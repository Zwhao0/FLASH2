# Tiny distributed MoE workload

`train.py` is a small systems diagnostic for distributed routing and training. It starts from deterministic random weights and synthetic inputs, so its loss says nothing about pretrained-model or Qwen quality. Each block is a top-2 MoE layer with a softmax router and GELU expert MLP. It has no attention, embeddings, tokenizer, dense transformer blocks, checkpoint loading, or Qwen implementation.

The default model has 2 layers, hidden size 64, FFN size 128, 16 experts, and 32 synthetic tokens per rank. `--num-layers 4`, `--hidden-size 128`, and `--seq-len 64` select the larger supported diagnostic shapes. The output is compared with a random target using mean squared error.

## Rank layout

The dimensions are `EP × TP × PP × expertDP = WORLD_SIZE`. `expertDP` is inferred from `WORLD_SIZE / (EP*TP*PP)`. The rank formula makes EP ranks contiguous:

```text
rank = (((expert_dp * PP + pp) * TP + tp) * EP + ep)
```

The process groups in the topology report have these meanings:

| Group | Ranks that vary | Coordinates held fixed |
| --- | --- | --- |
| `expert_ep` | `ep` | `expert_dp, pp, tp` |
| `expert_dp` | `expert_dp` | `pp, tp, ep` |
| `attention_dp` | `expert_dp, ep` | `pp, tp` |
| `expert_tp` | `tp` | `expert_dp, pp, ep` |
| `pp` | `pp` | `expert_dp, tp, ep` |

Singleton groups are omitted from the JSON topology value. The runner's four 16-rank configurations expand as follows (`[a-b]` means every rank from `a` through `b`):

| Configuration | expert EP groups | attention DP groups | Other groups |
| --- | --- | --- | --- |
| `EP8 TP1 PP1` (`expertDP=2`) | `[0-7]`, `[8-15]` | `[0-15]` | expertDP pairs `[0,8]` through `[7,15]` |
| `EP4 TP2 PP1` (`expertDP=2`) | `[0-3]`, `[4-7]`, `[8-11]`, `[12-15]` | `[0-3,8-11]`, `[4-7,12-15]` | expertDP pairs `[0,8]` through `[7,15]`; TP pairs `[0,4]` through `[3,7]` and `[8,12]` through `[11,15]` |
| `EP4 TP1 PP2` (`expertDP=2`) | `[0-3]`, `[4-7]`, `[8-11]`, `[12-15]` | `[0-3,8-11]`, `[4-7,12-15]` | expertDP pairs `[0,8]` through `[3,11]` and `[4,12]` through `[7,15]`; PP pairs `[0,4]` through `[3,7]` and `[8,12]` through `[11,15]` |
| `EP4 TP2 PP2` (`expertDP=1`) | `[0-3]`, `[4-7]`, `[8-11]`, `[12-15]` | the same four EP groups | TP pairs `[0,4]` through `[3,7]` and `[8,12]` through `[11,15]`; PP pairs `[0,8]` through `[3,11]` and `[4,12]` through `[7,15]` |

For example, launch EP8 on two 8-GPU nodes with:

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr=10.10.4.1 --master_port=29731 workload/train.py \
  --ep 8 --tp 1 --pp 1 --steps 160 --min-step-ms 500 --verify
```

`--verify` also works on smaller worlds when the dimensions divide evenly, such as `torchrun --standalone --nproc_per_node=2 workload/train.py --ep 2 --tp 1 --pp 1 --verify --steps 1 --min-step-ms 0 --no-redis`.

## Dispatch, tensor parallelism, and pipeline parallelism

The router picks the top two experts for each token and normalizes their selected softmax weights. A rank sorts routes by destination EP rank, sends token activations and route metadata with `all_to_all_single`, computes its local experts, then sends weighted results back with another `all_to_all_single`. A custom autograd function reverses each exchange in the backward pass.

For TP, each expert's first FFN matrix is split by output channels and its second matrix by input channels. The partial FFN outputs are summed across the expert TP group. Backward passes that output gradient to each shard and sums expert-input gradients across TP ranks. For PP, layers are split evenly across stages. A single microbatch moves forward with point-to-point activation sends and backward with point-to-point gradients; there is no pipeline overlap.

## Verification and gradient scaling

With `--verify`, step 0 builds a dense all-experts reference using the same deterministic initialization formulas. It checks the final output for the local tokens, the input gradient at that rank's pipeline boundary, each local router gradient, and each local expert weight shard against reference gradients. The pre-sync expert check sums the dense reference gradients over `expert_ep`, matching the local dispatched expert gradient before synchronization.

After `sync_gradients`, it independently reduces the dense reference router gradients by averaging over `attention_dp` and then TP, and reduces flattened full-expert reference gradients by summing `expert_ep`, averaging `expert_dp`, and dividing by EP. It extracts the current rank's TP expert shard and compares it with the synchronized local gradient. These expected gradients come only from the dense reference graph, not from the distributed gradients being checked. Missing gradients are treated as zero so unused or zero-gradient parameters remain verifiable. After AdamW updates, it checks router parameter replicas across `attention_dp` and TP, and expert parameter replicas across `expert_dp`, using group min/max differences. Per-rank JSONL contains a `verification` record with max absolute and relative error, tolerances, tensor sizes, replica differences, optimizer finiteness, and update status. The reference is not used in later steps.

TF32 matmul is disabled and float32 matmul precision is set to `highest` so GPU defaults do not create a reference mismatch; the comparison tolerances remain fixed.

The optimizer is AdamW. Before the first update, dense router gradients are averaged across `attention_dp`, then averaged across replicated TP routers. Expert weights receive contributions from all EP source-token ranks through dispatch, so their gradients are divided by EP and averaged across `expert_dp`; they are not reduced across EP a second time. This gives the mean-gradient scale over `EP × expertDP` local token batches. Verification checks that the loss and parameters are finite and that the first router parameter changes after the update.

## Run records and fault injection

`--run-id` must match `FLASH_RUN_ID` when both are set. Rank 0 writes `flash:{RUN}:topology` as a JSON object of semantic group names to rank cliques, appends `{"iteration": n, "time_ms": value}` to `flash:{RUN}:iter_time_0`, and sets `flash:{RUN}:done` to `1` after a normal finish. The default Redis URL is `redis://10.10.4.1:16389/0`; use `--redis-url` to override it or `--no-redis` to disable reporting.

Each rank writes JSONL to `--output DIR/rankN.jsonl` when `--output` is supplied. Otherwise it uses `--log-dir DIR/RUN_ID/rankN.jsonl`. Iteration records include loss, total `time_ms`, `compute_ms`, fixed `padding_ms`, and `injected_ms`; fault events include their wall-clock time and injection type. `--min-step-ms` is fixed host-side padding added to each step, not a minimum-time clamp. The default 160 steps with 500 ms padding takes at least about 80 seconds. Use `--min-step-ms 0` to measure the workload without that padding.

`--fault-rank`, `--fault-start`, `--fault-kind`, and `--fault-ms` select a rank-local fault. `compute` adds a host delay before expert compute (`cpu_pre_expert_compute_enqueue_delay`); `comm` adds a host delay immediately before `all_to_all_single` (`comm_pre_all_to_all_host_delay`), which is a timing proxy rather than physical link congestion. `transient` delays only the selected start step. `stop` writes an event, optionally sleeps for the requested duration, and exits only that rank without writing the normal `done` key.

For small-memory CUDA workers, the script defaults to `NCCL_BUFFSIZE=1048576`, `NCCL_MAX_NCHANNELS=2`, `NCCL_NVLS_ENABLE=0`, `CUDA_MODULE_LOADING=LAZY`, and `CUBLAS_WORKSPACE_CONFIG=:16:8`; caller-provided values take precedence. CUDA runs print allocated, reserved, and peak-reserved memory before and after process-group creation, around model creation, and around the first forward.
