# CUDA stream stall overlay

Build with `bash probe/build_stream_delay.sh`, then preload in this order:

```sh
LD_PRELOAD=/workspace/flash2/probe/libflash_nccl_probe.so:/workspace/flash2/probe/libflash_stream_delay.so
```

Set `RANK`, `FLASH_STALL_RANK` (defaults to `0`), and `FLASH_STALL_FILE`. The file must contain one finite positive number of milliseconds; a missing file, empty value, zero, negative value, or invalid text disables the delay. The overlay reads the file on the first Send/Recv in each outer NCCL group. It attempts at most one callback in that group; standalone Send/Recv calls each make their own attempt.

The overlay queues `cudaLaunchHostFunc` on the Send/Recv stream before forwarding the NCCL call. With the preload order above, the base probe has already queued its start event, so active event timing includes the stream stall. The callback only sleeps and frees its heap-allocated millisecond value; it does not call CUDA APIs. The overlay reads only the requested delay file and rank environment values.

For the shared-GPU fault simulation, use `FLASH_STALL_KIND=allreduce` and
`FLASH_STALL_COMM_SIZE=2`. This queues the same rank-local callback only on
size-2 NCCL AllReduce streams (the EP8 workload's inter-node expert-DP groups).
The default `p2p` mode preserves the earlier Send/Recv experiment. A positive
communicator-size filter is checked using public `ncclCommCount`.

This is an execution-latency proxy: it never changes clocks/NICs, starts traffic,
or allocates a delay kernel. It does not reproduce bandwidth contention, packet
loss, or RoCE congestion-control behavior. Peer synchronization can propagate
waiting within this experiment. Compare the paper's literal rule and the extended
cross-validation rule separately; neither is guaranteed to uniquely identify the
slow endpoint when both members measure nearly the same collective duration.
