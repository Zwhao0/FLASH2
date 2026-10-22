# 双 DP 视图 36-run 阶段筛查汇总

本页只描述当前 36 行矩阵，不与此前 22-run 矩阵或 3-run pilot 合并，也不据此宣称改善。负载为 559,104 参数、global batch 256、ETP=1，包含 Attention-DP 与 Expert-DP 两个 DP 视图；不含 attention 计算 workload。注入是 host-delay 代理，不是 DVFS 或 GPU 时钟变化。

线上预测指标取每行所有 terminal 输出的 `online_predictions_union`，不按 `valid` 过滤。线上命中只描述筛查输出，不能称为论文级验收或严格准确率。完整验收的 valid/exact 和不合格 profile 另列。 原计划第11行超时失败保留且不重试；前11行的容器CPU配额为8，修复后未启动后缀为16，启动超时由120秒增至180秒、两端采后导出并行。资源配置变化单独留档，不能把前后差异归因于定位算法改善。

计划分母固定为 28 个正例和 8 个对照。描述性线上 TP/FP/FN=25/1/3；唯一正确（terminal prediction union 恰为目标单例）=25/28；已终结 attempt、terminal log 可读且没有 rank 预测的 controls-clean=7/8。completed/failed/pending=35/1/0；engineering failure 等终结失败与 pending 分开计数。未运行或失败而无目标命中的正例仍按固定计划分母计入 FN。

完整验收：valid 28/36（valid 正例 21/28），验收 exact 19/28，对照 clean 3/8；不合格 profile 7，资格未知 profile 0，profile 验收缺失的行 0。

valid 子集条件指标（直接采用 `accuracy.json` 的 `all` 汇总）：TP/FP/FN=19/1/2；P=95.00%，R=90.48%（valid 正例 21）。固定计划保守召回单列为 19/28=67.86%；分母仍是全部 28 个计划正例，未通过资格或工程失败的行不会从中移除。描述性线上命中、valid 子集条件指标和固定计划召回是三种不同口径。

suite elapsed 从 `session.json` 原始 started 计到最新有 ended 的 attempt：4877.0 秒；最新结束为 1792596404.204 (fdm-1021-e4t2p2-control-after)。续跑不会重置起点。successful completed 且有共享 actual_steps 的训练 rank-iterations=200816（35 轮）；failed attempt 的原始 rank iteration 事件合计 720，pending attempt 的已观测部分合计 0，completed 但缺 actual_steps 的原始 iteration 合计 0；这些均不并入完成训练数。完整 attempt 中接受的 numerical rank-step=1120，完整 32 rank-step 验收=35 轮。

普通 TP 与专家 ETP 独立；16 rank、CP=1 时：Attention_DP=16/(TP×PP)，Expert_DP=16/(ETP×EP×PP)。

| EP | 普通 TP | PP | ETP | Attention_DP | Expert_DP |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 1 | 1 | 16 | 2 |
| 4 | 2 | 1 | 1 | 8 | 4 |
| 4 | 1 | 2 | 1 | 8 | 2 |
| 4 | 2 | 2 | 1 | 4 | 2 |

rank0/9 覆盖两物理节点的 PP stage0；PP=2 时另测 rank5/13 的 stage1。

## 按拓扑

| 分组 | 正例/对照 | completed | failed | pending | 描述性 TP/FP/FN | 单例命中 | controls-clean | 完整验收 valid/exact | 不合格/资格未知 profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EP4/TP1/PP2 | 8/2 | 10 | 0 | 0 | 7/1/1 | 7/8 | 1/2 | 10/10; 7/8 | 0/0 |
| EP4/TP2/PP1 | 6/2 | 7 | 1 | 0 | 5/0/1 | 5/6 | 2/2 | 5/8; 4/6 | 2/0 |
| EP4/TP2/PP2 | 8/2 | 10 | 0 | 0 | 8/0/0 | 8/8 | 2/2 | 10/10; 8/8 | 0/0 |
| EP8/TP1/PP1 | 6/2 | 8 | 0 | 0 | 5/0/1 | 5/6 | 2/2 | 3/8; 0/6 | 5/0 |

## 按延迟档位（对照单列）

| 分组 | 正例/对照 | completed | failed | pending | 描述性 TP/FP/FN | 单例命中 | controls-clean | 完整验收 valid/exact | 不合格/资格未知 profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 8/0 | 8 | 0 | 0 | 8/0/0 | 8/8 | 0/0 | 6/8; 6/8 | 2/0 |
| 14 | 8/0 | 8 | 0 | 0 | 7/0/1 | 7/8 | 0/0 | 6/8; 5/8 | 2/0 |
| 200 | 12/0 | 11 | 1 | 0 | 10/0/2 | 10/12 | 0/0 | 9/12; 8/12 | 2/0 |
| control | 0/8 | 8 | 0 | 0 | 0/1/0 | 0/0 | 7/8 | 7/8; 0/0 | 1/0 |

## 按物理节点

| 分组 | 正例/对照 | completed | failed | pending | 描述性 TP/FP/FN | 单例命中 | controls-clean | 完整验收 valid/exact | 不合格/资格未知 profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| node0 | 14/0 | 14 | 0 | 0 | 13/0/1 | 13/14 | 0/0 | 10/14; 9/14 | 4/0 |
| node1 | 14/0 | 13 | 1 | 0 | 12/0/2 | 12/14 | 0/0 | 11/14; 10/14 | 2/0 |

## 按 PP stage

| 分组 | 正例/对照 | completed | failed | pending | 描述性 TP/FP/FN | 单例命中 | controls-clean | 完整验收 valid/exact | 不合格/资格未知 profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PP0 | 24/0 | 23 | 1 | 0 | 22/0/2 | 22/24 | 0/0 | 17/24; 16/24 | 6/0 |
| PP1 | 4/0 | 4 | 0 | 0 | 3/0/1 | 3/4 | 0/0 | 4/4; 3/4 | 0/0 |

## 对照按拓扑

| 分组 | 正例/对照 | completed | failed | pending | 描述性 TP/FP/FN | 单例命中 | controls-clean | 完整验收 valid/exact | 不合格/资格未知 profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EP4/TP1/PP2 | 0/2 | 2 | 0 | 0 | 0/1/0 | 0/0 | 1/2 | 2/2; 0/0 | 0/0 |
| EP4/TP2/PP1 | 0/2 | 2 | 0 | 0 | 0/0/0 | 0/0 | 2/2 | 1/2; 0/0 | 1/0 |
| EP4/TP2/PP2 | 0/2 | 2 | 0 | 0 | 0/0/0 | 0/0 | 2/2 | 2/2; 0/0 | 0/0 |
| EP8/TP1/PP1 | 0/2 | 2 | 0 | 0 | 0/0/0 | 0/0 | 2/2 | 2/2; 0/0 | 0/0 |

## 每行结果

| Run | EP/TP/PP | node / stage | delay ms | actual_steps | observed partial rank-iterations | 线上预测并集 | valid | attempt state | refusal | wrong prediction ranks | terminal log parse error | 失败原因 |
|---|---|---|---:|---:|---:|---|---|---|---:|---|---|---|
| fdm-1021-e8t1p1-control-before | 8/1/1 | — / — | 0 | 667 | 0 | [] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e8t1p1-low-n0 | 8/1/1 | 0 / 0 | 14 | 435 | 0 | [0] | no | completed (completed) | 0 | [] | no | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-high-n1 | 8/1/1 | 1 / 0 | 200 | 146 | 0 | [9] | no | completed (completed) | 0 | [] | no | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-medium-n0 | 8/1/1 | 0 / 0 | 100 | 218 | 0 | [0] | no | completed (completed) | 0 | [] | no | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-low-n1 | 8/1/1 | 1 / 0 | 14 | 444 | 0 | [] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e8t1p1-high-n0 | 8/1/1 | 0 / 0 | 200 | 146 | 0 | [0] | no | completed (completed) | 0 | [] | no | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-medium-n1 | 8/1/1 | 1 / 0 | 100 | 216 | 0 | [9] | no | completed (completed) | 0 | [] | no | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-control-after | 8/1/1 | — / — | 0 | 539 | 0 | [] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p1-control-before | 4/2/1 | — / — | 0 | 275 | 0 | [] | no | completed (completed) | 0 | [] | no | missing controller_stopped; missing training_complete; one or more triggered profiles failed profile acceptance |
| fdm-1021-e4t2p1-low-n0 | 4/2/1 | 0 / 0 | 14 | 296 | 0 | [0] | no | completed (completed) | 0 | [] | no | one or more triggered profiles failed profile acceptance |
| fdm-1021-e4t2p1-high-n1 | 4/2/1 | 1 / 0 | 200 | — | 720 | [] | no | failed (engineering_failure) | 0 | [] | no | attempt not completed successfully: status='engineering_failure', exit_code=124; could not derive one shared actual_steps value from all 16 ranks; fault wall-time anchor missing; cannot evaluate pre-fault alarms; missing training_complete; node0: one or more worker exits are missing, failed, or still alive; node0: workers-node0.json success is not true; node1: one or more worker exits are missing, failed, or still alive; node1: workers-node1.json success is not true; rank0: expected one run_summary, found 0; rank10: expected one run_summary, found 0; rank11: expected one run_summary, found 0; rank12: expected one run_summary, found 0; rank13: expected one run_summary, found 0; rank14: expected one run_summary, found 0; rank15: expected one run_summary, found 0; rank1: expected one run_summary, found 0; rank2: expected one run_summary, found 0; rank3: expected one run_summary, found 0; rank4: expected one run_summary, found 0; rank5: expected one run_summary, found 0; rank6: expected one run_summary, found 0; rank7: expected one run_summary, found 0; rank8: expected one run_summary, found 0; rank9: expected one run_summary, found 0 |
| fdm-1021-e4t2p1-medium-n0 | 4/2/1 | 0 / 0 | 100 | 207 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p1-low-n1 | 4/2/1 | 1 / 0 | 14 | 444 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p1-high-n0 | 4/2/1 | 0 / 0 | 200 | 143 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p1-medium-n1 | 4/2/1 | 1 / 0 | 100 | 208 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p1-control-after | 4/2/1 | — / — | 0 | 641 | 0 | [] | yes | completed (completed) | 1 | [] | no | — |
| fdm-1021-e4t1p2-control-before | 4/1/2 | — / — | 0 | 721 | 0 | [] | yes | completed (completed) | 1 | [] | no | — |
| fdm-1021-e4t1p2-low-n0 | 4/1/2 | 0 / 0 | 14 | 545 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t1p2-high-n1 | 4/1/2 | 1 / 0 | 200 | 151 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t1p2-medium-n0 | 4/1/2 | 0 / 0 | 100 | 227 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t1p2-low-n1 | 4/1/2 | 1 / 0 | 14 | 549 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t1p2-high-n0 | 4/1/2 | 0 / 0 | 200 | 151 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t1p2-medium-n1 | 4/1/2 | 1 / 0 | 100 | 228 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t1p2-high-n0-stage1 | 4/1/2 | 0 / 1 | 200 | 151 | 0 | [] | yes | completed (completed) | 1 | [] | no | — |
| fdm-1021-e4t1p2-high-n1-stage1 | 4/1/2 | 1 / 1 | 200 | 151 | 0 | [13] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t1p2-control-after | 4/1/2 | — / — | 0 | 769 | 0 | [10] | yes | completed (completed) | 0 | [10] | no | — |
| fdm-1021-e4t2p2-control-before | 4/2/2 | — / — | 0 | 1038 | 0 | [] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-low-n0 | 4/2/2 | 0 / 0 | 14 | 542 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-high-n1 | 4/2/2 | 1 / 0 | 200 | 150 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-medium-n0 | 4/2/2 | 0 / 0 | 100 | 225 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-low-n1 | 4/2/2 | 1 / 0 | 14 | 534 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-high-n0 | 4/2/2 | 0 / 0 | 200 | 150 | 0 | [0] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-medium-n1 | 4/2/2 | 1 / 0 | 100 | 223 | 0 | [9] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-high-n0-stage1 | 4/2/2 | 0 / 1 | 200 | 149 | 0 | [5] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-high-n1-stage1 | 4/2/2 | 1 / 1 | 200 | 149 | 0 | [13] | yes | completed (completed) | 0 | [] | no | — |
| fdm-1021-e4t2p2-control-after | 4/2/2 | — / — | 0 | 723 | 0 | [] | yes | completed (completed) | 1 | [] | no | — |

## Failed terminal attempts

| Run | Status | Exit | actual_steps | Partial rank-iterations | 线上预测并集 | Missing rank logs |
|---|---|---:|---:|---:|---|---|
| fdm-1021-e4t2p1-high-n1 | engineering_failure | 124 | — | 720 | [] | [] |

## Pending 行

无。

逐组表的分母均来自固定计划行。Attempt 状态而非预测是否为空决定 pending、failed 或 completed；pending 行仍计入计划分母，已观测预测也照常保留，因此无预测不能反推为未运行。描述性线上 FP 保留无效 profile 运行中已读到的 terminal 错误 rank；global.jsonl 有解析错误时仍按 accuracy.json 给出的在线并集描述，但不计入 prediction-clean controls。

## 完整 36 行复核

[complete36-review.json](complete36-review.json) · [independent-audit-complete36.json](independent-audit-complete36.json)
