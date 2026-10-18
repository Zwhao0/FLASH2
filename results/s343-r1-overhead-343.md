# FLASH 第 3.4.3 节开销报告

总体状态：**ended_incomplete**；用户已结束全部测试。
运行前缀：`s343-r1-`；Profile 前缀：`s343-r2-`。

## 已完成结果

监控 ABBA 完成：完整周期均值 21.0693 → 21.3105 ms，延迟开销 1.1449%，吞吐变化 -1.1319%。主指标为 rank 0 相邻 step_start_time，排除 step 0–4。

Healthy 完成 1,800 步，训练实际观察 39.385 秒，自动触发 0 次；此短时观察不能外推长期误触发率。

## 误触发与未完成对照

未注入故障轮已观察到 **1 次自动剖析触发**（按正常负控协议记为误触）；显式强制触发不计入。统计直接读取已观测事件，不要求 controller_stopped。共享 GPU 的真实资源波动不能完全排除，不据此断言算法根因错误。

- `s343-r1-profile-1-flash`：自动 1 次，强制 1 次，exit=[0, 0]。
  变点 iteration=12；实际 trigger 前 rank 0 已完成 step 50，二者不能混为触发时刻。事件记录的变慢比例为 26.77%。
  历史恢复说明：Local suite orchestration terminated to prevent further contaminated profiles. Training completed both nodes exit0. Raw files recovered explicitly; source hashes inherited from immediately preceding baseline (same source at launch). Excluded from controlled profile comparison due to automatic trigger before forced trigger and late second trigger.

- `s343-r2-profile-1-flash`：自动 0 次，强制 1 次，exit=[1, 1]。
  终止原因：User requested ending all tests; current run stopped deliberately, excluded from completed performance comparisons.

Profile 未形成完整有效的受控 ABBA，不报告最终开销。用户终止的 exit=[1,1] 轮次不计有效对照；剩余轮次未开展。CUDA 覆盖截止保守取 min(local profile_start.time + period, profile_reported.time)，报告尾部不能充当 CUDA 覆盖时间。

## Collective

全量 Collective 对照未开展，不能报告开销。仅有 baseline 校准（8 次 warmup、100 calls、每档一批）：

| 大小 MiB | Baseline 均值 ms |
|---:|---:|
| 4 | 3.02 |
| 8 | 5.84 |
| 16 | 10.64 |
| 32 | 21.08 |
| 64 | 41.15 |
| 128 | 79.96 |

## 运行状态

| 运行 | 状态 | Exit codes |
|---|---|---|
| `s343-r1-monitor-0-base` | 完成 | [0, 0] |
| `s343-r1-monitor-1-flash` | 完成 | [0, 0] |
| `s343-r1-monitor-2-flash` | 完成 | [0, 0] |
| `s343-r1-monitor-3-base` | 完成 | [0, 0] |
| `s343-r2-profile-0-base` | 完成 | [0, 0] |
| `s343-r2-profile-1-flash` | 已终止 | [1, 1] |
| `s343-r2-profile-2-flash` | 未开展（已取消） | — |
| `s343-r2-profile-3-base` | 未开展（已取消） | — |
| `s343-r1-healthy` | 完成 | [0, 0] |
| `s343-r1-collective-0-base` | 未开展（已取消） | — |
| `s343-r1-collective-1-flash` | 未开展（已取消） | — |

本地为共享 H800、小型 MoE、NCCL Socket/bond0（关闭 IB）；不等同于论文 H200/Qwen 环境。原始参数、manifest/source SHA256、恢复与终止说明保存在配套 JSON。
