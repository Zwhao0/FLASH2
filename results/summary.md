# FLASH 工程结果汇总

生成时间：`2026-10-18T19:28:13+08:00` (Asia/Shanghai)。本文件由 `python3 scripts/summarize.py` 重复生成；源 JSON 为 [summary.json](summary.json)。

## 口径

- 仅实际存在的实验目录进入报告；不为未启动的矩阵项预填行或指标。仅当 manifest 的 `exit_codes` 精确为 `[0, 0]` 时计为正常完成；`stop` 注入的非零退出单列为预期失败。已有目录但无 manifest 会标为 missing / in-progress。
- 故障类型、目标 rank、注入步仅从 `manifest.arguments.args` 解析。`fault_injected` 事件只提供计时锚点；检测输出不参与标注 ground truth。
- baseline 使用 step 10 至 `fault_start - 5`（含边界），故障稳态从 `fault_start + 5` 开始。`time_ms` 包含 `min-step-ms` padding；同时报告 `compute_ms` 的变化作辅助检查。
- 剖析前/中/后按新日志的 iteration `step_start_time`/`wall_time` 与 global 事件时间切分。缺少这两个字段的旧 run 不硬推相位；其延迟/定位信息只作 preliminary。
- 延迟增幅 = `(fault steady median - baseline median) / baseline median`；吞吐下降 = `1 - baseline median / fault steady median`。命令行要求的延迟毫秒数不是实测严重度。低于或等于 20% 的故障不应期待触发 20% 检测阈值。
- `perf-*` 只在 `min-step-ms=0`、step 4 以后，以rank0相邻迭代起点之间的完整周期比较，包含准备/同步/上报；`perf-opt-*` 为监控态仅消费元数据的最终版本，旧全量解码轮保留但不混入最终对照。TP/FP/FN 只纳入完成的 fault/hybrid run；无故障或 transient 误触算 FP，目标 rank 未出现在预测集合时计 FN；唯一命中另计 exact localization。
- Strict minimum（旧逐组最小 rank）与 cross-validated 输出并列呈现。可从旧 profile 的聚合 records/topology 重算；profile 原始文件不修改。单次 run 的命中不代表统计显著性。

## 完成状态与故障检测

| Run | 状态 | manifest 故障配置 | 目标 rank | 基线 ms | 稳态 ms | 延迟增幅 % | 吞吐下降 % | 剖析相位稳态 ms | Observer s | Profiling s | 交叉验证 rank | Strict-min rank | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comm-delay126 | completed | comm | 0 | 276.44 | 405.31 | 46.62 | 31.80 | before=— ms; during=406.43 ms; after=402.52 ms | 1.74 | 21.02 | 0 | 0 | 1 | 0 | 0 |
| comm-delay14 | completed | comm | 0 | 280.69 | 293.87 | 4.69 | 4.48 | before=293.87 ms; during=— ms; after=— ms; no trigger | — | — | — | — | 0 | 0 | 1 |
| comm-delay182 | completed | comm | 0 | 277.13 | 460.46 | 66.15 | 39.81 | before=— ms; during=461.01 ms; after=459.04 ms | 1.98 | 21.00 | 0 | 0,6 | 1 | 0 | 0 |
| comm-delay238 | completed | comm | 0 | 282.55 | 515.68 | 82.51 | 45.21 | before=— ms; during=516.32 ms; after=513.58 ms | 2.31 | 20.85 | 0 | 0 | 1 | 0 | 0 |
| comm-delay70 | completed | comm | 0 | 282.06 | 348.18 | 23.44 | 18.99 | before=— ms; during=348.00 ms; after=350.39 ms | 1.45 | 21.18 | 0 | 0 | 1 | 0 | 0 |
| comm-stream-audit-r2 | completed | comm_stream | 0 | 277.74 | 386.52 | 39.17 | 28.14 | before=— ms; during=386.73 ms; after=378.52 ms | 1.63 | 21.17 | — | 7 | 0 | 0 | 1 |
| compute-delay126 | completed | compute | 0 | 277.89 | 404.28 | 45.48 | 31.26 | before=— ms; during=404.46 ms; after=401.74 ms | 1.50 | 21.18 | 0 | 0 | 1 | 0 | 0 |
| compute-delay14 | completed | compute | 0 | 278.26 | 292.10 | 4.98 | 4.74 | before=292.10 ms; during=— ms; after=— ms; no trigger | — | — | — | — | 0 | 0 | 1 |
| compute-delay182 | completed | compute | 0 | 279.13 | 466.16 | 67.00 | 40.12 | before=— ms; during=499.76 ms; after=459.56 ms | 2.28 | 21.01 | 0 | 0,7 | 1 | 0 | 0 |
| compute-delay238 | completed | compute | 0 | 277.93 | 527.80 | 89.90 | 47.34 | before=— ms; during=535.34 ms; after=514.79 ms | 2.26 | 21.01 | 0 | 0 | 1 | 0 | 0 |
| compute-delay70 | completed | compute | 0 | 282.17 | 347.17 | 23.04 | 18.72 | before=— ms; during=347.30 ms; after=344.00 ms | 1.49 | 21.31 | 0 | 0,7 | 1 | 0 | 0 |
| compute-othernode | completed | compute | 9 | 277.77 | 478.89 | 72.41 | 42.00 | before=— ms; during=479.40 ms; after=475.01 ms | 1.70 | 20.86 | 9 | 9,15 | 1 | 0 | 0 |
| hybrid-ep4-tp1-pp2 | completed | compute | 0 | 278.46 | 481.18 | 72.80 | 42.13 | before=— ms; during=481.68 ms; after=471.87 ms | 1.57 | 20.88 | 0 | 0 | 1 | 0 | 0 |
| hybrid-ep4-tp2-pp1 | completed | compute | 0 | 287.81 | 489.83 | 70.19 | 41.24 | before=— ms; during=490.28 ms; after=484.76 ms | 1.71 | 21.02 | 0 | 0 | 1 | 0 | 0 |
| hybrid-ep4-tp2-pp2 | completed | compute | 0 | 279.32 | 477.32 | 70.88 | 41.48 | before=— ms; during=478.80 ms; after=468.90 ms | 1.55 | 20.87 | 0 | 0 | 1 | 0 | 0 |
| hybrid-ep8-tp1-pp1 | completed | compute | 0 | 278.80 | 478.32 | 71.56 | 41.71 | before=— ms; during=478.81 ms; after=475.69 ms | 1.60 | 21.01 | 0 | 0 | 1 | 0 | 0 |
| normal | completed | no_injection | — | — | — | — | — | before=— ms; during=— ms; after=— ms | — | — | — | — | 0 | 0 | 0 |
| transient | completed | transient | 0 | 278.91 | 277.50 | -0.51 | -0.51 | before=277.50 ms; during=— ms; after=— ms; no trigger | — | — | — | — | 0 | 0 | 0 |

说明：阶段统计只对故障稳态样本计算；未完成的 profile 事件会按已观察到的活动窗口显示。`time_ms` 含固定 padding，因此小幅注入的 measured latency increase 可能低于请求注入时长所暗示的值。

## 汇总指标

| 完成且纳入 run 数 | Persistent fault run 数 | TP | FP | FN | Precision | Recall | Exact localization |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 16 | 13 | 0 | 3 | 1.00 | 0.81 | 13/16 (81.25%) |

不把 no-fault/transient 的无触发当成 TP；目标故障 rank 未出现在预测集合时计 FN，额外预测 rank 计为 FP；唯一命中另计 exact localization。上述汇总是当前实验次数的描述统计，不是总体准确率保证。

## 四种并行拓扑

| 拓扑 | Hybrid run | Hybrid 状态 | 交叉验证定位 | rank/tags/cliques/score | Verify run | Verify 状态 | 参数全 finite | 验证 rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EP8/TP1/PP1 | hybrid-ep8-tp1-pp1 | completed | 0 | r0 tags=2 cliques=2 score=1.989; r3 tags=1 cliques=1 score=0.995; r7 tags=1 cliques=1 score=0.995; r2 tags=1 cliques=1 score=0.994; r4 tags=1 cliques=1 score=0.993; r6 tags=1 cliques=1 score=0.993; r5 tags=1 cliques=1 score=0.992; r1 tags=1 cliques=1 score=0.989 | verify-ep8-tp1-pp1 | completed | 是 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 |
| EP4/TP2/PP1 | hybrid-ep4-tp2-pp1 | completed | 0 | r0 tags=3 cliques=3 score=2.902 | verify-ep4-tp2-pp1 | completed | 是 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 |
| EP4/TP1/PP2 | hybrid-ep4-tp1-pp2 | completed | 0 | r0 tags=3 cliques=3 score=2.860; r8 tags=1 cliques=1 score=0.629; r10 tags=1 cliques=1 score=0.557; r2 tags=1 cliques=1 score=0.543; r1 tags=1 cliques=1 score=0.523; r9 tags=1 cliques=1 score=0.504; r3 tags=1 cliques=1 score=0.459; r11 tags=1 cliques=1 score=0.455 | verify-ep4-tp1-pp2 | completed | 是 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 |
| EP4/TP2/PP2 | hybrid-ep4-tp2-pp2 | completed | 0 | r0 tags=5 cliques=4 score=4.831; r2 tags=1 cliques=1 score=0.579; r1 tags=1 cliques=1 score=0.575; r4 tags=1 cliques=1 score=0.561; r3 tags=1 cliques=1 score=0.560; r5 tags=1 cliques=1 score=0.537; r6 tags=1 cliques=1 score=0.528; r7 tags=1 cliques=1 score=0.523 | verify-ep4-tp2-pp2 | completed | 是 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 |

Topology 的定位分数来自该拓扑最新 profile 的交叉验证 ranking；Strict-min ranking 保留在 JSON 的 `rank_tag_clique_scores_literal_minimum` 与逐 profile 输出。参数 finite 来自 verify 事件的 `optimizer_update.all_parameters_finite`，不以 loss 曲线替代。

## Perf 监控开销（无 padding）

| Run | 监控实现 | 组别 | 完整周期 median ms | 计算片段 median ms | 纳入最终对照 | 预热后步数 |
| --- | --- | --- | --- | --- | --- | --- |
| perf-0-base | 全量解析 | base | 19.65 | 18.73 | 否 | 176 |
| perf-1-monitor | 全量解析 | monitor | 52.51 | 42.79 | 否 | 176 |
| perf-2-monitor | 全量解析 | monitor | 20.16 | 18.98 | 否 | 176 |
| perf-3-base | 全量解析 | base | 20.66 | 19.78 | 否 | 176 |
| perf-opt-perf-0-base | 元数据读取 | base | 19.72 | 18.86 | 是 | 176 |
| perf-opt-perf-1-monitor | 元数据读取 | monitor | 19.63 | 18.66 | 是 | 176 |
| perf-opt-perf-2-monitor | 元数据读取 | monitor | 21.16 | 19.93 | 是 | 176 |
| perf-opt-perf-3-base | 元数据读取 | base | 19.78 | 18.89 | 是 | 176 |

Base median：19.75 ms；monitor median：20.40 ms；监控开销：3.25%。该值包括该 run 实际启用的 probe callback 与 controller 开销。

## NCCL AllReduce microbenchmark

| MiB | Base runs | Base median rank ms | Monitor runs | Monitor median rank ms | 开销 % |
| --- | --- | --- | --- | --- | --- |
| 4 | 2 | 3.38 | 2 | 3.30 | -2.22 |
| 8 | 2 | 6.07 | 2 | 6.12 | 0.89 |
| 16 | 2 | 11.07 | 2 | 11.06 | -0.10 |
| 32 | 2 | 21.21 | 2 | 20.99 | -1.03 |
| 64 | 2 | 41.29 | 2 | 40.66 | -1.51 |
| 128 | 2 | 81.30 | 2 | 80.00 | -1.59 |
| 256 | 2 | 158.27 | 2 | 159.53 | 0.79 |

微基准结果按 JSON 的 per-rank latency 统计；`initial_profile` 可在 JSON 中核对。旧命名 `allreduce-*` 留在 diagnostics，避免混进最终 `collective-*` 矩阵。

## Preliminary / legacy diagnostics

| Run | 状态 | 分类 | Strict-min ranks | 重算 cross_validate ranks | Strict-min rank/tags/cliques/score | cross_validate rank/tags/cliques/score | Profile epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| allreduce-base-a | completed | diagnostic | — | — | — | — | 0 |
| allreduce-monitor-a | completed | diagnostic | — | — | — | — | 0 |
| allreduce-profile | completed | diagnostic | — | — | — | — | 0 |
| comm-stream-audit | failed | diagnostic | — | — | — | — | 0 |
| connectivity16 | completed | diagnostic | — | — | — | — | 0 |
| ep8-compute-e2e | completed | diagnostic | 0,7 | 0 | r7/tags1/cliques1/score0.995; r0/tags1/cliques1/score0.990 | r0/tags2/cliques2/score1.984; r7/tags1/cliques1/score0.995; r5/tags1/cliques1/score0.993; r4/tags1/cliques1/score0.992; r6/tags1/cliques1/score0.991; r3/tags1/cliques1/score0.990; r1/tags1/cliques1/score0.990; r2/tags1/cliques1/score0.989 | 1 |
| paper-clock600-auto | completed | diagnostic | — | — | — | — | 0 |
| profile-baseline | completed | diagnostic | — | — | — | — | 1 |
| profile-cost-0-base | completed | diagnostic | — | — | — | — | 0 |
| profile-cost-1-profile | completed | diagnostic | 6 | — | r6/tags1/cliques1/score0.528 | r6/tags1/cliques1/score0.528; r7/tags1/cliques1/score0.440; r4/tags1/cliques1/score0.422; r0/tags1/cliques1/score0.405; r5/tags1/cliques1/score0.377; r1/tags1/cliques1/score0.341; r3/tags1/cliques1/score0.341; r2/tags1/cliques1/score0.332 | 1 |
| profile-cost-2-profile | completed | diagnostic | 0 | — | r0/tags1/cliques1/score0.701 | r0/tags1/cliques1/score0.701; r2/tags1/cliques1/score0.660; r3/tags1/cliques1/score0.644; r1/tags1/cliques1/score0.635; r4/tags1/cliques1/score0.605; r7/tags1/cliques1/score0.587; r6/tags1/cliques1/score0.570; r5/tags1/cliques1/score0.509 | 1 |
| profile-cost-3-base | completed | diagnostic | — | — | — | — | 0 |
| sim-dpcomm-25 | completed | diagnostic | 1 | — | r1/tags1/cliques1/score0.453 | r1/tags1/cliques1/score0.453; r3/tags1/cliques1/score0.419; r6/tags1/cliques1/score0.388; r7/tags1/cliques1/score0.372; r5/tags1/cliques1/score0.347; r2/tags1/cliques1/score0.307 | 1 |
| sim-dpcomm-45 | completed | diagnostic | — | — | — | — | 1 |
| sim-dpcomm-5 | completed | diagnostic | — | — | — | — | 0 |
| sim-dpcomm-65 | completed | diagnostic | — | — | — | — | 1 |
| sim-dpcomm-85 | completed | diagnostic | 3 | — | r3/tags1/cliques1/score0.753 | r3/tags1/cliques1/score0.753; r2/tags1/cliques1/score0.733 | 1 |
| sim-dpcomm-pilot | completed | diagnostic | — | — | — | — | 1 |
| verify-ep8 | failed | diagnostic | — | — | — | — | 0 |
| verify-ep8-fp32 | completed | diagnostic | — | — | — | — | 0 |
| verify-ep8-lowmem | failed | diagnostic | — | — | — | — | 0 |

`ep8-compute-e2e` 是第一轮端到端诊断：旧 strict-min 结果与重算 cross_validate 并列留档，但不计入 final-suite precision/recall。早期失败 verify 同样只作诊断。summary 只读原始结果，不覆写 run 目录中的 raw/profile JSON。

## 缺失、进行中和退出状态

| Run | Suite | 状态 | exit_codes | iteration rows | 读取问题 |
| --- | --- | --- | --- | --- | --- |
| collective-0-base | collective | completed | [0, 0] | 0 | — |
| collective-1-monitor | collective | completed | [0, 0] | 0 | — |
| collective-2-monitor | collective | completed | [0, 0] | 0 | — |
| collective-3-base | collective | completed | [0, 0] | 0 | — |
| comm-delay126 | fault | completed | [0, 0] | 2000 | — |
| comm-delay14 | fault | completed | [0, 0] | 2000 | — |
| comm-delay182 | fault | completed | [0, 0] | 2000 | — |
| comm-delay238 | fault | completed | [0, 0] | 2000 | — |
| comm-delay70 | fault | completed | [0, 0] | 2000 | — |
| comm-stream-audit-r2 | fault | completed | [0, 0] | 2000 | — |
| compute-delay126 | fault | completed | [0, 0] | 2000 | — |
| compute-delay14 | fault | completed | [0, 0] | 2000 | — |
| compute-delay182 | fault | completed | [0, 0] | 2000 | — |
| compute-delay238 | fault | completed | [0, 0] | 2000 | — |
| compute-delay70 | fault | completed | [0, 0] | 2000 | — |
| compute-othernode | fault | completed | [0, 0] | 2000 | — |
| failstop | fault | expected_failure_stop | [1, 1] | 720 | — |
| hybrid-ep4-tp1-pp2 | hybrid | completed | [0, 0] | 2000 | — |
| hybrid-ep4-tp2-pp1 | hybrid | completed | [0, 0] | 2000 | — |
| hybrid-ep4-tp2-pp2 | hybrid | completed | [0, 0] | 2000 | — |
| hybrid-ep8-tp1-pp1 | hybrid | completed | [0, 0] | 2000 | — |
| normal | fault | completed | [0, 0] | 2000 | — |
| perf-0-base | perf | completed | [0, 0] | 2880 | — |
| perf-1-monitor | perf | completed | [0, 0] | 2880 | — |
| perf-2-monitor | perf | completed | [0, 0] | 2880 | — |
| perf-3-base | perf | completed | [0, 0] | 2880 | — |
| perf-opt-perf-0-base | perf | completed | [0, 0] | 2880 | — |
| perf-opt-perf-1-monitor | perf | completed | [0, 0] | 2880 | — |
| perf-opt-perf-2-monitor | perf | completed | [0, 0] | 2880 | — |
| perf-opt-perf-3-base | perf | completed | [0, 0] | 2880 | — |
| transient | fault | completed | [0, 0] | 2000 | — |
| verify-ep4-tp1-pp2 | verify | completed | [0, 0] | 48 | — |
| verify-ep4-tp2-pp1 | verify | completed | [0, 0] | 48 | — |
| verify-ep4-tp2-pp2 | verify | completed | [0, 0] | 48 | — |
| verify-ep8-tp1-pp1 | verify | completed | [0, 0] | 48 | — |
