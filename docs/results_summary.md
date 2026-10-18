# 结果汇总脚本

从本地 `results/` 重建工程结果表：

```bash
python3 scripts/summarize.py
```

脚本生成 `results/summary.json` 与 `results/summary.md`。有至少两组完成的故障测量时，如果当前 Python 环境可导入 Matplotlib，还会生成 `results/summary.png`。可以传 `--results-dir /path/to/results` 指向另一份已下载的结果，传 `--prefix expA-` 只汇总名称以该前缀开头的实验，或传 `--no-plot` 跳过绘图。对正在追加日志的目录可随时重跑；已有实验目录但缺失 manifest 或尚未结束时会标为 `missing` / `in_progress`。完全未创建的计划项不会被虚构成表格行。

最终矩阵按 `scripts/run_suite.py` 的命名组织：故障 suite（`normal`、`transient`、`compute-delay*`、`comm-delay*`、`compute-othernode`、`failstop`）、四个 `hybrid-*`、四个 `perf-*`、四个 `collective-*`、四个 `verify-ep*-tp*-pp*`。无 manifest 的计划项只显示 run 名和缺失状态，脚本不会从名称猜故障类型或目标 rank。非零退出的 stop 注入单独列为预期失败；正常完成严格要求 manifest 中 `exit_codes` 等于 `[0, 0]`。JSON 每个 run 的 `stop_or_timeout_events` 单列观测到的 `failstop` 与 `profile_timeout` 原始事件；`profile_timeout` 只表示观察到超时事件，本身不作为 failstop 成功证据。

脚本从 `manifest.arguments.args` 读取故障类型、目标 rank 与起始步。rank JSONL 中 `fault_injected.wall_time` 只用来计算观察延迟；detector 的 diagnosis 和 profile 标签从不作为 ground truth。训练的 verification 当前内嵌在 `node*/rankN.jsonl` 的 `verification` 事件中；脚本也兼容独立的 `rankN-verification.jsonl`。

故障性能基线使用 step 10 到 `fault_start - 5`（含），稳态从 `fault_start + 5` 起。脚本以 rank 0 的 iteration 为主测序列，缺 rank 0 时回退为每步已有 ranks 的中位数。`time_ms` 包括 `min-step-ms` 固定 padding；因此同时输出 `compute_ms` 变化供核查，但 latency increase 与 throughput drop 按 `time_ms` 计算。新的 iteration `step_start_time`、`wall_time` 与 global controller 事件时间用于把稳态测量分到 profile 前、中、后；旧日志没有这些时间戳时只给 preliminary 总体稳态数，不从步长推测 profiling phase。

Profile 报告里有 `literal_minimum_diagnosis` 时原样保留；没有该字段的旧 profile 会从其聚合 `records` 与 `topology` 分别重算 strict-minimum 和 cross-validation 两种解释。原始 profile 和 rank JSONL 不修改。跨验证结果用于主定位表与 TP/FP/FN，旧 strict-minimum 输出作为对照；目标 rank 未出现在预测集合时计 FN，额外预测 rank 计 FP，唯一命中另计 exact localization。`ep8-compute-e2e` 及早期失败 verify 只在 diagnostics 节，不并入最终矩阵统计。

性能主表优先使用最终 `perf-opt-perf-*` ABBA；原始 `perf-*` 中监控态全量解析的异常慢轮保留作诊断，不混入最终开销。只比较 `min-step-ms=0` 且 step 大于 3 的完成 run，以 rank0 相邻 `step_start_time` 之差统计完整周期（包含准备、同步、日志和 callback/controller 的影响），先取每轮中位数再比较两轮基线/监控中位数。`collective-*` 从每个 `collective-bench-*.json` 汇总不同 buffer 大小的 rank latency；旧 `allreduce-*` 文件留在 legacy diagnostics 范围。

最终交付图由 `scripts/plot_summary.py` 读取 summary.json 绘制，输出 summary.png/svg 与 iteration-timeline.png/svg。本轮在服务器既有 Matplotlib 环境生成，取回后已做视觉检查；本地没有 Matplotlib 时重建 summary 不会删除已交付图。健康状态强制 profile 的 `profile-baseline` 单列为负控，不计入自动检测误触。
