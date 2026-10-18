"""Refresh the repository's measured-outcome section without changing raw data."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'results/summary.json').read_text());runs=s['runs'];p=s['perf_monitoring_overhead']
strong=[r for r in runs if r['family'] in ['fault','hybrid'] and (r.get('confusion') or {}).get('exact_localization')]
lat=[r['performance']['injection_to_localization_seconds'] for r in strong if r['performance'].get('injection_to_localization_seconds') is not None]
controls=[r for r in runs if r['canonical_name'] in ['normal','transient']]
coll=s['collective_benchmark']['by_size'];lo=min(r['monitor_overhead_percent'] for r in coll);hi=max(r['monitor_overhead_percent'] for r in coll)
f=s['fault_detection_metrics'];t=f['totals'];text=f'''<!-- MEASURED_RESULTS_START -->
## 本轮实测结果（2026-10-18）

已在两节点16张H800上完成核心系统重建、真实分布式MoE训练及第三章对应的短测。结果支持部分机制，**不能据此声称复现了论文的全部性能与定位精度结论**。原作者FLASH源码未随材料提供，本项目依据论文明确实现；歧义、修正及原始最小值算法对照见[方法记录](docs/method.md)。

| 检查 | 本轮结果 |
| --- | --- |
| 四拓扑训练正确性 | EP8/TP1/PP1、EP4/TP2/PP1、EP4/TP1/PP2、EP4/TP2/PP2，共64个rank首步验证通过；输出最大绝对误差小于4e-7，同步后梯度对照通过，参数副本差为0 |
| 调用前持续延迟及混合拓扑 | 13个较强退化轮唯一定位正确，包含第二节点rank9；两个约5%弱退化轮未触发20%门限 |
| 通信执行区间停顿 | 检测到退化，但只能得到rank0–7候选，未唯一定位；按定位FN计。严格逐组最小值版本误指向7 |
| 停机 | 约2.13秒告警，健康rank随集合通信一起停滞；未唯一定位退出根因，不能称停机定位100% |
| 正常/瞬时抖动 | 两个125步轮无自动剖析误触；这不等同于数周误报率评估 |
| fail-slow定位描述统计 | TP={t['tp']}，FP={t['fp']}，FN={t['fn']}；16个持续故障/混合拓扑轮唯一命中13个（81.25%）；停机另列。每种设置样本很少，不代表总体准确率 |
| 最终监控端到端开销 | 无padding的ABBA：基线完整周期{p['baseline_median_ms']:.2f}ms，FLASH {p['monitor_median_ms']:.2f}ms，差异{p['monitoring_overhead_percent']:+.2f}%；两轮监控各为{p['monitor_run_median_range_ms'][0]:.2f}/{p['monitor_run_median_range_ms'][1]:.2f}ms |
| NCCL探针开销 | 4–256MiB AllReduce、两轮基线/两轮探针、每档8次预热+25次采样；变化范围{lo:+.2f}%至{hi:+.2f}%，负差异视作短测波动 |
| 定位时延 | 成功轮约{min(lat):.2f}–{max(lat):.2f}秒，主要来自20秒剖析下限，不能硬套论文“2–3个迭代” |

本轮使用约0.526M参数、两层、16专家top-2的随机初始化**MoE-only网络**和合成数据，GELU/AdamW；没有attention、预训练语言质量或论文Qwen 13.3B模型的吞吐结论。故障采用本进程的到达延迟、NCCL stream停顿和退出代理；硬件为共享H800、NCCL Socket路径。物理GPU降频、真实RoCE拥塞、Greyhound同负载精度对照和长期无故障运行未覆盖。

发现并修正了PP调度等待、TF32参考差异、显存占用、传播伪根因、pipeline-stage标签和监控态多余全量解析。初版异常慢轮与三次初始化/训练诊断失败均保留；没有用最终优化轮去覆盖原始记录。最终分析器对已有真实profiles重放，仅最早逐标签最小值歧义轮的结论发生预期修正。

完整数据：[结果表](results/summary.md)、[机器可读JSON](results/summary.json)、[完整性校验](results/artifact-validation.json)、[原始profile重放](results/profile-replay.json)、[计时独立核对](results/probe-timing-validation.json)。[总览图](results/summary.png)与[迭代图](results/iteration-timeline.png)用于查看这些实测结果。
<!-- MEASURED_RESULTS_END -->
'''
path=ROOT/'README.md';body=path.read_text();start='<!-- MEASURED_RESULTS_START -->';end='<!-- MEASURED_RESULTS_END -->'
if start in body:
 a=body.index(start);b=body.index(end,a)+len(end);body=body[:a]+text.strip()+body[b:]
else:
 marker='## 文件导航';body=body.replace(marker,text+'\n'+marker,1)
path.write_text(body);print('Updated measured README section')
