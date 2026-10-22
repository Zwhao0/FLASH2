# FLASH v1.1 — 独立 ETP 双视图实测版

基于正式 `v1`（`daf2ac047e67cab4b8fd85f67f68c1babdfe8fb9`）。`flash/`、`probe/`、原 `workload/`、`scripts/`、`tests/` 的58个源码/测试文件保持逐字节一致。新增双视图训练负载与36项固定测试资料；没有将后续定位改进工作树合入本版。

- [版本说明与复核方法](docs/releases/v1.1.md)
- [已实测双视图负载](experiments/flash-v1-20261021/dual-view-matrix/workload_dual_view.py)
- [36项完整报告](experiments/flash-v1-20261021/dual-view-matrix/screening-summary.md)
- [独立复核](experiments/flash-v1-20261021/dual-view-matrix/independent-audit-complete36.json)

Git中保存已实测代码、协议、36项attempt及报告。完整raw、逐rank资格验证、日志和包级SHA256清单保留在本地桌面`FLASHv1`资料包；这些大体积证据不包含在本分支，故克隆仓库后不能直接重跑全raw离线验收。

限定结果：36项计划，35项工程完成、1项失败，28项通过完整资格验收；合格子集 TP/FP/FN=19/1/2，精确率95.00%、召回率90.48%；全部28个计划正例的保守召回19/28=67.86%。故障为rank本地host计算到达延迟代理，不是GPU降频；未据此声称达到论文硬件故障定位水平。

以下保留v1使用文档；其默认训练入口仍是原负载。双视图入口见版本说明。

---

# FLASH 第三章小规模复现

当前版本：**v1 — 第一版本 FLASH 复现**。冻结范围及结果见[版本说明](docs/releases/v1.md)。

本项目在两台 H800（每台8卡）上实现和验证训练故障检测与定位。系统由 `LD_PRELOAD` NCCL C++ 探针、共享内存环、本地控制器、Redis 全局控制器和小型真实分布式 MoE 工作负载组成。原论文与 GREYHOUND PDF 保留在项目根目录。

当前按用户最新要求采用**效果优先的 FLASH 工程扩展版**，默认使用跨组候选消歧；正文定位基线通过 `--paper-text-only` 保留。扩展与论文差异见[实现状态](docs/implementation_status.md)，交接入口为[handoff.md](handoff.md)。优先保护共享GPU任务，只用进程内故障代理，排除fail-stop。故障结果见[专项报告](results/scoped-paper-faults.md)，开销见[专项数据](results/focus-metrics.md)；这些历史指标不是最新修订后重测结果。

## 最新定位与混合并行评估（2026-10-18）

默认采用FLASH工程扩展版。13轮进程内代理定位中，计算唯一命中6/7（实测超过20%的6轮全部命中），正文对照4/7；四拓扑3.4.6计算代理评估扩展4/4、正文2/4。通信三轮较强代理均命中受影响组，但未唯一识别端点；按用户要求停止通信测试。均为少量描述性样本，不是真实限频/背景拥塞精度。

结果：[定位矩阵](results/loc-r1-localization.md)、[3.4.6四拓扑](results/hyb346-r1-hybrid-346.md)。后续问题与证据：[改进记录F01–F07](docs/flash_improvement_log.md)。全部实验已收尾，数值与采样完整性检查通过。

<!-- MEASURED_RESULTS_START -->
## 本轮实测结果（2026-10-18）

已在两节点16张H800上完成核心系统重建、真实分布式MoE训练及第三章对应的短测。结果支持部分机制，**不能据此声称复现了论文的全部性能与定位精度结论**。原作者FLASH源码未随材料提供，本项目依据论文明确实现；歧义、修正及原始最小值算法对照见[方法记录](docs/method.md)。

| 检查 | 本轮结果 |
| --- | --- |
| 四拓扑训练正确性 | EP8/TP1/PP1、EP4/TP2/PP1、EP4/TP1/PP2、EP4/TP2/PP2，共64个rank首步验证通过；输出最大绝对误差小于4e-7，同步后梯度对照通过，参数副本差为0 |
| 调用前持续延迟及混合拓扑 | 13个较强退化轮唯一定位正确，包含第二节点rank9；两个约5%弱退化轮未触发20%门限 |
| 通信执行区间停顿 | 检测到退化，但只能得到rank0–7候选，未唯一定位；按定位FN计。严格逐组最小值版本误指向7 |
| 停机 | 约2.13秒告警，健康rank随集合通信一起停滞；未唯一定位退出根因，不能称停机定位100% |
| 正常/瞬时抖动 | 两个125步轮无自动剖析误触；这不等同于数周误报率评估 |
| fail-slow定位描述统计 | TP=13，FP=0，FN=3；16个持续故障/混合拓扑轮唯一命中13个（81.25%）；停机另列。每种设置样本很少，不代表总体准确率 |
| 最终监控端到端开销 | 无padding的ABBA：基线完整周期19.75ms，FLASH 20.40ms，差异+3.25%；两轮监控各为19.63/21.16ms |
| NCCL探针开销 | 4–256MiB AllReduce、两轮基线/两轮探针、每档8次预热+25次采样；变化范围-2.22%至+0.89%，负差异视作短测波动 |
| 定位时延 | 成功轮约22.41–23.29秒，主要来自20秒剖析下限，不能硬套论文“2–3个迭代” |

本轮使用约0.526M参数、两层、16专家top-2的随机初始化**MoE-only网络**和合成数据，GELU/AdamW；没有attention、预训练语言质量或论文Qwen 13.3B模型的吞吐结论。故障采用本进程的到达延迟、NCCL stream停顿和退出代理；硬件为共享H800、NCCL Socket路径。物理GPU降频、真实RoCE拥塞、Greyhound同负载精度对照和长期无故障运行未覆盖。

发现并修正了PP调度等待、TF32参考差异、显存占用、传播伪根因、pipeline-stage标签和监控态多余全量解析。初版异常慢轮与三次初始化/训练诊断失败均保留；没有用最终优化轮去覆盖原始记录。最终分析器对已有真实profiles重放，仅最早逐标签最小值歧义轮的结论发生预期修正。

完整数据：[结果表](results/summary.md)、[机器可读JSON](results/summary.json)、[完整性校验](results/artifact-validation.json)、[原始profile重放](results/profile-replay.json)、[计时独立核对](results/probe-timing-validation.json)。[总览图](results/summary.png)与[迭代图](results/iteration-timeline.png)用于查看这些实测结果。
<!-- MEASURED_RESULTS_END -->

## 文件导航

- [第三章要求核对](docs/chapter3_requirements.md)
- [方法、工程解释和差异](docs/method.md)
- [GREYHOUND 复用边界](docs/greyhound_reuse.md)
- [MoE 模型、并行与数值验证](workload/README.md)
- [探针 ABI](probe/ABI.md)
- [结果汇总](results/summary.md)（运行汇总脚本后生成）
- [运行交接](handoff.md)

## 环境与目录

服务器 `10.10.4.1` 和 `10.10.4.2`；远程开发根目录 `/home/campus/zhangweihao/FLASH1/flash2`。只复用既有独立 Docker daemon 和镜像；不修改原 FLASH1/GREYHOUND 实验源码。实验容器均为 `flash2-dev`，私有共享内存2GiB，主节点专用 Redis 端口16389，实验键前缀 `flash:{run_id}:`。

现有离线镜像 `tianyuanwu/greyhound:ae` 的 image ID：`sha256:2bd15284c071b8e65a81e5641d9d0a6208563582cd3ffc2ee34e615a3366dd66`。精确运行时见 `results/environment/` 与 `requirements-control.txt`。服务器不需要访问外网。

本地 `服务器.md` 仅用于读取登录凭据，已加入 `.gitignore`；同步脚本白名单不会上传它。不要把密码写进命令、日志或文档。

## 最快复跑路径

在本地 Mac 项目根目录执行：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-ops.txt
.venv/bin/python scripts/sync.py
.venv/bin/python scripts/prepare.py
.venv/bin/python scripts/run_cluster.py my-verify --probe --args '--ep 8 --verify --steps 3 --min-step-ms 0 --no-redis'
.venv/bin/python scripts/run_cluster.py my-fault --probe --controller --extended-cross-validation --args '--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind compute --fault-ms 200'
```

`prepare.py` 依赖服务器上已经安装好的 Docker wrapper 与离线镜像；它不会重新拉取镜像或修改共享 Kubernetes/containerd 服务。已有容器会直接复用。

本轮收尾已停止两端 `flash2-dev`（包括本实验 Redis），释放实验进程、共享内存和 GPU 占用；原始数据位于宿主机挂载目录，停止容器不会删除。再次运行 `prepare.py` 会启动容器及专用 Redis。收尾状态见 [final-state.json](results/final-state.json)。健康状态强制剖析负控也已完成，未产生慢卡候选，且无计时错误或丢记录。

每轮使用新的 run ID，避免和旧结果混淆。`run_cluster.py` 会启动两台节点的8进程 torchrun、设置低显存 NCCL 环境、收集退出码/源码 SHA256/原始 JSONL，并下载结果至 `results/<run_id>/node0|node1/`。`--timeout` 为所有训练进程提供上限。`--force-profile-at` 仅用于独立剖析验证，不应计入自动检测召回率。

当前两类进程内故障的完整十点矩阵（五档计算晚到、五档跨节点通信流迟滞；新prefix避免覆盖）：

```bash
.venv/bin/python scripts/run_scoped_fault_suite.py --prefix r2-
.venv/bin/python scripts/report_scoped_faults.py --prefix r2-
```

以下为历史完整矩阵，保留作工程对照；其中 `faults` 的通信是调用前晚到代理，当前通信执行迟滞评价使用上面的 scoped 矩阵。按顺序执行，不能让两套 GPU 矩阵同时运行：

```bash
.venv/bin/python scripts/run_suite.py verify
.venv/bin/python scripts/run_suite.py faults
.venv/bin/python scripts/run_suite.py hybrid
.venv/bin/python scripts/run_suite.py perf
.venv/bin/python scripts/run_suite.py collective
```

`faults` 默认不再运行停机注入；仅显式加 `--include-failstop` 时保留旧停机诊断，其预期非零退出，其他轮必须两端 exit 0。脚本保留失败结果并停止普通失败轮，便于先修正再重跑。`perf` 是无固定 padding 的 ABBA 对照；`collective` 是4–256MiB真实 NCCL AllReduce 对照。故障轮固定 padding250ms只用于给检测/20秒剖析提供足够时间，不能拿来声称模型吞吐或 TFLOP/s。

## 正确性与结果解释

22项本地单元测试可以使用带 NumPy 的 Python 执行：

```bash
python -m unittest discover -s tests
```

Rbeast 集成测试在容器中运行 `PYTHONPATH=/workspace/flash2 python tests/rbeast_integration.py`。训练的 `--verify` 在首步比较独立 dense-reference 的输出、输入/router/expert 梯度、同步后的全局平均梯度，并检查更新后的参数副本。后续步骤持续检查 finite loss。

共享环境可用显存约2–5GiB/卡。默认禁用 NVLS、降低 NCCL 缓冲/通道、启用 CUDA 懒加载；不更改已有 vLLM/GPU 时钟。模型为随机初始化的约0.526M参数两层 top-2 MoE-only 网络（无注意力、无预训练质量结论），并非论文13.3B Qwen模型。

故障是仅影响实验进程的 host enqueue 延迟和进程退出代理；没有把代理等同于物理 GPU 降频或真实网络拥塞。对于停机，同步阻塞会传播到健康 rank，观测停滞集合不等同于根因集合。最终精确率、召回率、开销和定位延迟应以原始短测数据为准，不套用论文数字。

严格每标签取最小 rank 的结果保存在 `literal_minimum_diagnosis`。最终交叉验证保留20%显著低时延候选集合并用独立通信组交叉消歧；这个解释与论文简写存在差异，详见方法文档。训练进程、控制器与离线分析器之间不传递注入故障标签；只有实验汇总器使用注入信息评估结果。

需要快速回归时，可先运行本地单元测试；两节点少卡数值 smoke 的参数示例为 `run_cluster.py fresh-smoke --gpus 1 --probe --args '--ep 2 --verify --steps 3 --min-step-ms 0 --no-redis'`。正式实验仍使用每节点8卡。已完成的 run ID 不允许重复覆盖；重跑整套矩阵时使用 `run_suite.py <suite> --prefix r2-` 生成新命名空间。

更严格的通信执行停顿审计（需已编译 overlay）：

```bash
.venv/bin/python scripts/run_cluster.py fresh-stream-audit --probe --controller --stream-delay --args '--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind comm_stream --fault-ms 10'
```

该选项只作用于本次 rank0 的 NCCL stream，10ms 为每个 P2P group 的 callback 延迟；不应把它直接理解为整次迭代仅增加10ms。
