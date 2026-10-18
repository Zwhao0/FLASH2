# FLASH 复现方法与证据口径

目标为《中文论文》第三章的故障检测和定位，不含第四章 SHIFT 恢复。使用论文允许的简化 communicator 模式，不解析 NCCL 内部 ring/tree ABI。

- 训练使用随机初始化小 MoE 与合成输入，验证真实 CUDA 前后向、专家路由、优化器更新；不代表语言模型质量或原论文 13.3B 模型的绝对性能。
- 官方 GREYHOUND Docker `tianyuanwu/greyhound:ae` 已离线部署于两节点。复用其 CUDA/PyTorch/NCCL/Redis/Rbeast 运行时及公开变点处理思路。上游源 commit `d641b3997c17de700e1ef7d37e353418df7e336d` 只读保留在 third_party/Greyhound。新探针无需其版本敏感的 NCCL 内部结构和 stop-the-world 微基准。
- 环境为 H800 两节点各8卡，共享既有 GPU 工作负载；不终止其他服务、不全局降频、不修改网络队列。NCCL 使用 bond0 Socket 与1MB缓冲、至多2通道，以减少显存占用和保证快速连通。此路径不代表 RoCE 性能上限。
- 主实验保留论文默认：30迭代窗口、前3迭代预热、z-score3、Rbeast候选概率0.8、退化阈值0.2、恢复比率0.75、变点距至少2迭代、右保护2迭代、组/rank CV0.1、离群20%、剖析 `max(2.2*最新迭代秒数,20)`。强制剖析只用于隔离验证定位机制，独立于端到端检测成功率统计。
- 论文称 BOCD，但指定 Rbeast；Rbeast 官方定义为 BEAST，非 Adams–MacKay 运行长度后验递推。实现遵从指定库，并记录版本与参数。论文没有完整给出变化阈值分母；这里定义 `(退化迭代均值-基线)/基线`，只检测变慢，恢复不触发。
- 算子层模拟故障仅影响实验 rank：计算延迟/通信调用前延迟/进程退出用于可控正确性验证。延迟代理不等价于物理 GPU 降频或背景网络拥塞。报告分开统计，绝不把设定延迟当成实测性能下降百分比。
- 无调用停滞并不自动证明根因：同步集合通信可让健康 rank 一起阻塞。停机报告区分观测停滞集合与已退出的进程，不将传播阻塞视为多个硬件故障。
- 聚合遵从论文绝对时延方案，原始事件额外保留 count/op 便审计不等消息量偏差。混合并行分离结果若不稳定，先检查GPU event边界、communicator identity、采样覆盖与训练梯度，再报告算法限制。

官方参考：
- [GREYHOUND 原仓库与复现指南](https://github.com/wutianyuan1/Greyhound)
- [NCCL Group Calls](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2243/user-guide/docs/usage/groups.html)：GroupEnd返回后通信才完整提交，禁止在其之前对组内通信做同步。
- [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)：NCCL all_to_all_single、send/recv、process group。
- [torchrun](https://docs.pytorch.org/docs/stable/elastic/run)：多节点独立启动器。
- [Rbeast 官方库](https://github.com/zhaokg/Rbeast)：cp/cpPr及BEAST定义。

## 同步传播歧义与两种定位输出

首轮 `ep8-compute-e2e` 的真实原始数据暴露出可重复的同步传播：EP 组精确指向 rank 0，但其下游 attention-DP 组中 rank 0–7 都已晚到，时延都很短，绝对最小值偶然落在 rank 7。逐标签仅取最小值得到 `[7,0]`；将略高的标准化得分当作最高置信度会错误优先 rank 7。

因此实现保留 `literal_minimum_diagnosis`（论文文字逐标签取最小 rank 的原样结果），同时在最终交叉验证中保留超过论文 20% 低时延阈值的候选集合，以独立 communicator 的共同支持进行消歧。只有一个候选获得最多独立组支持时才给出唯一定位；候选并列则报告 ambiguous，不擅自选根因。相同 communicator 的多个语义标签只算一个独立证据。此规则不读取故障注入位置、强度或故障类型。

这属于对论文“20% 筛选离群值”及“多标签交叉验证”的明确工程解释；相比“每标签严格只保留一个最小 rank”的实现有差异。两种结果、逐组得分和原始时延均保留，不能将消歧后的指标直接当作论文原实现的精度。抽象传播案例单元测试及原始数据重分析都表明，该消歧可以去除本轮的传播伪根因，仍需在其他 rank/并行拓扑和通信延迟实验中验证。

## 已发现并修正的训练验证问题

- 初版 PP 调度在接收激活前后放置了全局 barrier 和提前 loss broadcast，会形成循环等待，已改为先统一进入迭代、按 stage 顺序发送激活、反向返回梯度，最后广播 loss。
- 默认 NVLS 与 CUDA 运行时占用导致第一轮 MoE 训练 OOM，已禁用 NVLS、约束通道并启用懒加载。故障轮日志保留，不作为完成的训练实验。
- NGC 默认 TF32 导致不同专家批次形状与逐专家参考实现产生约 1e-4–1e-3 输出误差。明确关闭 TF32 后，严格 FP32 对照通过；没有放宽原有容差。
- 首步参考检查覆盖分布式输出、输入/router/expert 梯度、梯度同步后的全局平均、优化器参数更新和副本一致性。后续步骤检查 finite loss；首步通过不等同于长时收敛证明。

大量探针快照和逐事件剖析原始记录以 `.json.gz` 无损压缩保存，以缩短私网回传时间；统计、训练 JSONL、退出码和诊断 JSON 保持直接可读。`lost_on_snapshot` 指终态8192槽环之前已被覆盖的历史，不代表运行中控制器丢样；真实丢样指标在各节点 profile 上报的 `lost` 字段。

## 探针计时的独立核对

`allreduce-profile` 的1280条两节点记录中，1248条位于剖析模式并有非零 CUDA 时长，剩余32条是切换前初始化/握手记录；计时错误与未知 communicator 均为0。以 rank0 各档最后8次样本核对，探针 GPU 区间均值与外部 CUDA Event 区间均值接近，外部区间略长（它还覆盖包装器周围的提交开销）。逐档数值在 `results/probe-timing-validation.json`；不能把 host API 返回耗时当成 GPU 集合通信完成耗时。

BEAST 的显式调用参数为 `season='none', start=0, deltat=1, tcp_minmax=[0,5], torder_minmax=[0,0], tseg_minlength=2, mcmc_seed=12345, hasOutlier=True`；其他推断参数采用镜像内 Rbeast 0.1.19 的默认值。零阶趋势是对迭代平台时延变化的工程建模选择，论文没有公布对应先验/MCMC配置；不能把这组设定说成原作者的完整配置。

四种拓扑均保持每个数据源32个 token；TP/PP 使用复制或分段执行，因此不同拓扑的有效全局 token 批量不同。四拓扑实验用于验证正确性与定位证据，不用于宣称跨拓扑吞吐可直接比较。

## 补充的通信执行停顿审计

`comm-stream-audit` 将独立 overlay 放在探针之后加载，按 rank0 的专用控制文件，在每个 NCCL P2P group 首个操作前向原 stream 排入10ms host callback。探针 start event 已排在 callback 前，因此这个停顿处于测量区间内；它与 Python 调用前 sleep 的晚到代理具有不同计时含义。callback 仅休眠并释放自己的参数，不调用 CUDA 或等待其他 GPU 工作，遵循 [CUDA Runtime API](https://docs.nvidia.com/cuda/archive/12.3.2/cuda-runtime-api/group__CUDART__EXECUTION.html) 的限制。

这一测试仍是进程内受控代理，不是 RoCE 链路或 NIC 带宽实测；其用途是检验“慢 rank 反而通信更短”假设在通信执行区间本身变慢时是否成立。它独立纳入真实结果，不能只展示晚到代理成功率并据此推断真实网络拥塞的定位能力。

补充审计首轮在 NCCL 初始化之前失败，原因是 overlay 仅用 `RTLD_NEXT` 无法找到 PyTorch 以局部可见性加载的 NCCL 符号；修正为与主探针相同的显式 `dlopen`/`dlsym` 后，`comm-stream-audit-r2` 两端完成125步。初轮日志作为初始化诊断保留，不计故障定位精度。实际停顿轮触发了检测，但 EP 组内时延约10.14–10.23ms，rank CV仅0.00260，低于论文0.1阈值；下游 attention-DP 只能区分慢到的整个0–7 cohort。严格逐组最小值误指向7，交叉验证报告0–7候选并拒绝唯一定位。该轮按定位FN计，不算成功。

## 流水线标签与最终重放核对

最终分析器按框架 PP group 中的 stage 位置，为 DP、TP、EP 和 expert-DP 通信组增加 `ppstage{n}_` 标签。不同 stage 的工作量和通信相位不同，不能在阶段一作为同类 clique 互相竞争；这与论文 `ppstage0_dp` 的实现描述一致。新增单元测试覆盖“两个不同stage延迟量级相同，合并标签会错误跳过候选”的情况。已有真实 aggregate profiles 已用最终分析器重新执行并保留于 `results/profile-replay.json`；原始诊断不被覆写。

默认 world 同步 communicator 若不等同于某个模型并行组，会记录为 `special`。所以图表中“不同 communicator 数”不全等于模型并行维度数；玩具训练还包含迭代对齐的控制通信，不能将这些证据数量或跨拓扑分数直接当作论文大模型结论。

## 监控态开销修正与性能口径

初版 Python 本地控制器在监控态也以100ms周期把整个环解析成 Python 字典，尽管停机判断只需要 heartbeat/write_seq。无 padding 真实训练 ABBA 中出现一轮完整周期52.51ms、另一轮20.16ms，而基线为19.65/20.66ms；该异常没有触发 CUDA 剖析，全部原始数据保留。仅从现有样本不能把异常唯一归因为 GPU 共享或某个组件。

最终控制器在模式0只消费共享元数据、推进读序号，不构造逐事件对象；模式1仍完整解析、统计时延。这去掉了与论文低开销监控设计无关的额外工作。新增模式边界单元测试，正常监控中有意跳过的事件不计为剖析丢样。最终版本用独立 `perf-opt-*` ABBA 复测，旧 `perf-*` 轮显示于表中但不与最终实现混合。

性能主指标改为 rank0 **相邻 iteration start 的完整周期**，包含数据准备、迭代同步、日志和 Redis callback。原 `time_ms` 是 GPU训练段加padding/末尾同步的片段，不覆盖全部循环；它继续用于原故障实验的检测输入及阶段对比，并在性能表作为辅助列展示。固定250ms padding 的故障轮不用于训练监控开销结论。

## 当前评价范围与 profile 专项补测（2026-10-18）

按用户要求，当前主评价只保留常态/profile开销、计算/通信两类慢故障的唯一正确定位率。停机方法不作为当前方案结论；默认faults矩阵跳过停机，只有显式`--include-failstop`才运行历史诊断。原控制器实现和原始停机数据保留，未声称实现新的停机算法。

新增无故障、无padding的1800步ABBA：两轮无FLASH、两轮完整FLASH，后者约第200步强制进入20秒采样。统计完整迭代起点间隔；profile和常态各自按实际覆盖步号匹配两轮无FLASH基线，避免将训练前后不同路由/步段直接视作开销。边界2秒排除用于中位周期估计；完整事务另外不排除边界，统计均值和累计多耗时间。原始跨步段profile/常态周期差仅作描述，不代表严格因果增量。

本次匹配常态开销2.01%（原180步ABBA为3.25%），profile中位周期开销14.16%，两轮12.82%/15.50%；完整事务平均开销10.41%/12.86%，累计多耗2.794/3.118秒。事务27.38–29.64秒包括20秒采样和控制器聚合/原始记录写盘，并不代表训练暂停这么久。两轮无唯一故障输出，但均有rank0–7候选歧义，未把无故障强制profile描述成完全无候选。详见results/focus-metrics.md/json。

专项复跑（先sync/prepare，使用新prefix）：

```bash
.venv/bin/python scripts/run_profile_overhead.py --prefix r2-
.venv/bin/python scripts/report_focus.py --prefix r2-
```

这是共享H800、Socket网络与极小MoE下的测量，两个profile轮不足以推断长期稳定分布。计算/通信故障仍为进程范围的延迟/停顿代理；执行停顿未唯一定位的结果保留，不将调用晚到定位成功推广为真实网络拥塞成功。

## 最新边界：论文故障类型与共享任务保护

用户确认排除fail-stop，并补充“如果有别的进程，尽量不干扰，模拟故障就行”。因此不继续物理限频/共享链路背景流量。此前600MHz试验已恢复GPU0原应用时钟2619/1980MHz，所有16卡当前SM1980MHz；真实背景流量从未启动。物理工具与流量工具默认拒绝执行，只有显式硬件/共享链路opt-in才可作为独立备选，不属于当前默认矩阵。

论文3.4.4的原方法是限制GPU频率和注入背景流量；原有host sleep不是相同的注入手段。当前明确选择**进程内代理**，不能声称已经完成真实降频/背景拥塞的精度复现。计算类复用计算入队前等待的五档原始日志。通信类补测只延迟rank0跨节点expert-DP size=2 AllReduce的CUDA stream；callback不调用CUDA、不产生额外流量/大块显存/额外GPU核，记录等待包含在本进程通信CUDA Event区间内。此代理模拟通信执行迟滞，不模拟带宽、丢包或拥塞控制。

以论文文字逐标签最小rank规则作为本专项主结果（`--literal-minimum`），扩展跨组消歧单列。强度计划沿用5/25/45/65/85%的相对时延增加近似档位；原文性能下降分母不完整，报告同时给出实际时延增加和吞吐下降，绝不把计划值直接当成实测故障强度。

另：两轮用于开销的无故障强制profile，扩展规则无唯一输出，原文逐组最小规则分别给出[6]/[0]。因为是强制采样而非自动误触，这不计自动监控误报；它仍是原规则在本共享小模型环境下的条件定位局限，不能隐去。当前主结果入口为results/scoped-paper-faults.md/json，历史增强代理指标保留在focus-metrics中。

### 正文规则复核与本次模拟结论

逐项回看3.3.2后，将正文完整指定的步骤实现为独立`paper_text=True`：多个组的max_avg CV不足0.1时跳过，否则取最大组；组内rank平均时延CV不足0.1时跳过，否则取最小rank。正文未完整定义额外20%离群阈值的公式/位置，旧实现的clique/rank 20%门限解释另存，不冒充确定原文。原有cross_validate默认行为不变。17项单测通过，控制器`--paper-text-only`用已保存真实profile做Redis控制面重放通过；不涉及额外GPU工作。首次测试断言曾把JSON字符串键与Python整数键直接比较，统一JSON表示后完整输出一致，控制器判定未改动。

五档计算代理复用原始数据，正文解释唯一2/5（Precision66.67%、Recall80%，两轮另含错误rank7）；扩展消歧4/5。新跨节点通信执行代理唯一0/5，正文TP0/FP3/FN5，扩展拒判0/5而不是Precision100%；4个强退化轮的最高expert-DP组信号均为[0,8]，组内rank CV仅约0.0005–0.001，低于0.1，无法区分两端。正文还会从其他受同步传播影响的组误取rank1/7/3。这些结果不能推广为真实RoCE故障表现，模拟方式和论文原注入不同。

只读/数值/计时完整性审计累计56个训练run无异常；所有新通信轮125步、16rank、双端exit0，零计时错误/未知组/丢记录。独立开销数据和历史旧结论不覆盖；常态/profile开销仍按其独立无padding ABBA版本解释。当前完整主表为results/scoped-paper-faults.md/json。

## 最终主线决定：效果优先（交接版本）

用户最终确认不要求逐字还原：默认保留FLASH工程扩展版（跨组候选消歧），正文定位解释是显式`--paper-text-only`对照，旧20%放置解释另存。早先按正文切默认/拟去掉30步保护的过程记录不再代表当前默认。概率≥0.8及30步再剖析保护保持原策略；工程参数均标注，可后续改进。fail-stop默认关闭。

本次实际修复仅预热零基边界、500条历史原子上限和按迭代ID滚动消费，以及默认停机观测门控；22单测、真实Redis滚动、两版本控制器回放和真实Rbeast四类CPU测试通过。没有新GPU训练。最新版本须在新窗口重测数值、误触、召回和开销；不能承诺比旧版更优。当前权威入口为handoff.md和docs/implementation_status.md，验证及源码指纹为results/handoff-validation.json。
