# FLASH2 工作交接

2026-10-18 用户授权完整第三章复现，服务器.md包含私网凭据，不能输出/上传。

- 本地 /Users/henry/Desktop/FLASH2。两端独立开发目录 /home/campus/zhangweihao/FLASH1/flash2；不改现有FLASH1源码或vLLM。
- SSH `.venv/bin/python scripts/remote.py [--host 10.10.4.1] 'command'` 读私密文件使用paramiko。`scripts/sync.py`只上传白名单工程文件。
- 已创建两端 Docker `flash2-dev` 基于既有tianyuanwu/greyhound:ae，host网络、2G私有shm、目录挂载/workspace/flash2。Docker wrapper `/home/campus/zhangweihao/FLASH1/docker-gh`。node1 sudo wrapper、node2直连socket已可用。
- 运行时 torch2.1.0a0+32f93b1 CUDA12.2 NCCL2.19.3 Python3.10，nvcc/g++/redis/Rbeast可用。每H800只余2.3–5.5GB，不能停他人服务/降GPU时钟。
- 专用Redis 10.10.4.1:16389已在node1 flash2-dev启动。所有key `flash:{RUN}:...`，不能FLUSHDB。
- `flash/analysis.py`两阶段，`flash/detection.py`Rbeast/多重过滤，`flash/global_controller.py`状态机已写并上传，3项本地分析测试PASS。BEAST smoke persistent100→150成功变点。
- scripts/run_cluster.py 双端torchrun、环境、日志回收已写，待探针/训练完成后实际运行。
- 子agent nccl_probe 独占probe/实现C++；tiny_moe独占workload/train.py真实EP/TP/PP训练与验证；paper_requirements完成docs两份，正在workload/collective_bench.py和tests/test_detection.py。
- 探针ABI初版 header64 magic FLASHNCL mode24/rank28/write_seq32/heartbeat48/pid56，record128具体字段待agent报告。主窗口需写flash/local_controller.py。
- 下一步：完成local controller；sync/build两端；tiny EP双机16卡正确性smoke；修正错误；做有无probe对照+AllReduce4–256MB+持续/瞬态/轻微/停机故障+四拓扑，保存真实数据生成报告。不要仅安装后声称完成。

## 进度追加（实际测试阶段）
- connectivity16 两端exit0，16rank allreduce(1..16)=136 PASS，results本地已下载。
- probe/在两端已编译libflash_nccl_probe.so，ABI.md定稿；local_controller.py已实现读取和profile/failstop。11项unitPASS，实际Rbeast0.1.19 noisy normal/transient/weak/persistent integration PASS，persistent在故障45后观测48检测（3迭代）。集成文件在node1 results/rbeast-integration.json待取回。
- allreduce-base-a / allreduce-monitor-a 4–256MiB 3warmup8iters两端exit0；首轮4MiB监控+10.39%，其他约-0.6..1.8%，需新环境复测避免噪声误归因。
- verify-ep8 首次OOM（现有vLLM每卡只余2.3GB，NCCL多组/默认NVLS吃内存）；verify-ep8-lowmem设NVLS0、lazy、workspace小后不OOM但输出验证差3e-4..6e-4，梯度没报，疑NGC TF32默认。tiny_moe agent正在关闭TF32，不得放宽容差蒙混。
- scripts/run_cluster.py已加入低内存env，allreduce-profile 当前运行session11725 --probe --initial-profile。
- agent tiny_moe CPU单进程ep1 --verify2steps PASS，已修PP两个调度死锁；--output alias已支持。下一轮sync获取其TF32/mem日志/README后跑四topology verify，主窗口检查raw。
- 脚本scripts/stop_run.py只精确匹配自身--run-id和workload路径终止；失败实验已退出，无需停止别的任务。

## 进度追加（完整训练验证与故障矩阵）
- 四种真实16卡布局 `verify-ep8-tp1-pp1`, `verify-ep4-tp2-pp1`, `verify-ep4-tp1-pp2`, `verify-ep4-tp2-pp2` 全部两端exit0，64rank verification PASS；max output error3.87e-7，post-sync independent reference grads PASS，optimizer replica maxdiff0。
- 初次端到端 `ep8-compute-e2e` completed120steps exit0，BEAST触发、20sprofile完成；旧literal per-tag min结果[7,0]（真实注入0）。原因同步传播使rank0..7下游attentionDP都低latency，min7是微差。保留raw不改。
- `flash.analysis.diagnose`增加跨组消歧：保留score>=.2低latency集合，按独立comm交叉支持取唯一最佳，多候选并列则ambiguous。严格min版 `cross_validate=False` 同时在profile report `literal_minimum_diagnosis`保留。12项unitPASS，抽象传播测试与旧raw reanalysis定位0。docs/method.md明确这属于论文阈值/交叉验证的工程解释差异，不能冒充逐tag只取min的原始精度。
- 训练iteration JSONL新增wall_time+step_start_time，故障矩阵后可按profile时间分段；旧prelim无时间字段不硬算。
- 当前运行 `.venv/bin/python scripts/run_suite.py faults`，exec session93027，首轮normal。脚本依次normal/transient、compute-delay14/70/126/182/238、comm同五档、compute-othernode(rank9)、failstop(localexit前8s sleep)；每轮125steps padding250ms；单轮故障45开始；默认探针+controllers。故障模拟是本进程hostenqueue延迟，非物理降频/流量拥塞。弱14ms预期不触发，不应伪称召回100%。
- 后续尚待：faultsuite完成并检查异常；run_suite.py hybrid四拓扑故障；perf ABBA无padding180steps（baseline no-redis, monitor含callback）；collective ABBA 4–256MiB 8warm25iters；汇总/图表/报告/环境收尾。
- agent paper_requirements 正写scripts/summarize.py及结果图，读取partial可重跑。不要覆盖其脚本。其他agents完成。
- `scripts/capture_environment.py`已生成results/environment各host/container，image config sha256 2bd15284...；controlnumpy1.26.4 redis5.0.5 Rbeast0.1.19。低内存env见runner manifest，NVLS0 CUDA_MODULE_LOADING LAZY workspace16:8，禁用TF32。
- local_controller刚在本地追加profile计数timing_errors/unknowncomm/timed_samples，尚未sync（当前faults第一轮normal已启动，需下一轮前可同步flash）；运行controller已loaded旧版本也不影响。

## 进度追加（中后段矩阵，约18:17）
- 93027故障suite仍在运行，compute五档全完成：delay14实测+4.98%未检(FN)，70/126/182/238四档定位0正确。normal/transient125steps两端exit0且零trigger。comm14未检符合阈值，comm70/126/182已定位0；comm238已定位0等待收尾。后面会自动跑compute-othernode(rank9)和failstop(8s前停顿后exit)。不要并发启动其他GPU suite。
- 为提速回传，runner将probe快照/原始profile JSON无损gzip；每轮从~30s回传变几秒。原始统计/训练JSONL仍直接可读。现在可在run_suite后每60秒查简短global事件，不必大段dump。
- scripts/summarize.py和docs/results_summary.md已由agent交付，可用bundled Python执行更新summary.json/md（无matplotlib是实际情况）。脚本只有实际目录，无预填成功。已扩展comm_stream真值分类+`comm-stream-audit`纳入fault，避免隐藏额外通信限制。所有agents目前完成。
- scripts/validate_artifacts.py新增完成轮16rank/stepcount、finite loss/verification、CUDA错误/unknowncomm/lost/controller_error/profile_timeout审计。当前无完整性错误。早期OOM/TF32失败只diagnostics不冒充成功。
- scripts/telemetry.py 已加入runner，从后续轮采样总GPU memory/util/clock/power，每2秒，含共享任务。父controller停止stop-local也会停telemetry。
- 额外更严格通信测试：新probe/stream_delay.cpp及build_stream_delay.sh/STREAM_DELAY.md（agent写且两端已编译成功libflash_stream_delay.so）。LD_PRELOAD顺序baseprobe:overlay，RANK0每NCCL P2P group首op向其stream排hostcallback sleep，CUDA Event会计入。此测试不同于host pre-enqueue晚到代理，可能暴露min-latency定位算法的通信执行退化限制，需诚实计入结果。
- main已给workload/train.py加`--fault-kind comm_stream`，step45只写FLASH_STALL_FILE为fault_ms，之后overlay持续注入。runner新flag`--stream-delay`加overlay/环境并初始化stall-ms0；cleanup重置0+Redis mode0。源已sync，overlaylib已build两端。待fault suite结束后跑：`.venv/bin/python scripts/run_cluster.py comm-stream-audit --probe --controller --stream-delay --timeout 220 --args '--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind comm_stream --fault-ms 10'`，检查真实结果/根因。别给detector传groundtruth。
- 待测仍有：hybrid suite四拓扑故障；perf ABBA180step无padding；collective ABBA4-256MiB8warm25iters；profile-baseline强制剖析负控（125step250padding--force-profile-at45，无fault，单列不计自动误触）。若无padding自然MoE触发误报，别隐瞒/别用padding假称低开销，需查真实原因再决定符合论文哪个持久性条件。
- scripts/plot_summary.py已写静态2x2图和iteration-timeline图，需最终将summary.json上传node1 results/summary.json，在容器运行plot脚本（matplotlib3.8.0），然后SFTP取回PNG/SVG并view_image QA；未真正生成图。字段monitor_overhead_percent已匹配汇总结构。
- scripts/fetch_extras.py取回实际Rbeastintegration与allreduceprofile ring；results/probe-timing-validation.json已完成独立计时核对：两节点1280records/1248timed、0errors/0unknown，rank0每档探针与外部CUDAEvent差约0.05ms。Legacy基准NVLS设置不同，不混入最终ABBA开销。
- README.md、requirements-ops.txt/control.txt、scripts/prepare.py已准备复跑手册。正式结果中文报告尚未写；等所有数据收齐生成，明确tiny MoE-only/非Qwen/非真实物理故障/原文歧义/strict min对照/Greyhound同负载对照未做/短测非长时泛化。
- 最终需全tests/audit/summary/图QA/精确server自有worker清理确认/最终说明+summary归档上传。不要结束在“已安装”或pending测试状态。

## 最新追加（约18:25，前一段已过时以此为准）
- 93027 fault suite全部结束：13个普通轮两端exit0，failstop预期两端exit1。11个持续host-enqueue故障（compute五档+comm五档+othernode9）：9个强退化唯一命中正确root，两个约5%弱退化未触发。normal/transient零trigger。
- failstop约2.1277秒告警，但16rank全stalled，exited_process_ranks当时空，不能唯一root；这是真实限制，必须最终报告单列，不能称停机定位100%。注入root0先sleep8s再os._exit17，node1因通信abort退出-6，launcher均exit1属预期。
- comm-stream-audit初次加载失败（RTLD_NEXT找不到PyTorch RTLD_LOCAL的NCCL group symbol，发生在首个barrier、尚未注入）；修正overlay resolve_next为显式dlopen libnccl.so.2/.so+dlsym fallback后编译成功两端。保留失败原run在diagnostics。
- 真正stream trial `comm-stream-audit-r2`完成125steps两端exit0，自动触发profile，但最终no_localization candidate0..7；strict min错误指7。EP0 raw avg(us): root0=10135.62，others约10200，rankCV0.0025988<.1，无法分离；这是执行区间变慢与晚到假设不同的算法限制，计定位FN不假称复现错误或成功。docs/method.md已有解释。
- 发现cross_validate在只有一个comm证据时仍任意选min，已改成保留所有20%outlier candidates并列则ambiguous；新增单组cohort单测，现13tests PASS。strong host故障仍2comm唯一证据不受影响。README tests数已更新13。
- 当前唯一GPU工作：`.venv/bin/python scripts/run_suite.py hybrid` exec session72670，START hybrid-ep8-tp1-pp1（四种配置依次125steps250padding，fault0 compute200ms）。等完成并核查四拓扑真正故障定位。不要并发GPU suites。
- 剩余必须跑：run_suite.py perf（ABBA4轮180steps no padding），run_suite.py collective（ABBA4轮4–256MiB8warm25iters），profile-baseline（手动125steps250padding --force-profile-at45 无fault，用于baseline剖析负控，不计自动误触）。之后完整summaries/audit/tests/图/中文最终报告/服务器结果归档/自有GPU进程清理确认。
- summarize.py已识别comm_stream真值、`comm-stream-audit-r2`为fault；原失败`comm-stream-audit`为diagnostics。stop不混入fail-slow precision/recall，需最终正文额外解释alert1/1/root0/1。运行summary刷新命令见上。
- 图脚本plot_summary.py加了stream审计红X（在~36%强退化但unique=0），防只展示成功。已初步在server生成并view_image检查2x2布局；final下半两块尚无数据，所以必须最终所有结果后重画。timeline原来没生成因remote目录无node0层，已修成支持flat/node0双路径（尚未再次run）。summary.png/svg之前SFTP已取回，fetch iteration图片失败正因为该路径问题，修复已本地保存。
- 画图最终：sync脚本；SFTP summary.json 到 node1 root/results/summary.json；docker exec flash2-dev python scripts/plot_summary.py；SFTP四图summary.png/svg、iteration-timeline.png/svg取回；view_image两PNG QA。server有matplotlib3.8，Mac无，别本地安装/重新用imagegen。

## 完成交接（2026-10-18 18:48，以下为最终状态）

以上历史“待测/当前运行”均已完成，无后台 GPU 实验待收尾。

- 最终35个正式矩阵轮：34个正常完成，两端exit0；1个failstop预期非零退出。另保留10个诊断/负控轮。完整性审计共45个run，最终必要的数值、步骤、计时、丢记录与控制器检查无问题。15项单测通过，真实Rbeast集成通过。
- 四布局64个rank数值验证通过；四布局故障定位通过。持续故障/混合布局合计16轮：唯一命中13轮，两个约5%弱故障未触发20%门限，stream执行停顿无法唯一消歧，计FN。停机约2.13s告警但16rank传播停滞，无法唯一root。所有限制已进入README与summary，不修改原始数据。
- 监控态优化为metadata-only读取后，最终无padding ABBA完整周期19.75→20.40ms，短测+3.25%；单轮19.63/21.16ms。原始异常慢轮保留，不断言唯一原因。AllReduce4–256MiB四轮ABBA变化-2.22%至+0.89%，负值视为噪声。
- profile-baseline健康强制剖析负控完成，两端exit0，无慢卡候选、错误或丢记录。分析器按PP stage标记修正，16个已有profile重放，正式矩阵定位结论不变。
- results/summary.json/md、artifact-validation.json、profile-replay.json、probe-timing-validation.json、rbeast-integration.json、environment/、各run manifest和原始日志完整；summary与iteration-timeline PNG/SVG已生成并查看排版。README.md为用户入口，docs/method.md记录与原文解释差异。
- 两台宿主机已归档汇总、manifest及源码。2026-10-18T10:47:51Z确认两端flash2-dev均exited false；专用Redis随容器停止，zombie/共享内存一并清理。node1空闲显存2300–2468MiB/card，node2为5042–5210MiB/card，恢复共享环境基线。vLLM及其他服务未改动。结果保存在宿主机挂载目录。再次运行scripts/prepare.py会启动实验容器和Redis；新run ID复跑。
- 本地服务器.md凭据从未同步；无云发布/外部消息。最后状态凭据无关记录为results/final-state.json。

## 用户收敛后的专项完结（2026-10-18 19:05）

用户只关心常态/profile开销与计算/通信慢故障定位；停机方法暂不作为当前方案。新增scripts/run_profile_overhead.py与report_focus.py，1800步无padding健康ABBA四轮全部两端exit0，49run审计无完整性问题。results/focus-metrics.md/json为最新主评价入口，README已链接。

- 新ABBA同一步号匹配：常态2.01%，原短测3.25%作为历史参照；profile中位14.16%（单轮12.82%/15.50%）。profile完整事务均值10.41%/12.86%，累计多耗2.794/3.118秒。20秒采样后聚合/保存使事务27.38–29.64秒。两轮无唯一根因输出，但均有rank0–7歧义候选，诚实记录。
- 计算唯一正确9/10，其中超过20%门限9/9；通信晚到4/5、stream执行停顿0/1，合计4/6，其中超过门限4/5。两个弱故障未触发门限。默认faults不再包含failstop，--include-failstop为历史诊断opt-in，已用mock launch核对13/14jobs。运行期旧停机观测代码保留，未实现新方法。
- 新增四轮源SHA/环境/原始日志/manifest保留本地和服务器。再次停止两端flash2-dev并刷新results/final-state.json；复跑先prepare，run_profile_overhead.py与report_focus.py均支持--prefix避免覆盖。

## 新一轮按论文核对、共享约束下重测（进行中，2026-10-18 19:25左右）

用户先要求按论文故障方式测试，确认是排除fail-stop，随后新增“有其他进程时尽量不干扰，模拟即可”。已切回进程内代理，禁止继续真实GPU限频/背景流量。只做过paper-clock600-auto的GPU0应用时钟1965→600MHz物理诊断，1800step两端exit0、实测时延仅+1.68%、无触发；已恢复原2619/1980MHz并验证16卡1980。无任何背景流量启动。paper_clock_check/run_paper_clock/hardware_clock_injector/background_traffic现默认拒绝，显式opt-in才可启用，当前勿用。

- subagent traffic_method (gpt6luna max, 依服务器.md授权)完成网络只读调查，写scripts/background_traffic.py未启动流量，已结束。bond0为共享2x25Gb LACP，非一GPU一NIC，RoCE HCA存在但训练用Socket。已保存results/paper-network-preflight.json和paper-clock-preflight.json。
- 新增stream_delay的AllReduce拦截和FLASH_STALL_KIND=allreduce/FLASH_STALL_COMM_SIZE=2，仅自身rank0跨节点expertDP AllReduce流sleep callback。无额外流量/GPU核/硬件改动；默认p2p模式保留原实验。编译并pilot实测sim-dpcomm-pilot两端exit0，约33%退化自动触发，定位none；最高组信号[0,8]，组内rankCV0.000655<.1，无法唯一rank，无计时错误/未知comm/丢记录。
- 当前唯一GPU job：scripts/run_scoped_comm_suite.py，exec session24902；新sim-dpcomm-5、25、45已经完成，65正在跑（随后85）。每轮125step padding250，fault45开始，group delay1.75/8.75/15.75/22.75/29.75ms对应近似5/25/45/65/85%时延增加目标（非精确实测，更非吞吐下降），记录两种实际分母。不要并发GPU新suite。脚本内存中当前批次仍--literal-minimum；文件已改未来使用--paper-text-only，但未同步，统一离线重放以主规则评价。
- 复核发现旧analysis除CV外还把未定义20%门限放到clique_score/rank_score，论文无明确公式位置；新增diagnose(...paper_text=True)明确使用完整可读正文CV→最大组→CV→min，去掉未明确20%过滤。默认旧行为不变，旧literal_20pct与增强cross保留。global_controller新增--paper-text-only和paper_text_diagnosis/cross_validated_diagnosis/primary_rule；runner支持。17unitPASS，C++编译pilotPASS；新纸面模式尚未sync到服务器，不影响当前批已loaded任务。计算70/182的正文replay仍[7,0]，126/238为[0]，弱14无profile，所以唯一2/5，增强4/5，勿再把增强称作者原精度。
- scripts/report_scoped_faults.py已写，等待5个sim-dpcomm档齐后执行，主规则从真实profile统一离线replay paper_text，20%解释与enhanced分开JSON。五档计算复用compute-delay14/70/126/182/238，通信新五档，pilot单列不混主统计。还需补无故障forced-profile旧两轮原文min为[6]/[0]（扩展无唯一）的备注。
- 剩余收尾：等24902完成；report_scoped_faults；审计全部正常新轮；完成正文模式真实profile的控制面replay集成（不需GPU，可用独立Redis命名空间与真实nodes数据，强制trigger然后喂nodes，检查primary_rule/输出）；README主入口改scoped-paper-faults，方法记录补20%边界；可补probe实验clique得分用组内rank均值而非max_avg（runtime筛组按max，实验公式不同）；更新17tests与新报告归档列表；sync源码/root；停止我们两个容器清理SHM，capture_final_state，archive_summary（已有“无plot不删PNG”修复），验证不触碰vLLM。不要重新物理注入或共享链路traffic。

## 最新专项已完成（2026-10-18 19:30左右，覆盖上段进行中状态）

24902五档sim-dpcomm全部结束，两端exit0；加pilot共6轮新通信，56个训练run审计无数值/计时/未知comm/丢记录问题。计算五档复用原始raw。results/scoped-paper-faults.md/json为当前故障主入口；README已明确历史13/16为扩展代理指标，不是论文原方法精度。

- 新通信时延增加3.74/21.99/42.11/62.41/80.91%，实际吞吐下降3.60/18.03/29.63/38.43/44.72%，不冒充论文85%吞吐退化。弱档未触发。强档DP组[0,8]均最高，但rankCV约.0005–.001，无法唯一端点。
- 统一正文CV→max组→CV→min离线重放：计算TP4 FP2 FN1，precision66.67 recall80，唯一2/5；通信TP0 FP3 FN5，precision0 recall0，唯一0/5，误指1/7/3。扩展计算4/5，通信0/5拒判且无正预测，所以通信precision未定义，不写100%。旧20%placement解释字段保留；计算规则结果与旧min相同，通信65/85被旧20%组选过滤挡住，去掉后暴露伪根因，诚实记录。
- global_controller支持--paper-text-only，paper_text_diagnosis、legacy literal_20%与cross_validated并存；runner与未来scoped suite支持该flag。当前五档在线用旧literal20%（batch内存旧代码），报告是原始profile的统一正文重放，未覆盖原始在线输出。tests/paper_rule_integration.py用真实sim-dpcomm65 profile+独立Redis命名空间验证新CLI完整输出PASS，输出[7]，旧20%解释[]；此为控制面重放，不是GPU故障测试，receipt results/paper-rule-replay.json。首轮assert仅因JSON int/string键表示失败，normalize后全输出一致。
- 17单测PASS。未启动任何背景流量，所有GPU原应用时钟已恢复1980MHz。物理clock/共享link工具默认blocked，guards已静态验证。后续当前默认只用进程内代理。无需继续物理injector或新算法修改。
- 本段后只余例行收尾：capture_environment刷新新版so SHA；停止两个flash2-dev清理SHM；capture_final_state；最后sync --include-root并archive_summary（包含scoped-paper-faults、preflight、control replay、paper-fault metadata等）；final只需解释隔离代理范围与真实结果。常态/profile旧独立指标2.01–3.25%/14.16%不重测。图片summary.png保留（--no-plot不会再删）。

最终收尾补记：环境已刷新新版so SHA，2026-10-18T11:29:19Z两端flash2-dev已exited false，16卡SM1980MHz/空闲显存恢复原基线。新增run_scoped_fault_suite.py提供两类十点的新prefix复跑，report_scoped_faults的--prefix覆盖两类，若只重跑通信可--compute-prefix ''复用旧计算。正文重放的计算误判证据来自count1024的实际router梯度同步，通信误判来自EP组，不是count1的额外barrier直接被当作证据；仍需保留toy训练有控制同步、模拟并非真实拥塞等边界，不能断言作者算法真实硬件精度为本表数值。最后源码/root与66+汇总归档上传后无待执行GPU工作。


## 2026-10-18 最终实验收尾前交接快照

# FLASH 交接（2026-10-18）

## 最新收尾：定位与3.4.6（2026-10-18）
用户确认保留FLASH工程扩展版；正文规则只作同profile重放对照。因共享vLLM，用户再次选择进程内代理；通信最后补node1 rank9一轮后停止，后续仅补计算不同位置与混合并行。当前全部已结束，两端flash2-dev已停，显存恢复、应用时钟1980MHz；final-state已刷新。不要自动启动新实验。

- **定位矩阵** `results/loc-r1-localization.md/json`：13轮（2负控、7计算含1混合、4通信），全部16rank×150步、两端exit0；31,200个rank迭代及9份profile验收通过，源码匹配新smoke、无计时错误/未知组/丢样，见`loc-r1-validation.json`。计算目标覆盖rank0/3/9/10/15：扩展唯一6/7，正文4/7；唯一弱档实测+2.75%未过门限，超过20%的计算扩展6/6、正文4/6。正常/800ms单瞬态无自动profile（含250ms padding，不能宣称解决无padding误触）。
- **通信边界**：弱档+4.65%未触发；其余三轮+42.16/+81.73/+38.41%，均检测并命中[0,8]或[1,9]组，扩展唯一端点0/4（门限以上0/3），无正式错误rank；正文中/强rank0轮误报7/11。仅rank0注入的一轮，所有16rank迭代都慢约42%，不是16个根因。传播证据`loc-r1-communication-propagation.json`。这些不是实际背景流量/限频结果，不再追加通信。
- **3.4.6** `results/hyb346-r1-hybrid-346.md/json`：EP8/TP1/PP1、EP4/TP2/PP1、EP4/TP1/PP2、EP4/TP2/PP2均rank0、200ms计算晚到、150步，扩展唯一4/4；正文2/4，PP两配置额外报[4,5]与[11]。三轮新测、一轮复用`loc-r1-hybrid-compute`；四拓扑各16rank数值验证通过（复用配置另跑3步verify），完整性`hyb346-r1-validation.json`。定位约22.92–23.25秒。
- **累积分数与限制**：按式3-2逐comm累计、扩展筛选后累计、角色数/去重comm数分开。扩展原生分为1.9901/2.8958/3.8565/2.8558，不冒充论文原始组得分/误差线。一次配置无重复标准差；玩具无attention/独立SP、跨拓扑有效batch不同。当前角色仅按成员匹配，存在EP/DP别名及special控制证据，不强行匹配论文2/4/4/5组。
- **后续改进** `docs/flash_improvement_log.md` 已记录F01–F07：通信组件输出粒度、通信执行慢的最小rank假设、多层真值/指标、启动触发、profile生命周期、正文额外计算rank、组语义与累计口径。GreyHound方案有P2P主动验证，当前checkout树验证未启用/结果聚合；未在同负载跑基线，不声称实测优劣。后续修算法时固定这些原始profiles回放，不调阈值凑命中。

## 3.4.3 本次测试收尾（2026-10-18）
用户明确要求结束测试；不要自动继续剩余矩阵。两端flash2-dev已停止，显存恢复，时钟1980MHz；`results/final-state.json`已刷新。

- 最新源码16rank数值smoke全通过，容器22单测通过。监控ABBA：`s343-r1-monitor-{0-base,1-flash,2-flash,3-base}`，每轮250步，无padding；剔除零基0–4，rank0相邻step_start_time完整周期均值21.0693→21.3105ms，时延开销+1.1449%，吞吐变化-1.1319%。两轮纯monitor无自动profile。
- `s343-r1-healthy`：1800步约39.385秒，自动剖析0次、历史漏读0。但另一个无故障轮`s343-r1-profile-1-flash`出现1次自动误触：变点iteration12（零基），触发时已完成step50；随后强制剖析延迟至step1301。该轮有自动/强制混合且后段训练覆盖不足，排除受控剖析开销，原始数据已恢复归档。
- 新增`--force-profile-only`到global/runner，仅用于隔离受控剖析开销，需同时force-profile-at；默认自动检测不变。`s343-r2-profile-0-base`完成，`s343-r2-profile-1-flash`由用户结束而主动终止，两端exit1，manifest已注明原因；不报有效剖析开销。
- AllReduce仅完成`s343-r1-collective-calibration`：8次预热、100次测量/批100，六档约3.02/5.84/10.64/21.08/41.15/79.96ms。只是关闭FLASH的校准，不能计算探针开销。论文500预热/10000次的完整开关对照尚未开展。
- 新增`workload/collective_bench.py --batch-size`及CUDA/host两种批均值；`scripts/run_overhead_343.py`提供monitor/healthy/profile/collective串行矩阵，微基准默认论文开关一对，额外`--collective-abba`可补两轮；统计脚本`report_overhead_343.py`，方法`docs/performance_343.md`。原始结果和两版provenance在results/s343-r*；r1与r2源码差异只含受控开关/runner，训练执行源码远端核对一致。

## 用户最终决定
效果优先，默认保留 **FLASH扩展版：跨组候选消歧**，标明与论文差异。正文定位基线用 `--paper-text-only`；扩展可显式加 `--extended-cross-validation`。禁止混算两版指标。排除fail-stop，保护共享vLLM，只做本训练进程内故障代理，不再限频或启动共享链路流量。

## 环境与状态
本地 `/Users/henry/Desktop/FLASH2`；服务器 `10.10.4.1/.2`，远程均 `/home/campus/zhangweihao/FLASH1/flash2`。SSH用 `scripts/remote.py`，凭据只读本地 `服务器.md`，不得打印/上传。容器 `flash2-dev`，Redis节点0端口16389；离线GREYHOUND镜像，Torch2.1/NCCL2.19、Socket/bond0。每卡可用显存约2–5GiB。收尾容器已停、无GPU实验，时钟恢复1980MHz；复跑先prepare。当前非Git仓库，按manifest源码SHA追溯。

## 最新代码与验证
核心已实现：NCCL探针/共享环、监控与CUDA剖析切换、Redis双层控制、两阶段定位、真实EP/TP/PP小MoE。
本次修复：预热准确跳过零基0/1/2；Redis原子保留500条，按迭代ID读取，避免滚动后漏读；停机默认关闭。保留原概率≥0.8和30步再剖析保护（工程参数，可调），未为字面对齐改坏默认扩展。22单测、真实Redis滚动、两版控制器回放、Rbeast四案例通过；凭据无关证据 `results/handoff-validation.json`。
**最新修订已完成新smoke、13轮代理定位、四拓扑计算代理与数值验证，详见最上方收尾。全量AllReduce、受控剖析开销、真实物理注入精度仍未完成；不能用代理结果替代。** 旧四拓扑64rank数值验证及56run完整性审计通过，不能代替新版本验证。

## 历史结果及限制
同样五档计算晚到：扩展唯一4/5，正文解释2/5；此前9/10还包括异机rank及四拓扑。新跨节点通信执行迟滞两版均0/5；强档能筛中[0,8]组，但两端时延近似，不能唯一端点；扩展减少误判，不等于定位成功。历史常态开销2.01–3.25%、profile中位14.16%，均为旧版本共享环境短测。
模型0.526M、MoE-only无attention。计算用入队前等待，通信用size2跨节点AllReduce流callback等待；不是论文物理降频/背景拥塞。Rbeast实际是BEAST；论文20%离群公式不全，额外保护/消歧均标注工程选择。详见 `docs/implementation_status.md`、`results/scoped-paper-faults.md`、`results/focus-metrics.md`。不要声称作者源码完全还原或真实拥塞精度已验证。

## 用户再次明确要求复测时
```bash
.venv/bin/python scripts/sync.py --include-root
.venv/bin/python scripts/prepare.py
.venv/bin/python scripts/run_cluster.py win2-smoke --probe --args '--ep 8 --verify --steps 3 --min-step-ms 0 --no-redis'
```
smoke通过后先测正常/瞬态，再按需故障和开销。单轮命令见README；十点代理矩阵 `scripts/run_scoped_fault_suite.py --prefix win2-`，报告 `scripts/report_scoped_faults.py --prefix win2-`；开销 `scripts/run_profile_overhead.py --prefix win2-`。均用 `.venv/bin/python`。换新ID/prefix，不覆盖旧记录，不并发GPU矩阵。对照两版优先用同份profile重放，固定样本、故障语义和分母，仅切定位规则；同时报误判、召回、组命中和唯一rank。若通信仍失败，先验证模拟时序/可辨识性，再改算法，勿调阈值凑命中。

完成后只停止自己的flash2-dev、刷新final-state并归档。详细旧过程已移至 `docs/handoff-history-20261018.md`，通常无需先读。
