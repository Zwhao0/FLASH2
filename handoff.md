# FLASH 最终交接（2026-10-18）

## 决策与状态
默认保留**FLASH工程扩展版（跨组候选消歧）**；`--paper-text-only`仅作同份profile对照，禁止混算指标。用户确认本轮为最后实验，已全部结束；两端`flash2-dev`已停，显存恢复、时钟1980MHz。不要自动复测。保护共享vLLM，排除fail-stop，不做物理限频/背景流量；当前结果均为进程内代理。

## 最终结果
- **定位/位置矩阵**：13轮，16rank×150步；7个计算代理（含1混合）、4个通信代理、2负控。计算覆盖rank0/3/9/10/15，扩展唯一6/7、正文4/7；实测超过20%的计算为6/6与4/6。弱档+2.75%未触发。正常/单次瞬态均无自动profile，但含250ms固定等待，不代表修复了无padding误触。
- **通信已停止**：弱档+4.65%未触发，其余3轮均检测、命中[0,8]或[1,9]组，但均不能唯一识别注入端点。扩展无正式错误rank，正文两轮误报7/11。只在rank0注入的一轮，16rank迭代均慢约42%：注入点、目标组、受影响集合必须分开。不是实际背景拥塞精度。
- **3.4.6计算慢**：EP/TP/PP=8/1/1、4/2/1、4/1/2、4/2/2，统一rank0每步200ms入队前等待。扩展唯一4/4、正文2/4；正文PP配置额外报[4,5]及[11]。三轮新测、一轮复用；四拓扑共64rank数值验证通过，定位22.92–23.25秒。
- **3.4.3**：250步监控ABBA完整周期21.069→21.311ms，时延+1.14%、吞吐−1.13%。独立1800步无触发，但另一个无注入轮自动触发1次；长期零误触未验证。受控profile开销/全量AllReduce对照未完成。

## 证据入口
- `results/loc-r1-localization.md/json`、`loc-r1-validation.json`：13轮、31,200个rank迭代及9份profile验收通过。
- `results/hyb346-r1-hybrid-346.md/json`、`hyb346-r1-validation.json`：四拓扑、数值验证及逐组分数；复用轮`loc-r1-hybrid-compute`，勿重复算独立样本。
- `results/loc-r1-communication-propagation.json`：多rank传播；`results/s343-r1-overhead-343.md`：开销收尾；`results/final-state.json`：最终状态。各run的manifest保留源码SHA与原始输出。

## 新窗口：系统梳理三套方案，先不实验
对照 **GREYHOUND论文/当前checkout、中文论文FLASH、我们的FLASH扩展复现**，逐项分清“论文声称、代码实现、实测证据”：检测输入与算法、状态机/主动验证、分组与定位规则、输出粒度、注入真值、开销与局限。

先读`docs/flash_improvement_log.md`（F01–F07），再查`docs/chapter3_requirements.md`、`implementation_status.md`、`method.md`、`greyhound_reuse.md`；原文为根目录两份PDF，GreyHound源码在`third_party/Greyhound/detector/control_plane/`。

关键边界：Rbeast实际是BEAST；论文20%离群公式不全。GreyHound方案有暂停训练后的P2P验证，但当前checkout树验证未启用，结果聚合不能直接当NIC级定位；本轮未同负载跑GreyHound。FLASH图3-9未拆分计算/通信结果。当前约0.526M MoE-only、无attention/独立SP；组角色按成员匹配，EP/DP别名、special控制通信、角色数与comm哈希数需区分。全组式3-2累加与扩展筛选后得分不同；每设置一次，不制造标准差、不强行对齐论文组数/分数。

## 环境
本地`/Users/henry/Desktop/FLASH2`；服务器10.10.4.1/.2，远程`/home/campus/zhangweihao/FLASH1/flash2`。SSH用`scripts/remote.py`；凭据只读本地`服务器.md`，不得打印或上传。已纳入Git，首版标签`v1`；Git提交与实验manifest SHA分别追溯版本和实际运行源码。历史详情归档`docs/handoff-history-20261018.md`。
