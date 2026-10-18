# 进程内代理故障定位复核

矩阵：results/loc-r1-matrix.json。范围：Process-scoped timing proxies; not physical frequency restriction or network congestion.

主结果使用 FLASH 扩展规则（跨组候选消歧）；论文正文规则按同一聚合 profile 离线重放作对照。严重度仅由 rank0 step 10–40 与 50–149 中位迭代时长计算；20%子集按实测变化筛选。计划延迟只作元数据和故障日志核对，不用于定位精度。

## 单轮结果

| Run | 状态 | 类别/kind | 目标rank | 故障验收 | 基线→稳态ms | 时延增幅 | 吞吐下降 | 自动触发 | 有效profile | 正文输出 | 扩展输出 | 扩展歧义候选 |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|
| loc-r1-normal | completed | normal/normal | — | 是 | 277.17→276.23 | -0.34% | -0.34% | 0 | 0（not_triggered） | [] | [] | — |
| loc-r1-transient | completed | transient/transient | 0 | 是 | 282.52→282.52 | 0.00% | 0.00% | 0 | 0（not_triggered） | [] | [] | — |
| loc-r1-compute-weak | completed | compute/compute | 0 | 是 | 282.07→289.81 | 2.75% | 2.67% | 0 | 0（not_triggered） | [] | [] | — |
| loc-r1-compute-medium | completed | compute/compute | 9 | 是 | 277.63→404.21 | 45.59% | 31.31% | 1 | 1（valid） | [9] | [9] | — |
| loc-r1-compute-strong | completed | compute/compute | 0 | 是 | 278.05→514.92 | 85.19% | 46.00% | 1 | 1（valid） | [0] | [0] | — |
| loc-r1-comm-weak | completed | communication/comm_stream | 0 | 是 | 277.06→289.94 | 4.65% | 4.44% | 0 | 0（not_triggered） | [] | [] | — |
| loc-r1-comm-medium | completed | communication/comm_stream | 0 | 是 | 276.97→393.74 | 42.16% | 29.66% | 1 | 1（valid） | [7] | [] | [5, 7] |
| loc-r1-comm-strong | completed | communication/comm_stream | 0 | 是 | 278.20→505.56 | 81.73% | 44.97% | 1 | 1（valid） | [11] | [] | [9, 11, 12, 13, 14, 15] |
| loc-r1-hybrid-compute | completed | hybrid/compute | 0 | 是 | 274.80→470.54 | 71.23% | 41.60% | 1 | 1（valid） | [0, 11] | [0] | — |
| loc-r1-comm-node1-final | completed | communication/comm_stream | 9 | 是 | 284.64→393.96 | 38.41% | 27.75% | 1 | 1（valid） | [] | [] | — |
| loc-r1-compute-node0-rank3 | completed | compute/compute | 3 | 是 | 277.59→403.77 | 45.46% | 31.25% | 1 | 1（valid） | [3] | [3] | — |
| loc-r1-compute-node1-rank10 | completed | compute/compute | 10 | 是 | 276.51→403.50 | 45.93% | 31.47% | 1 | 1（valid） | [10, 15] | [10] | — |
| loc-r1-compute-node1-rank15 | completed | compute/compute | 15 | 是 | 277.69→405.40 | 45.99% | 31.50% | 1 | 1（valid） | [15] | [15] | — |

## 各类与总体定位指标

主指标只统计 identified_ranks 正式输出：TP为输出包含真实rank，FP为输出错误rank数，FN为未输出目标。扩展规则拒判时，即使歧义候选含目标也不算定位TP。候选覆盖另存JSON，不与正式预测混算；多个正式输出含目标仍非唯一正确。pending与有触发但profile无效的轮次不伪造成FN；唯一正确/全部计划注入保留计划分母，并单列pending。

### 扩展规则（主结果）

| 类别 | TP/FP/FN | Precision | Recall | 唯一正确/全部计划注入 | 可评估 | >20%唯一正确/实测>20% | >20% TP/FP/FN | Pending |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| communication | 0/0/4 | 未定义（无正预测） | 0.0% | 0/4 | 4 | 0/3 | 0/0/3 | 0 |
| compute | 5/0/1 | 100.0% | 83.3% | 5/6 | 6 | 5/5 | 5/0/0 | 0 |
| hybrid | 1/0/0 | 100.0% | 100.0% | 1/1 | 1 | 1/1 | 1/0/0 | 0 |
| all | 6/0/5 | 100.0% | 54.5% | 6/11 | 11 | 6/9 | 6/0/3 | 0 |

### 正文规则重放

| 类别 | TP/FP/FN | Precision | Recall | 唯一正确/全部计划注入 | 可评估 | >20%唯一正确/实测>20% | >20% TP/FP/FN | Pending |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| communication | 0/2/4 | 0.0% | 0.0% | 0/4 | 4 | 0/3 | 0/2/3 | 0 |
| compute | 5/1/1 | 83.3% | 83.3% | 4/6 | 6 | 4/5 | 5/1/0 | 0 |
| hybrid | 1/1/0 | 50.0% | 100.0% | 0/1 | 1 | 0/1 | 1/1/0 | 0 |
| all | 6/4/5 | 60.0% | 54.5% | 4/11 | 11 | 4/9 | 6/4/3 | 0 |

## 通信定位粒度（扩展版）

注入rank是代理控制点；对应size2组包含两个端点。命中目标组与唯一识别注入端点分开评价，不把所有受影响rank都当根因。此处目标组按当前EP8/TP1/PP1、expert-DP=2的拓扑定义，不是物理链路真值。

| Run | 注入rank | 目标组 | 触发 | 有效profile | 阶段一命中目标组 | 正式输出 | 唯一端点正确 |
|---|---:|---|---|---|---|---|---|
| loc-r1-comm-weak | 0 | [0, 8] | False | False | False | [] | False |
| loc-r1-comm-medium | 0 | [0, 8] | True | True | True | [] | False |
| loc-r1-comm-strong | 0 | [0, 8] | True | True | True | [] | False |
| loc-r1-comm-node1-final | 9 | [1, 9] | True | True | True | [] | False |

## 自动检测和时延

时延从 fault_injected.wall_time 起算；profile完成取同epoch终结事件，唯一线上定位要求 identified_ranks 恰等于单一真实rank。controller_stopped缺失单独留档，不抹去已有事件。

| Run | 注入→触发ms | 注入→profile完成ms | 注入→唯一线上定位ms | controller_stopped | 输出数 |
|---|---:|---:|---:|---|---:|
| loc-r1-normal | — | — | — | 是 | 0 |
| loc-r1-transient | — | — | — | 是 | 0 |
| loc-r1-compute-weak | — | — | — | 是 | 0 |
| loc-r1-compute-medium | 1690.71 | 22819.09 | 22819.09 | 是 | 1 |
| loc-r1-compute-strong | 1692.44 | 22803.11 | 22803.11 | 是 | 1 |
| loc-r1-comm-weak | — | — | — | 是 | 0 |
| loc-r1-comm-medium | 1745.09 | 22849.67 | — | 是 | 1 |
| loc-r1-comm-strong | 1669.03 | 22772.49 | — | 是 | 1 |
| loc-r1-hybrid-compute | 2102.21 | 22924.79 | 22924.79 | 是 | 1 |
| loc-r1-comm-node1-final | 1448.65 | 22553.98 | — | 是 | 1 |
| loc-r1-compute-node0-rank3 | 1787.66 | 22752.50 | 22752.50 | 是 | 1 |
| loc-r1-compute-node1-rank10 | 1753.79 | 22807.98 | 22807.98 | 是 | 1 |
| loc-r1-compute-node1-rank15 | 1451.85 | 22413.51 | 22413.51 | 是 | 1 |

## 正常与瞬态控制

正常与瞬态不进持续故障TP/FN分母；自动剖析次数和任何rank输出分列。pending仅表示当前证据不完整。

| Run | 状态 | 自动剖析次数 | SLOW_RANK_IDENTIFIED | no_localization | controller_stopped |
|---|---|---:|---|---:|---|
| loc-r1-normal | completed | 0 | [] | 0 | 是 |
| loc-r1-transient | completed | 0 | [] | 0 | 是 |

## Profile验收和通信组信号

有效profile须有global触发与同epoch终结、聚合profile、两端local上报的timed_samples>0/timing_errors=0/lost=0，且rank0时间覆盖触发至完成。无效profile仍保留触发记录。最高clique_score只作组信号；只有同tag/hash有rank_scores与rank_cv才记为通过阶段一。

| Run/epoch | 状态 | 原因 | 正文各tag最大信号；通过I组 | 扩展各tag最大信号；通过I组 | 目标 expert-DP rank对及CV 正文/扩展 |
|---|---|---|---|---|---|
| loc-r1-normal / — | not_triggered | 无 profile_trigger | — | — | — |
| loc-r1-transient / — | not_triggered | 无 profile_trigger | — | — | — |
| loc-r1-compute-weak / — | not_triggered | 无 profile_trigger | — | — | — |
| loc-r1-compute-medium / 1 | 有效 | — | expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.974 target=是; attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.949 target=是; expert_dp [1, 9] score=0.160 target=是；expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.3738 target=是, attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9812 target=是 | expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.974 target=是; attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.949 target=是; expert_dp [1, 9] score=0.160 target=是；expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.3738 target=是, attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9812 target=是 | — / — |
| loc-r1-compute-strong / 1 | 有效 | — | expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.988 target=是; attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.969 target=是; expert_dp [4, 12] score=0.086 target=否；expert_ep [0, 1, 2, 3, 4, 5, 6, 7] rankCV=0.3757 target=是, attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9927 target=是 | expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.988 target=是; attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.969 target=是; expert_dp [4, 12] score=0.086 target=否；expert_ep [0, 1, 2, 3, 4, 5, 6, 7] rankCV=0.3757 target=是, attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9927 target=是 | — / — |
| loc-r1-comm-weak / — | not_triggered | 无 profile_trigger | — | — | — |
| loc-r1-comm-medium / 1 | 有效 | — | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.000 target=是; expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.203 target=是; expert_dp [0, 8] score=5.648 target=是；expert_ep [0, 1, 2, 3, 4, 5, 6, 7] rankCV=0.3559 target=是, expert_dp [0, 8] rankCV=0.0011 target=是 | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.000 target=是; expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.203 target=是; expert_dp [0, 8] score=5.648 target=是；expert_ep [0, 1, 2, 3, 4, 5, 6, 7] rankCV=0.3559 target=是, expert_dp [0, 8] rankCV=0.0011 target=是 | [0, 8]:0.0011 / [0, 8]:0.0011 |
| loc-r1-comm-strong / 1 | 有效 | — | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.007 target=是; expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.495 target=否; expert_dp [0, 8] score=6.187 target=是；expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.8948 target=否, expert_dp [0, 8] rankCV=0.0005 target=是 | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.007 target=是; expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.495 target=否; expert_dp [0, 8] score=6.187 target=是；expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.8948 target=否, expert_dp [0, 8] rankCV=0.0005 target=是 | [0, 8]:0.0005 / [0, 8]:0.0005 |
| loc-r1-hybrid-compute / 1 | 有效 | — | special [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.000 target=是; ppstage0_expert_ep [0, 1, 2, 3] score=0.006 target=是; ppstage0_attention_dp [0, 1, 2, 3] score=0.006 target=是; ppstage0_expert_tp [0, 4] score=2.871 target=是; pp [0, 8] score=1.005 target=是; ppstage1_expert_ep [8, 9, 10, 11] score=0.147 target=否; ppstage1_attention_dp [8, 9, 10, 11] score=0.147 target=否; ppstage1_expert_tp [11, 15] score=0.372 target=否；special [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.5844 target=是, ppstage0_expert_tp [0, 4] rankCV=0.9975 target=是, pp [0, 8] rankCV=0.9053 target=是, ppstage1_expert_ep [8, 9, 10, 11] rankCV=0.1926 target=否, ppstage1_attention_dp [8, 9, 10, 11] rankCV=0.1926 target=否, ppstage1_expert_tp [11, 15] rankCV=0.8444 target=否 | special [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.000 target=是; ppstage0_expert_ep [0, 1, 2, 3] score=0.006 target=是; ppstage0_attention_dp [0, 1, 2, 3] score=0.006 target=是; ppstage0_expert_tp [0, 4] score=2.871 target=是; pp [0, 8] score=1.005 target=是; ppstage1_expert_ep [8, 9, 10, 11] score=0.147 target=否; ppstage1_attention_dp [8, 9, 10, 11] score=0.147 target=否; ppstage1_expert_tp [11, 15] score=0.372 target=否；special [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.5844 target=是, ppstage0_expert_tp [0, 4] rankCV=0.9975 target=是, pp [0, 8] rankCV=0.9053 target=是, ppstage1_expert_tp [11, 15] rankCV=0.8444 target=否 | — / — |
| loc-r1-comm-node1-final / 1 | 有效 | — | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.001 target=是; expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.015 target=否; expert_dp [1, 9] score=5.676 target=是；expert_dp [1, 9] rankCV=0.0003 target=是 | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.001 target=是; expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.015 target=否; expert_dp [1, 9] score=5.676 target=是；expert_dp [1, 9] rankCV=0.0003 target=是 | [1, 9]:0.0003 / [1, 9]:0.0003 |
| loc-r1-compute-node0-rank3 / 1 | 有效 | — | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.949 target=是; expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.935 target=是; expert_dp [4, 12] score=0.139 target=否；attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9899 target=是, expert_ep [0, 1, 2, 3, 4, 5, 6, 7] rankCV=0.3729 target=是 | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.949 target=是; expert_ep [0, 1, 2, 3, 4, 5, 6, 7] score=0.935 target=是; expert_dp [4, 12] score=0.139 target=否；attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9899 target=是, expert_ep [0, 1, 2, 3, 4, 5, 6, 7] rankCV=0.3729 target=是 | — / — |
| loc-r1-compute-node1-rank10 / 1 | 有效 | — | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.945 target=是; expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.960 target=是; expert_dp [2, 10] score=0.149 target=是；attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9830 target=是, expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.3604 target=是 | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.945 target=是; expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.960 target=是; expert_dp [2, 10] score=0.149 target=是；attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9830 target=是, expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.3604 target=是 | — / — |
| loc-r1-compute-node1-rank15 / 1 | 有效 | — | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.925 target=是; expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.969 target=是; expert_dp [7, 15] score=0.225 target=是；attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9779 target=是, expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.3729 target=是, expert_dp [7, 15] rankCV=0.0706 target=是 | attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] score=0.925 target=是; expert_ep [8, 9, 10, 11, 12, 13, 14, 15] score=0.969 target=是; expert_dp [7, 15] score=0.225 target=是；attention_dp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.9779 target=是, expert_ep [8, 9, 10, 11, 12, 13, 14, 15] rankCV=0.3729 target=是, expert_dp [7, 15] rankCV=0.0706 target=是 | [7, 15]:0.0706 / [7, 15]:0.0706 |

JSON保留完整matrix、manifest/source_sha256、rank步数验收、事件、local/profile完整性及逐profile diagnose scores/ranking。
