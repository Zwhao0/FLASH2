# Greyhound 的 Rbeast 调用：可复用部分与 FLASH 差异

通信组件定位的后续核查见 [FLASH 改进记录](flash_improvement_log.md#与-greyhound-比较的证据边界)：GreyHound 论文的组筛选后 P2P 验证、当前 checkout 的聚合输出，以及事件检测准确率与链路定位准确率的区别。下文仍专注 Rbeast 接入，不把 GreyHound 的主动验证混写为 FLASH 已有机制。

本文检查仓库文件 `third_party/Greyhound/detector/control_plane/slow_detection.py`、调用处 `local_analyzer.py` 和依赖声明 `detector/setup.py`。目标是说明 Rbeast 变点处理可借鉴的部分以及其与 FLASH 第三章描述的不同；本文不实现控制器。

## Greyhound 当前代码做什么

`local_analyzer.detect_failslow()` 对每个 rank 按 event id 排序，从 NCCL `call_number` 序列用 ACF 推断重复周期（`nlags=200`、显著峰高度 `0.95`），然后调用 `find_performance_drop()`。Rbeast 并不直接接收框架迭代时间；代码先用推断周期构造时间序列：

```python
ts.append(call_time[i + period] - call_time[i])
iter_start.append(call_time[i])
```

随后移除 `abs(z-score) > 3.0` 的样本，并对剩余的 `ts` 调用：

```python
rb.beast(
    ts,
    season='none',
    print_options=False,
    print_progress=False,
    quiet=True,
    hasOutlier=True,
)
```

调用没有传 `thresh_prob`，因此使用 `find_performance_drop()` 的 Python 默认值 `0.8`。Rbeast 的算法参数（例如 trend changepoint 数量先验、MCMC 设置）也没有在这里显式指定，使用所安装版本的包默认值。依赖仅约束为 `Rbeast>=0.1.19`，没有固定版本。

结果处理方式：

1. 取 `result.trend.ncp_mode[0]` 作为 changepoint 数量，再读取 `result.trend.cp` 和 `result.trend.cpPr`。
2. 对候选点按 `cpPr > 0.8` 保留（代码在 `<= 0.8` 时过滤）。
3. `validate_performance_drop()` 默认要求变点间隔大于 10 个样本；首个变点若位于序列前 10 个位置会被剔除。
4. 对保留的点比较变点前后均值，计算 `abs(mean_l - mean_r) / max(mean_l, mean_r)`；变化量 `<= 0.1` 时过滤，因此默认需要超过 10% 才报告。

`last_10_avg_ts` 变量实际计算 `mean(ts[-2:])`，是最近两个样本的平均值；它不参与变点筛选，变量名与计算窗口不一致。

## 与 FLASH 论文的对应关系

| 方面 | Greyhound 仓库代码 | FLASH 第三章描述 |
|---|---|---|
| 数据源 | NCCL 调用序列；先从 `call_number` 的重复模式用 ACF 推断周期，再按周期计算调用时间差 | 框架在每个迭代结束时直接上报迭代耗时；以 Rank 0 序列作为 BOCD 输入 |
| 变点实现 | 对当前构造出的有限 `ts` 调用 `Rbeast.beast()`，读取 BEAST 的 trend changepoints 和 `cpPr` | 文字称使用 Rbeast 实现 BOCD，并以 `P(r_t=0 | x_{1:t})` 越阈值判变点；未给出具体调用和输出字段 |
| 后验阈值 | `thresh_prob=0.8`，本仓库调用依赖函数默认值 | `thresh_prob=0.8` |
| 异常值 | 输入 Rbeast 前移除 `|z| > 3.0` | BOCD 前剔除 `z-score > 3.0`；文本未明确是否取绝对值 |
| 变化幅度 | `abs(mean_l-mean_r)/max(mean_l,mean_r) > 0.1` | `degradation_thresh=0.2`，并另有恢复比率 `0.75` 过滤；下降比例的基线定义未给出 |
| 相邻变点约束 | 默认间隔严格大于 10 个样本，首个 10 点内变点丢弃 | 小于 2 个迭代的相邻变点去重；另一个段落报告持续超过 20 个迭代，和 1–2 迭代右侧保护区存在未解口径冲突，见 [chapter3_requirements.md](chapter3_requirements.md#6-原文歧义和复现前需核对项) |
| 预热处理 | `slow_detection.py` 不跳过前 3 个训练迭代 | 前 3 个迭代跳过；使用 30 迭代窗口 |

## 关于 BOCPD 命名的论文歧义

仓库调用的 `Rbeast` 项目将其算法命名为 **BEAST**（Bayesian Estimator of Abrupt change, Seasonality, and Trend），描述为对多种趋势/季节/突变分解模型进行 Bayesian model averaging。Python 接口 `beast()` 返回趋势项及 changepoint 相关字段，例如 `ncp_mode`、`cp`、`cpPr`。`season='none'` 让模型按无季节项的趋势/突变模式处理序列，`hasOutlier=True` 则启用模型中的 outlier 分量。[Rbeast 官方说明](https://github.com/zhaokg/Rbeast)；[Python `beast()` 接口说明](https://github.com/zhaokg/Rbeast/blob/master/Python/py_src/Rbeast/beast.py)

Adams–MacKay 的严格 BOCPD 则是逐观测递推当前 run length（距上一个变点的长度）后验分布 `P(r_t | x_{1:t})` 的在线算法。[原论文](https://arxiv.org/abs/0710.3742) 明确以该 run-length 分布为核心。Greyhound 这里没有实现这套 run-length 递推；它对窗口序列调用 BEAST，再依据 BEAST 输出的 `trend.cpPr` 过滤候选点。因此，论文把 Rbeast 调用概括为“BOCD”与仓库中实际使用的 BEAST 接口之间存在术语/算法描述歧义。不能仅凭参数名相似就把 Rbeast 的 `cpPr` 等同于 Adams–MacKay 的 `P(r_t=0 | x_{1:t})`。

FLASH 论文也提到 Rbeast 和 run-length 后验，但没有提供调用代码、输出字段或推断配置。因此应将两件事分开记录：FLASH 的论文目标/描述是在线 BOCD 式决策；Greyhound 现有代码展示的是 BEAST changepoint 模型输出的使用方式。若复现要沿用该库，需明确把 BEAST 当作候选点产生器，并单独实现或验证窗口滚动、因果性、阈值语义和持续性过滤。

## 可以复用的部分

- **Rbeast Python 接入形态**：`import Rbeast as rb` 后用 `rb.beast(series, season='none', ...)`，可作为平滑趋势型的一维时序候选变点分析接口。
- **候选点输出的读取方式**：代码示范了读取 `trend.ncp_mode`、`trend.cp` 和 `trend.cpPr`，再把候选点位置映射回原始时间戳。
- **候选点后的独立验证**：将变点概率阈值、相邻点间隔、两侧均值变化拆开处理是可复用的结构；但阈值应按 FLASH 原文或复现配置明确设定，不能复制 Greyhound 的默认 `10` 点和 `10%`。
- **离群点预处理思路**：Greyhound 与 FLASH 文本都提到 z-score `3.0`。实现时应明确采用 `abs(z) > 3.0`，并保持时间戳和序列索引同步过滤。
- **可复现性要求**：固定 Rbeast 精确版本，并显式记录模型先验、MCMC 和 outlier 选项。当前 `>=0.1.19` 的范围和省略默认参数不足以锁定相同检测结果。

## 不能直接当作 FLASH 行为的部分

- 不要把 ACF 周期推断和 NCCL `call_number` 周期时延序列当成 FLASH 的迭代时间输入。FLASH 论文通过框架回调上报迭代时间。
- 不要把 Rbeast 的 `cpPr` 当成 Adams–MacKay run-length 后验。论文里的 `P(r_t=0 | x_{1:t})` 需要由被实际实现的算法定义。
- 不要把 Greyhound 的 `length_thresh=10`、`degradation_thresh=0.1` 作为 FLASH 阈值。FLASH 文本分别描述相邻变点 `<2` 迭代去重、`degradation_thresh=0.2` 和 recovery ratio `0.75`。
- 不要根据 `ncp_mode` 直接假设存在同样数量的可信候选位置；Rbeast 文档说明 `cp` 列出可能的 changepoint，可能包含不显著项，而 `ncp_mode` 是变点数量分布的众数。代码把 `cp[:ncp_mode]` 与整个 `cpPr` 列表按序配对；若改用该方式，应对安装版本输出的字段形状和索引语义作验证。

## 实现层面的稳健性注意

- 当前 `try/except` 只包住 Rbeast 调用循环，异常被静默吞掉，随后代码仍访问 `result`；若调用抛错，`result` 可能未定义。若返回对象没有 `trend`，`while True` 会持续调用，无法退出。
- `find_period()` 在 ACF 找不到峰时返回 `(-1, None)`；调用方会终止本轮检测。异常/无峰回退策略应与目标 FLASH 数据流单独定义。
- 数据量不足时 `stats.zscore()`、前后片段均值以及 `ts[-2:]` 可能不稳定；论文和该函数未规定最小样本数。
- 每轮全局窗口的滚动方式、输出对象如何与 rank/原始迭代号对齐，需结合调用方代码验证；本说明不据此新增 FLASH 控制逻辑。

## 源码与资料

- [slow_detection.py](../third_party/Greyhound/detector/control_plane/slow_detection.py)
- [local_analyzer.py](../third_party/Greyhound/detector/control_plane/local_analyzer.py)
- [setup.py](../third_party/Greyhound/detector/setup.py)
- [Rbeast 官方项目文档](https://github.com/zhaokg/Rbeast)
- [Adams & MacKay, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)
