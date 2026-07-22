# 合成长时域结构可辨识性实验 V1 正式结果

日期：2026-07-22

协议：`synthetic_long_horizon_identifiability_v1`
预注册结论：**failure**

本报告解释一次按冻结协议完成的合成长时域机制压力测试。它不是现实 LFP 电芯、海辰产品、储能电站或 15 至 25 年预测精度验证。机器可读结论以 [`score_report.json`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/score_report.json) 为准；冻结设计见 [`配置`](../configs/experiments/synthetic_long_horizon_identifiability_v1.json) 与 [`预注册说明`](synthetic_long_horizon_identifiability_prereg_v1.md)。

## 1. 预注册主终点

V1 规定三个主门槛必须同时通过。正式运行通过一个、未通过两个，因此总状态为 `failure`，不是 `inconclusive` 或 `void`。

| 主门槛 | 冻结通过条件 | 正式结果 | 判定 |
|---|---|---|---|
| 50% 发行率下的灾难风险降低 | 相对 10,000 个随机排名的平均灾难率至少降低 30%，且分层 bootstrap 单侧 95% 下界大于 0 | 已发行 162/500，灾难率 32.40%；随机均值 41.353%；相对降低 21.65%；5,000 次 bootstrap 单侧 95% 下界 16.31% | **未通过** |
| 匹配前缀反例双侧拒绝 | 至少 80% 的 200 对反例中，两侧都严格超过 calibration 阈值 | 54/200，即 27.0%；另有 7 对因分歧非有限而不计拒绝成功 | **未通过** |
| 已发行轨迹 IAE 非劣 | 候选均值不高于 calibration 选出的最强基线 `+0.10 pp` | 候选 2.15635 pp；平方根时间基线 2.14179 pp；差值 `+0.01456 pp` | **通过** |

随机排名分布、bootstrap 逐次结果和同覆盖率比较分别见 [`random_rejection.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/random_rejection.csv)、[`bootstrap.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/bootstrap.csv) 与 [`rejection_policy_metrics.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/rejection_policy_metrics.csv)。21.65% 的降低有稳定统计信号，但没有达到预先声明的 30% 最小效应，因此不能改称成功。

## 2. 必需安全门槛

四项安全门槛全部通过：

| 安全门槛 | 正式证据 | 判定 |
|---|---|---|
| 最小样本数与有限预测 | test 共 1,000 个 cluster，其中 781 个通过硬资格；test 点预测有限率 100%；200 对匹配反例全部合格 | 通过 |
| 随机排名完整 | 10,000/10,000 个随机排名均有定义 | 通过 |
| bootstrap 完整 | 5,000/5,000 个分层重采样均有定义 | 通过 |
| audit 方向一致 | audit 已发行灾难率 33.60%，随机期望 44.67%，相对降低 24.78% | 通过 |

这些门槛证明结果可评价且方向可复核，但不能替代未通过的两个主门槛。完整风险覆盖曲线见 [`risk_coverage.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/risk_coverage.csv)。

## 3. 协议与结果完整性

- 正式执行使用 Git commit `8244bd412e6dcd890ed182b338356ad6fa3b8f69`；配置字节 SHA-256 为 `503ec964bb2015fe3460433749d1b0d79f89187fc3dcd1c3809f9d4da2ffc319`，规范化 SHA-256 为 `6ad1e6dc1caa089ce0b9ee2c4e739a56c44f42f65436294649261a7676d4e320`。
- 预测进程未收到真值路径。预测与决策包先写入并完成字节承诺，之后评分器才打开真值；[`exposure_log.json`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/exposure_log.json) 记录了完整顺序。
- 真值包、预测包、决策包、成员诊断和全部输入承诺均由严格评分器复核；正式结果的 `protocol_deviations` 为空。
- 共处理 2,900 个 cluster、每个 85 个声明变体，即 246,500 个变体拟合；数值拟合失败为 0。匹配审计中的 14 个“模型失败成员”来自 7 对少于两个可信结构族而产生的非有限分歧，不是优化器失败。
- 可提交证据的逐文件哈希、仓库内保留范围与完整压缩包清单见 [`evidence_manifest.json`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/evidence_manifest.json) 和 [`full_bundle_manifest.json`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/full_bundle_manifest.json)。
- 完整发布资产 `lifetwin-v0.14.0-synthetic-long-horizon-full.zip` 为 14,468,311 字节，SHA-256 为 `66efc005c7207f7d2718edb77ad6f1e93029ee377ede6a2cece0ec32087e57bd`；可由 [`build_v014_release_asset.py`](../scripts/build_v014_release_asset.py) 确定性重建。

### 作废尝试披露

在成功运行前，有一次同 commit、同冻结协议的前台运行因控制通道中断而在预测阶段以退出码 120 终止，本地归档名为 `synthetic_long_horizon_identifiability_v1.void.20260722T031707Z`。该尝试没有生成预测承诺或评分结果，真值在作废前未打开，也未纳入证据。随后精确重跑的真值承诺以及前缀、坐标、匹配对和真值包字节均与作废尝试一致。披露副本和比对状态见 [`preoutcome_void_attempt`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/preoutcome_void_attempt/exposure_log.json) 与证据清单。不存在查看结果后换 seed 或筛选运行的情况。

## 4. 结果拆解

### 4.1 哪些机制有效，哪里发生反转

在 test 与独立 audit 分区，结构分歧策略呈现相同的族别模式：

| 真值族 | test 相对随机风险降低 | audit 相对随机风险降低 | 解释 |
|---|---:|---:|---|
| `single_power` | 59.24% | 55.69% | 稳定正向 |
| `dual_power` | 22.07% | 30.99% | 中等正向 |
| `saturating_plus_slow` | 43.53% | 41.81% | 稳定正向 |
| `early_activation_plus_power` | 47.44% | 45.83% | 稳定正向，但硬资格覆盖较低 |
| `late_knee` | **-5.45%** | **-2.17%** | 两个分区均方向反转 |

族别正式统计见 [`family_metrics.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/family_metrics.csv)；逐 cluster 记录见 [`trajectory_scores.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/trajectory_scores.csv)。总体分歧识别灾难错误的 AUROC 为 0.629；它有风险排序能力，但被 `late_knee` 的不可辨识性显著削弱。

硬资格本身也提供了有效预警：test 中 781/1,000 个 cluster 合格；不合格组灾难率为 79.45%，合格组为 41.36%。但覆盖率具有明显机制差异：`early_activation_plus_power` 仅 43.0%，`saturating_plus_slow` 为 58.5%，其余三族为 95.5% 至 97.0%。因此不能只报告已发行样本而忽略拒绝构成。

### 4.2 匹配前缀的可辨识性含义

200 对反例的两侧在 0 至 730 天无噪声前缀上完全相同，前缀 RMSE 与最大差异均为 0，并共享同一噪声实现；所有成对预测和分歧也完全相同。与此同时，两侧 25 年真值分离中位数为 14.84 pp，范围为 8.85 至 20.81 pp。原始配对与评分见 [`matched_prefix_pairs.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/matched_prefix_pairs.csv) 和 [`matched_pair_scores.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/matched_pair_scores.csv)。

calibration 冻结分歧阈值为 35.8018 pp，而匹配对有限分歧的中位数为 33.3906 pp，所以仅 27.0% 双侧拒绝。进一步检查显示，最大包络宽度在所有合格 cluster 上都出现在 25 年点；极端晚期 knee 场景贡献约 `0.004 * (9131.25 - 1095.75) = 32.142 pp` 的共同先验宽度。因而原始最大包络分数具有接近 32.142 pp 的底座，更像“普遍存在的晚期情景宽度”，而不是某个 cluster 的 knee 证据。

这项失败给出的是清晰边界：当两个未来共享完全相同的可见前缀时，任何确定性前缀模型都必须给出同一预测与决策，无法识别哪一侧会发生 knee。正确方向是报告部分可辨识区间、最坏情景或保守拒绝，而不是声称从同一前缀辨别未来分支。

### 4.3 均值预测与可信结构

calibration 选择的平方根时间基线 IAE 为 2.5265 pp，并在 test 上稳定为 2.5493 pp。候选在全部 test 上的 IAE 为 4.3104 pp，但通过选择性发行后降至 2.15635 pp，并满足非劣门槛。候选的主要均值误差来自 `early_activation_plus_power`：其 25 年平均绝对误差为 21.28 pp，而平方根时间基线为 2.71 pp。逐时点结果见 [`forecast_day_metrics.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/forecast_day_metrics.csv)。

未来观测噪声不是主要原因：全 test 带噪相对无噪的平均轨迹 IAE 仅增加 0.017 pp，灾难率变化为 -0.3 个百分点，见 [`noise_sensitivity_metrics.csv`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/noise_sensitivity_metrics.csv)。

## 5. 限制与允许结论

本实验只允许得出两点：结构分歧在四类前缀可辨识机制上具有可复现的风险筛选信号；对前缀完全相同、未来独立分叉的晚期 knee，它不能提供个体辨识，V1 门控尚未准备好覆盖全部声明的合成危险。

本结果不支持现实 LFP 25 年精度、海辰产品或电站验证、个体电芯保证、正式运行覆盖率、独立外部数据确认，也不证明五个合成真值族覆盖完整电池物理。50% 发行率是批量评价操作点，不是可部署阈值。

## 6. POST-HOC：v0.15 假设生成，不属于 V1 确证证据

以下分析在 V1 结果揭盲后完成，未预注册、未写入冻结预测承诺，也不属于 [`score_report.json`](../showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/score_report.json) 的正式结论。它只能用于设计新协议，不能补救或改写 V1 的 `failure`。

使用 396 个 development 硬资格 cluster 拟合了一个带 L2 正则、`C=1`、无类别加权的标准化逻辑回归风险头。输入均为预测时可见的八个摘要：可信结构族数、最佳前缀 RMSE、原始分歧、候选 25 年预测、候选与平方根基线之差、365 至 730 天下降、0 至 90 天变化和一个前缀曲率代理。保持 V1 的硬资格池与发行数量不变，探索性结果为：

| 已暴露分区 | 灾难风险相对降低 | AUROC |
|---|---:|---:|
| calibration | 32.32% | 0.712 |
| test | 31.33% | 0.691 |
| audit | 34.63% | 0.712 |

按 calibration 第 250 个低风险发行点冻结探索性阈值后，该风险头仍只拒绝 57/200 个匹配对，即 28.5%，没有解决完全相同前缀的根本歧义。现有 test 与 audit 已经暴露，因此上述数字不能作为独立验证或新版本成功证据；任何 v0.15 实现都必须先冻结新协议、特征、seed 和门槛，再使用全新未见分区验证。

由此形成的最高价值 v0.15 假设是：

1. 将均值预测头与安全拒绝头解耦，以独立 development/calibration 学到的收缩或堆叠中心预测替代按名义公式数投票。
2. 用透明、可校准的风险头组合预测与收缩基线之差、分歧、前缀形状、边界命中和有效独立预测形状，而不是只使用全局最大包络。
3. 增加显式早期激活结构，并按预测形状去重，避免多个退化到同一幂律外推的公式获得重复票数。
4. 将匹配前缀实验改为部分可辨识评价：同时报告成对真值覆盖、全轨迹覆盖和区间宽度；另设带温度、SOC、DOD、EFC、静置时间或批次等可见协变量的增量辨识实验。
5. 采用新 seed、留一机制族和未见真值族进行一次性复核；保留 V1、随机拒绝、前缀拟合误差和平方根时间基线作为冻结消融。
