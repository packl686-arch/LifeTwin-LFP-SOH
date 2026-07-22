# v0.14 合成长时域可辨识性证据

## 一句话结论

预注册协议 `synthetic_long_horizon_identifiability_v1` 的正式状态是 **failure**。结构分歧在普通合成混合中显示稳定风险筛选信号，但没有达到预先规定的 30% 最小效应，也没有保守拒绝足够多的完全相同前缀反例。

## 三个主门槛

| 门槛 | 正式结果 | 判定 |
|---|---:|---|
| 50% 发行下相对随机灾难风险降低至少 30%，bootstrap 下界大于 0 | 21.65%；单侧 95% 下界 16.31% | 未通过 |
| 200 对匹配前缀反例至少 80% 双侧拒绝 | 54/200，27.0% | 未通过 |
| 已发行轨迹 IAE 不劣于最强基线超过 0.10 pp | `+0.01456 pp` | 通过 |

三个门槛要求全部通过，因此不能把局部正向结果表述为 V1 成功。

## 安全与完整性

- 四项安全门槛全部通过：最小样本与有限预测、10,000 个随机排名完整、5,000 次 bootstrap 完整、audit 方向一致。
- audit 风险降低 24.78%，与 test 的 21.65% 同向。
- 预测承诺先于真值开启；正式运行 `protocol_deviations=[]`。
- 2,900 个 cluster、246,500 个声明变体拟合，数值拟合失败为 0。
- `late_knee` 是唯一在 test 与 audit 都发生族别风险反转的机制，分别为 -5.45% 和 -2.17%。

成功运行前的一次同 commit 运行因前台控制通道中断而在预测阶段作废。该尝试没有预测承诺、没有评分结果、真值未打开，也未纳入证据；精确重跑的真值承诺和生成包字节与其一致。详情见 [`evidence_manifest.json`](synthetic_long_horizon_identifiability_v1/evidence_manifest.json) 中的 `preoutcome_void_attempt`。

## 匹配前缀意味着什么

200 对反例的可见前缀完全相同，但 25 年真值分离中位数为 14.84 pp。成对预测与决策也完全相同，这既证明了身份防火墙与确定性不变性，也说明只看相同前缀不可能判断哪一侧会出现未来 knee。V1 的失败不应通过换 seed 或放宽门槛掩盖；后续应转向部分可辨识区间、最坏情景和额外可见协变量。

## 证据导航

- [`冻结配置`](../../configs/experiments/synthetic_long_horizon_identifiability_v1.json)
- [`冻结预注册`](../../reports/synthetic_long_horizon_identifiability_prereg_v1.md)
- [`正式中文结果与解释`](../../reports/synthetic_long_horizon_identifiability_result_v1_2026-07-22.md)
- [`机器可读评分`](synthetic_long_horizon_identifiability_v1/score_report.json)
- [`正式暴露日志`](synthetic_long_horizon_identifiability_v1/exposure_log.json)
- [`证据与作废尝试清单`](synthetic_long_horizon_identifiability_v1/evidence_manifest.json)
- [`族别结果`](synthetic_long_horizon_identifiability_v1/family_metrics.csv)
- [`匹配反例评分`](synthetic_long_horizon_identifiability_v1/matched_pair_scores.csv)
- [`风险覆盖曲线`](synthetic_long_horizon_identifiability_v1/risk_coverage.csv)
- [`随机排名分布`](synthetic_long_horizon_identifiability_v1/random_rejection.csv)
- [`bootstrap 分布`](synthetic_long_horizon_identifiability_v1/bootstrap.csv)

大文件未重复提交到 Git；其名称、字节数和 SHA-256 记录在 [`full_bundle_manifest.json`](synthetic_long_horizon_identifiability_v1/full_bundle_manifest.json)。确定性构建脚本为 [`build_v014_release_asset.py`](../../scripts/build_v014_release_asset.py)；完整压缩包对应发布资产 `lifetwin-v0.14.0-synthetic-long-horizon-full.zip`，大小 14,468,311 字节，SHA-256 为 `66efc005c7207f7d2718edb77ad6f1e93029ee377ede6a2cece0ec32087e57bd`。

## POST-HOC v0.15 假设

结果揭盲后的探索性逻辑风险头仅使用 development 标签和八个预测时可见摘要，在已暴露的 calibration/test/audit 上得到 32.32%/31.33%/34.63% 的风险降低，AUROC 为 0.712/0.691/0.712；它仍只拒绝 57/200 个匹配对。该结果**不是 V1 确证证据，也不是 v0.15 验证**，只能用于冻结下一版假设。下一版必须使用新协议、新 seed 与未见机制复核，并将“可排序风险”与“完全相同前缀的部分可辨识性”分开评价。

## 声明边界

本目录只证明冻结合成机制压力测试的执行和结果。它不验证现实 LFP、海辰产品、个体电芯、储能电站、正式运行区间或 15 至 25 年预测精度。
