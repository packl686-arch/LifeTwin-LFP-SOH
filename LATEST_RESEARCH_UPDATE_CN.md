# LifeTwin 最新研究进展（2026-08）

作者：Jincheng Liu

## 一句话结论

项目已经形成一条可审计的技术路线：只向预测器提供目标电芯的早期数据，利用训练电芯学习“相似轨迹迁移 + 风险门控”，并在证据不足时回退到训练集验证过的安全模型。最新公开 LFP 大样本实验中，安全硬门控模型将 cycle-300 轨迹 MAE 从持续值基线的 0.725 个百分点降至 0.286 个百分点，降幅 60.5%。

这是一项公开数据上的回顾性研发结果，不代表海辰产品精度，也不能直接证明 15-25 年寿命预测能力。

## 本轮解决了什么问题

1. **未来信息泄漏**：训练、预测、评分分成独立阶段；预测文件先生成并计算 SHA-256，评分器随后才能读取目标未来轨迹。评分时还会重放冻结算法，拒绝被修改的预测或清单。
2. **单一外推模型失稳**：同时保留持续值、线性、鲁棒近期趋势、物理形状约束和相似电芯增量迁移等专家，不依赖一种经验曲线覆盖所有工况。
3. **低置信度回退不安全**：V1 证明“所有模型等权平均”会把灾难性的线性外推重新引入。V2 改为严格训练集留一验证形成安全池，不合格专家权重固定为 0。
4. **模型动态修正**：根据早期容量、内阻、温度、充电时长和能效轨迹寻找相似训练电芯，再用局部历史误差决定是否切换专家。
5. **不确定性表达**：区间仅由训练电芯严格留一残差生成；项目明确将其称为诊断区间，而不是无条件的统计覆盖保证。

## 最新实验结果

公开 MATR FastCharge 数据包含 41 个训练电芯、41 个主测试电芯和 40 个次测试电芯。目标测试电芯只提供 P20、P40、P60 或 P100 的前缀，统一预测至 cycle 300。

| 方法 | 主测试集 MAE | 次测试集 MAE | 总体 MAE |
|---|---:|---:|---:|
| 安全硬门控 | **0.341** | 0.231 | **0.286** |
| 相似电芯增量迁移 | 0.361 | 0.231 | 0.297 |
| 安全先验连续混合 | 0.416 | **0.201** | 0.310 |
| 持续值基线 | 0.939 | 0.506 | 0.725 |

单位均为容量保持率百分点。冻结的 V2 主模型通过全部预设开发门槛；但实测最优的是安全硬门控，而不是连续混合模型。这个负责任的区分很重要：项目保留失败和非最优结果，不用事后改写主指标。

## 可复核材料

- [FastCharge V2 完整技术报告](reports/fastcharge_lfp_safe_prior_v2_2026-08-04.md)
- [FastCharge V1 失败分析](reports/fastcharge_lfp_trajectory_portability_v1_2026-08-04.md)
- [NASA V3 小样本方法开发报告](reports/nasa_evidence_weighted_moe_v3_development_2026-08-03.md)
- [FastCharge V2 冻结配置](configs/experiments/fastcharge_lfp_safe_prior_v2.json)
- [FastCharge V2 实现](src/lifetwin/experiments/fastcharge_safe_prior_v2.py)
- [FastCharge V2 测试](tests/test_fastcharge_safe_prior_v2.py)
- [数据来源与许可边界](docs/references.md)

原始数据、作者身份映射表以及生成的 Parquet/评分产物不会上传到公开仓库。公开报告记录配置、输入、预测和评分表的哈希，以便在合法取得数据后复核。

## 当前证据边界

已经验证的是：软件层面的未来标签隔离、公开循环老化数据上的轨迹迁移可行性、安全专家筛选机制，以及失败时的可追溯回退。

尚未验证的是：长期日历老化、储能电站真实工况、跨厂商和跨规格迁移、海辰内部产品表现，以及 15-25 年预测精度。下一阶段不应继续在已经看过结果的数据上调参，而应冻结安全硬门控方案，在新的长期 LFP 队列或海辰内部数据上做真正的独立验证。

## 2026-08-06 数据治理更新

本次更新只同步数据身份和发布完整性，没有新增模型精度。MATR 身份层已经改为在
`summary` 前停止；NASA ordinary-battery 解压快照包含 38 个 MAT、10 个 README/TXT，
按文件名和重复哈希规则得到 34 个唯一 `Bxxxx` 身份与 4 组同哈希重复表示。这里的
34 只表示当前快照中核验的身份，不是 34 个独立、同分布或合格盲测电芯。

NASA V3 的四个第三方 CSV 与上述 38 个 MAT 元数据接入是两个不同证据对象。新接入只读
文件元数据、128 字节头部和 `whosmat` 顶层 schema；MAT/容量值读取、训练、预测、评分
和 SNL 内容读取数均为 0。README 已暴露停止阈值、异常和部分结局结构，因此只能作为
`development_only_outcomes_and_protocol_structure_exposed` 的开发级治理证据。

数据集专属许可和公开聚合结果发布权仍未解决，正式 NASA 执行门保持关闭；NASA 化学
体系也未获权威确认，不能称为 LFP 验证集。V1.2 已恢复冻结的 `beep.py` 哈希，并通过
公开发布校验，但这同样不增加任何模型效果或独立验证证据。

- [数据治理原报告与日期化增补](docs/data_asset_intake_20260806.md)
- [MATR V1.1 与发布边界纠正](docs/data_asset_intake_20260806_v1_1_correction.md)
- [NASA 四 CSV 与解压 MAT 的来源边界](docs/nasa_pcoe_battery_data_provenance.md)
- [机器可读数据资产登记](docs/data_asset_register_20260806.csv)

## 2026-08-07 跨平台复现恢复

冻结提交 `b872c33` 的 public-release CI 已在 GitHub 托管的 Ubuntu 和 Windows 上完成
全量复现，quality、两端 reproduction 与 Pages build/deploy/report 均通过；两端均为
`914 passed, 0 skipped`。本次留痕保留了此前 GitHub 官方事故期间的失败和取消尝试，
并在事故 resolved、Actions/Pages operational 后复核了最终状态。

这只更新工程发布与跨平台可复现性事实，不增加模型精度、独立验证、NASA/BEEP、真实
电芯/电站或 15–25 年结论。运行、job、artifact 哈希和证据边界见
[跨平台 CI 恢复关闭报告](reports/cross_platform_ci_recovery_closeout_20260807.md)。

## 独立验证候选已冻结（2026-08-04）

项目已把 FastCharge V2 中表现最稳健的安全硬门控结构提名为下一份未接触长期 LFP 数据的主候选，并固定安全池门槛、局部风险算法、邻居数/风险余量训练网格、回退规则和区间发行门槛。候选语义 SHA-256 为 `596108e19ca0a8c7fb712bf82ca5be93817524f5f0c912f3b71b180a0fcba3af`。这是开发结果之后的候选提名，不是独立确认。

同时新增 metadata-only 数据 intake 编译器：它在读取目标容量值前检查数据许可、原始文件版本与哈希、物理电芯 ID、日历老化可分离性、时长、cluster 数、前后缀支持和项目结局接触史。通过 intake 也只会进入第二人复核，不会自动冻结或提高证据等级；失败草案自动降为 `unclassifiable + D0`。

- [独立验证执行手册](docs/independent_validation_execution_2026_08_cn.md)
- [冻结候选配置](configs/validation/independent_safe_hard_candidate_v1.json)
- [数据 intake 模板](configs/validation/independent_lfp_dataset_intake.template.json)
- [intake 编译器](scripts/compile_independent_lfp_intake.py)
- [对抗测试](tests/test_independent_lfp_intake.py)
