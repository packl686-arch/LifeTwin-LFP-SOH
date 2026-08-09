# LifeTwin 最新研究进展（2026-08）

作者：Jincheng Liu

## 一句话结论

项目已经形成一条可审计的技术路线：只向预测器提供目标电芯的早期数据，利用训练电芯学习“参考条件化残差 + 支持门控”，并在证据不足时回退到稳定模型。最新 V5 公开 LFP 开发实验中，成对参考残差模型将 cycle-300 轨迹 MAE 从 V2 稳定硬门控的 0.286 个百分点降至 0.208 个百分点，降幅 27.3%；支持门控后的 90% 共形区间覆盖率为 93.10%，平均宽度相对 V2 缩窄 50.4%。

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
| V5 参考条件化残差 | **0.222** | **0.194** | **0.208** |
| V5 支持门控中心 | 0.222 | 0.221 | 0.221 |
| 安全硬门控 | **0.341** | 0.231 | **0.286** |
| 相似电芯增量迁移 | 0.361 | 0.231 | 0.297 |
| 安全先验连续混合 | 0.416 | **0.201** | 0.310 |
| 持续值基线 | 0.939 | 0.506 | 0.725 |

单位均为容量保持率百分点。V5 模型只用 41 个训练电芯做物理电芯五折选择；81 个评估电芯的未来后缀未参与选择。V5 无门控中心的点精度最好，而支持门控中心略有退化；后者的价值在于更明确的回退和更紧的区间，不被包装成点精度提升。项目保留失败和非最优结果，不用事后改写主指标。

## 可复核材料

- [FastCharge V2 完整技术报告](reports/fastcharge_lfp_safe_prior_v2_2026-08-04.md)
- [FastCharge V5 完整技术报告](reports/fastcharge_v5_pairwise_development_2026-08-09.md)
- [FastCharge V5 小型公开证据包](showcase/evidence_v5/README.md)
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

## 企业私有盲测候选修订（2026-08-06）

面向海辰内部循环数据的接口已拆成开发、批次独立校准和一次性锁定测试，并密封预测、拒绝决策、运行配置、未来计划与完成清单。当前主模型仍是冻结V3双时钟模型。

完整V4曾尝试按未来温度、SOC窗口、放电倍率和分段EFC/天直接修正长期衰减系数。结果暴露后的开发诊断不支持该机制，因此它只保留为负对照，不晋升。新候选V4.1仅使用预测时已声明的未来日期与EFC坐标；其他计划字段只判断是否超出训练支持域，不直接推动容量曲线。V4.1仍不是海辰验证结果，必须通过不变的批次独立校准门槛后，才有资格进入一次锁定测试。

- [V4.1机器可读修订协议](configs/experiments/private_enterprise_schedule_v4_1_amendment.json)
- [V4.1修订记录](reports/private_schedule_v4_1_amendment_2026-08-06.md)
- [海辰私有盲测执行手册](docs/hithium_private_blind_execution_cn.md)

V4.2作为最后一个有限候选已经在接触海辰数据前冻结：工况修正权重最高为25%，且修正幅度不超过训练集内部LOCO诊断半区间的25%；离训练支持域越远，权重越接近零。它不是默认模式，也没有真实数据性能结论。合成全流程演练中，V4.1和V4.2都未满足预设改善门槛，系统按规则回退V3，并在不打开锁定真值的情况下密封了锁定预测。这一结果证明失败路径能正常工作，而不是证明V3具有企业精度。

- [V4.2预注册协议](configs/experiments/private_enterprise_schedule_v4_2_preregistered.json)
- [V4.2技术记录](reports/private_schedule_v4_2_preregistration_2026-08-06.md)
- [数据集证据矩阵](configs/validation/dataset_evidence_matrix_2026_08.json)
- [第二轮正确性审计](reports/private_enterprise_correctness_reaudit_2026-08-06.md)

## 文献驱动的 V5 研发与首轮结果（2026-08-09）

在系统阅读早期寿命、完整轨迹、参考电芯迁移、Gaussian Process、physics-informed、mixture-of-experts、knee、跨域迁移和不确定性论文后，下一条高价值路线被收敛为 V5-RCGP：保留冻结 V3/安全硬门控作为中心，只让“成对参考残差”和“GP 在线残差”在训练内部证据充分时修正它；没有 partial-charge、relaxation、ICA/DVA 或阻抗等诊断证据时，不启用带电化学名称的机理门控。

V5 不自动加入已经冻结的海辰锁定测试。首轮 H1 成对参考实验已经执行：41 个训练电芯内部选择 `12 个参考 + ExtraTrees 成对残差 + 加权均值`，训练内相对固定近邻改善 24.8%；在 81 个公开评估电芯上相对 V2 改善 27.3%，物理电芯 bootstrap 区间未跨零。支持门控和共形区间的子门槛也已执行，覆盖率为 93.10%、平均宽度 1.4155 pp、单区间 WIS 0.1284。

动态 landmark 与在线 residual audit 已按冻结协议执行。较长前缀重新运行 V5 在 81 个公开评估电芯上把 transition-equal MAE 从 0.2644 pp 降至 0.1738 pp，物理电芯聚类 bootstrap 95% 区间为 [-0.1195, -0.0659] pp；但 P40→P60 的改善很弱，三个 transition 合计只改善 64.2% 的 cell-transition。三种固定 Matern GP 均未达到 70% 电芯改善门槛；训练内唯一合格的 P40 轻量 offset 在公开评估只改善 66.7% 电芯，因此也不激活。完整 H2 明确为未通过，而不再是“尚未评估”；当前 V5 中心与 hybrid conformal 区间保持不变。Transformer/CNN 仍因训练电芯规模不足而不进入。

- [论文证据矩阵与完整 V5 实验设计](docs/literature_model_review_and_v5_experiment_plan_2026_08_cn.md)
- [V5 机器可读开发协议](configs/experiments/v5_rcgp_literature_informed_development.json)
- [V5 协议防回退测试](tests/test_v5_rcgp_development_plan.py)
- [V5 实验报告](reports/fastcharge_v5_pairwise_development_2026-08-09.md)
- [V5 动态 landmark 与在线残差审计](reports/fastcharge_v5_dynamic_landmark_audit_2026-08-09.md)
- [V5 公开证据](showcase/evidence_v5/README.md)
