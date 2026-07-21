# 独立长期 LFP 验证预注册草案

状态：`draft`，不可据此宣称验证成功。完整门禁由 JSON Schema 和公开的跨字段
semantic validator 共同构成。机器可读模板见
[`../configs/validation/independent_long_term_lfp_protocol.template.json`](../configs/validation/independent_long_term_lfp_protocol.template.json)，约束定义见
[`../configs/validation/independent_long_term_lfp_protocol.schema.json`](../configs/validation/independent_long_term_lfp_protocol.schema.json)，跨分区不重叠和计数一致性由
[`../src/lifetwin/validation/long_term_protocol.py`](../src/lifetwin/validation/long_term_protocol.py)复算，不能只填写一个“已验证”布尔值。

本文定义与具体数据集无关的验证协议。只有在数据许可、字段结构和可评分性确认后，才复制模板并冻结数据集专用实例。当前不建立 Lam/Joule 专用解析器，也不下载其许可未澄清的数据。

## 1. 研究问题

目标不是证明“短期数据可以准确预测 15 至 25 年”，而是在一个具有实际长期观测的独立 LFP/石墨日历老化队列上检验：在严格限制为 landmark 当时可见信息时，锁定的 LifeTwin 候选模型能否比最强的简单外推基线更准确，并在数据不足或域外输入时拒绝给出不受支持的区间。

主要统计单位是**物理电芯轨迹**，独立推断单位是预先定义的工况、批次或站点 cluster。重复检查点不是独立样本。

## 2. 数据资格

确认性轨迹评估必须同时满足以下硬门槛：

1. 数据或书面授权明确允许约定的非商业研究、竞赛评估、特征提取、聚合指标和图表发布；论文许可不能替代数据许可。
2. 正极为 LFP，负极为石墨；化学体系不明、混合体系或含硅负极不得并入主确认队列。
3. 有稳定的物理电芯 ID，且训练、校准、测试和审计分区不得跨分区复用同一物理电芯。
4. 日历老化可以与循环老化分离，并有机器可读的时间、工况和容量或容量保持率观测。
5. 最长可评分时长不少于 730 天；至少 8 个物理电芯和 8 个独立评分 cluster。
6. 每条进入动态 landmark 评分的轨迹至少有 4 个正时间前缀观测和 2 个未来观测，且未来跨度与 landmark 时间之比至少为 2.0。

十年起点/终点数据可进入 `D2_long_horizon_endpoint`，但不能验证动态 landmark。现场混合老化、系统级或短期加速数据最多作为 `D1_auxiliary_stress_or_field`。资格失败必须记录机器可读原因，不能为了得到正结果放宽门槛。

## 3. 结局盲法

每个数据集必须记录项目层面的结局接触史，并只能归入以下一类：

| 分类 | 定义 | 允许的最高表述 |
|---|---|---|
| `prospective_outcome_blind` | 预测冻结后才产生目标结局 | 前瞻性运行证据，仍受样本量与域边界限制 |
| `public_but_project_blind` | 结局已公开，但项目成员在冻结预测前未接触 | 带“公开数据但项目盲法”限定的锁定外部确认 |
| `locked_retrospective_replication` | 结局或其结构已被项目接触，但模型、阈值和评分器随后完整锁定 | 回顾性外部复现 |
| `development_only` | 数据参与了模型、landmark、门限或协议开发 | 假设生成和开发证据 |
| `unclassifiable` | 接触记录不完整 | 不作确认性声明 |

“预测时不看未来”是必要的防泄漏设计，不单独作为算法创新。任何接触论文图、汇总终点、公开 notebook 输出或数据结构的行为都要写入 exposure log。

冻结或执行状态下，证据等级与声明角色必须一一对应：`D5` 只对应
`prospective_outcome_blind` 和 `confirmatory`；`D4` 只对应
`public_but_project_blind` 和带公开数据限定的锁定外部确认；`D3` 只对应
`locked_retrospective_replication` 和回顾性复现。`development_only` 只能用于假设生成，锁定回顾性复现也不得改写为 outcome-blind confirmation。

## 4. 冻结顺序

必须按顺序执行，不得倒置：

1. 只查看许可、版本、字节数、文件名和字段说明等元数据，先判定权利与潜在资格。
2. 记录适用许可或作者书面授权；按原始字节保存每个文件的 SHA-256、字节数和版本。
3. 只基于字段结构和时间/索引可用性冻结适配器、单位换算、异常规则、分区、candidate landmarks 与共同未来评分窗口；不得读取目标容量值来选择这些设置。
4. 确认资格结果为 `eligible`，再冻结候选模型、三项强制基线、全部超参数、环境锁、评分器、成功门槛和本协议；记录 Git commit、模型/基线配置 SHA-256 及各输入文件 SHA-256。
5. 用前缀生成不含真实未来结局的预测 bundle，记录 SHA-256 和 UTC 时间戳。
6. 只有预测 bundle 已冻结后才链接未来真值，由独立评分器生成不可覆盖的结果 bundle。

`frozen` 状态不接受占位版本、空哈希、未取得的数据、未冻结的超参数、未选择的主 landmark 或超出观测支持的声明跨度。它还必须记录实际电芯、独立 cluster、前缀和未来支持计数；四个分区都非空且 ID 两两不相交。`executed` 还必须具有预测与结果 bundle 哈希、预测冻结与真值链接时间、真值确在预测冻结后链接的记录及独立评分器验证。任何内容变更都创建新协议 ID 和 amendment；若变更受目标结局启发，原运行终止为 `void`，新运行只能降级为回顾性或开发性证据。

## 5. Landmark 与评分窗口

landmark 可按“正时间观测数”或“经过天数”定义。候选值只由预先冻结的工程需求和时间支持决定。模板给出的 `[5, 8, 10, 14]` 是占位候选，数据集专用协议必须在看见目标值前冻结主 landmark。

所有模型在同一物理电芯、同一 landmark、同一未来时间戳上评分。主指标是未来轨迹积分绝对误差：先按实际经过时间做梯形积分，再除以评分跨度，单位为容量保持率百分点。条件先在 cluster 内等权，再跨 cluster 等权，避免检查点更密或重复更多的工况支配结论。

“最早可用 landmark”只有在该 landmark 及其所有更晚候选 landmark 都通过主要成功门槛和无回退门槛时才能确认；比较必须限制在冻结的共同未来支持区间。

## 6. 比较模型

候选模型必须与以下三项强制基线同时比较：

- `target_prefix_persistence`：保持 landmark 时的最后观测值。
- `target_prefix_sqrt_time`：只用目标电芯前缀拟合平方根时间衰减。
- `target_prefix_bounded_power_law`：只用目标电芯前缀拟合有界幂律衰减。

“最强基线”只能在独立 calibration partition 上按主指标预选，不能在 test 或 audit 结果出来后切换。候选模型和基线都不得使用 landmark 之后的目标结局拟合、选参或门控。

## 7. 成功、失败与证据不足

点预测只有同时满足以下门槛才记为 `success`：

1. 相对最强基线的 cluster 等权平均主指标至少改善 0.1 个百分点。
2. 相对改善比例至少 5%。
3. 以独立 cluster 为单位的单侧随机化检验 `p <= 0.05`。
4. 至少 60% 的独立 cluster 改善。
5. 最差 cluster 的回退不超过 0.5 个百分点。
6. 至少有 8 个独立评分 cluster，且所有更晚候选 landmark 同时通过规定门槛。

样本或 cluster 不足、随机化检验分辨率不足、评分窗口不满足等情况记为 `inconclusive`，不能记为成功。任一实质性能门槛失败记为 `failure`，保留完整负结果且不晋升候选模型。资格或哈希校验失败则该运行不得评分。

机器可读协议只有在资格为 `eligible`、实际观测计数达到冻结阈值、证据等级与盲法/声明角色匹配、候选和三项指定基线超参数已冻结、分区非空且经复算无重叠、点预测及 landmark 一致性门槛通过、执行哈希和时间齐全且独立评分器验证通过时，才允许 `executed + success`。把 `development_only` 或回顾性运行伪装成 outcome-blind confirmation 会导致 Schema 或 semantic validator 失败。

实际冻结或执行前运行统一生产入口：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\verify_independent_long_term_protocol.py path\to\dataset-specific-protocol.json
```

## 8. 区间与拒绝

诊断区间和可运行区间必须分开标记。可运行区间采用按预测路线分组的 **independent-cluster conformal calibration**。每个预先声明的工况、制造批次或站点 cluster 只产生一个校准分数：取该 cluster 内所有物理电芯轨迹及冻结未来窗口上的最大标准化误差；同一 cluster 内增加电芯或重复检查点不会增加 conformal 的 `n`。每条路线至少需要 20 个独立校准 cluster 和 20 个独立审计 cluster，并同时满足：

- 校准与待预测路线、时间跨度和协变量支持相符；
- 校准集独立于训练、测试和审计集；
- 点预测先通过上述成功门槛；
- 审计覆盖率相对目标的绝对短缺不超过 0.10。

任一门槛不满足时只允许输出诊断区间或明确拒绝，不能声称形式化覆盖率。预注册拒绝原因包括输入质量失败、前缀过短、温度/SOC/电芯类型超出支持、专用路线未就绪、同路线独立 cluster 校准不足、跨度不匹配、分位数非有限、区间过宽、缺少独立验证或哈希校验失败。

## 9. 声明边界

允许声明的最长时间只能等于冻结数据中真实观测且通过评分的最长未来跨度。即使结果成功，也不得据此声称：

- 已验证 15 至 25 年预测精度；
- 已验证海辰产品、电站或系统级性能；
- 单凭电芯数据完成了大型储能电站验证；
- 在没有独立同路线校准时具有形式化覆盖保证；
- 从工况均值推导了单体电芯不确定性；
- 论文 CC BY 自动赋予数据或代码商业使用权。

这些边界与模型得分同等重要，并由模板中的 `claim_boundaries` 和执行哈希共同固化。`fifteen_to_twenty_five_year_claim_allowed` 在 schema 中恒为 `false`；任何 frozen/executed 实例还必须保留 `15_to_25_year_accuracy_without_observed_support` 禁止项。
