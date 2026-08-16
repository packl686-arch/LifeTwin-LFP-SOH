# LifeTwin V0.18 / V2.3 数值契约稳定性预注册

设计日期：2026-08-10

协议 ID：`synthetic_long_horizon_identifiability_v2_3`

唯一正式 attempt：`v023-formal-20260810-a1`

当前状态：`preregistered_preimplementation`

机器可读修订：[`synthetic_long_horizon_identifiability_v2_3_amendment.json`](../configs/experiments/synthetic_long_horizon_identifiability_v2_3_amendment.json)

## 1. 历史边界与触发原因

V0.14 的正式状态保持 `failure`；V0.15/V2 保持 `inconclusive_not_success`。
V2.1 的唯一正式尝试 `v021-formal-20260808-a1` 永久保持
`terminal_pre_prediction / unclassified_terminal_not_success`，原因码为
`UNKNOWN_PRE_PREDICTION_EXCEPTION`。该尝试没有 prediction commitment，score 根为空；
不得重跑、续跑、修补、重新分类或创建 a2。

V2.1 的已确认根因是实现契约作用域不匹配：全量 label-free 表已经通过 formal reader，
但 runner 随后先切出 `center_development`，再把 `prefix_pack.csv` 全量 71,400 行规则应用到
合法的 7,200 行 center 子集。V2.2 已用 whole-bundle capability 修复该问题；其唯一正式尝试
`v022-formal-20260809-a1` 随后在预测前自然终止为
`terminal_pre_prediction / void / proven_integrity`，原因码为
`INTEGRITY_PARTITION_CONTRACT_MISMATCH`。该尝试已有 fit commitment，但没有 prediction
commitment、没有评分、`opened_truth_files=[]`，不得重跑、续跑、修补或创建 a2。

V2.2 的结果前数值根因已由不消费 seed 的合成 fixture 闭合：继承的 label-free pipeline
按既有 schema 对不可用 raw risk 以及未签发的 calibrated probability 有意写入结构性 `NaN`，
但 V2.2 新增的 blanket `isfinite` partition 门要求所有数值单元有限，因而必然拒绝合法的
`risk_bundle.csv`。V2.3 只修正这一数值契约实现，不改变任何科学门槛或结果规则。

以下前序锚点保持不可变：

| 对象 | SHA-256 |
|---|---|
| V2.1 amendment | `3c1348be3dd6e0f86df84283b7a27e57f4e5747c1356497f77bb39c695f06a4e` |
| V2.1 prereg | `ae346fe8aa5699a7b3ad0d124e8ec29851a16fcbf485309e8c3c441749540efc` |
| V2.1 implementation audit | `88470000f4b913ee36514e51aba646bda6d2dd6342b414e84190f7c2b9f39f29` |
| V2.1 freeze record | `c42414209ac365246fd3e52e142b6c1c9a64ff7bcc8a69d3f29348b1cc5369bd` |
| V2.1 public closeout | `a614b97df6d6d2891912d67410f1384d10141952fc07bd46551e7035eb185951` |
| V2.1 root-cause audit | `99d4dea6438b98e7e6c3c8ed213f499740f4c6176f4c36c4050472cab177a9cd` |

V2.3 只允许使用已披露的 V2.1/V2.2 结构性根因、冻结规则和历史身份。V2.1/V2.2 的生成行、
真值、opaque ID、pair ID、内容哈希、噪声、拟合/校准状态、prediction、score、临时文件和
缓存一律不得复用。不得读取 V2.2 sealed truth 或正式数值输出来设计本协议。

## 2. 唯一修订

V2.3 保留 V2.2 的 whole-bundle → capability-bound partition 路线，并增加 schema-aware
数值契约：

1. 先对五张完整 label-free 表执行原有 whole-bundle formal schema、行数、key、单位、
   有限值、成员关系与承诺哈希验证；
2. 全部门通过后，由私有 issuer 签发精确类型 `WholeBundleValidated`；
3. 只有该 capability 可以切出冻结分区，并再次校验分区精确 cardinality、key、单位、
   86 variants、8 horizons、全量到分区 key 证明与源文件哈希绑定；
4. 通过后签发不可伪造的 `ValidatedPartitionView`；数值 pipeline 只接受这一类型；
5. 对 pipeline 输出按 schema 与跨表状态校验“必须有限”与“必须为结构性 NaN”的精确位置；
6. 所需 label-free partition capability 与 label-free 输出数值契约全部通过之后，runner 才可
   打开相应开发真值。

冻结的数值语义如下：

- `risk_bundle.raw_risk_score` 当且仅当对应 `feature_bundle.all_features_finite=true`
  时必须有限，否则必须为 `NaN`；
- `risk_bundle.calibrated_catastrophic_probability` 当且仅当对应 cluster 是 hard-eligible，
  且 `score_id` 为 `prefix_only` 或 `visible_stress` 时必须为 `[0,1]` 内有限值，否则必须为
  `NaN`；
- risk 的三个计数列始终必须有限；所有正负 infinity 均禁止；
- decision/prediction 的每个 nullable 数值列必须由冻结的 fit/availability/issuance/rejection
  状态确定精确 mask，不得仅因“该列可空”而接受任意 NaN；
- 禁止填零、填哨兵值、clip、静默 coercion、只拒绝 infinity，或从 heldout truth 推导 mask。

禁止 `formal=False`、降低 `required_rows`、从观察行数反推期望值、跳过 Schema/哈希、
公开 capability 构造器、在 label-free 验证前打开真值、用填充值掩盖结构性 NaN，或通过
异常文本猜测分类。

## 3. 科学规则保持不变

除新协议/attempt 身份、全新 seed 与数据身份、上述精确数值契约和对应完整性异常外，
V2/V2.1/V2.2 全部规则继续逐字约束：

- 六个核心生成族、方程、分布和 row-generation logic；
- 七个分区及 cluster 数；12 个 prefix day、8 个 forecast day 与全部单位；
- 精确 86 个模型 variants、模型、特征、优化器和 fit credibility 门；
- risk/isotonic、mean baseline、900/811 split-conformal 规则；
- 五个主端点、八族与 novel safety gates；
- test/audit 发行数、250 对 matched audit 分母；
- bootstrap、Clopper–Pearson、placebo、stress permutation 和信息防火墙；
- 成功、scored failure、terminal inconclusive、integrity void 与报告义务。

任何未在本修订中明确改变的规则仍然有效，不得由实现解释放宽。

## 4. 全新 seed、身份和一次性 attempt

V2.3 固定 13 个新根：

| stream | root |
|---|---:|
| center_development | 202608100301 |
| risk_development | 202608100302 |
| calibration | 202608100303 |
| test | 202608100304 |
| audit | 202608100305 |
| novel_mechanism_test | 202608100306 |
| novel_mechanism_audit | 202608100307 |
| intrinsic_matched_pairs | 202608100308 |
| stress_plan_matched_pairs | 202608100309 |
| random_rankings | 202608100310 |
| bootstrap | 202608100311 |
| stress_permutations | 202608100312 |
| placebo_covariate | 202608100313 |

每个流使用首 16 位十六进制
`SHA256(protocol_id|seed_root|partition|family_id|zero_based_index|stream_name)`，转无符号整数后
模 `2^63-1`，并仅在冻结环境中使用 `numpy.random.Generator(PCG64DXSM)`。

新 protocol ID 与新 roots 必须产生全新行、ID、哈希、状态、预测和评分。正式 attempt
固定为 `v023-formal-20260810-a1`，只允许一次；路径、attempt 或历史碰撞均在生成前停止，
禁止自动建立 a2 或为结果更好重试。

## 5. 冻结 cardinality

五张全量表固定为：

| 表 | rows |
|---|---:|
| `prefix_pack.csv` | 71,400 |
| `forecast_coordinates.csv` | 47,600 |
| `operating_pack.csv` | 5,950 |
| `member_fit_diagnostics.csv` | 511,700 |
| `member_forecast_bundle.csv` | 4,093,600 |

各分区按 `clusters / prefix / coordinates / operating / diagnostics / forecasts` 固定为：

| partition | counts |
|---|---|
| center_development | `600 / 7,200 / 4,800 / 600 / 51,600 / 412,800` |
| risk_development | `600 / 7,200 / 4,800 / 600 / 51,600 / 412,800` |
| calibration | `900 / 10,800 / 7,200 / 900 / 77,400 / 619,200` |
| test | `1,900 / 22,800 / 15,200 / 1,900 / 163,400 / 1,307,200` |
| audit | `950 / 11,400 / 7,600 / 950 / 81,700 / 653,600` |
| intrinsic_matched_pairs | `500 / 6,000 / 4,000 / 500 / 43,000 / 344,000` |
| stress_plan_matched_pairs | `500 / 6,000 / 4,000 / 500 / 43,000 / 344,000` |

每个分区的 key 集必须两两不交；七个分区的 union 必须精确重建五张全量表的 key 集。

## 6. 结果前测试门

冻结前必须用手写/合成 fixture 完成以下测试；测试不得调用正式 generator、消费 V2.3
seed 或读取 sealed truth：

- runner 真实调用 whole-bundle formal 验证与 partition capability 路径；
- 71,400 行全量 prefix 和 7,200 行 center partition 同时通过；
- 五张表、七分区的 schema/key/cardinality/单位/有限值/conditional-null/variant/horizon 门；
- risk、decision、prediction 全部 nullable 数值列的精确结构性 NaN mask 与跨表对齐；
- 少一行、多一行、错列/列序、重复 key、错误分区、错协议/config/attempt/hash全部拒绝；
- forged capability、错误精确类型、签发后 dataframe mutation 全部拒绝；
- truth spy 证明 capability 完成前真值调用数为 0；
- prediction commitment 先于 heldout scoring；scored 与 terminal 注册表互斥；
- infinity、错误位置 NaN、结构性位置有限值、mask 漂移和填零绕过全部拒绝；
- V2/V2.1/V2.2/V2.3 seed、opaque ID、内容哈希、attempt 与路径碰撞门；
- 已知 contract 异常使用注册的 V2.3 完整性类型，不得落入 `unknown_default`。

## 7. 正式生命周期与停止规则

环境、源码、历史、碰撞、路径与测试门全部通过并形成 implementation audit、environment
lock、source-tree/semantic/byte SHA 和 freeze record 后，才允许调用正式 generator 一次。

runner 必须自行执行：fresh generation → whole-bundle validation → fit/calibration →
prediction commitment → heldout truth opening → scoring。不得手工拼接中间文件或边算边看。

- scored success：五主端点和全部冻结 gates 均通过；
- scored failure：有效评分但至少一个成功门失败，仍原样发布全部分母和结果；
- terminal inconclusive_not_success：冻结的科学可用性/拟合条件在预测前触发；
- integrity void：配置、源码、环境、历史、路径、seed/ID/content、capability、哈希、真值顺序
  或 commitment 任一违规；
- unknown：仅真正未注册异常可用 `UNKNOWN_PRE_PREDICTION_EXCEPTION`，已知 partition
  contract 错误不得使用该默认分类。

若 runner 写出 terminal registry，必须立即停止且不得伪造空评分表。若产生 prediction
commitment，只有在承诺字节完成后才能打开 heldout truth 并评分。平台中断不得创建新 attempt。

## 8. 隔离与历史保护

V2.3 使用独立 branch、worktree、四个正式根和 operator-evidence 根。四个正式根在启动前
必须不存在、解析后两两不重叠，并与 V2.1/V2.2 worktree 和根完全不相交。V2.3 进程无权
写入 V2.1/V2.2 路径；执行前后必须复核旧 Git refs、worktree 身份、正式根元数据和关键证据
SHA。

禁止 reset、rebase、amend、force push、删除/移动旧 branch、tag、Release、worktree、attempt、
artifact 或日志。任何历史漂移都使 V2.3 成为 integrity void。

## 9. 宣称边界

无论 V2.3 产生 success、failure、inconclusive 或 void，都只属于冻结六核心族合成 25 年
协议。它不能证明真实 LFP 电芯、海辰产品、储能电站或 15–25 年真实预测精度，也不增加
独立真实数据验证证据。
