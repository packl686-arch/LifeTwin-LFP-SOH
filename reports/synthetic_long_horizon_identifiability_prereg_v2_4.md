# LifeTwin V0.19 / V2.4 member-fit 契约与原子承诺预注册

设计日期：2026-08-10

协议 ID：`synthetic_long_horizon_identifiability_v2_4`

唯一预留正式 attempt：`v024-formal-20260810-a1`

当前状态：`preregistered_post_root_cause_pre_formalization`

机器可读修订：[`synthetic_long_horizon_identifiability_v2_4_amendment.json`](../configs/experiments/synthetic_long_horizon_identifiability_v2_4_amendment.json)

## 1. 时间与信息边界

本预注册形成于 V2.3 唯一正式 attempt 终止之后，也形成于 V0.19 根因候选代码
`e7af8150a92dcc339093bafdaaebe98f912543de` 之后，因此不能声称 preimplementation。
允许使用的信息仅包括冻结源码、schema、前序终态身份、公开触发点，以及不调用
generator/RNG/真值能力的全新手写确定性 fixture。V2.3 正式
`member_fit_diagnostics.csv` 的内容、数值、分布和哈希均未被读取、抽样、搜索或分析。

历史状态永久保持：V0.14=`failure`；V0.15/V2=`inconclusive_not_success`；V2.1 为
`terminal_pre_prediction / unclassified_terminal_not_success`；V2.2 与 V2.3 均为
`terminal_pre_prediction / void / proven_integrity`。V2.3 原因码为
`INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH`，没有 prediction commitment、没有评分、
`opened_truth_files=[]`。`fit_commitment.json` 只是 attempted fit 阶段的 partial artifact，
正式 manifest 中 registered fit commitment 为 `null`。所有前序尝试不得重跑、续跑、
改写、删除、重新分类或创建 a2。

## 2. 已确认根因与唯一变更面

冻结拟合器保留每个 cluster 的全部 86 variants。`fit_status="failed"` 时，参数为 `{}`、
`credible_variant=false`，三项诊断指标与八个 raw forecast 是结构性 `NaN`；成功状态的
相同位置必须有限。V0.18 whole-bundle blanket `isfinite` 把合法 failed mask 错判为
完整性异常。

V2.4 唯一允许的科学/协议差异是：

1. member-fit 两表的 exact fit-status mask；
2. member-fit、whole/partition capability、environment、terminal 的类型化异常；
3. fit commitment 的原子写入与 exposure phase 顺序；
4. 新 protocol/attempt、全新 seed roots、数据身份和隔离路径。

禁止填 0、填哨兵、删 failed 行、裁列、允许任意 NaN、`formal=False`、降低行数、跳过
schema/key/hash/read-back，或把 member-fit contract 错误解释为模型效果失败。

## 3. member-fit 数值契约

每个 `(partition, cluster_id)` 必须精确包含冻结 86 个 `(model_id, variant_id)`；每个
diagnostic key 必须对应冻结八个 forecast days。

| 列/状态 | succeeded | failed | 禁止 |
|---|---|---|---|
| `parameters_json` | canonical、数值有限的参数对象 | 精确 `{}` | 非 canonical、非数值或非有限参数 |
| `credible_variant` | 由冻结 RMSE≤1.0 pp、最大残差≤1.5 pp、八点 [40,105]% 规则重算 | 必须 false | 非严格 bool 或声明/重算漂移 |
| `prefix_rmse_pp` | 有限且 ≥0 | `NaN` | ±∞、numeric string、mask 反转 |
| `prefix_max_abs_residual_pp` | 有限且 ≥0 | `NaN` | 同上 |
| `parameter_boundary_hit_fraction` | 有限且在 [0,1] | `NaN` | 同上或整列退化 |
| `raw_forecast_retention_pct` | 八点全部有限 | 八点全部 `NaN` | 任一 horizon 缺失/多余、hash/key/status 漂移 |

V2.3 已冻结的 risk structural-NaN、prediction/decision nullable mask、rank/issuance 与
跨表状态规则继续有效，不因本修订放宽。

## 4. 科学规则逐项继承

以下项目逐项、无修改继承 V2.3；遗漏的前序规则仍然绑定：

- 六个核心生成族、所有方程、分布、噪声与 row-generation logic；
- 七个分区及 cluster 数；12 个 prefix days、8 个 forecast days 与单位；
- 精确 86 variants、全部模型、特征、优化器、参数边界与 fit credibility；
- risk/isotonic、mean baseline、900/811 split-conformal 与路由/回退规则；
- 五个主端点及其原阈值：coverage、conditional coverage、interval width、25 年点预测
  RMSE、catastrophic overprediction；
- 八族 gate、novel-mechanism/safety gate、test/audit 发行数及 250 对 matched audit；
- placebo、stress permutation、information firewall 负控制；
- bootstrap、Clopper–Pearson、permutation 与全部置信区间/显著性规则；
- success、scored failure、terminal inconclusive_not_success、integrity void、unknown
  terminal 的定义、分母和报告义务。

不得因实现便利修改阈值、分区、端点、gates、优化器或成功条件。若完整正式化证明必须
修改任一科学字段，V2.4 立即停止并新开版本，不回改本预注册。

## 5. 新 seed、身份和隔离

只将以下 13 个根写入协议；本阶段不得调用 RNG、generator 或派生函数：

| stream | root |
|---|---:|
| center_development | 202608100401 |
| risk_development | 202608100402 |
| calibration | 202608100403 |
| test | 202608100404 |
| audit | 202608100405 |
| novel_mechanism_test | 202608100406 |
| novel_mechanism_audit | 202608100407 |
| intrinsic_matched_pairs | 202608100408 |
| stress_plan_matched_pairs | 202608100409 |
| random_rankings | 202608100410 |
| bootstrap | 202608100411 |
| stress_permutations | 202608100412 |
| placebo_covariate | 202608100413 |

未来若另获执行授权，只能使用 `v024-formal-20260810-a1` 一次；不得创建 a2。四个正式
根必须在启动前不存在、两两不重叠，并与全部前序 worktree/attempt 根不相交。V2.4 不得
复用 V2/V2.1/V2.2/V2.3 的 seed、行、ID、哈希、拟合/校准状态、预测或评分。

## 6. 冻结 cardinality

全量五表固定为 71,400 prefix、47,600 coordinates、5,950 operating、511,700
diagnostics、4,093,600 forecasts。七分区仍为：

- center_development：`600 / 7,200 / 4,800 / 600 / 51,600 / 412,800`
- risk_development：`600 / 7,200 / 4,800 / 600 / 51,600 / 412,800`
- calibration：`900 / 10,800 / 7,200 / 900 / 77,400 / 619,200`
- test：`1,900 / 22,800 / 15,200 / 1,900 / 163,400 / 1,307,200`
- audit：`950 / 11,400 / 7,600 / 950 / 81,700 / 653,600`
- intrinsic_matched_pairs：`500 / 6,000 / 4,000 / 500 / 43,000 / 344,000`
- stress_plan_matched_pairs：`500 / 6,000 / 4,000 / 500 / 43,000 / 344,000`

## 7. fit commitment 原子性

顺序固定为：内存 member-fit 门 → canonical serialization → write-once 实际落盘 →
fresh read-back → whole-bundle schema/cardinality/key/hash/numeric contract → 唯一
`fit_commitment.json` → 验证 commitment → 更晚追加 completed fit exposure event。

非法结构 NaN、±∞、落盘后 hash/read-back 失败或 phase append 失败时，partial 文件必须
保留作证据，但不得存在正式 fit commitment 或 completed fit phase。若失败发生在
commitment 已形成之后、completed phase append 之前，terminal manifest 必须明确记录
该 commitment 已注册而 phase 未完成；不得删除 commitment 或倒写 completed event。
terminal manifest 必须分别列出 partial artifacts 与 registered commitments，且 scored/
terminal registry 继续互斥。

## 8. 结果前验证门

冻结前必须完成：逐列合法/非法 mask 与变异矩阵；真实文件 canonical write→fresh read→
whole validate→七分区 derive/consume 的 5,950-cluster 精确基数合成门；fit commitment
故障注入；V2.2/V2.3 risk/partition/terminal/preresult 和完整继承回归；ruff、compile/
import、CLI help/preflight、环境/线程池/包锁、source-tree、collision、路径与唯一 attempt
静态门。测试不得消费上述 seed、调用 generator/truth capability 或创建四个正式根。

## 9. 停止与宣称边界

本阶段只允许形成 P、I、F 三个 forward-only commit。F 完成后仍不得执行正式 attempt，
必须先交验。任何规则漂移、真实数值不稳定、历史/ref 漂移、资源不足或必须修改本预注册
字段时立即停止。

未来无论 V2.4 结果为何，都只能称为冻结六核心族合成 25 年协议结果；不能证明真实 LFP
电芯、海辰产品、储能电站或 15–25 年真实预测精度。
