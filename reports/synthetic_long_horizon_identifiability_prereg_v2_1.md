# LifeTwin V2.1 校准人群拆分修订预注册

设计日期：2026-07-26

设计见证人：Jincheng Liu

协议 ID：`synthetic_long_horizon_identifiability_v2_1`

当前状态：`design_candidate_preimplementation`

机器可读修订：[`synthetic_long_horizon_identifiability_v2_1_amendment.json`](../configs/experiments/synthetic_long_horizon_identifiability_v2_1_amendment.json)

## 1. 这不是 V2 的续跑或补救

V0.15/V2 正式尝试在校准阶段终止。900 个 calibration cluster 中，899 个具有至少两个成功可信结构族，1 个只有一个成功可信结构族。该尝试尚未写出 `model_state.json`，没有 prediction commitment，也没有评分包。

已打开的开发真值只有：

- `center_development_truth.csv`
- `risk_development_truth.csv`
- `calibration_truth.csv`

以下 heldout 真值与配对映射均未打开：

- `test_truth.csv`
- `audit_truth.csv`
- `intrinsic_matched_truth.csv`
- `stress_plan_matched_truth.csv`
- `intrinsic_matched_pairs.csv`
- `stress_plan_matched_pairs.csv`

V2 仍固定记为 `inconclusive_not_success`。V2.1 是在终止原因完全披露后提出的新设计，不能改写、恢复或替代 V2。

本修订绑定三个前序证据哈希：

| 对象 | SHA-256 |
|---|---|
| V2 config 原始字节 | `27dc7f89178f73779a52068c1878df26c9686faa7433686e60ba6496b6705796` |
| V2 prereg 原始字节 | `c1dee9f9b4ef134b1a52e9a51300c591e790c10a0e97b3fe6c15eb441b2c09f0` |
| V2 终止尝试 manifest | `5b2b2d300653d070ed107b67a1a11b4edc10a0d33b2bad491ef1be784e0f4b09` |

## 2. 允许和禁止继承的信息

V2.1 只允许使用：

1. `899/900` 具有至少两个可信结构族、`1/900` 只有一个可信结构族这一聚合触发事实；
2. 上述三个前序哈希、终止阶段、状态及 opened/unopened 文件清单；
3. V2 已冻结且与结果无关的公式、分布、阈值、schema 和科学动机。

V2 的任何生成行、前缀、工况、真值、噪声、映射、opaque ID、内容哈希、拟合状态、标签、预测、决策、排名或评分均不得进入 V2.1。即使 V2 heldout 从未打开，也禁止复用其行。V2.1 必须重新生成全部数据。

## 3. 修订的科学理由

V2 把两个不同统计任务错误地绑定成同一个 900/900 多结构资格要求：

- 风险与 isotonic 校准的目标人群，是最终能够进入排序和发行的硬资格池；
- split-conformal 的目标，是六核心族总体上固定分母的同时预测集覆盖率，包括最终 abstain 的 cluster。

因此，V2.1 冻结两个不同但都由结果无关规则确定的校准分母。它不放宽可信拟合门、不把单结构 cluster 提升为可发行对象，也不删除 conformal 中的不利行。

## 4. 风险与 isotonic 校准池

在打开 calibration truth 之前，必须从 label-free 工件计算并按字节承诺唯一 mask：

`risk_isotonic_eligible_v2_1`

mask 为真当且仅当同时满足：

1. 冻结日期上恰有 12 个有序且有限的 prefix 观测；
2. 恰有 8 个有序且有限的 forecast 坐标；
3. 8 个真实工况和 8 个 placebo 字段全部有限；
4. 七个声明结构族中至少两个各有一个 variant 通过 V2 不变的可信度规则；
5. 8 个冻结中心预测全部有限；
6. 14 个 Arm-A 特征、8 个真实 stress 特征和 8 个 placebo 特征全部有限；
7. 两个主 logistic 头的 raw score 均有限。

两个主臂必须使用同一个 mask。禁止按臂删行、补位或扩大资格池。mask 不得读取灾难标签、未来 SOH、真值族、真值参数或任何结果。

冻结的可用性要求为：

| 项目 | 要求 |
|---|---:|
| calibration 来源总数 | 恰为 900 |
| risk/isotonic mask 为真 | 至少 `ceil(0.95*900)=855` |
| mask 内灾难正例 | 至少 60 |
| mask 内负例 | 至少 60 |

`855` 来自事先声明的 95% 可用性原则，而不是把阈值贴在已暴露的 899 上。isotonic 只在 mask 为真的行上分别拟合两个主头；排序仍使用 raw logistic score，校准概率不能改变排序。

单结构或零结构 cluster 始终 risk-ineligible，绝不进入发行。若保留其 raw score 作为确定性审计量，也不得把它解释为可操作的校准概率。

任一来源计数、855 门槛、60/60 类别门槛、有限性或 isotonic 拟合要求未满足，尝试在预测前终止为 `inconclusive_not_success`。

## 5. 均值基线与 conformal 保持全 900

### 5.1 均值基线

三个原冻结均值基线继续在全部 900 个 calibration cluster 上计算 mean trajectory IAE，并以原词典序规则破除精确平局。禁止根据 risk eligibility 删除行。任何必需轨迹缺失或非有限均令预测前尝试无结论。

### 5.2 Split-conformal

conformal 分母固定为全部 900：

- 必须得到 900 个有限的 simultaneous nonconformity score；
- coverage 仍为 0.90；
- 排序后使用一基索引 `ceil((900+1)*0.90)=811`；
- 不插值、不删行、不补位、不截尾。

结构族成功数的冻结处理如下：

| 成功可信结构族数 | 中心 | risk/isotonic | conformal 基础带 |
|---:|---|---|---|
| 0 | sqrt-time fallback | 不合格 | 不存在；整次尝试预测前无结论 |
| 1 | sqrt-time fallback | 不合格 | 若真实单族支持有限且有序，则纳入全 900 |
| 2–7 | 原 V2 blended center | 其余 mask 条件通过时合格 | 原 V2 family-balanced band |

单族基础带只能来自该成功族中真实可信 variants，并继续使用原来的 float64 精确去重、完整八维 signature 聚合、族内权重和坐标 0.05/0.95 分位规则。不得把 sqrt-time 伪装成第二个结构族或结构支持。

零族没有结构基础带。禁止使用 sqrt 带、`[50,105]`、裁剪带或插补 score 代替；任一零族 calibration cluster 都令 conformal 不可定义并终止为 `inconclusive_not_success`。

同理，任一基础带在八个时点非有限或上下界逆序、任一 score 非有限、或数量不为 900，均不得删行，直接无结论。

conformal 声明边界不变：它只是在冻结六核心族混合分布上的边际同时覆盖，不保证逐族或逐支持数条件覆盖，也不是物理可辨识区间。

## 6. 必须报告的双分母

未来 calibration manifest 必须同时记录：

- `source_calibration_count`
- `risk_isotonic_eligible_count`
- 零族、单族和其他原因造成的 risk-ineligible 数
- mask 内正例与负例数
- `mean_baseline_count`
- `conformal_calibration_count`
- `conformal_order_statistic_index`
- label-free mask 的字节哈希

不能只报告 899 或只报告 900，从而掩盖两个 estimand 的人群差异。

## 7. 两套互斥工件注册表

一个 attempt 只能选择一套注册表，文件名集合严格不相交。

### 7.1 `scored`

只有 prediction commitment 已存在、heldout scoring capability 在其后合法打开且评分完成，才允许生成：

1. `point_scores.csv`
2. `trajectory_scores.csv`
3. `family_metrics.csv`
4. `matched_pair_scores.csv`
5. `bootstrap_replicates.csv`
6. `random_ranking_metrics.csv`
7. `stress_permutation_metrics.csv`
8. `negative_control_metrics.json`
9. `score_report.json`
10. `run_manifest.json`

### 7.2 `terminal_pre_prediction`

预测承诺前终止时，只生成：

1. `terminal_attempt_record.json`
2. `terminal_artifact_manifest.json`
3. `terminal_exposure_log_snapshot.jsonl`

该注册表不得包含 heldout truth、endpoint estimate、gate result、空评分表或成功/失败性能结论。terminal manifest 只能哈希终止时真实存在的工件，不能声称缺失的 model、prediction 或 score 文件存在。

## 8. 类型化终止原因

声明的科学无结论原因包括：

- `CALIBRATION_SOURCE_COUNT_NOT_900`
- `CALIBRATION_RISK_ELIGIBLE_BELOW_855`
- `CALIBRATION_RISK_POSITIVE_BELOW_60`
- `CALIBRATION_RISK_NEGATIVE_BELOW_60`
- `CALIBRATION_RISK_SCORE_NONFINITE`
- `CALIBRATION_ISOTONIC_FIT_UNDEFINED`
- `CALIBRATION_BASELINE_INCOMPLETE`
- `CALIBRATION_ZERO_FAMILY_NO_BAND`
- `CALIBRATION_BAND_NONFINITE_OR_UNORDERED`
- `CALIBRATION_CONFORMAL_COUNT_NOT_900`
- `CALIBRATION_CONFORMAL_SCORE_NONFINITE`
- `CALIBRATION_CONFORMAL_FIT_UNDEFINED`

配置、源码、环境或工件 hash 不一致，seed/ID/content collision，前序行复用，非法真值访问，信息泄漏或缺少承诺均属于 `INTEGRITY_*`，直接 `void`。

操作员或平台中断保持 `interrupted`，只能按同 commit、同 config、同 seed 和同承诺字节恢复。未知异常记为 `UNKNOWN_PRE_PREDICTION_EXCEPTION` 和 `unclassified_terminal_not_success`，禁止继续预测或伪造评分；若不可由不可变证据排除完整性问题，保守发布为 `void`。free text 不能替代 reason code。

## 9. 全新 seed 与禁止生成

V2.1 冻结 13 个新 root：

| stream | root |
|---|---:|
| center_development | 202607260201 |
| risk_development | 202607260202 |
| calibration | 202607260203 |
| test | 202607260204 |
| audit | 202607260205 |
| novel_mechanism_test | 202607260206 |
| novel_mechanism_audit | 202607260207 |
| intrinsic_matched_pairs | 202607260208 |
| stress_plan_matched_pairs | 202607260209 |
| random_rankings | 202607260210 |
| bootstrap | 202607260211 |
| stress_permutations | 202607260212 |
| placebo_covariate | 202607260213 |

这些 root 彼此不同，也不同于 V2 的 `202607230101` 至 `202607230113`。派生公式保持不变，但协议 ID 改为 `synthetic_long_horizon_identifiability_v2_1`。正式生成前必须在冻结实现中完成所有派生 seed、opaque ID 和 content hash 的碰撞审计。

当前没有 V2.1 实现，也没有运行 collision audit。`design_candidate_preimplementation` 状态下禁止调用任何 V2.1 generator、消费 seed、在生成数据上拟合、分析 pilot outcome 或打开 heldout truth。

## 10. 所有性能规则保持不变

除本修订明确列出的校准人群和预测前终止记录外，V2 的以下内容全部不变：

- 六核心族、两新机制、audit shift 和两类 250 对 matched audit；
- 中心公式、可信 fit 门、14+8 特征、两个主头及全部比较器；
- test 共同池至少 1,805，audit 至少 903；
- test 每臂发行 950，audit 每臂发行 475；
- 五个主终点的阈值、bootstrap 和 Clopper–Pearson 规则；
- 八族与 novel safety gates；
- placebo、stress permutation 和信息防火墙 gates；
- 全部报告表、负结果政策和声明边界。

V2.1 当前没有任何成功声明。它没有证明可行、改进、有效、失败或成功。只有在实现、环境、源码、schema、碰撞审计及 adversarial 0/1/2-family 测试另行提交并不可变冻结后，才允许进入生成阶段。
