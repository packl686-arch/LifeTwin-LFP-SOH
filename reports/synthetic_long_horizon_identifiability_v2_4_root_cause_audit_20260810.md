# LifeTwin V2.4 / V0.19 member-fit 数值契约根因审计（2026-08-10）

状态：`development_preimplementation`。本报告不是预注册、实现冻结或正式运行授权；
没有分配或消费正式 seed，没有创建正式 attempt、四根、runner CLI 或 freeze record。

## 结论

V2.3 已披露的直接触发事实是 `member_fit_diagnostics.csv` 在 whole-bundle
有限值门被拒绝。本次在禁止读取、抽样、统计或哈希分析该正式 CSV 内容的边界内，使用
冻结源码、冻结 schema 和全新确定性 fixture 将上游机制闭合为：

**A. 合法结构性缺失与 blanket finite 门冲突。**

冻结拟合器对每个 cluster 保留全部 86 个声明 variant。成功分支先验证八点预测有限，并
把诊断指标与公式重算逐项绑定；失败分支则明确写入八个 `NaN` 预测、三个 `NaN` 诊断
指标、空参数对象、`fit_status="failed"` 和非可信状态。V0.18 whole-bundle 校验随后对
所有数值 dtype 无条件执行 `isfinite`，因此会把注册过的失败状态误判为完整性错误。

这一结论不依赖 V2.3 正式数值。冻结成功分支不能正常产出非有限诊断指标：非有限预测
被直接拒绝，非有限指标也无法通过公式重算的精确相等检查。冻结失败分支是该生产路径
中唯一主动写入 member-fit 非有限值的分支。正式 CSV 已先通过冻结 canonical CSV
读取，V2.3 又以 `proven_integrity` 终止；因此 C（序列化/类型错误）与 D（状态掩码
矛盾）不是可达的已知触发机制。新测试另行证明 B（真正的 ±∞ 或成功位置 NaN）会被
V0.19 拒绝，而不会被结构性例外掩盖。

## 冻结源码证据

- `src/lifetwin/experiments/calendar_long_horizon_v015_fit.py:825-852`：成功 variant
  的八点预测必须有限，且 RMSE、最大残差、边界比例与公式重算逐项一致。
- `src/lifetwin/experiments/calendar_long_horizon_v015_fit.py:854-871`：失败 variant
  明确保留行，并把三个诊断指标和八点 raw forecast 写成结构性 `NaN`。
- `src/lifetwin/experiments/calendar_long_horizon_v018_partition.py:190-195`：旧门对
  dataframe 的全部数值 dtype 执行 blanket `np.isfinite(...).all()`。
- `configs/experiments/synthetic_long_horizon_identifiability_v2.json:921-947`：诊断表
  的冻结列和四列 variant key；失败 variant 不得删行。
- 同一冻结 schema 的 `member_forecast_bundle.csv` 规则要求每个诊断 variant 保留八个
  horizon，失败 variant 的 raw value 为空。因此 V0.19 必须联合校验两张表，不能只对
  首个报错文件做单列豁免。

V0.18 三个关键文件保持字节不变：

| 文件 | SHA-256 |
|---|---|
| `calendar_long_horizon_v018_numeric_contract.py` | `e91e0682899f5e41827c071e5304356758eef89087f6fff4d952d92f3106e783` |
| `calendar_long_horizon_v018_partition.py` | `7213b4877b06bbe5b71ef38306df8e1cc60f678382f04f16c6b564d6a57fb3be` |
| `calendar_long_horizon_v018_runner.py` | `d0c63b43872a923a8dd803c5124e1e0b99f5a7e614ec7f54b0e75a9084afa1b1` |

## `member_fit_diagnostics.csv` 数值语义表

冻结表没有名为 `objective`、`rank` 或 `issuance` 的列。拟合目标的已提交结果由
`prefix_rmse_pp` 与 `prefix_max_abs_residual_pp` 表达；rank/issuance 位于 decision
输出，继续由 V0.18 decision contract 约束，没有在本次接线中放宽。

| 数值/状态列 | 决定 mask 的状态 | 成功位置 | 结构性缺失位置 | 额外不变量 |
|---|---|---|---|---|
| `credible_variant` | `fit_status`、两项误差指标、八点 raw forecast | 严格 bool；只在 RMSE ≤ 1.0 pp、最大残差 ≤ 1.5 pp 且八点均在 [40,105]% 时为真 | failed 必须为假 | 禁止字符串/0/1 冒充 bool |
| `prefix_rmse_pp` | `fit_status` | 有限且 ≥ 0 | failed 必须为 `NaN` | 整列不得无有限值；不得为 ±∞ 或 numeric string |
| `prefix_max_abs_residual_pp` | `fit_status` | 有限且 ≥ 0 | failed 必须为 `NaN` | 同上 |
| `parameter_boundary_hit_fraction` | `fit_status` | 有限且在 [0,1] | failed 必须为 `NaN` | 同上 |

联合表不变量：

1. 每个 `(partition, cluster_id)` 精确包含冻结 86 个 `(model_id, variant_id)`；
   诊断 key 不重不漏，也不允许未声明 variant。
2. 每个诊断 key 在 forecast 表精确对应冻结八天 grid，forecast key 不重不漏。
3. succeeded 的八个 `raw_forecast_retention_pct` 必须有限；failed 的八个位置必须全
   为 `NaN`。±∞ 在任何状态都拒绝。
4. failed 的 `parameters_json` 必须精确为 `{}`；任意参数对象必须是 canonical、数值
   有限的 JSON object。
5. protocol、四列 variant key 与 `canonical_prefix_content_sha256` 在两表中逐行绑定。
6. 任何 mask 外 NaN、应缺失位置的伪有限值、状态/指标/可信度漂移均拒绝；没有填 0、
   删行、裁列、`formal=False` 或“任意 NaN 允许”的路径。

## V0.19 候选实现面

- `calendar_long_horizon_v019_numeric_contract.py`：新增联合 schema-aware validator；
  同时原样复用 V0.18 risk、prediction、decision numeric gates。
- `calendar_long_horizon_v019_partition.py`：复用 V0.18 冻结 contract 与 capability
  类型；非 member 表继续 blanket finite，member 两表改为联合状态 mask 校验；whole、
  derive、consume 三处均复核。
- `calendar_long_horizon_v019_runner.py`：仅提供 development-only `_fit_structure_stage`
  接线；无 CLI、seed、attempt 或正式协议身份。
- `tests/test_v019_member_fit_numeric_contract.py`：纯手写/确定性 fixture、逐列变异矩阵、
  formal canonicalization、精确基数和真实 fit-stage 调用图。

科学模型、特征、优化器、阈值、分区、端点、gates 与 V2.3 risk structural-NaN
contract 均未改变。

## 结果前复现与验证

全部命令使用主仓库既有 CPython 环境并把新 worktree 的 `src` 放在 import 路径首位；
未调用 generator、RNG、正式拟合或 truth capability。

1. 小型确定性/变异门：最终 `23 passed, 1 deselected in 18.90s`。
2. 精确基数门：全量 5,950 clusters、511,700 diagnostics、4,093,600 member
   forecasts，经 V0.19 `validate_whole_bundle_from_root` 和 development-only
   `_fit_structure_stage` 调用图；最终 `1 passed, 23 deselected in 15.51s`。fit/load/commit
   被无状态 sentinel 代替，未运行优化器。
3. V2.2/V2.3 risk/partition/terminal/preresult 回归：`39 passed in 940.52s`。
4. Ruff（3 个候选模块与新测试）：`All checks passed!`。

第一次精确基数 pytest 在测试体运行前因系统 `%TEMP%` pytest 根权限拒绝而中止；随后
使用新 worktree 内全新、受忽略的 basetemp 原样通过。该基础设施事件不是模型、契约或
科学终态，未删除或覆盖任何目录。

资源观测：精确基数门前可用内存约 10.11 GiB；长回归单 worker 的 Python 私有内存
约 1.69 GiB，C/D 可用空间保持约 11.7/76.0 GiB 量级，均低于 6-worker、10 GiB
scratch 上限且满足 8 GiB 可用内存目标。

## 边界与下一步

- 未读取或哈希分析 V2.3 正式 `member_fit_diagnostics.csv`，未读取密封真值、score、
  原始/第三方数据或旧正式模型输出。
- 未重跑/续跑 V2.3，未创建 a2；未创建 V2.4 正式 attempt、seed、四根或 runner。
- 本阶段只证明 member-fit contract 根因及候选实现正确性；不产生精度、评分、成功或
  真实 LFP/海辰产品/储能电站/15–25 年验证声称。
- 若本候选通过独立审查，唯一建议下一步是另立结果前 V2.4 prereg/implementation
  audit/freeze 阶段；在那之前本提交仍为 `development_preimplementation`。
