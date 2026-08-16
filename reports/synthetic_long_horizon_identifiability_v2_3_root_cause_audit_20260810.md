# LifeTwin V2.3 结果前数值契约根因审计

日期：2026-08-10
状态：development_only / no_formal_attempt / no_truth_access

## 结论

已确认一个足以独立解释 V2.2 终止、且可用纯合成 fixture 稳定复现的输出契约冲突：
既有 `risk_bundle.csv` 语义要求用结构性 `NaN` 表示非主 score 没有校准概率，以及不具
资格的 primary score 不发行校准概率；V2.2 新增的 partition validator 却对所有数值列
无差别要求 `isfinite`，因而必然拒绝合法的 risk bundle。

该结果把正式 traceback 的触发链闭合到契约实现，但不依赖、也没有检查 V2.2 正式输出
中是否还存在其他数值问题。V2.3 必须通过独立开发 fixture 和跨输出对抗测试排除这些
附加风险，不能仅删除旧检查后直接进入正式执行。

该结论不追溯改变 V2.2 正式终态。唯一正式 attempt 仍为
`terminal_pre_prediction / void / proven_integrity`，没有 prediction commitment、没有
评分，也没有打开真值。

## 结果前复现

复现只构造冻结 center cardinality 所需的 600 clusters × 9 score IDs，共 5,400 行。
fixture 不调用 generator、不使用 seed、不读取 sealed truth，也不读取 V2.2 正式 fit、预测
或评分数据。

复现得到三个相互独立的事实：

1. 冻结 CSV schema 接受结构性空值；
2. V2.2 partition validator 对同一合法 frame 抛出
   `risk_bundle.csv contains a nonfinite numeric value`；
3. 把结构性空值错误填成 0 会绕过旧门，但改变“未校准/不发行”的科学语义，因此禁止
   作为修复。

## V2.3 修复规则

V2.3 开发实现改为验证精确的跨表缺失掩码：

- `raw_risk_score` 只在 `all_features_finite=false` 时为空；
- `calibrated_catastrophic_probability` 只对 hard-eligible 的两个 primary score 有限，
  其他位置必须为空；
- 有限概率必须位于 `[0, 1]`；
- 所有 `+inf/-inf`、掩码外 NaN、结构性位置上的伪造有限值、risk/feature 掩码漂移继续
  失败关闭；
- 不填充、不裁剪、不关闭 formal 校验，不改变 seed、数据分区、阈值、端点或成功条件。

本阶段只是 V2.3 数值契约开发证据，不是预注册、实现冻结或新正式结果。完整 V2.3
pipeline 接入、跨输出契约、对抗测试、环境锁和 freeze record 仍须在任何新 seed 使用前
完成。

## 验证矩阵

- V2.3 结构性空值与数值破坏对抗测试：6 passed；
- V2.2 partition capability 与 terminal 分类回归：9 passed；
- V2.2 结果前 seed/collision/freeze 门回归：4 passed；
- ruff 与 Python 编译检查：通过。
