# V0.15 正式尝试证据

协议 `synthetic_long_horizon_identifiability_v2` 的首次正式尝试在校准阶段按冻结门槛
终止。其科学状态是 `inconclusive_not_success`，但没有到达预测承诺和盲态评分，
因此不存在可报告的 test、audit 或 matched endpoint 结果，也不存在完整 V2
评分包。

终止原因不是优化器崩溃。900 个 calibration cluster 中有 899 个满足“至少两个
可信结构族”的冻结硬资格条件，另 1 个只有一个可信结构族。协议要求完整
900/900 校准，程序没有静默删掉该行，也没有打开后续真值。

公开的小型证据副本包括：

- [`formal_attempt_termination_manifest.json`](synthetic_long_horizon_identifiability_v2/formal_attempt_termination_manifest.json)：
  尝试身份、状态、根因诊断、暴露边界和承诺链。
- [`exposure_log.jsonl`](synthetic_long_horizon_identifiability_v2/exposure_log.jsonl)：
  原始 13 行追加式阶段日志。
- [`truth_commitments.json`](synthetic_long_horizon_identifiability_v2/truth_commitments.json)
  与 [`fit_commitment.json`](synthetic_long_horizon_identifiability_v2/fit_commitment.json)：
  未提交大文件的字节数、行数和 SHA-256。
- [`center_state_checkpoint.json`](synthetic_long_horizon_identifiability_v2/center_state_checkpoint.json)、
  [`risk_state_checkpoint.json`](synthetic_long_horizon_identifiability_v2/risk_state_checkpoint.json)
  与 [`training_manifest.json`](synthetic_long_horizon_identifiability_v2/training_manifest.json)：
  终止前已经合法冻结的开发状态。

完整解释见
[`reports/synthetic_long_horizon_identifiability_v2_termination_2026-07-25.md`](../../reports/synthetic_long_horizon_identifiability_v2_termination_2026-07-25.md)。
冻结配置、代码、种子和原始尝试目录均保留不变；不会通过改阈值、换 seed 或
删除 calibration 行把这次尝试包装成成功。
