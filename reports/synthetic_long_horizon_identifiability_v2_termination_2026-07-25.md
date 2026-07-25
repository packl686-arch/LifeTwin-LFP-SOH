# LifeTwin V0.15 正式尝试终止报告

作者：Jincheng Liu

协议：`synthetic_long_horizon_identifiability_v2`

尝试：`v015-20260725T234818-cst`

冻结实现：`3631ee7bf86ebd8890b226551bca259390444ad3`

## 结论

本次正式尝试的状态是 **`inconclusive_not_success`（无结论，不能算成功）**。
程序在 calibration 阶段发现完整性门槛不满足后停止：900 个 calibration
cluster 中有 899 个具备至少两个可信结构族，另 1 个只有一个可信结构族。冻结
协议要求完整 900/900 校准，不能删掉该样本后继续。

这不是模型主终点的 `failure`。程序尚未生成 `model_state.json`、预测承诺或
评分包，也没有打开 test、audit 和两类 matched truth，因此没有任何留出集
性能数字可供解释。它同样不是 `success`，也不能通过改阈值、换 seed 或重跑
同一 V2 来补救。

## 发生了什么

正式运行先完成了无标签生成和 511,700 行结构拟合诊断，并提交全部真值与拟合
文件的 SHA-256。随后按顺序只打开 center、risk 和 calibration 三个开发真值：

| 阶段 | 状态 | 冻结证据 |
|---|---|---|
| 真值生成与承诺 | 完成 | `truth_commitments.json` |
| 结构库拟合与承诺 | 完成 | `fit_commitment.json` |
| center 开发 | 完成 | `center_beta=0.3452770144` |
| risk 开发 | 完成 | 600/600 合格，214 个正例、386 个负例 |
| calibration | 终止 | 899/900 满足硬资格 |
| 预测与盲态评分 | 未开始 | 无预测承诺，留出真值未打开 |

从第一条日志到终止日志共 6,521 秒。暴露日志的 13 行、末尾换行和字节哈希均
已验证；完整承诺链见
[`formal_attempt_termination_manifest.json`](../showcase/evidence_v015/synthetic_long_horizon_identifiability_v2/formal_attempt_termination_manifest.json)。
仓库内的
[`verify_v015_terminal_evidence.py`](../scripts/verify_v015_terminal_evidence.py)
会重新计算公开副本哈希、回放 13 行状态机，并拒绝伪造的评分状态、真值暴露
边界或承诺交叉引用。

## 根因

异常 cluster 为 `c_12275fb066dc74166b6382de519e626d`。它的 86 个候选变体
全部成功完成数值拟合，但只有
`target_prefix_early_activation_plus_power` 通过统一可信度规则：

- 可信变体的 prefix RMSE 为 `0.0777973 pp`，最大绝对残差为
  `0.156630 pp`。
- bounded power、dual power、sqrt time、saturating 以及 80 个 late-knee
  变体的 RMSE 均不超过 `1 pp`，但最大绝对残差的最小值为
  `1.510386 pp`，略高于冻结的 `1.5 pp` 上限。
- persistence 的 RMSE 为 `1.339761 pp`，最大绝对残差为 `2.012214 pp`。
- 该 cluster 的 12 个 prefix 点、16 个工况/安慰剂字段及 14 个 prefix
  特征均完整且有限；决定性条件只是
  `successful_structure_family_count=1`。

因此，这不是文件缺失、NaN、求解器失败或公式实现错误，而是新 seed 下真实
触发了预先声明的证据不足条件。程序拒绝悄悄删行是正确行为。

诊断只使用已经承诺的 label-free 文件。核心复核查询为：

```sql
WITH x AS (
  SELECT
    partition,
    cluster_id,
    count(DISTINCT model_id) FILTER (WHERE credible_variant)
      AS credible_families,
    count(*) FILTER (WHERE fit_status = 'failed') AS failed_variants
  FROM read_csv_auto('member_fit_diagnostics.csv', header = true)
  WHERE partition = 'calibration'
  GROUP BY 1, 2
)
SELECT *
FROM x
WHERE credible_families < 2
ORDER BY cluster_id;
```

可信结构族数量在 900 个 calibration cluster 上的分布为：

| 可信结构族数 | cluster 数 |
|---:|---:|
| 1 | 1 |
| 2 | 2 |
| 3 | 8 |
| 4 | 0 |
| 5 | 1 |
| 6 | 293 |
| 7 | 595 |

## 证据边界

原始暴露日志只记录打开了：

1. `center_development_truth.csv`
2. `risk_development_truth.csv`
3. `calibration_truth.csv`

`test_truth.csv`、`audit_truth.csv`、两类 matched truth 及 pair mapping 均未被
模型或评分器打开。公开报告也不包含这些文件的内容。它们已有生成前承诺，但
承诺哈希不等于揭盲。

当前能够保留的有效证据是冻结实现、数据与拟合承诺、center checkpoint、risk
checkpoint、training manifest 和追加式暴露日志。不能创建空的
`model_state.json`、预测表或评分表，因为那会伪造一个没有发生的阶段。

## 实现缺口

冻结规则已经把 calibration 不完整和拟合/校准不可定义列为
`inconclusive_not_success`，但 V2 的完整产物注册表又无条件要求模型状态、
预测承诺和评分表。训练在模型状态产生前终止时，这两组要求无法同时满足。

当前实现选择 fail closed：留下真实阶段日志，不伪造下游产物。代价是终止日志
只保存了通用错误文本，没有在运行当时把稳定 reason code、异常摘要和 traceback
哈希写入一个正式终止清单。本报告的精确根因来自当时控制台信息与随后对
label-free 文件的可复现诊断；它不是原始日志自身完整承诺的异常文本。

## 下一版修正

这次 V2 将原样保留。后续只能建立新的 `V2.1` 协议、重新预注册并使用全新种子，
而不是改写本次尝试。V2.1 至少需要：

1. 在执行前声明独立的 terminal-inconclusive 产物注册表，原子写入 reason
   code、异常摘要、traceback/stderr 哈希和暴露边界。
2. 明确区分用于风险概率校准的硬资格池与用于轨迹 conformal 的固定 900
   分母，避免把“不能发行风险决策”误等同于“不能形成有限预测区间”。
3. 对 0 或 1 个可信结构族的区间回退规则、风险 isotonic 最小样本数和所有固定
   分母重新预注册，再用新 seed 一次性检验。
4. 在新正式运行前加入至少一个 899/900、零可信族和校准异常的端到端故障注入
   测试，确保无结论可以被完整、诚实地发布。

## 声明边界

本实验是合成机制压力测试，不含海辰内部数据，也不构成真实 LFP 电芯、储能
电站或 15 至 25 年 SOH 精度证明。本次终止只说明冻结证据完整性门槛被触发；
它没有测得留出集上的模型效果。
