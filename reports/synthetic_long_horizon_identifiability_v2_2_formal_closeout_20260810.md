# LifeTwin V2.2 正式终态收口

日期：2026-08-10
证据等级：冻结六核心族合成 25 年协议的正式终态记录

## 1. 结论

V2.2 只执行了一次正式尝试：`v022-formal-20260809-a1`。该尝试自然结束为
`terminal_pre_prediction`；`scientific_status` 与 `attempt_disposition` 均为 `void`，
分类模式为 `proven_integrity`，原因码为
`INTEGRITY_PARTITION_CONTRACT_MISMATCH`。

流程已经形成 fit commitment，但没有形成 prediction commitment，没有评分，且
`opened_truth_files=[]`。因此本次没有五主端点、族 gate、negative control 或评分分母
可以报告，也不能称为成功或已评分失败。

## 2. 冻结身份与终态

- protocol：`synthetic_long_horizon_identifiability_v2_2`
- attempt：`v022-formal-20260809-a1`
- 冻结提交：`4f116325f6d414f3e3a452cc53541b989b4dfe5b`
- 冻结配置 SHA-256：
  `aaadd5b9d5436d6ccfa08806250f0a48bef93e04446d0c089cb2eb5cf8ce0f29`
- last completed phase：`label_free_fit_committed`
- attempted phase：`center_truth_opened`
- fit commitment SHA-256：
  `7cf3eca1e45e816386295eb032d56836ed0fa381ae73b3f4d604af33a2af883d`
- prediction commitment：不存在
- opened truth files：0
- score：空
- 正式运行时长：59,723.856 秒（16 小时 35 分 23.856 秒）
- termination registry：严格三个文件

正式终态文件的 SHA-256 锚点如下：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `terminal_artifact_manifest.json` | 3,208 | `4da9f3bb2182fab3da6fdad0b180353c7b16380fccf1bead2158132f99778b8e` |
| `terminal_attempt_record.json` | 2,539 | `686a0f18d2821c344d11502f119dac07f19b2b85ecb59fff05de8443307241ad` |
| `terminal_exposure_log_snapshot.jsonl` | 4,729 | `c4a1f3027f51fab9ed44d9a7b284bbd0a81974989e049623b2cec22d23e94827` |

终止快照是活动 exposure log 的精确前缀；活动日志随后只追加一条 terminal manifest
commitment，且其中引用的 manifest、attempt record 与 snapshot SHA-256 均与上述文件
一致。终止注册表与空 score 注册表互斥。

## 3. 触发点与完整性判断

直接触发点是 `risk_bundle.csv` 在冻结分区输出有限值契约检查中含有非有限数值。
冻结实现因此抛出 `V022PartitionContractError`，并按照结果前固定的异常分类映射，将本次
attempt 登记为 `proven_integrity` 的 `void`。这说明完整性契约阻止了非法输出继续进入
真值打开、prediction commitment 和评分阶段。

阶段名 `center_truth_opened` 表示尝试进入该阶段，不表示已经读取真值；正式记录中的
`opened_truth_files=[]` 才是实际暴露边界。当前证据可以确定非有限输出是契约触发点，
但尚未完成对其更上游数值成因的定位。不得据此写成问题已经彻底修复，也不得直接推断
整个模型已经失效。

## 4. 历史与宣称边界

- V0.14 的正式结论仍为 `failure`。
- V0.15 的正式结论仍为 `inconclusive_not_success`。
- V2.1 的唯一正式尝试仍为
  `terminal_pre_prediction / unclassified_terminal_not_success`，不重跑、不续跑。
- V2.2 没有预测和评分结果，不能报告精度、端点通过率或成功结论。
- 本次只属于冻结六核心族合成 25 年协议，不代表真实 LFP 电芯、海辰产品、储能电站或
  15–25 年真实精度验证。

此前赛前 No-Go 是执行授权前的历史判断；后续明确授权只允许以新协议、新数据、新 seed、
新 attempt 和隔离工作树前向执行 V2.2。该授权没有改变任何旧版本终态，也没有授权重试
当前 attempt。

## 5. v0.14.1 Release 日期化补充文案（待审，不构成 Release 修改）

> **2026-08-10 研究状态补充：**唯一 V2.2 正式尝试
> `v022-formal-20260809-a1` 自然结束为
> `terminal_pre_prediction / void / proven_integrity`，原因码为
> `INTEGRITY_PARTITION_CONTRACT_MISMATCH`。流程形成了 fit commitment，但没有
> prediction commitment、没有评分，也没有打开任何真值文件。直接触发点是
> `risk_bundle.csv` 在冻结分区输出有限值契约检查中出现非有限数值，完整性门按设计阻止
> 了非法输出；更上游数值成因尚未完成定位。本结果不构成预测精度、成功结论、真实 LFP
> 电芯、海辰产品或储能电站验证，也不改变 V0.14、V0.15 和 V2.1 的历史结论。
