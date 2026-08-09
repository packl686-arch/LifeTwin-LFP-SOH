# LifeTwin V0.16 / V2.1 正式终态与根因收口

日期：2026-08-09
证据等级：冻结六核心族合成 25 年协议的正式终态记录

## 1. 结论

V0.16 / V2.1 只执行了一次正式尝试：`v021-formal-20260808-a1`。该尝试在形成
prediction commitment 前终止，正式注册表为 `terminal_pre_prediction`，
`scientific_status` 与 `attempt_disposition` 均为
`unclassified_terminal_not_success`，原因码为
`UNKNOWN_PRE_PREDICTION_EXCEPTION`。

这不是成功，也不是已评分失败。该尝试没有生成 prediction commitment，score 根为空，
因此没有五主端点、八族 gate、novel safety gate 或评分分母可以报告。V0.14 的
`failure` 与 V0.15 的 `inconclusive_not_success` 均保持不变。

## 2. 冻结身份与终态

- attempt：`v021-formal-20260808-a1`
- 输入提交：`9f886202de07fd46bc6fd0604e60315b32801bc2`
- 输入提交父节点：`d676d7c3a7fffb7806e92822c35d4a027d8677bf`
- last completed phase：`center_truth_opened`
- attempted phase：`center_state_committed`
- 注册表显示仅打开：`center_development_truth.csv`
- prediction commitment：不存在
- score：空
- termination registry：严格三个文件

终态文件的 SHA-256 锚点如下：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `terminal_artifact_manifest.json` | 3,208 | `ea2afaf70cf253ecd28f24f700d716c8011a41fec44bfb0c5fd541dda602b0ea` |
| `terminal_attempt_record.json` | 2,489 | `a7feec9d99f8726e5f83a2dd0933f4d9f360b0ed3a1836ca9951970804bbefd7` |
| `terminal_exposure_log_snapshot.jsonl` | 6,971 | `8cf8040c3f1afd25f904bec76299e0699b7cb95ca740f3cf65b8896fa0716e40` |

活动 exposure log 为 7,852 bytes，SHA-256 为
`bbf2f83f5bdf84a05b0bec49605c8575821edc7e64ea93996559263eacf2c5b9`。
它以 6,971-byte 终止快照为精确前缀，之后只有一条 terminal manifest commitment；
这是冻结实现规定的双视图，不是哈希失败。

## 3. 根因

根因是实现与 artifact contract 的作用域不匹配：

1. [`_subset_partition`](../src/lifetwin/experiments/calendar_long_horizon_v016_runner.py)
   先把五张全量 label-free 表裁成 `center_development`；
2. [`_apply_partition`](../src/lifetwin/experiments/calendar_long_horizon_v016_runner.py)
   仍以 `formal=True` 调用 V2.1 pipeline；
3. [`_recompute_label_free_pipeline_with_state_v021`](../src/lifetwin/experiments/calendar_long_horizon_v016_pipeline.py)
   首先验证 `prefix_pack.csv`；
4. [`canonicalize_frame`](../src/lifetwin/experiments/calendar_long_horizon_v015_io.py)
   因而把全量 `required_rows=71,400` 的规则应用到 600 clusters × 12 prefix days =
   7,200 行的合法 center 子集，并抛出：

```text
V021PipelineError: prefix_pack.csv row count is 7200, expected 71400
```

只读复核排除了输入类型、列顺序、重复 key、缺失值、非有限值、协议 ID、分区名与
12 个 prefix day 单位/网格问题。由于有序 generator 在第一张表即终止，后四张表没有
参与本次触发。五张全量表此前已经通过正式 reader 并完成 label-free fit commitment。

`V021PipelineError` 不属于冻结 terminal classifier 注册的科学、完整性或中断异常类型，
所以历史终态按类型白名单落入 `unknown_default`。识别出实现根因不会追溯改写正式注册表。

相关冻结依据：

- [V2.1 预注册](synthetic_long_horizon_identifiability_prereg_v2_1.md)
- [V2.1 实现冻结审计](synthetic_long_horizon_identifiability_implementation_audit_v2_1.md)
- [V2.1 冻结记录](synthetic_long_horizon_identifiability_freeze_record_v2_1.json)
- [pipeline 定向测试](../tests/test_v016_pipeline.py)
- [runner 生命周期测试](../tests/test_v016_runner.py)
- [terminal 分类测试](../tests/test_v016_terminal.py)

## 4. 后续决策

V2.2 当前只允许作为未预注册、未授权执行的结果前候选设计，不是正式协议、实现冻结或
实验结果。候选方向必须保留 whole-bundle formal 校验，再通过承诺绑定的 capability
执行精确分区切片；禁止用 `formal=False` 关闭校验。

2026-08-16 前的决定为 No-Go：不实施、不执行新长期正式实验，优先完成 PPT、演示、
答辩与提交包。任何后续版本都必须先形成新协议、测试门和冻结记录，再单独取得执行授权；
不得重跑、续跑或改写 a1。

## 5. 宣称边界

本终态只说明冻结六核心族合成 25 年协议在预测前因实现契约问题终止。它不证明真实
LFP 电芯、海辰产品、储能电站或 15–25 年真实预测精度，也不增加独立验证证据。
