# Demo Fixture 说明

本目录存放 **评委可视化 Demo** 专用的公开 fixture 数据。所有内容均为界面演示用途，不代表正式模型实验结果，且与正式实验目录完全隔离。

## 目录用途

- 只被 `docs/demo/index.html` 读取，用于浏览器端展示。
- 不参与正式实验流水线、训练或评分流程。
- 文件命名与结构遵循 `docs/demo/schema/demo_summary.schema.json`。

## 文件列表

| 文件 | 说明 |
|------|------|
| `model_main.json` | 主模型公开状态卡。默认 `unavailable`，无指标，含 9 字段比较口径。 |
| `model_independent.json` | 独立模型公开状态卡。结构与主模型相同，仅边界文本不同。 |
| `model_unavailable.json` | 显式 unavailable 示例，用于展示缺省状态。 |
| `update_demo.json` | 动态更新演示序列：4 步前缀扩展，展示 predict → fallback → reject。 |
| `workbench_scenarios.json` | 工作台选择器场景：fallback / specialist_reject / ood_reject。 |

## 状态机约束

每个模型文件必须满足以下之一：

- **scored**：`prediction_commitment=true`, `scored=true`, `metrics` 非空, `terminal` 必须不存在。
- **terminal_pre_prediction**：`prediction_commitment=false`, `scored=false`, `metrics` 必须不存在, `terminal` 必须存在。
- **unavailable**：`prediction_commitment=false`, `scored=false`, `metrics` 必须不存在, `terminal` 必须不存在, `gages` 必须不存在, `public_version` 必须不存在, `protocol_id` 必须不存在。

## 比较口径（fail-closed）

`comparison` 对象必须包含以下 9 个字段，两侧完全一致且 `scored=true` 才允许数值比较：

- `data_version`
- `prefix_definition`
- `prediction_range`
- `partition_id`
- `scoring_rule_id`
- `metric_name`
- `metric_unit`
- `protocol_id`
- `scored`

## 命名规范

- 区间字段使用 `nominal_interval_level`（名义诊断区间等级），禁止使用 `diagnostic_interval_coverage`。
- 所有数值均为界面演示数据，不反映真实实验输出。

## 数据漂移预防

`docs/demo/index.html` 在构建时将这些 fixture 内嵌为 `<script id="demo-data" type="application/json">`。构建流程需保证内嵌 JSON 与 fixture 文件完全一致。验证命令：

```bash
node scripts/validate_demo.js
```

## 证据边界

- 本目录数据仅用于公开演示，不包含任何企业敏感信息。
- 历史案例（如 Naumann 0.305 pp）仅在 `index.html` 的“历史公开证据案例”区域作为证据展示，不作为模型默认指标。
- 正式实验结果存放于 `showcase/evidence_v011/`、`showcase/evidence_v012/`、`showcase/evidence_v014/` 等白名单目录，与本目录无数据共享。

## 修改注意事项

- 修改 fixture 后需同步重新构建 `index.html`。
- 不得引入 `terminal`、`metrics` 或状态字段违反状态机约束的数据。
- 不得删除或重命名 `comparison` 中的 9 个必填字段。
