# V8 测量稳定性盲测执行手册

## 1. 为什么做 V8

V7-P100 在同一批 41 个开发电芯的嵌套审计中表现很好，但冻结后的噪声审计发现其门控会被很小的前缀扰动误触发。因此 V8 不再继续调准确率阈值，而是先回答一个更基础的问题：**当前测量精度是否足以支持一次非零模型更新？**

当前结论不变：V5 是 champion，V7 已撤回，V8 只是待真实数据执行的盲测协议。

## 2. 三阶段实验闭环

### Stage A：只测量噪声，不读取寿命结局

输入 CSV 必须且只能包含以下 10 列：

| 字段 | 含义 |
|---|---|
| `record_role` | `cell_repeat`、`daily_reference` 或 `tester_bridge` |
| `physical_cell_id` | 物理电芯标识 |
| `landmark_cycle` | 重复测量对应循环数 |
| `repeat_index` | 从 0 开始的重复序号 |
| `retention_pct` | 容量保持率，单位 `%` |
| `tester_id` | 测试设备标识 |
| `temperature_chamber_id` | 温箱标识 |
| `measurement_date` | ISO 日期或时间 |
| `reference_channel_id` | 日参考通道标识；非参考行填空字符串 |
| `bridge_id` | 跨设备桥接组标识；非桥接行填空字符串 |

最低要求是 20 个物理电芯、循环 60 和 100 各至少 3 次独立重复测量、每个设备/温箱噪声组至少 5 个电芯和 20 条残差，同时包含日参考与跨设备桥接记录。

重复测量减去自身均值后会低估单次测量噪声。实现中按 `sqrt(1 - 1/n)` 修正该方差收缩，再用物理电芯留一法在零均值 Gaussian 与固定自由度 Student-t 候选中选择噪声族。以下任一项失败即停止 V8：

- 重复顺序斜率超限；
- 日参考漂移超限；
- 测试设备桥接偏差超限；
- 任一设备/温箱噪声组样本不足。

```powershell
$env:PYTHONPATH='src'
python scripts/prepare_fastcharge_v8_measurement_quality.py `
  --config configs/experiments/v8_measurement_stability_execution.template.json `
  --measurements D:\private-input\v8_repeatability.csv `
  --output-directory D:\private-output\v8-stage-a
```

输出目录必须为空，生成 `noise_candidate_scores.csv`、`noise_ledger.csv`、`decision.json` 和 `manifest.json`。这些文件只证明测量质量，不能证明模型准确率。

### Stage B：看得到前缀，看不到未来

每个新电芯在 P100 时生成一个严格 JSON 请求。请求模式见 `configs/experiments/v8_measurement_stability_request.schema.json`，只允许：

- 循环 61-100 的已观测保持率；
- 当时保存的 P60 V5 中心轨迹；
- 当前 P100 V5 中心轨迹；
- 设备、温箱和制造批次身份；
- 循环 101-300 的坐标，不含相应真实容量。

任何额外字段都会被拒绝，因此 `future_truth`、未来 SOH 或未来容量不能混入请求。

```powershell
python scripts/issue_fastcharge_v8_measurement_stability.py `
  --config configs/experiments/v8_measurement_stability_execution.template.json `
  --candidate configs/experiments/v7_p100_reissue_innovation_blind_candidate.json `
  --request D:\private-input\requests\CELL_001.json `
  --measurement-quality-decision D:\private-output\v8-stage-a\decision.json `
  --noise-ledger D:\private-output\v8-stage-a\noise_ledger.csv `
  --output-directory D:\private-output\v8-issuances\CELL_001
```

真实执行使用 1024 次测量噪声重采样。只有以下条件全部满足才发出非零修正：原始 V7 门控激活、重采样激活概率不低于 95%、修正方向概率不低于 95%、终点修正偏差 P95 不高于 0.05 pp、测量质量通过且设备/温箱映射完整。否则输出与当前 P100 V5 中心逐元素完全相同。

所有电芯签发后再编译队列承诺：

```powershell
python scripts/compile_fastcharge_v8_stage_b_commitment.py `
  --config configs/experiments/v8_measurement_stability_execution.template.json `
  --protocol configs/experiments/v8_measurement_stability_blind_protocol.template.json `
  --candidate configs/experiments/v7_p100_reissue_innovation_blind_candidate.json `
  --issuance-root D:\private-output\v8-issuances `
  --output-directory D:\private-output\v8-cohort-commitment
```

未来结局开放前至少需要 60 个不重复物理电芯、3 个制造批次、6 个稳定激活、10% 稳定激活覆盖率，并且激活来自至少 2 个批次。脚本逐文件校验预测、决策、清单和 SHA-256 承诺；未达到任一门槛时只生成“继续使用 V5、不得开放 Stage C”的结果。

### Stage C：一次性开放并评分

只有 `cohort_prediction_commitment.json` 明确授权后，独立保管方才可开放循环 101-300 的真实轨迹。评分必须一次完成，不得在同一批次上修改噪声模型或阈值后重跑。主要门槛仍按预注册协议执行：激活精度、全体 MAE 差、active P90/最差差值，以及分批次结果必须全部通过。即使通过，也只能提名下一次独立域验证，不能直接上线。

## 3. 当前已完成与尚缺证据

已完成的是软件级合成演练：24 个生成身份、192 行无结局测量记录、两个设备/温箱组。演练选择 Gaussian 噪声，估计尺度分别为 `0.002874 pp` 与 `0.003081 pp`；稳定路径的激活概率和方向概率均为 `1.0`，终点修正偏差 P95 为 `0.00794 pp`，缺失映射时精确回退 V5。

这些数字只用于验证代码路径，不是电池模型结果。当前仍缺：

1. 真实重复测量、日参考和跨设备桥接数据；
2. 至少 60 个新盲测电芯及 3 个制造批次；
3. 在预测承诺之后才开放的循环 101-300 真实轨迹；
4. 独立 Stage C 评分与批次级复核。

当前门控把已经签发的 P60/P100 V5 中心视为固定量，只传播容量测量对残差门控的影响；它尚未覆盖“测量扰动后重新运行整个 V5，中心轨迹本身如何变化”，也未建模跨循环相关噪声。Stage C 的新电芯评分能够检验最终净效果，但在生产化前仍应另做端到端 perturb-and-refit 与相关噪声实验。

## 4. 操作红线

- 私有原始数据和私有输出均放在仓库外，不提交 GitHub；
- Stage A、单电芯签发和队列承诺均使用全新空目录，禁止覆盖旧证据；
- Stage A 结果、噪声台账、请求、预测和代码均由哈希串联；
- 不把合成演练写成“模型提升”，不把 FastCharge 循环老化写成 15-25 年日历寿命验证；
- 任一数据缺失、映射缺失、漂移失败或稳定性失败都回退 V5，不人工强制激活。
