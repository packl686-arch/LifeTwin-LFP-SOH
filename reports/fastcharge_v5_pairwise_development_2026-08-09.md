# FastCharge V5 参考条件化残差与不确定性开发报告

作者：Jincheng Liu
日期：2026-08-09

## 结论先行

V5 在 MATR FastCharge 公开 LFP 队列上得到了一项值得保留、但不能越界解释的研发结果：由 41 个训练电芯内部五折选择的“12 参考电芯 + ExtraTrees 成对残差 + 加权均值”模型，在 81 个公开评估电芯上的 cycle-300 轨迹 MAE 为 **0.2082 个容量保持率百分点**，相对 V2 稳定硬门控的 0.2865 pp 降低 **27.3%**。

90% 区间分支使用训练电芯交叉拟合残差和参考集合分歧进行共形校准。支持门控后的评估覆盖率为 **93.10%**，平均区间宽度为 **1.4155 pp**，单区间 WIS 为 **0.1284**；V2 对应为 95.04%、2.8531 pp 和 0.2208。区间宽度减少 **50.4%**，WIS 减少 **41.8%**，最低前缀覆盖率为 **88.44%**。

但完整 H2 未通过。后续冻结的动态 landmark 审计已经执行：较长前缀重签发存在平均改善，但没有任何 GP 候选达到 70% 电芯改善门槛，训练内 P40 轻量 offset 在公开评估中也只改善 66.7% 电芯，因此在线残差分支保持关闭；详见 `reports/fastcharge_v5_dynamic_landmark_audit_2026-08-09.md`。当前仍没有跨数据集覆盖保证。支持门控使点预测 MAE 从无门控 V5 的 0.2082 pp 上升到 0.2215 pp，因此它暂时只能作为“签发/回退策略候选”，不能取代无门控 V5 的点精度结论。

全部结论均属于**结果已暴露的公开数据回顾性开发证据**。它们不是独立确认，不代表海辰产品精度，不验证日历老化，也不支持 15 至 25 年准确率宣称。

## 1. 研究问题

V5 检验两个可证伪问题：

1. 学习“目标电芯与参考电芯的早期差异如何映射为未来差异”，能否优于固定近邻轨迹迁移？
2. 参考集合分歧、参考距离和候选/回退分歧能否形成训练内选择的支持门控，并与严格物理电芯交叉拟合的共形区间结合？

Gaussian-process 在线残差、机理门控、主动试验选择和长期日历跨域验证没有在本轮完成。容量轨迹本身不能被命名为具体电化学机理。

## 2. 数据角色与防火墙

| 项目 | 设计 |
|---|---|
| 数据 | MATR FastCharge，LFP/石墨循环老化 |
| 模型选择 | 41 个训练电芯 |
| 公开评估 | 41 个 primary test + 40 个 secondary test |
| 早期前缀 | cycle 20、40、60、100 |
| 评分终点 | cycle 300 |
| 分组单位 | 物理电芯 |
| 训练内拆分 | 确定性、批次分层的五折物理电芯验证 |
| 成对样本防火墙 | 验证电芯同时不得作为 pair 的目标侧或参考侧 |
| 评估后缀 | 模型选择和预测阶段不可见；预测先写入并哈希，随后评分 |

训练周期表的规范化 SHA-256 为 `4a7ae36ac155f016fb8a0c7c6589c2e7e0cde78e5e292a77ec55ddca9c98b846`。训练成对防火墙审计保存在 `showcase/evidence_v5/pairwise_firewall_audit.json`。

## 3. 模型设计

### 3.1 成对参考残差

对目标电芯 `t` 和训练参考电芯 `r`，模型使用两者的早期轨迹描述、描述差异、参考未来变化和预测坐标，学习：

```text
目标未来保持率变化 - 参考未来保持率变化
```

预测时只允许从训练库检索参考电芯。每个参考产生一条候选未来轨迹，最终按早期描述距离聚合多个参考。筛选范围包含 Ridge、Huber、ExtraTrees、HistGradientBoosting，以及直接 Ridge/PCR/PLS/树模型对照；参考数为 4/8/12/16，聚合包含单近邻、加权均值和加权中位数。

### 3.2 支持门控

支持门控只使用训练交叉拟合输出选择阈值，观测三个信号：

- 目标与参考集合的平均距离；
- 多参考预测的平均离散度；
- 成对候选与固定近邻回退模型的平均分歧。

训练内选择 `union_q99`：任一信号超过训练诊断的 99% 分位时回退固定近邻。它在训练交叉拟合中触发 2.44% 的 cell-prefix，在公开评估中触发 1.23%。

### 3.3 共形不确定性

对每个训练电芯执行 leave-one-cell-out 共形校准：该电芯的残差绝不进入自己的校准集合。对每个前缀和未来 cycle 使用有限样本的 `ceil((n+1) * 0.9)` 绝对残差分位数。候选尺度包括：

- absolute：不缩放的绝对残差；
- reference_scaled：按参考集合离散度缩放；
- hybrid_scaled：按参考离散度与候选/回退分歧的组合缩放。

训练内以覆盖门槛和单区间 weighted interval score 联合选择 `hybrid_scaled`。这里的 WIS 按一个 90% 中央区间及其中位数预测计算；覆盖率与宽度必须同时报告。

## 4. 训练内选择结果

| 候选 | 五折平均轨迹 MAE（pp） |
|---|---:|
| V5 ExtraTrees，12 参考，加权均值 | **0.3544** |
| 最佳 HistGradientBoosting 成对模型 | 0.3692 |
| 直接 ExtraTrees | 0.3972 |
| 固定近邻对照的最佳配置 | 0.4614 |
| 固定近邻 k=8 加权均值 | 0.4710 |
| 最佳直接 PCR | 0.5299 |
| 最佳成对 Huber | 0.5829 |
| 最佳成对 Ridge | 0.5967 |

最终 V5 相对预先指定的固定近邻 k=8 加权均值改善 **24.75%**。10,000 次物理电芯配对 bootstrap 的 MAE 差异均值为 -0.1166 pp，95% 区间为 **[-0.2216, -0.0148] pp**，通过 H1 训练内门槛。

线性成对模型没有因为解释简单而被包装成成功；它们明显弱于非线性树模型，因此作为负结果保留。

## 5. 公开评估结果

### 5.1 点预测

| 指标 | V2 稳定硬门控 | V5 无支持门控 | 相对变化 |
|---|---:|---:|---:|
| 总体轨迹 MAE | 0.2865 | **0.2082** | -27.3% |
| cell-prefix P90 MAE | 0.6325 | **0.4798** | -24.1% |
| primary split MAE | 0.3411 | **0.2219** | -35.0% |
| secondary split MAE | 0.2305 | **0.1942** | -15.7% |

相对 V2 的 10,000 次物理电芯 bootstrap 差异 95% 区间为 **[-0.1503, -0.0175] pp**。两个作者 split 都改善，H1 公开开发门槛通过。

按前缀的 V5 MAE 为：P20 0.3114、P40 0.2017、P60 0.2089、P100 0.1109 pp。P40 到 P60 的非单调变化提醒我们：更多数据通常有帮助，但不能保证每个 landmark 都严格改善。

### 5.2 门控与区间

| 指标 | V2 区间 | V5 支持门控 + 共形区间 |
|---|---:|---:|
| 点预测 MAE | 0.2865 | **0.2215** |
| 90% 经验覆盖率 | 95.04% | 93.10% |
| 最低前缀覆盖率 | 未作为本轮比较主张 | **88.44%** |
| 平均区间宽度 | 2.8531 pp | **1.4155 pp** |
| 单区间 WIS | 0.2208 | **0.1284** |

V5 区间在保留接近名义覆盖率的同时显著变窄，区间子门槛通过。但无门控 V5 的点 MAE 仍更低，因此当前结论是：

- **无门控 V5**：公开开发点精度的主结果；
- **支持门控 V5**：待独立批次验证的签发策略候选；
- **完整 H2**：后续实验已执行但未通过；固定 GP 未达到改善电芯比例门槛，在线残差分支不激活；
- **正式覆盖保证**：不存在，必须在新的未接触批次或跨域队列中验证。

## 6. 可复核产物

公开仓库不包含 MATR 原始测量和大型预测 Parquet。以下小型派生产物足以复核选择规则、审计边界和汇总数字：

- `showcase/evidence_v5/pairwise_training_selection.json`
- `showcase/evidence_v5/pairwise_training_cv_summary.csv`
- `showcase/evidence_v5/pairwise_firewall_audit.json`
- `showcase/evidence_v5/pairwise_prediction_manifest.json`
- `showcase/evidence_v5/pairwise_evaluation_summary.json`
- `showcase/evidence_v5/pairwise_cell_prefix_scores.csv`
- `showcase/evidence_v5/support_uncertainty_development.json`
- `showcase/evidence_v5/support_uncertainty_prediction_manifest.json`
- `showcase/evidence_v5/support_uncertainty_score_summary.json`
- `showcase/evidence_v5/support_gate_screen.csv`
- `showcase/evidence_v5/interval_method_screen.csv`
- `showcase/evidence_v5/calibration_quantiles.csv`
- `showcase/evidence_v5/support_uncertainty_cell_prefix_scores.csv`

运行入口：

```powershell
$env:PYTHONPATH='src'
python scripts/run_fastcharge_v5_pairwise_development.py screen
python scripts/run_fastcharge_v5_pairwise_development.py predict
python scripts/run_fastcharge_v5_pairwise_development.py score
python scripts/run_fastcharge_v5_support_uncertainty.py crossfit
python scripts/run_fastcharge_v5_support_uncertainty.py predict
python scripts/run_fastcharge_v5_support_uncertainty.py score
```

原始数据路径、来源版本和预处理产物仍需按仓库数据说明准备；命令不会下载或重新分发上游原始数据。

## 7. 结论边界与下一步

本轮证明的是：在同一公开 LFP 循环老化域内，成对参考残差、多参考聚合、训练内支持门控和严格电芯级共形校准具有实际可行性。

本轮没有证明：

- 长期日历老化或储能电站真实混合工况精度；
- 跨厂商、跨规格、跨协议的迁移能力；
- 海辰内部电芯、系统或站级精度；
- 15 至 25 年寿命预测准确率；
- 完整 GP 在线更新或正式跨域覆盖保证。

公开开发队列的动态 landmark 预注册实验已经完成并因 GP/offset 分支未达门槛而回退。下一步应保持协议不变，在新的 outcome-blind 物理电芯批次或企业 truth vault 上复现；只有当在线更新改善比例、尾部误差、覆盖率和区间宽度同时通过，才允许晋级企业影子试点。
