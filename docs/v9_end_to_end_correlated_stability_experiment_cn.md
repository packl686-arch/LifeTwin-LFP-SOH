# V9 端到端相关扰动稳定性实验

## 1. 实验要回答什么

V8 已能回答“固定 P60/P100 V5 中心后，末端残差门控是否会被重复测量噪声误触发”，但还没有回答两个更接近真实部署的问题：

1. 测量误差进入原始前缀与历史参考库后，V5 的拟合结果、近邻集合和中心轨迹是否稳定；
2. 误差若带有设备偏置、跨循环相关性和缓慢漂移，V8 的 IID 重采样结论是否仍成立。

V9 因此不是新增一个追求更低 MAE 的模型，而是 V7/V8 进入真实盲测前的资格实验。核心假设是：**只有当“测量数据 -> V5 重拟合 -> 近邻重选 -> P60/P100 重签发 -> V7 更新”整个链条在相关扰动下仍稳定，非零动态修正才有资格进入后续一次性盲测。**

## 2. 为什么不能只加白噪声

电池测试误差既包括随机重复性，也包括环境、设备、流程造成的系统偏差。Taylor 等对电池表征实验的误差来源进行了系统拆分，并指出这些误差会沿模型开发链传播；Plett 等在 LFP 电芯上说明偏差造成的状态估计误差可能大于随机噪声；联合移动时域估计研究也把偏差与噪声的组合情形作为单独问题处理。因此 V9 将以下五类分量分开登记：

- 单循环 IID 重复性误差；
- 测试柜与温箱共同偏置；
- 跨循环 AR(1) 相关误差；
- 线性漂移或随机游走；
- 低概率重尾尖峰。

参考依据：

- Taylor et al., *Journal of Energy Storage* 24 (2019), 100761, [DOI: 10.1016/j.est.2019.100761](https://doi.org/10.1016/j.est.2019.100761)
- Plett et al., *Journal of Energy Storage* 11 (2017), 86-92, [DOI: 10.1016/j.est.2017.01.006](https://doi.org/10.1016/j.est.2017.01.006)
- Joint moving-horizon estimation under combined uncertainty, [DOI: 10.1016/j.est.2021.103316](https://doi.org/10.1016/j.est.2021.103316)
- Richardson, Osborne and Howey, Gaussian-process SOH forecasting, [arXiv:1703.05687](https://arxiv.org/abs/1703.05687)

这些论文只支持实验设计动机，不代表其结果能直接迁移到海辰电芯。

## 3. 三阶段协议

### Stage A：冻结相关误差模型

输入不需要目标未来 SOH，但至少需要：

- 同电芯、同 landmark 的独立重复测量；
- 每个测试柜/温箱组合不少于 20 个日期、跨度不少于 30 天的日参考序列；
- 不少于 5 个跨测试柜桥接电芯；
- 温箱参考记录、校准记录与维护事件日志。

先从重复测量估计 IID 分量，再从日参考序列估计共同偏置、AR(1) 和漂移，从桥接电芯估计设备间偏差。五类分量、参数、跨通道相关矩阵、设备映射、随机种子规则和源文件哈希必须在 Stage B 前冻结。若时间序列支持不足，实验停止或使用事先登记的最坏情形包络，不能默认退化为 IID。

### Stage B：端到端扰动与重拟合

真实执行为每个新电芯生成 1024 个 Monte Carlo draw。每个 draw 必须完成：

1. 扰动已完成的历史参考电芯测量；
2. 对目标电芯 1-100 周期只扰动一次，P60 与 P100 视图复用相同的前 60 个测量，避免人为制造两个世界；
3. 重建 V5 pairwise 训练矩阵并重新拟合冻结的 48-tree ExtraTrees；
4. 在 P60、P100 分别重新选择 12 个参考电芯；
5. 重新生成 P60/P100 V5 中心；
6. 用周期 61-100 的可见残差重新执行未经修改的 V7 门控。

评估器只接受十列严格 ledger：

| 字段 | 含义 |
|---|---|
| `schema_version` | 固定为 V9 ledger v1 |
| `issuance_id` | 单次签发身份 |
| `cell_id` | 物理电芯身份 |
| `manufacturing_batch_id` | 制造批次 |
| `draw_index` | 0 为无扰动基线，1-1024 为扰动 draw |
| `trajectory_role` | 可见历史、P60 中心或 P100 中心 |
| `cycle_index` | 周期坐标 |
| `retention_pct` | 可见值或模型中心，不是真实未来值 |
| `reference_cell_ids_json` | 当次 V5 选择的参考电芯 |
| `source_sha256` | 当次前缀或模型重拟合来源哈希 |

任何额外列都会被拒绝。目标周期 101-300 的真实容量、SOH、误差或 MAE 没有进入该接口的通道。

### Stage C：一次性开放未来结果

V9 通过只代表输入与管线稳定，不能代替准确率检验。仍需至少 60 个新电芯、3 个制造批次，在所有 ledger、决策、预测和哈希承诺完成后，才可一次性开放目标周期 101-300，并沿用 V8 已冻结的逐电芯与分批次准确率终点。同批次不得改噪声模型或门槛后重跑。

## 4. 冻结判定门槛

一个电芯只有同时满足以下条件才允许非零更新：

| 指标 | 门槛 |
|---|---:|
| 无扰动 V7 门控 | 必须激活 |
| 1024 次重拟合激活概率 | >= 0.95 |
| 修正方向一致概率 | >= 0.95 |
| P60/P100 参考集合 Jaccard 的 P05 | >= 0.80 |
| P60/P100 周期 300 中心偏差 P95 | <= 0.10 pp |
| P100 全轨迹平均绝对偏差 P95 | <= 0.05 pp |
| 最终签发周期 300 偏差 P95 | <= 0.10 pp |
| 最终签发全轨迹平均绝对偏差 P95 | <= 0.05 pp |

任一项失败，修正向量逐元素返回 0，即精确使用无扰动 P100 V5 中心。门槛不使用同一批目标电芯的未来结果选择。

## 5. 消融矩阵

真实执行应同时报告五组预注册消融，但只用完整组合模型决定资格：

1. IID-only 负对照：量化 V8 会低估多少风险；
2. IID + 共同偏置：识别设备校准敏感性；
3. IID + AR(1)：识别连续伪趋势；
4. IID + 漂移：识别长期斜率污染；
5. 五分量完整组合：唯一资格判定场景。

消融不能用于看到未来准确率后挑选“最好看”的噪声场景。

## 6. 已完成的合成软件演练

合成演练使用 16 个完整历史参考电芯和一个只生成到周期 100 的目标前缀。24 个软件 draw 均实际重建训练矩阵、重拟合 V5、重选近邻并重新运行 V7，而不是代理曲线。

| 软件检查项 | 结果 |
|---|---:|
| 目标未来数据行 | 0 |
| replicate ledger 行数 | 12,000 |
| 重拟合激活概率 | 1.000 |
| 修正方向概率 | 1.000 |
| P60/P100 参考集合 Jaccard P05 | 0.8462 / 0.8462 |
| 最终签发全轨迹偏差 P95 | 0.01261 pp |
| 最终签发终点偏差 P95 | 0.01660 pp |
| 人工压力负对照 | 失败并精确回退 0 修正 |

这些结果证明代码路径、严格 schema、端到端重拟合和失败回退可运行，**不证明真实测量噪声很小，也不构成准确率提升**。V5 仍是 champion。

## 7. 运行入口

合成软件演练：

```powershell
$env:PYTHONPATH='src'
python scripts/run_fastcharge_v9_end_to_end_synthetic_dry_run.py `
  --config configs/experiments/v9_end_to_end_correlated_stability_synthetic_dry_run.json `
  --output-directory artifacts/fastcharge-v9-end-to-end-synthetic-dry-run
```

真实 ledger 评估前，必须复制执行模板、填入 Stage A 的哈希绑定参数，并把状态改为 `frozen_for_real_v9_execution_before_target_outcome_access`。分发模板本身会拒绝运行：

```powershell
python scripts/evaluate_fastcharge_v9_replicate_ledger.py `
  --config D:/private-input/v9_execution_frozen.json `
  --replicate-ledger D:/private-input/CELL_001_replicates.csv `
  --output-directory D:/private-output/v9/CELL_001
```

私有原始测量、完整 replicate ledger 和任何海辰结果均应保留在仓库外；GitHub 只放协议、代码和不含真实电芯的合成证据。
