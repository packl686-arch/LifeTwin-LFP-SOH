# FastCharge V9 端到端相关稳定性软件演练

## 结论

V9 合成软件演练通过。它补齐了 V8 的两个实现缺口：测量扰动现在会穿过完整 V5 重拟合与近邻重选链路，且噪声同时包含共同偏置、AR(1)、漂移与尖峰，而不再只有 IID 重复性误差。

该结论严格属于软件证据。目标合成电芯只生成到周期 100，未来结果行数为 0；演练没有创建模型准确率证据，没有恢复 V7 的盲测资格，也没有改变 V5 champion。

## 实现核查

- 冻结 V5：`pairwise_extra_trees_leaf3_48`、12 近邻、weighted mean；
- 冻结 transition：P60 -> P100，预测周期 101-300；
- 每个 draw 重建 pairwise 矩阵并重新拟合两次 V5；
- 相同目标扰动在 P60/P100 之间保持嵌套一致；
- ledger 只含可见历史、模型中心、参考身份与哈希；
- 任一稳定性门槛失败时输出逐元素精确 0 修正。

## 合成结果

| 指标 | 结果 |
|---|---:|
| 历史参考电芯 | 16 |
| Monte Carlo draw | 24 |
| ledger 行 | 12,000 |
| 基线 V7 终点修正 | -0.38541 pp |
| 重拟合激活概率 | 1.000 |
| 修正方向概率 | 1.000 |
| P60 中心终点偏差 P95 | 0.01901 pp |
| P100 中心终点偏差 P95 | 0.01393 pp |
| P100 中心轨迹偏差 P95 | 0.01079 pp |
| 最终签发终点偏差 P95 | 0.01660 pp |
| 最终签发轨迹偏差 P95 | 0.01261 pp |
| P60/P100 近邻 Jaccard P05 | 0.84615 / 0.84615 |

人工软件压力对照同时破坏近邻集合和中心轨迹，七项稳定门槛失败，评估器返回精确 0 修正，证明失败路径不会悄悄发出不稳定更新。

## 尚未完成

1. 尚无真实测试柜/温箱的相关误差参数；
2. 尚无 60 个新电芯、3 个批次的 V9 承诺队列；
3. 真实执行应使用 1024 draw，而非软件演练的 24 draw；
4. 尚未开放任何新队列的周期 101-300 结果，因此不能声称准确率提升；
5. 循环老化稳定性不能替代储能日历老化或 15-25 年验证。

## 证据入口

- [盲测协议模板](../configs/experiments/v9_end_to_end_correlated_stability_blind_protocol.template.json)
- [真实执行配置模板](../configs/experiments/v9_end_to_end_correlated_stability_execution.template.json)
- [执行说明](../docs/v9_end_to_end_correlated_stability_experiment_cn.md)
- [合成证据](../showcase/evidence_v9_dry_run/README.md)
