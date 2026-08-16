# LifeTwin 评委三分钟简报

LifeTwin 解决的不是“用两年数据猜一个 25 年数字”，而是：**随着目标对象数据增长，
持续更新 SOH 轨迹；无法证明可靠时，主动回退或拒绝签发。** 当前版本是公开数据研究
原型，不含海辰数据，也不是产品精度承诺。

## 立即体验

- [打开评委可视化 Demo](https://packl686-arch.github.io/LifeTwin-LFP-SOH/demo/)：
  六个视图快速了解定位、工作台、动态更新、双模型比较、价值与证据边界。
  其中交互数据仅用于界面演示，不冒充正式模型实验结果。
- [打开在线评审控制台](https://packl686-arch.github.io/LifeTwin-LFP-SOH/judge-console/)：
  零安装回放三个冻结案例，查看轨迹、模型路由、诊断区间和拒绝原因；
  [离线单文件](docs/judge-console/index.html)也随仓库冻结。它是静态证据回放，不伪装成实时推理。
- [查看真实前缀预测请求](showcase/product_demo/README.md)：输入只含前 10 次容量观测，
  未来部分只允许时间坐标，输出 `forecast.csv` 和可审计的 `decision.json`。

```powershell
# 按 README 安装依赖后，在源码仓库根目录运行
python -m lifetwin.cli calendar-prefix-predict `
  --request showcase/product_demo/naumann_t40_soc37_5_request.json `
  --output-dir artifacts/product-demo
```

## 最新冻结结论

2026-08-16，独立预注册的 V3.0 运行时可靠性研究完成唯一获授权正式尝试
`v300-formal-20260815-a1`，终态为 **`success`**：7/7 正常作业完成，固定 8-case
故障矩阵通过，10/10 联合成功门通过，独立复算结论一致。完整证据见
[正式收口报告](reports/runtime_reliability_v3_0_formal_closeout_20260816.md)和
[机器可读结果](reports/runtime_reliability_v3_0_formal_result_20260816.json)。

这个成功只回答“冻结的混合合成结构拟合工作负载能否在声明的 Windows 环境中确定性、
有界且诊断透明地完成”。它没有接触密封真值，没有评估电池预测准确度，也不改写
V0.14/V1 的科学失败或 V2.10 的预测前异常终态。

## 方法闭环

`10 点短期前缀 → 严格未来标签防火墙 → 跨温度/SOC 层次先验更新 → 异常形状门控
→ 通用幂律或激活专用路由 → 回顾诊断区间或明确拒绝 → 全链路 SHA-256`

- 正常形状走稳定的层次幂律；低 SOC 早期容量回升有足够证据时才启用专用模型。
- 残差只能从训练条件的交叉拟合误差学习，并受时间支持和幅度上限约束。
- 校准按模型路由和“条件轨迹”计数，不把同一轨迹的多个时间点冒充独立样本。
- 域外、前缀超出参考支持、超出残差支持或同路由校准不足时，诊断区间失败关闭；
  校准非独立或缺少长期确认时，运营签发继续失败关闭。

## 四个关键结果

| 可核验事实 | 结果 | 正确解读 |
|---|---:|---|
| Naumann `p=10` V4 四条测试条件平均轨迹 IAE | `0.2260 pp` | 回顾性开发信号，不是独立验证 |
| 210 个重叠校准切分中的 fallback 80% 乘数 | `0.9243–2.1698` | 区间对切分敏感；specialist 区间仍不可用 |
| Geisbauer 15 电芯外部压力筛查的平均配对差 | `+0.0882 pp` | 候选平均更差，负结果被完整保留 |
| 当前运营区间签发数 / 合格独立长期公开队列数 | `0 / 0` | 证据不足就拒绝，不包装成 15–25 年能力 |

## 入围后的关键压力测试

V0.14 在查看真值前冻结代码、种子、终点和门槛，对 2,900 条合成 25 年轨迹执行
结构可辨识性测试。四项安全门全部通过，说明结果可评分且协议未偏离；但预注册
结论是 **failure**：50% 签发率下灾难性风险降低 21.65%，未达到 30% 门槛；
200 组“短期前缀完全相同、长期结局不同”的反例中，仅 27% 两侧同时拒绝，未达到
80% 门槛；已签发轨迹的 IAE 非劣性通过（相对平方根基线 +0.0146 pp，门槛
+0.10 pp）。

这不是 25 年真实电芯验证，也不是产品失败。它严格否定的是当前“最大结构包络
直接排序风险”的准备度，并暴露晚期 knee 风险在短期前缀下不可辨识。完整结果见
[V0.14 报告](reports/synthetic_long_horizon_identifiability_result_v1_2026-07-22.md)和
[机器可读证据](showcase/evidence_v014/README.md)。

## 三个预置案例

1. `T40 · SOC37.5%`：自动回退通用模型；80% 回顾诊断区间可算，运营仍拒签。
2. `T40 · SOC12.5%`：触发激活专用模型；同路由校准不足，诊断区间也拒绝。
3. `Geisbauer 60°C`：展示真实外部负迁移和方向不稳定，不只展示最好案例。

## 当前边界与下一步

公开 Naumann 证据是 17 条条件均值轨迹、最长约 885 天；Geisbauer 仅是 120 天
高温压力筛查。它们不能支持个体电芯、储能电站、海辰产品或 15–25 年准确度结论。
合成长时域结构可辨识性 V1 已按冻结协议运行并失败，因而不会晋升当前门控方法。
V2.10 唯一正式尝试在 prediction commitment 前异常终止且未评分；后续因果充分性门失败，
V2.11 永久关闭。新立项的 V3.0 已证明运行时可靠性，但没有重做科学评分。真正的模型确认
仍等待许可明确、未被查看过结果的独立长期 LFP 队列，并需要新的问题定义与独立预注册。

完整证据从 [V3.0 正式收口报告](reports/runtime_reliability_v3_0_formal_closeout_20260816.md)、
[V2.10 因果充分性终结审计](reports/synthetic_long_horizon_identifiability_v2_10_causal_sufficiency_closeout_20260815.md)、
[V0.14 长时域压力测试报告](reports/synthetic_long_horizon_identifiability_result_v1_2026-07-22.md)、
[开题报告补充材料](SUBMISSION_SUPPLEMENT.md)、
[V0.12 稳健性报告](reports/robustness_and_long_term_protocol_2026-07-21.md) 和
[发布复现入口](scripts/reproduce_public_release.py) 进入。
