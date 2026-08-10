# FastCharge V7 重发感知创新状态开发报告

## 结论先行

V7 找到了比 V6.1 更清晰、风险更低的 P100 候选机制，但 **V5 仍是当前正式模型**。

在 41 个训练电芯的外层留一电芯审计中，V7 在 P100 仅激活 `9/41` 个电芯，
`9/9` 均改善；全体 P100 电芯的平均轨迹 MAE 从 `0.2435991 pp` 降至
`0.2062829 pp`，绝对下降 `0.0373162 pp`，相对下降约 `15.32%`。两个 MATR
批次互相留出的压力测试也都通过。但是，这仍然是同一批 41 个训练电芯上的
outcome-informed 方法开发，不是独立确认。因此项目只冻结 P100 规则用于下一批
结果盲测，不激活 V7，也没有查看 81 个已暴露评估电芯。

## 1. V6.1 的剩余问题

V6.1 用上一次签发后已观测到的残差斜率修正当前 V5 轨迹。问题在于：到达新的
landmark 后，V5 本身会根据新增观测重新签发整条未来轨迹，其中可能已经吸收了
一部分旧残差趋势。如果仍把完整旧斜率再次投影，就会发生“重复修正”。P100 的
V6.1 结果正好暴露了这个风险：10 次激活中有 2 次退化，最差退化 `0.04747 pp`。

V7 将更新状态改为：

```text
历史残差斜率 h = Theil-Sen(P60 预测残差，cycles 61..100)
模型重发斜率 r = Theil-Sen(P100 V5 中心 - P60 V5 中心，cycles 101..300)
未吸收创新量 u = clip(h - r, -0.02, 0.02) pp/cycle
修正量 = clip(scale * u * (future_cycle - 100), -1.0, 1.0) pp
```

这里的 `r` 只由两次已经生成的模型预测计算，不读取未来真实容量。这个结构与
状态空间模型中的 innovation 思想一致：先计算新签发已经解释了多少变化，再只对
未解释部分做小幅更新。电池预测研究也普遍支持动态状态、递归更新和不确定性建模，
例如 [Gaussian-process transition model](https://doi.org/10.1016/j.est.2019.03.022)、
[Enhanced GP dynamical model](https://doi.org/10.1016/j.rser.2024.115045) 和
[hybrid physics/data-driven trajectory prediction](https://doi.org/10.1109/TTE.2022.3212024)。
V7 的具体“重发扣除 + 风险门控”组合是本项目针对多次 landmark 签发流程形成的
工程假设，不能因引用这些论文而被视为已得到外部证明。

## 2. 冻结实验设计

| 项目 | 冻结规则 |
|---|---|
| 开发数据 | 41 个 MATR FastCharge 训练电芯的 V5 cross-fit predictions |
| 禁止数据 | 81 个已暴露公开评估电芯、海辰私有数据、未来真实容量 |
| 外层审计 | leave-one-physical-cell-out，每次选择排除目标电芯 |
| 迁移压力测试 | MATR Batch 1 与 Batch 2 双向 leave-one-batch-out |
| 候选规模 | 3 个投影尺度 × 3 个创新幅度阈值 × 2 个单调性阈值 + 回退，共 19 个 |
| 选择次序 | 最坏激活退化、active p90、平均收益、覆盖率、规则 ID |
| 失败动作 | 修正严格为 0，完全回退当前 V5 中心 |
| 评分区间 | 当前 prefix 后至 cycle 300，按物理电芯等权 |

风险优先排序很重要。它阻止大尺度规则仅凭少数大收益赢得平均分，同时让 P100 在
所有激活样本均改善时选择更有价值的尺度。所有范围与门槛均明确标记为使用同一批
41 个训练电芯形成的开发结果。

## 3. 嵌套留一电芯结果

| 当前 prefix | 激活数 | 激活精度 | 全体 mean delta (pp) | active p90 / max (pp) | cell gate | batch gate | 决策 |
|---:|---:|---:|---:|---:|---|---|---|
| 40 | 14/41 | 71.43% | -0.0079200 | +0.00739 / +0.01355 | 通过 | 失败 | 淘汰 |
| 60 | 2/41 | 0.00% | +0.0007175 | +0.01984 / +0.02112 | 失败 | 失败 | 淘汰 |
| 100 | 9/41 | 100.00% | -0.0373162 | -0.02834 / -0.02410 | 通过 | 通过 | 冻结为盲测候选 |

![V7 重发感知创新状态训练内证据](../docs/assets/v7_reissue_innovation_results.png)

P40 是一个重要的反例：逐电芯嵌套指标看起来合格，但其规则不能跨 Batch 1/2 稳定
迁移。项目没有因为均值改善就保留它，而是按冻结协议淘汰。P60 则直接在外层审计中
退化。最终只有 P100 同时满足逐电芯和批次压力门槛。

P100 的外层 41 折全部选择同一门控 `a0p5_d0p01_r0p0`，modal stability 为
`100%`。冻结规则要求：历史全段与最近 10 点斜率同号；未吸收创新量与历史斜率
同号；`abs(u) * 40 >= 0.01 pp`；满足后用 `scale=0.5` 投影，否则完全回退 V5。

## 4. 批次迁移压力测试

| 用于选择的批次 | 留出批次 | P100 激活 | 激活精度 | 全体 mean delta (pp) | 最差 active delta (pp) |
|---|---|---:|---:|---:|---:|
| Batch 2 | Batch 1 | 1/20 | 100% | -0.0035406 | -0.0708126 |
| Batch 1 | Batch 2 | 6/21 | 100% | -0.0669355 | -0.0349287 |

这仍是同一个 41 电芯训练队列内的批次压力测试，不应写成“外部验证”。它的价值是
及时否决 P40/P60，并减少留一电芯结果被同批次相似性夸大的风险。

## 5. 同激活样本消融

在 P100 的同 9 个激活电芯、同一 `scale=0.5` 下，仅将 V7 的 `h-r` 换回 V6 的
原始历史斜率 `h`：

| 修正状态 | 改善电芯 | active mean delta (pp) | 最差 active delta (pp) |
|---|---:|---:|---:|
| V6 式原始历史斜率 | 8/9 | -0.1695621 | +0.0039075 |
| V7 未吸收创新量 | 9/9 | -0.1699959 | -0.0240988 |

两者平均收益接近，但 V7 消除了这一组中的唯一退化案例。这支持“扣除已吸收变化”
主要是在改善尾部安全性，而不是靠扩大总体修正量刷平均分。该消融仍与候选开发使用
同一批数据，因此只是机制证据。

## 6. 下一批结果盲测

机器可读规则已冻结在
`configs/experiments/v7_p100_reissue_innovation_blind_candidate.json`。至少需要 40 个
新物理电芯，并在打开 cycles 101..300 真值前完成以下动作：

1. 用冻结 V5 生成并保存 P60、P100 两次中心轨迹；
2. 计算并保存 activation flag、V7 预测及全部文件哈希；
3. 禁止在同一批数据上改尺度、阈值或筛选样本；
4. 打开真值后一次性计算激活数、覆盖率、精度、全体 mean delta、active p90 和最大退化；
5. 任一端点失败，保留 V5 并记录盲测负结果。

即使单批循环老化盲测通过，也只能说明 FastCharge 循环老化域得到一次独立确认；它
仍不能证明日历老化、15 至 25 年外推或海辰产品精度。

## 7. 可复核资产

- 实现：`src/lifetwin/experiments/fastcharge_v7_reissue_innovation.py`
- runner：`scripts/run_fastcharge_v7_reissue_innovation_development.py`
- 开发协议：`configs/experiments/v7_reissue_innovation_development.json`
- 盲测冻结协议：`configs/experiments/v7_p100_reissue_innovation_blind_candidate.json`
- 测试：`tests/test_fastcharge_v7_reissue_innovation.py`
- 小型派生证据：`showcase/evidence_v7/`

复现命令：

```powershell
python scripts\run_fastcharge_v7_reissue_innovation_development.py
python -m pytest -q tests\test_fastcharge_v7_reissue_innovation.py
```

FastCharge 数据来源与经典基准论文见
[Severson et al., Nature Energy 2019](https://doi.org/10.1038/s41560-019-0356-8)。
