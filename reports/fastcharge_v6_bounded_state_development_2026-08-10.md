# FastCharge V6 有界状态更新与 V6.1 选择性门控开发

## 结论先行

V6 没有替换 V5。对 41 个训练电芯的交叉拟合预测进行外层留一电芯审计后，
无门控的残差趋势更新虽然把三段 transition 合计平均误差降低了
`0.00787 pp`，但只改善 `47.97%` 的 cell-transition，且 P40 更新发生
退化。因此 V5 继续作为 champion。

V6.1 进一步把更新改为“有证据才激活”。在 P100，嵌套留一审计中有
`10/41` 个电芯触发更新，其中 `8/10` 改善；全体 P100 电芯的平均轨迹
MAE 从 `0.24360 pp` 降至 `0.22397 pp`，相对降低约 `8.06%`。但这仍是
同一批 41 个训练电芯上的 outcome-informed 方法开发，所以只冻结为下一批
盲测候选，不在当前模型中激活，也没有再次查看 81 个已暴露评估电芯。

## 1. 为什么做这轮实验

V5 动态 landmark 审计证明了“较长前缀重新签发”平均有效，但固定 offset
和 GP residual 没有稳定通过 70% 电芯改善门槛。进一步检查训练交叉拟合误差
后发现：历史残差的整体水平很弱，残差的缓慢斜率在 P60/P100 后更有信息。
因此新假设不是再叠加一个黑盒，而是估计一个受限的局部趋势状态：

1. 只读取上次签发后、当前 landmark 当时已经观测到的残差；
2. 用 Theil-Sen 或带截断 innovation 的 alpha-beta filter 估计残差斜率；
3. 未来修正随 horizon 线性投影，但绝对值不超过 `1.0 pp`；
4. 证据不足时修正严格为零，回到当前前缀 V5 中心。

## 2. 防泄漏设计

| 项目 | 冻结规则 |
|---|---|
| 开发数据 | 仅 41 个训练电芯的 V5 cross-fit predictions |
| 外层单位 | leave-one-physical-cell-out |
| 可用历史 | `previous_prefix < cycle <= current_prefix` |
| 禁止输入 | 当前 landmark 后的真实容量、实际未来工况、81 电芯评估 outcome |
| 共同评分区间 | 当前 prefix 后至 cycle 300 |
| champion | 冻结 V5，不修改其参数或已发布结果 |

候选选择和门控阈值都只能在每个外层 held-out 电芯之外完成。held-out 电芯的
未来结果只用于最后评分，不能决定它是否激活。

## 3. V6 无门控状态更新

候选库包含 1 个 no-update、6 个稳健局部趋势和 4 个有界 alpha-beta 状态
规则。训练候选提名门槛比正式晋级门槛宽松；正式晋级仍要求嵌套审计中至少
70% cell-transition 改善且每个 transition 不退化。

| 当前 prefix | V5 MAE (pp) | 更新后 MAE (pp) | delta (pp) | 改善电芯 |
|---:|---:|---:|---:|---:|
| 40 | 0.35218 | 0.36160 | +0.00941 | 39.02% |
| 60 | 0.30548 | 0.30512 | -0.00036 | 51.22% |
| 100 | 0.24360 | 0.21093 | -0.03266 | 53.66% |

三段合计平均 delta 为 `-0.00787 pp`，但改善比例仅 `47.97%`。P100 有
值得保留的均值信号，P40 的留一结果则揭示了小样本选择不稳定。正式晋级门槛
失败，决策为 `retain_frozen_v5_champion`。

## 4. V6.1 选择性门控

V6.1 不再要求所有电芯接受更新，而是同时约束激活覆盖率、激活精度、全体
平均收益和激活尾部风险。候选门控只读取三个已经可见的历史量：

- 全历史与最近 10 点 Theil-Sen 斜率是否同号；
- 历史斜率在上一段 landmark 上对应的绝对变化量；
- 历史残差的绝对 Spearman 单调性。

嵌套选择结果如下：

| 当前 prefix | 激活数 | 激活覆盖率 | 激活精度 | 全体 mean delta (pp) | active p90 / max (pp) | 决策 |
|---:|---:|---:|---:|---:|---:|---|
| 40 | 0/41 | 0.00% | 不适用 | 0.00000 | 不适用 | 回退 V5 |
| 60 | 1/41 | 2.44% | 0.00% | +0.00092 | 0.03767 / 0.03767 | 淘汰 |
| 100 | 10/41 | 24.39% | 80.00% | -0.01963 | 0.03202 / 0.04747 | 冻结为盲测候选 |

P100 外层选择在 `40/41` 个 fold 中选择同一门控，modal stability 为
`97.56%`。冻结候选的精确规则是：

```text
history = P60 issuance residuals observed at cycles 61..100
s_all   = Theil-Sen slope(history)
s_10    = Theil-Sen slope(last 10 history points)
activate if sign(s_all) == sign(s_10) and abs(s_all) * 40 >= 0.04 pp
correction(cycle) = clip(0.25 * s_all * (cycle - 100), -1.0, 1.0) pp
otherwise correction = 0 and use the P100 V5 center exactly
```

这个结果说明“异常专用模型”有一个可行方向：不是先判断电芯属于某种异常
类型，而是要求目标电芯自身已经出现方向一致、幅度足够的残差信号，再允许
小幅更新。它比全量更新更符合 LifeTwin 的保守签发逻辑。

## 5. 为什么现在仍不能说模型升级了

1. 门控阈值由这 41 个训练电芯启发，嵌套留一能减少乐观偏差，但不能把同一
   批数据变成独立确认。
2. P100 只有 10 次激活，`8/10` 仍是小样本信号。
3. 81 个公开评估电芯已经被项目查看，本轮主动不再使用，避免第二轮调参污染。
4. 该实验是快速充电循环老化，不是日历老化，更不是 15 至 25 年储能电站证据。

因此当前线上/演示默认仍是 V5。V6.1 只是一个已经写成机器可读规则、等待新
结果盲测的 challenger。

## 6. 下一批盲测

盲测协议已经冻结为
`configs/experiments/v6_1_p100_gated_state_blind_candidate.json`。至少需要
40 个新物理电芯，在打开 cycles 101-300 outcome 前提交预测与 activation
flag 哈希。所有端点必须同时通过：至少 6 次激活、覆盖率至少 15%、激活精度
至少 70%、全体 mean delta 小于 0、active p90 不超过 `0.05 pp`、单电芯
最大退化不超过 `0.1 pp`。失败就保留 V5 并公开记录负结果，不在同批数据上
改阈值重跑。

## 7. 可复核资产

- V6 实现：`src/lifetwin/experiments/fastcharge_v6_bounded_state.py`
- V6 runner：`scripts/run_fastcharge_v6_bounded_state_development.py`
- V6.1 runner：`scripts/run_fastcharge_v6_1_gated_state_development.py`
- V6 协议：`configs/experiments/v6_bounded_state_update_development.json`
- V6.1 协议：`configs/experiments/v6_1_gated_state_update_development.json`
- 未来盲测冻结规则：`configs/experiments/v6_1_p100_gated_state_blind_candidate.json`
- 防回退测试：`tests/test_fastcharge_v6_bounded_state.py`
- 小型证据包：`showcase/evidence_v6/`

复现命令：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_fastcharge_v6_bounded_state_development.py
.\.venv\Scripts\python.exe scripts\run_fastcharge_v6_1_gated_state_development.py
```
