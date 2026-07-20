# LifeTwin 开题报告补充材料

提交人：Jincheng Liu

项目方向：储能 LFP 电池短期数据驱动的长期 SOH 动态预测

## 1. 一分钟概览

储能电站设计寿命可达 15-25 年，而公开 LFP 日历老化数据通常只有数年。
LifeTwin 将问题从“一次性外推一个远期点值”改写为“随数据增长持续修正的寿命
数字孪生”：先学习温度、SOC 等工况下的共性规律，再根据目标对象短期表现
个性化更新；遇到低 SOC 早期容量回升时启用专用偏移模型，证据不足则回退
稳定主模型；最终同时输出 SOH 曲线、合理区间和数据充分性提示。

“目标电芯/单电芯”更新是产品架构目标。当前公开数据没有单电芯轨迹，实际实验对象
只是 target condition-mean trajectories（目标条件均值轨迹）：共 17 条条件均值轨迹，
条件级评估单位 `N=17`，不能表述为完成单电芯验证。

## 2. 为什么值得讨论

- **数据稀缺**：本仓库可公开复现的 Naumann 数据只有 17 条条件均值轨迹，
  最长约 885 天，却对应十年以上的业务决策问题。
- **形状并不统一**：低 SOC 条件可能出现早期容量高于初始基准，单一单调曲线
  会高估后续衰减速度。
- **模型必须持续更新**：研发、质保和电站运维看到的数据窗口不同，静态报告
  无法覆盖全生命周期。
- **置信度不能伪造**：样本不足时，窄区间比没有区间更加危险。

## 3. 方案组成

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 数据治理 | 容量、温度、SOC、倍率、时间 | 统一时间轴与质量标志 | 防止身份、单位和异常数据污染模型 |
| 层次主模型 | 多工况历史 + 目标前缀 | 个性化退化幅度和时间指数 | 用群体经验缓解目标对象数据不足 |
| 激活偏移模型 | 低 SOC 早期异常形状 | 老化项与偏移项 | 避免容量回升造成长期外推偏差 |
| 证据门控 | 异常证据 + 观测数量 | 模型选择与回退原因 | 只在确有证据时增加复杂度 |
| 有界残差 | 训练条件交叉拟合误差 | 小幅曲率修正 | 修正结构误差但不替代机理主模型 |
| 路由化校准 | 条件轨迹级误差与模型路由 | 诊断区间或拒绝原因 | 避免把时间点伪装成独立样本 |
| 风险输出 | 预测分布、证据状态与业务阈值 | SOH、区间、越限时间、提示 | 支持研发、质保和运维决策 |

## 4. 可核验结果

在公开 Naumann 数据的 `p=10` 回顾性开发中：

- 层次幂律相对传统平方根曲线，将未见温度和 SOC 插值场景的平均轨迹误差
  分别降低 57.46% 和 46.31%。
- 保守门控激活模型又相对层次幂律降低 23.72% 和 69.64%。
- 激活门控只触发 3 个唯一条件，严格 bootstrap 优越标准没有通过。
- `tau=3-14 day` 保持平均改善；`tau=20 day` 时仅未见温度场景反转，
  `tau=30 day` 时两个场景都反转，说明方法存在信号也存在明确边界。

全部数字可由仓库中的 CC BY 4.0 数据、冻结配置和测试重新计算，不要求评委
相信一张无法追溯的结果截图。

### 4.1 入围后的三项强化结果

1. **动态 landmark**：把 `p=5/8/10/14` 放到检查点 14-34 的同一未来窗口后，
   只有 `p=10` 同时满足两场景均值改善、逐条件零退化和唯一改善条件要求。由于
   Naumann 结果已被查看，它只叫回顾性信号点，确认 landmark 仍为空。
2. **V4 区间与拒绝**：7/6/4 条件级训练、校准、测试切分下，fallback 的 5 个
   校准条件只够产生 80% 诊断分位数；specialist 只有 1 个校准条件，所有覆盖率
   都拒绝。最终 4 条测试轨迹中 3 条获得回顾性 80% 诊断区间，运营区间发放为 0。
3. **独立外部应力筛查**：在许可明确的 15 个 Geisbauer LFP 电芯上，只用
   0/39/59 天预测 84/120 天。主候选 IAE 为 `3.9735 pp`，目标前缀平方根为
   `3.8852 pp`，候选没有胜出；100% SOC 的迁移问题最明显。该负结果被完整保留。

这三项强化的创新不在“没有偷看未来”，那只是研究正确性的底线；真正的技术增量是
机理层级均值与受约束残差的组合、按实际模型路由进行轨迹级小样本校准，以及把域支持
和长期独立证据直接接入区间发行决策。

## 5. 对抗审计与真实修复

为避免“代码能运行”被误当成“证据可靠”，项目增加了七项对抗检查：数据身份、
单位、重复与缺失；`p=5/8/10/14` 的逐 landmark 未来标签攻击；504 个条件-方法组
独立复算；共同支持与 6 组消融；gate 边界、专用模型故障注入和 V2 回退；fresh-clone
一键复现；以及跨四个 landmark 的 84 行失败条件表。

审计发现了一个真实问题：旧评分入口会验证预测包哈希，却仍可能信任“篡改时间轴或
final 标志后重新哈希”的预测包。现在 V2/V3 都从权威真值连接时间、温度和 SOC，验证
未来索引完整性，用真值时间积分并从真值派生末点；相应攻击测试会被拒绝。

`p=10` 时，V3 相对 V2 的 21 个场景-条件行是 4 行改善、17 行精确回退、0 行相对
退化。0 退化主要是未触发时逐点回退 V2 的结构结果，并非 21 个条件都证明 V3 更优。
4 个改善行只来自 3 个唯一条件；`T40_SOC12.5` 在两个场景重复出现，不能重复计作
两份证据。在这 21 个 scenario-condition occurrences 中，以所有正向 IAE gains
（`V2 IAE - V3 IAE > 0`）之和为分母，该条件贡献约 90.35%，并贡献未见温度场景
正向增益约 80.77%。

跨全部四个 landmark 的 84 行合计为 **72 exact fallback、9 improvement、3 relative
regressions**。三处相对退化（`V3 IAE - V2 IAE`）分别为：`p=8, T25_SOC0`
`+0.52968452368701047 pp`；`p=8, T40_SOC0` `+0.19573480305547392 pp`；
`p=14, T40_SOC0` `+0.048279793130974941 pp`。这说明 `p=10` 的零相对退化不能外推到
其他观测长度。

因此，**audit pass 不等于 model validation pass**。它证明实现和证据护栏经受住已定义
攻击，不证明 15-25 年精度、海辰产品适用性或独立外部有效性。完整说明见
[Phase 1 对抗性审计报告](reports/phase1_adversarial_audit_2026-07-20.md)。

## 6. 补充材料索引

| 材料 | 链接 | 评审价值 |
|---|---|---|
| 相关项目经验 | [docs/project_experience.md](docs/project_experience.md) | 展示从数据治理到模型审计的完整积累 |
| 数据分析样本 | [docs/data_analysis_sample.md](docs/data_analysis_sample.md) | 展示数据、指标、图表和结论链 |
| 可运行分析脚本 | [showcase/analyze_phase8_results.py](showcase/analyze_phase8_results.py) | 一条命令重建核心图表 |
| 对抗性审计报告 | [reports/phase1_adversarial_audit_2026-07-20.md](reports/phase1_adversarial_audit_2026-07-20.md) | 逐项解释攻击、修复、失败条件和证据边界 |
| V0.11 新证据报告 | [reports/landmark_v4_external_evidence_2026-07-20.md](reports/landmark_v4_external_evidence_2026-07-20.md) | 解释动态 landmark、V4 区间拒绝和外部负结果 |
| V0.11 机器可读证据 | [showcase/evidence_v011/README.md](showcase/evidence_v011/README.md) | 直接查看三组冻结结果、预测和条件级指标 |
| 机器可读审计总览 | [showcase/audit_results/phase1_adversarial_audit.json](showcase/audit_results/phase1_adversarial_audit.json) | 汇总审计状态、检查项和禁止宣称 |
| 未来标签攻击矩阵 | [showcase/audit_results/future_label_attack_cases.csv](showcase/audit_results/future_label_attack_cases.csv) | 核对四个 landmark 的预测不变与评分改变 |
| 独立指标复算 | [showcase/audit_results/independent_metric_audit.csv](showcase/audit_results/independent_metric_audit.csv) | 复核 504 个条件-方法指标组 |
| 失败条件表 | [showcase/audit_results/failure_condition_table.csv](showcase/audit_results/failure_condition_table.csv) | 展示 84 个场景-landmark 风险行，而非只展示均值 |
| 一键复现入口 | [scripts/reproduce_public_release.py](scripts/reproduce_public_release.py) | fresh clone 中运行预检、实验、审计、绘图和测试 |
| 研究笔记 | [docs/research_notes.md](docs/research_notes.md) | 展示假设、失败、修正和冻结决策 |
| 参考资料 | [docs/references.md](docs/references.md) | 展示论文、数据和许可核查 |
| 技术报告 | [reports/](reports/) | 展示 Phase 6-8 的详细实验 |
| 发布清单 | [PUBLICATION_MANIFEST.md](PUBLICATION_MANIFEST.md) | 说明上传和主动排除的内容 |

## 7. 建议演示路径

1. 打开首页架构图，说明“共性先验 + 个性更新 + 异常门控”。
2. 在使用 Python 3.12.x、按 `requirements/reproduction.txt` 安装冻结依赖的
   fresh clone 中运行：

   ```powershell
   .\.venv\Scripts\python.exe scripts\reproduce_public_release.py --mode full --output artifacts\reproduction
   ```

   Linux/macOS：

   ```bash
   .venv/bin/python scripts/reproduce_public_release.py --mode full --output artifacts/reproduction
   ```

3. 打开 V0.11 报告，先展示 `p=10` 只是回顾性 landmark，再展示为何大多数区间
   被拒绝发行。
4. 展示 Geisbauer 的外部负结果，说明项目没有为赢指标而重调协议。
5. 打开对抗审计报告和失败条件表，展示未来标签攻击、评分修复和故障回退。
6. 打开研究笔记中的 `T40_SOC12.5` 失效链，解释为什么模型会演进。
7. 最后展示证据边界：公开数据证明研究路径可行，但产品承诺必须等待独立长期和内部数据。

GitHub Actions 已配置 Ubuntu/Windows clean checkout；运行记录见
[public-release-ci](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/workflows/ci.yml)，
以对应提交的实际状态为准。

## 8. 当前边界

本项目没有海辰数据，也没有证明 15-25 年预测精度。当前最强结论是：一个低
参数、可解释、可滚动更新的混合模型在公开数据上表现出清晰的可行性信号，且
已经具备数据接入、验证、回退、受约束残差、区间拒绝和审计框架。独立的 120 天
高温队列暴露了 100% SOC 迁移风险，不能代替长期验证。下一步应冻结现有规则，
在许可明确的独立长期 LFP 队列和海辰目标产品数据上一次性验证，而不是继续对
同一公开数据调参。
