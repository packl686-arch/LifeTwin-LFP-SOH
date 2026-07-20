# Phase 1 对抗性审计报告

日期：2026-07-20

审计对象：LifeTwin 公开 Naumann 日历老化复现实验

审计状态：实现与防护检查通过；模型验证状态仍为 `not_confirmed`

## 先说结论

这次审计回答的是“公开结果是否由身份明确的数据、无未来泄漏的预测过程和可复算的
评分产生”，不是“模型是否已经能够预测 15-25 年”。审计覆盖数据、切分、预测、评分、
基线、门控、故障回退、复现入口与 CI 配置，并在评分器中发现并修复了一个真实完整性问题。

> **审计通过不等于模型验证通过。** 它说明当前回顾性实现经受住了规定的攻击；不证明
> 激活机理、海辰产品精度、储能电站适用性或 15-25 年外推精度。

| 审计项 | 做了什么 | 当前结论 |
|---|---|---|
| 1. 数据身份与单位 | 核对来源、哈希、单位、重复、缺失和统计单位 | 通过 |
| 2. 未来标签防火墙 | 在 `p=5/8/10/14` 分别篡改该时点及之后的容量标签 | 预测不变、评分改变，通过 |
| 3. 独立指标复算 | 不调用正式评分结果，独立复算 504 个条件-方法组 | 全部一致 |
| 4. 基线公平与消融 | 检查共同支持并完成 6 组比较、两场景共 12 行 | 支持一致，结论仅为开发诊断 |
| 5. 门控与故障回退 | 测边界、非法输入，并注入专用模型拟合失败 | 回退到 V2，流程不中断 |
| 6. 全新环境复现 | 单入口执行预检、实验、审计、绘图和测试 | 本地入口已建立；CI 链接待最终运行 |
| 7. 失败条件清单 | 四个 landmark 逐场景列出风险、回退原因和建议 | 84 行，全部禁止部署性宣称 |

## 1. 数据身份、单位、重复与缺失

规范化表共有 595 行，但它们是 17 条条件均值轨迹的重复检查点。原论文每个条件聚合
3 个物理电芯，因此可追溯到 51 个物理电芯；公开表没有保留 51 条单电芯轨迹，不能
把 51 或 595 当作样本量。**当前公开实验只有 17 条条件均值轨迹，条件级评估单位
`N=17`；没有单电芯级轨迹可供验证。**“目标电芯/单电芯”更新是产品架构目标，当前代码
在公开实验中更新的是 target condition-mean trajectory（目标条件均值轨迹）。

审计逐项检查：

- 数据集 ID、DOI、来源 URL、CC BY 4.0 许可和规范化 CSV 的 SHA-256；
- 17 个条件、每条件 35 个检查点以及公开温度-SOC 网格和时间轴；
- 缺失值、整行重复、`condition_id + checkup_index` 重复；
- 秒、小时、天三套时间单位的换算；
- `capacity_loss = 100 - capacity_retention`，以及保持率与 Ah 的一致性；
- 直流内阻增长率与欧姆值、10 秒脉冲、测试 SOC 百分数单位和 3 Ah 标称容量；
- `condition_id`、逻辑 `cell_id`、测试 ID、来源 ID 和条件均值语义的一一对应。

结果为 0 个缺失、0 个整行重复、0 个条件-检查点重复；所有单位与派生量误差都在数值
舍入范围内。逐条件结果见
[data_condition_audit.csv](../showcase/audit_results/data_condition_audit.csv)。

## 2. 每个 landmark 的未来标签攻击

“代码里写了不看未来”不是证据，所以审计直接修改未来答案：对每个冻结 landmark
`p=5、8、10、14` 单独生成攻击数据，把 `checkup_index >= p` 的容量保持率整体下移
4 个百分点，并同步修改容量 Ah 和容量损失，使数据内部仍然自洽。随后从头重跑实验。

每次攻击都要求同时满足：

1. 对应 landmark 的预测表和无标签预测哈希完全相同；
2. 拟合参数、诊断、训练/目标切分完全相同；
3. `p=10` 的 tau 敏感性预测也完全相同；
4. 评分必须发生变化，否则说明攻击没有真正触及未来答案。

四次攻击的最大预测变化为 `0 pp`，而评分发生变化。审计还改变同一折内一个目标条件的
已观测前缀：该目标自己的更新可以响应，但折先验和同折其他目标预测不得变化。这同时
检查了“目标前缀可用于自身更新”和“目标前缀不能污染其他目标”两条边界。
四个 landmark 的逐次哈希、变化量和判定见
[future_label_attack_cases.csv](../showcase/audit_results/future_label_attack_cases.csv)。

## 3. 504 组独立复算与评分器真实缺陷

正式指标是未来区间内的时间加权轨迹绝对误差：

```text
IAE = trapezoid(abs(prediction - truth), truth_elapsed_days)
      / (last_truth_day - first_truth_day)
```

审计从预测包和原始真值重新连接数据，独立计算未来点数、点 MAE、轨迹 IAE、末点真值、
末点预测和末点误差。总计为 `21 个场景-条件出现 × 4 个 landmark × 6 个方法 = 504`
组。504 组逐项差值均为 0，聚合均值最大差异约 `4.44e-16`。这里的 504 是重复评估组，
不是 504 个样本；条件级评估单位仍为 `N=17`。明细见
[independent_metric_audit.csv](../showcase/audit_results/independent_metric_audit.csv)。

### 审计发现并修复的真实问题

原评分入口先检查预测包哈希，但攻击者若同时篡改 `elapsed_days` 或 `is_final_checkup`，
再重新计算哈希，仍可能让评分器信任伪造的积分时间轴或末点标志。哈希只能证明“收到的
文件没有再变”，不能证明文件中的坐标是真实坐标。

V2 和 V3 评分器现已改为：

- 从权威真值表连接真实 `elapsed_days`、温度和 SOC，并与预测坐标逐项核对；
- 检查 `target_checkup_index >= prefix`、前缀末索引/日期，以及每条未来轨迹恰好为
  `range(prefix, 35)`；
- 按冻结协议检查每个场景、fold 和 landmark 的目标条件全集，并要求每个未来坐标
  同时包含全部六个方法；tau 敏感性包同样必须覆盖完整目标和冻结 tau 网格；
- 拒绝缺失、非有限值和不一致的验证 horizon；
- 用真值时间轴积分，并由真值最大检查点派生末点；预测包的 final 标志只用于一致性检查。

重新哈希后的时间、温度、SOC、final、prefix、缺失点和非有限值攻击均被拒绝。相关测试见
[test_prediction_scoring_integrity.py](../tests/test_prediction_scoring_integrity.py)。这项修复说明
对抗审计不只是补充说明文字，而是实际找到了并关闭了结果完整性漏洞。

## 4. 基线公平性与 6 组消融

六个方法在每个场景使用相同目标条件、fold、landmark 和未来坐标；预测包不含未来容量
或误差列。门控未触发时，两个门控候选与冻结 V2 逐点完全相同；触发且专用模型成功时，
它们与相应专用分支逐点相同。

公平性也有边界：传统平方根基线只使用目标前缀，而 V2 使用跨条件层次信息，因此
“平方根到 V2”同时改变了信息来源和指数自由度，不是单因素因果消融。`tau=7 day` 是
查看 Phase 7 失败后确定的 post-hoc 值，所有比较都只能作为回顾性开发证据。
敏感性结果也并非“20-30 天均同时失败”：`tau=20 day` 时仅未见温度场景反转，
`tau=30 day` 时未见温度和 SOC 插值两个场景都反转。

| 消融 | 隔离因素 | 是否单因素 | 主要用途 |
|---|---|:---:|---|
| 平方根 → V2 | 层次信息 + 自由指数 | 否 | 对比传统经验曲线 |
| V2 → 无门控层次激活 | 激活项 | 是 | 检查同一层次专家族内的激活项 |
| 无门控目标激活 → 门控目标激活 | 保守门控 | 是 | 量化精度与安全回退的代价 |
| 无门控层次激活 → 门控层次激活 | 保守门控 | 是 | 检查门控是否依赖专家类型 |
| 门控层次激活 → 门控目标激活 | 专用分支 | 是 | 比较同一门控下的专家选择 |
| V2 → 门控目标激活 | 组合候选 | 否 | 对应主候选的整体效果 |

两场景共 12 行结果见
[ablation_audit.csv](../showcase/audit_results/ablation_audit.csv)。无门控目标激活虽然均值更低，
但在 SOC 场景有 1 个、未见温度场景有 5 个条件相对 V2 变差；这正是保守门控存在的理由。

## 5. Gate 边界与执行回退故障注入

门控要求“正时间观测数达到 7”且“最小容量损失严格小于负损失阈值”。六个边界用例
覆盖 6/7 个观测、无负损失、恰好等于阈值、刚越过阈值和更多观测。等于阈值时不触发，
只有严格越界才触发。非法最小观测数、负阈值、多条件输入和没有正时间点均被拒绝。
边界表见 [gate_boundary_cases.csv](../showcase/audit_results/gate_boundary_cases.csv)。

边界正确还不够，数值优化器可能在实际运行中失败，因此增加了异常注入：

- gate 未就绪时强制目标专用拟合抛错，主预测仍逐点回退 V2，并记录错误与回退原因；
- gate 已就绪时同时让目标、层次和 tau 专用拟合失败，所有相关分支仍回退 V2，
  `activation_component_selected=false`，敏感性表仍完整产生，流程不会因专用模型失败中断。

故障注入测试见
[test_calendar_v3_gate_fallback.py](../tests/test_calendar_v3_gate_fallback.py)。这保证的是执行连续性和
可追溯回退，不保证回退后的 V2 在绝对意义上足够准确。

## 6. Fresh Clone 一键复现

在全新 clone 中使用 Python 3.12.x，并按
`requirements/reproduction.txt` 安装冻结依赖后，统一入口是：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_public_release.py --mode full --output artifacts\reproduction
```

Linux/macOS：

```bash
.venv/bin/python scripts/reproduce_public_release.py --mode full --output artifacts/reproduction
```

该命令先检查 Git 跟踪状态、发布清单、冻结文件哈希，并确认依赖版本与冻结约束一致，
再原子化运行 Phase 8、
无界面绘图、Phase 1 对抗审计和完整 pytest；任一步失败都不发布半成品目录。成功后生成
`artifacts/reproduction/reproduction_summary.json`、命令日志和带 SHA-256 的审计产物清单。
输出目录必须事先不存在，以免覆盖旧证据。

规范复现约束精确数值栈，清单所列的已发布证据文件必须逐字节匹配 SHA-256；Ubuntu/Windows
重算表允许 `1e-8` 数值容差。`training_state_sha256` 等派生状态哈希可能随底层数值库改变，
因此要求格式有效、行间等价类结构一致，并在同一次 baseline/attacked 攻击内成对相等；
不把跨操作系统哈希相等误写成模型结论。

GitHub Actions 已配置 Ubuntu 与 Windows 的 clean checkout 矩阵。运行记录见
[public-release-ci](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/workflows/ci.yml)，
以本报告对应提交的实际状态为准。

## 7. 84 行失败条件表

[failure_condition_table.csv](../showcase/audit_results/failure_condition_table.csv) 不是“成功案例表”，
而是主动回答模型何时不可信。它覆盖 4 个 landmark，每个 landmark 有 17 个未见温度场景
出现和 4 个 SOC 插值场景出现，共 `4 × 21 = 84` 行。表中包括：

- 温度、SOC、统计单位、前缀与预测 horizon；
- V3、V2、无门控目标激活和门控层次激活的误差及差值；
- 选中分支、门控证据、回退原因、末点误差和候选误差排名；
- 跨场景重复标志、风险标签、可信状态和建议动作；
- `deployment_trusted=false` 与 `claim_allowed=false`。

主分析 `p=10` 的 21 个场景-条件行中，V3 相对 V2 为 **4 行改善、17 行精确回退、
0 行相对退化**。这个“0 退化”主要由门控未触发时逐点复制 V2 的结构保证，不能解释为
V3 在 21 个条件上都获得了验证。4 个改善行只来自 3 个唯一条件：
`T25_SOC0`、`T40_SOC0` 和 `T40_SOC12.5`。

证据还高度集中：`T40_SOC12.5` 同时出现在未见温度和 SOC 插值两个场景，它们是同一条
条件轨迹的两次场景出现，不能计作两个条件样本。该条件贡献 p=10 全部正向增益的约
**90.35%**：这里的分母是 `p=10` 的 21 个 scenario-condition occurrences 中所有正向
IAE gains（`V2 IAE - V3 IAE > 0`）之和。它还贡献未见温度场景正向增益的约
**80.77%**。因此平均改善存在，但对单一条件高度敏感。与此同时，一些精确回退行的绝对
IAE 仍较高；“不比 V2 更差”并不等于“误差足够小”。

跨全部四个 landmark 的 84 行合计为 **72 exact fallback、9 improvement、3 relative
regressions**。三处相对退化（`V3 IAE - V2 IAE`）是：

- `p=8, T25_SOC0`：`+0.52968452368701047 pp`；
- `p=8, T40_SOC0`：`+0.19573480305547392 pp`；
- `p=14, T40_SOC0`：`+0.048279793130974941 pp`。

因此，`p=10` 的 0 行相对退化只是该 landmark 的结果，不能概括完整 84 行审计。

## 证据边界

- 数据最长约 885 天，远短于 15-25 年，不能据此声称完成 15-25 年寿命预测验证；
- 当前只有 17 条条件均值轨迹，条件级评估单位 `N=17`，不包含可用于估计电芯间差异的
  单电芯轨迹；
- 激活结构与 `tau=7` 来自查看复用数据后的开发，严格 bootstrap 优越标准未通过；
- 没有海辰内部电芯或电站数据，也没有独立长期 LFP 队列的冻结解盲结果；
- 当前允许的最强表述是“公开复用数据上的稀疏、可解释开发信号，且实现可审计”；
- 下一步应冻结当前规则，在独立电芯级队列上一次性验证，而不是继续对 Naumann 调参。

## 审计产物

| 产物 | 用途 |
|---|---|
| [phase1_adversarial_audit.json](../showcase/audit_results/phase1_adversarial_audit.json) | 机器可读总览、检查状态与证据边界 |
| [data_condition_audit.csv](../showcase/audit_results/data_condition_audit.csv) | 17 条条件的身份、单位和完整性 |
| [future_label_attack_cases.csv](../showcase/audit_results/future_label_attack_cases.csv) | 四个 landmark 的未来标签攻击矩阵 |
| [independent_metric_audit.csv](../showcase/audit_results/independent_metric_audit.csv) | 504 个条件-方法组的独立复算 |
| [ablation_audit.csv](../showcase/audit_results/ablation_audit.csv) | 6 组消融在两场景的结果 |
| [gate_boundary_cases.csv](../showcase/audit_results/gate_boundary_cases.csv) | 6 个门控边界用例 |
| [failure_condition_table.csv](../showcase/audit_results/failure_condition_table.csv) | 84 行失败条件与风险动作清单 |
| [run_phase1_adversarial_audit.py](../scripts/run_phase1_adversarial_audit.py) | 独立审计运行入口 |
| [reproduce_public_release.py](../scripts/reproduce_public_release.py) | 全新环境一键复现入口 |
