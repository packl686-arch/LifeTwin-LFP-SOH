# LifeTwin

[![public-release-ci](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/workflows/ci.yml/badge.svg)](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/workflows/ci.yml)

面向储能 LFP 电池的证据优先型 SOH 寿命数字孪生。

LifeTwin 不试图用一条短曲线直接“猜出 25 年寿命”，而是先从跨温度、
跨 SOC 的公开老化数据中学习共性规律，再用目标对象已经产生的短期数据
持续修正长期轨迹，并在证据不足时明确扩大区间或回退稳定模型。

> 本仓库是独立竞赛研究原型，不是海辰官方产品，不含海辰内部数据，当前
> 结果也不构成海辰电芯或储能电站的产品精度承诺。

## 评委快速入口

先用三分钟看到系统如何工作，再按需深入证据：

1. [评委三分钟简报](JUDGE_BRIEF.md)：一屏了解问题、方法、四个关键结果和边界。
2. [在线评审控制台](https://packl686-arch.github.io/LifeTwin-LFP-SOH/judge-console/)：零安装交互查看通用回退、专用路由和外部负迁移三个冻结案例；[离线单文件](docs/judge-console/index.html)也随仓库冻结。
3. [真实前缀预测入口](showcase/product_demo/README.md)：用不含未来容量的请求生成预测、区间状态、拒绝原因和哈希。

前缀 CLI 面向源码或 editable checkout；当前版本不把冻结参考数据拆离 Git 仓库包装成
standalone wheel 推理服务。

深入证据：[V0.13 可操作入口报告](reports/product_entry_v013_2026-07-22.md) ·
[开题报告补充材料](SUBMISSION_SUPPLEMENT.md) ·
[V0.12 稳健性报告](reports/robustness_and_long_term_protocol_2026-07-21.md) ·
[独立长期验证预注册](docs/independent_long_term_lfp_preregistration.md) ·
[Phase 1 对抗性审计](reports/phase1_adversarial_audit_2026-07-20.md) ·
[参考资料](docs/references.md)

## 核心方法

```mermaid
flowchart LR
    A["容量、温度、SOC、倍率与时间"] --> B["数据质检与统一时间轴"]
    B --> C["跨工况层次先验"]
    C --> D["目标对象短期数据更新"]
    D --> E{"出现早期容量回升且数据充足?"}
    E -- "否" --> F["稳定层次幂律"]
    E -- "是" --> G["老化项 + 激活偏移项"]
    F --> R["训练域内有界残差修正"]
    G --> R
    R --> H["SOH轨迹与路由化不确定性"]
    H --> U{"校准与域证据足够?"}
    U -- "是" --> I["发放区间并滚动更新"]
    U -- "否" --> J["拒绝预测或扩大区间"]
```

三个核心设计是：

- **跨工况到目标对象的动态更新**：用温度和 SOC 条件学习群体先验，再用目标
  对象少量观测同步修正退化幅度和时间指数。面向“目标电芯/单电芯”更新是产品
  架构目标；当前公开实验没有单电芯轨迹，实际更新的是 target condition-mean
  trajectory（目标条件均值轨迹）。
- **早期激活与不可逆老化分离**：针对低 SOC 下容量先升后降的形状，增加
  饱和激活偏移项，避免单调模型把早期回升误判成长期快速衰减。
- **证据驱动的模型门控**：只有异常形状和观测数量同时满足条件才启用专用
  模型，否则回退稳定主模型，避免复杂模型在普通工况中过拟合。
- **受约束的残差学习**：残差只从训练条件的交叉拟合误差中学习，在 landmark
  处锚定为零，并受时间支持和幅度上限约束，不允许黑盒修正吞掉机理主模型。
- **路由化校准与拒绝发行**：specialist 和 fallback 分开按条件轨迹校准；样本量、
  工况域、前缀轨迹支持或长期独立证据不足时，不输出貌似精确的运营区间。

## 回顾性开发结果

下表是公开 Naumann 日历老化数据上的回顾性开发结果。`p=10` 表示每条目标
轨迹只用前 10 次容量检查建模，再预测后续轨迹；误差指标为平均轨迹绝对误差
（IAE，越低越好）。

| 场景 | 传统平方根曲线 | 层次幂律 V2 | 门控激活 V3 | V2 相对传统方法 | V3 相对 V2 |
|---|---:|---:|---:|---:|---:|
| 未见温度层级 | 1.1287 pp | 0.4801 pp | 0.3662 pp | -57.46% | -23.72% |
| 40 C SOC 插值 | 1.2864 pp | 0.6907 pp | 0.2097 pp | -46.31% | -69.64% |

必须同时看到限制：V3 在主前缀只触发 3 个唯一低 SOC 条件，大量条件与 V2
完全相同，因此描述性 bootstrap 区间上界为 0，而不是小于 0；严格优越标准
没有通过。`tau=3-14 day` 的核心敏感性网格保持平均改善；`tau=20 day` 时只有
未见温度场景反转，`tau=30 day` 时两个场景都反转。主值 7 天又是在查看前一阶段
失效后确定的 post-hoc 值。结论只能称为有潜力的机制开发信号，不能称为独立验证。

![Phase 8 analysis summary](docs/assets/phase8_results.png)

## Phase 1 对抗性审计

审计不是再做一遍同样的实验，而是主动攻击证据链：核对数据身份和单位，在
`p=5/8/10/14` 分别修改未来容量标签，独立复算 504 个条件-方法指标组，检查
六种方法的共同支持与 6 组消融，并对门控边界和专用模型失败进行故障注入。

审计还发现了一个真实评分完整性问题：仅检查预测包哈希，无法阻止攻击者篡改
`elapsed_days` 或 `is_final_checkup` 后重新计算哈希。V2/V3 评分器现改为连接权威
真值坐标、用真实时间轴积分并由真值派生末点；重新哈希的坐标和 final 攻击会被拒绝。

`p=10` 的 21 个场景-条件行中，主候选相对 V2 为 4 行改善、17 行精确回退、
0 行相对退化。0 退化主要来自“门控未触发就复制 V2”的结构，不代表 21 个条件都
验证了 V3；4 个改善行只有 3 个唯一条件。`T40_SOC12.5` 在两个场景中重复出现，
不能重复计作两份证据。在 `p=10` 的 21 个 scenario-condition occurrences 中，以
所有正向 IAE gains（`V2 IAE - V3 IAE > 0`）之和为分母，该条件贡献约 90.35%。

跨全部四个 landmark 的 84 行合计为 **72 exact fallback、9 improvement、3 relative
regressions**。三处相对退化（`V3 IAE - V2 IAE`）分别是：`p=8, T25_SOC0`
`+0.52968452368701047 pp`；`p=8, T40_SOC0` `+0.19573480305547392 pp`；
`p=14, T40_SOC0` `+0.048279793130974941 pp`。因此不能把 `p=10` 的零相对退化外推到
其他观测长度。

完整解释见 [Phase 1 对抗性审计报告](reports/phase1_adversarial_audit_2026-07-20.md)。
机器可读入口包括
[审计总览](showcase/audit_results/phase1_adversarial_audit.json)、
[未来标签攻击矩阵](showcase/audit_results/future_label_attack_cases.csv)、
[独立指标复算](showcase/audit_results/independent_metric_audit.csv)、
[消融表](showcase/audit_results/ablation_audit.csv)、
[门控边界](showcase/audit_results/gate_boundary_cases.csv)和
[84 行失败条件表](showcase/audit_results/failure_condition_table.csv)。

**审计通过不等于模型验证通过。** 数据仍只有 17 条条件均值轨迹，条件级评估单位
`N=17`，最长约 885 天；这不是单电芯级验证，不能支持海辰产品、真实电站或
15-25 年预测精度宣称。

## V0.11：landmark、区间和外部负结果

本轮先完成正确性审计，再扩展方法。所有前缀被放到检查点 14-34 的相同未来窗口
比较后，只有 `p=10` 同时满足两场景均值改善、逐条件零退化和唯一改善条件要求。
它被记为**回顾性信号 landmark**，而不是预注册确认点；确认值仍为 `null`。

V4 将层次机理均值、训练条件 LOCO 有界残差和按路由校准的轨迹区间组合起来。
fallback 路由的 5 个校准条件只足以形成 80% 诊断分位数；specialist 仅 1 个校准
条件，80%/90%/95% 都拒绝。4 条测试轨迹中只有 3 条获得回顾性 80% 诊断区间，
运营区间发放数为 **0**，因为校准结果已被复用且没有独立长期数据。

项目还引入许可明确的 Geisbauer 独立电芯级 LFP 队列做 120 天、60 C 外部应力
筛查。15 个电芯全部因前缀不足而回退稳定模型；主候选平均 IAE 为 `3.9735 pp`，
目标前缀平方根比较器为 `3.8852 pp`，主候选**没有胜出**。项目保留这个负结果，
不重调协议，也不把短期高温筛查改称长期验证。详见
[完整报告](reports/landmark_v4_external_evidence_2026-07-20.md)和
[机器可读证据包](showcase/evidence_v011/README.md)。

## V0.12：先检验结论有多脆弱

V0.12 不晋升新均值模型，而是对 v0.11 的两个核心结果做反证优先审计。固定 V4
训练状态后，在 10 个非训练条件上枚举全部 `C(10,6)=210` 个校准/评估划分。
fallback 路线只有 80% 分位数可计算，乘数范围为 `0.9243-2.1698`；specialist
路线每个划分只有 0-2 条校准轨迹，因此 80%/90%/95% 全部不可用。原切分宽度又
主要由 `T40_SOC50` 一条高误差轨迹支配。这证明当前区间适合做拒绝诊断，不适合
包装成稳定覆盖保证。

Geisbauer 的逐电芯复核在 `1e-12 pp` 数值零阈值下得到 8 个候选更好、7 个更差；
采用事后 `0.05 pp` 等效界时为 5 个更好、4 个更差、6 个等效。平均配对差为
`+0.0882 pp`，中位数为 `-0.0022 pp`；15 个彼此重叠的单电芯删除场景中有 2 个
会使总体方向反转。负迁移风险主要位于 100% SOC，但样本不足以给出确认性推断。完整数字、
名义检验边界和 210 个重叠划分的正确分母见
[V0.12 报告](reports/robustness_and_long_term_protocol_2026-07-21.md)。

![V0.12 robustness audit](docs/assets/v012_robustness.png)

项目同时公开长期 LFP [数据集资格登记表](docs/long_term_lfp_dataset_registry.md)、
[数据无关预注册模板](configs/validation/independent_long_term_lfp_protocol.template.json)和
[跨字段语义验证器](src/lifetwin/validation/long_term_protocol.py)。
当前可立即用于独立长期轨迹确认的公开数据集数量仍为 **0**。这不是搜索失败后留下
一句“待补数据”，而是把许可、物理电芯 ID、老化模式、时长、结局接触史和证据等级
变成机器可读门槛；获得作者数据后也不能临时修改成功标准。

## 仓库结构

```text
LifeTwin-LFP-SOH/
├── src/lifetwin/              原型代码
├── configs/experiments/       冻结实验协议
├── configs/inference/         前缀推理请求 Schema
├── configs/validation/        长期数据资格与预注册 Schema/模板
├── data/interim/              CC BY 4.0 的规范化 Naumann 表
├── data/external/             CC BY 4.0 的 Geisbauer LFP 外部应力数据
├── scripts/                   实验、审计与一键复现入口
├── showcase/product_demo/     不含未来容量的真实推理请求
├── showcase/                  数据分析样本与公开审计产物
├── docs/judge-console/        自包含中文评审控制台
├── docs/                      补充材料、研究笔记与参考资料
├── reports/                   技术阶段报告与对抗性审计报告
├── tests/                     GitHub 公开版复现测试
└── release_manifest.json      发布文件哈希与证据边界
```

## 快速复现

规范复现使用 Python 3.12.x 和冻结依赖约束：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements\reproduction.txt -e ".[dev,showcase]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe showcase\analyze_phase8_results.py --output artifacts\quick\phase8_results.png
.\.venv\Scripts\lifetwin.exe calendar-prefix-predict --request showcase\product_demo\naumann_t40_soc37_5_request.json --output-dir artifacts\prefix-demo
```

Linux/macOS：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c requirements/reproduction.txt -e '.[dev,showcase]'
.venv/bin/python -m pytest -q
.venv/bin/python showcase/analyze_phase8_results.py --output artifacts/quick/phase8_results.png
.venv/bin/lifetwin calendar-prefix-predict --request showcase/product_demo/naumann_t40_soc37_5_request.json --output-dir artifacts/prefix-demo
```

安装依赖后，推荐用一个命令完成发布预检、Phase 8、动态 landmark、V4 区间与
校准切分审计、Geisbauer 外部应力与逐电芯稳健性审计、Phase 1 审计、无界面绘图
和完整测试，并把证据原子化
写入新目录：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_public_release.py --mode full --output artifacts\reproduction
```

Linux/macOS：

```bash
.venv/bin/python scripts/reproduce_public_release.py --mode full --output artifacts/reproduction
```

该命令拒绝覆盖已有输出。GitHub Actions 的 Ubuntu/Windows fresh-clone 运行记录见
[public-release-ci](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/workflows/ci.yml)，
以对应提交的实际状态为准。
清单所列的已发布证据文件必须逐字节匹配 SHA-256。Phase 8 核心表的跨平台数值重算按
`2e-4` 绝对容差核对；Phase 1 审计表按主键对齐并默认逐字段精确比较，仅对求解器派生的
误差指标使用 `5e-3 pp`、派生比例使用 `1e-4`，审计残差仍限制为 `1e-10`，且拒绝
NaN/Inf。身份、工况、前缀、计数、真值、门控状态和结论字段不使用宽松容差。
比较器还会重算独立指标残差、消融差值、失败表回退关系与风险标签，避免各列分别落在
容差内却彼此矛盾。
环境敏感的模型状态哈希不做跨操作系统相等宣称，但必须保持行间等价类结构，并在每次
未来标签攻击运行内部满足 baseline 与 attacked 哈希相等。

重新运行完整 Phase 8 开发实验（普通 CPU 约几十秒，输出目录不可覆盖）：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_calendar_v3_activation_development.py
```

## 数据与许可

仓库包含两份许可明确且可公开再分发的数据文件。Naumann 数据集由 Maik Naumann
发布于 Mendeley Data，DOI 为
[`10.17632/kxh42bfgtj.1`](https://doi.org/10.17632/kxh42bfgtj.1)，许可为
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。转换和统计单位说明见
[数据说明](data/interim/README.md)。

Geisbauer LFP 数据由 Christoph Geisbauer、Markus Woerle、Philip Keil 和 Andreas
Jossen 发布于 Zenodo，DOI 为
[`10.5281/zenodo.6685365`](https://doi.org/10.5281/zenodo.6685365)，许可同为
CC BY 4.0。仓库保留原始 2.7 KB LFP CSV，说明见
[外部数据说明](data/external/geisbauer_2022/README.md)。它只有 120 天、60 C，
只用于外部应力筛查。

Stanford Lam/Joule 长期数据和作者代码在本项目审计时未发现明确的数据/软件
许可，因此没有上传任何文件或代码副本，只在[参考资料](docs/references.md)中
保留公开链接与待澄清状态。Vachenauer 十年端点、Yagci 180 Ah 储能电芯、Sui
月度轨迹和 Aeppli Second-Life 数据也仅登记一手来源与资格，不上传请求型数据。

## 当前成熟度

- 研究逻辑与原型闭环：完成。
- 公开数据上的回顾性开发：完成。
- 数据身份、未来标签隔离、独立评分复算与预测包坐标完整性：完成。
- 门控边界、专用模型故障回退和 84 行失败条件清单：完成。
- 固定共同未来窗口的动态 landmark 回顾性诊断：完成；确认性 landmark 待独立数据。
- 机理层级模型、LOCO 有界残差、路由化区间与拒绝发行：完成回顾性实现。
- 15 个独立 LFP 电芯的 120 天、60 C 外部应力筛查：完成，主候选未胜出。
- V4 全部 210 个校准划分与 Geisbauer 逐电芯/LOCO 稳健性审计：完成，未形成确认性结论。
- 严格请求 Schema、真实前缀推理 API/CLI、域外/异常前缀失败关闭与零安装评审控制台：完成。
- 长期 LFP 数据集资格登记、数据无关 Schema 与预注册模板：完成；当前合格公开确认集为 0。
- Ubuntu/Windows fresh-clone CI：已配置，状态由仓库徽章和对应提交记录公开显示。
- 独立长期 LFP 队列验证：待完成。
- 海辰大容量电芯与真实电站验证：待完成。
- 15-25 年产品精度承诺：当前不允许。

提交人：Jincheng Liu

版本：`0.13.0`，2026-07-22
