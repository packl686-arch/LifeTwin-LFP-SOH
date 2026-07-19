# LifeTwin

面向储能 LFP 电池的证据优先型 SOH 寿命数字孪生。

LifeTwin 不试图用一条短曲线直接“猜出 25 年寿命”，而是先从跨温度、
跨 SOC 的公开老化数据中学习共性规律，再用目标电芯已经产生的短期数据
持续修正长期轨迹，并在证据不足时明确扩大区间或回退稳定模型。

> 本仓库是独立竞赛研究原型，不是海辰官方产品，不含海辰内部数据，当前
> 结果也不构成海辰电芯或储能电站的产品精度承诺。

## 评委快速入口

建议按以下顺序阅读：

1. [开题报告补充材料](SUBMISSION_SUPPLEMENT.md)：一页了解问题、方案、结果与边界。
2. [数据分析样本](docs/data_analysis_sample.md)：从公开数据到结论的完整示例。
3. [相关项目经验](docs/project_experience.md)：仓库中可核验的研究与工程积累。
4. [研究笔记](docs/research_notes.md)：为什么从固定曲线走到动态门控模型。
5. [参考资料](docs/references.md)：论文、数据集、代码来源和许可状态。

## 核心方法

```mermaid
flowchart LR
    A["容量、温度、SOC、倍率与时间"] --> B["数据质检与统一时间轴"]
    B --> C["跨工况层次先验"]
    C --> D["目标电芯短期数据更新"]
    D --> E{"出现早期容量回升且数据充足?"}
    E -- "否" --> F["稳定层次幂律"]
    E -- "是" --> G["老化项 + 激活偏移项"]
    F --> H["SOH轨迹、区间与越限时间"]
    G --> H
    H --> I["新数据进入后滚动更新"]
```

三个核心设计是：

- **跨工况到单电芯的动态更新**：用温度和 SOC 条件学习群体先验，再用目标
  电芯少量观测同步修正退化幅度和时间指数。
- **早期激活与不可逆老化分离**：针对低 SOC 下容量先升后降的形状，增加
  饱和激活偏移项，避免单调模型把早期回升误判成长期快速衰减。
- **证据驱动的模型门控**：只有异常形状和观测数量同时满足条件才启用专用
  模型，否则回退稳定主模型，避免复杂模型在普通工况中过拟合。

## 已验证结果

下表是公开 Naumann 日历老化数据上的回顾性开发结果。`p=10` 表示每条目标
轨迹只用前 10 次容量检查建模，再预测后续轨迹；误差指标为平均轨迹绝对误差
（IAE，越低越好）。

| 场景 | 传统平方根曲线 | 层次幂律 V2 | 门控激活 V3 | V2 相对传统方法 | V3 相对 V2 |
|---|---:|---:|---:|---:|---:|
| 未见温度层级 | 1.1287 pp | 0.4801 pp | 0.3662 pp | -57.46% | -23.72% |
| 40 C SOC 插值 | 1.2864 pp | 0.6907 pp | 0.2097 pp | -46.31% | -69.64% |

必须同时看到限制：V3 在主前缀只触发 3 个唯一低 SOC 条件，大量条件与 V2
完全相同，因此描述性 bootstrap 区间上界为 0，而不是小于 0；严格优越标准
没有通过。`tau=3-14 day` 的核心敏感性网格保持平均改善，20-30 天出现反转，
主值 7 天又是在查看前一阶段失效后确定的 post-hoc 值。结论只能称为有潜力的
机制开发信号，不能称为独立验证。

![Phase 8 analysis summary](docs/assets/phase8_results.png)

## 仓库结构

```text
LifeTwin-LFP-SOH/
├── src/lifetwin/              原型代码
├── configs/experiments/       冻结实验协议
├── data/interim/              CC BY 4.0 的规范化 Naumann 表
├── scripts/                   Phase 6-8 独立运行入口
├── showcase/                  可直接运行的数据分析样本
├── docs/                      补充材料、研究笔记与参考资料
├── reports/                   三个主要技术阶段报告
├── tests/                     GitHub 公开版复现测试
└── release_manifest.json      发布文件哈希与证据边界
```

## 快速复现

Python 3.11 或以上版本：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,showcase]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe showcase\analyze_phase8_results.py
```

重新运行完整 Phase 8 开发实验（普通 CPU 约几十秒，输出目录不可覆盖）：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_calendar_v3_activation_development.py
```

Linux/macOS 将 Python 路径替换为 `.venv/bin/python` 即可。

## 数据与许可

仓库只包含一份可公开再分发的规范化数据表。其上游 Naumann 数据集由 Maik
Naumann 发布于 Mendeley Data，DOI 为
[`10.17632/kxh42bfgtj.1`](https://doi.org/10.17632/kxh42bfgtj.1)，许可为
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。转换和统计单位说明见
[数据说明](data/interim/README.md)。

Stanford Lam/Joule 长期数据和作者代码在本项目审计时未发现明确的数据/软件
许可，因此没有上传任何文件或代码副本，只在[参考资料](docs/references.md)中
保留公开链接与待澄清状态。

## 当前成熟度

- 研究逻辑与原型闭环：完成。
- 公开数据上的回顾性开发：完成。
- 输入哈希、未来标签隔离、行序不变性和重复运行：完成。
- 独立长期 LFP 队列验证：待完成。
- 海辰大容量电芯与真实电站验证：待完成。
- 15-25 年产品精度承诺：当前不允许。

提交人：Jincheng Liu  
版本：`0.9.0`，2026-07-19

