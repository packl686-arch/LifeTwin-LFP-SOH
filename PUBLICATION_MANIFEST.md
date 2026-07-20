# GitHub 发布清单

## 发布目的

本目录是面向竞赛评委的轻量、可运行、可追溯版本。它从完整本地研究工作区中
筛选核心代码、许可明确的数据、聚合结果和说明材料，避免把数十 GB 原始数据、
重复快照、临时文件或许可不清内容推送到 GitHub。

## 已包含

- LifeTwin Python 源代码及 Phase 6-8、landmark、V4 和外部应力 runner。
- 六份冻结实验配置和阶段技术报告。
- CC BY 4.0 的 Naumann 规范化条件均值表与 Geisbauer 原始 LFP CSV。
- Phase 8 聚合比较、tau 敏感性和目标诊断表。
- 动态 landmark、V4 校准/拒绝和 Geisbauer 外部负结果的机器可读证据包。
- 公开版复现测试、GitHub Actions 和图表生成脚本。
- 开题补充材料、相关项目经验、数据分析样本、研究笔记和参考清单。

## 主动排除

| 排除内容 | 原因 |
|---|---|
| FastCharge、CellJAR 原始/处理后大文件 | 文件规模大，且应由上游来源和固定版本重新获取 |
| 大型 Phase 6-8 全量预测包、bootstrap 明细和重复运行快照 | GitHub 评审不需要重复数 GB 证据；V0.11 的小型预测包保留用于直接审计 |
| Lam/Joule 数据样本和作者代码 | 数据和软件许可仍待澄清 |
| 许可请求邮件与发送截图 | 不属于评委复现实验所需材料 |
| 非 canonical 和失败前临时产物 | 防止评委误用旧结果 |
| pytest 临时目录、缓存和本地绝对路径 | 环境噪声与隐私控制 |
| 海辰内部数据 | 本项目未获得，也未虚构 |

## 发布验证

机器可读哈希位于 [`release_manifest.json`](release_manifest.json)，运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_public_release.py
.\.venv\Scripts\python.exe -m pytest -q
```

发布版的目标是“评委可理解、核心结果可复算、边界可检查”，不是完整复制本地
研究归档。

## Phase 1 审计增量

`v0.10.0` 在原 Phase 8 结果之外新增对抗性审计代码、机器可读审计产物、
失败条件表、评分完整性测试、门控故障注入测试和跨平台一键复现入口。发布清单
同时冻结这些证据文件的 SHA-256。被 Git 忽略的 `artifacts/` 仅用于本地或 CI
重跑结果，不属于 GitHub 发布内容，也不会覆盖已发布证据。

fresh clone 使用 Python 3.12.x，并以 `requirements/reproduction.txt` 约束安装依赖后使用：

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_public_release.py --mode full --output artifacts\reproduction
```

Linux/macOS：

```bash
.venv/bin/python scripts/reproduce_public_release.py --mode full --output artifacts/reproduction
```

命令先验证 Git 跟踪文件和冻结哈希，再运行 Phase 8、Phase 1 审计、无界面绘图与
完整测试；失败时不发布半成品输出目录。
清单所列的发布证据文件必须精确匹配 SHA-256。Phase 8 核心表跨平台重算使用 `2e-4`
绝对容差；Phase 1 表按文件声明主键和字段策略核对，身份/协议/计数/真值/状态字段精确
匹配，求解器派生误差指标限于 `5e-3 pp`，派生比例限于 `1e-4`，审计残差限于
`1e-10`，并拒绝非有限值。派生模型状态哈希验证格式、行间等价类结构与同运行内部一致性。
独立重算残差、消融关系、回退关系、排名和风险标签还要满足同一行及同组内的不变量。

## V0.11 证据增量

`v0.11.0` 新增三条彼此独立的证据链：固定共同未来窗口的动态 landmark 诊断；
7/6/4 条件切分下的“机理层级均值 + LOCO 有界残差 + 路由化校准”；以及许可明确的
Geisbauer 15 电芯、120 天、60 C 外部应力筛查。后者的主候选没有优于目标前缀
平方根比较器，负结果与逐电芯指标一并发布。

一键复现会重新运行三项实验并核对冻结结果、核心 CSV 和禁止宣称。当前仍明确禁止：
把 `p=10` 称为确认性最早 landmark；把回顾性诊断区间称为正式覆盖保证；把 120 天
高温筛查称为长期外部验证；以及任何海辰产品、真实电站或 15-25 年精度声明。
