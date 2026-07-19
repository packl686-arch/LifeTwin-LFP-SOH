# GitHub 发布清单

## 发布目的

本目录是面向竞赛评委的轻量、可运行、可追溯版本。它从完整本地研究工作区中
筛选核心代码、许可明确的数据、聚合结果和说明材料，避免把数十 GB 原始数据、
重复快照、临时文件或许可不清内容推送到 GitHub。

## 已包含

- LifeTwin Python 源代码及 Phase 6-8 核心 runner。
- 三份冻结实验配置和 Phase 6-8 技术报告。
- CC BY 4.0 的 Naumann 规范化条件均值表。
- Phase 8 聚合比较、tau 敏感性和目标诊断表。
- 公开版复现测试、GitHub Actions 和图表生成脚本。
- 开题补充材料、相关项目经验、数据分析样本、研究笔记和参考清单。

## 主动排除

| 排除内容 | 原因 |
|---|---|
| FastCharge、CellJAR 原始/处理后大文件 | 文件规模大，且应由上游来源和固定版本重新获取 |
| 全量预测包、bootstrap 明细和重复运行快照 | GitHub 评审不需要重复数 GB 证据；聚合表足以复核主要结论 |
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
