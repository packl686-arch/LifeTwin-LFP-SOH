# 长期 LFP 数据集资格登记表

审计日期：2026-07-21

审计范围：论文、作者/机构页面、Zenodo、Mendeley Data、OSF 和作者 GitHub 仓库等一手来源。搜索目标是同时具备明确数据许可、独立物理电芯标识、LFP/石墨体系、可分离日历老化、至少 730 天和可供早期到未来评分的轨迹。

机器可读登记表见 [`../configs/validation/long_term_lfp_dataset_registry.json`](../configs/validation/long_term_lfp_dataset_registry.json)。本文只给出判定所需的高信号摘要。

## 结论先行

截至审计日，**没有发现一份现在即可用于独立长期轨迹确认的公开数据集**。缺口不是单纯“数据量不够”，而是资格条件不能同时满足：

1. Lam/Joule 最接近长期轨迹确认，但 OSF 数据许可和 GitHub 代码许可仍为空，且本项目已经检查过其公开汇总数据的结构与队列信息。因此即使后续获得许可，也只能做锁定的回顾性外部复现。
2. Vachenauer/TUM 提供最强的十年 LFP 端点证据，但十年期间没有中间检查点，数据又仅注明按请求提供。它最多支持十年端点检查，不能确认动态 landmark。
3. Yagci/Offenburg 的 180 Ah 储能电芯最贴近目标产品形态，约 850 天也越过两年门槛，但原始电芯级数据仅按请求提供。
4. Sui/Aalborg 有 27 至 43 个月、月度检查和 15 个 LFP/石墨电芯，但没有公开机器可读的个体轨迹。
5. 已公开且许可明确的 Geisbauer 只有 120 天；Naumann 已被用于开发；长期现场数据不是纯日历队列或缺少个体容量标签。

## 核心候选

| 数据源 | 时长与粒度 | 许可/可用性 | 当前判定 | 下一步 |
|---|---|---|---|---|
| [Lam et al., Joule 2025](https://doi.org/10.1016/j.joule.2024.11.013), [OSF](https://osf.io/ju325/), [作者代码](https://github.com/viveklam/Joule-Decade-Calendar-Aging) | 232 个商业电芯，8 类电芯，最长 13 年，含 LFP 子集和逐电芯 JSON | 论文开放；2026-07-21 审计时 OSF `node_license=null`，GitHub `license=null` | **许可阻塞**；获许可后最高为锁定回顾性轨迹复现 | 等作者书面澄清数据和代码许可，不下载、不改编、不分发 |
| [Vachenauer et al., JPS 2025](https://doi.org/10.1016/j.jpowsour.2025.237779), [TUM](https://portal.fis.tum.de/en/publications/shelf-life-of-lithium-ion-batteries-recommissioning-lifeposub4sub/) | 100 个 26650 LFP/石墨电芯，6 °C、50% SOC 连续存放十年；容量保留 96% 至 98% | 论文 CC BY 4.0；数据 `available on request`，数据许可未单列 | **十年端点候选**；无中间轨迹，不能验证 landmark | 请求 BOL/十年端点逐电芯表和明确数据使用许可 |
| [Sui et al., Energies 2021](https://doi.org/10.3390/en14061732) | 15 个 LFP/石墨电芯，5 个工况，每工况 3 个电芯；27 至 43 个月，约每月检查 | 论文 CC BY 4.0；Data Availability 为 `Not applicable` | **长期轨迹候选但数据未公开** | 请求月度逐电芯容量表和研究/竞赛使用条款 |
| [Yagci et al., JES 2025](https://doi.org/10.1016/j.est.2025.116774) | 180 Ah 储能 LFP/石墨电芯；约 850 天日历老化，35/50 C、75/100% SOC | 论文 CC BY 4.0；数据 `available on request` | **领域最接近的请求型候选** | 优先请求逐电芯 RPT 轨迹、工况表和许可 |
| [Aeppli et al., JES 2025](https://doi.org/10.1016/j.est.2025.117135), [Empa PDF](https://www.dora.lib4ri.ch/empa/islandora/object/empa%3A41733/datastream/PDF/Aeppli-2025-Aging_behavior_of_LiFePO4-based_battery-%28published_version%29.pdf) | 9 个入选的 100 Ah LFP 物理电芯，总年龄超过 14 年，Second-Life 约 7 年、最高 9600 次循环 | 论文 CC BY 4.0；数据 `available on request` | **混合/循环迁移候选**；First-Life 工况缺失，不能分离日历老化 | 只用于混合老化迁移和拒绝诊断，不作日历确认 |

## 已公开但不合格的数据

| 数据源 | 可用价值 | 不合格原因 |
|---|---|---|
| [Naumann/Mendeley](https://doi.org/10.17632/kxh42bfgtj.1) | 29 个月、17 工况、CC BY 4.0，是当前方法开发主数据 | 已用于模型、landmark 和协议开发；不是独立证据，公开评估文件的物理电芯 ID 也未在仓库元数据中确认 |
| [Geisbauer/Zenodo](https://doi.org/10.5281/zenodo.6685365) | 15 个独立 LFP 电芯、CC BY 4.0，可做外部负迁移检查 | 只有 120 天、单一 60 C，早期前缀只有两个正时间点 |
| [Figgener/RWTH home storage](https://doi.org/10.5281/zenodo.12091223) | 21 套户储系统、最长 8 年、CC BY 4.0、定期容量测试 | 系统级而非物理电芯级；含 LFP/NMC/LMO-NMC；现场循环和日历老化不可分离 |
| [Schaeffer LFP field data](https://doi.org/10.5281/zenodo.13715694) | 28 套 LFP 系统、224 路电芯电压、最长 5 年，适合故障与域偏移研究 | CC BY-NC 4.0；只有包电流，没有逐电芯容量/SOH 标签；使用工况未知且样本全部来自异常退货 |
| [WMG LGM50 calendar data](https://doi.org/10.5281/zenodo.14577286) | 117 个电芯、39 工况、496 至 770 天、CC BY 4.0 | NMC811/SiOx-石墨，不是 LFP；10.3 GB 下载对当前确认任务没有直接价值 |

## Lam/Joule 许可复核

本次只读取仓库和 OSF 元数据，没有下载原始或汇总测量文件。

- OSF API 返回项目名 `Stanford Long Term Calendar Aging Dataset`、最后修改时间 `2025-03-31T15:32:36.903932`，`node_license=null`。
- GitHub API 返回默认分支 `main`、最后推送时间 `2025-02-28T22:18:58Z`，仓库 `license=null`；公开文件列表中仍未发现 LICENSE。
- 论文的 CC BY 4.0 不能自动扩展到 OSF 数据和 GitHub 代码。作者回复前继续执行“不下载、不改编、不训练、不分发”的边界。

## 请求顺序

1. **Vachenauer/TUM**：十年端点是目前最稀缺的外部锚点。先问能否提供匿名逐电芯 BOL/十年容量、阻抗和元数据，并明确研究竞赛、派生指标和汇总图的许可。
2. **Yagci/Offenburg**：180 Ah 储能形态最贴题。请求约 850 天逐电芯 RPT 序列、工况、异常记录和许可。
3. **Sui/Aalborg**：请求月度逐电芯容量/内阻序列。若只能得到论文中位数，不满足个体电芯协议。
4. **Lam/Joule**：继续等待已发送的许可澄清。许可明确后才能冻结专用协议，但证据角色保持锁定回顾性复现。
5. **Aeppli/Empa**：只在需要混合老化、Second-Life 或拒绝机制证据时申请。

任何作者提供的数据都必须先记录授权文本、版本、字节数和 SHA-256，再检查内容。不能先浏览终点、再调整模型或成功门槛。
