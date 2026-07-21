# 参考资料与数据来源

本清单优先列原始论文、作者或机构数据页和作者代码仓库。链接可公开访问不等于
自动获得再分发或商业使用权；许可状态单独列示。

## 核心论文

1. Naumann, M. et al. **Analysis and modeling of calendar aging of a commercial
   LiFePO4/graphite cell.** *Journal of Energy Storage* 17, 153-169 (2018).
   DOI: <https://doi.org/10.1016/j.est.2018.01.019>  
   用途：17 个温度/SOC 条件、885 天日历老化和半经验时间规律，是本项目公开
   日历模型与回测协议的主要依据。

2. Severson, K. A. et al. **Data-driven prediction of battery cycle life before
   capacity degradation.** *Nature Energy* 4, 383-391 (2019).
   DOI: <https://doi.org/10.1038/s41560-019-0356-8>  
   用途：证明早期电压/容量曲线包含循环寿命信号，同时提醒其快充循环任务与
   储能日历老化任务并不相同。

3. Attia, P. M. et al. **Closed-loop optimization of fast-charging protocols for
   batteries with machine learning.** *Nature* 578, 397-402 (2020).
   DOI: <https://doi.org/10.1038/s41586-020-1994-5>  
   用途：早期预测与闭环实验决策；LifeTwin 借鉴“新证据进入后更新”的思路，
   但不直接迁移其快充寿命模型。

4. Schuster, S. F. et al. **Systematic aging of commercial LiFePO4|Graphite
   cylindrical cells including a theory explaining rise of capacity during
   aging.** *Journal of Power Sources* 345, 254-263 (2017).
   DOI: <https://doi.org/10.1016/j.jpowsour.2017.01.133>  
   用途：说明 LFP/石墨电芯早期表观容量上升可以是真实且与 SOC 有关的现象，
   支持将早期偏移与不可逆老化分开建模。

5. Grolleau, S. et al. **Calendar aging of commercial graphite/LiFePO4 cell -
   Predicting capacity fade under time dependent storage conditions.**
   *Journal of Power Sources* 255, 450-458 (2014).
   DOI: <https://doi.org/10.1016/j.jpowsour.2013.11.098>  
   用途：温度/SOC 依赖和时变储存条件下的经验退化建模。

6. Sui, X. et al. **The Degradation Behavior of LiFePO4/C Batteries during
   Long-Term Calendar Aging.** *Energies* 14, 1732 (2021).
   DOI: <https://doi.org/10.3390/en14061732>  
   用途：长期 LFP 日历趋势和不同温度/SOC 条件的论文级交叉检查。

7. Lam, V. N. et al. **A decade of insights: Delving into battery calendar aging
   trends and implications.** *Joule* 9, 101796 (2025).
   DOI: <https://doi.org/10.1016/j.joule.2024.11.013>  
   数据：<https://osf.io/ju325/>；作者代码：
   <https://github.com/viveklam/Joule-Decade-Calendar-Aging>  
   用途：长期、多化学体系日历老化和幂律/Arrhenius 外推局限。LifeTwin 审计时
   OSF 数据与 GitHub 代码未发现明确许可文件，因此本仓库不复制其数据或代码。

8. Angelopoulos, A. N. and Bates, S. **A Gentle Introduction to Conformal
   Prediction and Distribution-Free Uncertainty Quantification.** (2021).
   arXiv: <https://arxiv.org/abs/2107.07511>  
   用途：有限样本分位数、校准单位和覆盖率诊断的理论背景。

9. Geisbauer, C., Woerle, M., Keil, P. and Jossen, A. **Experimental Calendar
   Ageing Data for Lithium-Ion Battery Chemistries.** Version 3 (2022).
   DOI: <https://doi.org/10.5281/zenodo.6685365>
   用途：15 个独立 LFP 电芯在 60 C、三个 SOC 下的 120 天外部应力筛查。
   数据为 CC BY 4.0；由于时间短且工况单一，不作为长期验证。

10. Vachenauer, V. et al. **Shelf life of lithium-ion batteries:
    Recommissioning LiFePO4/C cells after ten years of uninterrupted calendar
    aging.** *Journal of Power Sources* 654, 237779 (2025).
    DOI: <https://doi.org/10.1016/j.jpowsour.2025.237779>
    用途：100 个商业 26650 LFP/C 电芯在 6 °C、50% SOC 下连续存放十年的
    端点证据。论文报告 96%-98% 容量保持，但原始数据仅按请求提供，且十年期间
    没有中间检查点；最多用于端点检查，不能验证动态 landmark。

11. Yagci, M. C. et al. **Degradation modes of large-format stationary-storage
    LFP-based lithium-ion cells during calendaric and cyclic aging.**
    *Journal of Energy Storage* (2025).
    DOI: <https://doi.org/10.1016/j.est.2025.116774>
    用途：180 Ah 储能 LFP 电芯、约 850 天日历实验，是目前最接近命题产品
    形态的请求型长期候选；逐电芯数据与复用权利仍需作者书面确认。

12. Aeppli, D. et al. **Aging behavior of LiFePO4-based battery cells at stack
    level: A Second-Life cycling study.** *Journal of Energy Storage* (2025).
    DOI: <https://doi.org/10.1016/j.est.2025.117135>
    用途：9 个入选物理电芯、约 7 年 Second-Life、最高 9600 cycles 的混合老化
    迁移候选。未知 First-Life 与循环/日历老化不可分离，不能作为长期日历确认。

## 数据与代码来源

| 资源 | 入口 | 本仓库处理 |
|---|---|---|
| Naumann 日历老化数据 | <https://doi.org/10.17632/kxh42bfgtj.1> | CC BY 4.0；发布规范化条件均值表并保留署名与变更说明 |
| Geisbauer LFP 日历老化数据 | <https://doi.org/10.5281/zenodo.6685365> | CC BY 4.0；发布原始 2.7 KB LFP CSV；仅用于 120 天、60 C 外部应力筛查 |
| Severson/MATR 项目 | <https://data.matr.io/1/projects/5c48dd2bc625d700019f3204> | 不上传原始大文件；只在研究经历中描述权威身份审计 |
| Severson 作者代码 | <https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation> | 用于核对公开实验身份和协议；不在本仓库复制第三方代码 |
| CellJAR 固定传输快照 | <https://huggingface.co/datasets/mihnathul/celljar> | 本地研究阶段使用固定 revision；GitHub 版不上传仓库副本 |
| Lam/Joule OSF | <https://osf.io/ju325/> | 许可待澄清；不上传数据样本、作者代码或邮件 |
| 长期 LFP 候选登记表 | [`configs/validation/long_term_lfp_dataset_registry.json`](../configs/validation/long_term_lfp_dataset_registry.json) | 只发布一手来源元数据、资格判定与许可状态；不复制请求型或许可不明的数据 |

## 许可边界

- LifeTwin 原创代码当前为公开可查看、保留全部权利，详见仓库 `LICENSE`。
- Naumann 规范化 CSV 单独遵循 CC BY 4.0，详见 `data/interim/README.md`。
- Geisbauer 原始 LFP CSV 单独遵循 CC BY 4.0，详见
  `data/external/geisbauer_2022/README.md`。
- 论文开放许可不自动等于配套数据或代码具有相同许可。
- 海辰名称只用于说明竞赛命题背景，不表示官方背书或产品验证。
