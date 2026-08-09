# 长期 LFP 作者数据申请包

日期：2026-08-04
署名：Jincheng Liu
状态：两封邮件均为可发送草稿，尚未发送

## 先说明边界

这两封邮件是在请求数据和书面使用许可，不是在索要论文。论文开放获取不自动代表
底层数据可以训练模型、参加竞赛或公开派生结果。LifeTwin 的首次分析仍限定为锁定的
回顾性外部评估：不根据申请到的结局重新选择候选方法或调参，不公开原始数据，也不
主张商业使用权。

两套数据的作用不同：

1. Yagci/Offenburg 是 180 Ah 储能 LFP，约 850 天且有多次检查，产品形态最贴题，
   优先级最高；若取得逐电芯轨迹，最高只能作为 D3 锁定回顾性轨迹复现。
2. Vachenauer/TUM 是 100 只 26650 LFP 电芯连续存放十年的 BOL/终点对照，长期锚点
   极强，但十年中没有中间检查，最高只能作为 D2 十年终点检验，不能验证动态 landmark。

机器可读的收件人、字段清单和证据边界见
[`author_data_request_queue_2026_08.json`](../configs/validation/author_data_request_queue_2026_08.json)。

## 发送顺序

1. 先发 Yagci：收件人 `mehmet.yagci@hs-offenburg.de`，抄送
   `wolfgang.bessler@hs-offenburg.de`。
2. 再发 Vachenauer：收件人 `veronika.vachenauer@tum.de`，第一封不必抄送学院秘书处。
3. 两封分开发送，不要群发；保留 Gmail 的已发送邮件和完整回复线程。
4. 7 至 10 个工作日无回复时，在原线程中发一次简短 follow-up，不另开新主题。

## 邮件一：Yagci/Offenburg

**To:** `mehmet.yagci@hs-offenburg.de`
**Cc:** `wolfgang.bessler@hs-offenburg.de`
**Subject:** `Data request: 180 Ah stationary-storage LFP calendar-aging study`

```text
Dear Mr. Yagci and Professor Bessler,

My name is Jincheng Liu. I am working on LifeTwin, a non-commercial research
and competition prototype for long-horizon state-of-health forecasting of LFP
batteries. It was prompted by an industry challenge, but it is not a deployed
product and we do not have access to Hithium internal data.

Your study is unusually relevant to our research because it examines commercial
180 Ah stationary-storage LFP/graphite cells over approximately 850 days:

M. C. Yagci et al., "Degradation modes of large-format stationary-storage
LFP-based lithium-ion cells during calendaric and cyclic aging," Journal of
Energy Storage 124 (2025), 116774.
https://doi.org/10.1016/j.est.2025.116774

The paper states that data will be made available on request. Would it be
possible to share an anonymized cell-level table for the calendar-aging campaign?
The smallest useful version would contain:

- anonymous physical-cell ID;
- storage temperature and SOC, with calendar/cycle campaign identification;
- elapsed storage day or measurement date for every reference test;
- discharge capacity in Ah, or capacity retention/SOH, at every reference test;
- reference-test temperature and charge/discharge protocol; and
- flags for missing tests, anomalies, exclusions, replacements, or interrupted
  storage.

If readily available, voltage-current-time reference-test curves, DC resistance,
energy/efficiency values, differential-voltage outputs, temperature excursions,
and batch identifiers would also be valuable, but they are optional. CSV, XLSX,
or MAT format would all be suitable. If the raw curves are too large to share,
the anonymized per-cell capacity table alone would already be very helpful.

For our first analysis, we would freeze the candidate method before inspecting
the cell-level outcomes and use the data only for a locked retrospective external
evaluation. We have already seen the published aggregate figures, so we would
not describe this as outcome-blind or prospective validation.

Could you also confirm whether you permit us to:

1. store and process the data locally for non-commercial research and competition
   evaluation;
2. perform normalization, feature extraction, and model evaluation;
3. publish only aggregate metrics and figures, and derived tables that do not
   reconstruct raw measurements; and
4. retain the data privately without redistributing the raw files?

We will not claim commercial-use rights. Please let us know if a data-use
agreement, different restrictions, or specific citation wording is required.

Thank you for considering the request and for making this unusually relevant
stationary-storage study openly accessible.

Best regards,
Jincheng Liu
```

## 邮件二：Vachenauer/TUM

**To:** `veronika.vachenauer@tum.de`
**Cc:** 留空
**Subject:** `Data request: ten-year LiFePO4/C shelf-life study`

```text
Dear Ms. Vachenauer,

My name is Jincheng Liu. I am working on LifeTwin, a non-commercial research
and competition prototype for long-horizon state-of-health forecasting of LFP
batteries. It was prompted by an industry challenge, but it is not a deployed
product and we do not have access to Hithium internal data.

I am writing about your ten-year shelf-life study:

V. Vachenauer et al., "Shelf life of lithium-ion batteries: Recommissioning
LiFePO4/C cells after ten years of uninterrupted calendar aging," Journal of
Power Sources 654 (2025), 237779.
https://doi.org/10.1016/j.jpowsour.2025.237779

The study's paired measurements before and after ten years of uninterrupted
storage provide a rare long-horizon reference. Would it be possible to share an
anonymized per-cell table for the 100 cells? The smallest useful version would
contain:

- an anonymous physical-cell ID preserved between the beginning-of-life and
  post-storage measurements;
- beginning-of-life and post-storage capacity;
- beginning-of-life and post-storage internal resistance;
- exact elapsed storage time, or storage start and end dates;
- storage temperature and SOC, including any known deviations; and
- capacity/resistance test protocols plus flags for missing values, anomalies,
  exclusions, or failed cells.

If readily available, pseudo-OCV or differential-voltage results, rate-capability
measurements, later cycle-aging trajectories, and batch identifiers would also be
valuable, but they are optional. CSV, XLSX, or MAT format would all be suitable.

Because there were no intermediate measurements during the uninterrupted storage
period, we would use this dataset only as a ten-year endpoint check, not as evidence
that a dynamic forecasting trajectory has been validated. We have already seen the
published aggregate 96%-98% capacity-retention result, so the analysis would be
reported as a locked retrospective evaluation rather than outcome-blind validation.

Could you also confirm whether you permit us to:

1. store and process the data locally for non-commercial research and competition
   evaluation;
2. perform normalization, feature extraction, and model evaluation;
3. publish only aggregate metrics and figures, and derived tables that do not
   reconstruct raw measurements; and
4. retain the data privately without redistributing the raw files?

We will not claim commercial-use rights. Please let us know if a data-use
agreement, different restrictions, or specific citation wording is required.

Thank you for considering the request and for carrying out this exceptionally
long-running experiment.

Best regards,
Jincheng Liu
```

## 简短追问模板

仅在 7 至 10 个工作日无回复后，于原邮件线程发送：

```text
Dear [Mr. Yagci / Ms. Vachenauer],

I am following up on the data and permission request below. Even a small
anonymized per-cell table containing only the core capacity measurements and
test metadata would be sufficient for our first non-commercial retrospective
evaluation. I would be grateful if you could let me know whether sharing it is
possible or whether a data-use agreement is required.

Best regards,
Jincheng Liu
```

## 收到回复后的处理顺序

1. 先不要打开数据表中的容量、SOH 或阻抗结果。
2. 保存完整邮件线程或书面许可原件，记录收到时间、文件字节数和 SHA-256；许可记录
   默认不放入公开 GitHub，只在 intake 中保存摘要和哈希。
3. 核对回复是否明确覆盖非商业研究、竞赛评估、本地处理、模型评估、汇总图表和
   非原始派生表；没有明确同意的用途一律视为未授权。
4. 由未参与模型选择的人先做仅元数据检查：文件清单、字段、物理电芯 ID、时间支持、
   工况、缺失标志和许可范围，不查看目标容量数值。
5. 复制并填写
   [`independent_lfp_dataset_intake.template.json`](../configs/validation/independent_lfp_dataset_intake.template.json)，
   运行 intake 编译器；只有通过硬门槛并经第二人复核后才能冻结数据集专用协议。
6. Yagci 数据若获准，只能晋升为锁定回顾性轨迹复现；Vachenauer 数据若获准，只能
   晋升为十年终点检查。两者都不能包装为海辰产品验证或 15 至 25 年精度证明。

## 一手来源核对

- Yagci 论文与开放获取记录：<https://opus.hs-offenburg.de/10762>
- Yagci 官方个人页：<https://www.hs-offenburg.de/personen-detail-seite/lsf/detail/1516>
- Bessler 官方团队页：<https://www.hs-offenburg.de/en/research/institutes/ines/electric-energy-storage>
- Vachenauer 论文的 TUM 记录：<https://portal.fis.tum.de/en/publications/shelf-life-of-lithium-ion-batteries-recommissioning-lifeposub4sub/>
- Vachenauer 官方个人页：<https://www.epe.ed.tum.de/en/ees/forschungsteams/vachenauer-veronika/>
