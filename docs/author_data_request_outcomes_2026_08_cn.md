# 长期LFP作者数据请求结果

记录日期：2026-08-06
记录人：Jincheng Liu

## Vachenauer/TUM

根据请求人提供的实际回复，Veronika Vachenauer不愿披露本项目请求的逐电芯数据。项目没有取得原始数据，也没有取得数据使用许可。

该来源从“等待许可的十年端点候选”调整为“访问受阻的文献背景”。后续只可引用论文已经公开的聚合结论，不能训练模型、计算逐电芯指标或宣称完成十年端点验证。为尊重作者决定，不再重复索取同一数据。

## Yagci/Offenburg

2026-08-05 10:12（Asia/Shanghai），Gmail返回`mehmet.yagci@hs-offenburg.de`地址不存在或无法接收邮件。该退信只证明主收件地址投递失败，不能证明抄送收件人`wolfgang.bessler@hs-offenburg.de`也投递失败。

当前状态为“主地址退信，抄送投递情况未确认”。不向失效地址重复发送；先给抄送作者保留正常回复时间。如仍无回复，只通过经学校官方目录核验的联系方式进行一次后续联系。

## 证据影响

1. 两个来源均未向LifeTwin提供任何可训练或可评分的新增数据。
2. Vachenauer不能继续列为可获取数据候选，只保留为十年公开聚合端点的文献背景。
3. Yagci仍是产品形态高度匹配的潜在来源，但在取得有效联系、书面许可和逐电芯文件之前保持`D0`和不可训练状态。
4. 这不会降低现有公开开发结果，但意味着项目的长期独立证据缺口仍未补齐。

机器可读状态见：

- `configs/validation/author_data_request_queue_2026_08.json`
- `configs/validation/dataset_evidence_matrix_2026_08.json`
