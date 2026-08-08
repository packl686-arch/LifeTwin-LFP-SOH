# 跨平台 CI 恢复收口报告——2026-08-07

## 状态

冻结提交 `b872c33ee6b5b2010e1478a808e48e0c64150928` 的 public-release CI 与 Pages 工作流均已通过。这是工程可复现性结果，不改变任何冻结科研结论，也不提高任何模型或数据集的证据等级。

## 事故与恢复时间线

[public-release-ci run 31120438473](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31120438473) 的前两次尝试发生在 GitHub 事故 `qcvjkzcs7j74` 期间：

- 第 1 次尝试：GitHub Actions 无法解析 action 下载信息，Windows job `92679779544` 因此失败；quality `92679779540` 与 Ubuntu `92679779599` 在仓库工作完成前被取消。
- 第 2 次尝试：Windows job `92682619571` 已跨过 setup 并进入完整复现，随后收到外部取消；quality `92682619528` 与 Ubuntu `92682619543` 也被取消。该记录作为事故证据保留，没有被描述成仓库失败。
- 第 3 次尝试：quality `92735031283`、Ubuntu reproduction `92735031282` 与 Windows reproduction `92735031392` 均成功完成。

[Pages run 31120438218](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31120438218) 也保留了事故期间第 1 次失败的尝试。第 2 次尝试中，build `92735032364`、deploy `92735124405` 与 report-build-status `92735124428` 均成功完成。

GitHub 于 `2026-08-07T00:06:24.906Z`（`2026-08-07T08:06:24.906+08:00`）首次将事故状态改为 monitoring。当时 Actions 与 Pages 均已 operational，因此触发了一次受控恢复重验。GitHub 随后于 `2026-08-07T02:04:44.460Z`（`2026-08-07T10:04:44.460+08:00`）将事故标记为 resolved；`2026-08-07T02:43:51.880699Z` 的无缓存检查确认事故已解决，两个组件继续保持 operational。

## Artifact 验证

第 3 次尝试产生的复现 artifacts 已下载并独立计算哈希：

| 平台 | Artifact ID | 摘要状态 | Pytest | 摘要 SHA-256 |
|---|---:|---|---:|---|
| Ubuntu | `8978157039` | 完整复现通过 | 914 passed, 0 skipped | `cfcf3d5b3ab4de746325d0540c374336abe600afcf6d9317c2206aaf96290c9e` |
| Windows | `8978128115` | 完整复现通过 | 914 passed, 0 skipped | `5b01ea991f9c2aecd9fa76af483197fe650d0d9ed543bcbba113b3c0f19318e2` |

下载的两个 artifacts 合计包含 144 个文件、19,549,204 bytes。64 条内部生成文件哈希均与下载文件一致，4 条图像哈希也全部一致。两个摘要都记录了 full mode、原子发布、`b872c33` 洁净 checkout、发布校验通过，以及所有复现命令返回码均为 0。

完整本地审计保留在 ignored 目录 `artifacts/ci-outage-recovery-validation-v1_6_1-20260807`，有意不提交。其 `output_manifest.json` 覆盖 154 个文件，SHA-256 为 `79c66bbe867412a88b941944798864d91ab338facdf47c1aa9560c9e952122eb`；所有 byte count 和 SHA-256 都已独立复算。

## 证据边界

本报告证明冻结提交通过了公开发布检查、GitHub 托管 Ubuntu 与 Windows 的完整复现，以及 Pages 工作流。它不增加模型精度、独立验证、NASA 或 BEEP 验证、真实电芯或真实电站证据，也不支持已经验证 15–25 年预测的说法。V0.14 仍为 `failure`（正式失败）；V0.15 仍为 `inconclusive_not_success`（未得出成功结论）；V0.16/V2.1 仍只完成实现冻结，尚无正式生成和评分结果。
