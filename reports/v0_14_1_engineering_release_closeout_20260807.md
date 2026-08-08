# LifeTwin v0.14.1 工程发布收口报告——2026-08-07

## 发布状态

`v0.14.1` 是在 `v0.14.0` 基础上发布的工程可复现性与数据治理补丁版本。它不替换、不修订，也不重新解释已经冻结的 `v0.14.0` 科研结果。V0.14 仍为 `failure`（正式失败）；V0.15 仍为 `inconclusive_not_success`（未得出成功结论）；V0.16/V2.1 仍只完成实现冻结，尚无正式生成和评分结果。

## 本次工程收口内容

- Windows 完整复现路径现在采用冻结的目录原子发布协议，并已完成公开的跨平台验证。
- GitHub 托管的质量检查、Ubuntu 复现、Windows 复现和 Pages 在 GitHub Actions/Pages 服务事故恢复后均成功完成。事故期间的失败和取消尝试仍保留在公开记录中。
- 数据治理改动均为仅向前纠正：MATR 的 identity-only 接入会在包含结局信息的 `summary` 内容之前停止；NASA 正式 `prepare`、`predict` 和 `score` 入口继续受权利门禁控制；生成的审计输出只能发布到新目录，并由清单覆盖。
- 解压后的 NASA 常规电池快照仅以元数据形式接入：38 个 MAT 文件、10 个 README/TXT 文件、34 个由文件名推导的 `Bxxxx` 身份，以及 4 组完全相同的重复表示。MAT/容量值读取、训练、预测、评分和 SNL 内容读取均为 0。
- 独立验证候选配置和 metadata-only 接入流程已经准备好，可在未来接入许可明确、结局盲化的数据集。当前没有符合条件的公开独立长期 LFP 确认队列。

NASA V3 的四个 CSV 工作与 FastCharge V1/V2 工作仍属于回顾性开发证据。NASA 化学体系尚未获得权威 LFP 认定；解压后的 NASA 元数据对象也与上述四个 CSV 是不同的证据对象。所有治理改动都没有产生新的模型精度结果。

## 公开证据

- 跨平台恢复与事故时间线：
  [`cross_platform_ci_recovery_closeout_20260807.md`](cross_platform_ci_recovery_closeout_20260807.md)
- 数据治理仅向前纠正的证据谱系：
  [`data_governance_forward_correction_closeout_20260807.md`](data_governance_forward_correction_closeout_20260807.md)
- 数据资产接入与 V1.1 纠正：
  [`../docs/data_asset_intake_20260806.md`](../docs/data_asset_intake_20260806.md) 和
  [`../docs/data_asset_intake_20260806_v1_1_correction.md`](../docs/data_asset_intake_20260806_v1_1_correction.md)
- NASA 来源与权利边界：
  [`../docs/nasa_pcoe_battery_data_provenance.md`](../docs/nasa_pcoe_battery_data_provenance.md)
- 独立验证执行边界：
  [`../docs/independent_validation_execution_2026_08_cn.md`](../docs/independent_validation_execution_2026_08_cn.md)

同一科研与工程树在发布前的跨平台证据保留于 [public-release-ci run 31142437998](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31142437998) 和 [Pages run 31142437189](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31142437189)。发布流程要求 `v0.14.1` tag 自身的 GitHub Actions run 也必须完成；更早的绿色 run 不能替代 tag 验收。

## 本地验证

在 Python 3.12、冻结复现约束和洁净 checkout 中执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements\reproduction.txt -e ".[dev,showcase]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\verify_public_release.py --project-root .
git diff --check
```

公开发布清单冻结除清单自身外的每一个 tracked 发布文件。发布包从 annotated tag 导出，不包含 NASA、BEEP、MATR 或 SNL 原始数据、下载的 CI artifacts、本地 competition 材料、凭证或机器特定路径。

## 证据边界

本补丁证明的是工程可复现性、发布完整性与数据治理控制。它没有新增独立验证、海辰产品证据、储能电站验证或真实 15–25 年精度证据。Naumann 的统计单位仍是条件均值轨迹；Geisbauer 仍是 60°C/120 天外部压力筛查；合成 25 年证据仍是结构性压力测试，而不是真实长期验证。
