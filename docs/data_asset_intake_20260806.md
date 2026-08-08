# LifeTwin 受治理的公开数据接入——2026-08-06

## 决定

数据治理接入通过。资产清点、物理身份规则、重复控制、访问角色和未来标签防火墙均已冻结。NASA 官方辅助基线在访问未来结局之前保持 **blocked**：虽然目录可公开访问，但没有找到数据集特定的许可证标识、许可证 URL，也没有找到允许发布汇总派生结果的确认。本次接入没有读取任何 NASA 未来容量或寿命值。

这是数据治理结果，不是模型精度提升，也不改变任何冻结的 LifeTwin 科研结论。

## 机器核验的资产清单

只读数据源包含 261 个文件，共 33,038,472,637 bytes（30.769475 GiB）。排除 3 个临时锁文件后，剩余 258 个有效文件。审计前后的源元数据快照完全一致。

| 资产 | 观察到的文件 | 受治理的统计单位 | 分配角色 |
|---|---:|---|---|
| MATR FastCharge | 140 JSON + 5 MAT | 物理电芯 barcode | 主模型开发训练 |
| SNL LFP holdout | 60 CSV + 1 个目录 workbook | 物理电芯 | 外部锁定 holdout |
| NASA 常规电池 | 6 ZIP | 物理电池 ID | 跨域循环/RUL 压力测试 |
| NASA randomized usage | 7 ZIP | 尚未核验 | 仅做身份清点 |
| CALCE A123 | 34 XLSX + 2 XLS | 电芯；与 session/workbook 区分 | 特征与输入验证 |
| Oxford Dataset 1 | 3 个文件 | 8 个物理电芯 | 跨化学体系/拒绝压力测试 |

3 个被排除的锁文件从未被打开、解析、复制或删除。

## 访问分配

1. **主模型开发训练：** MATR 只能沿用现有的 exposed-outcome 开发标签，不能作为独立确认。
2. **特征与输入验证：** CALCE 可用于字段、单位、温度、SOC、OCV 与动态工况检查。本次接入不把它作为寿命训练队列。
3. **跨域压力测试：** NASA 常规电池和 Oxford 只能在各自明确的化学体系边界内挑战循环轨迹或拒绝行为。NASA randomized bundles 仍只能做 identity-inventory。
4. **外部锁定 holdout：** SNL 被保留，除操作系统文件元数据外，主模型不得访问。

## SNL 强隔离

审计仅使用文件名后缀、扩展名和文件大小，将 30 个 `cycle_data` 文件名与 30 个 `timeseries` 文件名配对。审计没有打开这 60 个 CSV 文件或目录 workbook，也没有计算其内容哈希。

冻结政策如下：

- `main_model_access = metadata_inventory_only`
- `outcome_access = forbidden`
- `training_allowed = false`
- `model_selection_allowed = false`
- `reserved_external_holdout = true`
- `physical_cell_count = 30`
- `local_csv_count = 60`
- `metadata_catalog_record_count = 86`
- `locally_available_lfp_record_count = 30`

目录中的 86 条记录绝不能表述成已经下载 86 个电芯。

## MATR 身份与表示审计

140 个结构化 JSON 文件对应 135 个唯一物理 barcode。5 个 barcode 各有两个 JSON segment；这些 segment 在同一 barcode 内合并，绝不能跨越数据分区。审计发现 0 个解析失败和 0 个 batch/protocol 冲突。三个来源 batch 的物理电芯数量分别为 45、43、47；共有 70 个规范化 protocol 和 114,314 行 summary。

审计只读取 JSON header 与 summary 材料。大型 within-cycle arrays 被跳过，源文件哈希也被关闭。根目录 5 个 MAT 文件被识别为 2 个 MATLAB 7.3/HDF5 batch 表示和 3 个 MATLAB 5 作者结果表示；它们新增的物理电芯数为 0。MAT 与 JSON 的关联没有使用任何未来 cycle-life 或 final-capacity 值。无法通过权威身份元数据建立的逐电芯 MAT 链接继续视为有歧义并排除。

权威来源锚点继续采用 124 行 Severson crosswalk 和固定的作者代码提交 `1ef13d27c66dc3d73affdaa008fbeba5687b2ea4`、`0068fd0136bcd65884f5cd94b2b967c1ba73a668`。

## NASA 身份、重复与分区冻结

ZIP central directory 中包含 34 个唯一常规电池 ID。B0025、B0026、B0027 和 B0028 分别出现在两个 ZIP bundle 中，且未压缩大小与 CRC-32 一致，因此协议为每个物理电池规范化保留一个表示。未来如发现内容冲突的重复，自动接入立即停止。

物理电池分区在访问结局前冻结：18 个训练、8 个验证、8 个锁定测试电池。4 个重复 ID 全部位于验证组，因此重复表示不会跨分区。分区只使用电池 ID 和来源 bundle 身份，从不使用寿命、最终容量或模型结果。

NASA 化学体系记录为 `unspecified_li_ion_not_lfp_evidence`；唯一允许的定量角色是 `cross_domain_cycle_trajectory_and_rul_stress`。7 个 randomized-usage ZIP 中包含 28 个内部候选标识，但表示数量和 ZIP 数量都不是物理电芯数量。在物理身份得到单独核验前，randomized 数据不得训练或评分。

## 冻结的辅助协议与权利停止门

协议冻结前缀 20/40/60/100、cycle-200 最大评分范围、至少 20 个未来观测，以及 3 个基线：目标前缀 persistence、non-positive linear trend 和 constrained square-root-loss trend。预测只接受严格截断的 prefix table；未来标签单独写入；预测及其 manifest 在评分前计算哈希。

预定主指标是以百分点计的轨迹 MAE：先对物理电池等权平均，再对有效前缀等权平均。所有推断均为描述性推断。由于没有冻结预测区间，因此不提出预测区间结论。

执行继续保持 blocked：公开目录访问与预定的算法开发用途本身，不能解决数据集特定许可证或发布汇总派生结果的权利问题。在读取任何未来容量值或对锁定测试评分前，必须在新的协议版本中完成有记录的权利审查。原始数据继续禁止再分发。

## 宣称边界

本次接入不能说明新数据改善了主模型。NASA 和 Oxford 不能验证 LFP 性能；CALCE workbook 不是寿命训练样本；SNL 尚未验证主模型；重复使用的 MATR 不是独立确认。本次工作没有验证海辰电芯、储能电站或 15–25 年预测精度。V0.14 仍为 `failure`（正式失败）；V0.15 仍为 `inconclusive_not_success`（未得出成功结论）；V0.16/V2.1 仍只完成实现冻结。

## 2026-08-06 V1.1/V1.2 补充说明

上面的历史 V1 报告按原有事实保留。V1.1 纠正了 MATR 身份边界：identity-only reader 现在会在 `summary` key 之前停止，不调用 summary parser，不物化 `cycles_interpolated` 或结局值；仍将 140 个 JSON 文件解析为 135 个 barcode、5 个额外 segment 和 0 个身份冲突。

对解压后的 NASA 常规电池快照另行执行了零结局元数据接入：6 个顶层目录、38 个 MAT 文件、10 个 README/TXT 文件、34 个由文件名推导的唯一 `Bxxxx` 身份，以及 4 组 byte count/SHA-256 完全相同的重复表示。34 个身份不等于 34 个独立、同分布或合格测试电芯。只访问了文件元数据、128-byte MATLAB header、README/TXT 文本，以及通过 `whosmat` 得到的顶层 schema。MAT/容量值读取、NASA `prepare`/`predict`/`score` 和 SNL 内容读取均为 0。

README 暴露情况将角色固定为 `development_only_outcomes_and_protocol_structure_exposed`。NASA 化学体系没有得到权威确认，数据集特定许可与汇总结果发布权仍未解决，NASA 正式执行门继续关闭。NASA V3 的 4 个第三方 CSV 文件与本次 38-MAT 元数据接入仍是两个不同的证据对象。

V1.2 恢复了 `src/lifetwin/data/beep.py` 的冻结 SHA-256，将 identity reader 迁移到 `src/lifetwin/data/beep_identity.py`，并在不改变 `frozen_files_sha256`、release identity、版本、日期或任何冻结科研结论的情况下通过公开发布校验。所有这些纠正均未产生模型结果，也未提高证据等级。
