# 海辰私有循环老化数据盲测接入规程

## 目的与边界

本规程用于把企业内部 LFP 电芯 RPT 容量轨迹接入 LifeTwin，同时阻断模型开发进程读取锁定测试后缀。当前候选是 `v3_dual_clock_kernel_shrinkage`，适用对象是具有重复容量标定点的循环或混合老化轨迹。它不能仅凭本次接入证明储能电站 15 至 25 年寿命精度，也不能把 SNL 开发结果表述为海辰产品验证结果。

机器可读配置位于 `configs/validation/hithium_private_cycle_adapter_v1.json`。正式冻结前必须替换其中的 `hash_seed`，由项目负责人和数据保管人双人复核后保存配置哈希。种子不得在看到容量结果后更换。

## 数据角色分离

建议至少使用两个系统身份：

1. **数据保管进程**：可读取全部测量，负责格式校验、分区冻结、生成前缀和真值库。
2. **模型预测进程**：只能读取开发轨迹、校准或测试前缀、冻结配置与模型代码，不能读取任何 `*_truth_vault.private.parquet`。
3. **独立评分进程**：只在预测文件和完成清单冻结后读取真值库，输出评分与哈希。

同一台电脑上的不同目录不等于真正的权限隔离。正式盲测应使用不同服务账号、ACL 或对象存储访问策略，并保存访问日志。

## 标准字段

分区冻结阶段只接受以下三列，且每个物理电芯只能出现一次：

| 字段 | 含义 |
|---|---|
| `cell_id` | 脱敏后的物理电芯唯一标识，整个试验期间不变 |
| `batch_id` | 独立生产批次或更高层级分组，用于防止同批次跨分区 |
| `condition_id` | 固定老化工况标识，不得由未来容量结果构造 |

测量接入阶段严格接受以下字段，列名和顺序均须一致：

| 字段 | 单位或规则 |
|---|---|
| `record_id` | 全局唯一记录 ID |
| `cell_id` / `batch_id` / `condition_id` | 必须与冻结分区完全一致 |
| `cathode_chemistry` | `LFP`、`LiFePO4` 或 `lithium_iron_phosphate` |
| `temperature_c` | 老化工况温度，摄氏度；不是 RPT 室温的替代值 |
| `min_soc_pct` / `max_soc_pct` | 老化 SOC 窗口，0 至 100，且最小值小于最大值 |
| `charge_c_rate` / `discharge_c_rate` | 老化充放电倍率 |
| `visit_index` | 每个电芯从 0 开始连续编号的 RPT 序号 |
| `elapsed_days` | 自试验起点累计自然日，严格递增 |
| `equivalent_full_cycles` | 企业统一口径的累计 EFC，严格递增 |
| `capacity_ah` | 本次标准 RPT 的可用容量，不得使用日常部分循环吞吐量代替 |
| `reference_capacity_ah` | 冻结的容量归一化基准，必须大于 0 |
| `quality_status` | 当前配置仅接受 `accepted` |

适配器计算 `capacity_retention_pct = 100 * capacity_ah / reference_capacity_ah`。它不会猜测 EFC 口径、修补缺失 visit、插值容量或静默删除质量异常记录。

## 执行步骤

### 1. 冻结元数据分区

准备只含三列的 `partition_metadata.csv`，替换并复核配置种子，然后执行：

```powershell
$env:PYTHONPATH='src'
python scripts/prepare_private_cycle_blind_bundle.py freeze-partitions `
  D:\private-input\partition_metadata.csv `
  --config configs/validation/hithium_private_cycle_adapter_v1.json `
  --output-directory D:\private-vault\hithium-cycle-v1
```

这一步不会接受也不会读取容量、时间、EFC 或其他测量值。输出包括私有配置、分区清单和完成清单。记录终端显示的 `manifest_content_sha256`，完成双人复核后不再重分区。

### 2. 由数据保管进程构建盲测包

```powershell
$env:PYTHONPATH='src'
python scripts/prepare_private_cycle_blind_bundle.py build-bundle `
  D:\private-input\rpt_measurements.parquet `
  --config D:\private-vault\hithium-cycle-v1\adapter_config.private.json `
  --partition-manifest D:\private-vault\hithium-cycle-v1\partition_manifest.private.json `
  --output-directory D:\private-vault\hithium-cycle-v1\bundle
```

生成的文件角色如下：

| 文件 | 可访问进程 |
|---|---|
| `development_trajectories.private.parquet` | 模型训练进程 |
| `calibration_prefixes.private.parquet` | 模型预测进程 |
| `calibration_truth_vault.private.parquet` | 独立评分进程 |
| `locked_test_prefixes.private.parquet` | 最终模型预测进程 |
| `locked_test_truth_vault.private.parquet` | 最终独立评分进程 |
| `blind_bundle_manifest.private.json` | 三方均可读取，但不得修改 |

### 3. 校准阶段

只用 development 完整轨迹训练候选模型；只用 calibration 前缀生成预测。预测进程命令没有真值参数：

```powershell
$env:PYTHONPATH='src'
python scripts/run_private_enterprise_cycle.py predict `
  D:\private-vault\hithium-cycle-v1\bundle\development_trajectories.private.parquet `
  D:\private-vault\hithium-cycle-v1\bundle\calibration_prefixes.private.parquet `
  --adapter-config D:\private-vault\hithium-cycle-v1\adapter_config.private.json `
  --bundle-manifest D:\private-vault\hithium-cycle-v1\bundle\blind_bundle_manifest.private.json `
  --output-directory D:\private-prediction\hithium-calibration-v1
```

完成清单生成后，再由评分身份连接 calibration 真值库：

```powershell
$env:PYTHONPATH='src'
python scripts/run_private_enterprise_cycle.py score `
  D:\private-vault\hithium-cycle-v1\bundle\calibration_truth_vault.private.parquet `
  --prediction-directory D:\private-prediction\hithium-calibration-v1 `
  --adapter-config D:\private-vault\hithium-cycle-v1\adapter_config.private.json `
  --bundle-manifest D:\private-vault\hithium-cycle-v1\bundle\blind_bundle_manifest.private.json `
  --output-directory D:\private-score\hithium-calibration-v1
```

评分器会重新校验预测、决策、模型配置、数据包成员及完成清单哈希。校准阶段可以确定是否采用 V3、区间宽度和拒绝输出规则，但不得读取 locked-test 真值。

### 4. 锁定测试阶段

冻结最终模型和全部决策阈值后，用同一 `predict` 命令把输入替换为 `locked_test_prefixes.private.parquet`，且使用新的只写输出目录。先生成原子完成清单，再由独立评分进程用 `locked_test_truth_vault.private.parquet` 评分。任何崩溃重跑、人工删行、阈值修改或模型切换都必须作为新的试验版本登记，不能覆盖原结果。

## 最低验收条件

- 开发、校准、锁定测试的 `batch_id` 两两不重叠。
- 每个电芯至少支持冻结的最大 landmark 加两个未来 RPT 点。
- 预测进程命令行和 API 均不接受目标后缀参数。
- 预测、模型决策、配置、代码与运行环境均有哈希和完成清单。
- 同时报出条件等权误差、单体等权误差、改善工况比例和最坏工况回退。
- 区间必须报告覆盖率与宽度；证据不足时拒绝输出，不以扩大区间掩盖域外输入。
- 锁定测试结果无论成功或失败都保留，不能用校准结果替代。

## V4.1显式未来时间计划

默认命令继续使用V3的“前缀平均EFC/天保持不变”假设。只有在运行计划已于预测landmark之前形成时，才可额外提供严格列结构的私有Parquet计划表：

```powershell
$env:PYTHONPATH='src'
python scripts/run_private_enterprise_cycle.py predict `
  D:\private-vault\hithium-cycle-v1\bundle\development_trajectories.private.parquet `
  D:\private-vault\hithium-cycle-v1\bundle\calibration_prefixes.private.parquet `
  --forecast-schedule D:\private-input\declared_schedule.private.parquet `
  --adapter-config D:\private-vault\hithium-cycle-v1\adapter_config.private.json `
  --bundle-manifest D:\private-vault\hithium-cycle-v1\bundle\blind_bundle_manifest.private.json `
  --output-directory D:\private-prediction\hithium-calibration-schedule-v4-1
```

执行器会把规范化计划复制到预测目录，与预测、模型胶囊和决策共同写入完成清单。评分阶段会自动重新读取并校验该计划；文件缺失或任何字段变化都会使评分失败。

提供计划时，执行器默认选择`v4_1_explicit_elapsed_dual_clock`。V4.1仅把未来时间和EFC坐标送入双时钟预测器；计划温度、SOC窗口、放电倍率和分段EFC/天只用于支持域诊断，不直接修正容量曲线；充电倍率仅做溯源记录。这样保留真实日历时间信息，同时避免把不稳定的工况先验增量硬加到长期曲线上。

完整V4`v4_declared_schedule_delta_prior`仍可用`--schedule-mode v4_declared_schedule_delta_prior`显式运行，但只保留为失败机制对照，不是当前主候选。实际结束后才得到的未来工况必须标记为`oracle_upper_bound`，其结果不具备主证据资格。V4.1修订协议见`configs/experiments/private_enterprise_schedule_v4_1_amendment.json`，完整字段解释见`docs/hithium_private_data_dictionary_cn.md`。

V4.2`v4_2_support_gated_bounded_delta`是第二个、也是最后一个预注册候选。它只允许使用不超过25%的工况修正权重，并随计划工况到训练支持域的距离衰减；最终修正幅度还不得超过训练集内部LOCO诊断半区间的25%。V4.2不是默认模式，必须通过`--schedule-mode v4_2_support_gated_bounded_delta`显式运行。固定协议见`configs/experiments/private_enterprise_schedule_v4_2_preregistered.json`。

项目同时冻结了三个通用、结果无关的运行情景：参考计划、低温低利用率和高温高利用率。企业实际运行前必须由工程负责人审核这些参数，它们不是海辰运行建议或质保边界。生成命令如下：

```powershell
$env:PYTHONPATH='src'
python scripts/build_private_schedule_scenarios.py `
  D:\private-vault\hithium-cycle-v1\bundle\calibration_prefixes.private.parquet `
  --model-config D:\private-prediction\hithium-calibration-v1\model_config.private.json `
  --scenario-config configs/experiments/private_schedule_scenarios_v1.json `
  --output-directory D:\private-input\hithium-scenarios-v1
```

每个情景输出独立的Parquet计划和哈希，不能合并成一张表后选择最好结果。预测器必须逐情景运行；若高负荷情景超出训练支持域，应拒绝输出而不是强行给出数字。

完成V3基线和V4.1候选的两次独立校准评分后，使用冻结门槛生成唯一晋级决策：

```powershell
$env:PYTHONPATH='src'
python scripts/evaluate_private_schedule_v4.py `
  D:\private-score\hithium-calibration-v3\scores.private.csv `
  D:\private-score\hithium-calibration-v4-1\scores.private.csv `
  --baseline-summary D:\private-score\hithium-calibration-v3\score_summary.private.json `
  --candidate-summary D:\private-score\hithium-calibration-v4-1\score_summary.private.json `
  --preregistration configs/experiments/private_enterprise_schedule_v4_1_amendment.json `
  --output D:\private-score\hithium-calibration-v4-1\promotion_gate.private.json
```

候选缺失某个V3已覆盖的工况时，晋级器会按失败处理，不允许通过选择性拒绝困难工况来美化平均误差。只有对应候选的`promote_candidate=true`时，它才有资格参与最终选择；否则继续使用冻结V3。不得依据同一校准结果下调门槛或重新启用完整V4。

V4.2必须用同一晋级器和自己的预注册文件单独对比V3。若V4.1和V4.2都通过全部门槛，优先选择条件等权轨迹IAE更低者；两者差值不超过0.02个百分点时选择结构更简单的V4.1。选择完成前不得打开locked-test真值。

## 合成流程演练

在接触企业文件前，可运行不含真实电池数据的完整演练：

```powershell
$env:PYTHONPATH='src'
python scripts/run_private_enterprise_synthetic_dry_run.py `
  --output-directory D:\private-dry-run\lifetwin-v1
```

演练会生成合成开发、校准和锁定分区，比较V3、V4.1和V4.2，执行冻结晋级门槛，并生成锁定预测；它明确不打开锁定测试真值。`dry_run_complete.json`会密封全部输出。该结果只证明软件流程可执行，不构成任何电池精度证据。

## 当前尚未完成的企业侧事项

项目仓库目前没有海辰内部数据，因此没有海辰模型精度、区间覆盖率或长期外推结果。还需要企业侧确认 EFC 口径、RPT 标准、容量基准、批次独立性、异常记录处理、实际未来运行计划和锁定真值的访问控制。完成这些事项后，本适配器才是可执行的盲测入口，而不是一份数据格式说明。
