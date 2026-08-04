# LifeTwin 独立验证执行手册

日期：2026-08-04

负责人：Jincheng Liu

当前状态：候选方法已提名，尚未取得新的合格长期 LFP 数据，尚未完成独立验证。

## 1. 这次冻结了什么

下一份未接触结果的长期 LFP 数据不再用于继续挑模型，而用于检验已经提名的候选方法。机器可读候选配置为
[`independent_safe_hard_candidate_v1.json`](../configs/validation/independent_safe_hard_candidate_v1.json)，语义 SHA-256 为
`596108e19ca0a8c7fb712bf82ca5be93817524f5f0c912f3b71b180a0fcba3af`。

候选方法固定为“安全硬门控的局部风险选择器”：

1. 目标电芯只提供 landmark 当时可见的前缀，完整参考轨迹只能来自训练分区。
2. 基础专家固定为持续值、平方根时间、有界幂律和相似电芯增量迁移。
3. 专家安全池只由训练分区的嵌套留一验证形成；相对持续值 IAE 不得超过 `1.25` 倍，绝对劣化不得超过 `0.1 pp`，不合格专家权重为零。
4. 局部风险只由训练分区留一误差估计；`k=[3,5,8]` 和风险余量 `[0,0.1,0.2,0.3,0.5]` 只能在训练分区内按已冻结目标选择。
5. 有支持时选安全池中局部风险最低的专家；证据模糊时回退到训练集全局风险最低的安全专家；域外输入拒绝确认性数值输出。
6. 连续混合模型不是下一轮主候选。形式化区间还要求每条路线至少 20 个独立校准 cluster 和 20 个独立审计 cluster，否则只能给诊断区间或拒绝。

这个候选是在 FastCharge 和 NASA 开发结果已经可见后提名的，因此当前只能说“为下一份数据预先锁定了候选”，不能把提名称为独立确认。

## 2. 数据到达前的防火墙

先复制并填写
[`independent_lfp_dataset_intake.template.json`](../configs/validation/independent_lfp_dataset_intake.template.json)。这一阶段只记录许可、来源版本、文件哈希、字段名、物理电芯 ID 可用性、时间支持和结局接触史，不录入容量真值。

如果结构信息和容量列位于同一个原始文件，优先让数据托管方或未参与建模的审计人生成只含字段、计数、时间范围和 ID 支持的元数据清单。建模人员不得为了选择 landmark、分区或模型打开目标容量值。所有接触论文图、汇总终点、作者 notebook 输出或目标列的行为都必须写入 exposure log。

数据许可与论文许可分开记录。公开可下载不等于允许模型训练，私下不公开也不会自动产生使用权。当前 intake 只支持非商业研究和竞赛评估，不支持原始数据再分发或商业模型开发。书面授权或托管方协议必须保存记录哈希。

PowerShell 可用以下命令记录原始字节属性：

```powershell
(Get-Item -LiteralPath 'D:\path\to\archive.zip').Length
(Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\path\to\archive.zip').Hash.ToLower()
```

## 3. 编译 intake

在仓库根目录运行：

```powershell
$env:PYTHONPATH='src'
python scripts\compile_independent_lfp_intake.py `
  path\to\dataset-intake.json `
  --output-directory artifacts\independent-lfp-intake
```

程序生成 `intake_report.json` 和 `protocol_draft.json`，默认拒绝覆盖已有结果。需要把“尚未就绪”作为流水线失败时增加 `--require-ready`。

主要状态含义如下：

| 状态 | 含义 | 允许动作 |
|---|---|---|
| `blocked_before_dataset_specific_freeze` | 许可、结构、时长、样本或哈希等硬门槛未通过 | 修正原因并新建 intake 版本 |
| `blocked_outcome_evidence_classification` | 结局接触史不足或与盲法声明矛盾 | 补全记录或降低证据等级 |
| `development_only_not_confirmation` | 数据已参与模型或协议开发 | 仅保留开发证据，另找未接触数据 |
| `ready_for_locked_retrospective_freeze_review` | 可进行锁定回顾性复现 | 人工复核后冻结，最高 D3 |
| `ready_for_dataset_specific_freeze_review` | 元数据门槛通过且结局盲法分类成立 | 人工复核后进入数据集专用冻结 |

`ready` 仍不等于协议已冻结，更不等于模型验证成功。编译器固定将 `protocol_can_be_frozen_now` 设为 `false`，要求第二人复核；它拒绝未知字段，因此把容量数组塞入 intake 会直接失败。被阻断的 intake 生成的协议草案会降为 `unclassifiable + D0`，不会继承确认性声明。

## 4. Ready 之后的不可倒置顺序

1. 第二位审计人核对许可、artifact SHA-256、接触史和 intake 报告。
2. 冻结训练、校准、测试和审计物理电芯 ID，四个分区必须两两不重叠。
3. 只依据时间和索引支持冻结主 landmark 与共同未来窗口。
4. 仅在训练分区执行嵌套留一，确定安全池、邻居数和风险余量；不得读取 test/audit 后缀结果。
5. 冻结适配器、评分器、依赖环境、候选和基线配置哈希，再运行长期协议验证器。
6. 先生成不含真值的预测 bundle 并记录 SHA-256 与 UTC 时间；之后才允许链接未来真值。
7. 独立评分器在固定窗口、按 cluster 等权计算结果。测试和审计必须全部报告，不能只选较好的一组。

数据集专用协议验证命令：

```powershell
$env:PYTHONPATH='src'
python scripts\verify_independent_long_term_protocol.py `
  path\to\dataset-specific-protocol.json
```

## 5. 面向评委的准确表述

**“不看未来结果是不是创新？”** 不是。它是实验可信度的底线。技术创新候选是训练域内安全池、相似前缀局部风险、硬选择与证据不足回退/拒绝的组合；未来标签隔离让该创新可以被可信检验。

**“项目是不是已经完成？”** 原型、公开数据开发实验、泄漏防火墙和独立验证协议已经完成；独立长期日历老化确认尚未完成。当前不能宣称海辰产品、电站或 15-25 年精度。

**“为什么不继续在 FastCharge 上提分？”** 该数据的目标结果已经用于开发，再调参只能提高回顾性分数，不能提高证据等级。最高价值工作是把已经提名的方法送入新队列。

**“这套 intake 只是表格吗？”** 不是。解析器执行严格字段白名单、重复 JSON key、NaN/Inf、SHA-256、UTC 时间、许可范围、整数计数、样本门槛、结局接触史和候选哈希校验，并调用长期协议的 JSON Schema 与跨分区语义验证器。失败会关闭确认性声明，而不是只显示警告。

**“现在最可信的结果是什么？”** FastCharge V2 是大样本公开循环老化上的开发证据：安全硬门控总体 MAE 为 `0.286 pp`，持续值基线为 `0.725 pp`。它支持方法可行性，但不是长期日历老化或海辰产品验证。

## 6. 本阶段完成标准

- 候选配置语义哈希不变，任何直接或文件级篡改均被测试拒绝。
- intake 模板在未填状态下必须输出 `blocked_as_designed`，不能误进入 ready。
- 合格的纯元数据样例最多进入“人工冻结复核”，不能自动冻结。
- 许可身份、授权记录、artifact 身份、整数计数、结局误标和未知测量字段攻击全部 fail closed。
- 每次提交运行专项测试、完整测试、Ruff、Git diff 检查和公开发布清单验证。

在真正取得新数据前，正确的下一步不是制造一个更漂亮的分数，而是保持这条边界不动。
