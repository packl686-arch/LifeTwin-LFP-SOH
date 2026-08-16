# LifeTwin V2.10 因果充分性终结审计

日期：2026-08-15  
状态：`development_only / causal_gate_failed / v2_11_permanently_closed`

## 执行结论

已按批准计划完成基线复核、结果盲故障注入、资源遥测和全规模合成长稳重复探针。
新增证据进一步否定了“Windows 六 worker 路径必然失败”“运行超过 397 秒必然失败”以及
“5,950 簇规模必然失败”，但没有自然复现 V2.10 的底层异常。

因此，V2.11 因果充分性门判定为 **不通过**：没有 failing-before/passing-after 对照，不能把
冻结证据中的 A（capability/IO）、B（fit/worker）和 C（进程异常终止）唯一收敛到一个机制，
也就不存在可证明修复该机制的最小代码变更。

本研究链在 V2.10 的冻结终态永久停止。不会从这组证据创建 V2.11 seed、协议、attempt、
四根目录、实现冻结或正式 bundle，也不会把换样本运行包装成 V2.10 的重试、替代或续跑。
若未来开展研究，只能作为具有独立问题定义和独立预注册的新研究链，而不是 V2.11。

## 不变边界

- V2.10 冻结提交：`3b0e4008bad4c82fe5741391c65e004424e19f77`
- 唯一正式 attempt：`v030-formal-20260814-a1`
- 冻结终态：`terminal_pre_prediction / unclassified_terminal_not_success`
- 原因码：`UNKNOWN_PRE_PREDICTION_EXCEPTION`
- 没有 prediction commitment，没有评分，`opened_truth_files` 为空
- 本次未调用正式 generator，未使用正式 seed，未读取正式行或密封真值
- 没有修改 V2.10 的模型、优化器、阈值、分区、端点或成功门

开发诊断提交：

1. `9aa9f0863a4fd95c7643ca120821311ca8763ce5`：保留安全的异常 cause/context 链；
2. `14b1c3df388d96bae649298a2a8fccb364985a21`：增加结果盲故障矩阵和聚合资源遥测。

两者都只改善未来诊断能力，不构成 V2.10 底层触发修复。

## 故障注入矩阵

故障矩阵在纯合成、非正式输入边界上覆盖六个阶段，全部得到预期的安全结构：

| 注入点 | 结构化异常链 | 观测 |
|---|---|---|
| verified-bundle IO | `V024PredictionError -> V024IOError` | proven-integrity 分类 |
| pool startup | `V015PredictionError -> _InjectedPoolStartupError` | phase 可区分 |
| worker exception | `V015PredictionError -> _InjectedWorkerError` | phase 可区分 |
| abrupt worker | `V015PredictionError -> BrokenProcessPool` | 独立 spawn 校准退出码 `71` |
| invalid worker output | `V015PredictionError` | validation frame 可保留 |
| executor shutdown | `_InjectedShutdownError` | shutdown 边界可观测 |

输出不含异常原文、PID、cluster ID、输入值或真值。资源遥测只保存聚合进程数、工作集、
private bytes、可用物理内存与磁盘余量。矩阵证明开发版诊断能够区分已知注入，但不能反向恢复
V2.10 冻结记录中从未保存的异常链。

## 长稳重复探针

全部探针使用诊断专用 PCG64DXSM 生成的 randomized prefix-only fixture，固定 6 worker。
1024、2048、4096 每档在同一进程内重复两次并由脚本强制比较 canonical SHA-256；5950
采用两个独立进程各运行一次，以避免一次 UI 轮询中断损失全部证据。

| 簇数 | 重复 | 总时长（秒） | 峰值工作集（bytes） | 最低可用物理内存（bytes） | 结果 |
|---:|---:|---:|---:|---:|---|
| 1,024 | 2 | 1457.508 | 1,267,990,528 | 4,368,650,240 | passed, 内部哈希一致 |
| 2,048 | 2 | 3033.848 | 1,446,445,056 | 5,083,262,976 | passed, 内部哈希一致 |
| 4,096 | 2 | 5395.644 | 1,595,031,552 | 1,968,652,288 | passed, 内部哈希一致 |
| 5,950 A | 1 | 4881.756 | 1,719,750,656 | 2,412,793,856 | passed |
| 5,950 B | 1 | 4123.693 | 1,688,313,856 | 4,130,287,616 | passed |

全规模 A/B 的 canonical 输出完全一致：

- `member_fit_diagnostics.csv`：`441fe48ef175b7ac9e0592ea57c8dc6cbc7cc313c7a728f03464324a083ed457`
- `member_forecast_bundle.csv`：`d8b1bfb770ba98181d4322630b8a747ea6db5b2a99bf6719b6fca65faf4dba7d`

五个完成记录合计执行 26,236 个 cluster fit，产生 2,256,296 行合成诊断输出和
18,050,368 行合成 forecast 输出；累计受测时长 18,892.449 秒。每个完成记录都观测到 6 个
worker、零采样错误、空 stderr 和空 worker exit-code 列表。

一个合并式 5,950 重复探针曾被 UI/工具轮询中断，只留下 0-byte 占位文件；它没有产生终态
JSON，因此不被计为计算成功或计算失败，也不进入证据清单。随后两个独立完成记录取代它。

## 因果门判定

| 必要条件 | 判定 | 理由 |
|---|---|---|
| 在不接触正式行/真值时自然复现底层失败 | 不满足 | 全部自然探针通过；注入失败不是自然复现 |
| 同一复现可在底层修复前失败、修复后通过 | 不满足 | 没有底层机制修复，也没有 failing-before |
| 证据唯一指向一个机制 | 不满足 | 冻结 cause/context 已不可恢复，A/B/C 仍均非零 |
| 最小修复与该机制存在可验证因果关系 | 不满足 | 仅有诊断修复，不能声称修复未知触发 |

故障注入与压力测试都是负证据：它们能排除普遍性解释，却不能证明 V2.10 的输入特异、瞬态
或未记录 worker 异常属于哪一支。用新的 seed 再运行只会改变数据，不会补回缺失的因果证据。

## 条件步骤处置

因果门失败后，批准计划中的条件步骤按预注册规则不执行：

- 不创建底层修复提交 C；
- 不创建 V2.11 协议提交 P；
- 不创建 V0.26/V2.11 实现提交 I；
- 不创建 V2.11 冻结提交 F 或正式运行包装器；
- 不执行 V2.11 preflight，不请求新的正式运行授权；
- 绝不启动 V2.11 `a1`。

诊断记录可由 `scripts/verify_v031_expanded_diagnostics.py` 离线复核。正式运行包装器的原始
PowerShell `$ErrorActionPreference='Stop'` 问题仅作为未来新研究链的工程约束保留：任何新包装器
必须显式读取 native process 的 `$LASTEXITCODE` 并分别保存 stdout/stderr；本研究链不再生成包装器。

## 验证状态

- 扩展记录离线验证：`V031_EXPANDED_DIAGNOSTICS_OK`；
- V2.31 故障矩阵、终端链和精确基数目标测试：`5 passed in 8.77s`；
- 两个 matplotlib 展示测试在已有 `gridse` 环境中：`2 passed in 2.77s`；
- 其余全仓测试：`1195 passed, 8 failed in 929.61s`。8 个失败均为仓库当前历史状态门：
  public-release manifest 已早于 V2.2-V2.10 新增文件，V2.5-V2.10 formal attester 要求 HEAD 等于
  各自冻结提交且 worktree 干净，以及 V2.10 “授权前根目录不存在”测试与已经完成的唯一正式
  attempt 互斥；没有失败落在本次诊断算法或验证器；
- 主虚拟环境直接收集完整套件时缺少可选 `matplotlib`，所以展示测试改用本机已有环境单独执行；
- 全仓 Ruff：`All checks passed!`；Python `compileall`：passed。
