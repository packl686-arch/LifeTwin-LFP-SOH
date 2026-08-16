# LifeTwin V2.10 预测前终止根因审计

日期：2026-08-15  
状态：`development_only / v2_10_final_unchanged / no_new_formal_attempt`

## 结论

V2.10 唯一正式 attempt `v030-formal-20260814-a1` 的冻结终态保持不变：
`terminal_pre_prediction / unclassified_terminal_not_success / UNKNOWN_PRE_PREDICTION_EXCEPTION`。
它没有 prediction commitment、没有评分，`opened_truth_files` 为空。

本次审计确认了一个可由冻结源码和非正式 fixture 独立复现的工程缺陷：
**V2.10 终端诊断只序列化最外层异常的 traceback，完全丢弃 `__cause__` 和
`__context__` 链。** 同时，预测边界把 `V024FitError` 与 `V024IOError` 合并包装成同一个
`V024PredictionError`。因此，冻结证据无法区分下列底层分支：

1. 新鲜生成 capability 的 IO 或字节一致性拒绝；
2. 六进程拟合器中的模型、数值或输出契约异常；
3. worker pool 或 worker process 的异常终止。

这解释了为什么 V2.10 只留下三帧外层 traceback，也解释了为什么不能从现有证据恢复计算触发器。
它是“底层触发不可诊断”的确定根因，但不是对底层计算异常类型的事后猜测。

## 冻结证据

- 冻结提交：`3b0e4008bad4c82fe5741391c65e004424e19f77`
- 正式 attempt：`v030-formal-20260814-a1`
- 拟合阶段开始：`2026-08-14T15:20:30Z`
- 拟合阶段失败：`2026-08-14T15:27:07Z`
- 失败窗口：397 秒
- 最后完成阶段：`actual_analysis_hash_ledger_committed`
- 待完成阶段：`label_free_fit_committed`
- 外层异常：`V024PredictionError`
- 终端记录 SHA-256：`e1b826f72c5c335023a5dffef07158bb88be17cb6eb9ae4299d9a5061700bb05`
- 终端 manifest SHA-256：`493b7254d4cb060f470b67e70db15fc8da8efbc1d6a394446bd91950e67bd83d`
- 证据包 SHA-256：`73602810a3fefa18aedf293a0c864e0369f4abfd15a252b80fe1da27664d9e40`
- 冻结仓库 bundle SHA-256：`47a3dd4789b39562c791301771bec2ca4faa401c83b6547cdca360e92d9ec6b1`

冻结 traceback 仅包含：

1. `calendar_long_horizon_v019_runner.py::run_formal_attempt`
2. `calendar_long_horizon_v019_runner.py::_fit_structure_stage`
3. `calendar_long_horizon_v019_prediction.py::fit_verified_generation_bundle_v024`

冻结源码中的触发链闭合如下：

- `calendar_long_horizon_v019_prediction.py:133-136` 同时捕获 `V024FitError` 和
  `V024IOError`，再抛出相同的 `V024PredictionError`；
- `calendar_long_horizon_v019_fit.py:137-143` 把 inherited fit 的
  `V015PredictionError` 再包装成 `V024FitError`；
- `calendar_long_horizon_v015_prediction.py:193-256` 把 pool 启动、worker future、
  worker 输出和 pool shutdown 边界收敛到 `V015PredictionError`；
- 冻结版 `sanitized_structural_traceback` 只遍历最外层 `error.__traceback__`，没有遍历
  `__cause__` 或 `__context__`。

## 无真值复现矩阵

所有复现均使用手工或诊断专用 PCG64DXSM 流生成的 prefix-only fixture。没有调用正式
generator，没有使用正式 seed，没有读取 V2.10 正式行、密封真值、fit 输出、预测输出或评分。
真实 Windows `spawn` 与六 worker 路径在沙箱外执行，因为沙箱会以 `[WinError 5]` 拒绝
`multiprocessing.Pipe`。

| 复现 | 结果 | 时长 | 规模 A | 规模 B |
|---|---:|---:|---:|---:|
| 既有 one-worker / six-worker canonical equality 与 worker failure 回归 | 2 passed | 13.48 s | 小型 fixture | 小型 fixture |
| 96 簇 structured probe | passed | 110.707 s | 8,256 | 66,048 |
| 384 簇 randomized probe | passed | 246.916 s | 33,024 | 264,192 |
| 640 簇 randomized probe | passed | 415.886 s | 55,040 | 440,320 |
| 5,950 簇 exact-cardinality capability 提取与 fit adapter | passed | 3.12 s | 71,400 prefix | 47,600 coordinates |

640 簇 probe 已跨过 V2.10 的 397 秒失败窗口。它不能证明任意输入均成功，但足以否定
“Windows 六进程路径必然失败”以及“仅因运行到相同时长必然失败”。
5,950 簇 fixture 还通过了 fresh-bundle 物理 membership、逐文件哈希、ledger 前缀、
内存 frame canonical hash、deep-copy 提取、正式基数校验和 V2.10-to-V2 fit identity adapter。
这排除了 capability 提取和 adapter 对任意合法精确基数输入必然失败的解释。

可复现脚本：`scripts/diagnose_v210_fit_spawn.py`。

输出哈希：

| 复现 | member fit diagnostics | member forecast bundle |
|---|---|---|
| 96 structured | `adf228639e62a4d370afcbe7d192b59fc033d501a91dc62de16ba798a11cb125` | `5d1edf8e8b31f6001490e77cd9bd916eaef49e2298c3ee13151c862f9278c1eb` |
| 384 randomized | `161bbe1336ad9545fc75fe0333002be0b7a4d620e7852f9bacf51f0b5d6dc57b` | `bc5ceee47e839e49766cd5de3db3c960b6fd92c46764070bf397f1641e60e5b8` |
| 640 randomized | `24397573505ff9a577ad3ba901ab8a6eaca1031811883276d9ec4c47901be37b` | `dd102b61d97e2d2502004f56418d2c9c6cb3d8da8161f1f5c9e76f3e75eacce4` |

同一时间窗的 Windows System 日志未发现 Resource-Exhaustion 或 Kernel-Power 事件；
Application 日志未发现 Python、Application Error 或 Windows Error Reporting 事件。
用户 CrashDumps、WER ReportArchive 与 ReportQueue 也没有该时间窗的文件。
该负证据不能排除未记录的 worker 异常，但不支持已记录的系统崩溃或 OOM 解释。

## 根因树

### A. capability / IO 拒绝

`_extract_fresh_generation_frames_for_formal_fit_v024` 会重新验证物理 root、文件哈希、ledger
前缀和内存 frame。任一失败均为 `V024IOError`，随后在共同预测边界被擦除为
`V024PredictionError`。

全新 5,950 簇 fixture 已通过同一 capability 重验证与 adapter，因此通用逻辑错误的可信度较低；
仍不能排除 V2.10 字节特异的拒绝、并发改变或未记录的瞬态错误。

### B. inherited fit 拒绝

任一 worker future、拟合输出或 worker pool 异常可先成为 `V015PredictionError`，再成为
`V024FitError`，最后成为相同的 `V024PredictionError`。现有终端记录没有保存这两层 cause。

### C. 外部进程终止

worker 被平台终止通常会在 parent 中表现为 process-pool 异常，再进入 B。系统事件与跨时长
压力测试降低了该分支的可信度，但不能把它降为零。

冻结证据不能在 A、B、C 之间继续分叉。任何声称精确到某个模型、某个 cluster、OOM 或
某个 SciPy 异常的结论都超出现有证据。

## 最小修复

开发分支只修改终端诊断，不修改科学模型、优化器、特征、阈值、分区、端点、成功门或 seed：

1. structural traceback schema 从 `1.0.0` 升为 `1.1.0`；
2. 保存最多 16 层 `outer/cause/context` 异常链；
3. 每层只保存安全异常类名和结构化 frame，不保存异常原文、locals 或数据值；
4. 链检测循环并显式记录 `exception_chain_truncated`；
5. 继续拒绝绝对路径、地址、64 位十六进制 digest 与非 allowlisted 结构；
6. 嵌套 `V024IOError` 被分类为 proven-integrity artifact/capability failure；
7. 新增链保留、敏感文本排除、嵌套 IO 分类和循环有界回归。

提交后目标回归为 `35 passed in 5.41s`；完整 V0.15 prediction/spawn 回归为
`15 passed in 15.67s`。全仓 Ruff 与 Python compileall 均通过。

## V2.11 决策门

当前证据足以合并诊断修复，但不足以声称 V2.10 的底层计算触发已修复。直接用新 seed 建立
V2.11 正式 attempt 会成为数据依赖的替代运行，并可能通过换样本绕过未知缺陷。因此本审计
不建立 V2.11 seed、attempt、四根、预注册、实现冻结或正式 bundle，也不请求正式运行授权。

允许继续进入 V2.11 的必要条件是：在不读取 V2.10 正式行与密封真值的前提下，获得能够把
A、B、C 至少收敛到一个可修复机制的新增独立证据；否则 V2.10 保持最终终态，研究链在此停止。
