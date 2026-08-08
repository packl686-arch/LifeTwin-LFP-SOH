# LifeTwin 数据资产接入 V1.1 纠正——2026-08-06

## 状态

V1.1 是仅向前的数据治理完整性纠正。它不改变冻结的数据集角色、NASA 分区、前缀、指标、配置哈希或任何既有科研结果。

## MATR 纠正

V1 清单计数继续作为数据存在性事实成立：140 个结构化 JSON 文件、135 个唯一 barcode、5 个额外 segment、0 个身份冲突。根目录 5 个 MAT 表示新增的物理电芯仍为 0。

V1 在收集身份字段时使用了会物化 JSON summary 的 parser。因此，即使这些值没有用于映射或建模，V1 也 **没有** 证明其身份层避免接触含未来标签的 summary 值。V1.1 用 identity-field whitelist reader 替换该路径：在 summary object 之前停止，只返回身份与来源字段，不调用 summary parser，并报告物化的结局值为 0。只有 V1.1 对抗测试和定向 140 文件身份审计通过后，才接受这一边界。

## NASA 纠正

NASA 正式协议继续保持 blocked，因为权利门仍为 false。V1.1 将该门禁放入每一个公开 `prepare`、`predict` 和 `score` library entry point；command-line gate 会在打开输入前返回机器可读的 blocked 结果；合成测试中还实现了 append-only 的单次评分 receipt。没有打开真实 NASA 结局，也没有产生真实评分或模型精度结果。

## 输出完整性纠正

审计和评分输出目录现在只允许发布到新目录。既有输出目录会被拒绝；关键 child-audit 状态采用 allowlist；生成文件由记录 byte count 和 SHA-256 的 output manifest 覆盖。合成 locked-test 失败尝试会保留 append-only 的 failed receipt，且不能使用相同 protocol、prediction 和 future-label identity 自动重试。

## 历史边界

提交 `9e2884a82710c2d64ca9b4d412acca5030a21986` 的范围和提交行为偏离了其原始指令。V1.1 不重写、不回退，也不隐藏这段历史，只实施仅向前纠正。该纠正不增加模型精度证据，也不改变 V0.14 的 `failure`、V0.15 的 `inconclusive_not_success` 或 V0.16/V2.1 仅实现冻结的状态。

## V1.2 冻结发布边界

V1.2 将 `src/lifetwin/data/beep.py` 精确恢复为冻结 SHA-256 `555d47dd4c3bc3310667cbdb9ba01922e4b34b52720035b76fb3932bd3049c11`。identity-only 实现迁移到非冻结模块 `src/lifetwin/data/beep_identity.py`；公开发布校验在不改变冻结 hash map、release ID、版本或日期的情况下通过。

同步后的 NASA 治理证据继续是一个独立的零结局元数据对象：38 个 MAT 文件、10 个 README/TXT 文件、34 个由文件名推导的唯一 `Bxxxx` 身份，以及 4 组 byte count/SHA-256 完全相同的重复表示。MAT/容量值读取、训练、预测、评分和 SNL 内容读取均为 0；NASA 正式执行继续保持 blocked。这不产生新的模型结果、LFP 验证、独立测试集或更高证据等级。
