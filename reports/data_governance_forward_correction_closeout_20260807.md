# 数据治理仅向前纠正审计谱系——2026-08-07

## 状态

成功。本报告保存 V1.2 至 V1.4 数据治理审计的公开谱系，但不发布其 ignored 本地 artifact 包。下表中每个 output manifest 哈希都根据实际 manifest bytes 重新计算；写入本报告前，也逐项独立核对了 manifest 记录与对应文件的 byte count 和 SHA-256。

| 审计 | 状态 | Output manifest SHA-256 | 条目数 | 作用 |
|---|---|---:|---:|---|
| V1.2 发布边界收口 | success | `e70092f6d2ab141188eb95424d9de3e4c005839d29b8c6d5274708e323b4ced0` | 6 | 恢复冻结的 `beep.py` 边界，并将 identity-only 接入移至非冻结模块。 |
| V1.3 证据同步 | success | `c4312fe1261ca232400fe610337f82f1b31bdeb1f4bda43f30b93b518de1fb4f` | 7 | 将已经完成的 MATR、NASA metadata-only 与发布边界证据同步到项目文档。 |
| V1.3.1 规范计数纠正 | success | `0bf5878c5a9659485638450ece52184cf0d3994007e7d2daad93d8718fa9dd50` | 4 | 纠正规范源文件数量，不修改原始 V1.3 记录。 |
| V1.4 最终预提交审查 | success | `ad59c6b981651b98f6aa4594f172d174694e1b29de0cfe41fdaf3e61f5bed7e1` | 24 | 复核 Git 范围、安全边界、五个历史 manifest、完整测试套件、发布政策与规范源文件哈希。 |

## 规范源文件哈希计数纠正

V1.3 记录了 301 个未变化路径，但其递归工作区扫描包含临时 `.pytest-tmp` 仓库副本中的 88 个 Python 文件。对这些记录路径而言，“哈希未变化”的判断仍然成立，但 301 不是规范项目源文件数量。

V1.3.1 将规范集合定义为 `git ls-files -- '*.py'` 的实际输出加 `release_manifest.json`：212 个 Git 索引 Python 文件和 1 个 release manifest，共 213 个文件。V1.3 的 before/after 两个表完整覆盖这 213 个路径；过滤后的两个表完全一致，独立复算的当前哈希也一致。规范集合中的 artifact、pytest 临时路径和其他临时路径数量均为 0。因此，后续源文件哈希引用必须使用 213；原始 301 路径记录继续作为历史证据保留。

## V1.4 验证

V1.4 收集到 914 项测试。被接受的完整运行自然结束：913 passed、1 个既有的 Windows symlink 能力限制 skip、0 failure、0 error、0 xfail。第一次调用使用了 168 个字符的 Windows basetemp，产生 42 个由路径长度导致的失败；其 JUnit 显示每个失败都引用该过长 basetemp。该次调用作为无效运行保留，没有隐藏。唯一一次允许的纠正调用改用 93 个字符的 ignored basetemp，随后通过。

全仓 Ruff、公开发布校验、Git diff 检查、保持 blocked 的 NASA 执行门、五个历史 output manifest 的复算，以及 213 文件规范哈希复核均通过。整个 V1.4 审查期间，cached diff 保持为空，Git status 逐字节不变。

## 证据边界

本谱系记录数据身份、metadata-only 访问、权利门禁、发布完整性和验证状态。它不发布 NASA 或 BEEP 原始数据，不证明 NASA 化学体系为 LFP，也不产生模型结果、精度提升、独立验证、真实电站验证或更高证据等级。NASA 数据集特定许可与公开发布汇总派生结果的权利仍未解决，因此 NASA 正式 `prepare`、`predict` 和 `score` 执行继续保持 blocked。
