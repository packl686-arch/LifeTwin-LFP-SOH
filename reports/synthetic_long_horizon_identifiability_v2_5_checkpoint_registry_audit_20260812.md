# LifeTwin V2.5 / V0.20 checkpoint 注册表契约审计（2026-08-12）

状态：`development_preimplementation`。本报告不是预注册、实现冻结或正式运行授权；
没有分配正式 attempt 或 seed，也没有调用 generator、runner 或 truth capability。

## 结论

V2.4 唯一正式尝试在预测承诺前终止。已确认的直接工程根因是 checkpoint
生产者与 firewall 消费者对 `input_byte_hashes` 文件名集合的定义漂移，而不是已登记
文件的 SHA-256 发生变化，也不是模型评分失败。

完整源码追踪显示漂移不只影响 center：

- center 生产者登记 10 个输入，V0.19 firewall 只允许 7 个；
- risk 生产者登记 11 个输入，V0.19 firewall 只允许 8 个；
- 两处消费者都漏掉同样 3 个已承诺文件：
  `actual_analysis_hash_ledger_commitment.json`、
  `generation_plan_commitment.json`、`truth_commitments.json`。

因此，只补 center 会让生命周期在下一次 risk reveal 再次因同类 schema 漂移失败。

## Producer / consumer 矩阵

| 契约位置 | 生产者实际集合 | V0.19 消费方式 | 审计结论 |
|---|---:|---|---|
| center checkpoint | 10 个精确键 | 7 键静态 allowlist + 逐文件 SHA | 漏 3 键，确定漂移 |
| risk checkpoint | 11 个精确键 | 8 键静态 allowlist + 逐文件 SHA | 漏 3 键，确定漂移 |
| training manifest | 复用 center/risk 的完整映射 | 与两个 checkpoint 映射逐项等值 | 集合随上游绑定，无独立漂移 |
| calibration mask | 单独承诺文件 | reveal 前验证路径与 ledger SHA | 一致 |
| calibration manifest | 14 个精确键 | 非空映射、逐文件 SHA，再与 model state 等值 | 字节绑定存在，但 V0.19 未静态拒绝未知文件名 |
| model state | 复用 center/risk/calibration 三个映射 | fresh read 后逐项等值 | 集合随上游绑定，无独立漂移 |
| model-state commitment | 8 个精确文件 | 同序 8 文件、bytes/rows/SHA | 一致 |
| prediction commitment | 单一 IO 模块的精确文件表 | 同一文件表校验顺序与 SHA | 一致 |
| prediction reveal prerequisites | plan、ledger、fit、checkpoint、mask、prediction commitment 分阶段累加 | fail-closed 验证 | center/risk allowlist 漂移会传递到后续 reveal |

V0.20 候选把 center、risk、calibration 三个输入集合放入一个只读共享映射。生产者
哈希函数和消费者验证函数只能从该映射取值；验证仍要求精确键集合、64 位小写
SHA-256、受控根目录的直接文件以及 fresh byte hash 一致。没有填值、删键、忽略未知
键或放宽 truth firewall。

## 结果前合成验证

唯一新增测试使用全新手写确定性小文件，不调用 RNG 或 generator，覆盖：

1. 合法 center 10 键通过；
2. 10 个必需键逐个缺失均拒绝；
3. 未知额外键和重命名键均拒绝；
4. 任一已登记文件字节变化导致 SHA 拒绝；
5. center checkpoint 验证通过后只调用一次 risk truth spy；
6. center、risk、calibration 的生产与消费共享同一注册表；
7. model-state commitment 和 reveal prerequisite 的后续静态集合保持一致。

首次测试在测试体执行前遇到宿主默认临时目录权限错误，同时发现 calibration 数量断言
误写为 15；实际源码集合为 14。更正断言并改用隔离临时目录后，新契约与最小
V0.19/V2.4 回归合计为 `49 passed`，其中新契约矩阵为 31 个用例。该基础设施记录
不属于科学终态或模型结果。

## 边界与下一步

- V0.19 文件保持不变；V0.20 当前只有共享契约模块，没有复制完整生命周期。
- 没有读取、抽样、统计或哈希任何 V2.4 正式大型输出或 sealed truth；没有读取 score、
  raw/第三方数据或旧正式模型输出。
- 没有重跑或续跑 V2.4，没有创建 a2、正式 attempt、seed 或四个正式根。
- 本阶段不产生精度、评分、成功或真实 LFP、海辰产品、储能电站及 15–25 年真实验证
  声称。
- 唯一建议下一步：独立审查本开发提交后，再决定是否另立结果前 V2.5 正式化与
  V0.20 lifecycle 接线阶段；在此之前不得正式运行。
