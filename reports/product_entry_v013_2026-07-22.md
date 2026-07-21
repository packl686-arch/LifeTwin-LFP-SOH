# V0.13：从研究证据到可操作入口

日期：2026-07-22
作者：Jincheng Liu

## 1. 本轮为什么不继续调模型

V0.12 已经表明，当前最主要的限制不是再降低同一公开数据上的几个小数，而是：

- 评委无法提交一个目标前缀并得到预测与拒绝决定；
- V4 只有整套回顾实验 runner，没有用户级推理契约；
- specialist 同路由校准不足，fallback 区间又对校准切分敏感；
- 独立长期 LFP 确认队列仍为 0。

因此 V0.13 不晋升新的均值公式。它冻结 V0.12 为研究基线，优先完成“输入短期数据、
自动路由、生成研究轨迹、判断区间是否可用、证据不足时拒绝”的真实入口。

## 2. 新增入口

### 2.1 前缀预测 API 与 CLI

公共 API：

```python
predict_calendar_prefix(
    request,
    *,
    reference_observations,
    config,
    schema=None,
) -> tuple[decision, forecast]
```

命令行：

```powershell
lifetwin calendar-prefix-predict `
  --request showcase/product_demo/naumann_t40_soc37_5_request.json `
  --output-dir artifacts/product-demo
```

该 CLI 面向源码或 editable checkout，默认资产按源码位置解析，因此可从仓库外目录调用；
当前版本不宣称提供脱离 Git 冻结资产的 standalone wheel 推理服务。

输出目录采用暂存后原子发布，且拒绝覆盖已有目录：

- `forecast.csv`：25 个未来时间坐标的研究均值、预测尺度、路由和诊断区间状态；
- `decision.json`：未来标签防火墙、路由、校准条件数、运营拒绝原因、支持边界和哈希。

### 2.2 零安装评审控制台

[Judge Console](https://packl686-arch.github.io/LifeTwin-LFP-SOH/judge-console/) 是一个自包含 HTML 文件，不访问外部资源，
也不声称实时推理。它只回放八个冻结证据文件中的三个预置案例：

1. `T40 · SOC37.5%` 通用回退；
2. `T40 · SOC12.5%` 激活专用路由；
3. Geisbauer 外部负迁移。

控制台明确显示证据等级、诊断区间与运营区间的区别、拒绝原因和最长支持跨度。
冻结展示包没有提供 Naumann 前十点的逐点 SOH 时，页面只画输入位置，不补造纵坐标。

## 3. 请求契约与未来标签防火墙

请求 Schema 位于
[`configs/inference/calendar_prefix_request.schema.json`](../configs/inference/calendar_prefix_request.schema.json)。
根对象和每一行均使用 `additionalProperties: false`。

- `prefix` 必须正好包含索引 0–9 的容量保持率；
- `forecast` 必须正好包含索引 10–34 的时间坐标；
- `forecast` 不能出现容量、SOH、损失、阻抗或任意未知字段；
- V1 锁定 Naumann `p=10` 采样网格，改变到 15 年或任意新网格会被拒绝；
- 统计单位固定为温度–SOC 条件均值轨迹，不允许冒充单电芯接口。

拟合参考状态前，代码把四个测试条件的全部行从参考数据中隔离。参考状态只接收七个训练
条件和六个校准条件；修改四个测试条件的所有未来容量不会改变训练状态、校准状态或 API
输出；即使把测试条件未来结果全部遮蔽，状态与输出也保持不变。训练域外、前缀缺少
局部工况切片支持、偏离联合参考轨迹或模型拟合失败时，接口都会得到 25 行空数值、`route=unavailable`
和机器可读拒绝原因，而不是返回 traceback 或伪造数值。

## 4. 两个黄金案例

| 请求 | 均值路由 | 80% 回顾诊断区间 | 运营签发 |
|---|---|---|---|
| `T40_SOC37.5` | `hierarchical_power_fallback` | 可用 | 拒绝 |
| `T40_SOC12.5` | `hierarchical_activation_residual` | 同路由校准不足，不可用 | 拒绝 |

fallback 示例在第 34 次检查的研究均值为 `90.228059%`，回顾诊断上下界为
`88.506707%–91.949412%`。这些数字只用于确定性回归，不是独立精度声明。两案例的
运营决定都保留 `calibration_evidence_not_independent` 和
`independent_long_term_evidence_missing`，上下界为 `null`。

## 5. 验证

新增测试覆盖：

- 两个黄金 API 和 CLI 路由；
- future outcome 字段注入；
- 少点、重复、乱序、NaN、错误统计单位和 15 年时间点；
- 域外、工况错配/极端但 Schema 合法的前缀和拟合故障均失败关闭且不泄露数值；
- 四个测试条件未来容量置换及完全遮蔽不变性；
- 真实 CLI 参数解析、默认参考路径、25 行输出与文件哈希；
- 独立哈希复算、重复确定性、原子发布和拒绝覆盖；
- 既有 V4 预测包字节哈希保持
  `ac0bd25154954a603eab6dbbbcfd3a1f281b4ec59d0ce5207a95c015a87c4d0c`；
- Judge Console 三案例、八个来源哈希、自包含与确定性重建。

这证明入口遵守现有证据边界，不证明模型已经通过独立长期验证。

## 6. 下一项可证伪研究

下一项科学实验不再从 Naumann outcome 选择新公式，而是预注册
`synthetic_long_horizon_identifiability_v1`：用纯幂律、双机制、饱和、早期激活和晚期
knee 等已知真值族，测试短前缀能否识别长期结构，并比较“模型结构分歧”是否能成为有效
拒绝信号。核心成败条件包括：在 50% 发行率下，分歧拒绝必须较随机拒绝降低至少 30%
灾难性错误；构造 730 天内几乎相同、25 年相差至少 5 pp 的匹配前缀反例。通过也只能称为
合成机制压力测试，不能升级为 LFP、海辰或 15–25 年产品证据。
