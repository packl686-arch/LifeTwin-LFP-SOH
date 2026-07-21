# 校准稳健性、外部负迁移与长期数据资格报告

日期：2026-07-21

提交人：Jincheng Liu

状态：两项回顾性稳健性审计完成；独立长期确认仍未完成

## 1. 本轮结论

本轮没有继续在同一批数据上增加模型自由度，而是回答两个更紧迫的问题：V4 的
小样本区间是否依赖某一次校准切分，以及 Geisbauer 外部负结果是否由少数电芯造成。
结论并不“漂亮”，但更可信：

1. V4 fallback 路线的 80% 诊断区间对校准条件选择明显敏感；specialist 路线在
   所有切分中都没有足够样本形成有限分位数。
2. Geisbauer 上候选平均误差略差于平方根比较器，但逐电芯结果接近五五开，且
   leave-one-cell-out 会改变均值方向。当前只能判定存在负迁移风险，不能宣称稳定
   优越或稳定劣于比较器。
3. 一手来源复核没有找到一份同时满足许可明确、LFP/石墨、可分离日历老化、独立
   电芯轨迹、至少 730 天且结局未暴露的公开数据集。长期确认继续保持阻塞，而不是
   用短期或混合老化数据替代。

## 2. V4 校准切分敏感性

审计固定原 V4 的 7 个训练条件和全部均值模型状态，只在其余 10 个条件中枚举所有
`6 calibration + 4 evaluation` 划分，共 `C(10,6)=210` 个。每个条件仍以完整未来
轨迹的最大标准化误差作为一个校准分数，不能把 25 个时间点当成 25 个独立样本。

### 2.1 有限样本事实

| 路线 | 每个划分的校准轨迹数 | 80% | 90% | 95% |
|---|---:|:---:|:---:|:---:|
| fallback | 4-6 | 210/210 可计算 | 全部不可用 | 全部不可用 |
| specialist | 0-2 | 全部不可用 | 全部不可用 | 全部不可用 |

有限样本 higher-order quantile 在 80%/90%/95% 下至少分别需要 4/9/19 条同路线校准
轨迹。模型没有把不可计算的分位数替换成经验近似，也没有把不同路线混在一起凑样本。

### 2.2 结果

- fallback 80% 乘数在 210 个划分中为 `0.9243-2.1698`；平均诊断区间宽度为
  `0.7575-2.1251 pp`。
- 原切分乘数为 `2.16984`。LOCO 删除 `T40_SOC50` 后降为 `1.48407`，平均宽度从
  `1.68415 pp` 降为 `1.15188 pp`；其余删除不改变主导阶统计量。
- 210 个重叠划分产生 672 个 fallback `condition x partition` 评估实例，其中
  552 个整条轨迹同时覆盖，即 `82.14%`；但只有 126/210 个划分覆盖其各自全部
  fallback 评估轨迹，即 `60%`。

`552/672` 不是 672 次独立试验，`126/210` 也不是 210 次独立复现。它们重复使用
同一组已查看的 Naumann 条件轨迹，只能描述切分敏感性，不能建立正式覆盖保证。
工程含义是：当前 fallback 区间宽度受一条高误差校准轨迹支配，而 specialist 是
结构性欠校准；运营区间仍必须拒发。

## 3. Geisbauer 负结果稳健性

该队列包含 15 个独立物理电芯，只覆盖 60 °C、20%/50%/100% SOC 和 120 天。
候选因前缀不足在 15/15 个电芯上都等于层次幂律 fallback，激活 specialist 从未被
测试。本轮以物理电芯为配对单位，增加 exact sign、exhaustive sign-flip、SOC/时间
分层和 leave-one-cell-out 诊断。

| 指标 | 结果 |
|---|---:|
| 候选平均 IAE | 3.97345 pp |
| 平方根比较器平均 IAE | 3.88521 pp |
| 候选减比较器的平均配对差 | +0.08824 pp |
| 中位配对差 | -0.00219 pp |
| 数值零阈值下候选更好 / 更差电芯 | 8 / 7 |
| 数值零阈值下 exact sign 双侧诊断 | 1.0000 |
| exhaustive mean sign-flip 双侧诊断 | 0.5750 |

`8/7` 使用 `1e-12 pp` 数值零容差，只表示差值符号，不表示工程实质差异。作为
事后透明度检查，等效界设为 `0.01 pp` 时为 5 better / 7 worse / 3 equivalent；
设为 `0.05` 或 `0.10 pp` 时均为 5 / 4 / 6。这些等效界不是预先建立的测量或业务
门槛，不能据此挑选有利结论。

名义诊断是在查看 v0.11 结果后设计，未做多重性校正，只能用于探索。LOCO 后
平均差范围为 `-0.00990` 至 `+0.14932 pp`；在 15 个彼此重叠的单电芯删除场景中，
有 2 个场景会把总体方向从“候选更差”改为“候选更好”，并不是一次同时删除两个
电芯。SOC 均值分别为：20% `-0.21596 pp`、50%
`-0.13371 pp`、100% `+0.61438 pp`；负迁移主要集中在高 SOC，但 20% SOC 的均值
和中位数方向也冲突。

因此最准确的表述不是“模型已被外部数据否定”，而是：**层次 fallback 在这个短期
高温队列上没有稳定胜过简单目标前缀模型，且高 SOC 存在明显域迁移风险。** 这仍是
120 天 accelerated external stress check，不是长期验证。

## 4. 独立长期数据资格

长期数据登记表将“公开可见”和“允许复用”分开，并将端点、轨迹、锁定外部验证和
前瞻运营证据分级。当前最重要的候选如下：

| 候选 | 可提供的证据 | 当前阻塞 |
|---|---|---|
| [Lam/Joule](https://doi.org/10.1016/j.joule.2024.11.013) | 最长 13 年、含 LFP 的逐电芯长期轨迹 | OSF 数据与 GitHub 代码许可未明确；项目已看过汇总结构，获许可后也只能做锁定回顾性复现 |
| [Vachenauer/TUM](https://www.sciencedirect.com/science/article/pii/S0378775325016155) | 100 个 26650 LFP/C 电芯在 **6 °C、50% SOC** 连续存放十年的端点，论文报告 96%-98% 容量保持 | 原始逐电芯数据仅按请求提供；十年中无中间检查点，最多验证端点，不能验证 dynamic landmark |
| [Yagci/Offenburg](https://www.sciencedirect.com/science/article/pii/S2352152X25014872) | 180 Ah 储能 LFP、约 850 天，产品形态最接近命题 | 逐电芯数据和数据许可需向作者申请 |
| [Sui/Aalborg](https://doi.org/10.3390/en14061732) | 15 个 LFP/石墨电芯、27-43 个月、约月度检查 | 机器可读逐电芯轨迹未公开 |
| [Aeppli/Empa](https://www.dora.lib4ri.ch/empa/islandora/object/empa%3A41733/datastream/PDF/Aeppli-2025-Aging_behavior_of_LiFePO4-based_battery-%28published_version%29.pdf) | 9 个 Second-Life 电芯、总年龄超过 14 年、最高 9600 cycles | 未知 First-Life 与 Second-Life 循环/日历混合，只能做迁移压力测试，不能做日历确认 |

Vachenauer 的 `6 °C` 是存储温度，不是 `6C` 充放电倍率；Aeppli 的主实验是 9 个
selected cells，不是原车单包的 8 串电芯。这两处容易造成领域误读，已在机器可读
登记表中单独字段化。

## 5. 冻结协议如何防止“申请到数据后再改规则”

长期验证模板要求在读取目标结果前记录：数据授权文本、仓库版本、文件字节数与
SHA-256、物理电芯 ID、工况与时间单位、模型/适配器/评分器提交哈希、目标结局暴露
日志、分区、landmark、比较器和成功/失败门槛。核心决策规则包括：

1. 物理电芯不可跨训练、校准、测试分区；评分以电芯轨迹为单位并按条件聚类。
2. 目标未来结局不能影响均值预测、路由、区间或最强比较器选择。
3. 点预测必须同时通过平均改善、相对改善、改善电芯比例、最差条件回退和聚类随机化
   诊断；样本不足的结果记为 inconclusive，不记为成功。
4. 区间按实际模型路线校准；同路线样本、独立条件、时间支持或域支持不足即拒发。
5. 已公开并被项目看过的结果永远不能重新命名为 outcome-blind confirmation。

机器可读材料见[数据集登记表](../configs/validation/long_term_lfp_dataset_registry.json)、
[协议模板](../configs/validation/independent_long_term_lfp_protocol.template.json)、
[JSON Schema](../configs/validation/independent_long_term_lfp_protocol.schema.json)和
[跨字段验证器](../src/lifetwin/validation/long_term_protocol.py)。Schema 固定资格与成功阈值，
验证器再复算四个分区的非空、两两不相交以及声明计数，二者共同通过才可冻结或执行。

## 6. 对下一版模型的约束

当前最值得研究的不是再叠加黑盒，而是检验长期时间项是否需要更强的机理约束，例如
对幂律指数做收缩、加入可审计的饱和项，或把短期与长期机制分段。十年端点的公开
聚合结果已经属于“看过的开发线索”，不能一边用来调参、一边称作独立验证。正确顺序
应是：在开发分区比较候选结构，冻结模型和端点评分器，再用获得许可的逐电芯端点或
另一支未暴露队列评价。

本轮没有晋升新的均值模型。v0.12 的进步是把“区间可能很敏感”“外部结果可能由
少数样本驱动”“长期数据可能不够格”变成可运行、可失败、可由评委复核的审计规则。

## 7. 机器可读证据

- [V4 校准稳健性结果](../showcase/evidence_v012/v4_calibration_robustness/result.json)
- [V4 全枚举汇总](../showcase/evidence_v012/v4_calibration_robustness/sensitivity_summary.csv)
- [V4 LOCO 路由指标](../showcase/evidence_v012/v4_calibration_robustness/loco_route_metrics.csv)
- [Geisbauer 稳健性结果](../showcase/evidence_v012/geisbauer_robustness/result.json)
- [Geisbauer 逐电芯配对差](../showcase/evidence_v012/geisbauer_robustness/cell_paired_deltas.csv)
- [Geisbauer leave-one-cell-out](../showcase/evidence_v012/geisbauer_robustness/leave_one_cell_out.csv)
- [长期 LFP 数据集资格说明](../docs/long_term_lfp_dataset_registry.md)
- [审计总图](../docs/assets/v012_robustness.png)
