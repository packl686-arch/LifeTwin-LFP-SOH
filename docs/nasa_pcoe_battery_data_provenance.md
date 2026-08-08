# NASA PCoE Battery Data：来源与使用边界

审计日期：2026-08-03

状态：`auxiliary_benchmark_only`；本仓库不包含本地 4 个 CSV 文件。

## 来源记录

上游数据集是 NASA Ames Prognostics Center of Excellence（PCoE）的 **Li-ion Battery Aging Datasets** 集合，目录标识为 `DASHLINK_133`。

- Data.gov 官方目录：
  <https://catalog.data.gov/dataset/li-ion-battery-aging-datasets>
- NASA Open Data Portal 记录：
  <https://data.nasa.gov/dataset/li-ion-battery-aging-datasets>
- 目录记录的旧版 DASHlink 页面：
  <https://c3.nasa.gov/dashlink/resources/133/>
- 相关原始预测研究论文：Saha 与 Goebel，*Uncertainty Management for Diagnostics and Prognostics of Batteries using Bayesian Techniques*：
  <https://ntrs.nasa.gov/citations/20130010482>

官方目录描述的是商业化、2 Ah、18650 尺寸的锂离子电芯，按照充电、放电和电化学阻抗谱工况运行。目录报告采集频率约为 10 Hz，寿命终点判据是额定容量衰减 30%，即从 2 Ah 降至 1.4 Ah。为加速深度放电老化，部分放电截止电压被有意设置在 OEM 建议的 2.7 V 以下。

FY08Q4 文件附带的 README 也保存在第三方镜像中：
<https://labinfo.ing.he-arc.ch/gitlab/ticc/16TICc19/nasa-battery-dataset/-/blame/bac7f5812a70d05d0a79e3bca578efc80fb5b59c/data/BatteryAgingARC-FY08Q4/README.txt>。
该 README 记录：实验在室温下进行；先以 1.5 A 恒流充电至 4.2 V，再恒压充电至电流降为 20 mA；放电采用 2 A 恒流。B0005、B0006、B0007 与 B0018 报告的放电截止电压分别为 2.7 V、2.5 V、2.2 V 与 2.5 V。这些逐电芯数值来自随附 README，不是 LifeTwin 的新测量。

## 本地 CSV 获取情况

已检查的本地文件包只有 `B0005.csv`、`B0006.csv`、`B0007.csv` 和 `B0018.csv` 4 个文件，其中没有 README、来源 URL、转换脚本、转换版本、许可证文件或上游文件哈希。

CSV header 为 `type,temp,time,data`，每个 `data` 值都包含由测量数组构成的 serialized mapping。上游文档描述的表示形式是 `B0005.mat` 等 MATLAB 文件及嵌套 `cycle` 结构。因此，这些 CSV 必须表述为 **未经核验的 NASA 数据集第三方转换版本**，不能表述为 NASA 制作的 CSV。文件名相同、内容看似合理，并不能证明它们与上游 MAT 文件在 bytes 或语义上等价。

下列哈希只标识审计日期当天检查的本地文件，不是 NASA 官方 checksum。

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| `B0005.csv` | 49,218,466 | `d74b6352fde77fcb55543df48180914ca92d56d36320d70e0ebfcd57696b6105` |
| `B0006.csv` | 49,410,002 | `d544bdcfdf053861cc96736dd25b91a7de99fa91d1a6877aba3e05bb6a5d97c9` |
| `B0007.csv` | 49,943,430 | `251b6a074702fc07991db86c1760db843db967d6648f01f4270337d94461fd80` |
| `B0018.csv` | 26,358,323 | `9ce1516d47b3cb2a4a03d9a6c671fdbc6e703468795cbe5ee772605989ac011f` |

上游 MAT 文件目前已在本地存在，但文件存在并不授权评分，也不能证明其与这 4 个 CSV 语义等价。在开展任何评分实验前，项目仍需完成数据集特定权利确认、协议审查、冻结的确定性转换、完整 CSV/MAT crosswalk，以及新的明确执行授权。在此之前，禁止比较操作数、时间戳、容量、截止行为或抽样数组，也禁止正式评分。

## 2026-08-06 解压 MAT 元数据接入

本次接入与前述 4 个第三方 CSV 是不同的证据对象。解压快照包含 6 个顶层目录、38 个 MAT 文件和 10 个 README/TXT 文件。按文件名身份与 byte count/SHA-256 重复规则，共得到 34 个唯一 `Bxxxx` 物理电池 ID 和 4 组完全相同的重复表示：B0025、B0026、B0027、B0028。34 是该快照中的身份数量，不表示 34 个独立、同分布或合格测试电芯。

接入只读取文件元数据、流式哈希、128-byte MATLAB header、README/TXT 文本，以及通过 `scipy.io.whosmat` 得到的顶层变量名、MATLAB 类型和 shape。它没有加载 MAT array 或容量值，也没有准备 prefix、预测、训练或评分；所有此类访问和执行计数均为 0。SNL 内容访问也为 0。

README 文件暴露了停止阈值、实验异常以及部分结局/协议结构。因此，暴露角色固定为 `development_only_outcomes_and_protocol_structure_exposed`，不能称为 outcome-blind 或独立确认。尚未证明 4 个 CSV 文件与 38 个 MAT 文件语义等价。

持有文件和公开访问不能解决数据集特定许可、化学体系、再分发或汇总派生结果发布权。化学体系继续记录为 unspecified lithium-ion，而不是 LFP 证据；NASA 正式执行门继续关闭。任何正式评分前，仍需解决权利问题、审查协议、冻结转换，并获得新的明确授权。

## 化学体系与标签注意事项

官方目录只将这些电芯描述为商业化 lithium-ion 18650 电芯，没有权威说明正极或负极化学体系。二手文献对化学体系说法不一致，因此在没有制造商或 NASA 一手文件的情况下，LifeTwin 不把该队列标记为 LFP、LCO、NCA 或其他具体化学体系。

4 个电芯还采用不同的放电截止电压，因此其容量值依赖具体协议，不能在相同测量边界的假设下直接合并。后续 benchmark 必须明确保留电芯身份与截止协议；适用时在电芯内归一化；分区必须采用 cell-held-out，而不是随机 row split。

## 许可证状态

官方目录将数据访问级别标为 `public`，并明确把预测算法开发写作预定用途，但没有提供数据集特定的许可证标识或许可证 URL。当前 NASA portal 记录同样没有解决该集合的许可问题。

NASA 的一般科学数据指南区分 NASA 主导的任务数据与其他数据，并提醒用户：当 NASA 可能不是原始权利人时，需要自行核验来源权利：
<https://science.data.nasa.gov/about/license>。NASA STI 条款还说明，美国政府雇员作品通常不受美国版权保护，但由 NASA 公开发布的承包商或第三方材料仍可能受版权保护：
<https://sti.nasa.gov/disclaimers/>。

因此，LifeTwin 不推断该数据集或第三方 CSV 转换版本具有 CC0、CC BY、商业使用权或再分发权。项目可以在保留归属信息的前提下，为私下研究、parser 测试与数据质量评估检查本地文件；但不得提交或再分发 CSV，不得把它们称为 NASA 官方 CSV，也不得暗示公开访问已经解决所有下游权利。发布或商业复用需要有记录的权利审查；本说明是项目政策，不是法律意见。

官方目录将 Dawn McIntosh（`dawn.m.mcintosh@nasa.gov`）列为数据集联系人员，可用于进一步确认。

## 允许的证据角色

完成来源交叉核对后，该队列只能作为辅助循环老化 benchmark，用于：

- parser 与 schema 验证；
- 因果 early-prefix SOH 或 RUL 评估；
- leave-one-cell-out 迁移检查；
- 处理测量噪声、局部容量恢复与非单调性；
- 测试域偏移下的不确定性扩张和拒绝行为。

它不能支持 LFP 特定精度、海辰产品性能、储能电站部署就绪、日历老化有效性或 15–25 年预测精度等宣称。该数据集只有 4 个小型圆柱电芯、短期加速循环老化历史、固定实验室工况，也没有多年储能电站日历轨迹。它缺少支持这些宣称所需的温度、SOC window、低 C-rate、rest、维护、pack heterogeneity 和运行环境信息。

批准使用的公开表述为：

> NASA PCoE 数据只用于跨电芯循环老化压力测试，不能验证 LifeTwin 的 LFP 性能或 15–25 年储能电站寿命宣称。

## 本地复现

将 4 个文件放在 ignored 本地目录中，例如 `data/raw/nasa_pcoe/`。不要将其复制到 `data/external/`，也不要提交。安装项目依赖后运行：

```powershell
$env:PYTHONPATH='src'
python scripts/run_nasa_pcoe_benchmark.py prepare data/raw/nasa_pcoe

python scripts/run_nasa_pcoe_benchmark.py predict `
  artifacts/nasa-prefix-loco-v1/cycles.csv

python scripts/run_nasa_pcoe_benchmark.py score `
  artifacts/nasa-prefix-loco-v1/cycles.csv `
  artifacts/nasa-prefix-loco-v1/predictions.csv `
  artifacts/nasa-prefix-loco-v1/prediction_manifest.json
```

接入程序会拒绝 byte size 或 SHA-256 身份不同于 4 个已审计转换文件的输入。冻结 benchmark 使用 cycle 20、40、60、100 的目标前缀，并只在共同的 cycle-132 支持范围内评分。预测函数只接受物理截断的 prefix table；suffix outcomes 只能由独立 scorer 关联。

如需运行 V1 之后的 dynamic-gate 开发实验，执行：

```powershell
$env:PYTHONPATH='src'
python scripts/run_nasa_dynamic_gate_v2.py run-source data/raw/nasa_pcoe `
  --output-directory artifacts/nasa-dynamic-gate-v2
```

V2 adapter 只提取 within-cycle 放电曲线测量：电流积分、能量积分、电压阈值时间、3.8–3.4 V 持续时间、0.5 Ah 与 1.0 Ah 处的电压、平均 dV/dQ，以及温升。固定 bundle 中的 636 条放电记录全部覆盖两个公共 window。这些特征不能解决上述化学体系、转换来源、许可或域偏移限制。结果与负面发现记录在 `reports/nasa_dynamic_gate_v2_development_2026-08-03.md`。
