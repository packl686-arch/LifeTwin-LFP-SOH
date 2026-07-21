# LifeTwin Judge Console

零安装、单文件的中文评审证据控制台。可从
[GitHub Pages](https://packl686-arch.github.io/LifeTwin-LFP-SOH/judge-console/)
在线打开，也可直接打开仓库内的 `index.html`；页面不请求外部资源，也不执行实时模型推理。

## 重建

在仓库根目录运行：

```powershell
python showcase/build_judge_console.py
python showcase/build_judge_console.py --check
```

生成器只读取 `showcase/evidence_v011/` 与 `showcase/evidence_v012/` 中已冻结的公开 CSV/JSON，并将所用文件的 SHA-256 写入页面数据。重复运行应生成字节完全一致的 `index.html`。

## 三个预置案例

1. `T40_SOC37.5`：通用幂律回退；80% 诊断区间可显示，但因证据非独立且缺少长期确认而拒绝运营签发。
2. `T40_SOC12.5`：激活残差专用路由；同路由校准条件不足，主动拒绝区间与运营签发。
3. Geisbauer 外部压力筛查：候选方法平均配对误差增加 `0.088 pp`，展示负迁移而非隐藏失败。

Naumann 冻结展示包没有再分发前 10 点的逐点 SOH 数值，因此页面只显示十个输入位置与前缀截止日，不伪造纵坐标。揭盲开关仅显示可从两个冻结证据表交叉核对的最终真值点。

## 结论边界

这是回顾性证据回放，不是实时推理服务。Naumann 结果是最长 885 天的公开 LFP 条件均值诊断；Geisbauer 是 60°C、120 天的 15 电芯探索性压力筛查。二者都不是海辰产品或储能电站验证，也不能支持 15–25 年寿命结论。
