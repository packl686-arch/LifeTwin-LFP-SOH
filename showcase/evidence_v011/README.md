# V0.11 三条新增证据链

本目录公开三组可由 runner 重建的小型证据包。它们回答不同问题，不能互相替代。

| 目录 | 回答的问题 | 当前结论 |
|---|---|---|
| `landmark/` | 早期观测增长到何时，V3 才在固定未来窗口出现一致信号？ | `p=10` 是回顾性信号点；由于重复使用已查看的 Naumann 结果，确认点仍为空 |
| `v4/` | 机理层次模型加入有界残差和保守区间后，何时可以发区间？ | 只得到 3 条 fallback 路由的 80% 诊断区间；90%/95% 与 specialist 路由样本不足，运营区间全部拒发 |
| `geisbauer/` | 冻结方法能否迁移到独立电芯级 LFP 队列？ | 在 120 天、60 C 应力筛查中，主候选未优于目标前缀平方根比较器；这不是长期验证 |

三个包都保留无未来标签预测、评分结果、冻结哈希和明确的禁止宣称。完整解释见
[技术报告](../../reports/landmark_v4_external_evidence_2026-07-20.md)。

重新生成：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_calendar_landmark_readiness.py --output-dir artifacts\landmark
.\.venv\Scripts\python.exe scripts\run_calendar_v4_hybrid_development.py --output-dir artifacts\v4
.\.venv\Scripts\python.exe scripts\run_geisbauer_external_stress.py --output-dir artifacts\geisbauer
```

所有 runner 都拒绝覆盖已有证据；V4 与 Geisbauer runner 先在隐藏 staging 目录
完成写入，再原子发布完整目录。
