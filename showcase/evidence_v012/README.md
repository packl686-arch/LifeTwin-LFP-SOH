# V0.12 稳健性与长期资格证据

本目录公开两组由冻结 runner 重建的回顾性稳健性审计。它们检验 v0.11 结论对
校准切分和单个外部电芯的依赖程度，不晋升新均值模型，也不构成长期确认。

| 目录 | 审计单位 | 回答的问题 | 当前结论 |
|---|---|---|---|
| `v4_calibration_robustness/` | 10 条 Naumann 非训练条件轨迹的全部 210 个重叠划分 | 路由化分位数和宽度是否依赖某一次 6/4 校准切分？ | fallback 仅 80% 可计算且宽度敏感；specialist 所有覆盖等级都欠校准 |
| `geisbauer_robustness/` | 15 个物理电芯及 15 次 LOCO | 外部平均负迁移是否一致、位于哪些 SOC、是否由少数电芯驱动？ | 平均略差但逐电芯 8/7，LOCO 可改变方向；只支持负迁移风险诊断 |

## 重要分母

V4 的 210 个划分是同 10 条条件轨迹的穷举重排，不是 210 份独立数据。fallback
的 `552/672` 是重复的 `condition x partition` 评估实例；`126/210` 是各自全部
fallback 评估轨迹均覆盖的划分数。两者都不能称为正式覆盖率验证。

Geisbauer 的 sign 和 exhaustive sign-flip 结果是在 v0.11 结局已被查看后设计，
未做多重性校正，字段明确标为 exploratory nominal diagnostics。主 sign count 的
`1e-12 pp` 只是数值零容差；证据同时报告 `0/0.01/0.05/0.10 pp` 等效界敏感性，
但后两者仍是事后诊断，不是工程验收阈值。15 个 LOCO 行是高度重叠的单删场景，
不是独立复现。60 °C、120 天加速筛查不能替代常温长期日历队列。

## 重新生成

runner 都拒绝覆盖已有目录，并通过隐藏 staging 目录原子发布完整证据包：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_calendar_v4_calibration_robustness.py --output-dir artifacts\v4-calibration-robustness
.\.venv\Scripts\python.exe scripts\run_geisbauer_robustness_audit.py --output-dir artifacts\geisbauer-robustness
.\.venv\Scripts\python.exe showcase\analyze_v012_robustness.py --evidence-root showcase\evidence_v012 --output artifacts\v012-robustness.png
```

Linux/macOS：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_calendar_v4_calibration_robustness.py --output-dir artifacts/v4-calibration-robustness
PYTHONPATH=src .venv/bin/python scripts/run_geisbauer_robustness_audit.py --output-dir artifacts/geisbauer-robustness
.venv/bin/python showcase/analyze_v012_robustness.py --evidence-root showcase/evidence_v012 --output artifacts/v012-robustness.png
```

完整解释见[技术报告](../../reports/robustness_and_long_term_protocol_2026-07-21.md)，
长期数据资格与预冻结规则见[长期验证预注册](../../docs/independent_long_term_lfp_preregistration.md)。
