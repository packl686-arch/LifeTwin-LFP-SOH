"""Build the zero-install, evidence-only LifeTwin judge console.

The generated HTML is intentionally a retrospective evidence viewer, not an
inference service.  It reads only the frozen public evidence packs under
``showcase/evidence_v011`` and ``showcase/evidence_v012`` and embeds the
selected rows into one self-contained file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "judge-console" / "index.html"

V4_PREDICTIONS = ROOT / "showcase" / "evidence_v011" / "v4" / "label_free_predictions.csv"
V4_METRICS = ROOT / "showcase" / "evidence_v011" / "v4" / "condition_metrics.csv"
V4_RESULT = ROOT / "showcase" / "evidence_v011" / "v4" / "result.json"
LANDMARK_METRICS = ROOT / "showcase" / "evidence_v011" / "landmark" / "common_support_metrics.csv"
GEISBAUER_CELLS = ROOT / "showcase" / "evidence_v012" / "geisbauer_robustness" / "cell_paired_deltas.csv"
GEISBAUER_STRATA = ROOT / "showcase" / "evidence_v012" / "geisbauer_robustness" / "stratum_diagnostics.csv"
GEISBAUER_LOO = ROOT / "showcase" / "evidence_v012" / "geisbauer_robustness" / "leave_one_cell_out.csv"
GEISBAUER_RESULT = ROOT / "showcase" / "evidence_v012" / "geisbauer_robustness" / "result.json"

ALLOWED_EVIDENCE_ROOTS = (
    (ROOT / "showcase" / "evidence_v011").resolve(),
    (ROOT / "showcase" / "evidence_v012").resolve(),
)

ROUTE_LABELS = {
    "hierarchical_power_fallback": "通用幂律回退",
    "hierarchical_activation_residual": "激活残差专用路由",
}

REASON_LABELS = {
    "calibration_unavailable": "该路由没有可用校准量",
    "calibration_evidence_not_independent": "校准证据并非独立确认队列",
    "insufficient_same_route_calibration": "同路由校准条件不足",
    "independent_long_term_evidence_missing": "缺少独立长期 LFP 证据",
    "interval_width_invalid": "区间宽度不可用",
    "cross_dataset_domain_not_confirmed": "跨数据集适用域未确认",
}


def _assert_evidence_path(path: Path) -> None:
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) for root in ALLOWED_EVIDENCE_ROOTS):
        raise ValueError(f"Judge console may only read frozen evidence: {path}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    _assert_evidence_path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    _assert_evidence_path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value}")
    return result


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() == "true"


def _split_reasons(value: str) -> list[str]:
    keys = [item for item in value.split(";") if item and item != "none"]
    return [REASON_LABELS.get(item, item.replace("_", " ")) for item in keys]


def _final_truth(target_id: str, landmark_rows: list[dict[str, str]]) -> float:
    values = {
        round(float(row["common_support_final_true_retention_pct"]), 12)
        for row in landmark_rows
        if row["target_condition_id"] == target_id
        and int(row["prefix_checkups"]) == 10
    }
    if len(values) != 1:
        raise ValueError(f"Expected one evidence-backed final truth for {target_id}, got {values}")
    return values.pop()


def _naumann_case(
    *,
    case_id: str,
    target_id: str,
    title: str,
    subtitle: str,
    predictions: list[dict[str, str]],
    metrics: list[dict[str, str]],
    landmark_rows: list[dict[str, str]],
) -> dict[str, Any]:
    rows = [
        row
        for row in predictions
        if row["target_condition_id"] == target_id
        and math.isclose(float(row["requested_coverage"]), 0.8)
    ]
    rows.sort(key=lambda row: int(row["target_checkup_index"]))
    if len(rows) != 25:
        raise ValueError(f"Expected 25 forecast points for {target_id}, got {len(rows)}")

    metric_rows = [
        row
        for row in metrics
        if row["target_condition_id"] == target_id
        and math.isclose(float(row["requested_coverage"]), 0.8)
    ]
    if len(metric_rows) != 1:
        raise ValueError(f"Expected one 80% metric row for {target_id}")
    metric = metric_rows[0]
    first = rows[0]
    last = rows[-1]
    interval_available = last["diagnostic_interval_status"] == "available"
    route = last["mean_route"]
    final_truth = _final_truth(target_id, landmark_rows)
    inferred_truth = float(last["predicted_capacity_retention_pct"]) - float(metric["final_error_pp"])
    if not math.isclose(final_truth, inferred_truth, abs_tol=1e-9):
        raise ValueError(f"Final truth cross-check failed for {target_id}")

    curve = []
    for row in rows:
        curve.append(
            {
                "day": round(float(row["elapsed_days"]), 6),
                "prediction": round(float(row["predicted_capacity_retention_pct"]), 6),
                "lower": round(float(row["diagnostic_lower_pct"]), 6)
                if row["diagnostic_lower_pct"]
                else None,
                "upper": round(float(row["diagnostic_upper_pct"]), 6)
                if row["diagnostic_upper_pct"]
                else None,
            }
        )

    diagnostic_reasons = _split_reasons(last["diagnostic_abstention_reasons"])
    operational_reasons = _split_reasons(last["operational_abstention_reasons"])
    return {
        "id": case_id,
        "kind": "naumann",
        "title": title,
        "subtitle": subtitle,
        "dataset": "Naumann 2021 公开 LFP 日历老化",
        "target": target_id.replace("NAUMANN_CAL_", ""),
        "temperature": float(first["temperature_c"]),
        "soc": float(first["storage_soc_fraction"]) * 100,
        "prefix": {
            "count": int(first["prefix_checkups"]),
            "end_day": round(float(first["prefix_end_days"]), 3),
            "values_available_in_evidence_pack": False,
        },
        "curve": curve,
        "final_truth": {
            "day": curve[-1]["day"],
            "retention": round(final_truth, 6),
            "scope": "冻结证据仅提供最终揭盲值；中间真值不补造",
        },
        "route": ROUTE_LABELS.get(route, route),
        "route_code": route,
        "interval": {
            "status": "诊断区间可用" if interval_available else "诊断区间拒绝",
            "available": interval_available,
            "coverage": 0.8,
            "mean_width": _number(metric["diagnostic_mean_width_pp"]),
            "reasons": diagnostic_reasons,
        },
        "decision": {
            "status": "拒绝运营签发",
            "code": last["operational_issuance_status"],
            "reasons": operational_reasons,
        },
        "evidence": {
            "grade": "E2",
            "label": "回顾性条件均值诊断",
            "confirmed": False,
            "horizon": "136.9 → 885.0 天",
        },
        "metrics": [
            {"label": "未来点 MAE", "value": f"{float(metric['point_mae_pp']):.3f} pp"},
            {"label": "末点绝对误差", "value": f"{float(metric['final_absolute_error_pp']):.3f} pp"},
            {
                "label": "80% 同时覆盖",
                "value": "是（诊断）" if _bool(metric["diagnostic_simultaneous_covered"]) else "不可计算",
            },
            {"label": "运营区间", "value": "0 条签发"},
        ],
        "truth_note": "仅显示冻结 evidence 中可交叉核对的最终真值点。",
    }


def _geisbauer_case(
    cells: list[dict[str, str]],
    strata: list[dict[str, str]],
    loo: list[dict[str, str]],
    result: dict[str, Any],
) -> dict[str, Any]:
    overall_rows = [
        row
        for row in strata
        if row["scope_type"] == "all_physical_cells"
        and row["metric"] == "trajectory_iae_pp"
    ]
    if len(overall_rows) != 1 or len(cells) != 15 or len(loo) != 15:
        raise ValueError("Unexpected Geisbauer robustness evidence shape")
    overall = overall_rows[0]
    cell_points = [
        {
            "cell": row["cell_id"].replace("GEISBAUER_LFP_CELL_", "#"),
            "soc": round(float(row["storage_soc_fraction"]) * 100),
            "delta": round(float(row["paired_delta_trajectory_iae_pp"]), 6),
            "outcome": row["candidate_outcome"],
        }
        for row in sorted(cells, key=lambda row: int(row["source_cell_number"]))
    ]
    flips = sum(_bool(row["direction_flipped_from_full_sample"]) for row in loo)
    mean_delta = float(overall["mean_paired_delta_pp"])
    median_delta = float(overall["median_paired_delta_pp"])
    permutation_p = float(overall["exact_mean_sign_flip_two_sided_p"])
    route_reality = result["route_reality"]
    if not route_reality["candidate_exactly_equals_hierarchical_power_fallback"]:
        raise ValueError("Geisbauer route reality changed")

    return {
        "id": "geisbauer-negative-transfer",
        "kind": "external",
        "title": "Geisbauer 外部负迁移",
        "subtitle": "15 个物理电芯 · 60°C · 仅 120 天压力筛查",
        "dataset": "Geisbauer 2022 公开 LFP 日历老化",
        "target": "跨数据集压力筛查",
        "bars": cell_points,
        "route": "通用幂律回退（专用路由 0/15 就绪）",
        "route_code": "hierarchical_power_fallback_only",
        "interval": {
            "status": "未设计可签发区间",
            "available": False,
            "coverage": None,
            "mean_width": None,
            "reasons": ["跨数据集适用域未确认", "仅 120 天，不是长期确认"],
        },
        "decision": {
            "status": "拒绝迁移上线",
            "code": "abstained",
            "reasons": [
                "候选方法平均误差较简单前缀基线增加",
                "均值、中央値与逐电芯胜负方向冲突",
                "缺少独立长期确认队列",
            ],
        },
        "evidence": {
            "grade": "E1",
            "label": "探索性外部压力筛查",
            "confirmed": False,
            "horizon": "59 → 120 天",
        },
        "metrics": [
            {"label": "平均配对差", "value": f"+{mean_delta:.3f} pp（更差）"},
            {"label": "中位配对差", "value": f"{median_delta:.3f} pp"},
            {
                "label": "逐电芯胜 / 负",
                "value": f"{overall['candidate_better_count']} / {overall['candidate_worse_count']}",
            },
            {"label": "精确符号翻转 p", "value": f"{permutation_p:.3f}"},
        ],
        "external_summary": {
            "mean_delta": round(mean_delta, 6),
            "median_delta": round(median_delta, 6),
            "loo_flips": flips,
            "permutation_p": round(permutation_p, 6),
            "interpretation": result["negative_transfer_diagnosis"]["diagnostic_interpretation"],
        },
        "truth_note": "正值表示候选方法的轨迹误差更大；这是一项负结果压力筛查。",
    }


def build_payload() -> dict[str, Any]:
    sources = [
        V4_PREDICTIONS,
        V4_METRICS,
        V4_RESULT,
        LANDMARK_METRICS,
        GEISBAUER_CELLS,
        GEISBAUER_STRATA,
        GEISBAUER_LOO,
        GEISBAUER_RESULT,
    ]
    predictions = _read_csv(V4_PREDICTIONS)
    metrics = _read_csv(V4_METRICS)
    landmark_rows = _read_csv(LANDMARK_METRICS)
    v4_result = _read_json(V4_RESULT)
    geisbauer_result = _read_json(GEISBAUER_RESULT)

    if v4_result["design"]["target_future_outcomes_used_for_prediction"]:
        raise ValueError("V4 future-label firewall is not intact")
    if v4_result["confirmation"]["status"] != "not_confirmed":
        raise ValueError("Console wording must be reviewed after confirmation status changes")

    cases = [
        _naumann_case(
            case_id="naumann-fallback",
            target_id="NAUMANN_CAL_T40_SOC37.5",
            title="T40 · SOC 37.5% · 回退",
            subtitle="证据不足触发通用幂律；80% 诊断区间可画但不可运营签发",
            predictions=predictions,
            metrics=metrics,
            landmark_rows=landmark_rows,
        ),
        _naumann_case(
            case_id="naumann-specialist",
            target_id="NAUMANN_CAL_T40_SOC12.5",
            title="T40 · SOC 12.5% · 专用路由",
            subtitle="激活残差路由被选中；同路由校准仅 1 条，因此主动拒绝区间",
            predictions=predictions,
            metrics=metrics,
            landmark_rows=landmark_rows,
        ),
        _geisbauer_case(
            _read_csv(GEISBAUER_CELLS),
            _read_csv(GEISBAUER_STRATA),
            _read_csv(GEISBAUER_LOO),
            geisbauer_result,
        ),
    ]
    return {
        "console_version": "judge-console-v1",
        "mode": "retrospective_evidence_replay_not_live_inference",
        "cases": cases,
        "scope": {
            "chemistry": "公开 LFP 数据",
            "unit": "Naumann 为条件均值；Geisbauer 为 15 个物理电芯压力筛查",
            "maximum_horizon": "885 天",
            "prohibited": "不代表海辰产品，不支持储能电站或 15–25 年结论",
        },
        "evidence_scale": {
            "E2": "复用公开队列的回顾性条件均值诊断；结果已被项目查看，非确认性验证。",
            "E1": "短时外部探索性压力筛查；可暴露负迁移，不能确认长期性能。",
        },
        "sources": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for path in sources
        ],
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>LifeTwin · 评审证据控制台</title>
  <style>
    :root {
      --ink: #17201f;
      --muted: #61706d;
      --paper: #f3f4ef;
      --panel: #ffffff;
      --line: #d9ded9;
      --teal: #087b72;
      --teal-soft: #dcefeb;
      --orange: #c76618;
      --orange-soft: #fff0df;
      --red: #b93831;
      --red-soft: #fae9e6;
      --green: #397a49;
      --green-soft: #e5f1e7;
      --charcoal: #26312f;
      --grid: #e6e9e5;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    button, input { font: inherit; }
    .topbar {
      min-height: 70px;
      padding: 12px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      background: var(--charcoal);
      color: #fff;
      border-bottom: 4px solid var(--orange);
    }
    .brand { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
    .brand strong { font-size: 22px; font-weight: 750; }
    .brand span { color: #bec9c5; font-size: 13px; }
    .mode-badge {
      padding: 6px 9px;
      border: 1px solid #899692;
      border-radius: 4px;
      color: #fff;
      white-space: nowrap;
      font-size: 12px;
    }
    .scopebar {
      min-height: 48px;
      padding: 8px 22px;
      display: flex;
      align-items: center;
      gap: 10px;
      background: #fff7ee;
      border-bottom: 1px solid #e2c4a9;
      overflow-x: auto;
    }
    .scopebar strong { color: #8d3d12; white-space: nowrap; }
    .scope-tag {
      padding: 5px 8px;
      border: 1px solid #d8b18d;
      border-radius: 4px;
      background: #fff;
      color: #694633;
      white-space: nowrap;
      font-size: 12px;
    }
    .shell { max-width: 1540px; margin: 0 auto; padding: 14px 18px 18px; }
    .tabs {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      overflow: hidden;
      margin-bottom: 12px;
    }
    .tab {
      min-height: 48px;
      padding: 8px 14px;
      border: 0;
      border-right: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      cursor: pointer;
      text-align: left;
    }
    .tab:last-child { border-right: 0; }
    .tab strong { display: block; color: var(--ink); font-size: 14px; }
    .tab span { display: block; margin-top: 2px; font-size: 11px; }
    .tab[aria-selected="true"] { box-shadow: inset 0 -4px 0 var(--teal); background: #f7fbfa; }
    .tab:focus-visible, .truth-control:focus-within { outline: 3px solid #f2aa69; outline-offset: 2px; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 350px;
      gap: 12px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      min-width: 0;
    }
    .visual-panel { padding: 15px 17px 12px; }
    .visual-head {
      min-height: 52px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
    }
    .visual-head h1 { margin: 0; font-size: 19px; line-height: 1.3; }
    .visual-head p { margin: 4px 0 0; color: var(--muted); font-size: 12px; line-height: 1.45; }
    .truth-control {
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 4px;
      white-space: nowrap;
      color: var(--charcoal);
      cursor: pointer;
    }
    .truth-control input { width: 16px; height: 16px; accent-color: var(--teal); }
    .prefix-strip {
      display: grid;
      grid-template-columns: 155px minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      min-height: 42px;
      padding: 7px 9px;
      margin: 9px 0 5px;
      border: 1px solid var(--line);
      background: #f8f9f6;
    }
    .prefix-label strong { display: block; font-size: 12px; }
    .prefix-label span { color: var(--muted); font-size: 10px; }
    .prefix-dots { display: grid; grid-template-columns: repeat(10, minmax(12px, 1fr)); gap: 5px; }
    .prefix-dot {
      aspect-ratio: 1;
      max-width: 18px;
      min-width: 12px;
      border: 2px solid #7b8985;
      background: #fff;
      border-radius: 50%;
      position: relative;
    }
    .prefix-dot::after {
      content: attr(data-index);
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 8px;
      color: #7b8985;
    }
    .prefix-end { color: var(--muted); font-size: 11px; white-space: nowrap; }
    .chart-wrap { width: 100%; aspect-ratio: 820 / 340; min-height: 285px; }
    #chart { display: block; width: 100%; height: 100%; overflow: visible; }
    .chart-note {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; }
    .legend-item { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
    .legend-line { width: 18px; height: 3px; background: var(--teal); }
    .legend-band { width: 18px; height: 9px; background: var(--teal-soft); border: 1px solid #82b9b2; }
    .legend-truth { width: 9px; height: 9px; border-radius: 50%; background: var(--orange); }
    .decision-panel { display: flex; flex-direction: column; }
    .decision-head { padding: 14px 15px 12px; border-bottom: 1px solid var(--line); }
    .eyebrow { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .decision-title { margin-top: 5px; font-size: 21px; font-weight: 760; color: var(--red); }
    .status-grid { display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid var(--line); }
    .status-cell { min-height: 72px; padding: 10px 12px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .status-cell:nth-child(2n) { border-right: 0; }
    .status-cell:nth-last-child(-n+2) { border-bottom: 0; }
    .status-cell span { display: block; color: var(--muted); font-size: 10px; }
    .status-cell strong { display: block; margin-top: 5px; font-size: 13px; line-height: 1.35; }
    .grade { color: var(--orange); }
    .reasons { padding: 12px 15px; flex: 1; }
    .reasons h2 { margin: 0 0 8px; font-size: 12px; }
    .reasons ul { margin: 0; padding: 0; list-style: none; }
    .reasons li {
      padding: 7px 0 7px 16px;
      position: relative;
      border-bottom: 1px solid #edf0ec;
      color: #495753;
      font-size: 12px;
      line-height: 1.4;
    }
    .reasons li:last-child { border-bottom: 0; }
    .reasons li::before { content: ""; position: absolute; left: 0; top: 13px; width: 7px; height: 7px; background: var(--red); }
    .metrics {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      overflow: hidden;
    }
    .metric { min-height: 78px; padding: 12px 14px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric span { color: var(--muted); font-size: 11px; }
    .metric strong { display: block; margin-top: 8px; font-size: 18px; }
    .auditline {
      margin-top: 12px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-left: 4px solid var(--orange);
      background: #fff;
      color: #52605c;
      font-size: 11px;
      line-height: 1.45;
    }
    .auditline code { font-family: Consolas, monospace; font-size: 10px; color: #34413e; }
    .attribution { margin-top: 5px; color: #65716e; }
    .attribution a { color: #087b72; text-decoration-thickness: 1px; text-underline-offset: 2px; }
    .provenance { text-align: right; white-space: nowrap; }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0; }
    @media (max-width: 980px) {
      .workspace { grid-template-columns: 1fr; }
      .decision-panel { min-height: 360px; }
      .chart-wrap { min-height: 250px; }
    }
    @media (max-width: 680px) {
      .topbar { align-items: flex-start; flex-direction: column; padding: 11px 14px; }
      .brand { display: block; }
      .brand span { display: block; margin-top: 3px; }
      .scopebar { padding: 7px 14px; }
      .shell { padding: 10px; }
      .tabs { grid-template-columns: 1fr; }
      .tab { border-right: 0; border-bottom: 1px solid var(--line); }
      .tab:last-child { border-bottom: 0; }
      .visual-head { display: block; }
      .truth-control { margin-top: 9px; width: fit-content; }
      .prefix-strip { grid-template-columns: 1fr; padding-bottom: 18px; }
      .prefix-dots {
        grid-template-columns: repeat(5, minmax(20px, 1fr));
        row-gap: 16px;
        max-width: 260px;
        padding-bottom: 10px;
      }
      .prefix-end { white-space: normal; }
      .chart-wrap { height: 220px; min-height: 0; aspect-ratio: auto; }
      .chart-note { display: block; }
      .chart-note > div:last-child { margin-top: 7px; }
      .metrics { grid-template-columns: 1fr 1fr; }
      .metric:nth-child(2) { border-right: 0; }
      .metric { border-bottom: 1px solid var(--line); }
      .metric:nth-last-child(-n+2) { border-bottom: 0; }
      .auditline { grid-template-columns: 1fr; }
      .provenance { text-align: left; white-space: normal; }
    }

    /* 2026 judge-facing refinement */
    :root {
      --paper: #f5f7f6;
      --line: #dce3df;
      --teal: #0b7569;
      --teal-soft: #e1f0ec;
      --orange: #b65a19;
      --orange-soft: #fff1e5;
      --red: #ad3732;
      --red-soft: #fbeceb;
      --green: #2f7250;
      --green-soft: #e8f2eb;
      --charcoal: #172824;
    }
    body { line-height: 1.45; }
    a { color: inherit; }
    .topbar {
      min-height: 68px;
      padding: 12px max(18px, calc((100vw - 1440px) / 2 + 18px));
      border-bottom: 1px solid #31443f;
      background: var(--charcoal);
    }
    .brand { color: #fff; text-decoration: none; align-items: center; gap: 11px; }
    .brand .brand-mark {
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border: 1px solid #5c7a72;
      border-radius: 5px;
      color: #9ed3c9;
      font-size: 12px;
      font-weight: 800;
    }
    .brand-copy { display: block; }
    .brand-copy strong { display: block; font-size: 18px; line-height: 1.1; }
    .brand-copy span { display: block; margin-top: 4px; color: #b8c8c3; font-size: 11px; }
    .top-actions { display: flex; align-items: center; justify-content: flex-end; gap: 7px; flex-wrap: wrap; }
    .top-link, .mode-badge {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 7px 10px;
      border: 1px solid #50645f;
      border-radius: 4px;
      color: #ecf3f1;
      text-decoration: none;
      font-size: 12px;
      white-space: nowrap;
    }
    .top-link:hover { background: #243a34; border-color: #77948c; }
    .mode-badge { color: #a9d5cc; border-color: #315f56; background: #1a302b; }
    .scopebar {
      min-height: 50px;
      padding: 8px max(18px, calc((100vw - 1440px) / 2 + 18px));
      justify-content: space-between;
      gap: 16px;
      background: #fff;
      border-bottom: 1px solid var(--line);
      overflow: visible;
    }
    .scope-copy { display: flex; align-items: baseline; gap: 10px; min-width: 280px; }
    .scope-copy strong { color: var(--ink); }
    .scope-copy span { color: var(--muted); font-size: 12px; }
    .scope-tags { display: flex; gap: 7px; overflow-x: auto; padding: 1px 0; }
    .scope-tag { border-color: #e1c7b3; background: #fff9f4; color: #794826; }
    .shell { max-width: 1440px; padding: 16px 18px 24px; }
    .console-intro {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(420px, 1fr);
      gap: 24px;
      align-items: center;
      margin-bottom: 12px;
      padding: 18px 20px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--red);
      background: #fff;
    }
    .section-kicker { color: var(--red); font-size: 10px; font-weight: 800; text-transform: uppercase; }
    .console-intro h1 { margin: 4px 0 5px; font-size: 22px; line-height: 1.25; }
    .console-intro p { margin: 0; color: var(--muted); font-size: 12px; }
    .evidence-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-left: 1px solid var(--line); }
    .evidence-summary div { min-height: 58px; padding: 7px 12px; border-right: 1px solid var(--line); }
    .evidence-summary span { display: block; color: var(--muted); font-size: 10px; }
    .evidence-summary strong { display: block; margin-top: 5px; font-size: 15px; line-height: 1.2; }
    .tabs { position: sticky; top: 0; z-index: 20; margin-bottom: 12px; border-color: #cfd8d3; box-shadow: 0 6px 18px rgba(23, 40, 36, .06); }
    .tab { min-height: 76px; padding: 9px 12px; position: relative; }
    .tab small { display: block; color: #7a8783; font-size: 9px; font-weight: 700; }
    .tab strong { margin-top: 4px; font-size: 13px; line-height: 1.3; padding-right: 72px; }
    .tab span { max-width: calc(100% - 78px); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tab em {
      position: absolute;
      top: 10px;
      right: 10px;
      padding: 3px 6px;
      border: 1px solid #e1aaa6;
      border-radius: 4px;
      background: var(--red-soft);
      color: var(--red);
      font-size: 9px;
      font-style: normal;
    }
    .tab[aria-selected="true"] { box-shadow: inset 0 -3px 0 var(--teal); background: #f2f8f6; }
    .workspace { grid-template-columns: minmax(0, 1fr) 370px; gap: 12px; }
    .panel { border-color: #d8e0dc; box-shadow: 0 1px 2px rgba(23, 40, 36, .035); }
    .visual-panel { padding: 17px 19px 13px; }
    .visual-head h1 { font-size: 18px; }
    .truth-control { padding: 7px 10px; background: #fff; }
    .truth-control input {
      appearance: none;
      width: 32px;
      height: 18px;
      border: 1px solid #aab6b2;
      border-radius: 9px;
      background: #e7ebe9;
      position: relative;
      cursor: pointer;
      transition: background .16s ease, border-color .16s ease;
    }
    .truth-control input::after {
      content: "";
      position: absolute;
      width: 12px;
      height: 12px;
      left: 2px;
      top: 2px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 1px 2px rgba(23, 40, 36, .25);
      transition: transform .16s ease;
    }
    .truth-control input:checked { background: var(--teal); border-color: var(--teal); }
    .truth-control input:checked::after { transform: translateX(14px); }
    .prefix-strip { border-color: var(--line); background: #f8faf9; }
    .decision-panel { border-top: 4px solid var(--red); }
    .decision-head { padding: 15px 16px 13px; background: #fff9f8; }
    .decision-title { font-size: 22px; }
    .decision-code { margin-bottom: 6px; color: var(--red); font-size: 10px; font-weight: 800; }
    .status-cell strong { overflow-wrap: anywhere; }
    .reasons h2 { color: var(--red); }
    .metrics { border-color: #d8e0dc; box-shadow: 0 1px 2px rgba(23, 40, 36, .035); }
    .metric strong { font-size: 17px; }
    details.auditline {
      display: block;
      padding: 0;
      border-left: 0;
      border-top: 3px solid var(--orange);
      box-shadow: 0 1px 2px rgba(23, 40, 36, .03);
    }
    .auditline summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      min-height: 46px;
      padding: 9px 12px;
      cursor: pointer;
      list-style: none;
    }
    .auditline summary::-webkit-details-marker { display: none; }
    .auditline summary::before { content: "+"; color: var(--orange); font-size: 17px; font-weight: 700; }
    .auditline[open] summary::before { content: "−"; }
    .audit-summary-copy { flex: 1; }
    .audit-summary-copy span { display: block; margin-top: 2px; color: var(--muted); }
    .audit-detail { padding: 10px 12px 12px; border-top: 1px solid var(--line); }
    .top-actions, .scope-tags, .tabs { scrollbar-width: none; }
    .top-actions::-webkit-scrollbar, .scope-tags::-webkit-scrollbar, .tabs::-webkit-scrollbar { display: none; }
    @media (max-width: 980px) {
      .console-intro { grid-template-columns: 1fr; }
      .evidence-summary { border-left: 0; border-top: 1px solid var(--line); padding-top: 9px; }
      .workspace { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      .topbar { padding: 10px 12px; gap: 9px; }
      .brand { display: flex; }
      .brand-copy span { display: block; }
      .top-actions { width: 100%; justify-content: flex-start; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 1px; }
      .top-link, .mode-badge { min-height: 31px; padding: 5px 8px; }
      .scopebar { display: block; padding: 8px 12px; }
      .scope-copy { min-width: 0; }
      .scope-copy span { display: none; }
      .scope-tags { margin-top: 6px; }
      .shell { padding: 10px 10px 18px; }
      .console-intro { display: block; padding: 15px; }
      .console-intro h1 { font-size: 19px; }
      .evidence-summary { grid-template-columns: 1fr 1fr 1fr; margin-top: 12px; }
      .evidence-summary div { min-height: 62px; padding: 7px; }
      .evidence-summary strong { font-size: 13px; }
      .tabs { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(225px, 76vw); grid-template-columns: none; overflow-x: auto; }
      .tab { border-right: 1px solid var(--line); border-bottom: 0; }
      .visual-panel { padding: 14px 12px 11px; }
      .chart-wrap { height: 250px; }
      .decision-panel { min-height: 0; }
      .auditline summary { align-items: flex-start; }
      .auditline .provenance { text-align: right; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <a class="brand" href="./" aria-label="LifeTwin 评审证据控制台首页">
      <span class="brand-mark">LT</span>
      <span class="brand-copy"><strong>LifeTwin</strong><span>评审证据控制台</span></span>
    </a>
    <div class="top-actions">
      <span class="mode-badge">回顾性证据回放 · 非实时推理</span>
      <a class="top-link" href="../demo/">可视化 Demo</a>
      <a class="top-link" href="https://github.com/packl686-arch/LifeTwin-LFP-SOH">GitHub 证据仓库</a>
    </div>
  </header>
  <section class="scopebar" aria-label="结论边界">
    <div class="scope-copy"><strong>证据边界</strong><span>冻结证据只回答“系统是否知道何时不该承诺”。</span></div>
    <div class="scope-tags">
      <span class="scope-tag">公开 LFP</span>
      <span class="scope-tag">条件均值 / 短时外部筛查</span>
      <span class="scope-tag">最长 885 天</span>
      <span class="scope-tag">非海辰产品证据</span>
      <span class="scope-tag">不可支持 15–25 年</span>
    </div>
  </section>

  <main class="shell">
    <section class="console-intro" aria-labelledby="console-title">
      <div>
        <div class="section-kicker">FROZEN EVIDENCE · FAIL CLOSED</div>
        <h1 id="console-title">三条失败关闭路径，同一套审计口径</h1>
        <p>回放通用回退、专用路由校准不足与外部负迁移。所有当前案例均保留诊断证据，并拒绝越过证据边界。</p>
      </div>
      <div class="evidence-summary" aria-label="证据控制台摘要">
        <div><span>冻结案例</span><strong>3</strong></div>
        <div><span>证据文件</span><strong id="source-count-top">0</strong></div>
        <div><span>运营终态</span><strong>3 / 3 拒绝</strong></div>
      </div>
    </section>
    <nav id="tabs" class="tabs" aria-label="预置证据案例" role="tablist"></nav>
    <section class="workspace">
      <article class="panel visual-panel">
        <div class="visual-head">
          <div>
            <h1 id="case-title"></h1>
            <p id="case-subtitle"></p>
          </div>
          <label class="truth-control" id="truth-control" title="只揭示冻结证据中可交叉核对的最终真值点">
            <input id="truth-toggle" type="checkbox">
            <span>揭盲最终真值</span>
          </label>
        </div>
        <div class="prefix-strip" id="prefix-strip">
          <div class="prefix-label"><strong>前 10 点观测前缀</strong><span>已用于预测，未来结果未作输入</span></div>
          <div class="prefix-dots" id="prefix-dots" aria-label="十个前缀观测位置"></div>
          <div class="prefix-end" id="prefix-end"></div>
        </div>
        <div class="chart-wrap"><svg id="chart" viewBox="0 0 820 340" role="img"></svg></div>
        <div class="chart-note">
          <div class="legend" id="legend"></div>
          <div id="truth-note"></div>
        </div>
      </article>

      <aside class="panel decision-panel" aria-live="polite">
        <div class="decision-head">
          <div class="decision-code">FAIL CLOSED · 主动关闭</div>
          <div class="eyebrow">运营结论</div>
          <div class="decision-title" id="decision-title"></div>
        </div>
        <div class="status-grid">
          <div class="status-cell"><span>自动路由</span><strong id="route"></strong></div>
          <div class="status-cell"><span>区间状态</span><strong id="interval-status"></strong></div>
          <div class="status-cell"><span>证据等级</span><strong class="grade" id="evidence-grade"></strong></div>
          <div class="status-cell"><span>证据窗口</span><strong id="horizon"></strong></div>
        </div>
        <div class="reasons">
          <h2>拒绝原因 / 风险信号</h2>
          <ul id="reasons"></ul>
        </div>
      </aside>
    </section>

    <section class="metrics" id="metrics" aria-label="关键指标"></section>
    <details class="auditline">
      <summary>
        <span class="audit-summary-copy"><strong>证据来源与审计指纹</strong><span>展开核对数据归属、成熟度口径与冻结文件摘要</span></span>
        <span class="provenance"><span id="source-count"></span> 个冻结证据文件<br><code id="source-sha"></code></span>
      </summary>
      <div class="audit-detail">
        <strong>审计口径：</strong><span id="audit-copy"></span> E2/E1 是本控制台的证据成熟度标记，不是行业标准等级。
        <div class="attribution">
          数据归属：Naumann <a href="https://doi.org/10.17632/kxh42bfgtj.1">DOI 10.17632/kxh42bfgtj.1</a>；
          Geisbauer <a href="https://doi.org/10.5281/zenodo.6685365">DOI 10.5281/zenodo.6685365</a>；
          两份上游数据均按 <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> 标注。
          本页是 LifeTwin 派生证据回放，处理与归属见
          <a href="https://github.com/packl686-arch/LifeTwin-LFP-SOH/blob/main/NOTICE.md">NOTICE</a> 和
          <a href="https://github.com/packl686-arch/LifeTwin-LFP-SOH">证据仓库</a>。
        </div>
      </div>
    </details>
  </main>

  <noscript>此自包含证据控制台需要浏览器启用 JavaScript 才能绘图。</noscript>
  <script id="evidence-data" type="application/json">__EVIDENCE_DATA__</script>
  <script>
    "use strict";
    const DATA = JSON.parse(document.getElementById("evidence-data").textContent);
    const NS = "http://www.w3.org/2000/svg";
    const requestedCase = decodeURIComponent(window.location.hash.slice(1));
    const requestedIndex = DATA.cases.findIndex(item => item.id === requestedCase);
    const state = { active: requestedIndex >= 0 ? requestedIndex : 0, truth: false };

    function el(name, attrs = {}, text = "") {
      const node = document.createElementNS(NS, name);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      if (text) node.textContent = text;
      return node;
    }

    function fmt(value, digits = 1) {
      return Number(value).toFixed(digits);
    }

    function renderTabs() {
      const tabs = document.getElementById("tabs");
      tabs.replaceChildren();
      DATA.cases.forEach((item, index) => {
        const button = document.createElement("button");
        button.className = "tab";
        button.type = "button";
        button.role = "tab";
        button.id = `tab-${item.id}`;
        button.setAttribute("aria-selected", index === state.active ? "true" : "false");
        button.setAttribute("aria-controls", "chart");
        const small = document.createElement("small");
        small.textContent = `CASE 0${index + 1} · ${item.evidence.grade}`;
        const strong = document.createElement("strong");
        strong.textContent = item.title;
        const span = document.createElement("span");
        span.textContent = index < 2 ? item.route : "外部负结果审计";
        span.title = span.textContent;
        const status = document.createElement("em");
        status.textContent = index < 2 ? "拒绝签发" : "拒绝上线";
        button.append(small, strong, span, status);
        button.addEventListener("click", () => {
          state.active = index;
          state.truth = false;
          window.history.replaceState(null, "", `#${item.id}`);
          render();
        });
        button.addEventListener("keydown", event => {
          if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
          event.preventDefault();
          const step = event.key === "ArrowRight" ? 1 : -1;
          state.active = (state.active + step + DATA.cases.length) % DATA.cases.length;
          state.truth = false;
          window.history.replaceState(null, "", `#${DATA.cases[state.active].id}`);
          render();
          document.querySelectorAll(".tab")[state.active].focus();
        });
        tabs.appendChild(button);
      });
    }

    function renderPrefix(item) {
      const strip = document.getElementById("prefix-strip");
      if (item.kind !== "naumann") {
        strip.hidden = true;
        return;
      }
      strip.hidden = false;
      const dots = document.getElementById("prefix-dots");
      dots.replaceChildren();
      for (let index = 1; index <= item.prefix.count; index += 1) {
        const dot = document.createElement("span");
        dot.className = "prefix-dot";
        dot.dataset.index = index;
        dot.title = `观测位置 ${index}：逐点 SOH 未包含在冻结展示包中，故不伪造纵坐标`;
        dots.appendChild(dot);
      }
      document.getElementById("prefix-end").textContent = `截止 ${fmt(item.prefix.end_day, 1)} 天 · 逐点值未在展示包再分发`;
    }

    function drawAxes(svg, xScale, yScale, xTicks, yTicks, xLabel, yLabel) {
      xTicks.forEach(value => {
        const x = xScale(value);
        svg.appendChild(el("line", { x1: x, y1: 18, x2: x, y2: 295, stroke: "#e6e9e5", "stroke-width": 1 }));
        svg.appendChild(el("text", { x, y: 316, fill: "#61706d", "font-size": 11, "text-anchor": "middle" }, String(Math.round(value))));
      });
      yTicks.forEach(value => {
        const y = yScale(value);
        svg.appendChild(el("line", { x1: 58, y1: y, x2: 800, y2: y, stroke: "#e6e9e5", "stroke-width": 1 }));
        svg.appendChild(el("text", { x: 49, y: y + 4, fill: "#61706d", "font-size": 11, "text-anchor": "end" }, fmt(value, 0)));
      });
      svg.appendChild(el("line", { x1: 58, y1: 295, x2: 800, y2: 295, stroke: "#7b8985", "stroke-width": 1.2 }));
      svg.appendChild(el("line", { x1: 58, y1: 18, x2: 58, y2: 295, stroke: "#7b8985", "stroke-width": 1.2 }));
      svg.appendChild(el("text", { x: 429, y: 338, fill: "#61706d", "font-size": 11, "text-anchor": "middle" }, xLabel));
      svg.appendChild(el("text", { x: 13, y: 158, fill: "#61706d", "font-size": 11, "text-anchor": "middle", transform: "rotate(-90 13 158)" }, yLabel));
    }

    function renderNaumannChart(item) {
      const svg = document.getElementById("chart");
      svg.replaceChildren();
      svg.setAttribute("aria-label", `${item.title}：前十点之后的条件均值 SOH 预测曲线`);
      const points = item.curve;
      const xMin = item.prefix.end_day;
      const xMax = Math.max(...points.map(point => point.day));
      const values = points.flatMap(point => [point.prediction, point.lower, point.upper]).filter(value => value !== null);
      if (state.truth) values.push(item.final_truth.retention);
      const yMin = Math.floor(Math.min(...values) - 1);
      const yMax = Math.min(101, Math.ceil(Math.max(...values) + 1));
      const xScale = value => 58 + (value - xMin) / (xMax - xMin) * 742;
      const yScale = value => 295 - (value - yMin) / (yMax - yMin) * 277;
      const xTicks = [xMin, xMin + (xMax - xMin) / 3, xMin + 2 * (xMax - xMin) / 3, xMax];
      const yTicks = Array.from({ length: 5 }, (_, index) => yMin + index * (yMax - yMin) / 4);
      drawAxes(svg, xScale, yScale, xTicks, yTicks, "日历时间（天）", "容量保持率（%）");

      const boundaryX = xScale(xMin);
      svg.appendChild(el("line", { x1: boundaryX, y1: 18, x2: boundaryX, y2: 295, stroke: "#c76618", "stroke-width": 2, "stroke-dasharray": "5 4" }));
      svg.appendChild(el("text", { x: boundaryX + 7, y: 32, fill: "#9a4c12", "font-size": 10 }, "前缀截止 / 预测起点"));

      if (item.interval.available) {
        const upper = points.map(point => `${xScale(point.day)},${yScale(point.upper)}`).join(" ");
        const lower = [...points].reverse().map(point => `${xScale(point.day)},${yScale(point.lower)}`).join(" ");
        svg.appendChild(el("polygon", { points: `${upper} ${lower}`, fill: "#dcefeb", stroke: "#82b9b2", "stroke-width": 1 }));
      }

      const pathData = points.map((point, index) => `${index ? "L" : "M"}${xScale(point.day)},${yScale(point.prediction)}`).join(" ");
      svg.appendChild(el("path", { d: pathData, fill: "none", stroke: "#087b72", "stroke-width": 3, "stroke-linejoin": "round" }));
      points.forEach((point, index) => {
        if (index % 4 === 0 || index === points.length - 1) {
          svg.appendChild(el("circle", { cx: xScale(point.day), cy: yScale(point.prediction), r: 3.2, fill: "#fff", stroke: "#087b72", "stroke-width": 2 }));
        }
      });

      if (state.truth) {
        const truth = item.final_truth;
        const cx = xScale(truth.day);
        const cy = yScale(truth.retention);
        svg.appendChild(el("circle", { cx, cy, r: 6, fill: "#c76618", stroke: "#fff", "stroke-width": 2 }));
        svg.appendChild(el("text", { x: cx - 8, y: cy - 12, fill: "#9a4c12", "font-size": 11, "font-weight": 700, "text-anchor": "end" }, `末点真值 ${fmt(truth.retention, 2)}%`));
      }
    }

    function renderExternalChart(item) {
      const svg = document.getElementById("chart");
      svg.replaceChildren();
      svg.setAttribute("aria-label", "Geisbauer 十五个物理电芯的候选减基线轨迹误差配对差");
      const points = item.bars;
      const xMin = -0.9;
      const xMax = 1.6;
      const xScale = value => 110 + (value - xMin) / (xMax - xMin) * 675;
      const rowHeight = 17;
      const yStart = 27;
      const zeroX = xScale(0);
      [-0.5, 0, 0.5, 1, 1.5].forEach(value => {
        const x = xScale(value);
        svg.appendChild(el("line", { x1: x, y1: 12, x2: x, y2: 282, stroke: value === 0 ? "#26312f" : "#e6e9e5", "stroke-width": value === 0 ? 1.5 : 1 }));
        svg.appendChild(el("text", { x, y: 306, fill: "#61706d", "font-size": 11, "text-anchor": "middle" }, fmt(value, 1)));
      });
      points.forEach((point, index) => {
        const y = yStart + index * rowHeight;
        const endpoint = xScale(point.delta);
        const left = Math.min(zeroX, endpoint);
        const width = Math.max(1.5, Math.abs(endpoint - zeroX));
        const harm = point.delta > 0;
        svg.appendChild(el("text", { x: 47, y: y + 4, fill: "#52605c", "font-size": 10, "text-anchor": "end" }, point.cell));
        svg.appendChild(el("text", { x: 72, y: y + 4, fill: "#87918e", "font-size": 9, "text-anchor": "middle" }, `${point.soc}%`));
        svg.appendChild(el("rect", { x: left, y: y - 5, width, height: 10, fill: harm ? "#b93831" : "#397a49" }));
      });
      const meanX = xScale(item.external_summary.mean_delta);
      svg.appendChild(el("line", { x1: meanX, y1: 12, x2: meanX, y2: 282, stroke: "#c76618", "stroke-width": 2, "stroke-dasharray": "5 3" }));
      svg.appendChild(el("text", { x: meanX + 5, y: 16, fill: "#9a4c12", "font-size": 10 }, `均值 +${fmt(item.external_summary.mean_delta, 3)} pp`));
      svg.appendChild(el("text", { x: 448, y: 334, fill: "#61706d", "font-size": 11, "text-anchor": "middle" }, "轨迹误差配对差（候选 − 前缀基线，pp；正值为负迁移）"));
      svg.appendChild(el("text", { x: 72, y: 306, fill: "#61706d", "font-size": 9, "text-anchor": "middle" }, "SOC"));
    }

    function renderLegend(item) {
      const legend = document.getElementById("legend");
      if (item.kind === "naumann") {
        legend.innerHTML = '<span class="legend-item"><i class="legend-line"></i>条件均值预测</span>' +
          (item.interval.available ? '<span class="legend-item"><i class="legend-band"></i>80% 诊断区间</span>' : '') +
          '<span class="legend-item"><i class="legend-truth"></i>最终揭盲点</span>';
      } else {
        legend.innerHTML = '<span class="legend-item"><i class="legend-line" style="background:#397a49"></i>候选更好</span>' +
          '<span class="legend-item"><i class="legend-line" style="background:#b93831"></i>候选更差</span>';
      }
    }

    function render() {
      const item = DATA.cases[state.active];
      renderTabs();
      document.title = `${item.title} · LifeTwin 评审证据控制台`;
      document.getElementById("case-title").textContent = item.title;
      document.getElementById("case-subtitle").textContent = item.subtitle;
      document.getElementById("decision-title").textContent = item.decision.status;
      document.getElementById("route").textContent = item.route;
      document.getElementById("interval-status").textContent = item.interval.status;
      document.getElementById("evidence-grade").textContent = `${item.evidence.grade} · ${item.evidence.label}`;
      document.getElementById("horizon").textContent = item.evidence.horizon;
      document.getElementById("truth-note").textContent = item.truth_note;

      const truthControl = document.getElementById("truth-control");
      truthControl.hidden = item.kind !== "naumann";
      const truthToggle = document.getElementById("truth-toggle");
      truthToggle.checked = state.truth;
      truthToggle.disabled = item.kind !== "naumann";

      const reasons = [...item.decision.reasons, ...item.interval.reasons.filter(reason => !item.decision.reasons.includes(reason))];
      const reasonList = document.getElementById("reasons");
      reasonList.replaceChildren();
      reasons.forEach(reason => {
        const li = document.createElement("li");
        li.textContent = reason;
        reasonList.appendChild(li);
      });

      const metrics = document.getElementById("metrics");
      metrics.replaceChildren();
      item.metrics.forEach(metric => {
        const div = document.createElement("div");
        div.className = "metric";
        const label = document.createElement("span");
        label.textContent = metric.label;
        const value = document.createElement("strong");
        value.textContent = metric.value;
        div.append(label, value);
        metrics.appendChild(div);
      });

      renderPrefix(item);
      renderLegend(item);
      if (item.kind === "naumann") renderNaumannChart(item); else renderExternalChart(item);
      document.getElementById("audit-copy").textContent = DATA.evidence_scale[item.evidence.grade];
      document.getElementById("source-count").textContent = DATA.sources.length;
      document.getElementById("source-count-top").textContent = DATA.sources.length;
      document.getElementById("source-sha").textContent = `页面证据指纹 ${DATA.sources[0].sha256.slice(0, 12)}…`;
    }

    document.getElementById("truth-toggle").addEventListener("change", event => {
      state.truth = event.target.checked;
      renderNaumannChart(DATA.cases[state.active]);
    });
    render();
  </script>
</body>
</html>
'''


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__EVIDENCE_DATA__", data)


def _self_check(html: str, payload: dict[str, Any]) -> None:
    required = (
        "naumann-fallback",
        "naumann-specialist",
        "geisbauer-negative-transfer",
        "回顾性证据回放 · 非实时推理",
        "不可支持 15–25 年",
        "非海辰产品证据",
        "揭盲最终真值",
        "10.17632/kxh42bfgtj.1",
        "10.5281/zenodo.6685365",
        "CC BY 4.0",
    )
    for token in required:
        if token not in html:
            raise ValueError(f"Generated console is missing required token: {token}")
    if "<script src=" in html or "<link rel=\"stylesheet\" href=" in html:
        raise ValueError("Judge console must not load external assets")
    if len(payload["cases"]) != 3:
        raise ValueError("Judge console must contain exactly three preset cases")
    naumann = payload["cases"][:2]
    if naumann[0]["route_code"] != "hierarchical_power_fallback":
        raise ValueError("Fallback preset route changed")
    if naumann[1]["route_code"] != "hierarchical_activation_residual":
        raise ValueError("Specialist preset route changed")
    if any(case["decision"]["code"] != "abstained" for case in payload["cases"]):
        raise ValueError("Every current preset must fail closed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the self-contained LifeTwin judge console."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the existing output equals deterministic regeneration",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    payload = build_payload()
    html = render_html(payload)
    _self_check(html, payload)
    second = render_html(build_payload())
    if second != html:
        raise RuntimeError("Judge console generator is not deterministic")
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != html:
            raise RuntimeError(f"Judge console differs from regeneration: {output}")
        action = "verified"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8", newline="\n")
        action = "wrote"
    first_hash = _sha256(output)
    try:
        display_path = output.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = output.as_posix()
    print(f"{action} {display_path}")
    print(f"sha256 {first_hash}")
    print(f"cases {len(payload['cases'])}; evidence files {len(payload['sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
