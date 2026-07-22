from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = (
    PROJECT_ROOT
    / "showcase/evidence_v014/synthetic_long_horizon_identifiability_v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/assets/v014_synthetic_identifiability.png"

RISK_REDUCTION_GATE = 0.30
MATCHED_REJECTION_GATE = 0.80
NONINFERIORITY_CEILING_PP = 0.10
REPRESENTATIVE_ISSUANCE_COUNTS = (250, 500, 750)
TRUTH_FAMILY_ORDER = (
    "single_power",
    "dual_power",
    "saturating_plus_slow",
    "early_activation_plus_power",
    "late_knee",
)

INK = "#25343B"
TEAL = "#2B6F77"
GOLD = "#C6922E"
RED = "#B5473F"
LIGHT_RED = "#D9867E"
GRAY = "#7D898F"
LIGHT_GRAY = "#D9DEE1"


@dataclass(frozen=True)
class FigureInputs:
    report: dict[str, Any]
    risk_coverage: pd.DataFrame
    random_rejection: pd.DataFrame
    family_metrics: pd.DataFrame
    forecast_day_metrics: pd.DataFrame


def _load_inputs(evidence_root: Path) -> FigureInputs:
    required = {
        "score_report.json",
        "risk_coverage.csv",
        "random_rejection.csv",
        "family_metrics.csv",
        "forecast_day_metrics.csv",
    }
    missing = sorted(name for name in required if not (evidence_root / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Missing compact v0.14 evidence files: " + ", ".join(missing)
        )
    return FigureInputs(
        report=json.loads((evidence_root / "score_report.json").read_text("utf-8")),
        risk_coverage=pd.read_csv(evidence_root / "risk_coverage.csv"),
        random_rejection=pd.read_csv(evidence_root / "random_rejection.csv"),
        family_metrics=pd.read_csv(evidence_root / "family_metrics.csv"),
        forecast_day_metrics=pd.read_csv(
            evidence_root / "forecast_day_metrics.csv"
        ),
    )


def _require_frozen_failure(report: dict[str, Any]) -> None:
    if report.get("protocol_id") != "synthetic_long_horizon_identifiability_v1":
        raise ValueError("Unexpected synthetic identifiability protocol")
    if report.get("status") != "failure":
        raise ValueError("The v0.14 figure is frozen to the recorded failure result")
    if report.get("protocol_deviations") != []:
        raise ValueError("Cannot plot a run with protocol deviations as canonical evidence")
    expected_gates = {
        "catastrophic_risk_reduction_at_50_percent_issuance": False,
        "issued_trajectory_iae_noninferiority": True,
        "matched_prefix_both_members_rejected": False,
    }
    if report.get("primary_gates") != expected_gates:
        raise ValueError("Primary gate states differ from the frozen v0.14 result")
    if not all(report.get("required_safety_gates", {}).values()):
        raise ValueError("The canonical v0.14 result requires all safety gates to pass")


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Expected finite {label}")
    return number


def _primary_endpoint_values(report: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    risk_reduction = _finite(
        report["test_policy"]["risk_reduction_fraction"], "risk reduction"
    )
    matched_fraction = _finite(
        report["matched_prefix_audit"]["both_rejected_fraction"],
        "matched-pair rejection fraction",
    )
    iae_delta = _finite(
        report["mean_forecast_comparison"]["candidate_minus_baseline_iae_pp"],
        "issued IAE difference",
    )
    ratios = np.asarray(
        [
            risk_reduction / RISK_REDUCTION_GATE,
            matched_fraction / MATCHED_REJECTION_GATE,
            iae_delta / NONINFERIORITY_CEILING_PP,
        ],
        dtype=float,
    )
    labels = [
        f"{risk_reduction * 100:.1f}% / 30% minimum",
        f"{matched_fraction * 100:.0f}% / 80% minimum",
        f"{iae_delta:+.3f} pp / +0.10 pp ceiling",
    ]
    return ratios, labels


def _representative_risk(inputs: FigureInputs) -> tuple[pd.DataFrame, float]:
    frame = inputs.risk_coverage.copy()
    frame["issued_count"] = pd.to_numeric(frame["issued_count"], errors="raise")
    frame["catastrophic_rate"] = pd.to_numeric(
        frame["catastrophic_rate"], errors="raise"
    )
    selected = frame.loc[
        frame["issued_count"].isin(REPRESENTATIVE_ISSUANCE_COUNTS)
    ].sort_values("issued_count", kind="stable")
    if selected["issued_count"].astype(int).tolist() != list(
        REPRESENTATIVE_ISSUANCE_COUNTS
    ):
        raise ValueError("Missing predeclared 25%, 50%, or 75% issuance point")

    random_frame = inputs.random_rejection
    if len(random_frame) != 10_000 or set(random_frame["status"]) != {"defined"}:
        raise ValueError("Expected 10,000 fully defined random rankings")
    random_rates = pd.to_numeric(
        random_frame["catastrophic_rate"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(random_rates).all():
        raise ValueError("Random-ranking catastrophic rates must be finite")
    return selected, float(random_rates.mean())


def _family_risk_reductions(
    inputs: FigureInputs,
) -> tuple[np.ndarray, np.ndarray]:
    test = inputs.family_metrics.set_index("truth_family")
    if not set(TRUTH_FAMILY_ORDER).issubset(test.index):
        raise ValueError("Test family metrics are incomplete")
    test_values = (
        pd.to_numeric(
            test.loc[
                list(TRUTH_FAMILY_ORDER),
                "issued_vs_analytic_random_risk_reduction_fraction",
            ],
            errors="raise",
        )
        .to_numpy(dtype=float)
        * 100.0
    )

    audit_records = inputs.report.get("secondary", {}).get(
        "audit_family_metrics", []
    )
    audit = pd.DataFrame(audit_records).set_index("truth_family")
    if not set(TRUTH_FAMILY_ORDER).issubset(audit.index):
        raise ValueError("Audit family metrics are incomplete")
    audit_values = (
        pd.to_numeric(
            audit.loc[
                list(TRUTH_FAMILY_ORDER),
                "issued_vs_analytic_random_risk_reduction_fraction",
            ],
            errors="raise",
        )
        .to_numpy(dtype=float)
        * 100.0
    )
    if not np.isfinite(test_values).all() or not np.isfinite(audit_values).all():
        raise ValueError("Family risk reductions must be finite")
    return test_values, audit_values


def _forecast_horizon_metrics(inputs: FigureInputs) -> pd.DataFrame:
    frame = inputs.forecast_day_metrics
    selected = frame.loc[
        frame["truth_family"].astype(str).eq("__all__")
        & frame["model_id"].astype(str).eq("candidate")
    ].copy()
    selected["forecast_day"] = pd.to_numeric(
        selected["forecast_day"], errors="raise"
    )
    selected = selected.sort_values("forecast_day", kind="stable")
    if len(selected) != 8:
        raise ValueError("Expected eight candidate forecast horizons")
    years = selected["forecast_day"].to_numpy(dtype=float) / 365.25
    expected_years = np.asarray([3, 4, 5, 7, 10, 15, 20, 25], dtype=float)
    if not np.allclose(years, expected_years, rtol=0.0, atol=1e-12):
        raise ValueError("Forecast horizons differ from the frozen 3-25 year grid")
    for column in (
        "mean_absolute_error_pp_among_finite",
        "median_absolute_error_pp_among_finite",
        "p90_absolute_error_pp_among_finite",
    ):
        selected[column] = pd.to_numeric(selected[column], errors="raise")
        if not np.isfinite(selected[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Forecast metric {column} must be finite")
    selected["forecast_year"] = years
    return selected


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LIGHT_GRAY)
    axis.spines["bottom"].set_color(LIGHT_GRAY)
    axis.tick_params(colors=INK)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.72)
    axis.set_axisbelow(True)


def build_figure(evidence_root: Path, output: Path) -> Path:
    inputs = _load_inputs(evidence_root)
    _require_frozen_failure(inputs.report)
    endpoint_ratios, endpoint_labels = _primary_endpoint_values(inputs.report)
    representative_risk, empirical_random_rate = _representative_risk(inputs)
    test_family, audit_family = _family_risk_reductions(inputs)
    horizon = _forecast_horizon_metrics(inputs)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(15.8, 9.6))
    figure.patch.set_facecolor("white")

    axis = axes[0, 0]
    positions = np.arange(3)
    colors = [RED, RED, TEAL]
    axis.barh(
        positions,
        endpoint_ratios,
        height=0.52,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    axis.axvline(
        1.0,
        color=INK,
        linewidth=1.3,
        linestyle="--",
        label="frozen boundary",
    )
    axis.set_yticks(
        positions,
        [
            "Risk reduction\n(higher required)",
            "Matched-pair rejection\n(higher required)",
            "Issued IAE delta\n(lower required)",
        ],
    )
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.18)
    axis.set_xticks([0.0, 0.5, 1.0], ["0", "0.5x", "1.0x gate"])
    for index, (ratio, label) in enumerate(zip(endpoint_ratios, endpoint_labels)):
        inside_bar = ratio > 0.55
        axis.text(
            ratio - 0.02 if inside_bar else ratio + 0.025,
            index,
            label,
            va="center",
            ha="right" if inside_bar else "left",
            fontsize=9,
            fontweight="bold",
            color="white" if inside_bar else INK,
        )
    for index, (status, color) in enumerate(
        (("FAIL", RED), ("FAIL", RED), ("PASS", TEAL))
    ):
        axis.text(
            1.035,
            index,
            status,
            ha="left",
            va="center",
            color=color,
            fontsize=8.7,
            fontweight="bold",
        )
    axis.set_title("A  Frozen primary endpoints: 1 of 3 passed", loc="left")
    axis.legend(frameon=False, loc="lower right", fontsize=8.5)
    _style_axis(axis)
    axis.grid(False)

    axis = axes[0, 1]
    coverage_pct = (
        representative_risk["coverage_fraction_of_all_test_clusters"].to_numpy(
            dtype=float
        )
        * 100.0
    )
    catastrophic_pct = (
        representative_risk["catastrophic_rate"].to_numpy(dtype=float) * 100.0
    )
    axis.plot(
        coverage_pct,
        catastrophic_pct,
        color=TEAL,
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="structure-disagreement issuance",
    )
    axis.axhline(
        empirical_random_rate * 100.0,
        color=GOLD,
        linewidth=2.0,
        linestyle="--",
        label="10,000 random 50% rankings (mean)",
    )
    for x_value, y_value in zip(coverage_pct, catastrophic_pct):
        axis.annotate(
            f"{y_value:.1f}%",
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axis.text(
        74.0,
        empirical_random_rate * 100.0 + 0.7,
        f"random mean {empirical_random_rate * 100:.2f}%",
        ha="right",
        va="bottom",
        color="#7A5A18",
        fontsize=8.8,
    )
    axis.set_xlim(20, 80)
    axis.set_ylim(28, 46)
    axis.set_xticks([25, 50, 75])
    axis.set_xlabel("issued share of all test clusters")
    axis.set_ylabel("catastrophic endpoint error rate")
    axis.set_yticks([30, 35, 40, 45], ["30%", "35%", "40%", "45%"])
    axis.set_title("B  Selective risk signal is positive but insufficient", loc="left")
    axis.legend(frameon=False, loc="lower right", fontsize=8.3)
    _style_axis(axis)

    axis = axes[1, 0]
    positions = np.arange(len(TRUTH_FAMILY_ORDER), dtype=float)
    width = 0.35
    test_bars = axis.bar(
        positions - width / 2,
        test_family,
        width,
        color=TEAL,
        label="test (1,000 clusters)",
    )
    audit_bars = axis.bar(
        positions + width / 2,
        audit_family,
        width,
        color=GOLD,
        label="audit (500 clusters)",
    )
    knee_index = TRUTH_FAMILY_ORDER.index("late_knee")
    for bar in (test_bars[knee_index], audit_bars[knee_index]):
        bar.set_edgecolor(RED)
        bar.set_linewidth(2.2)
        bar.set_hatch("///")
    axis.axhline(0.0, color=INK, linewidth=1.0)
    axis.axvspan(
        knee_index - 0.48,
        knee_index + 0.48,
        color=LIGHT_RED,
        alpha=0.10,
        linewidth=0,
    )
    axis.annotate(
        "late-knee reversal\non both held-out splits",
        xy=(knee_index, min(test_family[knee_index], audit_family[knee_index])),
        xytext=(knee_index - 0.65, 12),
        arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.1},
        ha="center",
        va="bottom",
        color=RED,
        fontsize=8.8,
        fontweight="bold",
    )
    axis.set_xticks(
        positions,
        [
            "single\npower",
            "dual\npower",
            "saturating\n+ slow",
            "early\nactivation",
            "late\nknee",
        ],
    )
    axis.get_xticklabels()[knee_index].set_color(RED)
    axis.get_xticklabels()[knee_index].set_fontweight("bold")
    axis.set_ylabel("risk reduction vs analytic random")
    axis.set_yticks([-10, 0, 20, 40, 60], ["-10%", "0%", "20%", "40%", "60%"])
    axis.set_ylim(-10, 68)
    axis.set_title("C  Family analysis locates the failure mode", loc="left")
    axis.legend(frameon=False, loc="upper right", fontsize=8.5)
    _style_axis(axis)

    axis = axes[1, 1]
    years = horizon["forecast_year"].to_numpy(dtype=float)
    mean_error = horizon[
        "mean_absolute_error_pp_among_finite"
    ].to_numpy(dtype=float)
    median_error = horizon[
        "median_absolute_error_pp_among_finite"
    ].to_numpy(dtype=float)
    p90_error = horizon["p90_absolute_error_pp_among_finite"].to_numpy(dtype=float)
    axis.fill_between(
        years,
        median_error,
        p90_error,
        color=LIGHT_GRAY,
        alpha=0.65,
        label="median to 90th percentile",
    )
    axis.plot(
        years,
        mean_error,
        color=TEAL,
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        label="mean absolute error",
    )
    axis.plot(
        years,
        median_error,
        color=GRAY,
        linewidth=1.6,
        linestyle="--",
        label="median absolute error",
    )
    axis.plot(
        years,
        p90_error,
        color=RED,
        linewidth=1.4,
        linestyle=":",
        label="90th percentile",
    )
    axis.annotate(
        f"25-year mean {mean_error[-1]:.2f} pp\n90th pct {p90_error[-1]:.2f} pp",
        xy=(years[-1], mean_error[-1]),
        xytext=(18.0, 13.5),
        arrowprops={"arrowstyle": "->", "color": TEAL, "linewidth": 1.1},
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
    )
    axis.set_xlim(2.5, 25.8)
    axis.set_ylim(0, 26)
    axis.set_xticks(years.astype(int))
    axis.set_xlabel("forecast horizon (years)")
    axis.set_ylabel("absolute error against latent synthetic truth (pp)")
    axis.set_title("D  Forecast error widens across the 3-25 year horizon", loc="left")
    axis.legend(frameon=False, loc="upper left", fontsize=8.2)
    _style_axis(axis)

    figure.suptitle(
        "LifeTwin v0.14 frozen synthetic stress test: V1 fails 2 of 3 endpoints",
        fontsize=16,
        fontweight="bold",
        y=0.982,
        color=INK,
    )
    figure.text(
        0.5,
        0.942,
        (
            "Execution and safety gates passed; the candidate performance gates did not. "
            "Known synthetic truth only."
        ),
        ha="center",
        va="center",
        fontsize=10.2,
        color="#4F5C62",
    )
    figure.text(
        0.5,
        0.018,
        (
            "Frozen V1 result: failure, not void or inconclusive. This experiment does "
            "not validate real LFP, Hithium products, storage stations, or 15-25 year accuracy."
        ),
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#4F5C62",
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.975,
        bottom=0.10,
        top=0.89,
        hspace=0.38,
        wspace=0.29,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=180,
        facecolor="white",
        metadata={"Software": "LifeTwin v0.14 evidence figure"},
    )
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen LifeTwin v0.14 synthetic failure figure."
    )
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_figure(args.evidence_root, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
