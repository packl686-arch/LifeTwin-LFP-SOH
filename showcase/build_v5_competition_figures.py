from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5 = (
    PROJECT_ROOT
    / "artifacts/fastcharge-v5-pairwise-development/selected_evaluation_summary.json"
)
DEFAULT_V2_SCORES = PROJECT_ROOT / "artifacts/fastcharge-safe-prior-v2/scores.csv"
DEFAULT_UQ = (
    PROJECT_ROOT
    / "artifacts/fastcharge-v5-support-uncertainty/score_summary.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/assets/v5_fastcharge_development_results.png"


def build_figure(
    v5_path: Path,
    v2_scores_path: Path,
    uq_path: Path,
    output_path: Path,
) -> None:
    summary = json.loads(v5_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uq_path.read_text(encoding="utf-8"))
    v2_scores = pd.read_csv(v2_scores_path)
    hard = v2_scores.loc[
        v2_scores["model_id"] == "safe_hard_local_risk_selector"
    ]
    nearest = v2_scores.loc[
        v2_scores["model_id"] == "nearest_neighbor_delta_transfer"
    ]
    overall = [
        float(nearest["trajectory_mae_pp"].mean()),
        float(hard["trajectory_mae_pp"].mean()),
        float(summary["overall"]["trajectory_mae_pp"]),
        float(uncertainty["overall"]["trajectory_mae_pp"]),
    ]
    prefixes = [int(item["prefix_cycle"]) for item in summary["by_prefix"]]
    candidate_by_prefix = [
        float(item["trajectory_mae_pp"]) for item in summary["by_prefix"]
    ]
    hard_by_prefix = [
        float(
            hard.loc[hard["prefix_cycle"] == prefix, "trajectory_mae_pp"].mean()
        )
        for prefix in prefixes
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "figure.facecolor": "#F7F8FA",
            "axes.facecolor": "#FFFFFF",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), dpi=160)
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.82,
        bottom=0.11,
        hspace=0.48,
        wspace=0.28,
    )
    figure.suptitle(
        "LifeTwin V5 | FastCharge outcome-exposed development evidence",
        x=0.07,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17202A",
    )
    figure.text(
        0.07,
        0.89,
        (
            "81 evaluation cells | prefixes 20/40/60/100 | forecast through "
            "cycle 300 | physical-cell grouped selection"
        ),
        ha="left",
        fontsize=10.5,
        color="#566573",
    )

    colors = ["#5B6573", "#2F6B8A", "#1F8A70", "#D97706"]
    labels = ["Fixed neighbor", "V2 safe hard", "V5 pairwise", "V5 gated"]
    bars = axes[0, 0].bar(labels, overall, color=colors, width=0.62)
    axes[0, 0].set_title("A. Overall trajectory MAE")
    axes[0, 0].set_ylabel("Percentage points (lower is better)")
    axes[0, 0].set_ylim(0.0, max(overall) * 1.28)
    axes[0, 0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[0, 0].set_axisbelow(True)
    axes[0, 0].tick_params(axis="x", rotation=10)
    for bar, value in zip(bars, overall, strict=True):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#17202A",
        )
    improvement = float(
        summary["comparisons"]["safe_hard_local_risk_selector"][
            "relative_improvement"
        ]
    )
    axes[0, 0].text(
        0.98,
        0.95,
        f"{improvement:.1%} lower vs V2",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="#1F8A70",
        fontweight="bold",
    )

    positions = np.arange(len(prefixes), dtype=float)
    width = 0.36
    axes[0, 1].bar(
        positions - width / 2,
        hard_by_prefix,
        width,
        label="V2 safe hard",
        color="#2F6B8A",
    )
    axes[0, 1].bar(
        positions + width / 2,
        candidate_by_prefix,
        width,
        label="V5 pairwise",
        color="#1F8A70",
    )
    axes[0, 1].set_title("B. Performance as evidence accumulates")
    axes[0, 1].set_xlabel("Observed prefix cycle")
    axes[0, 1].set_ylabel("MAE (percentage points)")
    axes[0, 1].set_xticks(positions, [str(value) for value in prefixes])
    axes[0, 1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[0, 1].set_axisbelow(True)
    axes[0, 1].legend(frameon=False, loc="upper right")

    metric_labels = ["Mean interval width", "Single-interval WIS"]
    baseline_ratios = [1.0, 1.0]
    candidate_ratios = [
        float(uncertainty["width_ratio_vs_v2"]),
        float(uncertainty["wis_ratio_vs_v2"]),
    ]
    metric_positions = np.arange(len(metric_labels), dtype=float)
    interval_width = 0.34
    axes[1, 0].bar(
        metric_positions - interval_width / 2,
        baseline_ratios,
        interval_width,
        label="V2 = 100%",
        color="#2F6B8A",
    )
    candidate_bars = axes[1, 0].bar(
        metric_positions + interval_width / 2,
        candidate_ratios,
        interval_width,
        label="V5 gated",
        color="#D97706",
    )
    axes[1, 0].set_title("C. 90% interval efficiency")
    axes[1, 0].set_ylabel("Ratio to V2 (lower is better)")
    axes[1, 0].set_xticks(metric_positions, metric_labels)
    axes[1, 0].set_ylim(0.0, 1.18)
    axes[1, 0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[1, 0].set_axisbelow(True)
    axes[1, 0].legend(frameon=False, loc="upper right")
    for bar, value in zip(candidate_bars, candidate_ratios, strict=True):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            color="#9A5A00",
            fontweight="bold",
        )
    axes[1, 0].text(
        0.02,
        0.96,
        (
            f"Coverage: V2 {uncertainty['v2_safe_hard']['empirical_coverage']:.1%}"
            f" | V5 {uncertainty['overall']['empirical_coverage']:.1%}"
        ),
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        color="#17202A",
        fontweight="bold",
    )

    axes[1, 1].axis("off")
    axes[1, 1].set_title("D. What this evidence does and does not show", loc="left")
    axes[1, 1].text(
        0.02,
        0.9,
        (
            "SUPPORTED\n"
            "- Physical-cell five-fold model selection\n"
            "- Held-out cells excluded from both pair roles\n"
            "- Prediction hashes sealed before scoring\n"
            "- Cell-level bootstrap and interval diagnostics"
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#1F6F5C",
        linespacing=1.55,
    )
    axes[1, 1].text(
        0.02,
        0.42,
        (
            "NOT CLAIMED\n"
            "- Independent or cross-domain confirmation\n"
            "- Hithium product accuracy\n"
            "- Calendar-aging validation\n"
            "- 15-25 year accuracy"
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#8E3B2F",
        linespacing=1.55,
    )

    figure.text(
        0.07,
        0.035,
        (
            "Development evidence only: public outcomes were previously exposed; "
            "not Hithium validation and not a 15-25 year accuracy claim."
        ),
        ha="left",
        fontsize=9.5,
        color="#7B241C",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-summary", type=Path, default=DEFAULT_V5)
    parser.add_argument("--v2-scores", type=Path, default=DEFAULT_V2_SCORES)
    parser.add_argument("--uq-summary", type=Path, default=DEFAULT_UQ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_figure(args.v5_summary, args.v2_scores, args.uq_summary, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
