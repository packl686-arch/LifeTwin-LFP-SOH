from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISION = ROOT / "showcase/evidence_v5/dynamic_landmark_decision.json"
DEFAULT_CANDIDATES = (
    ROOT / "showcase/evidence_v5/dynamic_landmark_training_candidate_summary.csv"
)
DEFAULT_OUTPUT = ROOT / "docs/assets/v5_dynamic_landmark_audit.png"


def _reissue_panel(axis: plt.Axes, rows: list[dict[str, object]], title: str) -> None:
    labels = [
        f"P{row['previous_prefix_cycle']} to P{row['current_prefix_cycle']}"
        for row in rows
    ]
    previous = [float(row["mean_previous_trajectory_mae_pp"]) for row in rows]
    current = [float(row["mean_current_trajectory_mae_pp"]) for row in rows]
    positions = np.arange(len(rows), dtype=float)
    width = 0.34
    axis.bar(
        positions - width / 2,
        previous,
        width,
        color="#88939D",
        label="Previous issue",
    )
    bars = axis.bar(
        positions + width / 2,
        current,
        width,
        color="#2F6F91",
        label="Current-prefix reissue",
    )
    axis.set_title(title)
    axis.set_ylabel("Trajectory MAE (percentage points)")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, max(previous + current) * 1.28)
    axis.grid(axis="y", color="#E3E7E5", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, row in zip(bars, rows, strict=True):
        value = float(row["mean_current_trajectory_mae_pp"])
        improvement = float(row["relative_mae_improvement"])
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(previous + current) * 0.035,
            f"{value:.3f}\n-{improvement:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#214F68",
            fontweight="bold",
        )
    axis.legend(frameon=False, loc="upper right")


def build_figure(decision_path: Path, candidates_path: Path, output_path: Path) -> None:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidates_path)
    gp = candidates.loc[candidates["candidate_family"] == "fixed_gaussian_process"]
    best_gp = (
        gp.sort_values(
            ["current_prefix_cycle", "mean_updated_trajectory_mae_pp"],
            kind="stable",
        )
        .groupby("current_prefix_cycle", sort=True)
        .head(1)
        .sort_values("current_prefix_cycle", kind="stable")
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 10.5,
            "figure.facecolor": "#F4F6F3",
            "axes.facecolor": "#FFFFFF",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 8.4), dpi=170)
    figure.subplots_adjust(
        left=0.075,
        right=0.98,
        top=0.82,
        bottom=0.10,
        hspace=0.48,
        wspace=0.28,
    )
    figure.suptitle(
        "LifeTwin V5 | Dynamic-landmark and online residual audit",
        x=0.075,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17211F",
    )
    figure.text(
        0.075,
        0.89,
        (
            "Training-only rule selection | 41 cross-fit training cells | "
            "81 outcome-exposed public evaluation cells | cycle 300 endpoint"
        ),
        ha="left",
        fontsize=10.5,
        color="#64716E",
    )

    _reissue_panel(
        axes[0, 0],
        decision["training_base_reissue"]["transitions"],
        "A. Training cross-fit: reissue with a longer prefix",
    )
    _reissue_panel(
        axes[0, 1],
        decision["public_evaluation_base_reissue"]["transitions"],
        "B. Public evaluation: reissue with a longer prefix",
    )

    labels = [f"P{int(value)}" for value in best_gp["current_prefix_cycle"]]
    values = best_gp["mean_delta_mae_pp"].to_numpy(dtype=float)
    positions = np.arange(len(best_gp), dtype=float)
    colors = ["#2F6F91" if value < 0.0 else "#C76716" for value in values]
    bars = axes[1, 0].barh(positions, values, color=colors, height=0.54)
    axes[1, 0].axvline(0.0, color="#24302D", linewidth=1.0)
    axes[1, 0].set_yticks(positions, labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("C. Best fixed-GP candidate at each landmark")
    axes[1, 0].set_xlabel("Delta trajectory MAE vs frozen V5 center (pp)")
    axes[1, 0].grid(axis="x", color="#E3E7E5", linewidth=0.8)
    axes[1, 0].set_axisbelow(True)
    span = max(abs(values.min()), abs(values.max())) * 1.55
    axes[1, 0].set_xlim(-span, span)
    for bar, (_, row) in zip(bars, best_gp.iterrows(), strict=True):
        value = float(row["mean_delta_mae_pp"])
        fraction = float(row["fraction_cells_improved"])
        if value >= 0.0:
            label_x, label_color = value + span * 0.035, "#17211F"
        elif value < -span * 0.5:
            label_x, label_color = value + span * 0.04, "#FFFFFF"
        else:
            label_x, label_color = -span * 0.97, "#17211F"
        axes[1, 0].text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f} pp | {fraction:.1%} cells",
            va="center",
            ha="left",
            color=label_color,
            fontsize=9.5,
            fontweight="bold",
        )
    axes[1, 0].text(
        0.01,
        0.96,
        "Activation gate: at least 70% of cells improve",
        transform=axes[1, 0].transAxes,
        color="#64716E",
        fontsize=9,
    )

    axes[1, 1].axis("off")
    axes[1, 1].set_title("D. Frozen decision", loc="left")
    evaluation = decision["public_evaluation_base_reissue"]["overall_transition_equal"]
    axes[1, 1].text(
        0.02,
        0.91,
        (
            "REISSUE SIGNAL\n"
            f"Mean MAE: {evaluation['mean_previous_trajectory_mae_pp']:.3f} to "
            f"{evaluation['mean_current_trajectory_mae_pp']:.3f} pp\n"
            f"Cell-clustered bootstrap delta 95% CI: "
            f"[{evaluation['physical_cell_clustered_bootstrap']['lower_delta_mae_pp']:.3f}, "
            f"{evaluation['physical_cell_clustered_bootstrap']['upper_delta_mae_pp']:.3f}] pp"
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        color="#17211F",
        fontsize=10.2,
        linespacing=1.35,
    )
    axes[1, 1].text(
        0.02,
        0.56,
        (
            "ONLINE RESIDUAL BRANCH\n"
            "No GP candidate reached the 70% cell-improvement gate.\n"
            "Only a 0.25x robust offset was eligible at P40 in training;\n"
            "it improved 66.7% of evaluation cells and failed activation."
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        color="#17211F",
        fontsize=10.2,
        linespacing=1.35,
    )
    axes[1, 1].text(
        0.02,
        0.23,
        (
            "DECISION\n"
            "Retain current-prefix V5 plus the conformal interval; GP correction stays off."
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        color="#17211F",
        fontsize=9.8,
        linespacing=1.25,
    )
    axes[1, 1].add_patch(
        Rectangle(
            (0.01, 0.0),
            0.98,
            0.065,
            transform=axes[1, 1].transAxes,
            facecolor="#F9E9E7",
            edgecolor="none",
        )
    )
    axes[1, 1].text(
        0.02,
        0.032,
        "Development evidence only; not Hithium, calendar-aging, or 15-25 year validation.",
        transform=axes[1, 1].transAxes,
        color="#B63B34",
        fontsize=9.2,
        fontweight="bold",
        va="center",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_figure(args.decision, args.candidates, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
