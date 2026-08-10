"""Build the compact V7 competition evidence figure from public JSON results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "showcase/evidence_v6/gated_state_decision.json"
V7 = ROOT / "showcase/evidence_v7/reissue_innovation_decision.json"
OUTPUT = ROOT / "docs/assets/v7_reissue_innovation_results.png"


def _transition(decision: dict[str, object], prefix: int) -> dict[str, object]:
    key = (
        "nested_and_batch_future_blind_nomination"
        if "nested_and_batch_future_blind_nomination" in decision
        else "nested_future_blind_nomination"
    )
    return next(
        row
        for row in decision[key]["transitions"]
        if row["current_prefix_cycle"] == prefix
    )


def main() -> None:
    v6 = json.loads(V6.read_text(encoding="utf-8"))
    v7 = json.loads(V7.read_text(encoding="utf-8"))
    v6_p100 = _transition(v6, 100)
    v7_rows = [_transition(v7, prefix) for prefix in (40, 60, 100)]
    v7_p100 = v7_rows[-1]

    base = float(v7_p100["mean_base_trajectory_mae_pp"])
    v6_mae = base + float(v6_p100["mean_gated_delta_mae_pp"])
    v7_mae = float(v7_p100["mean_updated_trajectory_mae_pp"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#253238",
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=160)
    figure.patch.set_facecolor("#f6f8f7")
    figure.suptitle(
        "V7 reissue-aware innovation state | training-only evidence",
        fontsize=17,
        fontweight="bold",
        x=0.06,
        ha="left",
        color="#172126",
    )
    figure.text(
        0.06,
        0.925,
        "41 cross-fit training cells; V5 remains active until a new outcome-blind test passes",
        color="#536168",
        fontsize=10.5,
    )

    ax = axes[0, 0]
    labels = ["V5 center", "V6.1 gate", "V7 innovation"]
    values = [base, v6_mae, v7_mae]
    colors = ["#647780", "#d09a2d", "#147d73"]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("P100 mean trajectory MAE", loc="left", fontweight="bold")
    ax.set_ylabel("retention percentage points")
    ax.set_ylim(0.18, 0.255)
    ax.grid(axis="y", color="#d8dfdc", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#172126",
        )
    ax.text(
        0.98,
        0.88,
        "V7: -15.32% vs V5",
        transform=ax.transAxes,
        ha="right",
        color="#147d73",
        fontweight="bold",
    )

    ax = axes[0, 1]
    precision = [
        100.0 * float(v6_p100["activation_precision"]),
        100.0 * float(v7_p100["cell_holdout"]["activation_precision"]),
    ]
    worst = [
        float(v6_p100["active_maximum_delta_mae_pp"]),
        float(v7_p100["cell_holdout"]["active_maximum_delta_mae_pp"]),
    ]
    bars = ax.bar(
        ["V6.1", "V7"], precision, color=["#d09a2d", "#147d73"], width=0.55
    )
    ax.set_title("P100 activation safety", loc="left", fontweight="bold")
    ax.set_ylabel("activated cells improved (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", color="#d8dfdc", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value, tail in zip(bars, precision, worst, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value:.0f}%",
            ha="center",
            fontweight="bold",
        )
        tail_color = "#b43c37" if tail > 0 else "#ffffff"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            7,
            f"worst {tail:+.4f} pp",
            ha="center",
            color=tail_color,
            fontsize=8.5,
            fontweight="bold",
        )

    ax = axes[1, 0]
    prefixes = ["P40", "P60", "P100"]
    deltas = [float(row["cell_holdout"]["mean_gated_delta_mae_pp"]) for row in v7_rows]
    colors = ["#4f8db3", "#b43c37", "#147d73"]
    bars = ax.bar(prefixes, deltas, color=colors, width=0.58)
    ax.axhline(0.0, color="#253238", linewidth=0.9)
    ax.set_title("Nested all-cell gated delta", loc="left", fontweight="bold")
    ax.set_ylabel("delta MAE (pp); lower is better")
    ax.grid(axis="y", color="#d8dfdc", linewidth=0.7)
    ax.set_axisbelow(True)
    verdicts = ["batch reject", "cell + batch reject", "blind candidate"]
    label_y = [-0.011, 0.003, -0.030]
    label_color = ["#172126", "#172126", "#ffffff"]
    for bar, value, verdict, y, color in zip(
        bars, deltas, verdicts, label_y, label_color, strict=True
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.4f}\n{verdict}",
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=color,
        )
    ax.set_ylim(-0.047, 0.012)

    ax = axes[1, 1]
    batches = v7_p100["batch_holdout"]["batches"]
    batch_labels = [f"hold out Batch {row['held_out_batch']}" for row in batches]
    batch_deltas = [float(row["mean_gated_delta_mae_pp"]) for row in batches]
    bars = ax.barh(batch_labels, batch_deltas, color=["#4f8db3", "#147d73"])
    ax.axvline(0.0, color="#253238", linewidth=0.9)
    ax.set_title("P100 batch-shift stress test", loc="left", fontweight="bold")
    ax.set_xlabel("all-cell gated delta MAE (pp)")
    ax.grid(axis="x", color="#d8dfdc", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value, row in zip(bars, batch_deltas, batches, strict=True):
        if abs(value) > 0.02:
            x = value / 2.0
            alignment = "center"
            color = "#ffffff"
        else:
            x = value - 0.002
            alignment = "right"
            color = "#172126"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f} | {row['activation_count']} active | 100% precision",
            ha=alignment,
            va="center",
            color=color,
            fontweight="bold",
            fontsize=8.5,
        )
    ax.set_xlim(-0.078, 0.008)

    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.06,
        0.018,
        "Boundary: outcome-informed development on the same 41 cells; no Hithium, calendar-aging, or 15-25 year claim.",
        color="#69777d",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.045, 0.055, 0.99, 0.89), h_pad=2.3, w_pad=2.3)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
