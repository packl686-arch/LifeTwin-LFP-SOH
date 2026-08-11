"""Build the competition figure for the frozen V7 robustness rejection."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "showcase/evidence_v7_robustness/decision.json"
OUTPUT = ROOT / "docs/assets/v7_prefix_robustness_audit.png"


def main() -> None:
    result = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    summaries = {row["scenario_id"]: row for row in result["scenario_summaries"]}
    ids = [
        "reference_unperturbed",
        "iid_noise_sigma_0p02_pp",
        "iid_noise_sigma_0p05_pp",
        "random_missing_10pct",
        "noise_0p05_pp_plus_missing_10pct",
    ]
    labels = ["Reference", "Noise 0.02", "Noise 0.05", "10% missing", "Noise + missing"]
    rows = [summaries[key] for key in ids]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#26343a",
            "axes.linewidth": 0.8,
        }
    )
    figure = plt.figure(figsize=(12.8, 7.2), dpi=160, facecolor="#f4f7f6")
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.72])
    ax_stability = figure.add_subplot(grid[0, 0])
    ax_tail = figure.add_subplot(grid[0, 1])
    ax_decision = figure.add_subplot(grid[1, :])
    figure.suptitle(
        "Frozen V7 gate | prefix robustness audit",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#182328",
    )
    figure.text(
        0.055,
        0.925,
        "41 training cells, 128 deterministic Monte Carlo repeats per stochastic scenario; no threshold retuning",
        color="#58676d",
        fontsize=10.5,
    )

    x = np.arange(len(labels))
    width = 0.25
    agreement = [100.0 * float(row["decision_agreement"]) for row in rows]
    false_activation = [100.0 * float(row["false_activation_rate"]) for row in rows]
    precision = [100.0 * float(row["activation_precision"]) for row in rows]
    ax_stability.bar(
        x - width,
        agreement,
        width,
        label="Decision agreement",
        color="#247d78",
    )
    ax_stability.bar(
        x,
        precision,
        width,
        label="Activation precision",
        color="#4e82a8",
    )
    ax_stability.bar(
        x + width,
        false_activation,
        width,
        label="False activation",
        color="#c74a43",
    )
    ax_stability.set_title("Gate stability", loc="left", fontweight="bold")
    ax_stability.set_ylabel("percent")
    ax_stability.set_xticks(x, labels, rotation=18, ha="right")
    ax_stability.set_ylim(0, 112)
    ax_stability.grid(axis="y", color="#d7dfdc", linewidth=0.7)
    ax_stability.legend(frameon=False, fontsize=8.5, loc="upper right")

    tail = [float(row["p95_replicate_active_max_delta_mae_pp"]) for row in rows]
    colors = ["#247d78" if value <= 0.1 else "#c74a43" for value in tail]
    bars = ax_tail.bar(labels, tail, color=colors, width=0.62)
    ax_tail.axhline(
        0.1,
        color="#c74a43",
        linestyle="--",
        linewidth=1.1,
        label="registered maximum +0.10 pp",
    )
    ax_tail.axhline(0.0, color="#26343a", linewidth=0.9)
    ax_tail.set_title("Tail harm", loc="left", fontweight="bold")
    ax_tail.set_ylabel("P95 replicate worst active delta MAE (pp)")
    ax_tail.set_xticks(x, labels, rotation=18, ha="right")
    ax_tail.set_ylim(-0.04, 0.175)
    ax_tail.grid(axis="y", color="#d7dfdc", linewidth=0.7)
    ax_tail.legend(frameon=False, fontsize=8.5, loc="upper left")
    for bar, value in zip(bars, tail, strict=True):
        ax_tail.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.007 if value >= 0 else -0.012),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
            fontweight="bold",
        )

    ax_decision.axis("off")
    boxes = [
        (0.01, "Training audit", "9 / 9 improved", "#247d78"),
        (0.265, "Frozen stress", "3 required scenarios failed", "#c78d2e"),
        (0.52, "Scientific decision", "Withdraw V7 candidate", "#c74a43"),
        (0.775, "Current model", "V5 remains champion", "#354e59"),
    ]
    for left, title, value, color in boxes:
        ax_decision.add_patch(
            plt.Rectangle(
                (left, 0.28),
                0.215,
                0.48,
                transform=ax_decision.transAxes,
                facecolor="#ffffff",
                edgecolor=color,
                linewidth=1.6,
            )
        )
        ax_decision.text(
            left + 0.018,
            0.64,
            title,
            transform=ax_decision.transAxes,
            color="#67767c",
            fontsize=9,
            fontweight="bold",
        )
        ax_decision.text(
            left + 0.018,
            0.42,
            value,
            transform=ax_decision.transAxes,
            color=color,
            fontsize=12,
            fontweight="bold",
        )
    for left in (0.235, 0.49, 0.745):
        ax_decision.annotate(
            "",
            xy=(left + 0.02, 0.52),
            xytext=(left, 0.52),
            xycoords=ax_decision.transAxes,
            arrowprops={"arrowstyle": "->", "color": "#7b898e", "lw": 1.3},
        )
    ax_decision.text(
        0.01,
        0.08,
        "Boundary: same 41 outcome-exposed training cells; innovation layer only; no 81-cell, Hithium, calendar-aging, or 15-25 year claim.",
        transform=ax_decision.transAxes,
        color="#66757b",
        fontsize=9.2,
    )

    for axis in (ax_stability, ax_tail):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    figure.tight_layout(rect=(0.04, 0.03, 0.99, 0.9), h_pad=1.8, w_pad=2.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
