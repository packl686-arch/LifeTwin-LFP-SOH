from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "showcase/evidence_v012"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/assets/v012_robustness.png"
FALLBACK_ROUTE = "hierarchical_power_fallback"
PRIMARY_COVERAGE = 0.8


def _load_inputs(evidence_root: Path) -> tuple[pd.DataFrame, ...]:
    v4_root = evidence_root / "v4_calibration_robustness"
    geisbauer_root = evidence_root / "geisbauer_robustness"
    return (
        pd.read_csv(v4_root / "baseline_route_metrics.csv"),
        pd.read_csv(v4_root / "partition_route_metrics.csv"),
        pd.read_csv(geisbauer_root / "cell_paired_deltas.csv"),
        pd.read_csv(geisbauer_root / "leave_one_cell_out.csv"),
    )


def _fallback_80(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        frame["mean_route"].astype(str).eq(FALLBACK_ROUTE)
        & np.isclose(
            pd.to_numeric(frame["requested_coverage"]),
            PRIMARY_COVERAGE,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()
    selected["multiplier"] = pd.to_numeric(selected["multiplier"], errors="coerce")
    selected["diagnostic_mean_width_pp"] = pd.to_numeric(
        selected["diagnostic_mean_width_pp"], errors="coerce"
    )
    return selected


def build_figure(evidence_root: Path, output: Path) -> Path:
    baseline, partitions, cell_deltas, leave_one_out = _load_inputs(evidence_root)
    fallback_partitions = _fallback_80(partitions)
    fallback_baseline = _fallback_80(baseline)
    multipliers = fallback_partitions["multiplier"].dropna().to_numpy()
    widths = fallback_partitions["diagnostic_mean_width_pp"].dropna().to_numpy()
    if len(fallback_partitions) != 210 or len(multipliers) != 210:
        raise ValueError("Expected 210 finite fallback 80% partition audits")
    if len(fallback_baseline) != 1:
        raise ValueError("Expected one original fallback 80% split")
    if len(cell_deltas) != 15 or len(leave_one_out) != 15:
        raise ValueError("Expected the 15-cell Geisbauer robustness audit")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 5.2))
    figure.patch.set_facecolor("white")

    axis = axes[0]
    bins = np.linspace(float(multipliers.min()), float(multipliers.max()), 13)
    axis.hist(
        multipliers,
        bins=bins,
        color="#2B6F77",
        edgecolor="white",
        linewidth=0.8,
    )
    original_multiplier = float(fallback_baseline["multiplier"].iloc[0])
    axis.axvline(
        original_multiplier,
        color="#C43D3D",
        linewidth=2.0,
        label=f"original split: {original_multiplier:.2f}",
    )
    axis.set_title("A  V4 calibration multiplier")
    axis.set_xlabel("80% trajectory multiplier")
    axis.set_ylabel("partitions (out of 210)")
    axis.legend(frameon=False, loc="upper left")

    axis = axes[1]
    axis.hist(
        widths,
        bins=12,
        color="#D3942A",
        edgecolor="white",
        linewidth=0.8,
    )
    original_width = float(fallback_baseline["diagnostic_mean_width_pp"].iloc[0])
    axis.axvline(
        original_width,
        color="#C43D3D",
        linewidth=2.0,
        label=f"original split: {original_width:.2f} pp",
    )
    axis.set_title("B  V4 diagnostic interval width")
    axis.set_xlabel("mean width across fallback evaluations (pp)")
    axis.set_ylabel("partitions (out of 210)")
    axis.legend(frameon=False, loc="upper left")

    axis = axes[2]
    ordered = cell_deltas.sort_values(
        "paired_delta_trajectory_iae_pp", kind="stable"
    ).reset_index(drop=True)
    palette = {0.2: "#2B6F77", 0.5: "#D3942A", 1.0: "#C43D3D"}
    axis.axhspan(
        -0.05,
        0.05,
        color="#D9D9D9",
        alpha=0.65,
        label="post-hoc ±0.05 pp band",
    )
    for soc, group in ordered.groupby("storage_soc_fraction", sort=True):
        axis.scatter(
            group.index,
            group["paired_delta_trajectory_iae_pp"],
            s=42,
            color=palette[float(soc)],
            label=f"{float(soc) * 100:.0f}% SOC",
            zorder=3,
        )
    mean_delta = float(ordered["paired_delta_trajectory_iae_pp"].mean())
    axis.axhline(0.0, color="#222222", linewidth=1.0)
    axis.axhline(
        mean_delta,
        color="#6C4FA3",
        linewidth=1.8,
        linestyle="--",
        label=f"mean: {mean_delta:+.3f} pp",
    )
    axis.set_title("C  Geisbauer paired cell error")
    axis.set_xlabel("15 physical cells, sorted by paired delta")
    axis.set_ylabel("candidate minus sqrt IAE (pp)\npositive = candidate worse")
    axis.set_xticks([])
    axis.legend(frameon=False, loc="upper left", ncols=2, fontsize=8.2)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)

    figure.suptitle(
        "LifeTwin v0.12 robustness audit: calibration fragility and negative transfer",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Retrospective diagnostics only. The 210 partitions overlap; the "
            "Geisbauer cohort covers 120 days at 60 °C. Neither result supports "
            "formal coverage or 15–25 year accuracy claims."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.07, 0.99, 0.94), w_pad=2.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the LifeTwin v0.12 robustness audit figure."
    )
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_figure(args.evidence_root, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
