from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "showcase/results"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/showcase/phase8_results.png"
PRIMARY_PREFIX = 10
SQRT_METHOD = "target_prefix_only_sqrt_time_v1"
V3_METHOD = "mechanism_gated_target_activation_offset_hybrid_v1"
SCENARIOS = (
    "v3_unseen_temperature_level",
    "v3_soc_interpolation_at_40c",
)
SCENARIO_LABELS = {
    "v3_unseen_temperature_level": "Unseen temperature",
    "v3_soc_interpolation_at_40c": "40 C SOC interpolation",
}


def load_summary(results_root: Path = RESULTS_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = pd.read_csv(results_root / "comparison_summary.csv")
    sensitivity = pd.read_csv(results_root / "tau_sensitivity_summary.csv")
    required_comparison = {
        "scenario",
        "prefix_checkups",
        "candidate_method",
        "candidate_trajectory_iae_pp_mean",
        "comparator_trajectory_iae_pp_mean",
    }
    required_sensitivity = {
        "scenario",
        "activation_timescale_days",
        "candidate_trajectory_iae_pp_mean",
        "comparator_trajectory_iae_pp_mean",
    }
    if not required_comparison <= set(comparisons):
        raise ValueError("Phase 8 comparison summary schema is incomplete")
    if not required_sensitivity <= set(sensitivity):
        raise ValueError("Phase 8 tau summary schema is incomplete")
    return comparisons, sensitivity


def model_matrix(comparisons: pd.DataFrame) -> pd.DataFrame:
    primary = comparisons.loc[comparisons["prefix_checkups"] == PRIMARY_PREFIX]
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        group = primary.loc[primary["scenario"] == scenario]
        sqrt = group.loc[group["candidate_method"] == SQRT_METHOD]
        v3 = group.loc[group["candidate_method"] == V3_METHOD]
        if len(sqrt) != 1 or len(v3) != 1:
            raise ValueError(f"Missing primary comparison row for {scenario}")
        rows.append(
            {
                "scenario": scenario,
                "Traditional sqrt": float(
                    sqrt["candidate_trajectory_iae_pp_mean"].iloc[0]
                ),
                "Hierarchical V2": float(
                    v3["comparator_trajectory_iae_pp_mean"].iloc[0]
                ),
                "Gated activation V3": float(
                    v3["candidate_trajectory_iae_pp_mean"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def build_figure(
    matrix: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    output: Path,
) -> None:
    from matplotlib import pyplot as plt

    colors = ("#737373", "#0F766E", "#C2410C")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    x = np.arange(len(matrix))
    width = 0.23
    for offset, (method, color) in enumerate(zip(matrix.columns, colors, strict=True)):
        values = matrix[method].to_numpy(dtype=float)
        bars = axes[0].bar(
            x + (offset - 1) * width,
            values,
            width,
            label=method,
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_xticks(
        x,
        [SCENARIO_LABELS[value] for value in matrix.index],
    )
    axes[0].set_ylabel("Mean trajectory IAE (pp, lower is better)")
    axes[0].set_title("Model progression at prefix 10")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    line_colors = ("#2563EB", "#C2410C")
    for scenario, color in zip(SCENARIOS, line_colors, strict=True):
        group = sensitivity.loc[sensitivity["scenario"] == scenario].sort_values(
            "activation_timescale_days"
        )
        axes[1].plot(
            group["activation_timescale_days"],
            group["candidate_trajectory_iae_pp_mean"],
            marker="o",
            linewidth=2,
            color=color,
            label=SCENARIO_LABELS[scenario],
        )
        baseline = float(group["comparator_trajectory_iae_pp_mean"].iloc[0])
        axes[1].axhline(baseline, color=color, linestyle="--", alpha=0.45)
    axes[1].axvline(7.0, color="#111827", linestyle=":", linewidth=1.3)
    axes[1].annotate(
        "post-hoc primary tau = 7 d",
        xy=(7.0, 0.21),
        xytext=(10.0, 0.26),
        arrowprops={"arrowstyle": "->", "color": "#111827"},
        fontsize=8,
    )
    axes[1].set_xlabel("Activation timescale tau (day)")
    axes[1].set_ylabel("Mean trajectory IAE (pp)")
    axes[1].set_title("Timescale sensitivity at prefix 10")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    figure.suptitle(
        "LifeTwin Phase 8: retrospective public-data development results",
        fontsize=13,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the public LifeTwin Phase 8 analysis figure."
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    comparisons, sensitivity = load_summary(args.results_root)
    matrix = model_matrix(comparisons)
    build_figure(matrix, sensitivity, output=args.output)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "results_root": args.results_root.as_posix(),
                "prefix_checkups": PRIMARY_PREFIX,
                "trajectory_iae_pp": matrix.to_dict(orient="index"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
