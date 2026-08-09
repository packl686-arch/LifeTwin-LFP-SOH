"""Dynamic-landmark residual audit around the frozen FastCharge V5 center."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


@dataclass(frozen=True)
class ResidualRule:
    candidate_id: str
    family: str
    statistic: str | None = None
    shrinkage: float = 0.0
    length_scale: float | None = None


REQUIRED_COLUMNS = {
    "cell_id",
    "prefix_cycle",
    "forecast_cycle",
    "observed_retention_pct",
    "candidate_prediction_pct",
}


def candidate_rules(config: Mapping[str, object]) -> tuple[ResidualRule, ...]:
    settings = config["candidate_rules"]
    robust = settings["robust_offset"]
    gp = settings["fixed_gaussian_process"]
    rules = [ResidualRule("no_update", "no_update")]
    for statistic in robust["statistics"]:
        for shrinkage in robust["shrinkage"]:
            token = str(shrinkage).replace(".", "p")
            rules.append(
                ResidualRule(
                    f"offset_{statistic}_a{token}",
                    "robust_offset",
                    statistic=str(statistic),
                    shrinkage=float(shrinkage),
                )
            )
    for length_scale in gp["normalized_cycle_length_scales"]:
        token = str(length_scale).replace(".", "p")
        rules.append(
            ResidualRule(
                f"gp_matern32_l{token}",
                "fixed_gaussian_process",
                length_scale=float(length_scale),
            )
        )
    return tuple(rules)


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise FastChargeV5PairwiseError(
            f"Dynamic-landmark input is missing columns: {sorted(missing)}"
        )
    result = frame.copy()
    for column in (
        "prefix_cycle",
        "forecast_cycle",
        "observed_retention_pct",
        "candidate_prediction_pct",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric = result[
        [
            "prefix_cycle",
            "forecast_cycle",
            "observed_retention_pct",
            "candidate_prediction_pct",
        ]
    ].to_numpy(dtype=float)
    if len(result) == 0 or not np.isfinite(numeric).all():
        raise FastChargeV5PairwiseError("Dynamic-landmark input is empty or invalid")
    if result.duplicated(["cell_id", "prefix_cycle", "forecast_cycle"]).any():
        raise FastChargeV5PairwiseError(
            "Dynamic-landmark input contains duplicate prediction coordinates"
        )
    if (result["forecast_cycle"] <= result["prefix_cycle"]).any():
        raise FastChargeV5PairwiseError(
            "Dynamic-landmark predictions must lie after their prefix"
        )
    return result.sort_values(
        ["cell_id", "prefix_cycle", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )


def _offset_statistic(residuals: np.ndarray, statistic: str) -> float:
    if statistic == "last_residual":
        return float(residuals[-1])
    if statistic == "median_last_5":
        return float(np.median(residuals[-5:]))
    if statistic == "median_last_10":
        return float(np.median(residuals[-10:]))
    if statistic == "median_all":
        return float(np.median(residuals))
    raise FastChargeV5PairwiseError(f"Unknown residual statistic: {statistic}")


def residual_correction(
    history_cycles: np.ndarray,
    history_residuals: np.ndarray,
    future_cycles: np.ndarray,
    *,
    previous_prefix: int,
    score_end_cycle: int,
    rule: ResidualRule,
    config: Mapping[str, object],
) -> np.ndarray:
    history_x = np.asarray(history_cycles, dtype=float).reshape(-1)
    residuals = np.asarray(history_residuals, dtype=float).reshape(-1)
    future_x = np.asarray(future_cycles, dtype=float).reshape(-1)
    if (
        len(history_x) == 0
        or history_x.shape != residuals.shape
        or len(future_x) == 0
        or not np.isfinite(np.concatenate([history_x, residuals, future_x])).all()
    ):
        raise FastChargeV5PairwiseError("Residual update coordinates are invalid")
    if (history_x <= previous_prefix).any() or (
        future_x <= float(np.max(history_x))
    ).any():
        raise FastChargeV5PairwiseError("Residual update crossed its landmark firewall")
    if rule.family == "no_update":
        return np.zeros(len(future_x), dtype=float)
    if rule.family == "robust_offset":
        if rule.statistic is None:
            raise FastChargeV5PairwiseError("Robust residual rule lacks a statistic")
        value = rule.shrinkage * _offset_statistic(residuals, rule.statistic)
        return np.full(len(future_x), value, dtype=float)
    if rule.family != "fixed_gaussian_process" or rule.length_scale is None:
        raise FastChargeV5PairwiseError(f"Unknown residual rule: {rule.candidate_id}")

    denominator = float(score_end_cycle - previous_prefix)
    if denominator <= 0.0:
        raise FastChargeV5PairwiseError("Residual update score horizon is invalid")
    train = ((history_x - previous_prefix) / denominator).reshape(-1, 1)
    future = ((future_x - previous_prefix) / denominator).reshape(-1, 1)
    gp_settings = config["candidate_rules"]["fixed_gaussian_process"]
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        rule.length_scale,
        length_scale_bounds="fixed",
        nu=1.5,
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(gp_settings["observation_noise_variance_pp2"]),
        optimizer=None,
        normalize_y=bool(gp_settings["normalize_target"]),
    )
    model.fit(train, residuals)
    cap = float(gp_settings["absolute_correction_cap_pp"])
    return np.clip(model.predict(future), -cap, cap)


def score_residual_candidates(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    *,
    rules: Sequence[ResidualRule] | None = None,
) -> pd.DataFrame:
    data = validate_prediction_frame(frame)
    selected_rules = tuple(rules) if rules is not None else candidate_rules(config)
    score_end = int(config["landmarks"]["score_end_cycle"])
    prediction_clip = tuple(
        float(v) for v in config["candidate_rules"]["future_prediction_clip_pct"]
    )
    rows: list[dict[str, object]] = []
    for cell_id, cell in data.groupby("cell_id", sort=True):
        by_prefix = {
            int(prefix): group.set_index("forecast_cycle").sort_index()
            for prefix, group in cell.groupby("prefix_cycle", sort=True)
        }
        for previous_prefix, current_prefix in config["landmarks"][
            "registered_transitions"
        ]:
            previous = by_prefix[int(previous_prefix)]
            current = by_prefix[int(current_prefix)]
            history = previous.loc[
                (previous.index > int(previous_prefix))
                & (previous.index <= int(current_prefix))
            ]
            future = current.loc[
                (current.index > int(current_prefix)) & (current.index <= score_end)
            ]
            residuals = history["observed_retention_pct"].to_numpy(
                dtype=float
            ) - history["candidate_prediction_pct"].to_numpy(dtype=float)
            truth = future["observed_retention_pct"].to_numpy(dtype=float)
            center = future["candidate_prediction_pct"].to_numpy(dtype=float)
            base_mae = float(np.mean(np.abs(truth - center)))
            for rule in selected_rules:
                correction = residual_correction(
                    history.index.to_numpy(dtype=float),
                    residuals,
                    future.index.to_numpy(dtype=float),
                    previous_prefix=int(previous_prefix),
                    score_end_cycle=score_end,
                    rule=rule,
                    config=config,
                )
                updated = np.clip(center + correction, *prediction_clip)
                updated_mae = float(np.mean(np.abs(truth - updated)))
                rows.append(
                    {
                        "cell_id": str(cell_id),
                        "previous_prefix_cycle": int(previous_prefix),
                        "current_prefix_cycle": int(current_prefix),
                        "candidate_id": rule.candidate_id,
                        "candidate_family": rule.family,
                        "history_row_count": len(history),
                        "future_row_count": len(future),
                        "base_trajectory_mae_pp": base_mae,
                        "updated_trajectory_mae_pp": updated_mae,
                        "delta_mae_pp": updated_mae - base_mae,
                        "mean_absolute_correction_pp": float(
                            np.mean(np.abs(correction))
                        ),
                        "maximum_absolute_correction_pp": float(
                            np.max(np.abs(correction))
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "candidate_id", "cell_id"],
        kind="stable",
        ignore_index=True,
    )


def summarize_candidate_scores(
    scores: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    eligibility = config["selection"]["eligibility"]
    rows: list[dict[str, object]] = []
    for (previous, current, candidate_id, family), group in scores.groupby(
        [
            "previous_prefix_cycle",
            "current_prefix_cycle",
            "candidate_id",
            "candidate_family",
        ],
        sort=True,
    ):
        deltas = group["delta_mae_pp"].to_numpy(dtype=float)
        improved = float(np.mean(deltas < 0.0))
        mean_delta = float(np.mean(deltas))
        p90_delta = float(np.quantile(deltas, 0.9))
        eligible = bool(
            candidate_id != "no_update"
            and mean_delta < float(eligibility["mean_delta_mae_must_be_below_pp"])
            and improved >= float(eligibility["minimum_fraction_cells_improved"])
            and p90_delta <= float(eligibility["maximum_p90_cell_delta_mae_pp"])
        )
        rows.append(
            {
                "previous_prefix_cycle": int(previous),
                "current_prefix_cycle": int(current),
                "candidate_id": str(candidate_id),
                "candidate_family": str(family),
                "physical_cell_count": int(group["cell_id"].nunique()),
                "mean_base_trajectory_mae_pp": float(
                    group["base_trajectory_mae_pp"].mean()
                ),
                "mean_updated_trajectory_mae_pp": float(
                    group["updated_trajectory_mae_pp"].mean()
                ),
                "mean_delta_mae_pp": mean_delta,
                "fraction_cells_improved": improved,
                "p90_cell_delta_mae_pp": p90_delta,
                "eligible": eligible,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "mean_updated_trajectory_mae_pp", "candidate_id"],
        kind="stable",
        ignore_index=True,
    )


def select_rules(summary: pd.DataFrame) -> dict[int, str]:
    result: dict[int, str] = {}
    for current, group in summary.groupby("current_prefix_cycle", sort=True):
        eligible = group.loc[group["eligible"]].sort_values(
            ["mean_updated_trajectory_mae_pp", "candidate_id"], kind="stable"
        )
        result[int(current)] = (
            str(eligible.iloc[0]["candidate_id"]) if len(eligible) else "no_update"
        )
    return result


def nested_selector_audit(
    scores: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cell_ids = sorted(str(value) for value in scores["cell_id"].unique())
    for held_out in cell_ids:
        fit = scores.loc[scores["cell_id"] != held_out]
        selected = select_rules(summarize_candidate_scores(fit, config))
        for current, candidate_id in selected.items():
            row = scores.loc[
                (scores["cell_id"] == held_out)
                & (scores["current_prefix_cycle"] == current)
                & (scores["candidate_id"] == candidate_id)
            ]
            if len(row) != 1:
                raise FastChargeV5PairwiseError(
                    "Nested selector could not locate a held-out candidate score"
                )
            record = row.iloc[0].to_dict()
            record["selected_candidate_id"] = candidate_id
            record["selection_fit_cell_count"] = len(cell_ids) - 1
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "cell_id"], kind="stable", ignore_index=True
    )


def score_base_reissues(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    data = validate_prediction_frame(frame)
    score_end = int(config["landmarks"]["score_end_cycle"])
    rows: list[dict[str, object]] = []
    for cell_id, cell in data.groupby("cell_id", sort=True):
        by_prefix = {
            int(prefix): group.set_index("forecast_cycle").sort_index()
            for prefix, group in cell.groupby("prefix_cycle", sort=True)
        }
        for previous_prefix, current_prefix in config["landmarks"][
            "registered_transitions"
        ]:
            previous = by_prefix[int(previous_prefix)]
            current = by_prefix[int(current_prefix)]
            future_cycles = current.index[
                (current.index > int(current_prefix)) & (current.index <= score_end)
            ]
            previous = previous.loc[future_cycles]
            current = current.loc[future_cycles]
            truth = current["observed_retention_pct"].to_numpy(dtype=float)
            previous_mae = float(
                np.mean(
                    np.abs(
                        truth
                        - previous["candidate_prediction_pct"].to_numpy(dtype=float)
                    )
                )
            )
            current_mae = float(
                np.mean(
                    np.abs(
                        truth
                        - current["candidate_prediction_pct"].to_numpy(dtype=float)
                    )
                )
            )
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "previous_prefix_cycle": int(previous_prefix),
                    "current_prefix_cycle": int(current_prefix),
                    "future_row_count": len(future_cycles),
                    "previous_trajectory_mae_pp": previous_mae,
                    "current_trajectory_mae_pp": current_mae,
                    "delta_mae_pp": current_mae - previous_mae,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "cell_id"], kind="stable", ignore_index=True
    )
