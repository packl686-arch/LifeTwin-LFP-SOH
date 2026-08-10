"""Training-only bounded residual-state challenger around the frozen V5 center."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes

from lifetwin.experiments.fastcharge_v5_landmark import (
    nested_selector_audit,
    select_rules,
    summarize_candidate_scores,
    validate_prediction_frame,
)
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


@dataclass(frozen=True)
class BoundedStateRule:
    candidate_id: str
    family: str
    history_window: str | None = None
    projection_scale: float = 0.0
    level_gain: float | None = None
    slope_gain: float | None = None
    innovation_clip_pp: float | None = None


@dataclass(frozen=True)
class ActivationGate:
    gate_id: str
    correction_candidate_id: str | None = None
    projection_scale: float | None = None
    minimum_projected_history_change_pp: float | None = None
    minimum_absolute_spearman: float | None = None
    require_slope_sign_agreement: bool = False


def candidate_rules(config: Mapping[str, object]) -> tuple[BoundedStateRule, ...]:
    settings = config["candidate_rules"]
    robust = settings["robust_local_trend"]
    alpha_beta = settings["bounded_alpha_beta"]
    rules = [BoundedStateRule("no_update", "no_update")]
    for window in robust["history_windows"]:
        for scale in robust["projection_scales"]:
            token = str(scale).replace(".", "p")
            rules.append(
                BoundedStateRule(
                    f"theil_{window}_a{token}",
                    "robust_local_trend",
                    history_window=str(window),
                    projection_scale=float(scale),
                )
            )
    for profile in alpha_beta["profiles"]:
        for scale in alpha_beta["projection_scales"]:
            token = str(scale).replace(".", "p")
            rules.append(
                BoundedStateRule(
                    f"alpha_beta_{profile['profile_id']}_a{token}",
                    "bounded_alpha_beta",
                    projection_scale=float(scale),
                    level_gain=float(profile["level_gain"]),
                    slope_gain=float(profile["slope_gain"]),
                    innovation_clip_pp=float(profile["innovation_clip_pp"]),
                )
            )
    return tuple(rules)


def _history_slice(
    cycles: np.ndarray, residuals: np.ndarray, window: str
) -> tuple[np.ndarray, np.ndarray]:
    if window == "all":
        return cycles, residuals
    if window == "last_10":
        return cycles[-10:], residuals[-10:]
    raise FastChargeV5PairwiseError(f"Unknown robust trend window: {window}")


def _robust_local_slope(
    cycles: np.ndarray, residuals: np.ndarray, window: str
) -> float:
    selected_cycles, selected_residuals = _history_slice(
        cycles, residuals, window
    )
    if len(selected_cycles) < 2:
        return 0.0
    return float(theilslopes(selected_residuals, selected_cycles)[0])


def _alpha_beta_slope(
    cycles: np.ndarray,
    residuals: np.ndarray,
    *,
    level_gain: float,
    slope_gain: float,
    innovation_clip_pp: float,
    maximum_absolute_slope_pp_per_cycle: float,
) -> float:
    level = float(residuals[0])
    slope = 0.0
    previous_cycle = float(cycles[0])
    for cycle, observation in zip(cycles[1:], residuals[1:], strict=True):
        delta = float(cycle - previous_cycle)
        if delta <= 0.0:
            raise FastChargeV5PairwiseError(
                "Residual-state history cycles must be strictly increasing"
            )
        predicted_level = level + slope * delta
        innovation = float(
            np.clip(
                observation - predicted_level,
                -innovation_clip_pp,
                innovation_clip_pp,
            )
        )
        level = predicted_level + level_gain * innovation
        slope = float(
            np.clip(
                slope + slope_gain * innovation / delta,
                -maximum_absolute_slope_pp_per_cycle,
                maximum_absolute_slope_pp_per_cycle,
            )
        )
        previous_cycle = float(cycle)
    return slope


def bounded_state_correction(
    history_cycles: np.ndarray,
    history_residuals: np.ndarray,
    future_cycles: np.ndarray,
    *,
    previous_prefix: int,
    current_prefix: int,
    rule: BoundedStateRule,
    config: Mapping[str, object],
) -> np.ndarray:
    history_x = np.asarray(history_cycles, dtype=float).reshape(-1)
    residuals = np.asarray(history_residuals, dtype=float).reshape(-1)
    future_x = np.asarray(future_cycles, dtype=float).reshape(-1)
    values = np.concatenate([history_x, residuals, future_x])
    if (
        len(history_x) < 2
        or history_x.shape != residuals.shape
        or len(future_x) == 0
        or not np.isfinite(values).all()
    ):
        raise FastChargeV5PairwiseError("Residual-state coordinates are invalid")
    if (
        (np.diff(history_x) <= 0.0).any()
        or (history_x <= previous_prefix).any()
        or (history_x > current_prefix).any()
        or (future_x <= current_prefix).any()
    ):
        raise FastChargeV5PairwiseError(
            "Residual-state update crossed its landmark firewall"
        )
    if rule.family == "no_update":
        return np.zeros(len(future_x), dtype=float)

    settings = config["candidate_rules"]
    if rule.family == "robust_local_trend":
        if rule.history_window is None:
            raise FastChargeV5PairwiseError(
                "Robust local-trend rule lacks a history window"
            )
        slope = _robust_local_slope(
            history_x, residuals, rule.history_window
        )
        maximum_slope = float(
            settings["robust_local_trend"][
                "maximum_absolute_slope_pp_per_cycle"
            ]
        )
    elif rule.family == "bounded_alpha_beta":
        if (
            rule.level_gain is None
            or rule.slope_gain is None
            or rule.innovation_clip_pp is None
        ):
            raise FastChargeV5PairwiseError(
                "Bounded alpha-beta rule is incomplete"
            )
        maximum_slope = float(
            settings["bounded_alpha_beta"][
                "maximum_absolute_slope_pp_per_cycle"
            ]
        )
        slope = _alpha_beta_slope(
            history_x,
            residuals,
            level_gain=rule.level_gain,
            slope_gain=rule.slope_gain,
            innovation_clip_pp=rule.innovation_clip_pp,
            maximum_absolute_slope_pp_per_cycle=maximum_slope,
        )
    else:
        raise FastChargeV5PairwiseError(
            f"Unknown residual-state rule: {rule.candidate_id}"
        )

    bounded_slope = float(np.clip(slope, -maximum_slope, maximum_slope))
    horizon = future_x - float(current_prefix)
    raw = rule.projection_scale * bounded_slope * horizon
    cap = float(settings["absolute_correction_cap_pp"])
    return np.clip(raw, -cap, cap)


def score_bounded_state_candidates(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    *,
    rules: Sequence[BoundedStateRule] | None = None,
) -> pd.DataFrame:
    data = validate_prediction_frame(frame)
    selected_rules = tuple(rules) if rules is not None else candidate_rules(config)
    score_end = int(config["landmarks"]["score_end_cycle"])
    prediction_clip = tuple(
        float(value)
        for value in config["candidate_rules"]["future_prediction_clip_pct"]
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
                (current.index > int(current_prefix))
                & (current.index <= score_end)
            ]
            residuals = history["observed_retention_pct"].to_numpy(
                dtype=float
            ) - history["candidate_prediction_pct"].to_numpy(dtype=float)
            truth = future["observed_retention_pct"].to_numpy(dtype=float)
            center = future["candidate_prediction_pct"].to_numpy(dtype=float)
            base_mae = float(np.mean(np.abs(truth - center)))
            for rule in selected_rules:
                correction = bounded_state_correction(
                    history.index.to_numpy(dtype=float),
                    residuals,
                    future.index.to_numpy(dtype=float),
                    previous_prefix=int(previous_prefix),
                    current_prefix=int(current_prefix),
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


def promotion_summary(
    nested_scores: pd.DataFrame, config: Mapping[str, object]
) -> dict[str, object]:
    gate = config["promotion_gate"]
    transition_rows: list[dict[str, object]] = []
    for (previous, current), group in nested_scores.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle"], sort=True
    ):
        deltas = group["delta_mae_pp"].to_numpy(dtype=float)
        transition_rows.append(
            {
                "previous_prefix_cycle": int(previous),
                "current_prefix_cycle": int(current),
                "physical_cell_count": int(group["cell_id"].nunique()),
                "mean_delta_mae_pp": float(np.mean(deltas)),
                "fraction_cells_improved": float(np.mean(deltas < 0.0)),
                "p90_cell_delta_mae_pp": float(np.quantile(deltas, 0.9)),
            }
        )
    all_deltas = nested_scores["delta_mae_pp"].to_numpy(dtype=float)
    overall = {
        "physical_cell_transition_count": int(len(nested_scores)),
        "mean_delta_mae_pp": float(np.mean(all_deltas)),
        "fraction_cell_transitions_improved": float(
            np.mean(all_deltas < 0.0)
        ),
        "p90_cell_transition_delta_mae_pp": float(
            np.quantile(all_deltas, 0.9)
        ),
    }
    passed = bool(
        overall["mean_delta_mae_pp"]
        < float(gate["mean_delta_mae_must_be_below_pp"])
        and overall["fraction_cell_transitions_improved"]
        >= float(gate["minimum_fraction_cell_transitions_improved"])
        and overall["p90_cell_transition_delta_mae_pp"]
        <= float(gate["maximum_p90_cell_transition_delta_mae_pp"])
        and all(
            row["mean_delta_mae_pp"]
            <= float(gate["maximum_transition_mean_delta_mae_pp"])
            for row in transition_rows
        )
    )
    return {
        "passed": passed,
        "overall": overall,
        "transitions": transition_rows,
    }


def activation_gates(config: Mapping[str, object]) -> tuple[ActivationGate, ...]:
    settings = config["activation_gate_candidates"]
    gates = [ActivationGate("no_update")]
    for scale in settings["projection_scales"]:
        correction_id = settings["correction_candidate_id_template"].format(
            scale_token=str(scale).replace(".", "p")
        )
        for gate in settings["gate_profiles"]:
            gates.append(
                ActivationGate(
                    gate_id=f"a{str(scale).replace('.', 'p')}_{gate['profile_id']}",
                    correction_candidate_id=correction_id,
                    projection_scale=float(scale),
                    minimum_projected_history_change_pp=float(
                        gate["minimum_projected_history_change_pp"]
                    ),
                    minimum_absolute_spearman=float(
                        gate["minimum_absolute_spearman"]
                    ),
                    require_slope_sign_agreement=bool(
                        gate["require_slope_sign_agreement"]
                    ),
                )
            )
    return tuple(gates)


def _history_diagnostics(
    cycles: np.ndarray,
    residuals: np.ndarray,
    *,
    previous_prefix: int,
    current_prefix: int,
) -> dict[str, object]:
    all_slope = _robust_local_slope(cycles, residuals, "all")
    recent_slope = _robust_local_slope(cycles, residuals, "last_10")
    correlation = float(spearmanr(cycles, residuals).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "history_theil_slope_pp_per_cycle": all_slope,
        "history_recent_theil_slope_pp_per_cycle": recent_slope,
        "slope_sign_agreement": bool(
            np.sign(all_slope) == np.sign(recent_slope)
        ),
        "absolute_spearman_monotonicity": abs(correlation),
        "projected_history_change_pp": float(
            abs(all_slope) * (current_prefix - previous_prefix)
        ),
    }


def build_activation_table(
    frame: pd.DataFrame,
    correction_scores: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    data = validate_prediction_frame(frame)
    gates = activation_gates(config)
    correction_ids = sorted(
        {
            gate.correction_candidate_id
            for gate in gates
            if gate.correction_candidate_id is not None
        }
    )
    score_subset = correction_scores.loc[
        correction_scores["candidate_id"].isin(correction_ids)
    ].copy()
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
            history = previous.loc[
                (previous.index > int(previous_prefix))
                & (previous.index <= int(current_prefix))
            ]
            residuals = history["observed_retention_pct"].to_numpy(
                dtype=float
            ) - history["candidate_prediction_pct"].to_numpy(dtype=float)
            diagnostics = _history_diagnostics(
                history.index.to_numpy(dtype=float),
                residuals,
                previous_prefix=int(previous_prefix),
                current_prefix=int(current_prefix),
            )
            for correction_id in correction_ids:
                score = score_subset.loc[
                    (score_subset["cell_id"] == str(cell_id))
                    & (
                        score_subset["current_prefix_cycle"]
                        == int(current_prefix)
                    )
                    & (score_subset["candidate_id"] == correction_id)
                ]
                if len(score) != 1:
                    raise FastChargeV5PairwiseError(
                        "Activation audit could not locate a unique correction score"
                    )
                record = score.iloc[0]
                rows.append(
                    {
                        "cell_id": str(cell_id),
                        "previous_prefix_cycle": int(previous_prefix),
                        "current_prefix_cycle": int(current_prefix),
                        "correction_candidate_id": str(correction_id),
                        "base_trajectory_mae_pp": float(
                            record["base_trajectory_mae_pp"]
                        ),
                        "raw_updated_trajectory_mae_pp": float(
                            record["updated_trajectory_mae_pp"]
                        ),
                        "raw_delta_mae_pp": float(record["delta_mae_pp"]),
                        **diagnostics,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "correction_candidate_id", "cell_id"],
        kind="stable",
        ignore_index=True,
    )


def _gate_rows(
    table: pd.DataFrame, gate: ActivationGate
) -> tuple[pd.DataFrame, np.ndarray]:
    if gate.gate_id == "no_update":
        rows = table.drop_duplicates(
            ["cell_id", "previous_prefix_cycle", "current_prefix_cycle"]
        ).copy()
        return rows, np.zeros(len(rows), dtype=bool)
    rows = table.loc[
        table["correction_candidate_id"] == gate.correction_candidate_id
    ].copy()
    if (
        gate.minimum_projected_history_change_pp is None
        or gate.minimum_absolute_spearman is None
    ):
        raise FastChargeV5PairwiseError(
            f"Activation gate is incomplete: {gate.gate_id}"
        )
    active = (
        rows["projected_history_change_pp"].to_numpy(dtype=float)
        >= gate.minimum_projected_history_change_pp
    ) & (
        rows["absolute_spearman_monotonicity"].to_numpy(dtype=float)
        >= gate.minimum_absolute_spearman
    )
    if gate.require_slope_sign_agreement:
        active &= rows["slope_sign_agreement"].to_numpy(dtype=bool)
    return rows, active


def summarize_activation_gates(
    table: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    eligibility = config["activation_selection"]["eligibility"]
    rows: list[dict[str, object]] = []
    for (previous, current), transition in table.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle"], sort=True
    ):
        for gate in activation_gates(config):
            candidates, active = _gate_rows(transition, gate)
            raw_deltas = candidates["raw_delta_mae_pp"].to_numpy(dtype=float)
            gated_deltas = np.where(active, raw_deltas, 0.0)
            active_deltas = raw_deltas[active]
            coverage = float(np.mean(active))
            precision = (
                float(np.mean(active_deltas < 0.0))
                if len(active_deltas)
                else float("nan")
            )
            active_p90 = (
                float(np.quantile(active_deltas, 0.9))
                if len(active_deltas)
                else float("nan")
            )
            active_maximum = (
                float(np.max(active_deltas))
                if len(active_deltas)
                else float("nan")
            )
            eligible = bool(
                gate.gate_id != "no_update"
                and coverage
                >= float(eligibility["minimum_activation_coverage"])
                and precision
                >= float(eligibility["minimum_activation_precision"])
                and float(np.mean(gated_deltas))
                < float(eligibility["mean_gated_delta_mae_must_be_below_pp"])
                and active_p90
                <= float(eligibility["maximum_active_p90_delta_mae_pp"])
                and active_maximum
                <= float(eligibility["maximum_active_delta_mae_pp"])
            )
            rows.append(
                {
                    "previous_prefix_cycle": int(previous),
                    "current_prefix_cycle": int(current),
                    "gate_id": gate.gate_id,
                    "correction_candidate_id": gate.correction_candidate_id,
                    "physical_cell_count": int(candidates["cell_id"].nunique()),
                    "activation_count": int(np.sum(active)),
                    "activation_coverage": coverage,
                    "activation_precision": precision,
                    "mean_gated_delta_mae_pp": float(np.mean(gated_deltas)),
                    "active_p90_delta_mae_pp": active_p90,
                    "active_maximum_delta_mae_pp": active_maximum,
                    "eligible": eligible,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "mean_gated_delta_mae_pp", "gate_id"],
        kind="stable",
        ignore_index=True,
    )


def select_activation_gates(summary: pd.DataFrame) -> dict[int, str]:
    selected: dict[int, str] = {}
    for current, group in summary.groupby("current_prefix_cycle", sort=True):
        eligible = group.loc[group["eligible"]].sort_values(
            ["mean_gated_delta_mae_pp", "activation_coverage", "gate_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        selected[int(current)] = (
            str(eligible.iloc[0]["gate_id"]) if len(eligible) else "no_update"
        )
    return selected


def nested_activation_gate_audit(
    table: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    gate_lookup = {gate.gate_id: gate for gate in activation_gates(config)}
    cell_ids = sorted(str(value) for value in table["cell_id"].unique())
    rows: list[dict[str, object]] = []
    for held_out in cell_ids:
        fit = table.loc[table["cell_id"] != held_out]
        selected = select_activation_gates(
            summarize_activation_gates(fit, config)
        )
        for current, gate_id in selected.items():
            transition = table.loc[
                (table["cell_id"] == held_out)
                & (table["current_prefix_cycle"] == current)
            ]
            candidates, active = _gate_rows(transition, gate_lookup[gate_id])
            if len(candidates) != 1 or len(active) != 1:
                raise FastChargeV5PairwiseError(
                    "Nested activation audit could not locate a held-out gate row"
                )
            record = candidates.iloc[0]
            is_active = bool(active[0])
            raw_delta = float(record["raw_delta_mae_pp"])
            base_mae = float(record["base_trajectory_mae_pp"])
            rows.append(
                {
                    "cell_id": held_out,
                    "previous_prefix_cycle": int(
                        record["previous_prefix_cycle"]
                    ),
                    "current_prefix_cycle": int(current),
                    "selected_gate_id": gate_id,
                    "selected_correction_candidate_id": (
                        gate_lookup[gate_id].correction_candidate_id
                    ),
                    "selection_fit_cell_count": len(cell_ids) - 1,
                    "activated": is_active,
                    "base_trajectory_mae_pp": base_mae,
                    "updated_trajectory_mae_pp": (
                        base_mae + raw_delta if is_active else base_mae
                    ),
                    "raw_delta_mae_pp": raw_delta,
                    "gated_delta_mae_pp": raw_delta if is_active else 0.0,
                    "projected_history_change_pp": float(
                        record["projected_history_change_pp"]
                    ),
                    "absolute_spearman_monotonicity": float(
                        record["absolute_spearman_monotonicity"]
                    ),
                    "slope_sign_agreement": bool(
                        record["slope_sign_agreement"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "cell_id"], kind="stable", ignore_index=True
    )


def activation_nomination_summary(
    nested: pd.DataFrame, config: Mapping[str, object]
) -> dict[str, object]:
    gate = config["future_blind_nomination_gate"]
    transitions: list[dict[str, object]] = []
    nominated: list[int] = []
    for (previous, current), group in nested.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle"], sort=True
    ):
        active = group.loc[group["activated"]]
        active_deltas = active["gated_delta_mae_pp"].to_numpy(dtype=float)
        all_deltas = group["gated_delta_mae_pp"].to_numpy(dtype=float)
        has_active = len(active_deltas) > 0
        coverage = float(np.mean(group["activated"].to_numpy(dtype=bool)))
        precision = (
            float(np.mean(active_deltas < 0.0))
            if has_active
            else None
        )
        active_p90 = (
            float(np.quantile(active_deltas, 0.9))
            if has_active
            else None
        )
        active_maximum = (
            float(np.max(active_deltas))
            if has_active
            else None
        )
        modal_fraction = float(
            group["selected_gate_id"].value_counts(normalize=True).iloc[0]
        )
        passed = bool(
            has_active
            and len(active_deltas) >= int(gate["minimum_activation_count"])
            and coverage >= float(gate["minimum_activation_coverage"])
            and precision is not None
            and precision >= float(gate["minimum_activation_precision"])
            and float(np.mean(all_deltas))
            < float(gate["mean_gated_delta_mae_must_be_below_pp"])
            and active_p90 is not None
            and active_p90
            <= float(gate["maximum_active_p90_delta_mae_pp"])
            and active_maximum is not None
            and active_maximum
            <= float(gate["maximum_active_delta_mae_pp"])
            and modal_fraction
            >= float(gate["minimum_modal_gate_selection_fraction"])
        )
        row = {
            "previous_prefix_cycle": int(previous),
            "current_prefix_cycle": int(current),
            "physical_cell_count": int(group["cell_id"].nunique()),
            "activation_count": int(len(active)),
            "activation_coverage": coverage,
            "activation_precision": precision,
            "mean_gated_delta_mae_pp": float(np.mean(all_deltas)),
            "active_p90_delta_mae_pp": active_p90,
            "active_maximum_delta_mae_pp": active_maximum,
            "modal_gate_selection_fraction": modal_fraction,
            "selected_gate_counts": {
                str(key): int(value)
                for key, value in sorted(
                    group["selected_gate_id"].value_counts().items()
                )
            },
            "nominated_for_future_blind_test": passed,
        }
        transitions.append(row)
        if passed:
            nominated.append(int(current))
    return {
        "any_transition_nominated": bool(nominated),
        "nominated_current_prefix_cycles": nominated,
        "transitions": transitions,
    }


__all__ = [
    "ActivationGate",
    "BoundedStateRule",
    "activation_gates",
    "activation_nomination_summary",
    "bounded_state_correction",
    "build_activation_table",
    "candidate_rules",
    "nested_activation_gate_audit",
    "nested_selector_audit",
    "promotion_summary",
    "score_bounded_state_candidates",
    "select_activation_gates",
    "select_rules",
    "summarize_activation_gates",
    "summarize_candidate_scores",
]
