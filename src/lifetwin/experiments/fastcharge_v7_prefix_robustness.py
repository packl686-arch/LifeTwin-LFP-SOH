"""Frozen-rule prefix perturbation audit for the V7 P100 challenger.

The activation API is deliberately outcome-free. Future observations enter only
after a frozen decision has been produced, where they are used to quantify the
consequence of measurement noise or missing prefix observations.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.fastcharge_v5_landmark import (
    validate_prediction_frame,
)
from lifetwin.experiments.fastcharge_v5_pairwise import (
    FastChargeV5PairwiseError,
)
from lifetwin.experiments.fastcharge_v7_reissue_innovation import (
    reissue_innovation_correction,
)


def _correction_config(candidate: Mapping[str, object]) -> dict[str, object]:
    rule = candidate["frozen_update_rule"]
    return {
        "innovation_state": {
            "maximum_absolute_unassimilated_slope_pp_per_cycle": rule[
                "maximum_absolute_unassimilated_slope_pp_per_cycle"
            ],
            "absolute_correction_cap_pp": rule["absolute_correction_cap_pp"],
        }
    }


def frozen_gate_update(
    history_cycles: np.ndarray,
    history_residuals: np.ndarray,
    future_cycles: np.ndarray,
    previous_future_center: np.ndarray,
    current_future_center: np.ndarray,
    candidate: Mapping[str, object],
) -> tuple[np.ndarray, bool, dict[str, float | bool]]:
    """Return the frozen effective correction without reading future truth."""

    transition = candidate["eligible_transition"]
    rule = candidate["frozen_update_rule"]
    correction, diagnostics = reissue_innovation_correction(
        history_cycles,
        history_residuals,
        future_cycles,
        previous_future_center,
        current_future_center,
        previous_prefix=int(transition["previous_prefix_cycle"]),
        current_prefix=int(transition["current_prefix_cycle"]),
        projection_scale=float(rule["projection_scale"]),
        config=_correction_config(candidate),
    )
    active = bool(
        diagnostics["projected_unassimilated_change_pp"]
        >= float(rule["minimum_projected_unassimilated_change_pp"])
        and diagnostics["absolute_history_spearman"]
        >= float(rule["minimum_absolute_history_spearman"])
    )
    if bool(rule["require_history_slope_sign_agreement"]):
        active &= bool(diagnostics["history_slope_sign_agreement"])
    if bool(rule["require_unassimilated_history_slope_sign_agreement"]):
        active &= bool(diagnostics["unassimilated_history_slope_sign_agreement"])
    effective = correction if active else np.zeros_like(correction)
    return effective, active, diagnostics


def _stable_rng(
    base_seed: int,
    scenario_id: str,
    replicate_index: int,
    cell_id: str,
) -> np.random.Generator:
    material = (f"{base_seed}|{scenario_id}|{replicate_index}|{cell_id}").encode(
        "utf-8"
    )
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed)


def perturb_history(
    cycles: np.ndarray,
    residuals: np.ndarray,
    scenario: Mapping[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a registered perturbation to the observable residual history."""

    x = np.asarray(cycles, dtype=float).copy()
    y = np.asarray(residuals, dtype=float).copy()
    kind = str(scenario["kind"])
    if kind == "none":
        pass
    elif kind == "constant_offset":
        y += float(scenario["offset_pp"])
    elif kind == "iid_gaussian":
        y += rng.normal(0.0, float(scenario["sigma_pp"]), size=len(y))
    elif kind == "random_missing":
        x, y = _drop_random_history(x, y, scenario, rng)
    elif kind == "noise_plus_random_missing":
        y += rng.normal(0.0, float(scenario["sigma_pp"]), size=len(y))
        x, y = _drop_random_history(x, y, scenario, rng)
    elif kind == "single_spike":
        index = int(rng.integers(0, len(y)))
        direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        y[index] += direction * float(scenario["amplitude_pp"])
    elif kind == "linear_drift":
        direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        span = float(scenario["span_pp"])
        y += direction * np.linspace(-span / 2.0, span / 2.0, len(y))
    elif kind == "recent_block_missing":
        count = int(scenario["missing_count"])
        if count <= 0 or len(y) - count < 2:
            raise FastChargeV5PairwiseError(
                "Recent-block perturbation leaves insufficient history"
            )
        x = x[:-count]
        y = y[:-count]
    else:
        raise FastChargeV5PairwiseError(f"Unknown V7 robustness perturbation: {kind}")
    if len(x) < 2 or len(x) != len(y) or not np.isfinite(y).all():
        raise FastChargeV5PairwiseError(
            "V7 robustness perturbation produced invalid history"
        )
    return x, y


def _drop_random_history(
    cycles: np.ndarray,
    residuals: np.ndarray,
    scenario: Mapping[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    fraction = float(scenario["missing_fraction"])
    if not 0.0 < fraction < 1.0:
        raise FastChargeV5PairwiseError(
            "Random-missing fraction must be strictly between zero and one"
        )
    drop_count = max(1, int(round(len(cycles) * fraction)))
    if len(cycles) - drop_count < 2:
        raise FastChargeV5PairwiseError(
            "Random-missing perturbation leaves insufficient history"
        )
    keep = np.ones(len(cycles), dtype=bool)
    keep[rng.choice(len(cycles), size=drop_count, replace=False)] = False
    return cycles[keep], residuals[keep]


def _extract_cases(
    frame: pd.DataFrame,
    candidate: Mapping[str, object],
) -> dict[str, dict[str, np.ndarray]]:
    data = validate_prediction_frame(frame)
    transition = candidate["eligible_transition"]
    previous_prefix = int(transition["previous_prefix_cycle"])
    current_prefix = int(transition["current_prefix_cycle"])
    score_end = int(transition["score_end_cycle"])
    cases: dict[str, dict[str, np.ndarray]] = {}
    for cell_id, cell in data.groupby("cell_id", sort=True):
        previous = cell.loc[cell["prefix_cycle"] == previous_prefix].set_index(
            "forecast_cycle"
        )
        current = cell.loc[cell["prefix_cycle"] == current_prefix].set_index(
            "forecast_cycle"
        )
        history = previous.loc[
            (previous.index > previous_prefix) & (previous.index <= current_prefix)
        ].sort_index()
        future = current.loc[
            (current.index > current_prefix) & (current.index <= score_end)
        ].sort_index()
        if len(history) != current_prefix - previous_prefix:
            raise FastChargeV5PairwiseError(
                f"V7 robustness history is incomplete for {cell_id}"
            )
        if not future.index.isin(previous.index).all():
            raise FastChargeV5PairwiseError(
                f"V7 robustness common future support is incomplete for {cell_id}"
            )
        previous_future = previous.loc[future.index]
        cases[str(cell_id)] = {
            "history_cycles": history.index.to_numpy(dtype=float),
            "history_residuals": (
                history["observed_retention_pct"].to_numpy(dtype=float)
                - history["candidate_prediction_pct"].to_numpy(dtype=float)
            ),
            "future_cycles": future.index.to_numpy(dtype=float),
            "previous_future_center": previous_future[
                "candidate_prediction_pct"
            ].to_numpy(dtype=float),
            "current_future_center": future["candidate_prediction_pct"].to_numpy(
                dtype=float
            ),
            "future_truth": future["observed_retention_pct"].to_numpy(dtype=float),
        }
    return cases


def run_frozen_prefix_robustness(
    frame: pd.DataFrame,
    protocol: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run registered perturbations and return detailed and aggregate audits."""

    cases = _extract_cases(frame, candidate)
    expected_cells = int(protocol["input"]["physical_cell_count"])
    if len(cases) != expected_cells:
        raise FastChargeV5PairwiseError("V7 robustness physical-cell count changed")
    baseline = _baseline_rows(cases, candidate)
    baseline_by_cell = baseline.set_index("cell_id")
    baseline_corrections = {
        cell_id: frozen_gate_update(
            case["history_cycles"],
            case["history_residuals"],
            case["future_cycles"],
            case["previous_future_center"],
            case["current_future_center"],
            candidate,
        )[0]
        for cell_id, case in cases.items()
    }
    seed = int(protocol["randomization"]["base_seed"])
    decision_rows: list[dict[str, object]] = []
    for scenario in protocol["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        for replicate in range(int(scenario["replicates"])):
            for cell_id, case in sorted(cases.items()):
                rng = _stable_rng(seed, scenario_id, replicate, cell_id)
                history_x, residuals = perturb_history(
                    case["history_cycles"],
                    case["history_residuals"],
                    scenario,
                    rng,
                )
                correction, active, diagnostics = frozen_gate_update(
                    history_x,
                    residuals,
                    case["future_cycles"],
                    case["previous_future_center"],
                    case["current_future_center"],
                    candidate,
                )
                truth = case["future_truth"]
                center = case["current_future_center"]
                prediction_clip = tuple(
                    float(value)
                    for value in candidate["frozen_update_rule"][
                        "future_prediction_clip_pct"
                    ]
                )
                updated = np.clip(center + correction, *prediction_clip)
                base_mae = float(np.mean(np.abs(truth - center)))
                updated_mae = float(np.mean(np.abs(truth - updated)))
                original = baseline_by_cell.loc[cell_id]
                decision_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_role": str(scenario["role"]),
                        "replicate_index": replicate,
                        "cell_id": cell_id,
                        "history_row_count": len(history_x),
                        "baseline_activated": bool(original["activated"]),
                        "activated": active,
                        "decision_agreement": (active == bool(original["activated"])),
                        "false_activation": (
                            active and not bool(original["activated"])
                        ),
                        "missed_baseline_activation": (
                            not active and bool(original["activated"])
                        ),
                        "base_trajectory_mae_pp": base_mae,
                        "updated_trajectory_mae_pp": updated_mae,
                        "gated_delta_mae_pp": updated_mae - base_mae,
                        "improved": bool(active and updated_mae < base_mae),
                        "mean_absolute_effective_correction_pp": float(
                            np.mean(np.abs(correction))
                        ),
                        "effective_correction_deviation_pp": abs(
                            float(
                                np.mean(
                                    np.abs(correction - baseline_corrections[cell_id])
                                )
                            )
                        ),
                        **diagnostics,
                    }
                )
    decisions = pd.DataFrame(decision_rows).sort_values(
        ["scenario_id", "replicate_index", "cell_id"],
        kind="stable",
        ignore_index=True,
    )
    summaries = summarize_robustness(decisions, protocol)
    decision = robustness_decision(summaries, protocol)
    return baseline, decisions, summaries, decision


def _baseline_rows(
    cases: Mapping[str, Mapping[str, np.ndarray]],
    candidate: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_id, case in sorted(cases.items()):
        correction, active, diagnostics = frozen_gate_update(
            case["history_cycles"],
            case["history_residuals"],
            case["future_cycles"],
            case["previous_future_center"],
            case["current_future_center"],
            candidate,
        )
        truth = case["future_truth"]
        center = case["current_future_center"]
        prediction_clip = tuple(
            float(value)
            for value in candidate["frozen_update_rule"]["future_prediction_clip_pct"]
        )
        updated = np.clip(center + correction, *prediction_clip)
        base_mae = float(np.mean(np.abs(truth - center)))
        updated_mae = float(np.mean(np.abs(truth - updated)))
        rows.append(
            {
                "cell_id": cell_id,
                "activated": active,
                "base_trajectory_mae_pp": base_mae,
                "updated_trajectory_mae_pp": updated_mae,
                "gated_delta_mae_pp": updated_mae - base_mae,
                "mean_absolute_effective_correction_pp": float(
                    np.mean(np.abs(correction))
                ),
                **diagnostics,
            }
        )
    return pd.DataFrame(rows).sort_values("cell_id", kind="stable", ignore_index=True)


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def summarize_robustness(
    decisions: pd.DataFrame,
    protocol: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenarios = {str(item["scenario_id"]): item for item in protocol["scenarios"]}
    for scenario_id, group in decisions.groupby("scenario_id", sort=True):
        scenario = scenarios[str(scenario_id)]
        active = group.loc[group["activated"]]
        baseline_active = group["baseline_activated"].to_numpy(dtype=bool)
        activated = group["activated"].to_numpy(dtype=bool)
        replicate_maxima = np.asarray(
            [
                (
                    float(part.loc[part["activated"], "gated_delta_mae_pp"].max())
                    if bool(part["activated"].any())
                    else np.nan
                )
                for _, part in group.groupby("replicate_index", sort=True)
            ],
            dtype=float,
        )
        finite_maxima = replicate_maxima[np.isfinite(replicate_maxima)]
        summary: dict[str, object] = {
            "scenario_id": str(scenario_id),
            "scenario_role": str(scenario["role"]),
            "replicate_count": int(group["replicate_index"].nunique()),
            "physical_cell_count": int(group["cell_id"].nunique()),
            "decision_count": len(group),
            "activation_rate": float(np.mean(activated)),
            "decision_agreement": float(group["decision_agreement"].mean()),
            "baseline_active_retention": _optional_ratio(
                int(np.sum(activated & baseline_active)),
                int(np.sum(baseline_active)),
            ),
            "false_activation_rate": _optional_ratio(
                int(np.sum(activated & ~baseline_active)),
                int(np.sum(~baseline_active)),
            ),
            "activation_precision": (
                float(active["improved"].mean()) if len(active) else None
            ),
            "mean_all_cell_delta_mae_pp": float(group["gated_delta_mae_pp"].mean()),
            "active_p90_delta_mae_pp": (
                float(active["gated_delta_mae_pp"].quantile(0.9))
                if len(active)
                else None
            ),
            "active_global_max_delta_mae_pp": (
                float(active["gated_delta_mae_pp"].max()) if len(active) else None
            ),
            "p95_replicate_active_max_delta_mae_pp": (
                float(np.quantile(finite_maxima, 0.95)) if len(finite_maxima) else None
            ),
            "p95_effective_correction_deviation_pp": float(
                group["effective_correction_deviation_pp"].quantile(0.95)
            ),
        }
        passed, failures = _evaluate_thresholds(summary, scenario.get("thresholds"))
        summary["passed"] = passed
        summary["failed_thresholds"] = "|".join(failures)
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        "scenario_id", kind="stable", ignore_index=True
    )


def _evaluate_thresholds(
    summary: Mapping[str, object],
    thresholds: Mapping[str, object] | None,
) -> tuple[bool | None, list[str]]:
    if not thresholds:
        return None, []
    failures: list[str] = []
    for threshold, raw_limit in thresholds.items():
        if threshold.startswith("minimum_"):
            metric = threshold.removeprefix("minimum_")
            value = summary.get(metric)
            if value is None or float(value) < float(raw_limit):
                failures.append(threshold)
        elif threshold.startswith("maximum_"):
            metric = threshold.removeprefix("maximum_")
            value = summary.get(metric)
            if value is None or float(value) > float(raw_limit):
                failures.append(threshold)
        else:
            raise FastChargeV5PairwiseError(
                f"Unknown V7 robustness threshold: {threshold}"
            )
    return not failures, failures


def robustness_decision(
    summaries: pd.DataFrame,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    required_roles = set(protocol["decision_rule"]["required_roles"])
    required = summaries.loc[summaries["scenario_role"].isin(required_roles)]
    passed = bool(len(required) and required["passed"].eq(True).all())
    failed = required.loc[~required["passed"].eq(True), "scenario_id"].tolist()
    action_key = "pass_action" if passed else "failure_action"
    return {
        "all_required_scenarios_passed": passed,
        "required_scenario_count": len(required),
        "failed_required_scenarios": [str(value) for value in failed],
        "decision": str(protocol["decision_rule"][action_key]),
        "v5_champion_remains_active": True,
        "v7_candidate_activated": False,
        "same_41_training_cells_reused": True,
        "exposed_81_cell_evaluation_used": False,
    }


__all__ = [
    "frozen_gate_update",
    "perturb_history",
    "robustness_decision",
    "run_frozen_prefix_robustness",
    "summarize_robustness",
]
