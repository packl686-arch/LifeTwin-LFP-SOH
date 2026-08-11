"""Outcome-free end-to-end stability qualification for the frozen V7 update.

V8 perturbs the residual gate while holding the upstream V5 trajectories fixed.
V9 instead consumes a hash-bound ledger produced by repeated executions of the
complete V5 pipeline.  Each replicate may refit the historical reference model,
reselect neighbours, and rebuild the target P60/P100 trajectories.  The ledger
contains only completed reference histories, target-prefix observations, and
model predictions; target future outcomes are structurally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.fastcharge_v5_pairwise import (
    FastChargeV5PairwiseError,
)
from lifetwin.experiments.fastcharge_v7_prefix_robustness import (
    frozen_gate_update,
)


LEDGER_COLUMNS = (
    "schema_version",
    "issuance_id",
    "cell_id",
    "manufacturing_batch_id",
    "draw_index",
    "trajectory_role",
    "cycle_index",
    "retention_pct",
    "reference_cell_ids_json",
    "source_sha256",
)
LEDGER_SCHEMA_VERSION = "lifetwin.fastcharge_v9.end_to_end_replicate_ledger.v1"
TRAJECTORY_ROLES = (
    "p100_observed_prefix",
    "p60_v5_center",
    "p100_v5_center",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "future_truth",
    "future_capacity",
    "future_soh",
    "suffix_truth",
    "trajectory_mae",
    "absolute_error",
    "delta_mae",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DrawTrajectory:
    draw_index: int
    observed_history: np.ndarray
    previous_history_center: np.ndarray
    previous_future_center: np.ndarray
    current_future_center: np.ndarray
    p60_references: frozenset[str]
    p100_references: frozenset[str]


def validate_replicate_ledger(
    frame: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Validate one cell's outcome-free baseline and perturbation ledger."""

    if tuple(frame.columns) != LEDGER_COLUMNS:
        raise FastChargeV5PairwiseError(
            "V9 replicate ledger schema mismatch: "
            f"expected {list(LEDGER_COLUMNS)}, observed {list(frame.columns)}"
        )
    lowered = [str(column).lower() for column in frame.columns]
    if any(token in column for column in lowered for token in FORBIDDEN_OUTCOME_TOKENS):
        raise FastChargeV5PairwiseError("V9 replicate ledger exposes future outcomes")
    if frame.empty:
        raise FastChargeV5PairwiseError("V9 replicate ledger is empty")

    data = frame.copy()
    if set(data["schema_version"].astype(str)) != {LEDGER_SCHEMA_VERSION}:
        raise FastChargeV5PairwiseError("V9 replicate ledger schema version changed")
    for column in ("issuance_id", "cell_id", "manufacturing_batch_id"):
        values = data[column].astype(str)
        if values.str.strip().eq("").any() or values.nunique() != 1:
            raise FastChargeV5PairwiseError(
                f"V9 replicate ledger requires one nonempty {column}"
            )

    data["draw_index"] = pd.to_numeric(data["draw_index"], errors="coerce")
    data["cycle_index"] = pd.to_numeric(data["cycle_index"], errors="coerce")
    data["retention_pct"] = pd.to_numeric(data["retention_pct"], errors="coerce")
    numeric = data[["draw_index", "cycle_index", "retention_pct"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise FastChargeV5PairwiseError("V9 replicate ledger contains invalid numbers")
    if not np.equal(data["draw_index"], np.floor(data["draw_index"])).all():
        raise FastChargeV5PairwiseError("V9 draw indices must be integers")
    if not np.equal(data["cycle_index"], np.floor(data["cycle_index"])).all():
        raise FastChargeV5PairwiseError("V9 cycle indices must be integers")
    data["draw_index"] = data["draw_index"].astype(int)
    data["cycle_index"] = data["cycle_index"].astype(int)
    if ((data["retention_pct"] < 0.0) | (data["retention_pct"] > 110.0)).any():
        raise FastChargeV5PairwiseError("V9 retention values must lie in [0, 110]")

    roles = set(data["trajectory_role"].astype(str))
    if roles != set(TRAJECTORY_ROLES):
        raise FastChargeV5PairwiseError("V9 replicate ledger roles changed")
    if not data["source_sha256"].astype(str).map(_SHA256_RE.fullmatch).all():
        raise FastChargeV5PairwiseError("V9 source hashes must be canonical SHA-256")

    expected_draw_count = int(config["stability_gate"]["draw_count"])
    expected_draws = set(range(expected_draw_count + 1))
    if set(data["draw_index"]) != expected_draws:
        raise FastChargeV5PairwiseError(
            "V9 ledger must contain baseline draw 0 and every registered draw"
        )

    transition = config["eligible_transition"]
    previous_prefix = int(transition["previous_prefix_cycle"])
    current_prefix = int(transition["current_prefix_cycle"])
    score_end = int(transition["score_end_cycle"])
    expected_support = {
        "p100_observed_prefix": np.arange(
            previous_prefix + 1, current_prefix + 1, dtype=int
        ),
        "p60_v5_center": np.arange(previous_prefix + 1, score_end + 1, dtype=int),
        "p100_v5_center": np.arange(current_prefix + 1, score_end + 1, dtype=int),
    }
    expected_reference_count = int(config["model_contract"]["reference_count"])

    duplicate_key = ["draw_index", "trajectory_role", "cycle_index"]
    if data.duplicated(duplicate_key).any():
        raise FastChargeV5PairwiseError("V9 replicate ledger contains duplicate rows")
    for (draw_index, role), group in data.groupby(
        ["draw_index", "trajectory_role"], sort=True
    ):
        ordered = group.sort_values("cycle_index", kind="stable")
        if not np.array_equal(
            ordered["cycle_index"].to_numpy(dtype=int), expected_support[str(role)]
        ):
            raise FastChargeV5PairwiseError(
                f"V9 draw {draw_index} role {role} has incomplete support"
            )
        reference_values = set(ordered["reference_cell_ids_json"].astype(str))
        source_values = set(ordered["source_sha256"].astype(str))
        if len(reference_values) != 1 or len(source_values) != 1:
            raise FastChargeV5PairwiseError(
                "V9 per-draw role metadata must remain constant over its trajectory"
            )
        references = _parse_references(next(iter(reference_values)))
        if role == "p100_observed_prefix":
            if references:
                raise FastChargeV5PairwiseError(
                    "V9 observed-prefix rows cannot claim model references"
                )
        elif len(references) != expected_reference_count:
            raise FastChargeV5PairwiseError(
                "V9 V5-center rows have the wrong reference-cell count"
            )

    return data.sort_values(duplicate_key, kind="stable").reset_index(drop=True)


def evaluate_end_to_end_stability(
    frame: pd.DataFrame,
    candidate: Mapping[str, object],
    config: Mapping[str, object],
    *,
    protocol_sha256: str,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    """Qualify a V7 correction after full V5 perturb-and-refit replicates."""

    if _SHA256_RE.fullmatch(protocol_sha256) is None:
        raise FastChargeV5PairwiseError("V9 requires a canonical protocol SHA-256")
    _validate_candidate_transition(candidate, config)
    data = validate_replicate_ledger(frame, config)
    draws = {
        int(draw_index): _draw_trajectory(group, config)
        for draw_index, group in data.groupby("draw_index", sort=True)
    }
    baseline = draws[0]
    baseline_correction, baseline_active, baseline_diagnostics = _gate_draw(
        baseline, candidate, config
    )
    baseline_issued = baseline.current_future_center + baseline_correction
    baseline_sign = float(np.sign(baseline_correction[-1]))

    rows: list[dict[str, object]] = []
    corrections: list[np.ndarray] = []
    for draw_index in sorted(draw for draw in draws if draw != 0):
        draw = draws[draw_index]
        correction, active, diagnostics = _gate_draw(draw, candidate, config)
        issued = draw.current_future_center + correction
        corrections.append(correction)
        rows.append(
            {
                "draw_index": draw_index,
                "v7_activated": bool(active),
                "correction_sign_matches_baseline": bool(
                    active
                    and baseline_sign != 0.0
                    and np.sign(correction[-1]) == baseline_sign
                ),
                "p60_reference_jaccard": _jaccard(
                    baseline.p60_references, draw.p60_references
                ),
                "p100_reference_jaccard": _jaccard(
                    baseline.p100_references, draw.p100_references
                ),
                "p60_endpoint_abs_deviation_pp": float(
                    abs(
                        draw.previous_future_center[-1]
                        - baseline.previous_future_center[-1]
                    )
                ),
                "p100_endpoint_abs_deviation_pp": float(
                    abs(
                        draw.current_future_center[-1]
                        - baseline.current_future_center[-1]
                    )
                ),
                "p100_trajectory_mae_deviation_pp": float(
                    np.mean(
                        np.abs(
                            draw.current_future_center - baseline.current_future_center
                        )
                    )
                ),
                "issued_endpoint_abs_deviation_pp": float(
                    abs(issued[-1] - baseline_issued[-1])
                ),
                "issued_trajectory_mae_deviation_pp": float(
                    np.mean(np.abs(issued - baseline_issued))
                ),
                "history_slope_pp_per_cycle": float(
                    diagnostics["history_theil_slope_pp_per_cycle"]
                ),
                "unassimilated_slope_pp_per_cycle": float(
                    diagnostics["bounded_unassimilated_slope_pp_per_cycle"]
                ),
                "endpoint_correction_pp": float(correction[-1]),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("draw_index", kind="stable")
    gate = config["stability_gate"]
    activation_probability = float(metrics["v7_activated"].mean())
    sign_probability = float(metrics["correction_sign_matches_baseline"].mean())
    summaries = {
        "p05_p60_reference_jaccard": _quantile(metrics["p60_reference_jaccard"], 0.05),
        "p05_p100_reference_jaccard": _quantile(
            metrics["p100_reference_jaccard"], 0.05
        ),
        "p95_p60_endpoint_abs_deviation_pp": _quantile(
            metrics["p60_endpoint_abs_deviation_pp"], 0.95
        ),
        "p95_p100_endpoint_abs_deviation_pp": _quantile(
            metrics["p100_endpoint_abs_deviation_pp"], 0.95
        ),
        "p95_p100_trajectory_mae_deviation_pp": _quantile(
            metrics["p100_trajectory_mae_deviation_pp"], 0.95
        ),
        "p95_issued_endpoint_abs_deviation_pp": _quantile(
            metrics["issued_endpoint_abs_deviation_pp"], 0.95
        ),
        "p95_issued_trajectory_mae_deviation_pp": _quantile(
            metrics["issued_trajectory_mae_deviation_pp"], 0.95
        ),
    }

    reasons: list[str] = []
    if not baseline_active:
        reasons.append("baseline_v7_gate_inactive")
    _minimum_gate(
        reasons,
        activation_probability,
        float(gate["minimum_refit_activation_probability"]),
        "refit_activation_probability_below_threshold",
    )
    _minimum_gate(
        reasons,
        sign_probability,
        float(gate["minimum_refit_correction_sign_probability"]),
        "refit_correction_sign_probability_below_threshold",
    )
    for metric_name in (
        "p05_p60_reference_jaccard",
        "p05_p100_reference_jaccard",
    ):
        _minimum_gate(
            reasons,
            float(summaries[metric_name]),
            float(gate["minimum_p05_reference_jaccard"]),
            f"{metric_name}_below_threshold",
        )
    maximums = {
        "p95_p60_endpoint_abs_deviation_pp": ("maximum_p95_v5_endpoint_deviation_pp"),
        "p95_p100_endpoint_abs_deviation_pp": ("maximum_p95_v5_endpoint_deviation_pp"),
        "p95_p100_trajectory_mae_deviation_pp": (
            "maximum_p95_v5_trajectory_mae_deviation_pp"
        ),
        "p95_issued_endpoint_abs_deviation_pp": (
            "maximum_p95_issued_endpoint_deviation_pp"
        ),
        "p95_issued_trajectory_mae_deviation_pp": (
            "maximum_p95_issued_trajectory_mae_deviation_pp"
        ),
    }
    for metric_name, threshold_name in maximums.items():
        _maximum_gate(
            reasons,
            float(summaries[metric_name]),
            float(gate[threshold_name]),
            f"{metric_name}_above_threshold",
        )

    qualified = not reasons
    correction_matrix = np.vstack(corrections)
    stable_correction = (
        np.median(correction_matrix, axis=0)
        if qualified
        else np.zeros_like(baseline_correction)
    )
    status: dict[str, object] = {
        "schema_version": ("lifetwin.fastcharge_v9.end_to_end_stability_decision.v1"),
        "protocol_sha256": protocol_sha256,
        "draw_count": len(metrics),
        "baseline_v7_activated": bool(baseline_active),
        "baseline_endpoint_correction_pp": float(baseline_correction[-1]),
        "refit_activation_probability": activation_probability,
        "refit_correction_sign_probability": sign_probability,
        **summaries,
        "quality_activated": qualified,
        "reasons": reasons,
        "failed_action": (
            "none" if qualified else "exact_zero_update_to_unperturbed_p100_v5_center"
        ),
        "future_outcomes_read": False,
        "model_accuracy_evidence_created": False,
        "v5_champion_changed": False,
        "baseline_gate_diagnostics": {
            key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
            for key, value in baseline_diagnostics.items()
        },
    }
    return stable_correction, status, metrics


def _validate_candidate_transition(
    candidate: Mapping[str, object], config: Mapping[str, object]
) -> None:
    candidate_transition = candidate["eligible_transition"]
    expected = config["eligible_transition"]
    for field in (
        "previous_prefix_cycle",
        "current_prefix_cycle",
        "score_end_cycle",
    ):
        if int(candidate_transition[field]) != int(expected[field]):
            raise FastChargeV5PairwiseError(
                f"V9 candidate transition field changed: {field}"
            )


def _draw_trajectory(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> DrawTrajectory:
    draw_index = int(frame["draw_index"].iloc[0])

    def role_values(role: str) -> np.ndarray:
        return (
            frame.loc[frame["trajectory_role"] == role]
            .sort_values("cycle_index", kind="stable")["retention_pct"]
            .to_numpy(dtype=float)
        )

    def references(role: str) -> frozenset[str]:
        value = str(
            frame.loc[frame["trajectory_role"] == role, "reference_cell_ids_json"].iloc[
                0
            ]
        )
        return frozenset(_parse_references(value))

    transition = config["eligible_transition"]
    history_count = int(transition["current_prefix_cycle"]) - int(
        transition["previous_prefix_cycle"]
    )
    p60 = role_values("p60_v5_center")
    return DrawTrajectory(
        draw_index=draw_index,
        observed_history=role_values("p100_observed_prefix"),
        previous_history_center=p60[:history_count],
        previous_future_center=p60[history_count:],
        current_future_center=role_values("p100_v5_center"),
        p60_references=references("p60_v5_center"),
        p100_references=references("p100_v5_center"),
    )


def _gate_draw(
    draw: DrawTrajectory,
    candidate: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[np.ndarray, bool, dict[str, float | bool]]:
    transition = config["eligible_transition"]
    previous_prefix = int(transition["previous_prefix_cycle"])
    current_prefix = int(transition["current_prefix_cycle"])
    score_end = int(transition["score_end_cycle"])
    history_cycles = np.arange(previous_prefix + 1, current_prefix + 1, dtype=float)
    future_cycles = np.arange(current_prefix + 1, score_end + 1, dtype=float)
    residuals = draw.observed_history - draw.previous_history_center
    return frozen_gate_update(
        history_cycles,
        residuals,
        future_cycles,
        draw.previous_future_center,
        draw.current_future_center,
        candidate,
    )


def _parse_references(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise FastChargeV5PairwiseError(
            "V9 reference_cell_ids_json is invalid"
        ) from error
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) or not item.strip() for item in parsed
    ):
        raise FastChargeV5PairwiseError(
            "V9 reference_cell_ids_json must contain nonempty strings"
        )
    if len(parsed) != len(set(parsed)):
        raise FastChargeV5PairwiseError("V9 reference cell identifiers repeat")
    return tuple(parsed)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _quantile(values: pd.Series, probability: float) -> float:
    return float(
        np.quantile(values.to_numpy(dtype=float), probability, method="higher")
    )


def _minimum_gate(
    reasons: list[str], value: float, threshold: float, reason: str
) -> None:
    if not math.isfinite(value) or value < threshold:
        reasons.append(reason)


def _maximum_gate(
    reasons: list[str], value: float, threshold: float, reason: str
) -> None:
    if not math.isfinite(value) or value > threshold:
        reasons.append(reason)


__all__ = [
    "LEDGER_COLUMNS",
    "LEDGER_SCHEMA_VERSION",
    "TRAJECTORY_ROLES",
    "evaluate_end_to_end_stability",
    "validate_replicate_ledger",
]
