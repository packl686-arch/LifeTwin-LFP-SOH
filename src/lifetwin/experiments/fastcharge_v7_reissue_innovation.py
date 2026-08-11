"""Reissue-aware residual innovation challenger around the frozen V5 center.

The previous residual slope can contain a trajectory change that the current
V5 issuance has already absorbed.  V7 subtracts that model-only reissue shift
before applying a bounded correction, and abstains unless the remaining state
is directionally stable.  Future outcomes are used only by the scoring layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes

from lifetwin.experiments.fastcharge_v5_landmark import (
    validate_prediction_frame,
)
from lifetwin.experiments.fastcharge_v5_pairwise import (
    FastChargeV5PairwiseError,
)


@dataclass(frozen=True)
class InnovationGate:
    gate_id: str
    projection_scale: float | None = None
    minimum_projected_innovation_change_pp: float | None = None
    minimum_absolute_history_spearman: float | None = None
    require_history_slope_sign_agreement: bool = False
    require_innovation_history_sign_agreement: bool = False


def _token(value: float) -> str:
    return str(value).replace(".", "p")


def _theil_slope(cycles: np.ndarray, values: np.ndarray) -> float:
    if len(cycles) < 2:
        return 0.0
    return float(theilslopes(values, cycles)[0])


def reissue_innovation_correction(
    history_cycles: np.ndarray,
    history_residuals: np.ndarray,
    future_cycles: np.ndarray,
    previous_future_center: np.ndarray,
    current_future_center: np.ndarray,
    *,
    previous_prefix: int,
    current_prefix: int,
    projection_scale: float,
    config: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Return a bounded correction using only information available at issuance."""

    history_x = np.asarray(history_cycles, dtype=float).reshape(-1)
    residuals = np.asarray(history_residuals, dtype=float).reshape(-1)
    future_x = np.asarray(future_cycles, dtype=float).reshape(-1)
    previous_center = np.asarray(previous_future_center, dtype=float).reshape(-1)
    current_center = np.asarray(current_future_center, dtype=float).reshape(-1)
    arrays = (history_x, residuals, future_x, previous_center, current_center)
    if (
        len(history_x) < 2
        or history_x.shape != residuals.shape
        or len(future_x) < 2
        or future_x.shape != previous_center.shape
        or future_x.shape != current_center.shape
        or not all(np.isfinite(value).all() for value in arrays)
    ):
        raise FastChargeV5PairwiseError(
            "Reissue-innovation coordinates are invalid"
        )
    if (
        (np.diff(history_x) <= 0.0).any()
        or (np.diff(future_x) <= 0.0).any()
        or (history_x <= previous_prefix).any()
        or (history_x > current_prefix).any()
        or (future_x <= current_prefix).any()
    ):
        raise FastChargeV5PairwiseError(
            "Reissue-innovation update crossed its landmark firewall"
        )
    if projection_scale <= 0.0:
        raise FastChargeV5PairwiseError(
            "Reissue-innovation projection scale must be positive"
        )

    settings = config["innovation_state"]
    history_slope = _theil_slope(history_x, residuals)
    recent_count = min(10, len(history_x))
    recent_slope = _theil_slope(
        history_x[-recent_count:], residuals[-recent_count:]
    )
    reissue_shift = current_center - previous_center
    reissue_shift_slope = _theil_slope(future_x, reissue_shift)
    raw_innovation_slope = history_slope - reissue_shift_slope
    maximum_slope = float(
        settings["maximum_absolute_unassimilated_slope_pp_per_cycle"]
    )
    innovation_slope = float(
        np.clip(raw_innovation_slope, -maximum_slope, maximum_slope)
    )
    correlation = float(spearmanr(history_x, residuals).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0

    horizon = future_x - float(current_prefix)
    cap = float(settings["absolute_correction_cap_pp"])
    correction = np.clip(
        projection_scale * innovation_slope * horizon,
        -cap,
        cap,
    )
    diagnostics: dict[str, float | bool] = {
        "history_theil_slope_pp_per_cycle": history_slope,
        "history_recent_theil_slope_pp_per_cycle": recent_slope,
        "reissue_shift_theil_slope_pp_per_cycle": reissue_shift_slope,
        "raw_unassimilated_slope_pp_per_cycle": raw_innovation_slope,
        "bounded_unassimilated_slope_pp_per_cycle": innovation_slope,
        "history_slope_sign_agreement": bool(
            np.sign(history_slope) == np.sign(recent_slope)
        ),
        "unassimilated_history_slope_sign_agreement": bool(
            np.sign(innovation_slope) == np.sign(history_slope)
        ),
        "absolute_history_spearman": abs(correlation),
        "projected_unassimilated_change_pp": float(
            abs(innovation_slope) * (current_prefix - previous_prefix)
        ),
    }
    return correction, diagnostics


def score_innovation_candidates(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    data = validate_prediction_frame(frame)
    score_end = int(config["landmarks"]["score_end_cycle"])
    scales = [
        float(value)
        for value in config["innovation_state"]["projection_scales"]
    ]
    prediction_clip = tuple(
        float(value)
        for value in config["innovation_state"]["future_prediction_clip_pct"]
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
            if not future.index.isin(previous.index).all():
                raise FastChargeV5PairwiseError(
                    "Previous issuance lacks the current common future support"
                )
            previous_future = previous.loc[future.index]
            residuals = history["observed_retention_pct"].to_numpy(
                dtype=float
            ) - history["candidate_prediction_pct"].to_numpy(dtype=float)
            truth = future["observed_retention_pct"].to_numpy(dtype=float)
            center = future["candidate_prediction_pct"].to_numpy(dtype=float)
            previous_center = previous_future[
                "candidate_prediction_pct"
            ].to_numpy(dtype=float)
            base_mae = float(np.mean(np.abs(truth - center)))
            for scale in scales:
                correction, diagnostics = reissue_innovation_correction(
                    history.index.to_numpy(dtype=float),
                    residuals,
                    future.index.to_numpy(dtype=float),
                    previous_center,
                    center,
                    previous_prefix=int(previous_prefix),
                    current_prefix=int(current_prefix),
                    projection_scale=scale,
                    config=config,
                )
                updated = np.clip(center + correction, *prediction_clip)
                updated_mae = float(np.mean(np.abs(truth - updated)))
                maximum_slope = float(
                    config["innovation_state"][
                        "maximum_absolute_unassimilated_slope_pp_per_cycle"
                    ]
                )
                naive_slope = float(
                    np.clip(
                        diagnostics["history_theil_slope_pp_per_cycle"],
                        -maximum_slope,
                        maximum_slope,
                    )
                )
                naive_correction = np.clip(
                    scale
                    * naive_slope
                    * (
                        future.index.to_numpy(dtype=float)
                        - float(current_prefix)
                    ),
                    -float(
                        config["innovation_state"][
                            "absolute_correction_cap_pp"
                        ]
                    ),
                    float(
                        config["innovation_state"][
                            "absolute_correction_cap_pp"
                        ]
                    ),
                )
                naive_updated_mae = float(
                    np.mean(
                        np.abs(
                            truth
                            - np.clip(
                                center + naive_correction, *prediction_clip
                            )
                        )
                    )
                )
                rows.append(
                    {
                        "cell_id": str(cell_id),
                        "previous_prefix_cycle": int(previous_prefix),
                        "current_prefix_cycle": int(current_prefix),
                        "candidate_id": f"innovation_a{_token(scale)}",
                        "projection_scale": scale,
                        "history_row_count": len(history),
                        "future_row_count": len(future),
                        "base_trajectory_mae_pp": base_mae,
                        "updated_trajectory_mae_pp": updated_mae,
                        "raw_delta_mae_pp": updated_mae - base_mae,
                        "naive_history_slope_updated_mae_pp": (
                            naive_updated_mae
                        ),
                        "naive_history_slope_delta_mae_pp": (
                            naive_updated_mae - base_mae
                        ),
                        "mean_absolute_correction_pp": float(
                            np.mean(np.abs(correction))
                        ),
                        "maximum_absolute_correction_pp": float(
                            np.max(np.abs(correction))
                        ),
                        **diagnostics,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "projection_scale", "cell_id"],
        kind="stable",
        ignore_index=True,
    )


def innovation_gates(
    config: Mapping[str, object],
) -> tuple[InnovationGate, ...]:
    settings = config["activation_gate_candidates"]
    state = config["innovation_state"]
    history_sign = bool(
        settings["require_history_all_recent_slope_sign_agreement"]
    )
    innovation_sign = bool(
        settings["require_unassimilated_history_slope_sign_agreement"]
    )
    gates = [InnovationGate("no_update")]
    for scale in state["projection_scales"]:
        for change in settings[
            "minimum_projected_unassimilated_change_pp"
        ]:
            for correlation in settings[
                "minimum_absolute_history_spearman"
            ]:
                gates.append(
                    InnovationGate(
                        gate_id=(
                            f"a{_token(float(scale))}"
                            f"_d{_token(float(change))}"
                            f"_r{_token(float(correlation))}"
                        ),
                        projection_scale=float(scale),
                        minimum_projected_innovation_change_pp=float(change),
                        minimum_absolute_history_spearman=float(correlation),
                        require_history_slope_sign_agreement=history_sign,
                        require_innovation_history_sign_agreement=(
                            innovation_sign
                        ),
                    )
                )
    expected = int(settings["candidate_count_including_fallback"])
    if len(gates) != expected:
        raise FastChargeV5PairwiseError(
            "Configured V7 activation-gate count changed"
        )
    return tuple(gates)


def _gate_rows(
    table: pd.DataFrame, gate: InnovationGate
) -> tuple[pd.DataFrame, np.ndarray]:
    if gate.gate_id == "no_update":
        rows = table.drop_duplicates(
            ["cell_id", "previous_prefix_cycle", "current_prefix_cycle"]
        ).copy()
        return rows, np.zeros(len(rows), dtype=bool)
    if (
        gate.projection_scale is None
        or gate.minimum_projected_innovation_change_pp is None
        or gate.minimum_absolute_history_spearman is None
    ):
        raise FastChargeV5PairwiseError(
            f"Incomplete V7 innovation gate: {gate.gate_id}"
        )
    rows = table.loc[
        np.isclose(
            table["projection_scale"].to_numpy(dtype=float),
            gate.projection_scale,
        )
    ].copy()
    active = (
        rows["projected_unassimilated_change_pp"].to_numpy(dtype=float)
        >= gate.minimum_projected_innovation_change_pp
    ) & (
        rows["absolute_history_spearman"].to_numpy(dtype=float)
        >= gate.minimum_absolute_history_spearman
    )
    if gate.require_history_slope_sign_agreement:
        active &= rows["history_slope_sign_agreement"].to_numpy(dtype=bool)
    if gate.require_innovation_history_sign_agreement:
        active &= rows[
            "unassimilated_history_slope_sign_agreement"
        ].to_numpy(dtype=bool)
    return rows, active


def summarize_innovation_gates(
    table: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    eligibility = config["activation_selection"]["eligibility"]
    rows: list[dict[str, object]] = []
    for (previous, current), transition in table.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle"], sort=True
    ):
        for gate in innovation_gates(config):
            candidates, active = _gate_rows(transition, gate)
            raw = candidates["raw_delta_mae_pp"].to_numpy(dtype=float)
            if gate.gate_id == "no_update":
                raw = np.zeros(len(candidates), dtype=float)
            gated = np.where(active, raw, 0.0)
            active_deltas = raw[active]
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
                and float(np.mean(gated))
                < float(
                    eligibility["mean_gated_delta_mae_must_be_below_pp"]
                )
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
                    "projection_scale": gate.projection_scale,
                    "physical_cell_count": int(
                        candidates["cell_id"].nunique()
                    ),
                    "activation_count": int(np.sum(active)),
                    "activation_coverage": coverage,
                    "activation_precision": precision,
                    "mean_base_trajectory_mae_pp": float(
                        candidates["base_trajectory_mae_pp"].mean()
                    ),
                    "mean_gated_delta_mae_pp": float(np.mean(gated)),
                    "active_p90_delta_mae_pp": active_p90,
                    "active_maximum_delta_mae_pp": active_maximum,
                    "eligible": eligible,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "gate_id"],
        kind="stable",
        ignore_index=True,
    )


def select_innovation_gates(summary: pd.DataFrame) -> dict[int, str]:
    selected: dict[int, str] = {}
    for current, group in summary.groupby("current_prefix_cycle", sort=True):
        eligible = group.loc[group["eligible"]].sort_values(
            [
                "active_maximum_delta_mae_pp",
                "active_p90_delta_mae_pp",
                "mean_gated_delta_mae_pp",
                "activation_coverage",
                "gate_id",
            ],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
        selected[int(current)] = (
            str(eligible.iloc[0]["gate_id"])
            if len(eligible)
            else "no_update"
        )
    return selected


def _score_selected_holdout(
    held_out: pd.DataFrame,
    selected: Mapping[int, str],
    config: Mapping[str, object],
    *,
    selection_fit_cell_count: int,
    held_out_batch: str | None = None,
) -> list[dict[str, object]]:
    lookup = {gate.gate_id: gate for gate in innovation_gates(config)}
    rows: list[dict[str, object]] = []
    for current, gate_id in sorted(selected.items()):
        transition = held_out.loc[
            held_out["current_prefix_cycle"] == current
        ]
        candidates, active = _gate_rows(transition, lookup[gate_id])
        if len(candidates) != transition["cell_id"].nunique():
            raise FastChargeV5PairwiseError(
                "V7 holdout audit did not resolve one row per physical cell"
            )
        for index, (_, record) in enumerate(candidates.iterrows()):
            is_active = bool(active[index])
            raw_delta = (
                float(record["raw_delta_mae_pp"])
                if gate_id != "no_update"
                else 0.0
            )
            naive_delta = (
                float(record["naive_history_slope_delta_mae_pp"])
                if gate_id != "no_update"
                else 0.0
            )
            base_mae = float(record["base_trajectory_mae_pp"])
            row: dict[str, object] = {
                "cell_id": str(record["cell_id"]),
                "previous_prefix_cycle": int(
                    record["previous_prefix_cycle"]
                ),
                "current_prefix_cycle": int(current),
                "selected_gate_id": gate_id,
                "selection_fit_cell_count": selection_fit_cell_count,
                "activated": is_active,
                "base_trajectory_mae_pp": base_mae,
                "updated_trajectory_mae_pp": (
                    base_mae + raw_delta if is_active else base_mae
                ),
                "raw_delta_mae_pp": raw_delta,
                "gated_delta_mae_pp": raw_delta if is_active else 0.0,
                "matched_naive_history_slope_delta_mae_pp": (
                    naive_delta if is_active else 0.0
                ),
                "history_theil_slope_pp_per_cycle": float(
                    record["history_theil_slope_pp_per_cycle"]
                ),
                "reissue_shift_theil_slope_pp_per_cycle": float(
                    record["reissue_shift_theil_slope_pp_per_cycle"]
                ),
                "bounded_unassimilated_slope_pp_per_cycle": float(
                    record["bounded_unassimilated_slope_pp_per_cycle"]
                ),
                "projected_unassimilated_change_pp": float(
                    record["projected_unassimilated_change_pp"]
                ),
                "absolute_history_spearman": float(
                    record["absolute_history_spearman"]
                ),
                "history_slope_sign_agreement": bool(
                    record["history_slope_sign_agreement"]
                ),
                "unassimilated_history_slope_sign_agreement": bool(
                    record["unassimilated_history_slope_sign_agreement"]
                ),
            }
            if held_out_batch is not None:
                row["held_out_batch"] = held_out_batch
            rows.append(row)
    return rows


def nested_innovation_gate_audit(
    table: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    cell_ids = sorted(str(value) for value in table["cell_id"].unique())
    rows: list[dict[str, object]] = []
    for held_out in cell_ids:
        fit = table.loc[table["cell_id"] != held_out]
        selected = select_innovation_gates(
            summarize_innovation_gates(fit, config)
        )
        held = table.loc[table["cell_id"] == held_out]
        rows.extend(
            _score_selected_holdout(
                held,
                selected,
                config,
                selection_fit_cell_count=len(cell_ids) - 1,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "cell_id"],
        kind="stable",
        ignore_index=True,
    )


def batch_holdout_innovation_gate_audit(
    table: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    pattern = re.compile(
        config["future_blind_nomination_gate"]["batch_holdout"][
            "batch_id_regex"
        ]
    )
    cell_to_batch: dict[str, str] = {}
    for cell_id in sorted(str(value) for value in table["cell_id"].unique()):
        match = pattern.match(cell_id)
        if match is None:
            raise FastChargeV5PairwiseError(
                f"V7 batch challenge cannot parse cell id: {cell_id}"
            )
        cell_to_batch[cell_id] = str(match.group("batch"))
    batches = sorted(set(cell_to_batch.values()))
    if len(batches) < 2:
        raise FastChargeV5PairwiseError(
            "V7 batch challenge requires at least two batches"
        )
    batch_series = table["cell_id"].map(cell_to_batch)
    rows: list[dict[str, object]] = []
    for held_batch in batches:
        fit = table.loc[batch_series != held_batch]
        held = table.loc[batch_series == held_batch]
        selected = select_innovation_gates(
            summarize_innovation_gates(fit, config)
        )
        rows.extend(
            _score_selected_holdout(
                held,
                selected,
                config,
                selection_fit_cell_count=int(fit["cell_id"].nunique()),
                held_out_batch=held_batch,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["current_prefix_cycle", "held_out_batch", "cell_id"],
        kind="stable",
        ignore_index=True,
    )


def _optional_active_metrics(
    group: pd.DataFrame,
) -> tuple[int, float | None, float | None, float | None]:
    active = group.loc[group["activated"]]
    deltas = active["gated_delta_mae_pp"].to_numpy(dtype=float)
    if len(deltas) == 0:
        return 0, None, None, None
    return (
        len(deltas),
        float(np.mean(deltas < 0.0)),
        float(np.quantile(deltas, 0.9)),
        float(np.max(deltas)),
    )


def innovation_nomination_summary(
    nested: pd.DataFrame,
    batch_holdout: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, object]:
    cell_gate = config["future_blind_nomination_gate"]["cell_holdout"]
    batch_gate = config["future_blind_nomination_gate"]["batch_holdout"]
    transition_rows: list[dict[str, object]] = []
    nominated: list[int] = []
    for (previous, current), cell_group in nested.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle"], sort=True
    ):
        active_count, precision, active_p90, active_maximum = (
            _optional_active_metrics(cell_group)
        )
        active_rows = cell_group.loc[cell_group["activated"]]
        naive_active = active_rows[
            "matched_naive_history_slope_delta_mae_pp"
        ].to_numpy(dtype=float)
        coverage = float(cell_group["activated"].mean())
        mean_delta = float(cell_group["gated_delta_mae_pp"].mean())
        modal_fraction = float(
            cell_group["selected_gate_id"].value_counts(normalize=True).iloc[0]
        )
        cell_passed = bool(
            active_count >= int(cell_gate["minimum_activation_count"])
            and coverage >= float(cell_gate["minimum_activation_coverage"])
            and precision is not None
            and precision >= float(cell_gate["minimum_activation_precision"])
            and mean_delta
            < float(cell_gate["mean_gated_delta_mae_must_be_below_pp"])
            and active_p90 is not None
            and active_p90
            <= float(cell_gate["maximum_active_p90_delta_mae_pp"])
            and active_maximum is not None
            and active_maximum
            <= float(cell_gate["maximum_active_delta_mae_pp"])
            and modal_fraction
            >= float(cell_gate["minimum_modal_gate_selection_fraction"])
        )

        batch_rows: list[dict[str, object]] = []
        batch_subset = batch_holdout.loc[
            batch_holdout["current_prefix_cycle"] == current
        ]
        for held_batch, batch_group in batch_subset.groupby(
            "held_out_batch", sort=True
        ):
            (
                batch_active_count,
                batch_precision,
                _,
                batch_maximum,
            ) = _optional_active_metrics(batch_group)
            selected_counts = {
                str(key): int(value)
                for key, value in sorted(
                    batch_group["selected_gate_id"].value_counts().items()
                )
            }
            nonfallback = all(key != "no_update" for key in selected_counts)
            batch_mean_delta = float(
                batch_group["gated_delta_mae_pp"].mean()
            )
            passed = bool(
                (
                    nonfallback
                    or not bool(
                        batch_gate[
                            "require_nonfallback_selection_for_every_held_out_batch"
                        ]
                    )
                )
                and batch_active_count
                >= int(
                    batch_gate[
                        "minimum_activation_count_in_every_held_out_batch"
                    ]
                )
                and batch_precision is not None
                and batch_precision
                >= float(
                    batch_gate[
                        "minimum_activation_precision_in_every_held_out_batch"
                    ]
                )
                and batch_mean_delta
                <= float(
                    batch_gate[
                        "maximum_mean_gated_delta_mae_pp_in_every_held_out_batch"
                    ]
                )
                and batch_maximum is not None
                and batch_maximum
                <= float(
                    batch_gate[
                        "maximum_active_delta_mae_pp_in_every_held_out_batch"
                    ]
                )
            )
            batch_rows.append(
                {
                    "held_out_batch": str(held_batch),
                    "physical_cell_count": int(
                        batch_group["cell_id"].nunique()
                    ),
                    "selected_gate_counts": selected_counts,
                    "activation_count": batch_active_count,
                    "activation_precision": batch_precision,
                    "mean_gated_delta_mae_pp": batch_mean_delta,
                    "active_maximum_delta_mae_pp": batch_maximum,
                    "passed": passed,
                }
            )
        batch_passed = bool(
            len(batch_rows)
            >= int(batch_gate["minimum_held_out_batch_count"])
            and all(row["passed"] for row in batch_rows)
        )
        nominated_transition = bool(cell_passed and batch_passed)
        if nominated_transition:
            nominated.append(int(current))
        transition_rows.append(
            {
                "previous_prefix_cycle": int(previous),
                "current_prefix_cycle": int(current),
                "physical_cell_count": int(cell_group["cell_id"].nunique()),
                "mean_base_trajectory_mae_pp": float(
                    cell_group["base_trajectory_mae_pp"].mean()
                ),
                "mean_updated_trajectory_mae_pp": float(
                    cell_group["updated_trajectory_mae_pp"].mean()
                ),
                "cell_holdout": {
                    "activation_count": active_count,
                    "activation_coverage": coverage,
                    "activation_precision": precision,
                    "mean_gated_delta_mae_pp": mean_delta,
                    "active_p90_delta_mae_pp": active_p90,
                    "active_maximum_delta_mae_pp": active_maximum,
                    "modal_gate_selection_fraction": modal_fraction,
                    "selected_gate_counts": {
                        str(key): int(value)
                        for key, value in sorted(
                            cell_group[
                                "selected_gate_id"
                            ].value_counts().items()
                        )
                    },
                    "matched_active_ablation": {
                        "comparison": (
                            "same_activated_cells_and_projection_scale"
                        ),
                        "reissue_aware_mean_delta_mae_pp": (
                            float(
                                active_rows["gated_delta_mae_pp"].mean()
                            )
                            if active_count
                            else None
                        ),
                        "naive_history_slope_mean_delta_mae_pp": (
                            float(np.mean(naive_active))
                            if active_count
                            else None
                        ),
                        "reissue_aware_activation_precision": precision,
                        "naive_history_slope_activation_precision": (
                            float(np.mean(naive_active < 0.0))
                            if active_count
                            else None
                        ),
                        "reissue_aware_maximum_delta_mae_pp": active_maximum,
                        "naive_history_slope_maximum_delta_mae_pp": (
                            float(np.max(naive_active))
                            if active_count
                            else None
                        ),
                    },
                    "passed": cell_passed,
                },
                "batch_holdout": {
                    "held_out_batch_count": len(batch_rows),
                    "batches": batch_rows,
                    "passed": batch_passed,
                },
                "nominated_for_future_blind_test": nominated_transition,
            }
        )
    return {
        "any_transition_nominated": bool(nominated),
        "nominated_current_prefix_cycles": nominated,
        "transitions": transition_rows,
    }


__all__ = [
    "InnovationGate",
    "batch_holdout_innovation_gate_audit",
    "innovation_gates",
    "innovation_nomination_summary",
    "nested_innovation_gate_audit",
    "reissue_innovation_correction",
    "score_innovation_candidates",
    "select_innovation_gates",
    "summarize_innovation_gates",
]
