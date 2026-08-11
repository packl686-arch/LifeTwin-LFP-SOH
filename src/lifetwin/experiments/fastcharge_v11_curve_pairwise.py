"""P100 Delta-Q curve-feature challengers for the frozen V5 pairwise model.

Only cycle-10-to-100 curve features are accepted.  Model selection is confined
to the 41 public MATR training cells, with held-out physical cells removed from
both sides of every pair and from robust-scaling estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

import lifetwin.experiments.fastcharge_v5_pairwise as v5
from lifetwin.experiments.fastcharge_trajectory_portability import (
    _normalization_capacity,
    _retention,
)


CURVE_FEATURE_IDS = (
    "log10_delta_q_variance",
    "delta_q_min_ah",
    "delta_q_mean_ah",
    "delta_q_abs_area_ah_v",
    "delta_q_skewness",
    "delta_q_kurtosis",
)
CURVE_INPUT_COLUMNS = (
    "cell_id",
    "early_cycle",
    "late_cycle",
    *CURVE_FEATURE_IDS,
)
CHALLENGER_IDS = (
    "v11_delta_q_residual_only",
    "v11_delta_q_residual_plus_geometry",
)
PREFIX_CYCLE = 100
SCORE_END_CYCLE = 300


@dataclass(frozen=True)
class RobustCurveScaler:
    cell_ids: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray


def validate_curve_features(
    frame: pd.DataFrame,
    *,
    required_cell_ids: Sequence[str],
) -> pd.DataFrame:
    if tuple(frame.columns) != CURVE_INPUT_COLUMNS:
        raise v5.FastChargeV5PairwiseError("V11 curve-feature columns changed")
    if frame.empty:
        raise v5.FastChargeV5PairwiseError("V11 curve-feature input is empty")
    data = frame.copy()
    data["cell_id"] = data["cell_id"].astype(str).str.strip()
    if (data["cell_id"] == "").any() or data["cell_id"].duplicated().any():
        raise v5.FastChargeV5PairwiseError(
            "V11 curve-feature cell identifiers are empty or duplicated"
        )
    numeric_columns = ("early_cycle", "late_cycle", *CURVE_FEATURE_IDS)
    numeric = data.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise v5.FastChargeV5PairwiseError("V11 curve features are non-finite")
    data.loc[:, numeric_columns] = numeric
    if set(data["early_cycle"].astype(int)) != {10} or set(
        data["late_cycle"].astype(int)
    ) != {PREFIX_CYCLE}:
        raise v5.FastChargeV5PairwiseError(
            "V11 accepts only cycle-10-to-100 Delta-Q features"
        )
    required = set(str(value) for value in required_cell_ids)
    observed = set(data["cell_id"])
    if not required <= observed:
        raise v5.FastChargeV5PairwiseError(
            f"V11 curve features miss cells: {sorted(required - observed)}"
        )
    return data.loc[data["cell_id"].isin(required), CURVE_INPUT_COLUMNS].sort_values(
        "cell_id", kind="stable", ignore_index=True
    )


def fit_curve_scaler(
    features: pd.DataFrame,
    fit_cell_ids: Sequence[str],
) -> RobustCurveScaler:
    fit_ids = tuple(sorted(set(str(value) for value in fit_cell_ids)))
    subset = features.set_index("cell_id").loc[list(fit_ids), CURVE_FEATURE_IDS]
    values = subset.to_numpy(dtype=float)
    median = np.median(values, axis=0)
    absolute = np.median(np.abs(values - median), axis=0) * 1.4826
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    standard = np.std(values, axis=0)
    scale = np.where(absolute > 1e-10, absolute, iqr_scale)
    scale = np.where(scale > 1e-10, scale, standard)
    scale = np.maximum(scale, 1e-10)
    if not np.isfinite(median).all() or not np.isfinite(scale).all():
        raise v5.FastChargeV5PairwiseError("V11 robust curve scaling failed")
    return RobustCurveScaler(fit_ids, median.astype(float), scale.astype(float))


def transform_curve_features(
    features: pd.DataFrame,
    scaler: RobustCurveScaler,
    cell_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    indexed = features.set_index("cell_id")
    result: dict[str, np.ndarray] = {}
    for cell_id in sorted(set(str(value) for value in cell_ids)):
        values = indexed.loc[cell_id, CURVE_FEATURE_IDS].to_numpy(dtype=float)
        transformed = (values - scaler.median) / scaler.scale
        if (
            transformed.shape != (len(CURVE_FEATURE_IDS),)
            or not np.isfinite(transformed).all()
        ):
            raise v5.FastChargeV5PairwiseError(
                f"V11 transformed curve features are invalid for {cell_id}"
            )
        result[cell_id] = transformed
    return result


def build_curve_pairwise_training_matrix(
    cells: Mapping[str, pd.DataFrame],
    curve_features: pd.DataFrame,
    core_config: Mapping[str, object],
    *,
    scaler: RobustCurveScaler,
    anchor_stride: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    cell_ids = sorted(cells)
    if tuple(cell_ids) != scaler.cell_ids:
        raise v5.FastChargeV5PairwiseError(
            "V11 scaler must be fit on exactly the pair-training cells"
        )
    base = v5._cell_resources(cells, PREFIX_CYCLE, core_config)
    curves = transform_curve_features(curve_features, scaler, cell_ids)
    descriptors = {
        cell_id: np.concatenate([base["descriptors"][cell_id], curves[cell_id]])
        for cell_id in cell_ids
    }
    anchors = v5._anchor_cycles(PREFIX_CYCLE, SCORE_END_CYCLE, anchor_stride)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    pair_count = 0
    for left_index, left_id in enumerate(cell_ids[:-1]):
        left_change = (
            base["retentions"][left_id][anchors - 1]
            - base["retentions"][left_id][PREFIX_CYCLE - 1]
        )
        for right_id in cell_ids[left_index + 1 :]:
            right_change = (
                base["retentions"][right_id][anchors - 1]
                - base["retentions"][right_id][PREFIX_CYCLE - 1]
            )
            difference = descriptors[left_id] - descriptors[right_id]
            for forecast_cycle, label in zip(
                anchors, left_change - right_change, strict=True
            ):
                horizon = (forecast_cycle - PREFIX_CYCLE) / (
                    SCORE_END_CYCLE - PREFIX_CYCLE
                )
                vector = v5._pair_features(difference, horizon)
                rows.extend((vector, -vector))
                labels.extend((float(label), float(-label)))
            pair_count += 1
    matrix = np.vstack(rows).astype(float)
    target = np.asarray(labels, dtype=float)
    audit = {
        "physical_cell_count": len(cell_ids),
        "unordered_pair_count": pair_count,
        "ordered_pair_count": pair_count * 2,
        "anchor_cycles": anchors.tolist(),
        "training_row_count": len(target),
        "feature_count": matrix.shape[1],
        "target_cell_ids": cell_ids,
        "reference_cell_ids": cell_ids,
        "curve_scaler_fit_cell_ids": list(scaler.cell_ids),
        "held_out_curve_features_used_for_scaling": False,
    }
    return matrix, target, audit


def predict_curve_pairwise_trajectory(
    estimator: RegressorMixin,
    target_cell_id: str,
    target_prefix: pd.DataFrame,
    reference_cells: Mapping[str, pd.DataFrame],
    curve_features: pd.DataFrame,
    core_config: Mapping[str, object],
    *,
    scaler: RobustCurveScaler,
    geometry_mode: str,
    geometry_curve_weight: float,
    neighbor_count: int = 12,
    aggregation: str = "weighted_mean",
    reference_resources: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if geometry_mode not in {"base_only", "base_plus_curve"}:
        raise v5.FastChargeV5PairwiseError("Unknown V11 geometry mode")
    if not 0.0 <= geometry_curve_weight <= 1.0:
        raise v5.FastChargeV5PairwiseError("Invalid V11 curve geometry weight")
    if reference_resources is None:
        reference_resources = v5._cell_resources(
            reference_cells, PREFIX_CYCLE, core_config
        )
    reference_ids = sorted(reference_cells)
    if tuple(reference_ids) != scaler.cell_ids:
        raise v5.FastChargeV5PairwiseError(
            "V11 prediction references differ from scaler-fit cells"
        )
    base_target = v5._descriptor(target_prefix, core_config)
    curve_map = transform_curve_features(
        curve_features, scaler, [*reference_ids, str(target_cell_id)]
    )
    target_curve = curve_map[str(target_cell_id)]
    target_descriptor = np.concatenate([base_target, target_curve])
    base_nearest, _, base_distances = v5._reference_geometry(
        base_target,
        reference_resources["descriptors"],
        core_config,
        neighbor_count=len(reference_ids),
    )
    if set(base_nearest) != set(reference_ids):
        raise v5.FastChargeV5PairwiseError("V11 base geometry lost references")
    if geometry_mode == "base_only":
        distances = base_distances
    else:
        distances = {}
        for reference_id in reference_ids:
            curve_distance = float(
                np.sqrt(np.mean(np.square(target_curve - curve_map[reference_id])))
            )
            distances[reference_id] = math.sqrt(
                (1.0 - geometry_curve_weight) * float(base_distances[reference_id]) ** 2
                + geometry_curve_weight * curve_distance**2
            )
    nearest = sorted(reference_ids, key=lambda cell_id: (distances[cell_id], cell_id))[
        :neighbor_count
    ]
    epsilon = float(
        core_config["base_experts"]["nearest_neighbor_delta_transfer"][
            "distance_epsilon"
        ]
    )
    raw_weights = np.asarray(
        [1.0 / (float(distances[cell_id]) + epsilon) for cell_id in nearest]
    )
    weights = raw_weights / float(np.sum(raw_weights))
    target_normalization = _normalization_capacity(target_prefix)
    target_last = float(_retention(target_prefix, target_normalization)[-1])
    forecast = np.arange(PREFIX_CYCLE + 1, SCORE_END_CYCLE + 1, dtype=int)
    feature_blocks: list[np.ndarray] = []
    reference_changes: list[np.ndarray] = []
    for reference_id in nearest:
        reference_descriptor = np.concatenate(
            [
                reference_resources["descriptors"][reference_id],
                curve_map[reference_id],
            ]
        )
        difference = target_descriptor - reference_descriptor
        feature_blocks.append(
            np.vstack(
                [
                    v5._pair_features(
                        difference,
                        (cycle - PREFIX_CYCLE) / (SCORE_END_CYCLE - PREFIX_CYCLE),
                    )
                    for cycle in forecast
                ]
            )
        )
        retention = reference_resources["retentions"][reference_id]
        reference_changes.append(retention[forecast - 1] - retention[PREFIX_CYCLE - 1])
    positive = np.vstack(feature_blocks)
    correction = 0.5 * (
        np.asarray(estimator.predict(positive), dtype=float).reshape(-1)
        - np.asarray(estimator.predict(-positive), dtype=float).reshape(-1)
    )
    correction = correction.reshape(len(nearest), len(forecast))
    trajectories = target_last + np.vstack(reference_changes) + correction
    prediction = v5.aggregate_reference_trajectories(
        np.clip(trajectories, 0.0, 110.0), weights, aggregation
    )
    return prediction, {
        "reference_cell_ids": nearest,
        "reference_weights": {
            cell_id: float(value)
            for cell_id, value in zip(nearest, weights, strict=True)
        },
        "reference_distances": {
            cell_id: float(distances[cell_id]) for cell_id in nearest
        },
        "geometry_mode": geometry_mode,
        "geometry_curve_weight": (
            geometry_curve_weight if geometry_mode == "base_plus_curve" else 0.0
        ),
        "aggregation": aggregation,
        "target_curve_cycles": [10, PREFIX_CYCLE],
        "target_future_outcomes_used": False,
    }


def assert_curve_pair_fold_firewall(
    fit_cell_ids: Sequence[str],
    validation_cell_ids: Sequence[str],
    pair_audit: Mapping[str, object],
    scaler: RobustCurveScaler,
) -> None:
    v5.assert_pair_fold_firewall(fit_cell_ids, validation_cell_ids, pair_audit)
    fit = set(str(value) for value in fit_cell_ids)
    held = set(str(value) for value in validation_cell_ids)
    if set(scaler.cell_ids) != fit or held & set(scaler.cell_ids):
        raise v5.FastChargeV5PairwiseError(
            "Held-out cell entered V11 curve-feature scaling"
        )
    if set(pair_audit["curve_scaler_fit_cell_ids"]) != fit:
        raise v5.FastChargeV5PairwiseError(
            "V11 pair audit does not match the curve scaler"
        )


__all__ = [
    "CHALLENGER_IDS",
    "CURVE_FEATURE_IDS",
    "CURVE_INPUT_COLUMNS",
    "PREFIX_CYCLE",
    "RobustCurveScaler",
    "SCORE_END_CYCLE",
    "assert_curve_pair_fold_firewall",
    "build_curve_pairwise_training_matrix",
    "fit_curve_scaler",
    "predict_curve_pairwise_trajectory",
    "transform_curve_features",
    "validate_curve_features",
]
