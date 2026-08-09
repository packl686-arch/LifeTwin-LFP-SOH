"""Outcome-exposed V5 pairwise development on the MATR FastCharge cohort.

The module deliberately keeps the public evaluation suffix out of fitting.
Candidate selection uses physical-cell folds drawn only from the 41 training
cells.  A held-out cell is removed from both sides of every training pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lifetwin.experiments.fastcharge_trajectory_portability import (
    FEATURE_IDS,
    _distance_map,
    _normalization_capacity,
    _retention,
    _trajectory_signature,
)


DEVELOPMENT_EVIDENCE_ROLE = (
    "outcome_exposed_public_development_not_independent_confirmation"
)
PAIRWISE_AGGREGATIONS = ("single_nearest", "weighted_mean", "weighted_median")
PAIR_BASIS_IDS = ("constant", "sqrt_horizon", "horizon", "horizon_squared")
DESCRIPTOR_IDS = (
    *FEATURE_IDS,
    "retention_last_pct",
    "retention_recent_mean_pct",
    "retention_recent_std_pct",
    "retention_change_from_cycle_5_pp",
    "log_internal_resistance_last",
    "log_internal_resistance_change",
    "temperature_recent_mean_c",
    "log_charge_duration_recent_mean",
    "energy_efficiency_recent_mean",
)


class FastChargeV5PairwiseError(ValueError):
    """Raised when a V5 development firewall or data invariant is violated."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    parameters: tuple[tuple[str, float | int], ...] = ()

    def parameter_dict(self) -> dict[str, float | int]:
        return dict(self.parameters)


DIRECT_MODEL_SPECS = (
    ModelSpec("direct_ridge_a1", "ridge", (("alpha", 1.0),)),
    ModelSpec("direct_ridge_a100", "ridge", (("alpha", 100.0),)),
    ModelSpec("direct_pcr95", "pcr", (("variance", 0.95), ("alpha", 1.0))),
    ModelSpec("direct_pls6", "pls", (("components", 6),)),
    ModelSpec(
        "direct_extra_trees_leaf5",
        "extra_trees",
        (("min_samples_leaf", 5), ("n_estimators", 128)),
    ),
    ModelSpec(
        "direct_hist_gbdt_leaf20",
        "hist_gbdt",
        (("min_samples_leaf", 20), ("max_iter", 160)),
    ),
)

PAIRWISE_MODEL_SPECS = (
    ModelSpec("pairwise_ridge_a0p1", "ridge", (("alpha", 0.1),)),
    ModelSpec("pairwise_ridge_a1", "ridge", (("alpha", 1.0),)),
    ModelSpec("pairwise_ridge_a10", "ridge", (("alpha", 10.0),)),
    ModelSpec(
        "pairwise_huber",
        "huber",
        (("alpha", 0.0001), ("epsilon", 1.35), ("max_iter", 500)),
    ),
    ModelSpec(
        "pairwise_extra_trees_leaf5",
        "extra_trees",
        (("min_samples_leaf", 5), ("n_estimators", 128)),
    ),
    ModelSpec(
        "pairwise_hist_gbdt_leaf40",
        "hist_gbdt",
        (("min_samples_leaf", 40), ("max_iter", 160)),
    ),
)


def _validated_cells(
    cycles: pd.DataFrame,
    *,
    required_support: int,
) -> dict[str, pd.DataFrame]:
    required = {
        "cell_id",
        "cycle_index",
        "discharge_capacity_ah",
        "internal_resistance_ohm",
        "temperature_max_c",
        "charge_time_s",
        "energy_efficiency",
    }
    missing = required - set(cycles.columns)
    if missing:
        raise FastChargeV5PairwiseError(
            f"V5 cycles are missing columns: {sorted(missing)}"
        )
    result: dict[str, pd.DataFrame] = {}
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        cell = group.sort_values("cycle_index", kind="stable").reset_index(drop=True)
        observed = cell["cycle_index"].to_numpy(dtype=int)
        expected = np.arange(1, required_support + 1, dtype=int)
        if len(observed) < required_support or not np.array_equal(
            observed[:required_support], expected
        ):
            raise FastChargeV5PairwiseError(
                f"Cell {cell_id} lacks contiguous support through {required_support}"
            )
        cell = cell.loc[cell["cycle_index"] <= required_support].reset_index(drop=True)
        numeric = cell.loc[:, sorted(required - {"cell_id"})].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any() or not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all():
            raise FastChargeV5PairwiseError(f"Cell {cell_id} contains invalid values")
        result[str(cell_id)] = cell
    if len(result) < 2:
        raise FastChargeV5PairwiseError("V5 requires at least two physical cells")
    return result


def _descriptor(
    prefix: pd.DataFrame,
    core_config: Mapping[str, object],
) -> np.ndarray:
    signature = _trajectory_signature(prefix, core_config)
    normalization = _normalization_capacity(prefix)
    retention = _retention(prefix, normalization)
    recent_count = min(10, len(prefix))
    resistance = np.log(
        np.maximum(prefix["internal_resistance_ohm"].to_numpy(dtype=float), 1e-4)
    )
    charge_duration = np.log(
        np.maximum(prefix["charge_time_s"].to_numpy(dtype=float), 1.0)
    )
    values = [
        *(float(signature[feature_id]) for feature_id in FEATURE_IDS),
        float(retention[-1]),
        float(np.mean(retention[-recent_count:])),
        float(np.std(retention[-recent_count:])),
        float(retention[-1] - retention[4]),
        float(resistance[-1]),
        float(resistance[-1] - np.median(resistance[:5])),
        float(np.mean(prefix["temperature_max_c"].to_numpy(dtype=float)[-recent_count:])),
        float(np.mean(charge_duration[-recent_count:])),
        float(
            np.mean(
                prefix["energy_efficiency"].to_numpy(dtype=float)[-recent_count:]
            )
        ),
    ]
    array = np.asarray(values, dtype=float)
    if array.shape != (len(DESCRIPTOR_IDS),) or not np.isfinite(array).all():
        raise FastChargeV5PairwiseError("V5 prefix descriptor is invalid")
    return array


def _cell_resources(
    cells: Mapping[str, pd.DataFrame],
    prefix_cycle: int,
    core_config: Mapping[str, object],
) -> dict[str, object]:
    descriptors: dict[str, np.ndarray] = {}
    signatures: dict[str, dict[str, float]] = {}
    retentions: dict[str, np.ndarray] = {}
    prefixes: dict[str, pd.DataFrame] = {}
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        prefix = cell.loc[cell["cycle_index"] <= prefix_cycle].reset_index(drop=True)
        if len(prefix) != prefix_cycle:
            raise FastChargeV5PairwiseError(
                f"Cell {cell_id} prefix {prefix_cycle} is incomplete"
            )
        normalization = _normalization_capacity(cell)
        prefixes[cell_id] = prefix
        descriptors[cell_id] = _descriptor(prefix, core_config)
        signatures[cell_id] = {
            feature_id: float(value)
            for feature_id, value in zip(
                FEATURE_IDS, descriptors[cell_id][: len(FEATURE_IDS)], strict=True
            )
        }
        retentions[cell_id] = _retention(cell, normalization)
    return {
        "descriptors": descriptors,
        "signatures": signatures,
        "retentions": retentions,
        "prefixes": prefixes,
    }


def _anchor_cycles(prefix_cycle: int, score_end_cycle: int, stride: int) -> np.ndarray:
    if prefix_cycle < 5 or score_end_cycle <= prefix_cycle or stride < 1:
        raise FastChargeV5PairwiseError("Invalid V5 horizon design")
    values = np.arange(prefix_cycle + 1, score_end_cycle + 1, stride, dtype=int)
    return np.unique(np.append(values, score_end_cycle)).astype(int)


def _pair_features(difference: np.ndarray, horizon_fraction: float) -> np.ndarray:
    difference = np.asarray(difference, dtype=float)
    horizon = float(horizon_fraction)
    basis = np.asarray([1.0, math.sqrt(horizon), horizon, horizon * horizon])
    return np.concatenate([difference * value for value in basis])


def _direct_features(descriptor: np.ndarray, horizon_fraction: float) -> np.ndarray:
    descriptor = np.asarray(descriptor, dtype=float)
    horizon = float(horizon_fraction)
    root = math.sqrt(horizon)
    return np.concatenate(
        [
            descriptor,
            descriptor * root,
            descriptor * horizon,
            descriptor * horizon * horizon,
            np.asarray([root, horizon, horizon * horizon]),
        ]
    )


def build_pairwise_training_matrix(
    cells: Mapping[str, pd.DataFrame],
    prefix_cycle: int,
    score_end_cycle: int,
    core_config: Mapping[str, object],
    *,
    anchor_stride: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    resources = _cell_resources(cells, prefix_cycle, core_config)
    cell_ids = sorted(cells)
    anchors = _anchor_cycles(prefix_cycle, score_end_cycle, anchor_stride)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    pair_count = 0
    for left_index, left_id in enumerate(cell_ids[:-1]):
        left_descriptor = resources["descriptors"][left_id]
        left_retention = resources["retentions"][left_id]
        left_change = left_retention[anchors - 1] - left_retention[prefix_cycle - 1]
        for right_id in cell_ids[left_index + 1 :]:
            right_descriptor = resources["descriptors"][right_id]
            right_retention = resources["retentions"][right_id]
            right_change = (
                right_retention[anchors - 1] - right_retention[prefix_cycle - 1]
            )
            difference = left_descriptor - right_descriptor
            target = left_change - right_change
            for forecast_cycle, label in zip(anchors, target, strict=True):
                horizon = (forecast_cycle - prefix_cycle) / (
                    score_end_cycle - prefix_cycle
                )
                feature = _pair_features(difference, horizon)
                rows.extend((feature, -feature))
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
    }
    return matrix, target, audit


def build_direct_training_matrix(
    cells: Mapping[str, pd.DataFrame],
    prefix_cycle: int,
    score_end_cycle: int,
    core_config: Mapping[str, object],
    *,
    anchor_stride: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    resources = _cell_resources(cells, prefix_cycle, core_config)
    anchors = _anchor_cycles(prefix_cycle, score_end_cycle, anchor_stride)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for cell_id in sorted(cells):
        retention = resources["retentions"][cell_id]
        change = retention[anchors - 1] - retention[prefix_cycle - 1]
        for forecast_cycle, label in zip(anchors, change, strict=True):
            horizon = (forecast_cycle - prefix_cycle) / (
                score_end_cycle - prefix_cycle
            )
            rows.append(_direct_features(resources["descriptors"][cell_id], horizon))
            labels.append(float(label))
    matrix = np.vstack(rows).astype(float)
    target = np.asarray(labels, dtype=float)
    audit = {
        "physical_cell_count": len(cells),
        "anchor_cycles": anchors.tolist(),
        "training_row_count": len(target),
        "feature_count": matrix.shape[1],
        "target_cell_ids": sorted(cells),
    }
    return matrix, target, audit


def make_estimator(
    spec: ModelSpec,
    *,
    pairwise: bool,
    random_state: int = 20260809,
) -> RegressorMixin:
    parameters = spec.parameter_dict()
    intercept = not pairwise
    if spec.family == "ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(parameters["alpha"]), fit_intercept=intercept),
        )
    if spec.family == "huber":
        return make_pipeline(
            StandardScaler(),
            HuberRegressor(
                alpha=float(parameters["alpha"]),
                epsilon=float(parameters["epsilon"]),
                max_iter=int(parameters["max_iter"]),
                fit_intercept=intercept,
            ),
        )
    if spec.family == "pcr":
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=float(parameters["variance"]), svd_solver="full"),
            Ridge(alpha=float(parameters["alpha"]), fit_intercept=intercept),
        )
    if spec.family == "pls":
        return PLSRegression(
            n_components=int(parameters["components"]),
            scale=True,
            max_iter=1000,
        )
    if spec.family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(parameters["n_estimators"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=0.75,
            # A single worker keeps the research replay deterministic and also
            # works in restricted Windows execution environments.
            n_jobs=1,
            random_state=random_state,
        )
    if spec.family == "hist_gbdt":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=15,
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=1.0,
            random_state=random_state,
        )
    raise FastChargeV5PairwiseError(f"Unknown model family: {spec.family}")


def _reference_geometry(
    target_descriptor: np.ndarray,
    reference_descriptors: Mapping[str, np.ndarray],
    core_config: Mapping[str, object],
    *,
    neighbor_count: int,
) -> tuple[list[str], np.ndarray, dict[str, float]]:
    if len(reference_descriptors) < neighbor_count:
        raise FastChargeV5PairwiseError("Insufficient V5 reference support")
    target_signature = {
        feature_id: float(value)
        for feature_id, value in zip(
            FEATURE_IDS, target_descriptor[: len(FEATURE_IDS)], strict=True
        )
    }
    reference_signatures = {
        cell_id: {
            feature_id: float(value)
            for feature_id, value in zip(
                FEATURE_IDS, descriptor[: len(FEATURE_IDS)], strict=True
            )
        }
        for cell_id, descriptor in reference_descriptors.items()
    }
    distances = _distance_map(
        target_signature,
        reference_signatures,
        FEATURE_IDS,
        core_config,
    )
    nearest = sorted(distances, key=lambda cell_id: (distances[cell_id], cell_id))[
        :neighbor_count
    ]
    epsilon = float(
        core_config["base_experts"]["nearest_neighbor_delta_transfer"][
            "distance_epsilon"
        ]
    )
    raw = np.asarray([1.0 / (distances[cell_id] + epsilon) for cell_id in nearest])
    return nearest, raw / float(np.sum(raw)), distances


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or weight.shape != (matrix.shape[0],):
        raise FastChargeV5PairwiseError("Weighted median dimensions are invalid")
    if (weight < 0).any() or not math.isclose(
        float(weight.sum()), 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise FastChargeV5PairwiseError("Weighted median weights are invalid")
    order = np.argsort(matrix, axis=0, kind="stable")
    sorted_values = np.take_along_axis(matrix, order, axis=0)
    sorted_weights = np.take_along_axis(
        np.broadcast_to(weight[:, None], matrix.shape), order, axis=0
    )
    positions = np.argmax(np.cumsum(sorted_weights, axis=0) >= 0.5, axis=0)
    return sorted_values[positions, np.arange(matrix.shape[1])]


def pairwise_reference_trajectories(
    estimator: RegressorMixin,
    target_prefix: pd.DataFrame,
    reference_cells: Mapping[str, pd.DataFrame],
    prefix_cycle: int,
    score_end_cycle: int,
    core_config: Mapping[str, object],
    *,
    neighbor_count: int = 8,
    reference_resources: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target_descriptor = _descriptor(target_prefix, core_config)
    target_normalization = _normalization_capacity(target_prefix)
    target_retention = _retention(target_prefix, target_normalization)
    target_last = float(target_retention[-1])
    if reference_resources is None:
        reference_resources = _cell_resources(
            reference_cells, prefix_cycle, core_config
        )
    elif set(reference_resources["descriptors"]) != set(reference_cells):
        raise FastChargeV5PairwiseError(
            "Cached reference resources do not match the reference cells"
        )
    nearest, weights, distances = _reference_geometry(
        target_descriptor,
        reference_resources["descriptors"],
        core_config,
        neighbor_count=neighbor_count,
    )
    forecast = np.arange(prefix_cycle + 1, score_end_cycle + 1, dtype=int)
    feature_blocks: list[np.ndarray] = []
    reference_changes: list[np.ndarray] = []
    for reference_id in nearest:
        difference = target_descriptor - reference_resources["descriptors"][
            reference_id
        ]
        feature_blocks.append(
            np.vstack(
            [
                _pair_features(
                    difference,
                    (cycle - prefix_cycle) / (score_end_cycle - prefix_cycle),
                )
                for cycle in forecast
            ]
            )
        )
        retention = reference_resources["retentions"][reference_id]
        reference_changes.append(
            retention[forecast - 1] - retention[prefix_cycle - 1]
        )
    positive = np.vstack(feature_blocks)
    correction = 0.5 * (
        np.asarray(estimator.predict(positive), dtype=float).reshape(-1)
        - np.asarray(estimator.predict(-positive), dtype=float).reshape(-1)
    )
    correction = correction.reshape(len(nearest), len(forecast))
    matrix = target_last + np.vstack(reference_changes) + correction
    audit = {
        "reference_cell_ids": nearest,
        "reference_weights": {
            cell_id: float(value)
            for cell_id, value in zip(nearest, weights, strict=True)
        },
        "reference_distances": {
            cell_id: float(distances[cell_id]) for cell_id in nearest
        },
        "mean_reference_distance": float(
            np.mean([distances[cell_id] for cell_id in nearest])
        ),
        "reference_dispersion_mean_pp": float(np.mean(np.std(matrix, axis=0))),
    }
    return np.clip(matrix, 0.0, 110.0), weights, audit


def aggregate_reference_trajectories(
    trajectories: np.ndarray,
    weights: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    if aggregation not in PAIRWISE_AGGREGATIONS:
        raise FastChargeV5PairwiseError(f"Unknown aggregation: {aggregation}")
    matrix = np.asarray(trajectories, dtype=float)
    normalized = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or normalized.shape != (matrix.shape[0],):
        raise FastChargeV5PairwiseError("Reference aggregation dimensions changed")
    normalized = normalized / float(np.sum(normalized))
    if aggregation == "single_nearest":
        prediction = matrix[0]
    elif aggregation == "weighted_mean":
        prediction = np.sum(normalized[:, None] * matrix, axis=0)
    else:
        prediction = weighted_median(matrix, normalized)
    return np.clip(prediction, 0.0, 110.0)


def predict_pairwise_trajectory(
    estimator: RegressorMixin,
    target_prefix: pd.DataFrame,
    reference_cells: Mapping[str, pd.DataFrame],
    prefix_cycle: int,
    score_end_cycle: int,
    core_config: Mapping[str, object],
    *,
    aggregation: str = "weighted_median",
    neighbor_count: int = 8,
    reference_resources: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    matrix, weights, audit = pairwise_reference_trajectories(
        estimator,
        target_prefix,
        reference_cells,
        prefix_cycle,
        score_end_cycle,
        core_config,
        neighbor_count=neighbor_count,
        reference_resources=reference_resources,
    )
    prediction = aggregate_reference_trajectories(matrix, weights, aggregation)
    return prediction, {**audit, "aggregation": aggregation}


def predict_direct_trajectory(
    estimator: RegressorMixin,
    target_prefix: pd.DataFrame,
    prefix_cycle: int,
    score_end_cycle: int,
    core_config: Mapping[str, object],
) -> np.ndarray:
    descriptor = _descriptor(target_prefix, core_config)
    normalization = _normalization_capacity(target_prefix)
    last = float(_retention(target_prefix, normalization)[-1])
    forecast = np.arange(prefix_cycle + 1, score_end_cycle + 1, dtype=int)
    matrix = np.vstack(
        [
            _direct_features(
                descriptor,
                (cycle - prefix_cycle) / (score_end_cycle - prefix_cycle),
            )
            for cycle in forecast
        ]
    )
    change = np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)
    return np.clip(last + change, 0.0, 110.0)


def deterministic_cell_folds(
    cell_ids: Sequence[str],
    *,
    fold_count: int = 5,
) -> list[tuple[list[str], list[str]]]:
    if fold_count < 2 or len(cell_ids) < fold_count * 2:
        raise FastChargeV5PairwiseError("Invalid physical-cell fold design")
    by_batch: dict[str, list[str]] = {}
    for cell_id in sorted(set(cell_ids)):
        match = re.match(r"MATR_B(\d+)C", cell_id)
        batch = match.group(1) if match else "unknown"
        by_batch.setdefault(batch, []).append(cell_id)
    validation: list[list[str]] = [[] for _ in range(fold_count)]
    for batch in sorted(by_batch):
        for index, cell_id in enumerate(sorted(by_batch[batch])):
            validation[index % fold_count].append(cell_id)
    all_ids = set(cell_ids)
    folds: list[tuple[list[str], list[str]]] = []
    for held_out in validation:
        held_out = sorted(held_out)
        fit = sorted(all_ids - set(held_out))
        if set(fit) & set(held_out) or set(fit) | set(held_out) != all_ids:
            raise FastChargeV5PairwiseError("Physical-cell fold leakage detected")
        folds.append((fit, held_out))
    return folds


def batch_holdout_folds(cell_ids: Sequence[str]) -> list[tuple[list[str], list[str]]]:
    batches: dict[str, list[str]] = {}
    for cell_id in sorted(set(cell_ids)):
        match = re.match(r"MATR_B(\d+)C", cell_id)
        if match is None:
            raise FastChargeV5PairwiseError(
                f"Cannot derive MATR batch from {cell_id}"
            )
        batches.setdefault(match.group(1), []).append(cell_id)
    if len(batches) < 2:
        raise FastChargeV5PairwiseError("Batch challenge requires at least two batches")
    all_ids = set(cell_ids)
    return [
        (sorted(all_ids - set(batches[batch])), sorted(batches[batch]))
        for batch in sorted(batches)
    ]


def assert_pair_fold_firewall(
    fit_cell_ids: Sequence[str],
    validation_cell_ids: Sequence[str],
    pair_audit: Mapping[str, object],
) -> None:
    fit = set(fit_cell_ids)
    validation = set(validation_cell_ids)
    pair_targets = set(pair_audit["target_cell_ids"])
    pair_references = set(pair_audit["reference_cell_ids"])
    if fit & validation:
        raise FastChargeV5PairwiseError("Fold fit and validation cells overlap")
    if pair_targets != fit or pair_references != fit:
        raise FastChargeV5PairwiseError(
            "Pair matrix does not exactly match the fit-cell firewall"
        )
    if validation & (pair_targets | pair_references):
        raise FastChargeV5PairwiseError(
            "Held-out physical cell entered a pairwise training role"
        )


def trajectory_mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise FastChargeV5PairwiseError("Trajectory score dimensions changed")
    return float(np.mean(np.abs(observed - predicted)))


def paired_cell_bootstrap(
    cell_deltas: Mapping[str, float],
    *,
    repetitions: int = 10000,
    confidence: float = 0.95,
    random_state: int = 20260809,
) -> dict[str, float | int]:
    cell_ids = sorted(cell_deltas)
    values = np.asarray([cell_deltas[cell_id] for cell_id in cell_ids], dtype=float)
    if len(values) < 2 or repetitions < 100:
        raise FastChargeV5PairwiseError("Bootstrap support is insufficient")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = np.mean(values[indices], axis=1)
    alpha = 1.0 - confidence
    return {
        "physical_cell_count": len(values),
        "repetitions": repetitions,
        "mean_delta_mae_pp": float(np.mean(values)),
        "lower_delta_mae_pp": float(np.quantile(means, alpha / 2.0)),
        "upper_delta_mae_pp": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def finite_sample_absolute_quantile(
    residuals: np.ndarray,
    *,
    coverage: float = 0.9,
) -> tuple[float, float]:
    values = np.asarray(residuals, dtype=float).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise FastChargeV5PairwiseError("Conformal residuals are invalid")
    if not 0.0 < coverage < 1.0:
        raise FastChargeV5PairwiseError("Conformal coverage must lie in (0, 1)")
    level = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, level, method="higher")), float(level)


def weighted_interval_score(
    observed: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    coverage: float = 0.9,
) -> np.ndarray:
    truth = np.asarray(observed, dtype=float)
    median = np.asarray(center, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if (
        truth.shape != median.shape
        or truth.shape != low.shape
        or truth.shape != high.shape
    ):
        raise FastChargeV5PairwiseError("Interval score dimensions changed")
    if not 0.0 < coverage < 1.0 or (low > high).any():
        raise FastChargeV5PairwiseError("Interval score inputs are invalid")
    alpha = 1.0 - coverage
    below = np.maximum(low - truth, 0.0)
    above = np.maximum(truth - high, 0.0)
    interval_score = (high - low) + (2.0 / alpha) * (below + above)
    # Proper WIS for one central interval and its median forecast.
    return (0.5 * np.abs(truth - median) + (alpha / 2.0) * interval_score) / 1.5
