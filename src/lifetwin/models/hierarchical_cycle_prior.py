"""Hierarchical condition priors for sparse battery-cycle trajectories.

The models learn degradation-shape priors from reference conditions, then
shrink a target cell's short prefix toward that prior.  They never require the
target suffix and can therefore be reused for private enterprise adaptation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls


REQUIRED_COLUMNS = (
    "cell_id",
    "condition_id",
    "temperature_c",
    "dod_fraction",
    "discharge_c_rate",
    "visit_index",
    "equivalent_full_cycles",
    "capacity_retention_pct",
)
DUAL_CLOCK_REQUIRED_COLUMNS = (*REQUIRED_COLUMNS, "elapsed_days")


class HierarchicalCyclePriorError(ValueError):
    """Raised when a hierarchical cycle-prior contract is violated."""


@dataclass(frozen=True)
class PowerConditionPrior:
    exponent: float
    alpha: float
    condition_equal: bool
    condition_center: tuple[float, float, float]
    condition_scale: tuple[float, float, float]
    regression_coefficients: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PowerConditionPrior:
        return cls(
            exponent=float(value["exponent"]),
            alpha=float(value["alpha"]),
            condition_equal=bool(value["condition_equal"]),
            condition_center=tuple(
                float(item) for item in value["condition_center"]
            ),
            condition_scale=tuple(float(item) for item in value["condition_scale"]),
            regression_coefficients=tuple(
                float(item) for item in value["regression_coefficients"]
            ),
        )


@dataclass(frozen=True)
class BasisKernelPrior:
    basis_exponents: tuple[float, float]
    gamma: float
    condition_center: tuple[float, float, float]
    condition_scale: tuple[float, float, float]
    support_condition_ids: tuple[str, ...]
    support_condition_vectors: tuple[tuple[float, float, float], ...]
    support_coefficients: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BasisKernelPrior:
        return cls(
            basis_exponents=tuple(
                float(item) for item in value["basis_exponents"]
            ),
            gamma=float(value["gamma"]),
            condition_center=tuple(
                float(item) for item in value["condition_center"]
            ),
            condition_scale=tuple(float(item) for item in value["condition_scale"]),
            support_condition_ids=tuple(
                str(item) for item in value["support_condition_ids"]
            ),
            support_condition_vectors=tuple(
                tuple(float(item) for item in row)
                for row in value["support_condition_vectors"]
            ),
            support_coefficients=tuple(
                tuple(float(item) for item in row)
                for row in value["support_coefficients"]
            ),
        )


@dataclass(frozen=True)
class DualClockKernelPrior:
    time_exponent: float
    cycle_exponent: float
    gamma: float
    condition_center: tuple[float, float, float, float]
    condition_scale: tuple[float, float, float, float]
    support_condition_ids: tuple[str, ...]
    support_condition_vectors: tuple[tuple[float, float, float, float], ...]
    support_coefficients: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DualClockKernelPrior:
        return cls(
            time_exponent=float(value["time_exponent"]),
            cycle_exponent=float(value["cycle_exponent"]),
            gamma=float(value["gamma"]),
            condition_center=tuple(
                float(item) for item in value["condition_center"]
            ),
            condition_scale=tuple(
                float(item) for item in value["condition_scale"]
            ),
            support_condition_ids=tuple(
                str(item) for item in value["support_condition_ids"]
            ),
            support_condition_vectors=tuple(
                tuple(float(item) for item in row)
                for row in value["support_condition_vectors"]
            ),
            support_coefficients=tuple(
                tuple(float(item) for item in row)
                for row in value["support_coefficients"]
            ),
        )


def _validated(frame: pd.DataFrame, *, minimum_rows: int = 2) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise HierarchicalCyclePriorError(
            f"Cycle-prior input is missing columns: {missing}"
        )
    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in ("cell_id", "condition_id"):
        if result[column].isna().any():
            raise HierarchicalCyclePriorError(f"Null cycle-prior identity: {column}")
        result[column] = result[column].astype(str)
    numeric_columns = [
        column for column in REQUIRED_COLUMNS if column not in {"cell_id", "condition_id"}
    ]
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise HierarchicalCyclePriorError("Cycle-prior inputs must be finite")
    for column in numeric_columns:
        result[column] = numeric[column].astype(float)
    if len(result) < minimum_rows:
        raise HierarchicalCyclePriorError("Cycle-prior input has insufficient rows")
    if (result["equivalent_full_cycles"] < 0.0).any():
        raise HierarchicalCyclePriorError("Equivalent full cycles cannot be negative")
    if (result["dod_fraction"] <= 0.0).any() or (
        result["discharge_c_rate"] <= 0.0
    ).any():
        raise HierarchicalCyclePriorError("Condition values must be positive")
    return result


def _validated_dual_clock(
    frame: pd.DataFrame,
    *,
    minimum_rows: int = 2,
) -> pd.DataFrame:
    data = _validated(frame, minimum_rows=minimum_rows)
    if "elapsed_days" not in frame.columns:
        raise HierarchicalCyclePriorError("Dual-clock input is missing elapsed_days")
    elapsed = pd.to_numeric(frame["elapsed_days"], errors="coerce")
    if elapsed.isna().any() or not np.isfinite(elapsed.to_numpy(dtype=float)).all():
        raise HierarchicalCyclePriorError("Elapsed days must be finite")
    if (elapsed < 0.0).any():
        raise HierarchicalCyclePriorError("Elapsed days cannot be negative")
    data["elapsed_days"] = elapsed.to_numpy(dtype=float)
    return data


def condition_vector(frame: pd.DataFrame) -> np.ndarray:
    validated = _validated(frame)
    values = validated.loc[
        :, ["temperature_c", "dod_fraction", "discharge_c_rate"]
    ].drop_duplicates()
    if len(values) != 1:
        raise HierarchicalCyclePriorError("Condition changes within one trajectory")
    return values.iloc[0].to_numpy(dtype=float)


def _power_coordinate(exposure_efc: np.ndarray, exponent: float) -> np.ndarray:
    if not 0.0 < exponent <= 2.0:
        raise HierarchicalCyclePriorError("Power exponent is outside (0, 2]")
    exposure = np.asarray(exposure_efc, dtype=float)
    if not np.isfinite(exposure).all() or (exposure < 0.0).any():
        raise HierarchicalCyclePriorError("Forecast exposure must be finite and non-negative")
    return np.power(exposure / 1000.0, exponent)


def power_fade_rate(frame: pd.DataFrame, exponent: float) -> float:
    validated = _validated(frame)
    coordinate = _power_coordinate(
        validated["equivalent_full_cycles"].to_numpy(dtype=float), exponent
    )
    fade = 100.0 - validated["capacity_retention_pct"].to_numpy(dtype=float)
    denominator = float(np.dot(coordinate, coordinate))
    if denominator <= 0.0:
        raise HierarchicalCyclePriorError("Power-rate fit lacks positive exposure")
    return max(0.0, float(np.dot(coordinate, fade) / denominator))


def fit_power_condition_prior(
    references: pd.DataFrame,
    *,
    exponent: float,
    alpha: float,
    condition_equal: bool = False,
) -> PowerConditionPrior:
    data = _validated(references)
    if data["cell_id"].nunique() < 3 or data["condition_id"].nunique() < 2:
        raise HierarchicalCyclePriorError("Power prior needs multiple cells and conditions")
    if alpha <= 0.0:
        raise HierarchicalCyclePriorError("Power-prior ridge alpha must be positive")
    rows: list[dict[str, object]] = []
    for (condition_id, cell_id), cell in data.groupby(
        ["condition_id", "cell_id"], sort=True
    ):
        vector = condition_vector(cell)
        rows.append(
            {
                "condition_id": str(condition_id),
                "cell_id": str(cell_id),
                "temperature_c": vector[0],
                "dod_fraction": vector[1],
                "discharge_c_rate": vector[2],
                "fade_rate": power_fade_rate(cell, exponent),
            }
        )
    table = pd.DataFrame(rows)
    if condition_equal:
        table = (
            table.groupby("condition_id", sort=True)
            .agg(
                temperature_c=("temperature_c", "first"),
                dod_fraction=("dod_fraction", "first"),
                discharge_c_rate=("discharge_c_rate", "first"),
                fade_rate=("fade_rate", "mean"),
            )
            .reset_index()
        )
    matrix = table.loc[
        :, ["temperature_c", "dod_fraction", "discharge_c_rate"]
    ].to_numpy(dtype=float)
    target = table["fade_rate"].to_numpy(dtype=float)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (matrix - center) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return PowerConditionPrior(
        exponent=float(exponent),
        alpha=float(alpha),
        condition_equal=bool(condition_equal),
        condition_center=tuple(float(value) for value in center),
        condition_scale=tuple(float(value) for value in scale),
        regression_coefficients=tuple(float(value) for value in coefficients),
    )


def power_prior_rate(prefix: pd.DataFrame, model: PowerConditionPrior) -> float:
    vector = condition_vector(prefix)
    center = np.asarray(model.condition_center, dtype=float)
    scale = np.asarray(model.condition_scale, dtype=float)
    design = np.concatenate([[1.0], (vector - center) / scale])
    return max(
        0.0,
        float(design @ np.asarray(model.regression_coefficients, dtype=float)),
    )


def predict_power_condition_prior(
    prefix: pd.DataFrame,
    forecast_efc: Sequence[float] | np.ndarray,
    model: PowerConditionPrior,
    *,
    prefix_rate_weight: float,
    anchor_weight: float = 1.0,
) -> np.ndarray:
    data = _validated(prefix)
    if not 0.0 <= prefix_rate_weight <= 1.0:
        raise HierarchicalCyclePriorError("Prefix-rate weight must be in [0, 1]")
    if not 0.0 <= anchor_weight <= 1.0:
        raise HierarchicalCyclePriorError("Anchor weight must be in [0, 1]")
    ordered = data.sort_values("visit_index", kind="stable")
    prior_rate = power_prior_rate(ordered, model)
    observed_rate = power_fade_rate(ordered, model.exponent)
    fade_rate = (
        (1.0 - prefix_rate_weight) * prior_rate
        + prefix_rate_weight * observed_rate
    )
    forecast_coordinate = _power_coordinate(
        np.asarray(forecast_efc, dtype=float), model.exponent
    )
    last = ordered.iloc[-1]
    last_coordinate = float(
        _power_coordinate(
            np.asarray([float(last["equivalent_full_cycles"])]), model.exponent
        )[0]
    )
    latent_last = 100.0 - fade_rate * last_coordinate
    residual = float(last["capacity_retention_pct"]) - latent_last
    return 100.0 - fade_rate * forecast_coordinate + anchor_weight * residual


def _basis_matrix(
    exposure_efc: Sequence[float] | np.ndarray,
    exponents: tuple[float, float],
) -> np.ndarray:
    first, second = (float(value) for value in exponents)
    if not 0.0 < first < second <= 2.0:
        raise HierarchicalCyclePriorError(
            "Basis exponents must satisfy 0 < first < second <= 2"
        )
    exposure = np.asarray(exposure_efc, dtype=float)
    if not np.isfinite(exposure).all() or (exposure < 0.0).any():
        raise HierarchicalCyclePriorError("Basis exposure must be finite and non-negative")
    scaled = exposure / 1000.0
    return np.column_stack([np.power(scaled, first), np.power(scaled, second)])


def basis_fade_coefficients(
    frame: pd.DataFrame,
    exponents: tuple[float, float],
) -> np.ndarray:
    data = _validated(frame)
    matrix = _basis_matrix(
        data["equivalent_full_cycles"].to_numpy(dtype=float), exponents
    )
    fade = 100.0 - data["capacity_retention_pct"].to_numpy(dtype=float)
    coefficients, _ = nnls(matrix, fade)
    return coefficients


def fit_basis_kernel_prior(
    references: pd.DataFrame,
    *,
    basis_exponents: tuple[float, float],
    gamma: float,
) -> BasisKernelPrior:
    data = _validated(references)
    if data["condition_id"].nunique() < 2:
        raise HierarchicalCyclePriorError("Basis kernel needs multiple conditions")
    if gamma <= 0.0:
        raise HierarchicalCyclePriorError("Basis-kernel gamma must be positive")
    condition_ids: list[str] = []
    vectors: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    for condition_id, condition in data.groupby("condition_id", sort=True):
        condition_ids.append(str(condition_id))
        vectors.append(condition_vector(condition))
        coefficients.append(
            np.mean(
                [
                    basis_fade_coefficients(cell, basis_exponents)
                    for _, cell in condition.groupby("cell_id", sort=True)
                ],
                axis=0,
            )
        )
    matrix = np.vstack(vectors)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (matrix - center) / scale
    return BasisKernelPrior(
        basis_exponents=tuple(float(value) for value in basis_exponents),
        gamma=float(gamma),
        condition_center=tuple(float(value) for value in center),
        condition_scale=tuple(float(value) for value in scale),
        support_condition_ids=tuple(condition_ids),
        support_condition_vectors=tuple(
            tuple(float(value) for value in row) for row in standardized
        ),
        support_coefficients=tuple(
            tuple(float(value) for value in row) for row in coefficients
        ),
    )


def basis_prior_coefficients(
    prefix: pd.DataFrame,
    model: BasisKernelPrior,
) -> tuple[np.ndarray, np.ndarray]:
    vector = condition_vector(prefix)
    center = np.asarray(model.condition_center, dtype=float)
    scale = np.asarray(model.condition_scale, dtype=float)
    target = (vector - center) / scale
    support = np.asarray(model.support_condition_vectors, dtype=float)
    squared_distance = np.sum(np.square(support - target), axis=1)
    log_weights = -model.gamma * squared_distance
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise HierarchicalCyclePriorError("Target condition has no kernel support")
    weights /= total
    coefficients = weights @ np.asarray(model.support_coefficients, dtype=float)
    return np.maximum(0.0, coefficients), np.sqrt(squared_distance)


def posterior_basis_coefficients(
    prefix: pd.DataFrame,
    model: BasisKernelPrior,
    *,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    if shrinkage <= 0.0:
        raise HierarchicalCyclePriorError("Basis shrinkage must be positive")
    data = _validated(prefix)
    prior, distances = basis_prior_coefficients(data, model)
    matrix = _basis_matrix(
        data["equivalent_full_cycles"].to_numpy(dtype=float),
        model.basis_exponents,
    )
    fade = 100.0 - data["capacity_retention_pct"].to_numpy(dtype=float)
    augmented_matrix = np.vstack(
        [matrix, math.sqrt(shrinkage) * np.eye(len(prior), dtype=float)]
    )
    augmented_target = np.concatenate(
        [fade, math.sqrt(shrinkage) * prior]
    )
    posterior, _ = nnls(augmented_matrix, augmented_target)
    return posterior, distances


def predict_basis_kernel_prior(
    prefix: pd.DataFrame,
    forecast_efc: Sequence[float] | np.ndarray,
    model: BasisKernelPrior,
    *,
    shrinkage: float,
    anchor_weight: float,
) -> np.ndarray:
    if not 0.0 <= anchor_weight <= 1.0:
        raise HierarchicalCyclePriorError("Anchor weight must be in [0, 1]")
    data = _validated(prefix).sort_values("visit_index", kind="stable")
    coefficients, _ = posterior_basis_coefficients(
        data, model, shrinkage=shrinkage
    )
    forecast_matrix = _basis_matrix(forecast_efc, model.basis_exponents)
    last = data.iloc[-1]
    last_matrix = _basis_matrix(
        [float(last["equivalent_full_cycles"])], model.basis_exponents
    )
    latent_last = float((100.0 - last_matrix @ coefficients)[0])
    residual = float(last["capacity_retention_pct"]) - latent_last
    return 100.0 - forecast_matrix @ coefficients + anchor_weight * residual


def _dual_clock_basis(
    elapsed_days: Sequence[float] | np.ndarray,
    exposure_efc: Sequence[float] | np.ndarray,
    *,
    time_exponent: float,
    cycle_exponent: float,
) -> np.ndarray:
    if not 0.0 < time_exponent <= 2.0 or not 0.0 < cycle_exponent <= 2.0:
        raise HierarchicalCyclePriorError(
            "Dual-clock exponents must lie in (0, 2]"
        )
    elapsed = np.asarray(elapsed_days, dtype=float)
    exposure = np.asarray(exposure_efc, dtype=float)
    if elapsed.shape != exposure.shape:
        raise HierarchicalCyclePriorError("Dual-clock coordinates must align")
    if (
        not np.isfinite(elapsed).all()
        or not np.isfinite(exposure).all()
        or (elapsed < 0.0).any()
        or (exposure < 0.0).any()
    ):
        raise HierarchicalCyclePriorError(
            "Dual-clock coordinates must be finite and non-negative"
        )
    return np.column_stack(
        [
            np.power(elapsed / 365.0, time_exponent),
            np.power(exposure / 1000.0, cycle_exponent),
        ]
    )


def prefix_duty_rate_efc_per_day(frame: pd.DataFrame) -> float:
    data = _validated_dual_clock(frame).sort_values("visit_index", kind="stable")
    last = data.iloc[-1]
    elapsed = float(last["elapsed_days"])
    exposure = float(last["equivalent_full_cycles"])
    if elapsed <= 1e-9 or exposure <= 0.0:
        raise HierarchicalCyclePriorError(
            "Dual-clock prefix needs positive elapsed time and exposure"
        )
    return max(exposure / elapsed, 1e-4)


def dual_clock_condition_vector(frame: pd.DataFrame) -> np.ndarray:
    data = _validated_dual_clock(frame)
    conditions = data.loc[
        :, ["temperature_c", "dod_fraction", "discharge_c_rate"]
    ].drop_duplicates()
    if len(conditions) != 1:
        raise HierarchicalCyclePriorError("Condition changes within one trajectory")
    log_duty_rates = [
        math.log(prefix_duty_rate_efc_per_day(cell))
        for _, cell in data.groupby("cell_id", sort=True)
    ]
    condition = conditions.iloc[0].to_numpy(dtype=float)
    return np.concatenate([condition, [float(np.mean(log_duty_rates))]])


def dual_clock_fade_coefficients(
    frame: pd.DataFrame,
    *,
    time_exponent: float,
    cycle_exponent: float,
) -> np.ndarray:
    data = _validated_dual_clock(frame)
    matrix = _dual_clock_basis(
        data["elapsed_days"].to_numpy(dtype=float),
        data["equivalent_full_cycles"].to_numpy(dtype=float),
        time_exponent=time_exponent,
        cycle_exponent=cycle_exponent,
    )
    fade = 100.0 - data["capacity_retention_pct"].to_numpy(dtype=float)
    coefficients, _ = nnls(matrix, fade)
    return coefficients


def fit_dual_clock_kernel_prior(
    references: pd.DataFrame,
    *,
    time_exponent: float,
    cycle_exponent: float,
    gamma: float,
) -> DualClockKernelPrior:
    data = _validated_dual_clock(references)
    if data["condition_id"].nunique() < 2:
        raise HierarchicalCyclePriorError(
            "Dual-clock kernel needs multiple conditions"
        )
    if gamma <= 0.0:
        raise HierarchicalCyclePriorError(
            "Dual-clock kernel gamma must be positive"
        )
    condition_ids: list[str] = []
    vectors: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    for condition_id, condition in data.groupby("condition_id", sort=True):
        condition_ids.append(str(condition_id))
        vectors.append(dual_clock_condition_vector(condition))
        coefficients.append(
            np.mean(
                [
                    dual_clock_fade_coefficients(
                        cell,
                        time_exponent=time_exponent,
                        cycle_exponent=cycle_exponent,
                    )
                    for _, cell in condition.groupby("cell_id", sort=True)
                ],
                axis=0,
            )
        )
    matrix = np.vstack(vectors)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (matrix - center) / scale
    return DualClockKernelPrior(
        time_exponent=float(time_exponent),
        cycle_exponent=float(cycle_exponent),
        gamma=float(gamma),
        condition_center=tuple(float(value) for value in center),
        condition_scale=tuple(float(value) for value in scale),
        support_condition_ids=tuple(condition_ids),
        support_condition_vectors=tuple(
            tuple(float(value) for value in row) for row in standardized
        ),
        support_coefficients=tuple(
            tuple(float(value) for value in row) for row in coefficients
        ),
    )


def dual_clock_prior_coefficients(
    prefix: pd.DataFrame,
    model: DualClockKernelPrior,
) -> tuple[np.ndarray, np.ndarray]:
    vector = dual_clock_condition_vector(prefix)
    return dual_clock_condition_prior_coefficients(vector, model)


def dual_clock_condition_prior_coefficients(
    condition: Sequence[float] | np.ndarray,
    model: DualClockKernelPrior,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate dual-clock coefficients for one declared condition vector."""
    vector = np.asarray(condition, dtype=float)
    if vector.shape != (4,) or not np.isfinite(vector).all():
        raise HierarchicalCyclePriorError(
            "Dual-clock condition vector must contain four finite values"
        )
    if vector[1] <= 0.0 or vector[2] <= 0.0:
        raise HierarchicalCyclePriorError(
            "Dual-clock DOD and discharge rate must be positive"
        )
    center = np.asarray(model.condition_center, dtype=float)
    scale = np.asarray(model.condition_scale, dtype=float)
    target = (vector - center) / scale
    support = np.asarray(model.support_condition_vectors, dtype=float)
    squared_distance = np.sum(np.square(support - target), axis=1)
    log_weights = -model.gamma * squared_distance
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise HierarchicalCyclePriorError(
            "Target condition has no dual-clock kernel support"
        )
    weights /= total
    coefficients = weights @ np.asarray(model.support_coefficients, dtype=float)
    return np.maximum(0.0, coefficients), np.sqrt(squared_distance)


def dual_clock_basis_coordinates(
    elapsed_days: Sequence[float] | np.ndarray,
    exposure_efc: Sequence[float] | np.ndarray,
    model: DualClockKernelPrior,
) -> np.ndarray:
    """Return the two monotone basis coordinates used by a fitted dual clock."""
    return _dual_clock_basis(
        elapsed_days,
        exposure_efc,
        time_exponent=model.time_exponent,
        cycle_exponent=model.cycle_exponent,
    )


def posterior_dual_clock_coefficients(
    prefix: pd.DataFrame,
    model: DualClockKernelPrior,
    *,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    if shrinkage <= 0.0:
        raise HierarchicalCyclePriorError(
            "Dual-clock shrinkage must be positive"
        )
    data = _validated_dual_clock(prefix)
    prior, distances = dual_clock_prior_coefficients(data, model)
    matrix = _dual_clock_basis(
        data["elapsed_days"].to_numpy(dtype=float),
        data["equivalent_full_cycles"].to_numpy(dtype=float),
        time_exponent=model.time_exponent,
        cycle_exponent=model.cycle_exponent,
    )
    fade = 100.0 - data["capacity_retention_pct"].to_numpy(dtype=float)
    augmented_matrix = np.vstack(
        [matrix, math.sqrt(shrinkage) * np.eye(len(prior), dtype=float)]
    )
    augmented_target = np.concatenate(
        [fade, math.sqrt(shrinkage) * prior]
    )
    posterior, _ = nnls(augmented_matrix, augmented_target)
    return posterior, distances


def infer_constant_duty_elapsed_days(
    prefix: pd.DataFrame,
    forecast_efc: Sequence[float] | np.ndarray,
) -> np.ndarray:
    data = _validated_dual_clock(prefix).sort_values("visit_index", kind="stable")
    forecast = np.asarray(forecast_efc, dtype=float)
    if not np.isfinite(forecast).all() or (forecast < 0.0).any():
        raise HierarchicalCyclePriorError(
            "Dual-clock forecast exposure must be finite and non-negative"
        )
    duty_rate = prefix_duty_rate_efc_per_day(data)
    return forecast / duty_rate


def predict_dual_clock_kernel_prior(
    prefix: pd.DataFrame,
    forecast_efc: Sequence[float] | np.ndarray,
    model: DualClockKernelPrior,
    *,
    shrinkage: float,
    anchor_weight: float,
    forecast_elapsed_days: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    if not 0.0 <= anchor_weight <= 1.0:
        raise HierarchicalCyclePriorError(
            "Dual-clock anchor weight must be in [0, 1]"
        )
    data = _validated_dual_clock(prefix).sort_values("visit_index", kind="stable")
    forecast = np.asarray(forecast_efc, dtype=float)
    elapsed = (
        infer_constant_duty_elapsed_days(data, forecast)
        if forecast_elapsed_days is None
        else np.asarray(forecast_elapsed_days, dtype=float)
    )
    coefficients, _ = posterior_dual_clock_coefficients(
        data, model, shrinkage=shrinkage
    )
    forecast_matrix = _dual_clock_basis(
        elapsed,
        forecast,
        time_exponent=model.time_exponent,
        cycle_exponent=model.cycle_exponent,
    )
    last = data.iloc[-1]
    last_matrix = _dual_clock_basis(
        [float(last["elapsed_days"])],
        [float(last["equivalent_full_cycles"])],
        time_exponent=model.time_exponent,
        cycle_exponent=model.cycle_exponent,
    )
    latent_last = float((100.0 - last_matrix @ coefficients)[0])
    residual = float(last["capacity_retention_pct"]) - latent_last
    return 100.0 - forecast_matrix @ coefficients + anchor_weight * residual


def prefix_residual_rms(
    prefix: pd.DataFrame,
    predicted_at_prefix: Sequence[float] | np.ndarray,
) -> float:
    data = _validated(prefix)
    predicted = np.asarray(predicted_at_prefix, dtype=float)
    observed = data.sort_values("visit_index", kind="stable")[
        "capacity_retention_pct"
    ].to_numpy(dtype=float)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise HierarchicalCyclePriorError("Prefix residual inputs are inconsistent")
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


__all__ = [
    "BasisKernelPrior",
    "DUAL_CLOCK_REQUIRED_COLUMNS",
    "DualClockKernelPrior",
    "HierarchicalCyclePriorError",
    "PowerConditionPrior",
    "REQUIRED_COLUMNS",
    "basis_fade_coefficients",
    "basis_prior_coefficients",
    "condition_vector",
    "dual_clock_condition_vector",
    "dual_clock_condition_prior_coefficients",
    "dual_clock_basis_coordinates",
    "dual_clock_fade_coefficients",
    "dual_clock_prior_coefficients",
    "fit_basis_kernel_prior",
    "fit_dual_clock_kernel_prior",
    "fit_power_condition_prior",
    "infer_constant_duty_elapsed_days",
    "posterior_dual_clock_coefficients",
    "posterior_basis_coefficients",
    "predict_dual_clock_kernel_prior",
    "prefix_duty_rate_efc_per_day",
    "power_fade_rate",
    "power_prior_rate",
    "predict_basis_kernel_prior",
    "predict_power_condition_prior",
    "prefix_residual_rms",
]
