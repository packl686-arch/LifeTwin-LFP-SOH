from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lifetwin.data.schema import validate_cell_labels, validate_cycle_summary


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return math.nan
    x_valid = x[finite]
    y_valid = y[finite]
    x_centered = x_valid - x_valid.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0:
        return math.nan
    return float(np.dot(x_centered, y_valid - y_valid.mean()) / denominator)


def _value_at_or_before(frame: pd.DataFrame, column: str, cycle: int) -> float:
    subset = frame.loc[frame["cycle_index"] <= cycle, column]
    if subset.empty:
        return math.nan
    value = pd.to_numeric(subset.iloc[-1], errors="coerce")
    return float(value) if pd.notna(value) else math.nan


def _optional_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def extract_early_cycle_features(
    cycles: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    observation_cycle: int = 100,
    minimum_observed_cycles: int = 20,
) -> pd.DataFrame:
    """Build one feature row per cell without reading beyond observation_cycle."""
    if observation_cycle < 2:
        raise ValueError("observation_cycle must be at least 2")
    if minimum_observed_cycles < 2 or minimum_observed_cycles > observation_cycle:
        raise ValueError("minimum_observed_cycles must be in [2, observation_cycle]")

    validate_cycle_summary(cycles)
    validate_cell_labels(labels)
    observed = cycles.loc[cycles["cycle_index"] <= observation_cycle].copy()
    if observed.empty:
        raise ValueError("No cycles fall inside the requested observation window")

    rows: list[dict[str, object]] = []
    for (dataset_id, cell_id), cell in observed.groupby(
        ["dataset_id", "cell_id"], sort=True
    ):
        cell = cell.sort_values("cycle_index")
        if len(cell) < minimum_observed_cycles:
            continue

        cycle_index = cell["cycle_index"].to_numpy(dtype=float)
        capacity = cell["discharge_capacity_ah"].to_numpy(dtype=float)
        capacity_diff = np.diff(capacity)
        diff_variance = float(np.var(capacity_diff, ddof=1)) if len(capacity_diff) > 1 else 0.0
        initial_capacity = float(np.median(capacity[: min(5, len(capacity))]))

        resistance = _optional_numeric(cell, "internal_resistance_ohm")
        temperature = _optional_numeric(cell, "temperature_avg_c")
        temperature_max = _optional_numeric(cell, "temperature_max_c")
        charge_time = _optional_numeric(cell, "charge_time_s")

        rows.append(
            {
                "dataset_id": dataset_id,
                "cell_id": cell_id,
                "batch_id": str(cell["batch_id"].iloc[0]),
                "protocol_id": str(cell["protocol_id"].iloc[0]),
                "observation_cycle": observation_cycle,
                "observed_cycle_count": len(cell),
                "capacity_initial_ah": initial_capacity,
                "capacity_cycle_10_ah": _value_at_or_before(cell, "discharge_capacity_ah", 10),
                "capacity_cycle_50_ah": _value_at_or_before(cell, "discharge_capacity_ah", 50),
                "capacity_cycle_n_ah": float(capacity[-1]),
                "capacity_delta_10_to_n_ah": float(
                    capacity[-1] - _value_at_or_before(cell, "discharge_capacity_ah", 10)
                ),
                "capacity_slope_ah_per_cycle": _slope(cycle_index, capacity),
                "log10_capacity_diff_variance": math.log10(max(diff_variance, 1e-16)),
                "resistance_initial_ohm": float(np.nanmedian(resistance[:5]))
                if np.isfinite(resistance[:5]).any()
                else math.nan,
                "resistance_slope_ohm_per_cycle": _slope(cycle_index, resistance),
                "temperature_avg_c": float(np.nanmean(temperature))
                if np.isfinite(temperature).any()
                else math.nan,
                "temperature_max_c": float(np.nanmax(temperature_max))
                if np.isfinite(temperature_max).any()
                else math.nan,
                "charge_time_avg_s": float(np.nanmean(charge_time))
                if np.isfinite(charge_time).any()
                else math.nan,
            }
        )

    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No cell has enough cycles for feature extraction")
    result = features.merge(
        labels,
        on=["dataset_id", "cell_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(result) != len(features):
        missing = len(features) - len(result)
        raise ValueError(f"Missing labels for {missing} feature rows")
    return result

