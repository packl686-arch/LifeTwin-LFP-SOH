from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


REQUIRED_CURVE_COLUMNS = {
    "test_id",
    "cycle_number",
    "voltage_V",
    "coulomb_count_Ah",
    "step_type",
}


def _interpolate_discharge_curve(
    frame: pd.DataFrame,
    voltage_grid: np.ndarray,
) -> np.ndarray:
    numeric = frame[["voltage_V", "coulomb_count_Ah"]].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    numeric = numeric.loc[
        (numeric["voltage_V"] >= voltage_grid.min() - 0.02)
        & (numeric["voltage_V"] <= voltage_grid.max() + 0.10)
    ]
    if len(numeric) < 50:
        raise ValueError("Discharge curve has fewer than 50 valid samples")

    by_voltage = (
        numeric.groupby("voltage_V", as_index=False)["coulomb_count_Ah"]
        .median()
        .sort_values("voltage_V")
    )
    voltage = by_voltage["voltage_V"].to_numpy(dtype=float)
    capacity = by_voltage["coulomb_count_Ah"].to_numpy(dtype=float)
    if voltage[0] > voltage_grid[0] or voltage[-1] < voltage_grid[-1]:
        raise ValueError("Discharge curve does not cover the fixed voltage grid")
    return np.interp(voltage_grid, voltage, capacity)


def extract_delta_q_features(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    early_cycle: int = 10,
    late_cycle: int = 100,
    voltage_min_v: float = 2.0,
    voltage_max_v: float = 3.5,
    voltage_points: int = 1000,
) -> pd.DataFrame:
    """Reconstruct Severson-style Delta Q(V) features from two discharge cycles."""
    missing = sorted(REQUIRED_CURVE_COLUMNS - set(curves.columns))
    if missing:
        raise ValueError(f"Missing curve columns: {missing}")
    if early_cycle < 1 or late_cycle <= early_cycle:
        raise ValueError("Expected 1 <= early_cycle < late_cycle")
    if voltage_points < 50 or voltage_max_v <= voltage_min_v:
        raise ValueError("Invalid voltage grid")

    requested = curves.loc[
        (curves["step_type"] == "discharge")
        & (curves["cycle_number"].isin([early_cycle, late_cycle]))
    ].copy()
    if requested.empty:
        raise ValueError("No requested discharge curves are present")

    voltage_grid = np.linspace(voltage_min_v, voltage_max_v, voltage_points)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for test_id, test in requested.groupby("test_id", sort=True):
        available = set(test["cycle_number"].astype(int).unique())
        if not {early_cycle, late_cycle}.issubset(available):
            failures.append(f"{test_id}: missing requested cycle")
            continue
        try:
            early = _interpolate_discharge_curve(
                test.loc[test["cycle_number"] == early_cycle], voltage_grid
            )
            late = _interpolate_discharge_curve(
                test.loc[test["cycle_number"] == late_cycle], voltage_grid
            )
        except ValueError as exc:
            failures.append(f"{test_id}: {exc}")
            continue

        delta = late - early
        variance = float(np.var(delta, ddof=0))
        temperature = (
            pd.to_numeric(test["temperature_C"], errors="coerce")
            if "temperature_C" in test
            else pd.Series(np.nan, index=test.index)
        )
        cell_id = str(test_id).removesuffix("_CYCLING")
        rows.append(
            {
                "dataset_id": "MATR_SEVERSON_2019",
                "cell_id": cell_id,
                "early_cycle": early_cycle,
                "late_cycle": late_cycle,
                "voltage_min_v": voltage_min_v,
                "voltage_max_v": voltage_max_v,
                "voltage_points": voltage_points,
                "delta_q_min_ah": float(np.min(delta)),
                "delta_q_max_ah": float(np.max(delta)),
                "delta_q_mean_ah": float(np.mean(delta)),
                "delta_q_variance_ah2": variance,
                "log10_delta_q_variance": math.log10(max(variance, 1e-16)),
                "delta_q_skewness": float(skew(delta, bias=False)),
                "delta_q_kurtosis": float(kurtosis(delta, bias=False)),
                "delta_q_abs_area_ah_v": float(np.trapezoid(np.abs(delta), voltage_grid)),
                "q_early_mean_ah": float(np.mean(early)),
                "q_late_mean_ah": float(np.mean(late)),
                "temperature_mean_c": float(temperature.mean())
                if temperature.notna().any()
                else math.nan,
            }
        )

    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError(f"No usable curve pairs; failures: {failures[:5]}")
    required_metadata = {
        "dataset_id",
        "cell_id",
        "batch_id",
        "protocol_id",
        "cycle_life",
        "split_cell",
        "split_protocol",
    }
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    if missing_metadata:
        raise ValueError(f"Missing metadata columns: {missing_metadata}")
    result = features.merge(
        metadata,
        on=["dataset_id", "cell_id"],
        how="left",
        validate="one_to_one",
    )
    if result["cycle_life"].isna().any():
        raise ValueError("Curve features could not be matched to all life labels")
    result.attrs["failed_curve_pairs"] = failures
    return result
