"""Label-free Delta Q(V) features for external cycling campaigns.

This module deliberately accepts only a small, pre-outcome metadata contract.  Life
labels and censoring information belong in a later scoring step, never in curve
feature extraction.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from lifetwin.features.curves import _interpolate_discharge_curve


EARLY_CYCLE = 10
LATE_CYCLE = 100
VOLTAGE_MIN_V = 2.0
VOLTAGE_MAX_V = 3.5
VOLTAGE_POINTS = 1000
MINIMUM_SAMPLES_PER_CURVE = 50

REQUIRED_CURVE_COLUMNS = frozenset(
    {
        "test_id",
        "cycle_number",
        "voltage_V",
        "coulomb_count_Ah",
        "step_type",
        "temperature_C",
    }
)
REQUIRED_METADATA_COLUMNS = frozenset({"cell_id", "test_id", "protocol_id"})

# Keep this contract intentionally narrow. Descriptive campaign inventories can be
# audited separately without making their columns candidate model inputs.
ALLOWED_METADATA_COLUMNS = (
    "cell_id",
    "test_id",
    "protocol_id",
    "batch_id",
    "source_cell_id",
    "source_protocol_id",
)

FEATURE_COLUMNS = (
    "early_cycle",
    "late_cycle",
    "voltage_min_v",
    "voltage_max_v",
    "voltage_points",
    "delta_q_min_ah",
    "delta_q_max_ah",
    "delta_q_mean_ah",
    "delta_q_variance_ah2",
    "log10_delta_q_variance",
    "delta_q_skewness",
    "delta_q_kurtosis",
    "delta_q_abs_area_ah_v",
    "q_early_mean_ah",
    "q_late_mean_ah",
    "temperature_mean_c",
)

_PROTECTED_COLUMN_TOKENS = frozenset(
    {
        "capacity_fade",
        "capacity_loss",
        "capacity_retention",
        "censor",
        "cycle_life",
        "eol",
        "event",
        "failure",
        "is_censored",
        "life",
        "lifetime",
        "num_cycles",
        "observed_time",
        "remaining_useful_life",
        "resistance",
        "rul",
        "soh",
        "source_num_cycles",
        "target",
    }
)


def _normalise_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _is_protected_column(column: object) -> bool:
    normalised = _normalise_column_name(column)
    tokens = set(normalised.split("_"))
    if normalised in _PROTECTED_COLUMN_TOKENS:
        return True
    if any(token in tokens for token in {"censor", "eol", "event", "failure"}):
        return True
    if any(marker in normalised for marker in ("cycle_life", "num_cycles")):
        return True
    if normalised == "r" or re.fullmatch(r"r(?:_?[0-9]+|_.+)", normalised):
        return True
    return "resistance" in normalised


def _reject_protected_columns(frame: pd.DataFrame, *, frame_name: str) -> None:
    protected = sorted(str(column) for column in frame if _is_protected_column(column))
    if protected:
        raise ValueError(
            f"{frame_name} contains protected outcome columns: {protected}"
        )


def _validate_dataset_id(dataset_id: str) -> None:
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if dataset_id != dataset_id.strip():
        raise ValueError("dataset_id must not contain leading or trailing whitespace")


def _validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    _reject_protected_columns(metadata, frame_name="metadata")
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    unsupported = sorted(set(metadata.columns) - set(ALLOWED_METADATA_COLUMNS))
    if unsupported:
        raise ValueError(
            "Unsupported metadata columns in label-free contract: "
            f"{unsupported}"
        )
    if metadata.empty:
        raise ValueError("metadata must contain at least one cell")

    result = metadata.copy()
    for column in ("cell_id", "test_id", "protocol_id"):
        values = result[column]
        strings = values.map(lambda value: isinstance(value, str))
        if values.isna().any() or not strings.all():
            raise ValueError(f"metadata {column} values must be non-null strings")
        if values.map(lambda value: not value or value != value.strip()).any():
            raise ValueError(
                f"metadata {column} values must be non-empty and whitespace-trimmed"
            )
    if result["cell_id"].duplicated().any():
        raise ValueError("metadata cell_id values must be unique")
    if result["test_id"].duplicated().any():
        raise ValueError("metadata test_id values must be unique")
    return result


def _valid_curve_sample_count(frame: pd.DataFrame) -> tuple[int, int]:
    numeric = frame[["voltage_V", "coulomb_count_Ah"]].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    numeric = numeric.loc[
        (numeric["voltage_V"] >= VOLTAGE_MIN_V - 0.02)
        & (numeric["voltage_V"] <= VOLTAGE_MAX_V + 0.10)
    ]
    return len(numeric), int(numeric["voltage_V"].nunique())


def _validate_and_select_curves(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    _reject_protected_columns(curves, frame_name="curves")
    missing = sorted(REQUIRED_CURVE_COLUMNS - set(curves.columns))
    if missing:
        raise ValueError(f"Missing curve columns: {missing}")

    cycle_number = pd.to_numeric(curves["cycle_number"], errors="coerce")
    if cycle_number.isna().any() or not np.isfinite(cycle_number).all():
        raise ValueError("curve cycle_number values must be finite integers")
    if not np.equal(cycle_number, np.floor(cycle_number)).all():
        raise ValueError("curve cycle_number values must be finite integers")

    normalised = curves.copy()
    normalised["cycle_number"] = cycle_number.astype(np.int64)
    requested = normalised.loc[
        (normalised["step_type"] == "discharge")
        & normalised["cycle_number"].isin((EARLY_CYCLE, LATE_CYCLE))
    ].copy()
    if requested.empty:
        raise ValueError("No cycle 10/100 discharge curves are present")
    if requested["test_id"].isna().any():
        raise ValueError("requested curve test_id values must be non-null")

    expected_tests = set(metadata["test_id"])
    observed_tests = set(requested["test_id"])
    missing_tests = sorted(expected_tests - observed_tests)
    unexpected_tests = sorted(observed_tests - expected_tests)
    if missing_tests or unexpected_tests:
        raise ValueError(
            "Curve/metadata test_id mismatch: "
            f"missing={missing_tests}, unexpected={unexpected_tests}"
        )

    observed_pairs = set(
        requested[["test_id", "cycle_number"]].itertuples(index=False, name=None)
    )
    expected_pairs = {
        (test_id, cycle)
        for test_id in expected_tests
        for cycle in (EARLY_CYCLE, LATE_CYCLE)
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise ValueError(f"Missing required test/cycle curve pairs: {missing_pairs}")
    return requested


def extract_external_delta_q_features(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    dataset_id: str,
    minimum_samples_per_curve: int = MINIMUM_SAMPLES_PER_CURVE,
) -> pd.DataFrame:
    """Extract fixed-grid cycle-10-to-100 Delta Q features without outcomes.

    ``metadata`` is an explicit one-to-one test-to-cell map. The returned frame is
    sorted by identity and contains no lifetime, event, censoring, SOH, resistance,
    or other outcome columns.
    """
    _validate_dataset_id(dataset_id)
    if minimum_samples_per_curve < MINIMUM_SAMPLES_PER_CURVE:
        raise ValueError(
            "minimum_samples_per_curve cannot be below the audited minimum of "
            f"{MINIMUM_SAMPLES_PER_CURVE}"
        )
    safe_metadata = _validate_metadata(metadata)
    requested = _validate_and_select_curves(curves, safe_metadata)
    voltage_grid = np.linspace(VOLTAGE_MIN_V, VOLTAGE_MAX_V, VOLTAGE_POINTS)

    rows: list[dict[str, object]] = []
    for test_id, test in requested.groupby("test_id", sort=True):
        interpolated: dict[int, np.ndarray] = {}
        for cycle in (EARLY_CYCLE, LATE_CYCLE):
            curve = test.loc[test["cycle_number"] == cycle]
            sample_count, unique_voltage_count = _valid_curve_sample_count(curve)
            if sample_count < minimum_samples_per_curve:
                raise ValueError(
                    f"{test_id} cycle {cycle} has fewer than "
                    f"{minimum_samples_per_curve} valid samples"
                )
            if unique_voltage_count < minimum_samples_per_curve:
                raise ValueError(
                    f"{test_id} cycle {cycle} has fewer than "
                    f"{minimum_samples_per_curve} distinct voltage points"
                )
            interpolated[cycle] = _interpolate_discharge_curve(curve, voltage_grid)

        temperature = pd.to_numeric(test["temperature_C"], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(temperature).all():
            raise ValueError(f"{test_id} contains non-finite temperature values")
        temperature_mean = math.fsum(
            float(value) for value in np.sort(temperature)
        ) / len(temperature)

        early = interpolated[EARLY_CYCLE]
        late = interpolated[LATE_CYCLE]
        delta = late - early
        variance = float(np.var(delta, ddof=0))
        rows.append(
            {
                "test_id": test_id,
                "early_cycle": EARLY_CYCLE,
                "late_cycle": LATE_CYCLE,
                "voltage_min_v": VOLTAGE_MIN_V,
                "voltage_max_v": VOLTAGE_MAX_V,
                "voltage_points": VOLTAGE_POINTS,
                "delta_q_min_ah": float(np.min(delta)),
                "delta_q_max_ah": float(np.max(delta)),
                "delta_q_mean_ah": float(np.mean(delta)),
                "delta_q_variance_ah2": variance,
                "log10_delta_q_variance": math.log10(max(variance, 1e-16)),
                "delta_q_skewness": float(skew(delta, bias=False)),
                "delta_q_kurtosis": float(kurtosis(delta, bias=False)),
                "delta_q_abs_area_ah_v": float(
                    np.trapezoid(np.abs(delta), voltage_grid)
                ),
                "q_early_mean_ah": float(np.mean(early)),
                "q_late_mean_ah": float(np.mean(late)),
                "temperature_mean_c": temperature_mean,
            }
        )

    features = pd.DataFrame(rows)
    numeric_features = features.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("External Delta Q extraction produced non-finite features")

    metadata_columns = [
        column for column in ALLOWED_METADATA_COLUMNS if column in safe_metadata
    ]
    result = features.merge(
        safe_metadata.loc[:, metadata_columns],
        on="test_id",
        how="left",
        validate="one_to_one",
    )
    if len(result) != len(safe_metadata) or result["cell_id"].isna().any():
        raise ValueError("Curve features could not be matched to every metadata cell")
    result.insert(0, "dataset_id", dataset_id)
    output_columns = ["dataset_id", *metadata_columns, *FEATURE_COLUMNS]
    return result.loc[:, output_columns].sort_values(
        ["cell_id", "test_id"], kind="stable", ignore_index=True
    )


__all__ = [
    "ALLOWED_METADATA_COLUMNS",
    "EARLY_CYCLE",
    "FEATURE_COLUMNS",
    "LATE_CYCLE",
    "MINIMUM_SAMPLES_PER_CURVE",
    "VOLTAGE_MAX_V",
    "VOLTAGE_MIN_V",
    "VOLTAGE_POINTS",
    "extract_external_delta_q_features",
]
