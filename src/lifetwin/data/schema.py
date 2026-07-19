from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_CYCLE_COLUMNS = (
    "dataset_id",
    "cell_id",
    "batch_id",
    "protocol_id",
    "cycle_index",
    "discharge_capacity_ah",
)

REQUIRED_LABEL_COLUMNS = (
    "dataset_id",
    "cell_id",
    "cycle_life",
)


class DataValidationError(ValueError):
    """Raised when canonical battery data violate a hard contract rule."""


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    cell_count: int
    warnings: tuple[str, ...] = ()


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(frame.columns))


def validate_cycle_summary(frame: pd.DataFrame) -> ValidationReport:
    missing = _missing_columns(frame, REQUIRED_CYCLE_COLUMNS)
    if missing:
        raise DataValidationError(f"Missing required cycle columns: {missing}")
    if frame.empty:
        raise DataValidationError("Cycle summary is empty")

    key_columns = ["dataset_id", "cell_id", "cycle_index"]
    if frame[key_columns].isna().any().any():
        raise DataValidationError("Cycle identity columns cannot contain null values")
    if frame.duplicated(key_columns).any():
        duplicates = int(frame.duplicated(key_columns, keep=False).sum())
        raise DataValidationError(
            f"Found {duplicates} rows with duplicate dataset/cell/cycle identity"
        )

    cycle_index = pd.to_numeric(frame["cycle_index"], errors="coerce")
    capacity = pd.to_numeric(frame["discharge_capacity_ah"], errors="coerce")
    if cycle_index.isna().any() or not np.allclose(cycle_index, np.round(cycle_index)):
        raise DataValidationError("cycle_index must contain finite integer values")
    if (cycle_index < 1).any():
        raise DataValidationError("cycle_index must be one-based and positive")
    if capacity.isna().any() or (~np.isfinite(capacity)).any():
        raise DataValidationError("discharge_capacity_ah must contain finite values")
    if (capacity <= 0).any():
        raise DataValidationError("discharge_capacity_ah must be positive")

    warnings: list[str] = []
    if frame[["batch_id", "protocol_id"]].isna().any().any():
        warnings.append("batch_id/protocol_id contains null; use explicit 'unknown' values")
    if "temperature_avg_c" in frame:
        temperature = pd.to_numeric(frame["temperature_avg_c"], errors="coerce")
        if ((temperature < -60) | (temperature > 150)).any():
            warnings.append("temperature_avg_c contains values outside [-60, 150] degC")

    return ValidationReport(
        row_count=len(frame),
        cell_count=frame[["dataset_id", "cell_id"]].drop_duplicates().shape[0],
        warnings=tuple(warnings),
    )


def validate_cell_labels(frame: pd.DataFrame) -> ValidationReport:
    missing = _missing_columns(frame, REQUIRED_LABEL_COLUMNS)
    if missing:
        raise DataValidationError(f"Missing required label columns: {missing}")
    if frame.empty:
        raise DataValidationError("Cell label table is empty")

    key_columns = ["dataset_id", "cell_id"]
    if frame[key_columns].isna().any().any():
        raise DataValidationError("Label identity columns cannot contain null values")
    if frame.duplicated(key_columns).any():
        raise DataValidationError("Cell label table must contain one row per cell")

    cycle_life = pd.to_numeric(frame["cycle_life"], errors="coerce")
    if cycle_life.isna().any() or (~np.isfinite(cycle_life)).any():
        raise DataValidationError("cycle_life must contain finite values")
    if (cycle_life <= 0).any():
        raise DataValidationError("cycle_life must be positive")

    warnings: list[str] = []
    if "is_censored" in frame:
        censored = frame["is_censored"]
        if censored.isna().any():
            raise DataValidationError("is_censored cannot contain null values")
        valid_boolean = censored.map(lambda value: isinstance(value, (bool, np.bool_)))
        if not valid_boolean.all():
            raise DataValidationError("is_censored must contain boolean values")
        if censored.astype(bool).all():
            warnings.append("all labels are right-censored; EOL events are unavailable")
    else:
        warnings.append("is_censored is absent; labels are assumed to be observed EOL events")

    if "eol_threshold" in frame:
        threshold = pd.to_numeric(frame["eol_threshold"], errors="coerce")
        if threshold.isna().any() or (~np.isfinite(threshold)).any():
            raise DataValidationError("eol_threshold must contain finite values")
        if ((threshold <= 0) | (threshold >= 1)).any():
            raise DataValidationError("eol_threshold must lie strictly between zero and one")

    return ValidationReport(
        row_count=len(frame), cell_count=len(frame), warnings=tuple(warnings)
    )
