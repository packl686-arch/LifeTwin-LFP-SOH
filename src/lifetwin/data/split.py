from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import pandas as pd


class SplitError(ValueError):
    """Raised when a requested group split is invalid."""


@dataclass(frozen=True)
class SplitSummary:
    train_rows: int
    validation_rows: int
    test_rows: int
    train_groups: int
    validation_groups: int
    test_groups: int


def _stable_rank(group: tuple[object, ...], seed: int) -> str:
    value = "|".join(str(item) for item in group)
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def stable_group_split(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> pd.Series:
    """Assign complete groups to train/validation/test independent of row order."""
    if not group_columns:
        raise SplitError("At least one group column is required")
    missing = sorted(set(group_columns) - set(frame.columns))
    if missing:
        raise SplitError(f"Missing group columns: {missing}")
    if validation_fraction < 0 or test_fraction < 0:
        raise SplitError("Split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise SplitError("Validation and test fractions must sum to less than one")
    if frame.empty:
        raise SplitError("Cannot split an empty frame")
    if frame[list(group_columns)].isna().any().any():
        raise SplitError("Group columns cannot contain null values")

    groups = [
        tuple(row)
        for row in frame[list(group_columns)].drop_duplicates().itertuples(index=False, name=None)
    ]
    if len(groups) < 3:
        raise SplitError("At least three distinct groups are required")
    groups.sort(key=lambda group: _stable_rank(group, seed))

    n_groups = len(groups)
    n_test = max(1, math.floor(n_groups * test_fraction)) if test_fraction else 0
    n_validation = (
        max(1, math.floor(n_groups * validation_fraction)) if validation_fraction else 0
    )
    if n_test + n_validation >= n_groups:
        raise SplitError("Fractions leave no groups for training")

    assignment: dict[tuple[object, ...], str] = {}
    for group in groups[:n_test]:
        assignment[group] = "test"
    for group in groups[n_test : n_test + n_validation]:
        assignment[group] = "validation"
    for group in groups[n_test + n_validation :]:
        assignment[group] = "train"

    row_groups = frame[list(group_columns)].itertuples(index=False, name=None)
    result = pd.Series(
        (assignment[tuple(group)] for group in row_groups),
        index=frame.index,
        name="split",
        dtype="string",
    )
    assert_group_isolation(frame.assign(split=result), group_columns)
    return result


def explicit_holdout_split(
    frame: pd.DataFrame,
    column: str,
    *,
    test_values: Iterable[object],
    validation_values: Iterable[object] = (),
) -> pd.Series:
    if column not in frame:
        raise SplitError(f"Missing holdout column: {column}")
    test_set = set(test_values)
    validation_set = set(validation_values)
    overlap = test_set & validation_set
    if overlap:
        raise SplitError(f"Values cannot be both validation and test: {sorted(overlap)}")
    if not test_set:
        raise SplitError("At least one test holdout value is required")
    missing_test = test_set - set(frame[column].dropna().unique())
    if missing_test:
        raise SplitError(f"Requested test values are absent: {sorted(missing_test)}")

    result = pd.Series("train", index=frame.index, name="split", dtype="string")
    result.loc[frame[column].isin(validation_set)] = "validation"
    result.loc[frame[column].isin(test_set)] = "test"
    return result


def assert_group_isolation(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    split_column: str = "split",
) -> None:
    missing = sorted(set([*group_columns, split_column]) - set(frame.columns))
    if missing:
        raise SplitError(f"Missing isolation columns: {missing}")
    split_counts = frame.groupby(list(group_columns), dropna=False)[split_column].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        raise SplitError(f"Found {len(leaking)} groups spanning multiple splits")


def summarize_split(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    split_column: str = "split",
) -> SplitSummary:
    assert_group_isolation(frame, group_columns, split_column=split_column)
    row_counts = frame[split_column].value_counts()
    group_counts = (
        frame[[*group_columns, split_column]]
        .drop_duplicates()
        [split_column]
        .value_counts()
    )
    return SplitSummary(
        train_rows=int(row_counts.get("train", 0)),
        validation_rows=int(row_counts.get("validation", 0)),
        test_rows=int(row_counts.get("test", 0)),
        train_groups=int(group_counts.get("train", 0)),
        validation_groups=int(group_counts.get("validation", 0)),
        test_groups=int(group_counts.get("test", 0)),
    )

