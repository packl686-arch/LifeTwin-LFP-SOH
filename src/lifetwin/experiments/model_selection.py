from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
import pandas as pd

from lifetwin.models.probabilistic import LogNormalAFT


@dataclass(frozen=True)
class AFTPenaltyCandidateScore:
    l2_penalty: float
    mean_group_negative_log_likelihood: float
    fold_mean_negative_log_likelihood: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AFTPenaltySelection:
    selected_l2_penalty: float
    group_column: str
    group_count: int
    fold_count: int
    seed: int
    scoring: str
    tie_break: str
    candidates: tuple[AFTPenaltyCandidateScore, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return result


def _stable_group_rank(value: object, seed: int) -> str:
    payload = f"{seed}|{type(value).__name__}|{value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_group_folds(
    frame: pd.DataFrame,
    group_column: str,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Assign complete groups to deterministic, approximately balanced folds."""
    if group_column not in frame:
        raise ValueError(f"Missing group column: {group_column}")
    if frame.empty:
        raise ValueError("Cannot create folds for an empty frame")
    if frame[group_column].isna().any():
        raise ValueError("Model-selection groups cannot contain null values")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")

    groups = list(frame[group_column].drop_duplicates())
    if len(groups) < n_splits:
        raise ValueError(
            f"Expected at least {n_splits} distinct groups, found {len(groups)}"
        )
    groups.sort(key=lambda value: _stable_group_rank(value, seed))
    assignment = {group: index % n_splits for index, group in enumerate(groups)}
    return pd.Series(
        (assignment[group] for group in frame[group_column]),
        index=frame.index,
        name="inner_fold",
        dtype=int,
    )


def _validated_penalties(values: Sequence[float]) -> tuple[float, ...]:
    penalties = tuple(float(value) for value in values)
    if not penalties:
        raise ValueError("At least one L2 penalty candidate is required")
    if any(not math.isfinite(value) or value < 0 for value in penalties):
        raise ValueError("L2 penalty candidates must be finite and non-negative")
    if len(set(penalties)) != len(penalties):
        raise ValueError("L2 penalty candidates must be unique")
    return tuple(sorted(penalties))


def select_lognormal_aft_l2(
    train_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    l2_candidates: Sequence[float],
    group_column: str = "protocol_id",
    label_column: str = "cycle_life",
    censor_column: str = "is_censored",
    n_splits: int = 5,
    seed: int = 42,
) -> AFTPenaltySelection:
    """Select AFT shrinkage using group-isolated inner cross-validation.

    Every protocol contributes equal weight to the selection score, regardless
    of how many cells it contains. The caller must pass only the outer-training
    rows; outer validation remains available for conformal calibration.
    """
    features = list(feature_columns)
    if not features:
        raise ValueError("At least one feature column is required")
    required = {label_column, censor_column, group_column, *features}
    missing = sorted(required - set(train_frame.columns))
    if missing:
        raise ValueError(f"Missing model-selection columns: {missing}")
    if train_frame.empty:
        raise ValueError("Outer-training frame is empty")

    working = train_frame.copy()
    if working[censor_column].isna().any() or working[censor_column].dtype != bool:
        raise ValueError(f"{censor_column} must contain non-null boolean values")
    numeric = working[[*features, label_column]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Features and lifetime labels must contain finite numeric values")
    if (numeric[label_column] <= 0).any():
        raise ValueError("Lifetime labels must be positive")
    working[[*features, label_column]] = numeric

    penalties = _validated_penalties(l2_candidates)
    folds = stable_group_folds(
        working,
        group_column,
        n_splits=n_splits,
        seed=seed,
    )
    matrix_columns = features
    candidate_scores: list[AFTPenaltyCandidateScore] = []
    for penalty in penalties:
        fold_scores: list[float] = []
        group_scores: list[float] = []
        for fold_index in range(n_splits):
            inner_train = working.loc[folds != fold_index]
            inner_validation = working.loc[folds == fold_index]
            model = LogNormalAFT(l2_penalty=penalty).fit(
                inner_train[matrix_columns].to_numpy(dtype=float),
                inner_train[label_column].to_numpy(dtype=float),
                is_censored=inner_train[censor_column].to_numpy(dtype=bool),
            )
            current_fold_scores: list[float] = []
            for _, group in inner_validation.groupby(group_column, sort=False):
                score = model.negative_log_likelihood(
                    group[matrix_columns].to_numpy(dtype=float),
                    group[label_column].to_numpy(dtype=float),
                    is_censored=group[censor_column].to_numpy(dtype=bool),
                )
                current_fold_scores.append(score)
                group_scores.append(score)
            fold_scores.append(float(np.mean(current_fold_scores)))
        candidate_scores.append(
            AFTPenaltyCandidateScore(
                l2_penalty=penalty,
                mean_group_negative_log_likelihood=float(np.mean(group_scores)),
                fold_mean_negative_log_likelihood=tuple(fold_scores),
            )
        )

    selected = min(
        candidate_scores,
        key=lambda candidate: (
            candidate.mean_group_negative_log_likelihood,
            -candidate.l2_penalty,
        ),
    )
    return AFTPenaltySelection(
        selected_l2_penalty=selected.l2_penalty,
        group_column=group_column,
        group_count=int(working[group_column].nunique()),
        fold_count=n_splits,
        seed=seed,
        scoring="mean protocol-balanced validation negative log-likelihood",
        tie_break="prefer stronger L2 penalty when scores are exactly equal",
        candidates=tuple(candidate_scores),
    )
