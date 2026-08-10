"""Schema-aware numeric contracts for the V2.3 development line.

V2.2 added a blanket ``isfinite`` check after the inherited pipeline.  That
check rejected the structural missing values that the frozen risk schema uses
for non-primary scores and abstained clusters.  This module keeps those missing
values explicit and validates their exact masks instead of filling or clipping
them.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class V023NumericContractError(ValueError):
    """Raised when a V2.3 development output violates numeric semantics."""


_KEY_COLUMNS = ("partition", "cluster_id")
_PRIMARY_SCORE_IDS = frozenset({"prefix_only", "visible_stress"})
_RISK_REQUIRED_COLUMNS = frozenset(
    {
        *_KEY_COLUMNS,
        "score_id",
        "raw_risk_score",
        "calibrated_catastrophic_probability",
        "all_features_finite",
        "successful_structure_family_count",
        "fit_failure_count",
        "effective_unique_shape_count",
    }
)
_FEATURE_REQUIRED_COLUMNS = frozenset(
    {*_KEY_COLUMNS, "hard_eligible", "all_features_finite"}
)


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    context: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise V023NumericContractError(f"{context} columns are missing: {missing}")


def _numeric(values: pd.Series, *, context: str) -> np.ndarray:
    converted = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & converted.isna()
    if invalid.any():
        raise V023NumericContractError(f"{context} contains a non-numeric value")
    result = converted.to_numpy(dtype=float)
    if np.isinf(result).any():
        raise V023NumericContractError(f"{context} contains infinity")
    return result


def validate_risk_bundle_numeric_contract(
    risk_bundle: pd.DataFrame,
    feature_bundle: pd.DataFrame,
) -> None:
    """Validate finite values and structural-NA masks for one risk partition.

    Raw scores exist exactly when all model features are finite.  Calibrated
    probabilities exist exactly for hard-eligible primary scores.  Every other
    missing value is rejected, as is every infinity or out-of-range
    probability.
    """

    if not isinstance(risk_bundle, pd.DataFrame) or not isinstance(
        feature_bundle, pd.DataFrame
    ):
        raise V023NumericContractError("Risk numeric inputs must be dataframes")
    _require_columns(risk_bundle, _RISK_REQUIRED_COLUMNS, context="Risk bundle")
    _require_columns(
        feature_bundle,
        _FEATURE_REQUIRED_COLUMNS,
        context="Feature bundle",
    )
    if risk_bundle.empty or feature_bundle.empty:
        raise V023NumericContractError("Risk numeric inputs cannot be empty")
    if feature_bundle.duplicated(list(_KEY_COLUMNS)).any():
        raise V023NumericContractError("Feature bundle contains duplicate cluster keys")

    feature_flags = feature_bundle.set_index(list(_KEY_COLUMNS))[
        ["hard_eligible", "all_features_finite"]
    ]
    risk_keys = pd.MultiIndex.from_frame(risk_bundle.loc[:, list(_KEY_COLUMNS)])
    aligned = feature_flags.reindex(risk_keys)
    if aligned.isna().any().any():
        raise V023NumericContractError("Risk rows do not align with feature rows")

    risk_finite_flags = risk_bundle["all_features_finite"].to_numpy(dtype=bool)
    feature_finite_flags = aligned["all_features_finite"].to_numpy(dtype=bool)
    if not np.array_equal(risk_finite_flags, feature_finite_flags):
        raise V023NumericContractError(
            "Risk and feature all_features_finite masks differ"
        )
    hard_eligible = aligned["hard_eligible"].to_numpy(dtype=bool)
    score_ids = risk_bundle["score_id"].astype(str).to_numpy()
    primary = np.isin(score_ids, tuple(sorted(_PRIMARY_SCORE_IDS)))

    raw_scores = _numeric(risk_bundle["raw_risk_score"], context="raw_risk_score")
    if not np.isfinite(raw_scores[risk_finite_flags]).all() or not np.isnan(
        raw_scores[~risk_finite_flags]
    ).all():
        raise V023NumericContractError(
            "raw_risk_score does not match all_features_finite"
        )

    calibrated = _numeric(
        risk_bundle["calibrated_catastrophic_probability"],
        context="calibrated_catastrophic_probability",
    )
    calibrated_required = primary & hard_eligible
    required_values = calibrated[calibrated_required]
    if (
        not np.isfinite(required_values).all()
        or np.any(required_values < 0.0)
        or np.any(required_values > 1.0)
        or not np.isnan(calibrated[~calibrated_required]).all()
    ):
        raise V023NumericContractError(
            "calibrated probability does not match the primary eligibility mask"
        )

    for column in (
        "successful_structure_family_count",
        "fit_failure_count",
        "effective_unique_shape_count",
    ):
        values = _numeric(risk_bundle[column], context=column)
        if not np.isfinite(values).all():
            raise V023NumericContractError(f"{column} must be finite")


__all__ = ["V023NumericContractError", "validate_risk_bundle_numeric_contract"]
