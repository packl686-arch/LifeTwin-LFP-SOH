"""Schema-aware numeric contracts for the V2.3 development line.

V2.2 added a blanket ``isfinite`` check after the inherited pipeline.  That
check rejected the structural missing values that the frozen risk schema uses
for non-primary scores and abstained clusters.  This module keeps those missing
values explicit and validates their exact masks instead of filling or clipping
them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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
_PREDICTION_NUMERIC_COLUMNS = (
    "forecast_day",
    "center_forecast_pct",
    "sqrt_time_forecast_pct",
    "bounded_power_forecast_pct",
    "base_interval_lower_pct",
    "base_interval_upper_pct",
    "calibrated_interval_lower_pct",
    "calibrated_interval_upper_pct",
)
_DECISION_REQUIRED_COLUMNS = frozenset(
    {
        *_KEY_COLUMNS,
        "arm",
        "raw_risk_score",
        "hard_eligible",
        "issuance_rank",
        "issued",
    }
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


def validate_feature_bundle_numeric_contract(feature_bundle: pd.DataFrame) -> None:
    """Bind the all-features-finite flag to the actual feature values."""

    if not isinstance(feature_bundle, pd.DataFrame) or feature_bundle.empty:
        raise V023NumericContractError("Feature bundle must be a nonempty dataframe")
    _require_columns(
        feature_bundle,
        _FEATURE_REQUIRED_COLUMNS,
        context="Feature bundle",
    )
    if feature_bundle.duplicated(list(_KEY_COLUMNS)).any():
        raise V023NumericContractError("Feature bundle contains duplicate cluster keys")
    numeric = feature_bundle.select_dtypes(include=[np.number]).drop(
        columns=["hard_eligible", "all_features_finite"],
        errors="ignore",
    )
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise V023NumericContractError("Feature bundle contains infinity")
    observed_finite = np.isfinite(values).all(axis=1)
    declared_finite = feature_bundle["all_features_finite"].to_numpy(dtype=bool)
    if not np.array_equal(observed_finite, declared_finite):
        raise V023NumericContractError(
            "Feature all_features_finite flag does not match its numeric values"
        )
    hard_eligible = feature_bundle["hard_eligible"].to_numpy(dtype=bool)
    if np.any(hard_eligible & ~declared_finite):
        raise V023NumericContractError("A hard-eligible feature row is nonfinite")


def validate_prediction_bundle_numeric_contract(
    prediction_bundle: pd.DataFrame,
    feature_bundle: pd.DataFrame,
) -> None:
    """Require complete finite trajectories for every label-free cluster."""

    if not isinstance(prediction_bundle, pd.DataFrame) or prediction_bundle.empty:
        raise V023NumericContractError("Prediction bundle must be nonempty")
    _require_columns(
        prediction_bundle,
        {*_KEY_COLUMNS, *_PREDICTION_NUMERIC_COLUMNS},
        context="Prediction bundle",
    )
    values = prediction_bundle.loc[:, _PREDICTION_NUMERIC_COLUMNS].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise V023NumericContractError("Prediction bundle must be fully finite")
    keys = prediction_bundle.loc[:, list(_KEY_COLUMNS)].drop_duplicates()
    feature_keys = feature_bundle.loc[:, list(_KEY_COLUMNS)].drop_duplicates()
    if set(map(tuple, keys.to_numpy())) != set(map(tuple, feature_keys.to_numpy())):
        raise V023NumericContractError(
            "Prediction and feature cluster key sets differ"
        )
    for lower, upper in (
        ("base_interval_lower_pct", "base_interval_upper_pct"),
        ("calibrated_interval_lower_pct", "calibrated_interval_upper_pct"),
    ):
        if np.any(
            prediction_bundle[lower].to_numpy(float)
            > prediction_bundle[upper].to_numpy(float)
        ):
            raise V023NumericContractError(f"{lower} exceeds {upper}")


def validate_decision_bundle_numeric_contract(
    decision_bundle: pd.DataFrame,
    feature_bundle: pd.DataFrame,
    risk_bundle: pd.DataFrame,
    *,
    primary_issue_counts: Mapping[str, int],
) -> None:
    """Validate risk alignment, eligibility ranks and issuance exactly."""

    if not isinstance(decision_bundle, pd.DataFrame) or decision_bundle.empty:
        raise V023NumericContractError("Decision bundle must be nonempty")
    _require_columns(
        decision_bundle,
        _DECISION_REQUIRED_COLUMNS,
        context="Decision bundle",
    )
    feature_state = feature_bundle.set_index(list(_KEY_COLUMNS))[
        ["hard_eligible", "all_features_finite"]
    ]
    decision_keys = pd.MultiIndex.from_frame(
        decision_bundle.loc[:, list(_KEY_COLUMNS)]
    )
    aligned = feature_state.reindex(decision_keys)
    if aligned.isna().any().any():
        raise V023NumericContractError("Decision rows do not align with feature rows")
    hard_eligible = decision_bundle["hard_eligible"].to_numpy(dtype=bool)
    if not np.array_equal(
        hard_eligible,
        aligned["hard_eligible"].to_numpy(dtype=bool),
    ):
        raise V023NumericContractError("Decision hard_eligible state differs")

    primary_risk = risk_bundle.loc[
        risk_bundle["score_id"].isin(_PRIMARY_SCORE_IDS),
        [*_KEY_COLUMNS, "score_id", "raw_risk_score"],
    ].rename(columns={"score_id": "arm", "raw_risk_score": "risk_raw"})
    merged = decision_bundle.merge(
        primary_risk,
        on=[*_KEY_COLUMNS, "arm"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(decision_bundle) or merged["risk_raw"].isna().ne(
        merged["raw_risk_score"].isna()
    ).any():
        raise V023NumericContractError("Decision and risk raw-score masks differ")
    finite = ~merged["risk_raw"].isna()
    if not np.array_equal(
        merged.loc[finite, "risk_raw"].to_numpy(float),
        merged.loc[finite, "raw_risk_score"].to_numpy(float),
    ):
        raise V023NumericContractError("Decision and risk raw scores differ")

    ranks = _numeric(decision_bundle["issuance_rank"], context="issuance_rank")
    issued = decision_bundle["issued"].to_numpy(dtype=bool)
    if np.any(issued & ~hard_eligible):
        raise V023NumericContractError("An ineligible decision was issued")
    for (partition, arm), group in decision_bundle.assign(
        _rank=ranks,
    ).groupby(["partition", "arm"], sort=True):
        group_eligible = group["hard_eligible"].to_numpy(dtype=bool)
        group_ranks = group["_rank"].to_numpy(float)
        issue_count = primary_issue_counts.get(str(partition))
        if issue_count is None:
            if not np.isnan(group_ranks).all() or group["issued"].any():
                raise V023NumericContractError(
                    "A non-issuance partition contains a rank or issuance"
                )
            continue
        if not np.isnan(group_ranks[~group_eligible]).all():
            raise V023NumericContractError("An ineligible decision has a rank")
        eligible_ranks = group_ranks[group_eligible]
        expected = np.arange(1, len(eligible_ranks) + 1, dtype=float)
        if not np.array_equal(np.sort(eligible_ranks), expected):
            raise V023NumericContractError("Eligible issuance ranks are not exact")
        expected_issued = np.isfinite(group_ranks) & (group_ranks <= issue_count)
        if not np.array_equal(group["issued"].to_numpy(dtype=bool), expected_issued):
            raise V023NumericContractError("Issued flags do not match frozen ranks")


def validate_pipeline_numeric_contract(
    *,
    prediction_bundle: pd.DataFrame,
    feature_bundle: pd.DataFrame,
    risk_bundle: pd.DataFrame,
    decision_bundle: pd.DataFrame,
    primary_issue_counts: Mapping[str, int],
) -> None:
    """Validate every label-free numeric output without filling a value."""

    validate_feature_bundle_numeric_contract(feature_bundle)
    validate_prediction_bundle_numeric_contract(prediction_bundle, feature_bundle)
    validate_risk_bundle_numeric_contract(risk_bundle, feature_bundle)
    validate_decision_bundle_numeric_contract(
        decision_bundle,
        feature_bundle,
        risk_bundle,
        primary_issue_counts=primary_issue_counts,
    )


__all__ = [
    "V023NumericContractError",
    "validate_decision_bundle_numeric_contract",
    "validate_feature_bundle_numeric_contract",
    "validate_pipeline_numeric_contract",
    "validate_prediction_bundle_numeric_contract",
    "validate_risk_bundle_numeric_contract",
]
