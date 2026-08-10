"""Result-blind member-fit numeric contracts for the V0.19 development line.

The inherited fitter deliberately records failed variants instead of dropping
them.  Their fit metrics and eight raw forecasts are structural missing values;
successful variants require finite values in those same positions.  This module
binds that mask to the frozen variant registry, fit status, credibility state,
forecast grid and content-hash identity without filling or clipping a value.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEY_SET,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import FORECAST_DAYS
from lifetwin.experiments.calendar_long_horizon_v018_numeric_contract import (
    validate_decision_bundle_numeric_contract,
    validate_feature_bundle_numeric_contract,
    validate_pipeline_numeric_contract,
    validate_prediction_bundle_numeric_contract,
    validate_risk_bundle_numeric_contract,
)


class V024MemberFitNumericContractError(ValueError):
    """Raised when candidate V0.19 member-fit numeric semantics are violated."""


DIAGNOSTIC_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "model_id",
    "variant_id",
    "parameters_json",
    "fit_status",
    "credible_variant",
    "prefix_rmse_pp",
    "prefix_max_abs_residual_pp",
    "parameter_boundary_hit_fraction",
    "canonical_prefix_content_sha256",
)
FORECAST_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "model_id",
    "variant_id",
    "forecast_day",
    "raw_forecast_retention_pct",
    "canonical_prefix_content_sha256",
)
DIAGNOSTIC_KEY = ("partition", "cluster_id", "model_id", "variant_id")
FORECAST_KEY = (*DIAGNOSTIC_KEY, "forecast_day")
FIT_METRIC_COLUMNS = (
    "prefix_rmse_pp",
    "prefix_max_abs_residual_pp",
    "parameter_boundary_hit_fraction",
)
_FORECAST_BOUNDS_PCT = (40.0, 105.0)
_MAXIMUM_PREFIX_RMSE_PP = 1.0
_MAXIMUM_PREFIX_RESIDUAL_PP = 1.5


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    *,
    context: str,
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise V024MemberFitNumericContractError(
            f"{context} must be a nonempty dataframe"
        )
    if tuple(frame.columns) != expected:
        raise V024MemberFitNumericContractError(
            f"{context} columns differ from the frozen schema"
        )


def _require_nonempty_identity(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).eq("").any():
            raise V024MemberFitNumericContractError(
                f"{context} contains an empty {column}"
            )


def _strict_bool_array(values: pd.Series, *, context: str) -> np.ndarray:
    if any(type(value) not in {bool, np.bool_} for value in values.tolist()):
        raise V024MemberFitNumericContractError(
            f"{context} must contain strict booleans"
        )
    return values.to_numpy(dtype=bool)


def _numeric_array(values: pd.Series, *, context: str) -> np.ndarray:
    if not is_numeric_dtype(values.dtype):
        raise V024MemberFitNumericContractError(
            f"{context} must already have a numeric dtype"
        )
    result = values.to_numpy(dtype=float)
    if np.isinf(result).any():
        raise V024MemberFitNumericContractError(f"{context} contains infinity")
    return result


def _validate_parameters_json(
    diagnostics: pd.DataFrame,
    *,
    succeeded: np.ndarray,
) -> None:
    payloads = diagnostics["parameters_json"].tolist()
    for index, (payload, is_success) in enumerate(
        zip(payloads, succeeded, strict=True)
    ):
        if not isinstance(payload, str):
            raise V024MemberFitNumericContractError(
                "parameters_json must contain canonical strings"
            )
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise V024MemberFitNumericContractError(
                "parameters_json contains invalid JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise V024MemberFitNumericContractError(
                "parameters_json must encode an object"
            )
        try:
            canonical = json.dumps(
                parsed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise V024MemberFitNumericContractError(
                "parameters_json contains a nonfinite or unsupported value"
            ) from exc
        if canonical != payload:
            raise V024MemberFitNumericContractError("parameters_json is not canonical")
        if not is_success and payload != "{}":
            raise V024MemberFitNumericContractError(
                "A failed variant retained fitted parameters"
            )
        for name, value in parsed.items():
            if (
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise V024MemberFitNumericContractError(
                    f"parameters_json row {index} contains a nonfinite parameter"
                )


def _validate_variant_registry(diagnostics: pd.DataFrame) -> None:
    if diagnostics.duplicated(list(DIAGNOSTIC_KEY)).any():
        raise V024MemberFitNumericContractError(
            "member_fit_diagnostics.csv contains duplicate variant keys"
        )
    observed = pd.MultiIndex.from_frame(
        diagnostics.loc[:, ["model_id", "variant_id"]].astype(str)
    )
    allowed = pd.MultiIndex.from_tuples(sorted(FROZEN_VARIANT_KEY_SET))
    if not observed.isin(allowed).all():
        raise V024MemberFitNumericContractError(
            "member_fit_diagnostics.csv contains an undeclared variant key"
        )
    group_sizes = diagnostics.groupby(
        ["partition", "cluster_id"], sort=False, observed=True
    ).size()
    if group_sizes.empty or not group_sizes.eq(len(FROZEN_VARIANT_KEY_SET)).all():
        raise V024MemberFitNumericContractError(
            "Every cluster must contain the exact frozen 86-variant registry"
        )


def validate_member_fit_numeric_contract(
    member_fit_diagnostics: pd.DataFrame,
    member_forecast_bundle: pd.DataFrame,
) -> None:
    """Validate exact fit-status masks for diagnostics and raw forecasts.

    There is no literal objective, rank or issuance column in the frozen
    member-fit schemas.  The committed fit objectives are the RMSE and maximum
    residual metrics checked here.  Rank and issuance invariants remain bound
    by :func:`validate_decision_bundle_numeric_contract`, re-exported above.
    """

    diagnostics = member_fit_diagnostics
    forecasts = member_forecast_bundle
    _require_exact_columns(
        diagnostics,
        DIAGNOSTIC_COLUMNS,
        context="member_fit_diagnostics.csv",
    )
    _require_exact_columns(
        forecasts,
        FORECAST_COLUMNS,
        context="member_forecast_bundle.csv",
    )
    _require_nonempty_identity(
        diagnostics,
        (
            "protocol_id",
            "partition",
            "cluster_id",
            "model_id",
            "variant_id",
            "fit_status",
            "canonical_prefix_content_sha256",
        ),
        context="member_fit_diagnostics.csv",
    )
    _require_nonempty_identity(
        forecasts,
        (
            "protocol_id",
            "partition",
            "cluster_id",
            "model_id",
            "variant_id",
            "canonical_prefix_content_sha256",
        ),
        context="member_forecast_bundle.csv",
    )
    protocol_ids = set(diagnostics["protocol_id"].astype(str))
    if (
        len(protocol_ids) != 1
        or set(forecasts["protocol_id"].astype(str)) != protocol_ids
    ):
        raise V024MemberFitNumericContractError("Member-fit protocol identities differ")

    _validate_variant_registry(diagnostics)
    statuses = diagnostics["fit_status"].astype(str).to_numpy()
    if not np.isin(statuses, ("succeeded", "failed")).all():
        raise V024MemberFitNumericContractError(
            "fit_status must be succeeded or failed"
        )
    succeeded = statuses == "succeeded"
    credible = _strict_bool_array(
        diagnostics["credible_variant"], context="credible_variant"
    )
    if np.any(credible & ~succeeded):
        raise V024MemberFitNumericContractError(
            "A failed variant was declared credible"
        )
    _validate_parameters_json(diagnostics, succeeded=succeeded)

    metric_arrays: dict[str, np.ndarray] = {}
    for column in FIT_METRIC_COLUMNS:
        values = _numeric_array(diagnostics[column], context=column)
        if not np.isfinite(values).any():
            raise V024MemberFitNumericContractError(f"{column} is entirely degenerate")
        if not np.isfinite(values[succeeded]).all():
            raise V024MemberFitNumericContractError(
                f"A succeeded variant has nonfinite {column}"
            )
        if not np.isnan(values[~succeeded]).all():
            raise V024MemberFitNumericContractError(
                f"A failed variant must contain structural NaN in {column}"
            )
        metric_arrays[column] = values
    if np.any(metric_arrays["prefix_rmse_pp"][succeeded] < 0.0):
        raise V024MemberFitNumericContractError("prefix_rmse_pp cannot be negative")
    if np.any(metric_arrays["prefix_max_abs_residual_pp"][succeeded] < 0.0):
        raise V024MemberFitNumericContractError(
            "prefix_max_abs_residual_pp cannot be negative"
        )
    boundary = metric_arrays["parameter_boundary_hit_fraction"][succeeded]
    if np.any((boundary < 0.0) | (boundary > 1.0)):
        raise V024MemberFitNumericContractError(
            "parameter_boundary_hit_fraction is outside [0, 1]"
        )

    if forecasts.duplicated(list(FORECAST_KEY)).any():
        raise V024MemberFitNumericContractError(
            "member_forecast_bundle.csv contains duplicate forecast keys"
        )
    forecast_days = _numeric_array(forecasts["forecast_day"], context="forecast_day")
    if (
        not np.isfinite(forecast_days).all()
        or not np.isin(forecast_days, np.asarray(FORECAST_DAYS, dtype=float)).all()
    ):
        raise V024MemberFitNumericContractError(
            "member_forecast_bundle.csv contains an invalid forecast day"
        )

    state = diagnostics.loc[
        :,
        [
            *DIAGNOSTIC_KEY,
            "fit_status",
            "credible_variant",
            "prefix_rmse_pp",
            "prefix_max_abs_residual_pp",
            "canonical_prefix_content_sha256",
        ],
    ].rename(
        columns={
            "canonical_prefix_content_sha256": "diagnostic_content_sha256",
            "credible_variant": "diagnostic_credible_variant",
        }
    )
    merged = forecasts.merge(
        state,
        on=list(DIAGNOSTIC_KEY),
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    if len(merged) != len(forecasts) or not merged["_merge"].eq("both").all():
        raise V024MemberFitNumericContractError(
            "Forecast rows do not align with diagnostic variant keys"
        )
    forecast_group_sizes = merged.groupby(
        list(DIAGNOSTIC_KEY), sort=False, observed=True
    ).size()
    if (
        len(forecast_group_sizes) != len(diagnostics)
        or not forecast_group_sizes.eq(len(FORECAST_DAYS)).all()
    ):
        raise V024MemberFitNumericContractError(
            "Every diagnostic variant must have the exact eight-row forecast grid"
        )
    if (
        not merged["canonical_prefix_content_sha256"]
        .astype(str)
        .eq(merged["diagnostic_content_sha256"].astype(str))
        .all()
    ):
        raise V024MemberFitNumericContractError(
            "Diagnostic and forecast content hashes differ"
        )

    raw_forecasts = _numeric_array(
        merged["raw_forecast_retention_pct"],
        context="raw_forecast_retention_pct",
    )
    forecast_succeeded = merged["fit_status"].eq("succeeded").to_numpy()
    if not np.isfinite(raw_forecasts[forecast_succeeded]).all():
        raise V024MemberFitNumericContractError(
            "A succeeded variant has a nonfinite raw forecast"
        )
    if not np.isnan(raw_forecasts[~forecast_succeeded]).all():
        raise V024MemberFitNumericContractError(
            "A failed variant must contain structural NaN raw forecasts"
        )

    lower, upper = _FORECAST_BOUNDS_PCT
    credible_by_forecast = (
        forecast_succeeded
        & (merged["prefix_rmse_pp"].to_numpy(float) <= _MAXIMUM_PREFIX_RMSE_PP)
        & (
            merged["prefix_max_abs_residual_pp"].to_numpy(float)
            <= _MAXIMUM_PREFIX_RESIDUAL_PP
        )
        & np.isfinite(raw_forecasts)
        & (raw_forecasts >= lower)
        & (raw_forecasts <= upper)
    )
    recomputed_credible = (
        pd.Series(credible_by_forecast)
        .groupby([merged[column] for column in DIAGNOSTIC_KEY], sort=False)
        .all()
    )
    declared_credible = diagnostics.set_index(list(DIAGNOSTIC_KEY))[
        "credible_variant"
    ].astype(bool)
    recomputed_credible = recomputed_credible.reindex(declared_credible.index)
    if recomputed_credible.isna().any() or not np.array_equal(
        recomputed_credible.to_numpy(bool), declared_credible.to_numpy(bool)
    ):
        raise V024MemberFitNumericContractError(
            "credible_variant differs from the frozen fit objective and forecast rule"
        )


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "DIAGNOSTIC_KEY",
    "FIT_METRIC_COLUMNS",
    "FORECAST_COLUMNS",
    "FORECAST_KEY",
    "V024MemberFitNumericContractError",
    "validate_decision_bundle_numeric_contract",
    "validate_feature_bundle_numeric_contract",
    "validate_member_fit_numeric_contract",
    "validate_pipeline_numeric_contract",
    "validate_prediction_bundle_numeric_contract",
    "validate_risk_bundle_numeric_contract",
]
