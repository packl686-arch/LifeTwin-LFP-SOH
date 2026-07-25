"""Prefix-only structure fitting for the frozen V0.15 synthetic protocol.

The public entry point accepts only the two label-free tables needed to fit the
structure library.  In particular, it has no argument for a protocol path,
truth table, family label, future outcome, or matched-pair metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

from lifetwin.experiments import calendar_long_horizon_synthetic as _v1
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    load_artifact_contract,
    predictor_content_hashes,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FORECAST_COORDINATE_COLUMNS,
    FROZEN_PROTOCOL_ID,
    MATCHED_PARTITIONS,
    ORDINARY_PARTITIONS,
    PLACEBO_FIELDS,
    PREFIX_COLUMNS,
    REAL_OPERATING_FIELDS,
    ValidatedV015Protocol,
    load_frozen_protocol_config as load_v2_protocol_config,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_V1_CONFIG_PATH = (
    _REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v1.json"
)
_V2_CONFIG_PATH = (
    _REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2.json"
)

_EXPECTED_V1_BYTE_SHA256 = (
    "503ec964bb2015fe3460433749d1b0d79f89187fc3dcd1c3809f9d4da2ffc319"
)
_EXPECTED_V1_CANONICAL_SHA256 = (
    "6ad1e6dc1caa089ce0b9ee2c4e739a56c44f42f65436294649261a7676d4e320"
)
_EARLY_MODEL_ID = "target_prefix_early_activation_plus_power"
_EARLY_PARAMETER_NAMES = (
    "a",
    "b",
    "activation_amplitude_pp",
    "tau_rise_days",
    "tau_decay_days",
)
_EARLY_PARAMETER_BOUNDS = {
    "a": (0.0, 5.0),
    "b": (0.05, 1.5),
    "activation_amplitude_pp": (0.0, 3.0),
    "tau_rise_days": (3.0, 60.0),
    "tau_decay_days": (30.0, 730.0),
}
_FORECAST_BOUNDS_PCT = (40.0, 105.0)
_MAXIMUM_PREFIX_RMSE_PP = 1.0
_MAXIMUM_PREFIX_MAX_ABS_RESIDUAL_PP = 1.5
_BOUNDARY_RELATIVE_TOLERANCE = 1e-6
_LATE_KNEE_K_GRID = (0.0005, 0.001, 0.002, 0.004)
_LATE_KNEE_T_GRID = (1095.75, 1826.25, 3652.5, 5478.75, 7305.0)
_LATE_KNEE_W_GRID = (30.0, 90.0, 180.0, 365.0)

FROZEN_VARIANT_KEYS = (
    ("target_prefix_persistence", "persistence"),
    ("target_prefix_sqrt_time", "sqrt_time"),
    (
        "target_prefix_bounded_power_law",
        "target_prefix_bounded_power_law",
    ),
    (
        "target_prefix_saturating_plus_slow",
        "target_prefix_saturating_plus_slow",
    ),
    ("target_prefix_dual_power", "target_prefix_dual_power"),
    *(
        (
            "target_prefix_late_knee_prior_grid",
            f"k={k:g}|t={t_knee:g}|w={width:g}",
        )
        for k in _LATE_KNEE_K_GRID
        for t_knee in _LATE_KNEE_T_GRID
        for width in _LATE_KNEE_W_GRID
    ),
    (_EARLY_MODEL_ID, _EARLY_MODEL_ID),
)
FROZEN_VARIANT_KEY_SET = frozenset(FROZEN_VARIANT_KEYS)
_EXPECTED_VARIANT_COUNT = len(FROZEN_VARIANT_KEYS)
if _EXPECTED_VARIANT_COUNT != 86 or len(FROZEN_VARIANT_KEY_SET) != 86:
    raise RuntimeError("The frozen V0.15 structure-library cardinality changed")

_LATE_KNEE_FIXED_BY_VARIANT = {
    variant_id: {
        "k_pp_per_day": k,
        "t_knee_days": t_knee,
        "w_days": width,
    }
    for k in _LATE_KNEE_K_GRID
    for t_knee in _LATE_KNEE_T_GRID
    for width in _LATE_KNEE_W_GRID
    for variant_id in (f"k={k:g}|t={t_knee:g}|w={width:g}",)
}

_DIAGNOSTIC_COLUMNS = (
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
_FORECAST_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "model_id",
    "variant_id",
    "forecast_day",
    "raw_forecast_retention_pct",
    "canonical_prefix_content_sha256",
)


class V015FitError(ValueError):
    """Raised when prefix-only fitting violates the frozen V0.15 contract."""


@dataclass(frozen=True)
class V015FitResult:
    """The two committed structure-library tables."""

    member_fit_diagnostics: pd.DataFrame
    member_forecast_bundle: pd.DataFrame


@dataclass(frozen=True)
class RecomputedVariantCommitment:
    """Formula-derived quantities used to audit one successful fit row."""

    prefix_rmse_pp: float
    prefix_max_abs_residual_pp: float
    forecast_retention_pct: tuple[float, ...]
    parameter_values: tuple[tuple[str, float], ...]
    parameter_bounds: tuple[tuple[str, float, float], ...]
    parameter_boundary_hit_fraction: float


@lru_cache(maxsize=1)
def _legacy_protocol() -> _v1.ValidatedSyntheticProtocol:
    """Load the byte- and canonical-hash-verified immutable V1 protocol."""

    if _v1.FROZEN_CONFIG_BYTE_SHA256 != _EXPECTED_V1_BYTE_SHA256:
        raise V015FitError("Imported V1 byte commitment changed")
    if _v1.FROZEN_CONFIG_CANONICAL_SHA256 != _EXPECTED_V1_CANONICAL_SHA256:
        raise V015FitError("Imported V1 canonical commitment changed")
    try:
        return _v1.load_frozen_protocol_config(_V1_CONFIG_PATH)
    except (OSError, ValueError) as exc:
        raise V015FitError("Frozen V1 config failed double-hash validation") from exc


@lru_cache(maxsize=1)
def _v2_protocol() -> ValidatedV015Protocol:
    try:
        return load_v2_protocol_config(_V2_CONFIG_PATH)
    except (OSError, ValueError) as exc:
        raise V015FitError("Frozen V2 config failed validation") from exc


def validate_frozen_variant_keys(
    keys: Iterable[tuple[str, str]],
    *,
    context: str = "structure library",
) -> None:
    """Require the exact frozen 86-key universe, including variant identities."""

    observed = tuple((str(model_id), str(variant_id)) for model_id, variant_id in keys)
    observed_set = frozenset(observed)
    if len(observed) != len(observed_set):
        raise V015FitError(f"{context} contains duplicate variant keys")
    if observed_set != FROZEN_VARIANT_KEY_SET:
        missing = sorted(FROZEN_VARIANT_KEY_SET - observed_set)
        extra = sorted(observed_set - FROZEN_VARIANT_KEY_SET)
        raise V015FitError(
            f"{context} differs from the frozen exact 86 variant set; "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}"
        )


@lru_cache(maxsize=1)
def _validate_legacy_variant_declaration() -> None:
    candidate = _legacy_protocol().candidate_config()
    specs = {
        str(spec["model_id"]): spec for spec in candidate["structure_member_specs"]
    }
    expected_models = {
        model_id for model_id, _ in FROZEN_VARIANT_KEYS if model_id != _EARLY_MODEL_ID
    }
    if set(specs) != expected_models:
        raise V015FitError("Imported V1 structure-family declaration changed")
    grid = specs["target_prefix_late_knee_prior_grid"]["fixed_grid"]
    if (
        tuple(float(value) for value in grid["k_pp_per_day"]) != _LATE_KNEE_K_GRID
        or tuple(float(value) for value in grid["t_knee_days"]) != _LATE_KNEE_T_GRID
        or tuple(float(value) for value in grid["w_days"]) != _LATE_KNEE_W_GRID
    ):
        raise V015FitError("Imported V1 late-knee fixed grid changed")


def _canonical_parameters(parameters: tuple[tuple[str, float], ...]) -> str:
    payload: dict[str, float] = {}
    for name, value in parameters:
        if name in payload:
            raise V015FitError(f"Duplicate fitted parameter: {name}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise V015FitError("Fitted parameters must be finite")
        payload[name] = numeric
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V015FitError("Fitted parameters are not canonical JSON") from exc


def parse_canonical_parameters_json(value: object) -> dict[str, float]:
    """Parse one committed parameter object and reject ambiguous encodings."""

    if not isinstance(value, str):
        raise V015FitError("parameters_json must be a canonical JSON object string")

    def reject_duplicate_pairs(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, raw in pairs:
            if name in result:
                raise V015FitError(f"parameters_json contains duplicate key: {name}")
            result[name] = raw
        return result

    try:
        raw_payload = json.loads(value, object_pairs_hook=reject_duplicate_pairs)
    except (json.JSONDecodeError, V015FitError) as exc:
        if isinstance(exc, V015FitError):
            raise
        raise V015FitError("parameters_json is invalid JSON") from exc
    if not isinstance(raw_payload, dict):
        raise V015FitError("parameters_json must encode a JSON object")

    parameters: dict[str, float] = {}
    for name, raw in raw_payload.items():
        if not isinstance(name, str) or not name or isinstance(raw, (bool, np.bool_)):
            raise V015FitError("parameters_json contains an invalid field")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as exc:
            raise V015FitError(f"parameters_json/{name} must be numeric") from exc
        if not math.isfinite(numeric):
            raise V015FitError(f"parameters_json/{name} must be finite")
        parameters[name] = numeric

    canonical = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if value != canonical:
        raise V015FitError("parameters_json is not in frozen canonical form")
    return parameters


def _early_prediction(parameters: Mapping[str, float], days: np.ndarray) -> np.ndarray:
    elapsed = np.asarray(days, dtype=float)
    years = elapsed / 365.25
    base = parameters["a"] * np.power(years, parameters["b"])
    activation = (
        parameters["activation_amplitude_pp"]
        * (1.0 - np.exp(-elapsed / parameters["tau_rise_days"]))
        * np.exp(-elapsed / parameters["tau_decay_days"])
    )
    return 100.0 - base + activation


def _sobol_starts(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if lower.shape != upper.shape or lower.shape != (len(_EARLY_PARAMETER_NAMES),):
        raise V015FitError("Early-activation parameter bounds changed")
    sampler = qmc.Sobol(d=len(lower), scramble=False)
    unit = sampler.random_base2(m=4)
    starts = lower + unit * (upper - lower)
    return np.minimum(np.maximum(starts, lower), upper)


def _fit_early_activation(
    prefix_days: np.ndarray,
    observed: np.ndarray,
    forecast_days: np.ndarray,
) -> _v1.CandidateVariant:
    lower = np.asarray(
        [_EARLY_PARAMETER_BOUNDS[name][0] for name in _EARLY_PARAMETER_NAMES],
        dtype=float,
    )
    upper = np.asarray(
        [_EARLY_PARAMETER_BOUNDS[name][1] for name in _EARLY_PARAMETER_NAMES],
        dtype=float,
    )
    starts = _sobol_starts(lower, upper)
    optimizer = _legacy_protocol().candidate_config()["optimizer"]
    successes: list[tuple[float, tuple[float, ...], np.ndarray, np.ndarray]] = []

    def parameter_map(values: np.ndarray) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(_EARLY_PARAMETER_NAMES, values, strict=True)
        }

    def residual(values: np.ndarray) -> np.ndarray:
        return _early_prediction(parameter_map(values), prefix_days) - observed

    for start in starts:
        try:
            fitted = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                method="trf",
                loss="linear",
                max_nfev=int(optimizer["maximum_function_evaluations_per_start"]),
                ftol=float(optimizer["relative_objective_tolerance"]),
                xtol=float(optimizer["relative_parameter_tolerance"]),
                gtol=float(optimizer["relative_parameter_tolerance"]),
            )
            values = np.asarray(fitted.x, dtype=float)
            prefix_residual = residual(values)
            forecast = _early_prediction(parameter_map(values), forecast_days)
            if (
                not fitted.success
                or not np.isfinite(values).all()
                or not np.isfinite(prefix_residual).all()
                or not np.isfinite(forecast).all()
            ):
                continue
            sse = float(prefix_residual @ prefix_residual)
            successes.append(
                (
                    sse,
                    tuple(float(value) for value in values),
                    prefix_residual,
                    forecast,
                )
            )
        except (FloatingPointError, RuntimeError, ValueError):
            continue

    if not successes:
        return _v1.CandidateVariant(
            model_id=_EARLY_MODEL_ID,
            variant_id=_EARLY_MODEL_ID,
            parameters=(),
            prefix_rmse_pp=math.inf,
            prefix_max_absolute_residual_pp=math.inf,
            forecast_retention_pct=(),
            fit_succeeded=False,
            failure_reason="all_declared_sobol_starts_failed",
        )

    successes.sort(key=lambda item: (item[0], item[1]))
    best_sse = successes[0][0]
    tied = [item for item in successes if item[0] <= best_sse + 1e-12]
    _, values, prefix_residual, forecast = min(tied, key=lambda item: item[1])
    return _v1.CandidateVariant(
        model_id=_EARLY_MODEL_ID,
        variant_id=_EARLY_MODEL_ID,
        parameters=tuple(zip(_EARLY_PARAMETER_NAMES, values, strict=True)),
        prefix_rmse_pp=float(np.sqrt(np.mean(np.square(prefix_residual)))),
        prefix_max_absolute_residual_pp=float(np.max(np.abs(prefix_residual))),
        forecast_retention_pct=tuple(float(value) for value in forecast),
        fit_succeeded=True,
    )


def _candidate_parameter_bounds(
    model_id: str,
) -> tuple[tuple[str, float, float], ...]:
    if model_id == "target_prefix_persistence":
        return ()
    if model_id == _EARLY_MODEL_ID:
        return tuple(
            (name, *_EARLY_PARAMETER_BOUNDS[name]) for name in _EARLY_PARAMETER_NAMES
        )

    candidate = _legacy_protocol().candidate_config()
    specs = {
        str(spec["model_id"]): spec for spec in candidate["structure_member_specs"]
    }
    try:
        spec = specs[model_id]
    except KeyError as exc:
        raise V015FitError(f"Undeclared candidate model: {model_id}") from exc
    bounds_key = (
        "fitted_parameter_bounds"
        if model_id == "target_prefix_late_knee_prior_grid"
        else "parameter_bounds"
    )
    raw_bounds = spec[bounds_key]
    parameter_order = _v1._CANDIDATE_PARAMETERS[model_id]
    return tuple(
        (name, float(raw_bounds[name][0]), float(raw_bounds[name][1]))
        for name in parameter_order
    )


def frozen_parameter_metadata(
    model_id: str,
    variant_id: str,
    parameters: Mapping[str, float],
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float, float], ...],
    float,
]:
    """Validate one key's parameter contract without fitting or optimizing."""

    _validate_legacy_variant_declaration()
    key = (str(model_id), str(variant_id))
    if key not in FROZEN_VARIANT_KEY_SET:
        raise V015FitError("Variant key is not part of the frozen exact 86 set")
    numeric = {str(name): float(value) for name, value in parameters.items()}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise V015FitError("Fitted parameters must be finite")

    if model_id == "target_prefix_persistence":
        if set(numeric) != {"last_retention_pct"}:
            raise V015FitError("Persistence parameters differ from the freeze")
        return (), (), 0.0

    fitted_bounds = _candidate_parameter_bounds(model_id)
    expected_names = {name for name, _, _ in fitted_bounds}
    fixed: Mapping[str, float] = {}
    if model_id == "target_prefix_late_knee_prior_grid":
        fixed = _LATE_KNEE_FIXED_BY_VARIANT[variant_id]
        expected_names |= set(fixed)
    if set(numeric) != expected_names:
        raise V015FitError(f"{model_id} parameters_json keys differ from the freeze")
    for name, expected in fixed.items():
        if numeric[name] != expected:
            raise V015FitError(
                f"{model_id} fixed-grid parameters do not match variant_id"
            )

    parameter_values: list[tuple[str, float]] = []
    parameter_bounds: list[tuple[str, float, float]] = []
    boundary_hits = 0
    for name, lower, upper in fitted_bounds:
        value = numeric[name]
        if not lower <= value <= upper:
            raise V015FitError(
                f"{model_id} fitted parameter lies outside its frozen bounds"
            )
        tolerance = _BOUNDARY_RELATIVE_TOLERANCE * max(1.0, upper - lower)
        if min(value - lower, upper - value) <= tolerance:
            boundary_hits += 1
        parameter_values.append((name, value))
        parameter_bounds.append((name, lower, upper))
    if model_id == "target_prefix_dual_power" and (numeric["b1"] > numeric["b2"]):
        raise V015FitError("Dual-power identifiability constraint is violated")

    fraction = boundary_hits / len(fitted_bounds) if fitted_bounds else 0.0
    return tuple(parameter_values), tuple(parameter_bounds), fraction


def _evaluate_frozen_variant(
    model_id: str,
    variant_id: str,
    parameters: Mapping[str, float],
    elapsed_days: np.ndarray,
) -> np.ndarray:
    parameter_values, _, _ = frozen_parameter_metadata(model_id, variant_id, parameters)
    if model_id == _EARLY_MODEL_ID:
        return _early_prediction(parameters, elapsed_days)
    if model_id == "target_prefix_persistence":
        fitted = {"last_retention_pct": parameters["last_retention_pct"]}
        fixed = None
    else:
        fitted = dict(parameter_values)
        fixed = (
            _LATE_KNEE_FIXED_BY_VARIANT[variant_id]
            if model_id == "target_prefix_late_knee_prior_grid"
            else None
        )
    try:
        return _v1._predict_structure(
            model_id,
            fitted,
            elapsed_days,
            time_scale_days=_legacy_protocol().time_scale_days,
            fixed=fixed,
        )
    except (KeyError, ValueError, FloatingPointError) as exc:
        raise V015FitError("Frozen variant formula evaluation failed") from exc


def recompute_variant_commitment(
    *,
    model_id: str,
    variant_id: str,
    parameters: Mapping[str, float],
    prefix_days: Sequence[float],
    observed_retention_pct: Sequence[float],
    forecast_days: Sequence[float],
) -> RecomputedVariantCommitment:
    """Recompute a successful row from its prefix and committed parameters."""

    prefix = np.asarray(prefix_days, dtype=np.float64)
    observed = np.asarray(observed_retention_pct, dtype=np.float64)
    forecast_grid = np.asarray(forecast_days, dtype=np.float64)
    protocol = _v2_protocol()
    if (
        prefix.shape != observed.shape
        or prefix.shape != (len(protocol.prefix_days),)
        or forecast_grid.shape != (len(protocol.forecast_days),)
        or tuple(float(value) for value in prefix) != protocol.prefix_days
        or tuple(float(value) for value in forecast_grid) != protocol.forecast_days
        or not np.isfinite(observed).all()
    ):
        raise V015FitError("Variant commitment uses a non-frozen predictor grid")

    parameter_values, parameter_bounds, boundary_fraction = frozen_parameter_metadata(
        model_id, variant_id, parameters
    )
    if model_id == "target_prefix_persistence" and (
        np.float64(parameters["last_retention_pct"]).tobytes()
        != np.float64(observed[-1]).tobytes()
    ):
        raise V015FitError(
            "Persistence parameter differs from the last prefix observation"
        )

    prefix_prediction = _evaluate_frozen_variant(
        model_id, variant_id, parameters, prefix
    )
    forecast = _evaluate_frozen_variant(model_id, variant_id, parameters, forecast_grid)
    residual = prefix_prediction - observed
    if (
        prefix_prediction.shape != observed.shape
        or forecast.shape != forecast_grid.shape
        or not np.isfinite(prefix_prediction).all()
        or not np.isfinite(forecast).all()
        or not np.isfinite(residual).all()
    ):
        raise V015FitError("Frozen variant formula produced invalid values")
    return RecomputedVariantCommitment(
        prefix_rmse_pp=float(np.sqrt(np.mean(np.square(residual)))),
        prefix_max_abs_residual_pp=float(np.max(np.abs(residual))),
        forecast_retention_pct=tuple(float(value) for value in forecast),
        parameter_values=parameter_values,
        parameter_bounds=parameter_bounds,
        parameter_boundary_hit_fraction=boundary_fraction,
    )


def _parameter_boundary_hit_fraction(variant: _v1.CandidateVariant) -> float:
    if not variant.fit_succeeded:
        return math.nan
    fitted_bounds = _candidate_parameter_bounds(variant.model_id)
    if not fitted_bounds:
        return 0.0
    values = variant.parameter_map()
    hits = 0
    for name, lower, upper in fitted_bounds:
        try:
            value = float(values[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise V015FitError(f"Missing fitted parameter {name}") from exc
        if not math.isfinite(value) or value < lower or value > upper:
            raise V015FitError(f"Fitted parameter {name} lies outside its bounds")
        tolerance = _BOUNDARY_RELATIVE_TOLERANCE * max(1.0, upper - lower)
        if min(abs(value - lower), abs(upper - value)) <= tolerance:
            hits += 1
    return hits / len(fitted_bounds)


def _credible_variant(variant: _v1.CandidateVariant) -> bool:
    forecast = np.asarray(variant.forecast_retention_pct, dtype=float)
    return bool(
        variant.fit_succeeded
        and math.isfinite(variant.prefix_rmse_pp)
        and variant.prefix_rmse_pp <= _MAXIMUM_PREFIX_RMSE_PP
        and math.isfinite(variant.prefix_max_absolute_residual_pp)
        and (
            variant.prefix_max_absolute_residual_pp
            <= _MAXIMUM_PREFIX_MAX_ABS_RESIDUAL_PP
        )
        and forecast.shape == (8,)
        and np.isfinite(forecast).all()
        and np.all(forecast >= _FORECAST_BOUNDS_PCT[0])
        and np.all(forecast <= _FORECAST_BOUNDS_PCT[1])
    )


def _validate_input_tables(
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(prefix_pack, pd.DataFrame) or not isinstance(
        forecast_coordinates, pd.DataFrame
    ):
        raise V015FitError("Prefix and coordinate inputs must be dataframes")
    if tuple(prefix_pack.columns) != PREFIX_COLUMNS:
        raise V015FitError("prefix_pack has unknown or missing columns")
    if tuple(forecast_coordinates.columns) != FORECAST_COORDINATE_COLUMNS:
        raise V015FitError("forecast_coordinates has unknown or missing columns")
    if prefix_pack.empty or forecast_coordinates.empty:
        raise V015FitError("Prefix and coordinate inputs cannot be empty")

    prefix = prefix_pack.copy()
    coordinates = forecast_coordinates.copy()
    allowed_partitions = set(ORDINARY_PARTITIONS + MATCHED_PARTITIONS)
    for frame, day_column in (
        (prefix, "prefix_day"),
        (coordinates, "forecast_day"),
    ):
        if not frame["protocol_id"].astype(str).eq(FROZEN_PROTOCOL_ID).all():
            raise V015FitError("Predictor protocol ID differs from frozen V2")
        if not frame["partition"].astype(str).isin(allowed_partitions).all():
            raise V015FitError("Predictor partition is invalid")
        if (
            frame["cluster_id"].isna().any()
            or frame["cluster_id"].astype(str).eq("").any()
        ):
            raise V015FitError("Predictor cluster IDs must be nonempty")
        frame[day_column] = pd.to_numeric(frame[day_column], errors="coerce")
        if (
            frame[day_column].isna().any()
            or not np.isfinite(frame[day_column].to_numpy(dtype=float)).all()
        ):
            raise V015FitError("Predictor time coordinates must be finite")

    prefix["observed_retention_pct"] = pd.to_numeric(
        prefix["observed_retention_pct"], errors="coerce"
    )
    if (
        prefix["observed_retention_pct"].isna().any()
        or not np.isfinite(prefix["observed_retention_pct"].to_numpy(dtype=float)).all()
    ):
        raise V015FitError("Prefix observations must be finite")
    if prefix.duplicated(["partition", "cluster_id", "prefix_day"]).any():
        raise V015FitError("Prefix coordinates must be unique")
    if coordinates.duplicated(["partition", "cluster_id", "forecast_day"]).any():
        raise V015FitError("Forecast coordinates must be unique")

    prefix_clusters = set(
        zip(prefix["partition"].astype(str), prefix["cluster_id"].astype(str))
    )
    forecast_clusters = set(
        zip(
            coordinates["partition"].astype(str),
            coordinates["cluster_id"].astype(str),
        )
    )
    if prefix_clusters != forecast_clusters:
        raise V015FitError("Prefix and coordinate cluster sets differ")

    protocol = _v2_protocol()
    for key, group in prefix.groupby(["partition", "cluster_id"], sort=False):
        observed_days = tuple(sorted(group["prefix_day"].to_numpy(dtype=float)))
        if observed_days != protocol.prefix_days:
            raise V015FitError(f"Cluster {key} prefix grid is incomplete")
    for key, group in coordinates.groupby(["partition", "cluster_id"], sort=False):
        observed_days = tuple(sorted(group["forecast_day"].to_numpy(dtype=float)))
        if observed_days != protocol.forecast_days:
            raise V015FitError(f"Cluster {key} forecast grid is incomplete")

    return (
        prefix.sort_values(
            ["partition", "cluster_id", "prefix_day"], kind="stable"
        ).reset_index(drop=True),
        coordinates.sort_values(
            ["partition", "cluster_id", "forecast_day"], kind="stable"
        ).reset_index(drop=True),
    )


def _prefix_content_hash(prefix_rows: pd.DataFrame, forecast_rows: pd.DataFrame) -> str:
    empty_operating = {name: 0.0 for name in (*REAL_OPERATING_FIELDS, *PLACEBO_FIELDS)}
    try:
        return predictor_content_hashes(
            prefix_rows,
            forecast_rows,
            empty_operating,
        ).arm_a
    except ValueError as exc:
        raise V015FitError("Cannot hash prefix-only predictor content") from exc


def _fit_variants(
    prefix_days: np.ndarray,
    observed: np.ndarray,
    forecast_days: np.ndarray,
) -> tuple[_v1.CandidateVariant, ...]:
    _validate_legacy_variant_declaration()
    legacy = _v1.fit_structure_family_variants(
        prefix_days,
        observed,
        forecast_days,
        _legacy_protocol(),
    )
    if len(legacy) != _EXPECTED_VARIANT_COUNT - 1:
        raise V015FitError("Imported V1 library no longer has exactly 85 variants")
    early = _fit_early_activation(prefix_days, observed, forecast_days)
    variants = (*legacy, early)
    observed_keys = [(item.model_id, item.variant_id) for item in variants]
    validate_frozen_variant_keys(observed_keys, context="fitted structure library")
    if tuple(observed_keys) != FROZEN_VARIANT_KEYS:
        raise V015FitError(
            "Frozen V2 variant IDs are missing, reordered, or duplicated"
        )
    return variants


def _validate_output_contract(
    diagnostics: pd.DataFrame,
    forecasts: pd.DataFrame,
    cluster_count: int,
) -> None:
    contract = load_artifact_contract()
    if _DIAGNOSTIC_COLUMNS != contract.csv_schema("member_fit_diagnostics.csv").columns:
        raise V015FitError("Diagnostic output schema differs from the freeze")
    if _FORECAST_COLUMNS != contract.csv_schema("member_forecast_bundle.csv").columns:
        raise V015FitError("Forecast output schema differs from the freeze")
    if len(diagnostics) != cluster_count * _EXPECTED_VARIANT_COUNT:
        raise V015FitError("Diagnostic output does not contain exactly 86 variants")
    if len(forecasts) != cluster_count * _EXPECTED_VARIANT_COUNT * 8:
        raise V015FitError("Forecast output does not contain eight rows per variant")
    for key, group in diagnostics.groupby(["partition", "cluster_id"], sort=False):
        validate_frozen_variant_keys(
            zip(
                group["model_id"].astype(str),
                group["variant_id"].astype(str),
                strict=True,
            ),
            context=f"diagnostics {key}",
        )
    for key, group in forecasts.groupby(["partition", "cluster_id"], sort=False):
        distinct_keys = tuple(
            group.loc[:, ["model_id", "variant_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        validate_frozen_variant_keys(
            distinct_keys,
            context=f"forecasts {key}",
        )
        sizes = group.groupby(["model_id", "variant_id"], sort=False).size()
        if len(sizes) != _EXPECTED_VARIANT_COUNT or not sizes.eq(8).all():
            raise V015FitError(
                f"Forecast output {key} does not contain eight rows per key"
            )


def fit_structure_library(
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
) -> V015FitResult:
    """Fit all 86 frozen variants using only prefix observations and coordinates."""

    prefix, coordinates = _validate_input_tables(prefix_pack, forecast_coordinates)
    diagnostic_records: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []

    grouped = prefix.groupby(["partition", "cluster_id"], sort=True)
    for (partition, cluster_id), prefix_group in grouped:
        forecast_group = coordinates.loc[
            coordinates["partition"].astype(str).eq(str(partition))
            & coordinates["cluster_id"].astype(str).eq(str(cluster_id))
        ].sort_values("forecast_day", kind="stable")
        prefix_group = prefix_group.sort_values("prefix_day", kind="stable")
        content_hash = _prefix_content_hash(prefix_group, forecast_group)
        forecast_days = forecast_group["forecast_day"].to_numpy(dtype=float)
        variants = _fit_variants(
            prefix_group["prefix_day"].to_numpy(dtype=float),
            prefix_group["observed_retention_pct"].to_numpy(dtype=float),
            forecast_days,
        )

        identity = {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "partition": str(partition),
            "cluster_id": str(cluster_id),
        }
        for variant in variants:
            succeeded = bool(variant.fit_succeeded)
            credible = _credible_variant(variant)
            if succeeded:
                raw_forecast = np.asarray(variant.forecast_retention_pct, dtype=float)
                if raw_forecast.shape != (8,) or not np.isfinite(raw_forecast).all():
                    raise V015FitError("Successful variant has an invalid forecast")
                parameters_json = _canonical_parameters(variant.parameters)
                prefix_rmse = float(variant.prefix_rmse_pp)
                prefix_max_residual = float(variant.prefix_max_absolute_residual_pp)
                boundary_fraction = _parameter_boundary_hit_fraction(variant)
                recomputed = recompute_variant_commitment(
                    model_id=variant.model_id,
                    variant_id=variant.variant_id,
                    parameters=variant.parameter_map(),
                    prefix_days=prefix_group["prefix_day"].to_numpy(dtype=float),
                    observed_retention_pct=prefix_group[
                        "observed_retention_pct"
                    ].to_numpy(dtype=float),
                    forecast_days=forecast_days,
                )
                if (
                    recomputed.forecast_retention_pct
                    != tuple(float(value) for value in raw_forecast)
                    or recomputed.prefix_rmse_pp != prefix_rmse
                    or (recomputed.prefix_max_abs_residual_pp != prefix_max_residual)
                    or (recomputed.parameter_boundary_hit_fraction != boundary_fraction)
                ):
                    raise V015FitError(
                        "Fitted variant differs from formula recomputation"
                    )
            else:
                raw_forecast = np.full(8, math.nan, dtype=float)
                parameters_json = "{}"
                prefix_rmse = math.nan
                prefix_max_residual = math.nan
                boundary_fraction = math.nan

            diagnostic_records.append(
                {
                    **identity,
                    "model_id": variant.model_id,
                    "variant_id": variant.variant_id,
                    "parameters_json": parameters_json,
                    "fit_status": "succeeded" if succeeded else "failed",
                    "credible_variant": credible,
                    "prefix_rmse_pp": prefix_rmse,
                    "prefix_max_abs_residual_pp": prefix_max_residual,
                    "parameter_boundary_hit_fraction": boundary_fraction,
                    "canonical_prefix_content_sha256": content_hash,
                }
            )
            for day, value in zip(forecast_days, raw_forecast, strict=True):
                forecast_records.append(
                    {
                        **identity,
                        "model_id": variant.model_id,
                        "variant_id": variant.variant_id,
                        "forecast_day": float(day),
                        "raw_forecast_retention_pct": float(value),
                        "canonical_prefix_content_sha256": content_hash,
                    }
                )

    diagnostics = (
        pd.DataFrame(diagnostic_records, columns=_DIAGNOSTIC_COLUMNS)
        .sort_values(
            ["partition", "cluster_id", "model_id", "variant_id"], kind="stable"
        )
        .reset_index(drop=True)
    )
    forecasts = (
        pd.DataFrame(forecast_records, columns=_FORECAST_COLUMNS)
        .sort_values(
            [
                "partition",
                "cluster_id",
                "model_id",
                "variant_id",
                "forecast_day",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    _validate_output_contract(diagnostics, forecasts, len(grouped))
    return V015FitResult(
        member_fit_diagnostics=diagnostics,
        member_forecast_bundle=forecasts,
    )


__all__ = [
    "FROZEN_VARIANT_KEYS",
    "FROZEN_VARIANT_KEY_SET",
    "RecomputedVariantCommitment",
    "V015FitError",
    "V015FitResult",
    "fit_structure_library",
    "frozen_parameter_metadata",
    "parse_canonical_parameters_json",
    "recompute_variant_commitment",
    "validate_frozen_variant_keys",
]
