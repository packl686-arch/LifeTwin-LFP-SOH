"""Capability-minimal V2.2 prediction runtime.

This module deliberately contains no generator, truth reader, fitting routine,
protocol loader, or scoring entry point.  It accepts only the committed
label-free directory, validates its byte commitments, decodes the prediction
subset of ``model_state.json``, and writes the three prediction artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    ConformalExpansionState,
    IsotonicState,
    LogisticRiskState,
    StandardizerState,
    V2ModelError,
    blend_center_forecast,
    build_library_forecast,
    canonical_float64_vector_bytes,
    coordinatewise_weighted_quantile,
    deduplicate_vectors,
    expand_intervals,
    family_balanced_support,
    family_representative,
    leave_day730_out_sqrt_error,
    quantized_shape_signature,
    rank_for_issuance,
)
from lifetwin.experiments.calendar_long_horizon_v017_ledger import (
    AttemptProgress,
    V022LedgerError,
    read_exposure_log,
)


class V022PredictionCapsuleError(RuntimeError):
    """Raised when the isolated prediction capsule fails closed."""


V022_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_2"

REAL_OPERATING_FIELDS = (
    "past_mean_temperature_c",
    "past_mean_soc_fraction",
    "past_mean_dod_fraction",
    "past_efc_per_year",
    "planned_mean_temperature_c",
    "planned_mean_soc_fraction",
    "planned_mean_dod_fraction",
    "planned_efc_per_year",
)
PLACEBO_FIELDS = tuple(f"placebo_control_{index}" for index in range(1, 9))
VISIBLE_STRESS_FEATURE_NAMES = PREFIX_FEATURE_NAMES + REAL_OPERATING_FIELDS
PLACEBO_FEATURE_NAMES = PREFIX_FEATURE_NAMES + PLACEBO_FIELDS
ARM_A_PLUS_S_PLAN_FEATURE_NAMES = PREFIX_FEATURE_NAMES + ("planned_stress_index",)
DECLARED_STRUCTURE_FAMILIES = (
    "target_prefix_persistence",
    "target_prefix_sqrt_time",
    "target_prefix_bounded_power_law",
    "target_prefix_saturating_plus_slow",
    "target_prefix_dual_power",
    "target_prefix_late_knee_prior_grid",
    "target_prefix_early_activation_plus_power",
)
PRIMARY_ISSUE_COUNTS = MappingProxyType({"test": 950, "audit": 475})
PRIMARY_SCORE_IDS = ("prefix_only", "visible_stress")
RISK_SCORE_IDS = (
    "prefix_only",
    "visible_stress",
    "placebo_8",
    "arm_a_plus_s_plan",
    "strongest_single_feature",
    "planned_stress_only",
    "prefix_rmse_only",
    "v1_max_envelope_only",
    "center_sqrt_abs_difference_only",
)
PARTITIONS = (
    "center_development",
    "risk_development",
    "calibration",
    "test",
    "audit",
    "intrinsic_matched_pairs",
    "stress_plan_matched_pairs",
)
PARTITION_MEMBER_COUNTS = MappingProxyType(
    {
        "center_development": 600,
        "risk_development": 600,
        "calibration": 900,
        "test": 1900,
        "audit": 950,
        "intrinsic_matched_pairs": 500,
        "stress_plan_matched_pairs": 500,
    }
)

_LATE_KNEE_K_GRID = (0.0005, 0.001, 0.002, 0.004)
_LATE_KNEE_T_GRID = (1095.75, 1826.25, 3652.5, 5478.75, 7305.0)
_LATE_KNEE_W_GRID = (30.0, 90.0, 180.0, 365.0)
_PARAMETER_BOUNDS = MappingProxyType(
    {
        "target_prefix_sqrt_time": (("c", 0.0, 5.0),),
        "target_prefix_bounded_power_law": (
            ("a", 0.0, 5.0),
            ("b", 0.05, 1.5),
        ),
        "target_prefix_saturating_plus_slow": (
            ("a_sat", 0.0, 8.0),
            ("tau_sat_days", 30.0, 1826.25),
            ("b_sat", 0.25, 2.0),
            ("a_slow", 0.0, 2.0),
            ("b_slow", 0.05, 1.2),
        ),
        "target_prefix_dual_power": (
            ("a1", 0.0, 3.0),
            ("b1", 0.05, 0.8),
            ("a2", 0.0, 3.0),
            ("b2", 0.8, 1.5),
        ),
        "target_prefix_late_knee_prior_grid": (
            ("a", 0.0, 3.0),
            ("b", 0.05, 1.5),
        ),
        "target_prefix_early_activation_plus_power": (
            ("a", 0.0, 5.0),
            ("b", 0.05, 1.5),
            ("activation_amplitude_pp", 0.0, 3.0),
            ("tau_rise_days", 3.0, 60.0),
            ("tau_decay_days", 30.0, 730.0),
        ),
    }
)
_LATE_KNEE_FIXED_BY_VARIANT = MappingProxyType(
    {
        f"k={k:g}|t={t_knee:g}|w={width:g}": MappingProxyType(
            {
                "k_pp_per_day": k,
                "t_knee_days": t_knee,
                "w_days": width,
            }
        )
        for k in _LATE_KNEE_K_GRID
        for t_knee in _LATE_KNEE_T_GRID
        for width in _LATE_KNEE_W_GRID
    }
)
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
    (
        "target_prefix_early_activation_plus_power",
        "target_prefix_early_activation_plus_power",
    ),
)
FROZEN_VARIANT_KEY_SET = frozenset(FROZEN_VARIANT_KEYS)
if len(FROZEN_VARIANT_KEYS) != 86 or len(FROZEN_VARIANT_KEY_SET) != 86:
    raise RuntimeError("The prediction capsule variant registry changed")


@dataclass(frozen=True, slots=True)
class PredictionCsvSchema:
    filename: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    required_rows: int | None = None
    required_values: tuple[str, ...] = ()
    required_value_column: str | None = None


def _schema(
    filename: str,
    columns: tuple[str, ...],
    key: tuple[str, ...],
    *,
    required_rows: int | None = None,
    required_values: tuple[str, ...] = (),
    required_value_column: str | None = None,
) -> PredictionCsvSchema:
    return PredictionCsvSchema(
        filename=filename,
        columns=columns,
        key=key,
        required_rows=required_rows,
        required_values=required_values,
        required_value_column=required_value_column,
    )


_SCHEMAS = MappingProxyType(
    {
        "prefix_pack.csv": _schema(
            "prefix_pack.csv",
            (
                "protocol_id",
                "partition",
                "cluster_id",
                "prefix_day",
                "observed_retention_pct",
            ),
            ("partition", "cluster_id", "prefix_day"),
            required_rows=71_400,
        ),
        "forecast_coordinates.csv": _schema(
            "forecast_coordinates.csv",
            ("protocol_id", "partition", "cluster_id", "forecast_day"),
            ("partition", "cluster_id", "forecast_day"),
            required_rows=47_600,
        ),
        "operating_pack.csv": _schema(
            "operating_pack.csv",
            (
                "protocol_id",
                "partition",
                "cluster_id",
                *REAL_OPERATING_FIELDS,
                *PLACEBO_FIELDS,
            ),
            ("partition", "cluster_id"),
            required_rows=5_950,
        ),
        "member_fit_diagnostics.csv": _schema(
            "member_fit_diagnostics.csv",
            (
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
            ),
            ("partition", "cluster_id", "model_id", "variant_id"),
        ),
        "member_forecast_bundle.csv": _schema(
            "member_forecast_bundle.csv",
            (
                "protocol_id",
                "partition",
                "cluster_id",
                "model_id",
                "variant_id",
                "forecast_day",
                "raw_forecast_retention_pct",
                "canonical_prefix_content_sha256",
            ),
            (
                "partition",
                "cluster_id",
                "model_id",
                "variant_id",
                "forecast_day",
            ),
        ),
        "prediction_bundle.csv": _schema(
            "prediction_bundle.csv",
            (
                "protocol_id",
                "partition",
                "cluster_id",
                "forecast_day",
                "center_forecast_pct",
                "sqrt_time_forecast_pct",
                "bounded_power_forecast_pct",
                "base_interval_lower_pct",
                "base_interval_upper_pct",
                "calibrated_interval_lower_pct",
                "calibrated_interval_upper_pct",
                "canonical_prefix_content_sha256",
            ),
            ("partition", "cluster_id", "forecast_day"),
            required_rows=47_600,
        ),
        "risk_bundle.csv": _schema(
            "risk_bundle.csv",
            (
                "protocol_id",
                "partition",
                "cluster_id",
                "score_id",
                "raw_risk_score",
                "calibrated_catastrophic_probability",
                "all_features_finite",
                "successful_structure_family_count",
                "fit_failure_count",
                "effective_unique_shape_count",
                "canonical_predictor_content_sha256",
            ),
            ("partition", "cluster_id", "score_id"),
            required_values=RISK_SCORE_IDS,
            required_value_column="score_id",
        ),
        "decision_bundle.csv": _schema(
            "decision_bundle.csv",
            (
                "protocol_id",
                "partition",
                "cluster_id",
                "arm",
                "raw_risk_score",
                "hard_eligible",
                "issuance_rank",
                "issued",
                "abstention_reasons",
                "canonical_predictor_content_sha256",
            ),
            ("partition", "cluster_id", "arm"),
            required_values=PRIMARY_SCORE_IDS,
            required_value_column="arm",
        ),
    }
)

_LABEL_INPUTS = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
)
_FIT_OUTPUTS = (
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
)
_PREDICTION_OUTPUTS = (
    "prediction_bundle.csv",
    "risk_bundle.csv",
    "decision_bundle.csv",
)
_FIT_COMMITMENT_FILES = (
    "generation_plan_commitment.json",
    *_LABEL_INPUTS,
    "truth_commitments.json",
    "actual_analysis_hash_ledger_commitment.json",
    *_FIT_OUTPUTS,
)
_MODEL_STATE_COMMITMENT_FILES = (
    "fit_commitment.json",
    "center_state_checkpoint.json",
    "risk_state_checkpoint.json",
    "training_manifest.json",
    "calibration_mask_commitment.json",
    "calibration_manifest.json",
    "calibration_population_audit.json",
    "model_state.json",
)
_PRE_PREDICTION_FILES = frozenset(
    {
        "generation_plan_commitment.json",
        *_LABEL_INPUTS,
        "truth_commitments.json",
        "exposure_log.jsonl",
        "actual_analysis_hash_ledger_commitment.json",
        *_FIT_OUTPUTS,
        *_MODEL_STATE_COMMITMENT_FILES,
        "model_state_commitment.json",
    }
)
_SEALED_FILENAMES = (
    "center_development_truth.csv",
    "risk_development_truth.csv",
    "calibration_truth.csv",
    "test_truth.csv",
    "audit_truth.csv",
    "intrinsic_matched_truth.csv",
    "stress_plan_matched_truth.csv",
    "intrinsic_matched_pairs.csv",
    "stress_plan_matched_pairs.csv",
)
_FILE_ENTRY_KEYS = frozenset({"path", "row_count", "byte_count", "byte_sha256"})
_FIT_COMMITMENT_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "git_commit",
        "worker_count",
        "files",
        "created_utc",
    }
)
_MODEL_COMMITMENT_KEYS = frozenset(
    {"protocol_id", "config_sha256", "git_commit", "files", "created_utc"}
)
_MODEL_STATE_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "center_state",
        "risk_states",
        "calibration_state",
        "comparator_states",
        "feature_orders",
        "input_byte_hashes",
        "software_versions",
        "created_utc",
    }
)
_STRICT_BOOLEAN_COLUMNS = frozenset(
    {"credible_variant", "all_features_finite", "hard_eligible", "issued"}
)
_STRING_KEY_COLUMNS = frozenset(
    {
        "partition",
        "cluster_id",
        "model_id",
        "variant_id",
        "score_id",
        "arm",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_FORMULA_ABSOLUTE_TOLERANCE = 1e-12
_FORECAST_BOUNDS_PCT = (40.0, 105.0)
_MAXIMUM_PREFIX_RMSE_PP = 1.0
_MAXIMUM_PREFIX_RESIDUAL_PP = 1.5
_EXPECTED_SOFTWARE_VERSIONS = {
    "numpy": "2.5.1",
    "python": "3.12.13",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
}
_MODEL_STATE_PHASES = frozenset(
    {"center_development", "risk_development", "calibration"}
)
_BUNDLE_SEAL = object()
_PIPELINE_OUTPUT_FIELDS = (
    ("prediction_bundle.csv", "prediction_bundle"),
    ("risk_bundle.csv", "primary_risk_bundle"),
    ("decision_bundle.csv", "decision_bundle"),
)
_MASK_HASH_DOMAIN = b"lifetwin-v022-calibration-mask-v2\0"
_MASK_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "source_calibration_count",
        "risk_isotonic_eligible_count",
        "eligibility_mask_sha256",
        "rows",
    }
)
_MASK_ROW_KEYS = frozenset(
    {
        "cluster_id",
        "label_free_row_sha256",
        "structural_support_sha256",
        "successful_structure_family_ids",
        "eligible",
        "ineligibility_reasons",
    }
)
_CENTER_CHECKPOINT_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "state_kind",
        "center_state_sha256",
        "center_beta",
        "development_cluster_count",
        "forecast_horizon_count",
        "ridge_penalty",
        "completeness_rule",
        "input_byte_hashes",
        "created_utc",
    }
)
_RISK_CHECKPOINT_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "state_kind",
        "center_checkpoint_byte_sha256",
        "training_manifest_byte_sha256",
        "risk_state_sha256",
        "development_cluster_count",
        "eligible_cluster_count",
        "positive_label_count",
        "negative_label_count",
        "input_byte_hashes",
        "created_utc",
    }
)
_TRAINING_MANIFEST_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "center_development_input_hashes",
        "risk_development_input_hashes",
        "opened_truth_files",
        "forbidden_v1_evidence_matches",
        "center_state_sha256",
        "risk_state_sha256",
        "created_utc",
    }
)
_CALIBRATION_MANIFEST_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "calibration_input_hashes",
        "opened_truth_files",
        "isotonic_state_sha256",
        "conformal_state_sha256",
        "selected_mean_baseline",
        "created_utc",
    }
)
_CALIBRATION_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "config_sha256",
        "source_calibration_count",
        "risk_isotonic_eligible_count",
        "risk_isotonic_ineligible_zero_family_count",
        "risk_isotonic_ineligible_one_family_count",
        "risk_isotonic_ineligible_other_count",
        "risk_isotonic_positive_label_count",
        "risk_isotonic_negative_label_count",
        "mean_baseline_count",
        "conformal_calibration_count",
        "conformal_order_statistic_index",
        "eligibility_mask_cluster_ids",
        "eligibility_mask",
        "eligibility_mask_sha256",
        "calibration_mask_commitment_byte_sha256",
        "isotonic_state_sha256",
        "conformal_state_sha256",
        "selected_mean_baseline",
        "created_utc",
    }
)
_MASK_INELIGIBILITY_REASONS = frozenset(
    {
        "invalid_prefix_grid_or_observations",
        "invalid_forecast_coordinates",
        "insufficient_structure_families",
        "nonfinite_center_forecast",
        "nonfinite_prefix_features",
        "nonfinite_real_operating_fields",
        "nonfinite_placebo_operating_fields",
        "nonfinite_real_operating_features",
        "nonfinite_placebo_features",
        "nonfinite_primary_risk_scores",
    }
)
_CLUSTER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TRUTH_REQUIRED_ROWS = MappingProxyType(
    {
        "center_development_truth.csv": 4_800,
        "risk_development_truth.csv": 4_800,
        "calibration_truth.csv": 7_200,
        "test_truth.csv": 15_200,
        "audit_truth.csv": 7_600,
        "intrinsic_matched_truth.csv": 4_000,
        "stress_plan_matched_truth.csv": 4_000,
        "intrinsic_matched_pairs.csv": 500,
        "stress_plan_matched_pairs.csv": 500,
    }
)


@dataclass(frozen=True, slots=True)
class PredictionArtifactMetadata:
    path: str
    row_count: int
    byte_count: int
    byte_sha256: str


@dataclass(frozen=True, slots=True)
class PredictionState:
    center_beta: float
    prefix_only_risk: LogisticRiskState
    visible_stress_risk: LogisticRiskState
    placebo_risk: LogisticRiskState
    arm_a_plus_s_plan_risk: LogisticRiskState
    strongest_single_feature_name: str
    strongest_single_feature_orientation: int
    prefix_only_isotonic: IsotonicState
    visible_stress_isotonic: IsotonicState
    conformal: ConformalExpansionState


@dataclass(frozen=True, slots=True)
class DecodedPredictionState:
    state: PredictionState
    input_byte_hashes: Mapping[str, Mapping[str, str]]
    model_state_byte_sha256: str


@dataclass(frozen=True, slots=True)
class _MaskEvidence:
    cluster_ids: tuple[str, ...]
    eligible: tuple[bool, ...]
    family_counts: tuple[int, ...]
    eligibility_mask_sha256: str


@dataclass(frozen=True, slots=True)
class PredictionPipelineResult:
    prediction_bundle: pd.DataFrame
    feature_bundle: pd.DataFrame
    primary_risk_bundle: pd.DataFrame
    decision_bundle: pd.DataFrame
    predictor_content_bundle: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PredictorContentHashes:
    random_policy: str
    arm_a: str
    arm_b: str
    placebo: str


@dataclass(frozen=True, slots=True)
class _SafeVariant:
    forecast: tuple[float, ...]
    prefix_rmse_pp: float
    boundary_fraction: float
    parameter_payload: str


class PredictionBundle:
    """Opaque committed label-free capability for the isolated process."""

    __slots__ = (
        "_seal",
        "_config_sha256",
        "_file_hashes",
        "_frames",
        "_git_commit",
        "_ledger_raw",
        "_root",
        "_state",
    )

    def __init__(
        self,
        *,
        _seal: object,
        root: Path,
        config_sha256: str,
        git_commit: str,
        frames: Mapping[str, pd.DataFrame],
        state: PredictionState,
        file_hashes: Mapping[str, str],
        ledger_raw: bytes,
    ) -> None:
        if _seal is not _BUNDLE_SEAL or type(self) is not PredictionBundle:
            raise TypeError(
                "Prediction bundles are issued only by the strict capsule loader"
            )
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_config_sha256", config_sha256)
        object.__setattr__(self, "_git_commit", git_commit)
        object.__setattr__(
            self,
            "_frames",
            tuple((name, frame.copy(deep=True)) for name, frame in frames.items()),
        )
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_file_hashes", tuple(sorted(file_hashes.items())))
        object.__setattr__(self, "_ledger_raw", ledger_raw)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Prediction capsule bundles are immutable")


_PIPELINE_RESULT_BINDINGS: dict[
    int,
    tuple[
        PredictionPipelineResult,
        PredictionBundle,
        tuple[tuple[str, str], ...],
    ],
] = {}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V022PredictionCapsuleError(f"{context} must be lowercase SHA256")
    return value


def _utc(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise V022PredictionCapsuleError(f"{context} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise V022PredictionCapsuleError(f"{context} is invalid") from exc
    if parsed.utcoffset() is None:
        raise V022PredictionCapsuleError(f"{context} lacks a timezone")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V022PredictionCapsuleError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V022PredictionCapsuleError(f"Nonfinite JSON constant is forbidden: {token}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise V022PredictionCapsuleError(
            "Canonical JSON cannot contain NaN or infinity"
        )
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            _json_ready(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise V022PredictionCapsuleError("Payload is not finite JSON") from exc
    return (text + "\n").encode("utf-8")


def _strict_json(
    raw: bytes,
    *,
    filename: str,
    compact: bool = False,
) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise V022PredictionCapsuleError(f"{filename} must be exact bytes")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V022PredictionCapsuleError(
            f"{filename} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise V022PredictionCapsuleError(f"{filename} must be a JSON object")
    if compact:
        canonical = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    else:
        canonical = canonical_json_bytes(payload)
    if canonical != raw:
        raise V022PredictionCapsuleError(f"{filename} is not canonical JSON")
    return payload


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise V022PredictionCapsuleError(f"{context} keys changed")
    return value


def _real(value: object, *, context: str) -> float:
    if type(value) is not float:
        raise V022PredictionCapsuleError(f"{context} must be a canonical JSON real")
    result = value
    if not math.isfinite(result):
        raise V022PredictionCapsuleError(f"{context} must be finite")
    return result


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise V022PredictionCapsuleError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise V022PredictionCapsuleError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise V022PredictionCapsuleError(f"{context} must be finite")
    return result


def _integer(value: object, *, context: str) -> int:
    if type(value) is not int:
        raise V022PredictionCapsuleError(f"{context} must be an integer")
    return value


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise V022PredictionCapsuleError(f"{context} must be a nonempty string")
    return value


def _real_list(
    value: object,
    *,
    context: str,
    length: int | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise V022PredictionCapsuleError(f"{context} must be an array")
    result = tuple(
        _real(item, context=f"{context}[{index}]") for index, item in enumerate(value)
    )
    if length is not None and len(result) != length:
        raise V022PredictionCapsuleError(f"{context} length changed")
    return result


def _bool_list(
    value: object,
    *,
    context: str,
    length: int,
) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) is not bool for item in value)
    ):
        raise V022PredictionCapsuleError(f"{context} must be a boolean array")
    return tuple(value)


def _parse_logistic(
    value: object,
    *,
    context: str,
    expected_names: tuple[str, ...],
) -> LogisticRiskState:
    payload = _exact_mapping(
        value,
        frozenset({"feature_names", "standardizer", "intercept", "coefficients"}),
        context=context,
    )
    names = payload["feature_names"]
    if not isinstance(names, list) or tuple(names) != expected_names:
        raise V022PredictionCapsuleError(f"{context}.feature_names changed")
    dimension = len(expected_names)
    standardizer = _exact_mapping(
        payload["standardizer"],
        frozenset({"mean", "scale", "zero_variance"}),
        context=f"{context}.standardizer",
    )
    mean = _real_list(
        standardizer["mean"],
        context=f"{context}.standardizer.mean",
        length=dimension,
    )
    scale = _real_list(
        standardizer["scale"],
        context=f"{context}.standardizer.scale",
        length=dimension,
    )
    zero = _bool_list(
        standardizer["zero_variance"],
        context=f"{context}.standardizer.zero_variance",
        length=dimension,
    )
    coefficients = _real_list(
        payload["coefficients"],
        context=f"{context}.coefficients",
        length=dimension,
    )
    if any(item <= 0.0 for item in scale):
        raise V022PredictionCapsuleError(
            f"{context}.standardizer.scale must be positive"
        )
    for index, is_zero in enumerate(zero):
        if is_zero and (scale[index] != 1.0 or abs(coefficients[index]) > 1e-12):
            raise V022PredictionCapsuleError(f"{context} zero-variance state changed")
    return LogisticRiskState(
        feature_names=expected_names,
        standardizer=StandardizerState(
            mean=mean,
            scale=scale,
            zero_variance=zero,
        ),
        intercept=_real(payload["intercept"], context=f"{context}.intercept"),
        coefficients=coefficients,
    )


def _parse_isotonic(value: object, *, context: str) -> IsotonicState:
    payload = _exact_mapping(
        value,
        frozenset({"x_thresholds", "y_thresholds"}),
        context=context,
    )
    x = _real_list(payload["x_thresholds"], context=f"{context}.x_thresholds")
    y = _real_list(payload["y_thresholds"], context=f"{context}.y_thresholds")
    if (
        len(x) < 2
        or len(x) != len(y)
        or any(right <= left for left, right in zip(x, x[1:]))
        or any(right < left for left, right in zip(y, y[1:]))
        or any(item < 0.0 or item > 1.0 for item in y)
    ):
        raise V022PredictionCapsuleError(f"{context} is not an isotonic map")
    return IsotonicState(x_thresholds=x, y_thresholds=y)


def _parse_conformal(value: object) -> ConformalExpansionState:
    context = "calibration_state.conformal"
    payload = _exact_mapping(
        value,
        frozenset(
            {
                "coverage",
                "calibration_count",
                "order_statistic_index",
                "expansion_pp",
            }
        ),
        context=context,
    )
    coverage = _real(payload["coverage"], context=f"{context}.coverage")
    count = _integer(payload["calibration_count"], context=f"{context}.count")
    index = _integer(
        payload["order_statistic_index"],
        context=f"{context}.order_statistic_index",
    )
    expansion = _real(
        payload["expansion_pp"],
        context=f"{context}.expansion_pp",
    )
    if coverage != 0.9 or count != 900 or index != 811 or expansion < 0.0:
        raise V022PredictionCapsuleError(
            "Conformal state differs from the frozen 900/811 rule"
        )
    return ConformalExpansionState(
        coverage=coverage,
        calibration_count=count,
        order_statistic_index=index,
        expansion_pp=expansion,
    )


def _validate_prediction_state(state: PredictionState) -> None:
    if not math.isfinite(state.center_beta) or not 0.0 <= state.center_beta <= 1.0:
        raise V022PredictionCapsuleError("center_beta is invalid")
    expected = (
        (state.prefix_only_risk, PREFIX_FEATURE_NAMES),
        (state.visible_stress_risk, VISIBLE_STRESS_FEATURE_NAMES),
        (state.placebo_risk, PLACEBO_FEATURE_NAMES),
        (state.arm_a_plus_s_plan_risk, ARM_A_PLUS_S_PLAN_FEATURE_NAMES),
    )
    if any(risk.feature_names != names for risk, names in expected):
        raise V022PredictionCapsuleError("Prediction feature order changed")
    if (
        state.strongest_single_feature_name not in PREFIX_FEATURE_NAMES
        or state.strongest_single_feature_orientation not in {-1, 1}
    ):
        raise V022PredictionCapsuleError("Strongest single-feature state is invalid")


def decode_prediction_state(
    raw: bytes,
    *,
    expected_config_sha256: str,
) -> DecodedPredictionState:
    config_sha256 = _digest(
        expected_config_sha256,
        context="expected config hash",
    )
    payload = _strict_json(raw, filename="model_state.json")
    top = _exact_mapping(payload, _MODEL_STATE_KEYS, context="model_state.json")
    if (
        top.get("protocol_id") != V022_PROTOCOL_ID
        or top.get("config_sha256") != config_sha256
    ):
        raise V022PredictionCapsuleError("model_state.json identity changed")

    feature_orders = _exact_mapping(
        top["feature_orders"],
        frozenset({"prefix_only", "visible_stress", "placebo_8", "arm_a_plus_s_plan"}),
        context="feature_orders",
    )
    expected_orders = {
        "prefix_only": list(PREFIX_FEATURE_NAMES),
        "visible_stress": list(VISIBLE_STRESS_FEATURE_NAMES),
        "placebo_8": list(PLACEBO_FEATURE_NAMES),
        "arm_a_plus_s_plan": list(ARM_A_PLUS_S_PLAN_FEATURE_NAMES),
    }
    if dict(feature_orders) != expected_orders:
        raise V022PredictionCapsuleError("feature_orders changed")

    center = _exact_mapping(
        top["center_state"],
        frozenset(
            {
                "beta",
                "development_cluster_count",
                "forecast_horizon_count",
                "ridge_penalty",
                "completeness_rule",
            }
        ),
        context="center_state",
    )
    center_beta = _real(center["beta"], context="center_state.beta")
    if (
        _integer(
            center["development_cluster_count"],
            context="center_state.development_cluster_count",
        )
        != 600
        or _integer(
            center["forecast_horizon_count"],
            context="center_state.forecast_horizon_count",
        )
        != 8
        or _real(center["ridge_penalty"], context="center_state.ridge_penalty") != 0.01
        or center["completeness_rule"]
        != "exactly_600_complete_rows_all_8_horizons_finite"
        or not 0.0 <= center_beta <= 1.0
    ):
        raise V022PredictionCapsuleError("center_state changed")

    risk = _exact_mapping(
        top["risk_states"],
        frozenset(
            {
                "development_cluster_count",
                "eligible_cluster_count",
                "positive_label_count",
                "negative_label_count",
                "catastrophic_threshold_pp",
                "prefix_only",
                "visible_stress",
            }
        ),
        context="risk_states",
    )
    development_count = _integer(
        risk["development_cluster_count"],
        context="risk_states.development_cluster_count",
    )
    eligible_count = _integer(
        risk["eligible_cluster_count"],
        context="risk_states.eligible_cluster_count",
    )
    positive_count = _integer(
        risk["positive_label_count"],
        context="risk_states.positive_label_count",
    )
    negative_count = _integer(
        risk["negative_label_count"],
        context="risk_states.negative_label_count",
    )
    if (
        development_count != 600
        or not 0 < eligible_count <= development_count
        or positive_count + negative_count != eligible_count
        or positive_count < 60
        or negative_count < 60
        or _real(
            risk["catastrophic_threshold_pp"],
            context="risk_states.catastrophic_threshold_pp",
        )
        != 5.0
    ):
        raise V022PredictionCapsuleError("risk development state changed")

    comparators = _exact_mapping(
        top["comparator_states"],
        frozenset({"placebo_8", "arm_a_plus_s_plan", "strongest_single_feature"}),
        context="comparator_states",
    )
    strongest = _exact_mapping(
        comparators["strongest_single_feature"],
        frozenset({"feature_name", "danger_orientation", "oriented_empirical_auroc"}),
        context="comparator_states.strongest_single_feature",
    )
    strongest_name = _string(
        strongest["feature_name"],
        context="strongest_single_feature.feature_name",
    )
    strongest_orientation = _integer(
        strongest["danger_orientation"],
        context="strongest_single_feature.danger_orientation",
    )
    strongest_auroc = _real(
        strongest["oriented_empirical_auroc"],
        context="strongest_single_feature.oriented_empirical_auroc",
    )
    if (
        strongest_name not in PREFIX_FEATURE_NAMES
        or strongest_orientation not in {-1, 1}
        or not 0.5 <= strongest_auroc <= 1.0
    ):
        raise V022PredictionCapsuleError("strongest single-feature state changed")

    calibration = _exact_mapping(
        top["calibration_state"],
        frozenset(
            {
                "calibration_cluster_count",
                "positive_label_count",
                "negative_label_count",
                "prefix_only_isotonic",
                "visible_stress_isotonic",
                "conformal",
                "selected_mean_baseline",
                "mean_baseline_iae_pp",
            }
        ),
        context="calibration_state",
    )
    calibration_count = _integer(
        calibration["calibration_cluster_count"],
        context="calibration_state.calibration_cluster_count",
    )
    calibration_positive = _integer(
        calibration["positive_label_count"],
        context="calibration_state.positive_label_count",
    )
    calibration_negative = _integer(
        calibration["negative_label_count"],
        context="calibration_state.negative_label_count",
    )
    if (
        calibration_count != 900
        or calibration_positive + calibration_negative != calibration_count
        or calibration_positive < 60
        or calibration_negative < 60
    ):
        raise V022PredictionCapsuleError("calibration counts changed")
    baseline_value = calibration["mean_baseline_iae_pp"]
    baseline_ids = (
        "target_prefix_persistence",
        "target_prefix_sqrt_time",
        "target_prefix_bounded_power_law",
    )
    if not isinstance(baseline_value, Mapping) or set(baseline_value) != set(
        baseline_ids
    ):
        raise V022PredictionCapsuleError("mean baseline registry changed")
    baseline_iae = {
        model_id: _real(
            baseline_value[model_id],
            context=f"mean_baseline_iae_pp.{model_id}",
        )
        for model_id in baseline_ids
    }
    if any(value < 0.0 for value in baseline_iae.values()):
        raise V022PredictionCapsuleError("mean baseline IAE is negative")
    selected = _string(
        calibration["selected_mean_baseline"],
        context="calibration_state.selected_mean_baseline",
    )
    if selected != min(
        baseline_iae,
        key=lambda model_id: (baseline_iae[model_id], model_id),
    ):
        raise V022PredictionCapsuleError("selected mean baseline changed")

    state = PredictionState(
        center_beta=center_beta,
        prefix_only_risk=_parse_logistic(
            risk["prefix_only"],
            context="risk_states.prefix_only",
            expected_names=PREFIX_FEATURE_NAMES,
        ),
        visible_stress_risk=_parse_logistic(
            risk["visible_stress"],
            context="risk_states.visible_stress",
            expected_names=VISIBLE_STRESS_FEATURE_NAMES,
        ),
        placebo_risk=_parse_logistic(
            comparators["placebo_8"],
            context="comparator_states.placebo_8",
            expected_names=PLACEBO_FEATURE_NAMES,
        ),
        arm_a_plus_s_plan_risk=_parse_logistic(
            comparators["arm_a_plus_s_plan"],
            context="comparator_states.arm_a_plus_s_plan",
            expected_names=ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
        ),
        strongest_single_feature_name=strongest_name,
        strongest_single_feature_orientation=strongest_orientation,
        prefix_only_isotonic=_parse_isotonic(
            calibration["prefix_only_isotonic"],
            context="calibration_state.prefix_only_isotonic",
        ),
        visible_stress_isotonic=_parse_isotonic(
            calibration["visible_stress_isotonic"],
            context="calibration_state.visible_stress_isotonic",
        ),
        conformal=_parse_conformal(calibration["conformal"]),
    )
    _validate_prediction_state(state)

    raw_hashes = top["input_byte_hashes"]
    if not isinstance(raw_hashes, Mapping) or set(raw_hashes) != _MODEL_STATE_PHASES:
        raise V022PredictionCapsuleError("model input phase registry changed")
    input_hashes: dict[str, Mapping[str, str]] = {}
    for phase in _MODEL_STATE_PHASES:
        phase_hashes = raw_hashes[phase]
        if not isinstance(phase_hashes, Mapping) or not phase_hashes:
            raise V022PredictionCapsuleError(
                f"model input hashes are empty for {phase}"
            )
        input_hashes[phase] = MappingProxyType(
            {
                name: _digest(
                    digest,
                    context=f"{phase} input hash/{name}",
                )
                for name, digest in phase_hashes.items()
                if isinstance(name, str) and _IDENTIFIER.fullmatch(name) is not None
            }
        )
        if len(input_hashes[phase]) != len(phase_hashes):
            raise V022PredictionCapsuleError(f"{phase} input filename is invalid")
    if (
        not isinstance(top["software_versions"], Mapping)
        or dict(top["software_versions"]) != _EXPECTED_SOFTWARE_VERSIONS
    ):
        raise V022PredictionCapsuleError("software_versions changed")
    _utc(top["created_utc"], context="model_state.json created_utc")
    return DecodedPredictionState(
        state=state,
        input_byte_hashes=MappingProxyType(input_hashes),
        model_state_byte_sha256=_sha256(raw),
    )


def _decode_mask_commitment(raw: bytes) -> _MaskEvidence:
    payload = _strict_json(
        raw,
        filename="calibration_mask_commitment.json",
        compact=True,
    )
    if set(payload) != _MASK_TOP_LEVEL_KEYS:
        raise V022PredictionCapsuleError("Calibration mask commitment schema changed")
    if (
        payload["schema_version"] != "1.0.0"
        or payload["protocol_id"] != V022_PROTOCOL_ID
    ):
        raise V022PredictionCapsuleError("Calibration mask commitment identity changed")
    source_count = _integer(
        payload["source_calibration_count"],
        context="mask source_calibration_count",
    )
    declared_eligible = _integer(
        payload["risk_isotonic_eligible_count"],
        context="mask risk_isotonic_eligible_count",
    )
    declared_digest = _digest(
        payload["eligibility_mask_sha256"],
        context="mask eligibility_mask_sha256",
    )
    rows = payload["rows"]
    if not isinstance(rows, list) or source_count != 900 or len(rows) != 900:
        raise V022PredictionCapsuleError(
            "Calibration mask must contain exactly 900 rows"
        )

    cluster_ids: list[str] = []
    eligible: list[bool] = []
    family_counts: list[int] = []
    label_free_hashes: list[str] = []
    for index, value in enumerate(rows):
        row = _exact_mapping(
            value,
            _MASK_ROW_KEYS,
            context=f"mask rows[{index}]",
        )
        cluster_id = row["cluster_id"]
        if not isinstance(cluster_id, str) or _CLUSTER_ID.fullmatch(cluster_id) is None:
            raise V022PredictionCapsuleError(
                f"mask rows[{index}] cluster ID is invalid"
            )
        label_free_hash = _digest(
            row["label_free_row_sha256"],
            context=f"mask rows[{index}] label-free hash",
        )
        _digest(
            row["structural_support_sha256"],
            context=f"mask rows[{index}] support hash",
        )
        families = row["successful_structure_family_ids"]
        reasons = row["ineligibility_reasons"]
        if (
            not isinstance(families, list)
            or any(not isinstance(item, str) for item in families)
            or tuple(families) != tuple(sorted(set(families)))
            or not set(families).issubset(set(DECLARED_STRUCTURE_FAMILIES))
        ):
            raise V022PredictionCapsuleError(
                f"mask rows[{index}] family IDs are invalid"
            )
        if (
            not isinstance(reasons, list)
            or any(not isinstance(item, str) for item in reasons)
            or tuple(reasons) != tuple(sorted(set(reasons)))
            or not set(reasons).issubset(_MASK_INELIGIBILITY_REASONS)
        ):
            raise V022PredictionCapsuleError(f"mask rows[{index}] reasons are invalid")
        is_eligible = row["eligible"]
        if type(is_eligible) is not bool or is_eligible == bool(reasons):
            raise V022PredictionCapsuleError(
                f"mask rows[{index}] eligibility is inconsistent"
            )
        cluster_ids.append(cluster_id)
        eligible.append(is_eligible)
        family_counts.append(len(families))
        label_free_hashes.append(label_free_hash)

    identifiers = tuple(cluster_ids)
    if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != 900:
        raise V022PredictionCapsuleError(
            "Calibration mask rows are reordered or duplicated"
        )
    eligible_count = sum(eligible)
    if eligible_count < 855 or declared_eligible != eligible_count:
        raise V022PredictionCapsuleError("Calibration mask eligible count is invalid")
    hasher = hashlib.sha256()
    hasher.update(_MASK_HASH_DOMAIN)
    hasher.update(struct.pack("<Q", source_count))
    for cluster_id, label_free_hash, is_eligible in zip(
        identifiers,
        label_free_hashes,
        eligible,
        strict=True,
    ):
        encoded = cluster_id.encode("ascii")
        hasher.update(struct.pack("<Q", len(encoded)))
        hasher.update(encoded)
        hasher.update(bytes.fromhex(label_free_hash))
        hasher.update(struct.pack("<B", int(is_eligible)))
    if hasher.hexdigest() != declared_digest:
        raise V022PredictionCapsuleError(
            "Calibration mask digest does not match its rows"
        )
    return _MaskEvidence(
        cluster_ids=identifiers,
        eligible=tuple(eligible),
        family_counts=tuple(family_counts),
        eligibility_mask_sha256=declared_digest,
    )


def _model_substate_hashes(
    model_payload: Mapping[str, Any],
) -> Mapping[str, str]:
    risk_payload = {
        "risk_states": model_payload["risk_states"],
        "comparator_states": model_payload["comparator_states"],
        "feature_orders": model_payload["feature_orders"],
    }
    calibration = model_payload["calibration_state"]
    if not isinstance(calibration, Mapping):
        raise V022PredictionCapsuleError("calibration_state must be an object")
    isotonic_payload = {
        "prefix_only_isotonic": calibration["prefix_only_isotonic"],
        "visible_stress_isotonic": calibration["visible_stress_isotonic"],
    }
    return MappingProxyType(
        {
            "center": _sha256(canonical_json_bytes(model_payload["center_state"])),
            "risk": _sha256(canonical_json_bytes(risk_payload)),
            "isotonic": _sha256(canonical_json_bytes(isotonic_payload)),
            "conformal": _sha256(canonical_json_bytes(calibration["conformal"])),
        }
    )


def _verify_training_chain_semantics(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    raw_by_name: Mapping[str, bytes],
    decoded: DecodedPredictionState,
    mask: _MaskEvidence,
    config_sha256: str,
) -> None:
    model = payloads["model_state.json"]
    hashes = _model_substate_hashes(model)
    center = _exact_mapping(
        payloads["center_state_checkpoint.json"],
        _CENTER_CHECKPOINT_KEYS,
        context="center_state_checkpoint.json",
    )
    risk = _exact_mapping(
        payloads["risk_state_checkpoint.json"],
        _RISK_CHECKPOINT_KEYS,
        context="risk_state_checkpoint.json",
    )
    training = _exact_mapping(
        payloads["training_manifest.json"],
        _TRAINING_MANIFEST_KEYS,
        context="training_manifest.json",
    )
    calibration = _exact_mapping(
        payloads["calibration_manifest.json"],
        _CALIBRATION_MANIFEST_KEYS,
        context="calibration_manifest.json",
    )
    audit = _exact_mapping(
        payloads["calibration_population_audit.json"],
        _CALIBRATION_AUDIT_KEYS,
        context="calibration_population_audit.json",
    )
    for filename, payload in (
        ("center_state_checkpoint.json", center),
        ("risk_state_checkpoint.json", risk),
        ("training_manifest.json", training),
        ("calibration_manifest.json", calibration),
        ("calibration_population_audit.json", audit),
    ):
        _identity_json(
            payload,
            filename=filename,
            config_sha256=config_sha256,
        )
        _utc(payload["created_utc"], context=f"{filename} created_utc")

    center_inputs = dict(decoded.input_byte_hashes["center_development"])
    risk_inputs = dict(decoded.input_byte_hashes["risk_development"])
    calibration_inputs = dict(decoded.input_byte_hashes["calibration"])
    model_center = model["center_state"]
    model_risk = model["risk_states"]
    model_calibration = model["calibration_state"]
    if (
        not isinstance(model_center, Mapping)
        or not isinstance(model_risk, Mapping)
        or not isinstance(model_calibration, Mapping)
    ):
        raise V022PredictionCapsuleError("Model substates are not objects")

    if (
        center["state_kind"] != "center_development"
        or center["center_state_sha256"] != hashes["center"]
        or center["center_beta"] != model_center["beta"]
        or center["development_cluster_count"] != 600
        or center["forecast_horizon_count"] != len(FORECAST_DAYS)
        or center["ridge_penalty"] != 0.01
        or center["completeness_rule"]
        != "exactly_600_complete_rows_all_8_horizons_finite"
        or center["input_byte_hashes"] != center_inputs
    ):
        raise V022PredictionCapsuleError("Center checkpoint semantics changed")
    if (
        risk["state_kind"] != "risk_development"
        or risk["center_checkpoint_byte_sha256"]
        != _sha256(raw_by_name["center_state_checkpoint.json"])
        or risk["training_manifest_byte_sha256"]
        != _sha256(raw_by_name["training_manifest.json"])
        or risk["risk_state_sha256"] != hashes["risk"]
        or risk["development_cluster_count"] != model_risk["development_cluster_count"]
        or risk["eligible_cluster_count"] != model_risk["eligible_cluster_count"]
        or risk["positive_label_count"] != model_risk["positive_label_count"]
        or risk["negative_label_count"] != model_risk["negative_label_count"]
        or risk["input_byte_hashes"] != risk_inputs
    ):
        raise V022PredictionCapsuleError("Risk checkpoint semantics changed")
    if (
        training["center_development_input_hashes"] != center_inputs
        or training["risk_development_input_hashes"] != risk_inputs
        or training["opened_truth_files"]
        != [
            "center_development_truth.csv",
            "risk_development_truth.csv",
        ]
        or training["forbidden_v1_evidence_matches"] != []
        or training["center_state_sha256"] != hashes["center"]
        or training["risk_state_sha256"] != hashes["risk"]
    ):
        raise V022PredictionCapsuleError("Training manifest semantics changed")
    selected = model_calibration["selected_mean_baseline"]
    if (
        calibration["calibration_input_hashes"] != calibration_inputs
        or calibration["opened_truth_files"]
        != [
            "calibration_truth.csv",
            "center_development_truth.csv",
            "risk_development_truth.csv",
        ]
        or calibration["isotonic_state_sha256"] != hashes["isotonic"]
        or calibration["conformal_state_sha256"] != hashes["conformal"]
        or calibration["selected_mean_baseline"] != selected
        or calibration_inputs.get("calibration_mask_commitment.json")
        != _sha256(raw_by_name["calibration_mask_commitment.json"])
    ):
        raise V022PredictionCapsuleError("Calibration manifest semantics changed")

    count_fields = (
        "source_calibration_count",
        "risk_isotonic_eligible_count",
        "risk_isotonic_ineligible_zero_family_count",
        "risk_isotonic_ineligible_one_family_count",
        "risk_isotonic_ineligible_other_count",
        "risk_isotonic_positive_label_count",
        "risk_isotonic_negative_label_count",
        "mean_baseline_count",
        "conformal_calibration_count",
        "conformal_order_statistic_index",
    )
    counts = {
        name: _integer(audit[name], context=f"audit {name}") for name in count_fields
    }
    ineligible_zero = sum(
        not is_eligible and count == 0
        for is_eligible, count in zip(
            mask.eligible,
            mask.family_counts,
            strict=True,
        )
    )
    ineligible_one = sum(
        not is_eligible and count == 1
        for is_eligible, count in zip(
            mask.eligible,
            mask.family_counts,
            strict=True,
        )
    )
    ineligible_other = sum(
        not is_eligible and count >= 2
        for is_eligible, count in zip(
            mask.eligible,
            mask.family_counts,
            strict=True,
        )
    )
    eligible_count = sum(mask.eligible)
    audit_ids = audit["eligibility_mask_cluster_ids"]
    audit_mask = audit["eligibility_mask"]
    if (
        audit["schema_version"] != "1.0.0"
        or counts["source_calibration_count"] != 900
        or counts["risk_isotonic_eligible_count"] != eligible_count
        or counts["risk_isotonic_ineligible_zero_family_count"] != ineligible_zero
        or counts["risk_isotonic_ineligible_one_family_count"] != ineligible_one
        or counts["risk_isotonic_ineligible_other_count"] != ineligible_other
        or counts["risk_isotonic_positive_label_count"]
        + counts["risk_isotonic_negative_label_count"]
        != eligible_count
        or counts["risk_isotonic_positive_label_count"] < 60
        or counts["risk_isotonic_negative_label_count"] < 60
        or counts["mean_baseline_count"] != 900
        or counts["conformal_calibration_count"] != 900
        or counts["conformal_order_statistic_index"] != 811
        or not isinstance(audit_ids, list)
        or tuple(audit_ids) != mask.cluster_ids
        or not isinstance(audit_mask, list)
        or any(type(value) is not bool for value in audit_mask)
        or tuple(audit_mask) != mask.eligible
        or audit["eligibility_mask_sha256"] != mask.eligibility_mask_sha256
        or audit["calibration_mask_commitment_byte_sha256"]
        != _sha256(raw_by_name["calibration_mask_commitment.json"])
        or audit["isotonic_state_sha256"] != hashes["isotonic"]
        or audit["conformal_state_sha256"] != hashes["conformal"]
        or audit["selected_mean_baseline"] != selected
    ):
        raise V022PredictionCapsuleError(
            "Calibration population audit semantics changed"
        )


def _is_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise V022PredictionCapsuleError(
            f"Cannot inspect physical path: {path}"
        ) from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & flag)


def _physical_root(raw: str | Path) -> Path:
    root = Path(os.path.abspath(os.fspath(raw)))
    if not root.is_dir() or _is_reparse(root):
        raise V022PredictionCapsuleError(
            "Label-free root must be a direct physical directory"
        )
    for parent in root.parents:
        if parent.exists() and _is_reparse(parent):
            raise V022PredictionCapsuleError(
                "Label-free root traverses a reparse point"
            )
    return root


def _direct_file(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise V022PredictionCapsuleError(f"Unsafe label-free filename: {filename!r}")
    path = root / filename
    if not path.exists() or _is_reparse(path) or not path.is_file():
        raise V022PredictionCapsuleError(f"{filename} is not a direct physical file")
    return path


def _require_membership(
    root: Path,
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise V022PredictionCapsuleError(f"Cannot inspect {context} root") from exc
    observed = {entry.name for entry in entries}
    if observed != set(expected):
        raise V022PredictionCapsuleError(
            f"{context} root membership changed: "
            f"missing={sorted(set(expected) - observed)}, "
            f"unexpected={sorted(observed - set(expected))}"
        )
    for entry in entries:
        if _is_reparse(entry) or not entry.is_file():
            raise V022PredictionCapsuleError(
                f"{context} contains a nonphysical artifact"
            )


def _strict_boolean_columns(
    frame: pd.DataFrame,
    *,
    schema: PredictionCsvSchema,
) -> None:
    for column in _STRICT_BOOLEAN_COLUMNS.intersection(frame.columns):
        values = frame[column]
        if values.isna().any() or any(
            not isinstance(value, (bool, np.bool_)) for value in values.tolist()
        ):
            raise V022PredictionCapsuleError(
                f"{schema.filename} column {column!r} must contain strict booleans"
            )


def _cluster_counts(frame: pd.DataFrame) -> dict[str, int]:
    clusters = frame.loc[:, ["partition", "cluster_id"]].drop_duplicates()
    reused = clusters.groupby("cluster_id", sort=False)["partition"].nunique()
    if (reused > 1).any():
        raise V022PredictionCapsuleError(
            "An opaque cluster ID is reused across partitions"
        )
    return {
        str(partition): int(count)
        for partition, count in clusters.groupby(
            "partition",
            sort=False,
        )
        .size()
        .items()
    }


def _require_group_grid(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    expected: Sequence[object],
    context: str,
) -> None:
    allowed = set(expected)
    if not set(frame[value_column]).issubset(allowed):
        raise V022PredictionCapsuleError(
            f"{context} contains a value outside the frozen grid"
        )
    sizes = frame.groupby(list(group_columns), sort=False).size()
    if (sizes != len(expected)).any():
        raise V022PredictionCapsuleError(f"{context} group membership is incomplete")


def _validate_formal_cardinality(
    frame: pd.DataFrame,
    *,
    schema: PredictionCsvSchema,
) -> None:
    if schema.required_rows is not None and len(frame) != schema.required_rows:
        raise V022PredictionCapsuleError(
            f"{schema.filename} row count is {len(frame)}, "
            f"expected {schema.required_rows}"
        )
    if {"partition", "cluster_id"}.issubset(frame.columns):
        observed = _cluster_counts(frame)
        if observed != dict(PARTITION_MEMBER_COUNTS):
            raise V022PredictionCapsuleError(
                f"{schema.filename} partition member counts changed"
            )

    if schema.filename == "prefix_pack.csv":
        _require_group_grid(
            frame,
            group_columns=("partition", "cluster_id"),
            value_column="prefix_day",
            expected=PREFIX_DAYS,
            context=schema.filename,
        )
    elif schema.filename in {
        "forecast_coordinates.csv",
        "prediction_bundle.csv",
    }:
        _require_group_grid(
            frame,
            group_columns=("partition", "cluster_id"),
            value_column="forecast_day",
            expected=FORECAST_DAYS,
            context=schema.filename,
        )
    elif schema.filename == "member_fit_diagnostics.csv":
        keys = set(
            frame.loc[:, ["model_id", "variant_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        sizes = frame.groupby(
            ["partition", "cluster_id"],
            sort=False,
        ).size()
        if keys != FROZEN_VARIANT_KEY_SET or (sizes != 86).any():
            raise V022PredictionCapsuleError(
                "member_fit_diagnostics.csv exact variant registry changed"
            )
    elif schema.filename == "member_forecast_bundle.csv":
        variant_keys = set(
            frame.loc[:, ["model_id", "variant_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        cluster_sizes = frame.groupby(
            ["partition", "cluster_id"],
            sort=False,
        ).size()
        if (
            variant_keys != FROZEN_VARIANT_KEY_SET
            or (cluster_sizes != 86 * len(FORECAST_DAYS)).any()
        ):
            raise V022PredictionCapsuleError(
                "member_forecast_bundle.csv per-cluster exact variant registry changed"
            )
        _require_group_grid(
            frame,
            group_columns=(
                "partition",
                "cluster_id",
                "model_id",
                "variant_id",
            ),
            value_column="forecast_day",
            expected=FORECAST_DAYS,
            context=schema.filename,
        )
    elif schema.required_value_column is not None:
        _require_group_grid(
            frame,
            group_columns=("partition", "cluster_id"),
            value_column=schema.required_value_column,
            expected=schema.required_values,
            context=schema.filename,
        )


def canonicalize_frame(
    frame: pd.DataFrame,
    filename: str,
    *,
    formal: bool = True,
) -> pd.DataFrame:
    """Validate and stable-key-sort one prediction-capsule dataframe."""

    try:
        schema = _SCHEMAS[filename]
    except KeyError as exc:
        raise V022PredictionCapsuleError(
            f"Unknown prediction artifact: {filename}"
        ) from exc
    if type(frame) is not pd.DataFrame:
        raise V022PredictionCapsuleError(f"{filename} must be an exact dataframe")
    if tuple(frame.columns) != schema.columns:
        raise V022PredictionCapsuleError(
            f"{filename} columns differ from the frozen allowlist"
        )
    if frame.empty:
        raise V022PredictionCapsuleError(f"{filename} cannot be empty")

    for column in schema.key:
        values = frame[column]
        if values.isna().any():
            raise V022PredictionCapsuleError(
                f"{filename} key column {column!r} contains NA"
            )
        if column in _STRING_KEY_COLUMNS:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in values.tolist()
            ):
                raise V022PredictionCapsuleError(
                    f"{filename} string key column {column!r} is invalid"
                )
        elif column in {"prefix_day", "forecast_day"}:
            try:
                numeric = values.to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise V022PredictionCapsuleError(
                    f"{filename} numeric key column {column!r} is invalid"
                ) from exc
            if not np.isfinite(numeric).all():
                raise V022PredictionCapsuleError(
                    f"{filename} numeric key column {column!r} is nonfinite"
                )
    if frame.duplicated(list(schema.key)).any():
        raise V022PredictionCapsuleError(f"{filename} contains duplicate key rows")
    _strict_boolean_columns(frame, schema=schema)

    if "protocol_id" in frame.columns:
        values = frame["protocol_id"].tolist()
        if any(type(value) is not str for value in values) or set(values) != {
            V022_PROTOCOL_ID
        }:
            raise V022PredictionCapsuleError(
                f"{filename} contains a non-frozen protocol_id"
            )
    if "partition" in frame.columns:
        values = frame["partition"].tolist()
        if any(type(value) is not str for value in values) or not set(values).issubset(
            set(PARTITIONS)
        ):
            raise V022PredictionCapsuleError(
                f"{filename} contains an unknown partition"
            )

    key_index = pd.MultiIndex.from_frame(
        frame.loc[:, list(schema.key)],
        names=list(schema.key),
    )
    if (
        key_index.is_monotonic_increasing
        and isinstance(frame.index, pd.RangeIndex)
        and frame.index.start == 0
        and frame.index.step == 1
    ):
        ordered = frame.copy(deep=False)
    else:
        ordered = frame.sort_values(
            list(schema.key),
            kind="stable",
        ).reset_index(drop=True)
    if formal:
        _validate_formal_cardinality(ordered, schema=schema)
    return ordered


def canonical_csv_bytes(
    frame: pd.DataFrame,
    filename: str,
    *,
    formal: bool = True,
) -> bytes:
    ordered = canonicalize_frame(frame, filename, formal=formal)
    try:
        return ordered.to_csv(
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V022PredictionCapsuleError(
            f"{filename} cannot be serialized canonically"
        ) from exc


def read_canonical_csv(
    path: Path,
    *,
    formal: bool = True,
) -> pd.DataFrame:
    raw = path.read_bytes()
    try:
        frame = pd.read_csv(
            io.BytesIO(raw),
            encoding="utf-8",
            encoding_errors="strict",
            float_precision="round_trip",
        )
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise V022PredictionCapsuleError(
            f"{path.name} is not strict canonical CSV"
        ) from exc
    ordered = canonicalize_frame(frame, path.name, formal=formal)
    if canonical_csv_bytes(ordered, path.name, formal=formal) != raw:
        raise V022PredictionCapsuleError(
            f"{path.name} is not the frozen canonical CSV serialization"
        )
    return ordered


def _identity_json(
    payload: Mapping[str, Any],
    *,
    filename: str,
    config_sha256: str,
) -> None:
    if (
        payload.get("protocol_id") != V022_PROTOCOL_ID
        or payload.get("config_sha256") != config_sha256
    ):
        raise V022PredictionCapsuleError(f"{filename} identity changed")


def _require_phase_hash(
    progress: AttemptProgress,
    *,
    field: str,
    raw: bytes,
    context: str,
) -> None:
    expected = getattr(progress, field)
    if expected is None or expected != _sha256(raw):
        raise V022PredictionCapsuleError(
            f"{context} differs from its ledger commitment"
        )


def _load_prediction_progress(
    root: Path,
    *,
    attempt_id: str,
    expected_config_sha256: str,
    expected_git_commit: str,
) -> tuple[AttemptProgress, bytes]:
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise V022PredictionCapsuleError("attempt_id is invalid")
    if (
        not isinstance(expected_git_commit, str)
        or _GIT_COMMIT.fullmatch(expected_git_commit) is None
    ):
        raise V022PredictionCapsuleError("Expected git commit is invalid")
    _digest(expected_config_sha256, context="Expected config hash")
    try:
        _, states, raw = read_exposure_log(
            _direct_file(root, "exposure_log.jsonl"),
            expected_config_sha256=expected_config_sha256,
            sealed_filenames=_SEALED_FILENAMES,
        )
    except V022LedgerError as exc:
        raise V022PredictionCapsuleError(str(exc)) from exc
    try:
        progress = states[attempt_id]
    except KeyError as exc:
        raise V022PredictionCapsuleError(
            "Attempt is absent from the canonical ledger"
        ) from exc
    if (
        progress.identity.attempt_id != attempt_id
        or progress.identity.git_commit != expected_git_commit
        or progress.identity.config_byte_sha256 != expected_config_sha256
        or progress.terminal_failed
        or progress.opened_truth_files
        != (
            "calibration_truth.csv",
            "center_development_truth.csv",
            "risk_development_truth.csv",
        )
        or progress.completed_phase != "model_state_committed"
        or progress.pending_phase != "prediction_started"
    ):
        raise V022PredictionCapsuleError(
            "Attempt is not at the sole label-free prediction boundary"
        )
    return progress, raw


def _verify_truth_commitment(
    raw: bytes,
    *,
    progress: AttemptProgress,
    config_sha256: str,
) -> dict[str, str]:
    _require_phase_hash(
        progress,
        field="truth_commitments_byte_sha256",
        raw=raw,
        context="Truth commitment",
    )
    payload = _strict_json(raw, filename="truth_commitments.json")
    if set(payload) != {
        "protocol_id",
        "config_sha256",
        "files",
        "created_utc",
        "truth_values_withheld_by_physical_path",
    }:
        raise V022PredictionCapsuleError("truth_commitments.json schema changed")
    _identity_json(
        payload,
        filename="truth_commitments.json",
        config_sha256=config_sha256,
    )
    _utc(payload["created_utc"], context="truth commitment created_utc")
    if payload["truth_values_withheld_by_physical_path"] is not True:
        raise V022PredictionCapsuleError(
            "Truth commitment does not attest physical withholding"
        )
    entries = payload["files"]
    if not isinstance(entries, list) or len(entries) != len(_SEALED_FILENAMES):
        raise V022PredictionCapsuleError("Truth commitment file registry changed")
    result: dict[str, str] = {}
    for filename, entry in zip(_SEALED_FILENAMES, entries, strict=True):
        if (
            not isinstance(entry, Mapping)
            or set(entry) != _FILE_ENTRY_KEYS
            or entry.get("path") != filename
            or type(entry.get("row_count")) is not int
            or entry.get("row_count") != _TRUTH_REQUIRED_ROWS[filename]
            or type(entry.get("byte_count")) is not int
            or entry["byte_count"] < 1
        ):
            raise V022PredictionCapsuleError(
                f"Truth commitment metadata changed for {filename}"
            )
        result[filename] = _digest(
            entry.get("byte_sha256"),
            context=f"truth commitment hash/{filename}",
        )
    return result


def _verify_file_commitment(
    root: Path,
    *,
    raw: bytes,
    filename: str,
    expected_keys: frozenset[str],
    expected_files: Sequence[str],
    progress: AttemptProgress,
    progress_field: str,
    config_sha256: str,
    git_commit: str,
    frames: Mapping[str, pd.DataFrame],
    require_worker_count: bool,
) -> None:
    _require_phase_hash(
        progress,
        field=progress_field,
        raw=raw,
        context=filename,
    )
    payload = _strict_json(raw, filename=filename)
    if (
        set(payload) != expected_keys
        or payload.get("git_commit") != git_commit
        or not isinstance(payload.get("files"), list)
    ):
        raise V022PredictionCapsuleError(f"{filename} semantics changed")
    if require_worker_count and payload.get("worker_count") != 6:
        raise V022PredictionCapsuleError("fit_commitment.json worker count changed")
    _identity_json(
        payload,
        filename=filename,
        config_sha256=config_sha256,
    )
    _utc(payload["created_utc"], context=f"{filename} created_utc")
    entries = payload["files"]
    if len(entries) != len(expected_files):
        raise V022PredictionCapsuleError(f"{filename} file registry changed")
    for expected_name, entry in zip(expected_files, entries, strict=True):
        if not isinstance(entry, Mapping) or set(entry) != _FILE_ENTRY_KEYS:
            raise V022PredictionCapsuleError(
                f"{filename} contains an invalid file entry"
            )
        artifact = _direct_file(root, expected_name).read_bytes()
        expected_rows = (
            len(frames[expected_name]) if expected_name.endswith(".csv") else 1
        )
        if (
            entry.get("path") != expected_name
            or type(entry.get("row_count")) is not int
            or entry.get("row_count") != expected_rows
            or type(entry.get("byte_count")) is not int
            or entry.get("byte_count") != len(artifact)
            or entry.get("byte_sha256") != _sha256(artifact)
        ):
            raise V022PredictionCapsuleError(
                f"{filename} does not bind {expected_name}"
            )


def _verify_model_input_hashes(
    decoded: DecodedPredictionState,
    *,
    local_hashes: Mapping[str, str],
    truth_hashes: Mapping[str, str],
) -> None:
    available = {**local_hashes, **truth_hashes}
    for phase, hashes in decoded.input_byte_hashes.items():
        for filename, digest in hashes.items():
            if Path(filename).name != filename or filename not in available:
                raise V022PredictionCapsuleError(
                    f"{phase} model input is outside committed artifacts"
                )
            if available[filename] != digest:
                raise V022PredictionCapsuleError(
                    f"{phase} model input changed: {filename}"
                )


def load_prediction_bundle(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    expected_config_sha256: str,
    expected_git_commit: str,
) -> PredictionBundle:
    """Load the exact pre-prediction capability without a truth path."""

    root = _physical_root(label_free_root)
    _require_membership(
        root,
        _PRE_PREDICTION_FILES,
        context="Pre-prediction",
    )
    progress, ledger_raw = _load_prediction_progress(
        root,
        attempt_id=attempt_id,
        expected_config_sha256=expected_config_sha256,
        expected_git_commit=expected_git_commit,
    )
    verified_hashes: dict[str, str] = {"exposure_log.jsonl": _sha256(ledger_raw)}

    plan_raw = _direct_file(
        root,
        "generation_plan_commitment.json",
    ).read_bytes()
    _strict_json(
        plan_raw,
        filename="generation_plan_commitment.json",
        compact=True,
    )
    _require_phase_hash(
        progress,
        field="generation_plan_commitment_byte_sha256",
        raw=plan_raw,
        context="Generation plan",
    )
    verified_hashes["generation_plan_commitment.json"] = _sha256(plan_raw)

    truth_raw = _direct_file(root, "truth_commitments.json").read_bytes()
    truth_hashes = _verify_truth_commitment(
        truth_raw,
        progress=progress,
        config_sha256=expected_config_sha256,
    )
    verified_hashes["truth_commitments.json"] = _sha256(truth_raw)

    actual_raw = _direct_file(
        root,
        "actual_analysis_hash_ledger_commitment.json",
    ).read_bytes()
    _strict_json(
        actual_raw,
        filename="actual_analysis_hash_ledger_commitment.json",
        compact=True,
    )
    _require_phase_hash(
        progress,
        field="actual_analysis_hash_ledger_commitment_byte_sha256",
        raw=actual_raw,
        context="Actual-analysis hash ledger",
    )
    verified_hashes["actual_analysis_hash_ledger_commitment.json"] = _sha256(actual_raw)

    frames: dict[str, pd.DataFrame] = {}
    for filename in (*_LABEL_INPUTS, *_FIT_OUTPUTS):
        frame = read_canonical_csv(_direct_file(root, filename), formal=True)
        frames[filename] = frame
        verified_hashes[filename] = _sha256(
            canonical_csv_bytes(frame, filename, formal=True)
        )
    _validate_prediction_input_alignment(frames)

    fit_raw = _direct_file(root, "fit_commitment.json").read_bytes()
    _verify_file_commitment(
        root,
        raw=fit_raw,
        filename="fit_commitment.json",
        expected_keys=_FIT_COMMITMENT_KEYS,
        expected_files=_FIT_COMMITMENT_FILES,
        progress=progress,
        progress_field="fit_commitment_byte_sha256",
        config_sha256=expected_config_sha256,
        git_commit=expected_git_commit,
        frames=frames,
        require_worker_count=True,
    )
    verified_hashes["fit_commitment.json"] = _sha256(fit_raw)

    state_json_names = (
        "center_state_checkpoint.json",
        "risk_state_checkpoint.json",
        "training_manifest.json",
        "calibration_manifest.json",
        "calibration_population_audit.json",
        "model_state.json",
    )
    state_raw: dict[str, bytes] = {}
    state_payloads: dict[str, Mapping[str, Any]] = {}
    for filename in state_json_names:
        raw = _direct_file(root, filename).read_bytes()
        payload = _strict_json(raw, filename=filename)
        _identity_json(
            payload,
            filename=filename,
            config_sha256=expected_config_sha256,
        )
        state_raw[filename] = raw
        state_payloads[filename] = payload
        verified_hashes[filename] = _sha256(raw)

    mask_filename = "calibration_mask_commitment.json"
    mask_raw = _direct_file(root, mask_filename).read_bytes()
    mask = _decode_mask_commitment(mask_raw)
    state_raw[mask_filename] = mask_raw
    verified_hashes[mask_filename] = _sha256(mask_raw)

    for filename, field in (
        (
            "center_state_checkpoint.json",
            "center_state_checkpoint_byte_sha256",
        ),
        (
            "risk_state_checkpoint.json",
            "risk_state_checkpoint_byte_sha256",
        ),
        (
            "calibration_mask_commitment.json",
            "calibration_mask_commitment_byte_sha256",
        ),
    ):
        _require_phase_hash(
            progress,
            field=field,
            raw=state_raw[filename],
            context=filename,
        )

    decoded = decode_prediction_state(
        state_raw["model_state.json"],
        expected_config_sha256=expected_config_sha256,
    )
    _verify_training_chain_semantics(
        payloads=state_payloads,
        raw_by_name=state_raw,
        decoded=decoded,
        mask=mask,
        config_sha256=expected_config_sha256,
    )
    model_commitment_raw = _direct_file(
        root,
        "model_state_commitment.json",
    ).read_bytes()
    _verify_file_commitment(
        root,
        raw=model_commitment_raw,
        filename="model_state_commitment.json",
        expected_keys=_MODEL_COMMITMENT_KEYS,
        expected_files=_MODEL_STATE_COMMITMENT_FILES,
        progress=progress,
        progress_field="model_state_commitment_byte_sha256",
        config_sha256=expected_config_sha256,
        git_commit=expected_git_commit,
        frames=frames,
        require_worker_count=False,
    )
    verified_hashes["model_state_commitment.json"] = _sha256(model_commitment_raw)
    if decoded.model_state_byte_sha256 != verified_hashes["model_state.json"]:
        raise V022PredictionCapsuleError(
            "Decoded model state differs from its committed bytes"
        )
    _verify_model_input_hashes(
        decoded,
        local_hashes=verified_hashes,
        truth_hashes=truth_hashes,
    )

    bundle = PredictionBundle(
        _seal=_BUNDLE_SEAL,
        root=root,
        config_sha256=expected_config_sha256,
        git_commit=expected_git_commit,
        frames=frames,
        state=decoded.state,
        file_hashes=verified_hashes,
        ledger_raw=ledger_raw,
    )
    return _require_bundle_unchanged(bundle)


def _require_bundle_unchanged(
    value: object,
    *,
    allowed_outputs: frozenset[str] = frozenset(),
) -> PredictionBundle:
    if (
        type(value) is not PredictionBundle
        or value._seal is not _BUNDLE_SEAL
        or not allowed_outputs.issubset(set(_PREDICTION_OUTPUTS))
        or not isinstance(value._config_sha256, str)
        or _SHA256.fullmatch(value._config_sha256) is None
        or not isinstance(value._git_commit, str)
        or _GIT_COMMIT.fullmatch(value._git_commit) is None
    ):
        raise V022PredictionCapsuleError("Prediction bundle capability is invalid")
    expected_membership = frozenset({*_PRE_PREDICTION_FILES, *allowed_outputs})
    _require_membership(
        value._root,
        expected_membership,
        context="Committed prediction operation",
    )
    file_hashes = dict(value._file_hashes)
    if set(file_hashes) != set(_PRE_PREDICTION_FILES):
        raise V022PredictionCapsuleError("Committed file-hash registry changed")
    for filename, digest in file_hashes.items():
        if _sha256(_direct_file(value._root, filename).read_bytes()) != digest:
            raise V022PredictionCapsuleError("A committed label-free artifact changed")
    if (
        _direct_file(
            value._root,
            "exposure_log.jsonl",
        ).read_bytes()
        != value._ledger_raw
    ):
        raise V022PredictionCapsuleError("Committed prediction ledger changed")
    decoded = decode_prediction_state(
        _direct_file(value._root, "model_state.json").read_bytes(),
        expected_config_sha256=value._config_sha256,
    )
    if decoded.state != value._state:
        raise V022PredictionCapsuleError("Sealed in-memory prediction state changed")
    frames = tuple(value._frames)
    if tuple(name for name, _ in frames) != (
        *_LABEL_INPUTS,
        *_FIT_OUTPUTS,
    ):
        raise V022PredictionCapsuleError("Sealed in-memory frame registry changed")
    for filename, frame in frames:
        if type(frame) is not pd.DataFrame:
            raise V022PredictionCapsuleError(f"Sealed frame type changed: {filename}")
        raw = canonical_csv_bytes(frame, filename, formal=True)
        if file_hashes[filename] != _sha256(raw):
            raise V022PredictionCapsuleError(
                f"Sealed in-memory frame changed: {filename}"
            )
    return value


def _exclusive_create(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise V022PredictionCapsuleError(
            f"Prediction artifact already exists: {path.name}"
        ) from exc
    try:
        written = os.write(descriptor, raw)
        if written != len(raw):
            raise OSError("exclusive write was incomplete")
        os.fsync(descriptor)
    except OSError as exc:
        raise V022PredictionCapsuleError(
            f"Could not write {path.name} exclusively"
        ) from exc
    finally:
        os.close(descriptor)
    if path.read_bytes() != raw:
        raise V022PredictionCapsuleError(
            f"{path.name} changed after its exclusive write"
        )


def _bound_prediction_output_bytes(
    bundle: PredictionBundle,
    result: PredictionPipelineResult,
) -> dict[str, bytes]:
    binding = _PIPELINE_RESULT_BINDINGS.get(id(result))
    if (
        type(result) is not PredictionPipelineResult
        or binding is None
        or binding[0] is not result
        or binding[1] is not bundle
    ):
        raise V022PredictionCapsuleError(
            "Prediction outputs were not issued for this exact sealed bundle"
        )
    frames = {
        filename: getattr(result, attribute)
        for filename, attribute in _PIPELINE_OUTPUT_FIELDS
    }
    raw_by_name = {
        filename: canonical_csv_bytes(
            frame,
            filename,
            formal=True,
        )
        for filename, frame in frames.items()
    }
    observed_hashes = tuple(
        (filename, _sha256(raw_by_name[filename]))
        for filename, _ in _PIPELINE_OUTPUT_FIELDS
    )
    if observed_hashes != binding[2]:
        raise V022PredictionCapsuleError(
            "A sealed prediction result changed after capsule execution"
        )
    return raw_by_name


def write_prediction_outputs(
    bundle: PredictionBundle,
    *,
    result: PredictionPipelineResult,
) -> tuple[PredictionArtifactMetadata, ...]:
    value = _require_bundle_unchanged(bundle)
    raw_by_name = _bound_prediction_output_bytes(value, result)
    _require_bundle_unchanged(value)
    metadata: list[PredictionArtifactMetadata] = []
    written: set[str] = set()
    written_hashes: dict[str, str] = {}
    for filename in _PREDICTION_OUTPUTS:
        raw = raw_by_name[filename]
        _exclusive_create(value._root / filename, raw)
        written.add(filename)
        written_hashes[filename] = _sha256(raw)
        _require_bundle_unchanged(
            value,
            allowed_outputs=frozenset(written),
        )
        for written_name, digest in written_hashes.items():
            if _sha256(_direct_file(value._root, written_name).read_bytes()) != digest:
                raise V022PredictionCapsuleError(
                    "A written prediction artifact changed"
                )
        metadata.append(
            PredictionArtifactMetadata(
                path=filename,
                row_count=len(
                    getattr(
                        result,
                        dict(_PIPELINE_OUTPUT_FIELDS)[filename],
                    )
                ),
                byte_count=len(raw),
                byte_sha256=_sha256(raw),
            )
        )
    _PIPELINE_RESULT_BINDINGS.pop(id(result), None)
    return tuple(metadata)


@dataclass(frozen=True, slots=True)
class _PrimaryArmRanking:
    prefix_only_ranks: tuple[int | None, ...]
    visible_stress_ranks: tuple[int | None, ...]
    prefix_only_issued: tuple[bool, ...]
    visible_stress_issued: tuple[bool, ...]


def _strict_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise V022PredictionCapsuleError(f"{context} must be a strict boolean")
    return bool(value)


def _require_exact_grid(
    frame: pd.DataFrame,
    *,
    value_column: str,
    expected: Sequence[float],
    context: str,
) -> pd.DataFrame:
    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    ).to_numpy(float)
    if (
        len(values) != len(expected)
        or not np.isfinite(values).all()
        or tuple(sorted(float(value) for value in values)) != tuple(expected)
    ):
        raise V022PredictionCapsuleError(f"{context} does not contain the frozen grid")
    return frame.sort_values(value_column, kind="stable").reset_index(drop=True)


def _parse_parameter_payload(value: object) -> tuple[dict[str, float], str]:
    if not isinstance(value, str):
        raise V022PredictionCapsuleError(
            "parameters_json must be a canonical JSON object string"
        )
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise V022PredictionCapsuleError("parameters_json is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise V022PredictionCapsuleError("parameters_json must encode a JSON object")
    parameters: dict[str, float] = {}
    for name, raw in payload.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(
                raw,
                (bool, np.bool_),
            )
        ):
            raise V022PredictionCapsuleError(
                "parameters_json contains an invalid field"
            )
        parameters[name] = _real(
            raw,
            context=f"parameters_json/{name}",
        )
    try:
        canonical = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V022PredictionCapsuleError("parameters_json is not finite") from exc
    if value != canonical:
        raise V022PredictionCapsuleError(
            "parameters_json is not in frozen canonical form"
        )
    return parameters, canonical


def _parameter_boundary_fraction(
    model_id: str,
    variant_id: str,
    parameters: Mapping[str, float],
) -> float:
    if (model_id, variant_id) not in FROZEN_VARIANT_KEY_SET:
        raise V022PredictionCapsuleError(
            "Variant key is outside the frozen exact registry"
        )
    if model_id == "target_prefix_persistence":
        if set(parameters) != {"last_retention_pct"}:
            raise V022PredictionCapsuleError(
                "Persistence parameters differ from the freeze"
            )
        return 0.0
    try:
        bounds = _PARAMETER_BOUNDS[model_id]
    except KeyError as exc:
        raise V022PredictionCapsuleError(
            f"Undeclared candidate model: {model_id}"
        ) from exc
    expected_names = {name for name, _, _ in bounds}
    fixed: Mapping[str, float] = {}
    if model_id == "target_prefix_late_knee_prior_grid":
        try:
            fixed = _LATE_KNEE_FIXED_BY_VARIANT[variant_id]
        except KeyError as exc:
            raise V022PredictionCapsuleError(
                "Late-knee variant identity changed"
            ) from exc
        expected_names.update(fixed)
    if set(parameters) != expected_names:
        raise V022PredictionCapsuleError(
            f"{model_id} parameters_json keys differ from the freeze"
        )
    for name, expected in fixed.items():
        if parameters[name] != expected:
            raise V022PredictionCapsuleError(
                "Late-knee fixed parameters differ from variant_id"
            )

    boundary_hits = 0
    for name, lower, upper in bounds:
        value = parameters[name]
        if not lower <= value <= upper:
            raise V022PredictionCapsuleError(
                f"{model_id} parameter {name} lies outside frozen bounds"
            )
        tolerance = 1e-6 * max(1.0, upper - lower)
        if min(value - lower, upper - value) <= tolerance:
            boundary_hits += 1
    if model_id == "target_prefix_dual_power" and parameters["b1"] > parameters["b2"]:
        raise V022PredictionCapsuleError(
            "Dual-power identifiability constraint is violated"
        )
    return boundary_hits / len(bounds)


def _evaluate_frozen_variant(
    model_id: str,
    variant_id: str,
    parameters: Mapping[str, float],
    elapsed_days: np.ndarray,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    years = elapsed / 365.25
    if model_id == "target_prefix_persistence":
        return np.full_like(
            elapsed,
            parameters["last_retention_pct"],
            dtype=float,
        )
    if model_id == "target_prefix_sqrt_time":
        return 100.0 - parameters["c"] * np.sqrt(years)
    if model_id == "target_prefix_bounded_power_law":
        return 100.0 - parameters["a"] * np.power(
            years,
            parameters["b"],
        )
    if model_id == "target_prefix_saturating_plus_slow":
        saturated = parameters["a_sat"] * (
            1.0
            - np.exp(
                -np.power(
                    elapsed / parameters["tau_sat_days"],
                    parameters["b_sat"],
                )
            )
        )
        slow = parameters["a_slow"] * np.power(
            years,
            parameters["b_slow"],
        )
        return 100.0 - saturated - slow
    if model_id == "target_prefix_dual_power":
        loss = parameters["a1"] * np.power(years, parameters["b1"])
        loss += parameters["a2"] * np.power(years, parameters["b2"])
        return 100.0 - loss
    if model_id == "target_prefix_late_knee_prior_grid":
        fixed = _LATE_KNEE_FIXED_BY_VARIANT[variant_id]
        base = parameters["a"] * np.power(years, parameters["b"])
        knee = (
            fixed["k_pp_per_day"]
            * fixed["w_days"]
            * (
                np.logaddexp(
                    0.0,
                    (elapsed - fixed["t_knee_days"]) / fixed["w_days"],
                )
                - np.logaddexp(
                    0.0,
                    -fixed["t_knee_days"] / fixed["w_days"],
                )
            )
        )
        return 100.0 - base - knee
    if model_id == "target_prefix_early_activation_plus_power":
        base = parameters["a"] * np.power(years, parameters["b"])
        activation = (
            parameters["activation_amplitude_pp"]
            * (1.0 - np.exp(-elapsed / parameters["tau_rise_days"]))
            * np.exp(-elapsed / parameters["tau_decay_days"])
        )
        return 100.0 - base + activation
    raise V022PredictionCapsuleError(f"Unknown candidate structure: {model_id}")


def _verify_variant_commitment(
    *,
    model_id: str,
    variant_id: str,
    parameters: Mapping[str, float],
    prefix_days: np.ndarray,
    observed: np.ndarray,
    forecast_days: np.ndarray,
    committed_rmse: object,
    committed_residual: object,
    committed_boundary: object,
    committed_forecast: np.ndarray,
) -> float:
    boundary = _parameter_boundary_fraction(
        model_id,
        variant_id,
        parameters,
    )
    if model_id == "target_prefix_persistence" and (
        np.float64(parameters["last_retention_pct"]).tobytes()
        != np.float64(observed[-1]).tobytes()
    ):
        raise V022PredictionCapsuleError(
            "Persistence parameter differs from the last prefix observation"
        )
    try:
        prefix_prediction = _evaluate_frozen_variant(
            model_id,
            variant_id,
            parameters,
            prefix_days,
        )
        forecast = _evaluate_frozen_variant(
            model_id,
            variant_id,
            parameters,
            forecast_days,
        )
    except (
        FloatingPointError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise V022PredictionCapsuleError(
            "Frozen variant formula evaluation failed"
        ) from exc
    residual = prefix_prediction - observed
    if (
        prefix_prediction.shape != observed.shape
        or forecast.shape != forecast_days.shape
        or committed_forecast.shape != forecast_days.shape
        or not np.isfinite(prefix_prediction).all()
        or not np.isfinite(forecast).all()
        or not np.isfinite(residual).all()
    ):
        raise V022PredictionCapsuleError(
            "Frozen variant formula produced invalid values"
        )
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    max_residual = float(np.max(np.abs(residual)))
    comparisons = (
        (
            _finite_number(committed_rmse, context="prefix_rmse_pp"),
            rmse,
            "prefix_rmse_pp",
        ),
        (
            _finite_number(
                committed_residual,
                context="prefix_max_abs_residual_pp",
            ),
            max_residual,
            "prefix_max_abs_residual_pp",
        ),
        (
            _finite_number(
                committed_boundary,
                context="parameter_boundary_hit_fraction",
            ),
            boundary,
            "parameter_boundary_hit_fraction",
        ),
    )
    for committed, recomputed, context in comparisons:
        if not math.isclose(
            committed,
            recomputed,
            rel_tol=0.0,
            abs_tol=_FORMULA_ABSOLUTE_TOLERANCE,
        ):
            raise V022PredictionCapsuleError(
                f"{context} differs from frozen formula recomputation"
            )
    if not np.allclose(
        committed_forecast,
        forecast,
        rtol=0.0,
        atol=_FORMULA_ABSOLUTE_TOLERANCE,
        equal_nan=False,
    ):
        raise V022PredictionCapsuleError(
            "raw_forecast_retention_pct differs from frozen formula recomputation"
        )
    return boundary


def _recomputed_credible(
    diagnostic: pd.Series,
    raw_forecast: np.ndarray,
    *,
    parameters: Mapping[str, float],
) -> bool:
    status = str(diagnostic["fit_status"])
    if status not in {"succeeded", "failed"}:
        raise V022PredictionCapsuleError("fit_status must be succeeded or failed")
    declared = _strict_bool(
        diagnostic["credible_variant"],
        context="credible_variant",
    )
    if status == "failed":
        if declared or parameters:
            raise V022PredictionCapsuleError(
                "A failed variant retained credible fitted state"
            )
        if not np.isnan(raw_forecast).all():
            raise V022PredictionCapsuleError(
                "A failed variant must contain only empty forecasts"
            )
        for name in (
            "prefix_rmse_pp",
            "prefix_max_abs_residual_pp",
            "parameter_boundary_hit_fraction",
        ):
            try:
                value = float(diagnostic[name])
            except (TypeError, ValueError) as exc:
                raise V022PredictionCapsuleError(
                    f"A failed variant has invalid {name}"
                ) from exc
            if not math.isnan(value):
                raise V022PredictionCapsuleError(
                    f"A failed variant must contain empty {name}"
                )
        return False

    rmse = _finite_number(
        diagnostic["prefix_rmse_pp"],
        context="prefix_rmse_pp",
    )
    residual = _finite_number(
        diagnostic["prefix_max_abs_residual_pp"],
        context="prefix_max_abs_residual_pp",
    )
    boundary = _finite_number(
        diagnostic["parameter_boundary_hit_fraction"],
        context="parameter_boundary_hit_fraction",
    )
    if not 0.0 <= boundary <= 1.0:
        raise V022PredictionCapsuleError(
            "parameter_boundary_hit_fraction is outside [0, 1]"
        )
    recomputed = bool(
        rmse <= _MAXIMUM_PREFIX_RMSE_PP
        and residual <= _MAXIMUM_PREFIX_RESIDUAL_PP
        and np.isfinite(raw_forecast).all()
        and np.all(raw_forecast >= _FORECAST_BOUNDS_PCT[0])
        and np.all(raw_forecast <= _FORECAST_BOUNDS_PCT[1])
    )
    if declared != recomputed:
        raise V022PredictionCapsuleError(
            "credible_variant differs from the frozen credibility rule"
        )
    return recomputed


def _unique_safe_variants(
    variants: Sequence[_SafeVariant],
) -> tuple[_SafeVariant, ...]:
    unique: dict[bytes, _SafeVariant] = {}
    for variant in variants:
        key = canonical_float64_vector_bytes(variant.forecast)
        if (
            not math.isfinite(variant.prefix_rmse_pp)
            or not math.isfinite(variant.boundary_fraction)
            or not 0.0 <= variant.boundary_fraction <= 1.0
        ):
            raise V022PredictionCapsuleError("Credible variant metadata is invalid")
        previous = unique.get(key)
        if previous is not None:
            previous_metadata = (
                previous.prefix_rmse_pp,
                previous.boundary_fraction,
                previous.parameter_payload,
            )
            current_metadata = (
                variant.prefix_rmse_pp,
                variant.boundary_fraction,
                variant.parameter_payload,
            )
            if previous_metadata != current_metadata:
                raise V022PredictionCapsuleError(
                    "Exact duplicate forecasts have conflicting metadata"
                )
            continue
        unique[key] = variant
    return tuple(unique[key] for key in sorted(unique))


def _safe_boundary_fraction(
    family_variants: Mapping[str, Sequence[_SafeVariant]],
) -> float:
    family_values: list[float] = []
    for family_id in sorted(family_variants):
        variants = _unique_safe_variants(family_variants[family_id])
        if not variants:
            continue
        by_signature: dict[tuple[int, ...], list[float]] = {}
        for variant in variants:
            signature = quantized_shape_signature(variant.forecast)
            by_signature.setdefault(signature, []).append(variant.boundary_fraction)
        signature_values = [
            float(np.mean(by_signature[signature]))
            for signature in sorted(by_signature)
        ]
        family_values.append(float(np.mean(signature_values)))
    if not family_values:
        raise V022PredictionCapsuleError(
            "Boundary fraction requires a successful family"
        )
    return float(np.mean(family_values))


def _value_at_day(
    days: np.ndarray,
    values: np.ndarray,
    day: float,
) -> float:
    matches = np.flatnonzero(days == day)
    if matches.size != 1:
        raise V022PredictionCapsuleError(
            f"prefix_days must contain day {day} exactly once"
        )
    return float(values[matches[0]])


def _extract_prefix_features_safe(
    *,
    prefix_days: Sequence[float],
    observed_retention_pct: Sequence[float],
    family_variants: Mapping[str, Sequence[_SafeVariant]],
    sqrt_forecast: Sequence[float],
    center_forecast: Sequence[float],
) -> tuple[float, ...]:
    days = np.asarray(prefix_days, dtype=np.float64)
    observed = np.asarray(observed_retention_pct, dtype=np.float64)
    sqrt = np.asarray(sqrt_forecast, dtype=np.float64)
    center = np.asarray(center_forecast, dtype=np.float64)
    if (
        days.shape != (len(PREFIX_DAYS),)
        or tuple(float(value) for value in days) != PREFIX_DAYS
        or observed.shape != days.shape
        or sqrt.shape != (len(FORECAST_DAYS),)
        or center.shape != (len(FORECAST_DAYS),)
        or not np.isfinite(np.concatenate((days, observed, sqrt, center))).all()
    ):
        raise V022PredictionCapsuleError("Prefix feature inputs are invalid")

    unique_by_family = {
        family_id: _unique_safe_variants(variants)
        for family_id, variants in family_variants.items()
    }
    successful = {
        family_id: variants
        for family_id, variants in unique_by_family.items()
        if variants
    }
    successful_count = len(successful)
    if successful_count == 0 or successful_count > len(DECLARED_STRUCTURE_FAMILIES):
        raise V022PredictionCapsuleError("Successful structure-family count is invalid")
    family_vectors = {
        family_id: tuple(variant.forecast for variant in variants)
        for family_id, variants in successful.items()
    }
    support_vectors, support_weights = family_balanced_support(family_vectors)
    q10 = np.asarray(
        coordinatewise_weighted_quantile(
            support_vectors,
            support_weights,
            0.10,
        )
    )
    q25 = np.asarray(
        coordinatewise_weighted_quantile(
            support_vectors,
            support_weights,
            0.25,
        )
    )
    q75 = np.asarray(
        coordinatewise_weighted_quantile(
            support_vectors,
            support_weights,
            0.75,
        )
    )
    q90 = np.asarray(
        coordinatewise_weighted_quantile(
            support_vectors,
            support_weights,
            0.90,
        )
    )
    family_representatives = [
        family_representative(family_vectors[family_id])
        for family_id in sorted(family_vectors)
    ]
    unique_family_representatives = deduplicate_vectors(family_representatives)
    effective_shape_count = len(
        {quantized_shape_signature(vector) for vector in unique_family_representatives}
    )
    best_rmse = min(
        variant.prefix_rmse_pp
        for variants in successful.values()
        for variant in variants
    )
    boundary_fraction = _safe_boundary_fraction(successful)
    leave_out_error = leave_day730_out_sqrt_error(days, observed)

    q0_value = _value_at_day(days, observed, 0.0)
    q90_value = _value_at_day(days, observed, 90.0)
    q180_value = _value_at_day(days, observed, 180.0)
    q365_value = _value_at_day(days, observed, 365.0)
    q730_value = _value_at_day(days, observed, 730.0)
    slope_180_365 = (q365_value - q180_value) / (365.0 - 180.0) * 365.25
    slope_365_730 = (q730_value - q365_value) / (730.0 - 365.0) * 365.25
    width_25y = float(q90[-1] - q10[-1])
    width_10y = float(q90[4] - q10[4])
    values = (
        float(successful_count),
        float(len(DECLARED_STRUCTURE_FAMILIES) - successful_count),
        float(best_rmse),
        width_25y,
        float(np.mean(q75 - q25)),
        float(effective_shape_count),
        boundary_fraction,
        leave_out_error,
        float(center[-1]),
        float(center[-1] - sqrt[-1]),
        q365_value - q730_value,
        q0_value - q90_value,
        slope_180_365 - slope_365_730,
        max(0.0, width_25y - width_10y),
    )
    if not np.isfinite(values).all():
        raise V022PredictionCapsuleError("Extracted prefix features must all be finite")
    return values


def _numeric_records(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_column: str | None = None,
) -> list[list[object]]:
    working = frame.loc[:, list(columns)]
    if sort_column is not None:
        working = working.sort_values(sort_column, kind="stable")
    records: list[list[object]] = []
    for row in working.itertuples(index=False, name=None):
        clean: list[object] = []
        for raw in row:
            value = raw.item() if isinstance(raw, np.generic) else raw
            if isinstance(value, float) and not math.isfinite(value):
                raise V022PredictionCapsuleError(
                    "Predictor content contains a nonfinite value"
                )
            clean.append(value)
        records.append(clean)
    return records


def predictor_content_hashes(
    prefix_rows: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    operating_row: Mapping[str, Any] | pd.Series,
) -> PredictorContentHashes:
    if (
        len(prefix_rows) != len(PREFIX_DAYS)
        or len(forecast_rows) != len(FORECAST_DAYS)
        or prefix_rows["prefix_day"].duplicated().any()
        or forecast_rows["forecast_day"].duplicated().any()
    ):
        raise V022PredictionCapsuleError(
            "Predictor content does not have frozen 12/8 coordinates"
        )
    operating = dict(operating_row)
    if set((*REAL_OPERATING_FIELDS, *PLACEBO_FIELDS)).difference(operating):
        raise V022PredictionCapsuleError("Predictor operating content is incomplete")
    base = {
        "prefix": _numeric_records(
            prefix_rows,
            ("prefix_day", "observed_retention_pct"),
            sort_column="prefix_day",
        ),
        "forecast": _numeric_records(
            forecast_rows,
            ("forecast_day",),
            sort_column="forecast_day",
        ),
    }
    arm_a_raw = canonical_json_bytes(base)
    arm_b_raw = canonical_json_bytes(
        {
            **base,
            "real_operating": [
                _json_ready(operating[name]) for name in REAL_OPERATING_FIELDS
            ],
        }
    )
    placebo_raw = canonical_json_bytes(
        {
            **base,
            "placebo_operating": [
                _json_ready(operating[name]) for name in PLACEBO_FIELDS
            ],
        }
    )
    return PredictorContentHashes(
        random_policy=_sha256(arm_a_raw),
        arm_a=_sha256(arm_a_raw),
        arm_b=_sha256(arm_b_raw),
        placebo=_sha256(placebo_raw),
    )


def _select_baseline(
    forecasts: Mapping[tuple[str, str], np.ndarray],
    diagnostics: Mapping[tuple[str, str], pd.Series],
    model_id: str,
) -> np.ndarray:
    keys = sorted(key for key in forecasts if key[0] == model_id)
    successful = [
        forecasts[key]
        for key in keys
        if str(diagnostics[key]["fit_status"]) == "succeeded"
        and np.isfinite(forecasts[key]).all()
    ]
    if not successful:
        raise V022PredictionCapsuleError(f"{model_id} has no finite succeeded forecast")
    first = successful[0]
    if any(not np.array_equal(first, candidate) for candidate in successful[1:]):
        raise V022PredictionCapsuleError(
            f"{model_id} has conflicting baseline variants"
        )
    return first


def _stress_index(
    temperature_c: float,
    soc_fraction: float,
    dod_fraction: float,
    efc_per_year: float,
) -> float:
    values = np.asarray(
        [temperature_c, soc_fraction, dod_fraction, efc_per_year],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise V022PredictionCapsuleError("Stress-index inputs must be finite")
    return float(
        0.35 * ((values[0] - 27.5) / 12.5)
        + 0.25 * ((values[1] - 0.55) / 0.35)
        + 0.15 * ((values[2] - 0.55) / 0.35)
        + 0.25 * ((values[3] - 275.0) / 175.0)
    )


def _abstention_reasons(
    *,
    successful_family_count: int,
    center: Sequence[float],
    prefix_features: Sequence[float],
    real_operating: Sequence[float],
    placebo_operating: Sequence[float],
) -> tuple[bool, str]:
    reasons: list[str] = []
    if successful_family_count < 2:
        reasons.append("insufficient_structure_families")
    if not np.isfinite(np.asarray(center, dtype=float)).all():
        reasons.append("nonfinite_center_forecast")
    if not np.isfinite(np.asarray(prefix_features, dtype=float)).all():
        reasons.append("nonfinite_prefix_features")
    if not np.isfinite(np.asarray(real_operating, dtype=float)).all():
        reasons.append("nonfinite_real_operating_features")
    if not np.isfinite(np.asarray(placebo_operating, dtype=float)).all():
        reasons.append("nonfinite_placebo_features")
    return not reasons, ";".join(reasons)


def _tie_hash(arm: str, content_hash: str) -> str:
    material = f"{V022_PROTOCOL_ID}|{arm}|{content_hash}".encode("ascii")
    return _sha256(material)


def _rank_primary_arms(
    *,
    prefix_only_scores: Sequence[float],
    visible_stress_scores: Sequence[float],
    prefix_only_hashes: Sequence[str],
    visible_stress_hashes: Sequence[str],
    hard_eligible: Sequence[bool],
    issue_count: int,
) -> _PrimaryArmRanking:
    prefix_scores = np.asarray(prefix_only_scores, dtype=np.float64)
    visible_scores = np.asarray(visible_stress_scores, dtype=np.float64)
    eligible_raw = np.asarray(hard_eligible, dtype=object)
    hashes_a = tuple(str(value) for value in prefix_only_hashes)
    hashes_b = tuple(str(value) for value in visible_stress_hashes)
    size = prefix_scores.size
    if (
        prefix_scores.ndim != 1
        or visible_scores.shape != (size,)
        or eligible_raw.shape != (size,)
        or len(hashes_a) != size
        or len(hashes_b) != size
        or type(issue_count) is not int
        or issue_count < 0
    ):
        raise V022PredictionCapsuleError("Primary ranking inputs are invalid")
    if any(not isinstance(value, (bool, np.bool_)) for value in eligible_raw):
        raise V022PredictionCapsuleError("hard_eligible must contain strict booleans")
    indices = np.flatnonzero(eligible_raw.astype(bool))
    if issue_count > len(indices):
        raise V022PredictionCapsuleError(
            "The common hard-eligibility pool is smaller than the issue count"
        )
    if (
        not np.isfinite(prefix_scores[indices]).all()
        or not np.isfinite(visible_scores[indices]).all()
    ):
        raise V022PredictionCapsuleError("An eligible primary risk score is nonfinite")
    ranking_a = rank_for_issuance(
        prefix_scores[indices],
        tuple(_tie_hash("prefix_only", hashes_a[index]) for index in indices),
        issue_count,
    )
    ranking_b = rank_for_issuance(
        visible_scores[indices],
        tuple(_tie_hash("visible_stress", hashes_b[index]) for index in indices),
        issue_count,
    )
    ranks_a: list[int | None] = [None] * size
    ranks_b: list[int | None] = [None] * size
    issued_a = [False] * size
    issued_b = [False] * size
    for local, global_index in enumerate(indices):
        index = int(global_index)
        ranks_a[index] = ranking_a.ranks[local]
        ranks_b[index] = ranking_b.ranks[local]
        issued_a[index] = ranking_a.issued[local]
        issued_b[index] = ranking_b.issued[local]
    if sum(issued_a) != issue_count or sum(issued_b) != issue_count:
        raise V022PredictionCapsuleError(
            "Primary arms did not issue the same fixed count"
        )
    return _PrimaryArmRanking(
        prefix_only_ranks=tuple(ranks_a),
        visible_stress_ranks=tuple(ranks_b),
        prefix_only_issued=tuple(issued_a),
        visible_stress_issued=tuple(issued_b),
    )


def _cluster_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["partition"].astype(str),
            frame["cluster_id"].astype(str),
            strict=True,
        )
    )


def _validate_cluster_alignment(
    frames: Sequence[pd.DataFrame],
) -> None:
    expected = _cluster_keys(frames[0])
    if not expected:
        raise V022PredictionCapsuleError("Predictor inputs contain no clusters")
    cluster_partitions: dict[str, set[str]] = {}
    for partition, cluster_id in expected:
        cluster_partitions.setdefault(cluster_id, set()).add(partition)
    if any(len(partitions) != 1 for partitions in cluster_partitions.values()):
        raise V022PredictionCapsuleError(
            "An opaque cluster ID is reused across partitions"
        )
    for frame in frames[1:]:
        if _cluster_keys(frame) != expected:
            raise V022PredictionCapsuleError("Predictor input cluster sets differ")


def _validate_prediction_input_alignment(
    frames: Mapping[str, pd.DataFrame],
) -> None:
    expected_names = (*_LABEL_INPUTS, *_FIT_OUTPUTS)
    if set(frames) != set(expected_names):
        raise V022PredictionCapsuleError("Prediction input file registry changed")
    _validate_cluster_alignment(tuple(frames[name] for name in expected_names))

    diagnostics = frames["member_fit_diagnostics.csv"]
    member_forecasts = frames["member_forecast_bundle.csv"]
    targets = frames["forecast_coordinates.csv"]
    variant_columns = [
        "partition",
        "cluster_id",
        "model_id",
        "variant_id",
    ]
    diagnostic_coordinates = diagnostics.loc[:, variant_columns].reset_index(drop=True)
    forecast_variant_coordinates = member_forecasts.loc[
        :, variant_columns
    ].drop_duplicates(ignore_index=True)
    if not diagnostic_coordinates.equals(forecast_variant_coordinates):
        raise V022PredictionCapsuleError(
            "Diagnostic and member-forecast variant coordinates differ"
        )

    target_columns = ["partition", "cluster_id", "forecast_day"]
    declared_targets = targets.loc[:, target_columns].reset_index(drop=True)
    forecast_targets = (
        member_forecasts.loc[:, target_columns]
        .drop_duplicates(ignore_index=True)
        .sort_values(target_columns, kind="stable")
        .reset_index(drop=True)
    )
    if not declared_targets.equals(forecast_targets):
        raise V022PredictionCapsuleError(
            "Member forecasts differ from declared target coordinates"
        )


def recompute_prediction_pipeline(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    member_fit_diagnostics: pd.DataFrame,
    member_forecast_bundle: pd.DataFrame,
    state: PredictionState,
    formal: bool,
) -> PredictionPipelineResult:
    """Recompute the V2.2 label-free outputs in the minimal capsule."""

    if type(state) is not PredictionState:
        raise V022PredictionCapsuleError(
            "Prediction state must have the exact capsule type"
        )
    _validate_prediction_state(state)
    try:
        return _recompute_prediction_pipeline_impl(
            prefix_pack=prefix_pack,
            forecast_coordinates=forecast_coordinates,
            operating_pack=operating_pack,
            member_fit_diagnostics=member_fit_diagnostics,
            member_forecast_bundle=member_forecast_bundle,
            state=state,
            formal=formal,
        )
    except V2ModelError as exc:
        raise V022PredictionCapsuleError(
            "A frozen statistical primitive rejected prediction inputs"
        ) from exc


def _recompute_prediction_pipeline_impl(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    member_fit_diagnostics: pd.DataFrame,
    member_forecast_bundle: pd.DataFrame,
    state: PredictionState,
    formal: bool,
) -> PredictionPipelineResult:
    supplied = (
        prefix_pack,
        forecast_coordinates,
        operating_pack,
        member_fit_diagnostics,
        member_forecast_bundle,
    )
    filenames = (*_LABEL_INPUTS, *_FIT_OUTPUTS)
    inputs = tuple(
        canonicalize_frame(frame, filename, formal=formal)
        for frame, filename in zip(supplied, filenames, strict=True)
    )
    prefix, coordinates, operating, diagnostics, member_forecasts = inputs
    _validate_prediction_input_alignment(dict(zip(filenames, inputs, strict=True)))

    prediction_records: list[dict[str, object]] = []
    feature_records: list[dict[str, object]] = []
    risk_records: list[dict[str, object]] = []
    content_records: list[dict[str, object]] = []

    diagnostics_by_cluster = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in diagnostics.groupby(
            ["partition", "cluster_id"],
            sort=True,
        )
    }
    forecasts_by_cluster = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in member_forecasts.groupby(
            ["partition", "cluster_id"],
            sort=True,
        )
    }
    prefix_by_cluster = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in prefix.groupby(
            ["partition", "cluster_id"],
            sort=True,
        )
    }
    coordinates_by_cluster = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in coordinates.groupby(
            ["partition", "cluster_id"],
            sort=True,
        )
    }
    operating_by_cluster = operating.set_index(["partition", "cluster_id"])
    if not operating_by_cluster.index.is_unique:
        raise V022PredictionCapsuleError(
            "Operating pack contains duplicate cluster keys"
        )

    for partition, cluster_id in sorted(_cluster_keys(prefix)):
        prefix_rows = _require_exact_grid(
            prefix_by_cluster[(partition, cluster_id)],
            value_column="prefix_day",
            expected=PREFIX_DAYS,
            context=f"{partition}/{cluster_id} prefix",
        )
        coordinate_rows = _require_exact_grid(
            coordinates_by_cluster[(partition, cluster_id)],
            value_column="forecast_day",
            expected=FORECAST_DAYS,
            context=f"{partition}/{cluster_id} coordinates",
        )
        observed = pd.to_numeric(
            prefix_rows["observed_retention_pct"],
            errors="coerce",
        ).to_numpy(float)
        if not np.isfinite(observed).all():
            raise V022PredictionCapsuleError("Prefix observations must be finite")

        operating_row = operating_by_cluster.loc[(partition, cluster_id)]
        operating_values = tuple(
            _finite_number(operating_row[name], context=name)
            for name in REAL_OPERATING_FIELDS
        )
        placebo_values = tuple(
            _finite_number(operating_row[name], context=name) for name in PLACEBO_FIELDS
        )
        hashes = predictor_content_hashes(
            prefix_rows,
            coordinate_rows,
            operating_row,
        )

        diagnostic_group = diagnostics_by_cluster[(partition, cluster_id)]
        forecast_group = forecasts_by_cluster[(partition, cluster_id)]
        diagnostic_key_rows = tuple(
            zip(
                diagnostic_group["model_id"].astype(str),
                diagnostic_group["variant_id"].astype(str),
                strict=True,
            )
        )
        forecast_key_rows = tuple(
            forecast_group.loc[:, ["model_id", "variant_id"]]
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if (
            len(diagnostic_key_rows) != 86
            or set(diagnostic_key_rows) != FROZEN_VARIANT_KEY_SET
            or len(forecast_key_rows) != 86
            or set(forecast_key_rows) != FROZEN_VARIANT_KEY_SET
            or len(forecast_group) != 86 * len(FORECAST_DAYS)
        ):
            raise V022PredictionCapsuleError(
                "Diagnostic and forecast exact 86-variant sets differ"
            )

        diagnostic_map: dict[tuple[str, str], pd.Series] = {}
        raw_forecast_map: dict[tuple[str, str], np.ndarray] = {}
        credible_by_family: dict[str, list[_SafeVariant]] = {}
        for key in sorted(FROZEN_VARIANT_KEY_SET):
            model_id, variant_id = key
            diagnostic_matches = diagnostic_group.loc[
                diagnostic_group["model_id"].astype(str).eq(model_id)
                & diagnostic_group["variant_id"].astype(str).eq(variant_id)
            ]
            if len(diagnostic_matches) != 1:
                raise V022PredictionCapsuleError("Variant diagnostics are not unique")
            diagnostic = diagnostic_matches.iloc[0]
            parameters, parameter_payload = _parse_parameter_payload(
                diagnostic["parameters_json"]
            )
            forecast_rows = _require_exact_grid(
                forecast_group.loc[
                    forecast_group["model_id"].astype(str).eq(model_id)
                    & forecast_group["variant_id"].astype(str).eq(variant_id)
                ],
                value_column="forecast_day",
                expected=FORECAST_DAYS,
                context=(f"{partition}/{cluster_id}/{model_id}/{variant_id}"),
            )
            raw_forecast = pd.to_numeric(
                forecast_rows["raw_forecast_retention_pct"],
                errors="coerce",
            ).to_numpy(float)
            diagnostic_hashes = set(
                diagnostic_matches["canonical_prefix_content_sha256"].astype(str)
            )
            forecast_hashes = set(
                forecast_rows["canonical_prefix_content_sha256"].astype(str)
            )
            if diagnostic_hashes != {hashes.arm_a} or forecast_hashes != {hashes.arm_a}:
                raise V022PredictionCapsuleError(
                    "Committed prefix-content hash differs from recomputation"
                )
            status = str(diagnostic["fit_status"])
            if status == "succeeded":
                recomputed_boundary = _verify_variant_commitment(
                    model_id=model_id,
                    variant_id=variant_id,
                    parameters=parameters,
                    prefix_days=prefix_rows["prefix_day"].to_numpy(float),
                    observed=observed,
                    forecast_days=forecast_rows["forecast_day"].to_numpy(float),
                    committed_rmse=diagnostic["prefix_rmse_pp"],
                    committed_residual=diagnostic["prefix_max_abs_residual_pp"],
                    committed_boundary=diagnostic["parameter_boundary_hit_fraction"],
                    committed_forecast=raw_forecast,
                )
            elif status == "failed":
                recomputed_boundary = math.nan
            else:
                raise V022PredictionCapsuleError(
                    "fit_status must be succeeded or failed"
                )
            credible = _recomputed_credible(
                diagnostic,
                raw_forecast,
                parameters=parameters,
            )
            diagnostic_map[key] = diagnostic
            raw_forecast_map[key] = raw_forecast
            if credible:
                credible_by_family.setdefault(model_id, []).append(
                    _SafeVariant(
                        forecast=tuple(float(value) for value in raw_forecast),
                        prefix_rmse_pp=_finite_number(
                            diagnostic["prefix_rmse_pp"],
                            context="prefix_rmse_pp",
                        ),
                        boundary_fraction=recomputed_boundary,
                        parameter_payload=parameter_payload,
                    )
                )

        sqrt_forecast = _select_baseline(
            raw_forecast_map,
            diagnostic_map,
            "target_prefix_sqrt_time",
        )
        bounded_forecast = _select_baseline(
            raw_forecast_map,
            diagnostic_map,
            "target_prefix_bounded_power_law",
        )
        family_vectors = {
            model_id: tuple(variant.forecast for variant in variants)
            for model_id, variants in credible_by_family.items()
        }
        library = build_library_forecast(
            family_vectors,
            sqrt_forecast,
        )
        center = blend_center_forecast(
            sqrt_forecast,
            library.forecast,
            state.center_beta,
        )

        if library.support_vectors:
            base_lower = coordinatewise_weighted_quantile(
                library.support_vectors,
                library.support_weights,
                0.05,
            )
            base_upper = coordinatewise_weighted_quantile(
                library.support_vectors,
                library.support_weights,
                0.95,
            )
            calibrated_lower_array, calibrated_upper_array = expand_intervals(
                (base_lower,),
                (base_upper,),
                state.conformal.expansion_pp,
            )
            calibrated_lower = calibrated_lower_array[0]
            calibrated_upper = calibrated_upper_array[0]
        else:
            base_lower = (math.nan,) * len(FORECAST_DAYS)
            base_upper = (math.nan,) * len(FORECAST_DAYS)
            calibrated_lower = np.full(len(FORECAST_DAYS), math.nan)
            calibrated_upper = np.full(len(FORECAST_DAYS), math.nan)

        if credible_by_family:
            prefix_features = _extract_prefix_features_safe(
                prefix_days=PREFIX_DAYS,
                observed_retention_pct=observed,
                family_variants=credible_by_family,
                sqrt_forecast=sqrt_forecast,
                center_forecast=center,
            )
        else:
            prefix_features = (math.nan,) * len(PREFIX_FEATURE_NAMES)
        arm_a_features = tuple(float(value) for value in prefix_features)
        arm_b_features = arm_a_features + operating_values
        placebo_features = arm_a_features + placebo_values
        planned_stress = _stress_index(*operating_values[4:])
        arm_a_plus_s_plan_features = arm_a_features + (planned_stress,)
        credible_vectors = [
            np.asarray(variant.forecast, dtype=float)
            for variants in credible_by_family.values()
            for variant in variants
        ]
        if credible_vectors:
            credible_matrix = np.vstack(credible_vectors)
            v1_max_envelope = float(
                np.max(
                    np.max(credible_matrix, axis=0) - np.min(credible_matrix, axis=0)
                )
            )
        else:
            v1_max_envelope = math.nan
        best_prefix_rmse = arm_a_features[
            PREFIX_FEATURE_NAMES.index("best_prefix_rmse_pp")
        ]
        center_sqrt_difference = abs(center[-1] - sqrt_forecast[-1])
        strongest_feature = arm_a_features[
            PREFIX_FEATURE_NAMES.index(state.strongest_single_feature_name)
        ]
        all_features_finite = bool(
            np.isfinite(
                (
                    *arm_b_features,
                    *placebo_values,
                    planned_stress,
                    v1_max_envelope,
                    center_sqrt_difference,
                    strongest_feature,
                )
            ).all()
        )
        hard_eligible, abstention_reasons = _abstention_reasons(
            successful_family_count=library.successful_family_count,
            center=center,
            prefix_features=arm_a_features,
            real_operating=operating_values,
            placebo_operating=placebo_values,
        )

        if all_features_finite:
            raw_a = float(
                state.prefix_only_risk.decision_function((arm_a_features,))[0]
            )
            raw_b = float(
                state.visible_stress_risk.decision_function((arm_b_features,))[0]
            )
            raw_placebo = float(
                state.placebo_risk.decision_function((placebo_features,))[0]
            )
            raw_arm_a_plus_s_plan = float(
                state.arm_a_plus_s_plan_risk.decision_function(
                    (arm_a_plus_s_plan_features,)
                )[0]
            )
            raw_strongest = float(
                state.strongest_single_feature_orientation * strongest_feature
            )
            raw_planned_stress = float(planned_stress)
            raw_prefix_rmse = float(best_prefix_rmse)
            raw_v1_envelope = float(v1_max_envelope)
            raw_center_sqrt = float(center_sqrt_difference)
            calibrated_a = float(state.prefix_only_isotonic.predict((raw_a,))[0])
            calibrated_b = float(state.visible_stress_isotonic.predict((raw_b,))[0])
            if not hard_eligible:
                calibrated_a = math.nan
                calibrated_b = math.nan
        else:
            (
                raw_a,
                raw_b,
                raw_placebo,
                raw_arm_a_plus_s_plan,
                raw_strongest,
                raw_planned_stress,
                raw_prefix_rmse,
                raw_v1_envelope,
                raw_center_sqrt,
                calibrated_a,
                calibrated_b,
            ) = (math.nan,) * 11

        identity = {
            "protocol_id": V022_PROTOCOL_ID,
            "partition": partition,
            "cluster_id": cluster_id,
        }
        for index, day in enumerate(FORECAST_DAYS):
            prediction_records.append(
                {
                    **identity,
                    "forecast_day": day,
                    "center_forecast_pct": center[index],
                    "sqrt_time_forecast_pct": float(sqrt_forecast[index]),
                    "bounded_power_forecast_pct": float(bounded_forecast[index]),
                    "base_interval_lower_pct": base_lower[index],
                    "base_interval_upper_pct": base_upper[index],
                    "calibrated_interval_lower_pct": (calibrated_lower[index]),
                    "calibrated_interval_upper_pct": (calibrated_upper[index]),
                    "canonical_prefix_content_sha256": hashes.arm_a,
                }
            )
        feature_records.append(
            {
                **identity,
                "hard_eligible": hard_eligible,
                "all_features_finite": all_features_finite,
                "abstention_reasons": abstention_reasons,
                **dict(
                    zip(
                        PREFIX_FEATURE_NAMES,
                        arm_a_features,
                        strict=True,
                    )
                ),
                **dict(
                    zip(
                        REAL_OPERATING_FIELDS,
                        operating_values,
                        strict=True,
                    )
                ),
                **dict(
                    zip(
                        PLACEBO_FIELDS,
                        placebo_values,
                        strict=True,
                    )
                ),
            }
        )
        for score_id, raw, calibrated, content_hash in (
            ("prefix_only", raw_a, calibrated_a, hashes.arm_a),
            ("visible_stress", raw_b, calibrated_b, hashes.arm_b),
            ("placebo_8", raw_placebo, math.nan, hashes.placebo),
            (
                "arm_a_plus_s_plan",
                raw_arm_a_plus_s_plan,
                math.nan,
                hashes.arm_b,
            ),
            (
                "strongest_single_feature",
                raw_strongest,
                math.nan,
                hashes.arm_a,
            ),
            (
                "planned_stress_only",
                raw_planned_stress,
                math.nan,
                hashes.arm_b,
            ),
            (
                "prefix_rmse_only",
                raw_prefix_rmse,
                math.nan,
                hashes.arm_a,
            ),
            (
                "v1_max_envelope_only",
                raw_v1_envelope,
                math.nan,
                hashes.arm_a,
            ),
            (
                "center_sqrt_abs_difference_only",
                raw_center_sqrt,
                math.nan,
                hashes.arm_a,
            ),
        ):
            risk_records.append(
                {
                    **identity,
                    "score_id": score_id,
                    "raw_risk_score": raw,
                    "calibrated_catastrophic_probability": calibrated,
                    "all_features_finite": all_features_finite,
                    "successful_structure_family_count": (
                        library.successful_family_count
                    ),
                    "fit_failure_count": (
                        len(DECLARED_STRUCTURE_FAMILIES)
                        - library.successful_family_count
                    ),
                    "effective_unique_shape_count": (
                        arm_a_features[
                            PREFIX_FEATURE_NAMES.index("effective_unique_shape_count")
                        ]
                    ),
                    "canonical_predictor_content_sha256": content_hash,
                }
            )
        content_records.append(
            {
                **identity,
                "random_policy_content_sha256": hashes.random_policy,
                "arm_a_content_sha256": hashes.arm_a,
                "arm_b_content_sha256": hashes.arm_b,
                "placebo_content_sha256": hashes.placebo,
            }
        )

    feature_bundle = (
        pd.DataFrame(feature_records)
        .sort_values(
            ["partition", "cluster_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    primary_risk = (
        pd.DataFrame(risk_records)
        .sort_values(
            ["partition", "cluster_id", "score_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    content_bundle = (
        pd.DataFrame(content_records)
        .sort_values(
            ["partition", "cluster_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    for partition in PRIMARY_ISSUE_COUNTS:
        ordinary = content_bundle.loc[content_bundle["partition"].eq(partition)]
        for column in (
            "random_policy_content_sha256",
            "arm_b_content_sha256",
            "placebo_content_sha256",
        ):
            if ordinary[column].duplicated().any():
                raise V022PredictionCapsuleError(
                    f"{partition} contains duplicate predictor content: {column}"
                )

    decision_records: list[dict[str, object]] = []
    for partition, raw_features in feature_bundle.groupby(
        "partition",
        sort=True,
    ):
        features = raw_features.sort_values(
            "cluster_id",
            kind="stable",
        ).reset_index(drop=True)
        risks = primary_risk.loc[primary_risk["partition"].eq(partition)].pivot(
            index="cluster_id",
            columns="score_id",
            values="raw_risk_score",
        )
        contents = content_bundle.loc[
            content_bundle["partition"].eq(partition)
        ].set_index("cluster_id")
        cluster_ids = features["cluster_id"].astype(str).tolist()
        if set(risks.index.astype(str)) != set(cluster_ids) or set(
            contents.index.astype(str)
        ) != set(cluster_ids):
            raise V022PredictionCapsuleError(
                "Decision inputs contain different cluster sets"
            )
        risks = risks.loc[cluster_ids]
        contents = contents.loc[cluster_ids]
        if partition in PRIMARY_ISSUE_COUNTS:
            ranking = _rank_primary_arms(
                prefix_only_scores=risks["prefix_only"].to_numpy(float),
                visible_stress_scores=risks["visible_stress"].to_numpy(float),
                prefix_only_hashes=contents["arm_a_content_sha256"].astype(str),
                visible_stress_hashes=contents["arm_b_content_sha256"].astype(str),
                hard_eligible=features["hard_eligible"].tolist(),
                issue_count=PRIMARY_ISSUE_COUNTS[str(partition)],
            )
        else:
            empty_ranks: tuple[int | None, ...] = (None,) * len(features)
            empty_issued = (False,) * len(features)
            ranking = _PrimaryArmRanking(
                prefix_only_ranks=empty_ranks,
                visible_stress_ranks=empty_ranks,
                prefix_only_issued=empty_issued,
                visible_stress_issued=empty_issued,
            )
        for index, row in features.iterrows():
            cluster_id = str(row["cluster_id"])
            for arm, ranks, issued, hash_column in (
                (
                    "prefix_only",
                    ranking.prefix_only_ranks,
                    ranking.prefix_only_issued,
                    "arm_a_content_sha256",
                ),
                (
                    "visible_stress",
                    ranking.visible_stress_ranks,
                    ranking.visible_stress_issued,
                    "arm_b_content_sha256",
                ),
            ):
                decision_records.append(
                    {
                        "protocol_id": V022_PROTOCOL_ID,
                        "partition": str(partition),
                        "cluster_id": cluster_id,
                        "arm": arm,
                        "raw_risk_score": float(risks.loc[cluster_id, arm]),
                        "hard_eligible": bool(row["hard_eligible"]),
                        "issuance_rank": ranks[index],
                        "issued": issued[index],
                        "abstention_reasons": str(row["abstention_reasons"]),
                        "canonical_predictor_content_sha256": str(
                            contents.loc[cluster_id, hash_column]
                        ),
                    }
                )

    prediction_bundle = canonicalize_frame(
        pd.DataFrame(prediction_records),
        "prediction_bundle.csv",
        formal=formal,
    )
    primary_risk_bundle = canonicalize_frame(
        primary_risk,
        "risk_bundle.csv",
        formal=formal,
    )
    decision_bundle = canonicalize_frame(
        pd.DataFrame(decision_records),
        "decision_bundle.csv",
        formal=formal,
    )
    return PredictionPipelineResult(
        prediction_bundle=prediction_bundle,
        feature_bundle=feature_bundle,
        primary_risk_bundle=primary_risk_bundle,
        decision_bundle=decision_bundle,
        predictor_content_bundle=content_bundle,
    )


def run_prediction_bundle(
    bundle: PredictionBundle,
    *,
    formal: bool = True,
) -> PredictionPipelineResult:
    """Extract immutable inputs and execute the capsule numerical core."""

    if formal is not True:
        raise V022PredictionCapsuleError(
            "A sealed prediction bundle can only run in formal mode"
        )
    value = _require_bundle_unchanged(bundle)
    frames = {name: frame.copy(deep=True) for name, frame in value._frames}
    state = value._state
    result = recompute_prediction_pipeline(
        prefix_pack=frames["prefix_pack.csv"],
        forecast_coordinates=frames["forecast_coordinates.csv"],
        operating_pack=frames["operating_pack.csv"],
        member_fit_diagnostics=frames["member_fit_diagnostics.csv"],
        member_forecast_bundle=frames["member_forecast_bundle.csv"],
        state=state,
        formal=formal,
    )
    _require_bundle_unchanged(value)
    raw_by_name = {
        filename: canonical_csv_bytes(
            getattr(result, attribute),
            filename,
            formal=True,
        )
        for filename, attribute in _PIPELINE_OUTPUT_FIELDS
    }
    sealed_result = PredictionPipelineResult(
        prediction_bundle=result.prediction_bundle,
        feature_bundle=result.feature_bundle,
        primary_risk_bundle=result.primary_risk_bundle,
        decision_bundle=result.decision_bundle,
        predictor_content_bundle=result.predictor_content_bundle,
    )
    _PIPELINE_RESULT_BINDINGS[id(sealed_result)] = (
        sealed_result,
        value,
        tuple(
            (filename, _sha256(raw_by_name[filename]))
            for filename, _ in _PIPELINE_OUTPUT_FIELDS
        ),
    )
    return sealed_result


__all__ = [
    "DecodedPredictionState",
    "PredictionArtifactMetadata",
    "PredictionBundle",
    "PredictionPipelineResult",
    "PredictionState",
    "V022PredictionCapsuleError",
    "V022_PROTOCOL_ID",
    "canonical_csv_bytes",
    "canonicalize_frame",
    "decode_prediction_state",
    "load_prediction_bundle",
    "predictor_content_hashes",
    "read_canonical_csv",
    "recompute_prediction_pipeline",
    "run_prediction_bundle",
    "write_prediction_outputs",
]
