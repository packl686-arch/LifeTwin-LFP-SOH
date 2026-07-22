from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc


FROZEN_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v1"
FROZEN_CONFIG_CANONICAL_SHA256 = (
    "6ad1e6dc1caa089ce0b9ee2c4e739a56c44f42f65436294649261a7676d4e320"
)
FROZEN_CONFIG_BYTE_SHA256 = (
    "503ec964bb2015fe3460433749d1b0d79f89187fc3dcd1c3809f9d4da2ffc319"
)
FROZEN_PREFIX_END_DAY = 730.0
FROZEN_PRIMARY_ENDPOINT_DAY = 9131.25
TRUTH_FAMILY_IDS = (
    "single_power",
    "dual_power",
    "saturating_plus_slow",
    "early_activation_plus_power",
    "late_knee",
)
STRUCTURE_MEMBER_IDS = (
    "target_prefix_persistence",
    "target_prefix_sqrt_time",
    "target_prefix_bounded_power_law",
    "target_prefix_saturating_plus_slow",
    "target_prefix_dual_power",
    "target_prefix_late_knee_prior_grid",
)
PARTITION_NAMES = ("development", "calibration", "test", "audit")
MATCHED_PARTITION = "matched_prefix_counterexamples"
PREDICTOR_PARTITION_NAMES = PARTITION_NAMES + (MATCHED_PARTITION,)
_VERIFIED_SCORE_TOKEN = object()

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "freeze",
    "scientific_question",
    "evidence_role",
    "design_status",
    "explicitly_deferred_questions",
    "time_grid",
    "truth_generation",
    "observation_model",
    "partitions",
    "candidate",
    "comparators",
    "matched_prefix_counterexample_audit",
    "endpoints",
    "decision_rules",
    "firewall_and_artifacts",
    "reporting",
    "claim_boundaries",
}
_TRUTH_PARAMETERS = {
    "single_power": ("a", "b"),
    "dual_power": ("a1", "b1", "a2", "b2"),
    "saturating_plus_slow": (
        "a_sat",
        "tau_sat_days",
        "b_sat",
        "a_slow",
        "b_slow",
    ),
    "early_activation_plus_power": (
        "a",
        "b",
        "activation_amplitude_pp",
        "tau_rise_days",
        "tau_decay_days",
    ),
    "late_knee": ("a", "b", "k_pp_per_day", "t_knee_days", "w_days"),
}
_CANDIDATE_PARAMETERS = {
    "target_prefix_persistence": (),
    "target_prefix_sqrt_time": ("c",),
    "target_prefix_bounded_power_law": ("a", "b"),
    "target_prefix_saturating_plus_slow": (
        "a_sat",
        "tau_sat_days",
        "b_sat",
        "a_slow",
        "b_slow",
    ),
    "target_prefix_dual_power": ("a1", "b1", "a2", "b2"),
    "target_prefix_late_knee_prior_grid": ("a", "b"),
}
_FORBIDDEN_PREDICTION_FIELDS = {
    "truth_family",
    "truth_parameters",
    "future_capacity_retention_pct",
    "future_error_pp",
    "catastrophic_error_label",
    "true_capacity_retention_pct",
    "error_pp",
    "absolute_error_pp",
}
PREFIX_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "prefix_day",
    "observed_retention_pct",
)
FORECAST_COORDINATE_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "forecast_day",
)
PREDICTION_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "forecast_day",
    "candidate_point_forecast_pct",
    "persistence_forecast_pct",
    "sqrt_time_forecast_pct",
    "bounded_power_forecast_pct",
    "structure_envelope_lower_pct",
    "structure_envelope_upper_pct",
    "canonical_prefix_content_sha256",
)
DECISION_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "canonical_prefix_content_sha256",
    "credible_structure_family_count",
    "fit_failure_count",
    "best_prefix_rmse_pp",
    "disagreement_score_pp",
    "hard_eligible",
    "primary_issuance_rank",
    "primary_issued",
    "abstention_reasons",
)
MEMBER_DIAGNOSTIC_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "model_id",
    "variant_id",
    "fit_status",
    "credible_variant",
    "prefix_rmse_pp",
    "prefix_max_abs_residual_pp",
    "forecast_min_pct",
    "forecast_max_pct",
    "canonical_prefix_content_sha256",
)
TRUTH_PACK_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "truth_family",
    "truth_parameters_json",
    "forecast_day",
    "latent_retention_pct",
    "noisy_retention_pct",
)
MATCHED_PAIR_COLUMNS = (
    "protocol_id",
    "pair_id",
    "left_cluster_id",
    "right_cluster_id",
    "left_family",
    "right_family",
    "latent_prefix_rmse_pp",
    "latent_prefix_max_abs_difference_pp",
    "truth_separation_25y_pp",
    "max_forecast_truth_separation_pp",
)


class SyntheticProtocolError(ValueError):
    """Raised when the frozen synthetic protocol is malformed or violated."""


@dataclass(frozen=True)
class ParameterDistribution:
    kind: str
    minimum: float
    maximum: float


@dataclass(frozen=True)
class TruthFamilyDefinition:
    family_id: str
    formula: str
    parameters: tuple[tuple[str, ParameterDistribution], ...]
    mechanism_role: str

    def parameter_map(self) -> dict[str, ParameterDistribution]:
        return dict(self.parameters)


@dataclass(frozen=True)
class TruthSpec:
    cluster_id: str
    partition: str
    family_id: str
    cluster_index: int
    parameters: tuple[tuple[str, float], ...]
    truth_seed: int
    measurement_seed: int

    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class CandidateVariant:
    model_id: str
    variant_id: str
    parameters: tuple[tuple[str, float], ...]
    prefix_rmse_pp: float
    prefix_max_absolute_residual_pp: float
    forecast_retention_pct: tuple[float, ...]
    fit_succeeded: bool
    failure_reason: str | None = None

    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class ValidatedSyntheticProtocol:
    protocol_id: str
    config_sha256: str
    time_scale_days: float
    prefix_days: tuple[float, ...]
    forecast_days: tuple[float, ...]
    primary_prefix_end_day: float
    primary_endpoint_day: float
    truth_families: tuple[TruthFamilyDefinition, ...]
    truth_retention_bounds_pct: tuple[float, float]
    maximum_single_interval_change_pp: float
    maximum_draw_attempts_per_cluster: int
    noise_sigma_bounds_pp: tuple[float, float]
    noise_ar1_bounds: tuple[float, float]
    cluster_counts_per_truth_family: tuple[tuple[str, int], ...]
    partition_seed_roots: tuple[tuple[str, int], ...]
    candidate_config_json: str
    matched_config_json: str
    endpoint_config_json: str
    decision_config_json: str
    bundle_columns_json: str
    forbidden_prediction_fields: tuple[str, ...]

    def truth_family_map(self) -> dict[str, TruthFamilyDefinition]:
        return {item.family_id: item for item in self.truth_families}

    def candidate_config(self) -> dict[str, Any]:
        return json.loads(self.candidate_config_json)

    def matched_config(self) -> dict[str, Any]:
        return json.loads(self.matched_config_json)

    def endpoint_config(self) -> dict[str, Any]:
        return json.loads(self.endpoint_config_json)

    def decision_config(self) -> dict[str, Any]:
        return json.loads(self.decision_config_json)

    def bundle_columns(self) -> dict[str, tuple[str, ...]]:
        return {
            name: tuple(columns)
            for name, columns in json.loads(self.bundle_columns_json).items()
        }


@dataclass(frozen=True)
class LabelFreePredictionResult:
    prediction_bundle: pd.DataFrame
    member_diagnostics: pd.DataFrame
    prediction_sha256: str
    member_diagnostics_sha256: str


@dataclass(frozen=True)
class DisagreementDecisionResult:
    decision_bundle: pd.DataFrame
    decision_sha256: str
    target_issue_count: int
    actual_issue_count: int


@dataclass(frozen=True)
class SyntheticClusterPacks:
    prefix_pack: pd.DataFrame
    forecast_coordinates: pd.DataFrame
    truth_pack: pd.DataFrame
    truth_spec: TruthSpec


@dataclass(frozen=True)
class MatchedPairPacks:
    """Predictor-visible packs plus the sealed truth-side pair record."""

    prefix_pack: pd.DataFrame
    forecast_coordinates: pd.DataFrame
    truth_pack: pd.DataFrame
    matched_prefix_pairs: pd.DataFrame
    truth_specs: tuple[TruthSpec, ...]


@dataclass(frozen=True)
class FrozenScoreResult:
    point_scores: pd.DataFrame
    trajectory_scores: pd.DataFrame
    prediction_sha256: str
    decision_sha256: str
    prefix_sha256: str
    forecast_coordinates_sha256: str
    member_diagnostics_sha256: str
    truth_sha256: str
    verified_decision_bytes: bytes = field(repr=False, compare=False)
    _verification_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class MatchedPairAuditResult:
    pair_scores: pd.DataFrame
    calibration_disagreement_threshold_pp: float | None
    endpoint_available: bool
    unavailable_reason: str | None
    qualified_pair_count: int
    both_rejected_pair_count: int
    both_rejected_fraction: float


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SyntheticProtocolError("Protocol values must be finite JSON") from exc
    return encoded.encode("ascii")


def load_frozen_protocol_config(
    path: str | Path,
) -> ValidatedSyntheticProtocol:
    """Load the exact frozen file, checking its byte and canonical commitments."""
    raw = Path(path).read_bytes()
    observed_byte_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_byte_sha256 != FROZEN_CONFIG_BYTE_SHA256:
        raise SyntheticProtocolError("Protocol file byte SHA-256 differs from v1")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticProtocolError("Frozen protocol is not valid UTF-8 JSON") from exc
    return validate_protocol_config(payload)


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticProtocolError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SyntheticProtocolError(
            f"{context} keys changed; missing={missing}, extra={extra}"
        )


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SyntheticProtocolError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SyntheticProtocolError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise SyntheticProtocolError(f"{context} must be a finite number")
    return result


def _positive_int(value: Any, *, context: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise SyntheticProtocolError(f"{context} must be a positive integer")
    return int(value)


def _number_pair(
    value: Any, *, context: str, lower_allowed: float | None = None
) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise SyntheticProtocolError(f"{context} must contain two bounds")
    lower = _finite_float(value[0], context=f"{context}[0]")
    upper = _finite_float(value[1], context=f"{context}[1]")
    if lower >= upper or (lower_allowed is not None and lower < lower_allowed):
        raise SyntheticProtocolError(f"{context} bounds are invalid")
    return lower, upper


def _strict_increasing_numbers(value: Any, *, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise SyntheticProtocolError(f"{context} must be a non-empty array")
    result = tuple(
        _finite_float(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if any(right <= left for left, right in zip(result, result[1:])):
        raise SyntheticProtocolError(f"{context} must be strictly increasing")
    return result


def _validate_distribution(value: Any, *, context: str) -> ParameterDistribution:
    mapping = _require_mapping(value, context=context)
    _require_exact_keys(
        mapping,
        {"distribution", "minimum", "maximum"},
        context=context,
    )
    kind = mapping["distribution"]
    if kind not in {"uniform", "log_uniform"}:
        raise SyntheticProtocolError(f"{context}.distribution is unsupported")
    minimum = _finite_float(mapping["minimum"], context=f"{context}.minimum")
    maximum = _finite_float(mapping["maximum"], context=f"{context}.maximum")
    if minimum >= maximum or (kind == "log_uniform" and minimum <= 0.0):
        raise SyntheticProtocolError(f"{context} bounds are invalid")
    return ParameterDistribution(kind=kind, minimum=minimum, maximum=maximum)


def _validate_truth_families(value: Any) -> tuple[TruthFamilyDefinition, ...]:
    if not isinstance(value, list):
        raise SyntheticProtocolError("truth_generation.truth_families must be an array")
    families: list[TruthFamilyDefinition] = []
    for index, raw in enumerate(value):
        context = f"truth_generation.truth_families[{index}]"
        mapping = _require_mapping(raw, context=context)
        _require_exact_keys(
            mapping,
            {"family_id", "formula", "parameters", "mechanism_role"},
            context=context,
        )
        family_id = mapping["family_id"]
        if family_id not in _TRUTH_PARAMETERS:
            raise SyntheticProtocolError(f"{context}.family_id is unsupported")
        parameter_mapping = _require_mapping(
            mapping["parameters"], context=f"{context}.parameters"
        )
        expected_parameters = set(_TRUTH_PARAMETERS[str(family_id)])
        _require_exact_keys(
            parameter_mapping,
            expected_parameters,
            context=f"{context}.parameters",
        )
        parameters = tuple(
            (
                name,
                _validate_distribution(
                    parameter_mapping[name], context=f"{context}.parameters.{name}"
                ),
            )
            for name in _TRUTH_PARAMETERS[str(family_id)]
        )
        formula = mapping["formula"]
        role = mapping["mechanism_role"]
        if not isinstance(formula, str) or not formula.strip():
            raise SyntheticProtocolError(f"{context}.formula must be non-empty")
        if not isinstance(role, str) or not role.strip():
            raise SyntheticProtocolError(f"{context}.mechanism_role must be non-empty")
        families.append(
            TruthFamilyDefinition(
                family_id=str(family_id),
                formula=formula,
                parameters=parameters,
                mechanism_role=role,
            )
        )
    observed = tuple(item.family_id for item in families)
    if observed != TRUTH_FAMILY_IDS:
        raise SyntheticProtocolError(
            "Frozen truth family order or membership changed: " f"{observed}"
        )
    return tuple(families)


def _validate_candidate(value: Any) -> str:
    candidate = _require_mapping(value, context="candidate")
    _require_exact_keys(
        candidate,
        {
            "model_id",
            "fit_data",
            "uses_future_outcomes",
            "structure_members",
            "structure_member_specs",
            "optimizer",
            "credible_member_rule",
            "family_aggregation",
            "point_forecast",
            "disagreement_score_pp",
            "primary_issuance_policy",
        },
        context="candidate",
    )
    if candidate["uses_future_outcomes"] is not False:
        raise SyntheticProtocolError("candidate must not use future outcomes")
    if tuple(candidate["structure_members"]) != STRUCTURE_MEMBER_IDS:
        raise SyntheticProtocolError("candidate.structure_members changed")
    raw_specs = candidate["structure_member_specs"]
    if not isinstance(raw_specs, list):
        raise SyntheticProtocolError("candidate.structure_member_specs must be an array")
    if tuple(item.get("model_id") for item in raw_specs) != STRUCTURE_MEMBER_IDS:
        raise SyntheticProtocolError("candidate structure specs changed")
    for raw_spec in raw_specs:
        model_id = str(raw_spec["model_id"])
        if model_id == "target_prefix_persistence":
            _require_exact_keys(
                raw_spec,
                {"model_id", "formula", "free_parameters"},
                context=f"candidate.{model_id}",
            )
            if raw_spec["free_parameters"] != []:
                raise SyntheticProtocolError("Persistence cannot have free parameters")
        elif model_id == "target_prefix_late_knee_prior_grid":
            _require_exact_keys(
                raw_spec,
                {
                    "model_id",
                    "formula",
                    "fitted_parameter_bounds",
                    "fixed_grid",
                    "grid_variant_policy",
                },
                context=f"candidate.{model_id}",
            )
            bounds_key = "fitted_parameter_bounds"
            _require_exact_keys(
                _require_mapping(raw_spec[bounds_key], context=bounds_key),
                set(_CANDIDATE_PARAMETERS[model_id]),
                context=f"candidate.{model_id}.{bounds_key}",
            )
            grid = _require_mapping(raw_spec["fixed_grid"], context="fixed_grid")
            _require_exact_keys(
                grid,
                {"k_pp_per_day", "t_knee_days", "w_days"},
                context=f"candidate.{model_id}.fixed_grid",
            )
            for grid_name, grid_values in grid.items():
                values = _strict_increasing_numbers(
                    grid_values,
                    context=f"candidate.{model_id}.fixed_grid.{grid_name}",
                )
                if values[0] <= 0.0:
                    raise SyntheticProtocolError("Late-knee grid values must be positive")
        else:
            _require_exact_keys(
                raw_spec,
                {"model_id", "formula", "parameter_bounds"}
                | ({"identifiability_constraint"} if model_id == "target_prefix_dual_power" else set()),
                context=f"candidate.{model_id}",
            )
            bounds = _require_mapping(
                raw_spec["parameter_bounds"],
                context=f"candidate.{model_id}.parameter_bounds",
            )
            _require_exact_keys(
                bounds,
                set(_CANDIDATE_PARAMETERS[model_id]),
                context=f"candidate.{model_id}.parameter_bounds",
            )
            for name, raw_bounds in bounds.items():
                _number_pair(
                    raw_bounds,
                    context=f"candidate.{model_id}.parameter_bounds.{name}",
                    lower_allowed=0.0,
                )
    optimizer = _require_mapping(candidate["optimizer"], context="candidate.optimizer")
    _require_exact_keys(
        optimizer,
        {
            "objective",
            "bounded_parameters",
            "algorithm",
            "deterministic_multistart_count",
            "multistart_design",
            "parameter_transform",
            "maximum_function_evaluations_per_start",
            "relative_parameter_tolerance",
            "relative_objective_tolerance",
            "best_fit_rule",
            "closed_form_exceptions",
            "fit_failure_rule",
        },
        context="candidate.optimizer",
    )
    _positive_int(
        optimizer["deterministic_multistart_count"],
        context="candidate.optimizer.deterministic_multistart_count",
    )
    _positive_int(
        optimizer["maximum_function_evaluations_per_start"],
        context="candidate.optimizer.maximum_function_evaluations_per_start",
    )
    if optimizer["bounded_parameters"] is not True:
        raise SyntheticProtocolError("candidate optimizer must remain bounded")
    if (
        optimizer["algorithm"]
        != "scipy.optimize.least_squares_method_trf_with_linear_loss"
        or optimizer["parameter_transform"]
        != "native_declared_parameter_scale_without_log_transform"
        or "no random seed" not in optimizer["multistart_design"]
    ):
        raise SyntheticProtocolError("Frozen deterministic optimizer changed")
    credible = _require_mapping(
        candidate["credible_member_rule"], context="candidate.credible_member_rule"
    )
    _require_exact_keys(
        credible,
        {
            "maximum_rmse_above_best_prefix_rmse_pp",
            "maximum_prefix_absolute_residual_pp",
            "minimum_structurally_distinct_family_ids",
            "forecast_retention_bounds_pct",
            "forecast_bounds_application",
            "member_fit_failure",
        },
        context="candidate.credible_member_rule",
    )
    if _positive_int(
        credible["minimum_structurally_distinct_family_ids"],
        context="candidate.credible_member_rule.minimum_structurally_distinct_family_ids",
    ) < 2:
        raise SyntheticProtocolError("At least two credible families are required")
    _number_pair(
        credible["forecast_retention_bounds_pct"],
        context="candidate.credible_member_rule.forecast_retention_bounds_pct",
    )
    issuance = _require_mapping(
        candidate["primary_issuance_policy"],
        context="candidate.primary_issuance_policy",
    )
    _require_exact_keys(
        issuance,
        {
            "type",
            "target_issuance_fraction",
            "ranking",
            "hard_eligibility",
            "tie_break",
            "truth_family_or_future_outcome_used",
            "required_eligible_test_cluster_count",
            "ineligible_cluster_policy",
            "minimum_finite_point_forecast_fraction",
            "deployment_threshold_claim_allowed",
        },
        context="candidate.primary_issuance_policy",
    )
    fraction = _finite_float(
        issuance["target_issuance_fraction"],
        context="candidate.primary_issuance_policy.target_issuance_fraction",
    )
    if not 0.0 < fraction < 1.0:
        raise SyntheticProtocolError("Issuance fraction must be in (0, 1)")
    if issuance["truth_family_or_future_outcome_used"] is not False:
        raise SyntheticProtocolError("Issuance cannot use truth family or outcomes")
    if issuance["deployment_threshold_claim_allowed"] is not False:
        raise SyntheticProtocolError("Synthetic rank cannot become a deployment threshold")
    return _canonical_json_bytes(candidate).decode("ascii")


def validate_protocol_config(config: Mapping[str, Any]) -> ValidatedSyntheticProtocol:
    """Strictly validate the frozen v1 protocol without reading any data files."""
    root = _require_mapping(config, context="protocol")
    _require_exact_keys(root, _TOP_LEVEL_KEYS, context="protocol")
    canonical_sha256 = hashlib.sha256(_canonical_json_bytes(root)).hexdigest()
    if canonical_sha256 != FROZEN_CONFIG_CANONICAL_SHA256:
        raise SyntheticProtocolError(
            "Protocol bytes or an otherwise unvalidated field differ from the frozen v1"
        )
    if root["schema_version"] != "1.0.0":
        raise SyntheticProtocolError("Unsupported synthetic protocol schema")
    if root["protocol_id"] != FROZEN_PROTOCOL_ID:
        raise SyntheticProtocolError("Synthetic protocol_id changed")
    if root["status"] != "frozen_before_first_simulation_execution":
        raise SyntheticProtocolError("Synthetic protocol is not frozen")
    if root["design_status"] != (
        "preregistered_before_implementation_and_simulation_outcome_generation"
    ):
        raise SyntheticProtocolError("Synthetic design status changed")

    time_grid = _require_mapping(root["time_grid"], context="time_grid")
    _require_exact_keys(
        time_grid,
        {
            "time_unit",
            "days_per_year_for_labels",
            "prefix_days",
            "forecast_days",
            "primary_prefix_end_day",
            "primary_endpoint_day",
            "prefix_observation_count",
            "positive_time_prefix_observation_count",
            "forecast_point_count",
            "future_capacity_or_soh_visible_to_predictor",
        },
        context="time_grid",
    )
    if time_grid["time_unit"] != "day":
        raise SyntheticProtocolError("Synthetic time unit must be day")
    time_scale = _finite_float(
        time_grid["days_per_year_for_labels"],
        context="time_grid.days_per_year_for_labels",
    )
    prefix_days = _strict_increasing_numbers(
        time_grid["prefix_days"], context="time_grid.prefix_days"
    )
    forecast_days = _strict_increasing_numbers(
        time_grid["forecast_days"], context="time_grid.forecast_days"
    )
    prefix_end = _finite_float(
        time_grid["primary_prefix_end_day"],
        context="time_grid.primary_prefix_end_day",
    )
    endpoint = _finite_float(
        time_grid["primary_endpoint_day"],
        context="time_grid.primary_endpoint_day",
    )
    if (
        prefix_days[0] != 0.0
        or prefix_days[-1] != prefix_end
        or forecast_days[0] <= prefix_end
        or forecast_days[-1] != endpoint
        or prefix_end != FROZEN_PREFIX_END_DAY
        or endpoint != FROZEN_PRIMARY_ENDPOINT_DAY
    ):
        raise SyntheticProtocolError("Frozen prefix or forecast horizon changed")
    if (
        _positive_int(
            time_grid["prefix_observation_count"],
            context="time_grid.prefix_observation_count",
        )
        != len(prefix_days)
        or _positive_int(
            time_grid["positive_time_prefix_observation_count"],
            context="time_grid.positive_time_prefix_observation_count",
        )
        != len([value for value in prefix_days if value > 0.0])
        or _positive_int(
            time_grid["forecast_point_count"],
            context="time_grid.forecast_point_count",
        )
        != len(forecast_days)
        or time_grid["future_capacity_or_soh_visible_to_predictor"] is not False
    ):
        raise SyntheticProtocolError("Synthetic time-grid counts or firewall changed")

    truth_generation = _require_mapping(
        root["truth_generation"], context="truth_generation"
    )
    _require_exact_keys(
        truth_generation,
        {
            "retention_unit",
            "time_scale_days",
            "random_generator",
            "log_uniform_definition",
            "softplus_definition",
            "family_sampling",
            "family_label_visible_to_predictor",
            "parameter_sampling",
            "truth_families",
            "admissibility_rules",
        },
        context="truth_generation",
    )
    if truth_generation["family_label_visible_to_predictor"] is not False:
        raise SyntheticProtocolError("Truth family must remain hidden from predictor")
    if not math.isclose(
        _finite_float(
            truth_generation["time_scale_days"],
            context="truth_generation.time_scale_days",
        ),
        time_scale,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise SyntheticProtocolError("Truth and label year scales disagree")
    if truth_generation["random_generator"] != (
        "numpy.random.Generator(PCG64DXSM)_under_the_frozen_environment"
    ):
        raise SyntheticProtocolError("Frozen random generator changed")
    truth_families = _validate_truth_families(truth_generation["truth_families"])
    admissibility = _require_mapping(
        truth_generation["admissibility_rules"],
        context="truth_generation.admissibility_rules",
    )
    _require_exact_keys(
        admissibility,
        {
            "day_zero_retention_exact",
            "minimum_truth_retention_pct_over_full_grid",
            "maximum_truth_retention_pct_over_full_grid",
            "finite_at_every_grid_point",
            "nonincreasing_required_except_family",
            "maximum_single_interval_change_pp",
            "rejected_draw_handling",
            "maximum_draw_attempts_per_cluster",
            "attempt_limit_result",
        },
        context="truth_generation.admissibility_rules",
    )
    truth_bounds = (
        _finite_float(
            admissibility["minimum_truth_retention_pct_over_full_grid"],
            context="admissibility.minimum_truth_retention_pct_over_full_grid",
        ),
        _finite_float(
            admissibility["maximum_truth_retention_pct_over_full_grid"],
            context="admissibility.maximum_truth_retention_pct_over_full_grid",
        ),
    )
    if (
        truth_bounds[0] >= truth_bounds[1]
        or _finite_float(
            admissibility["day_zero_retention_exact"],
            context="admissibility.day_zero_retention_exact",
        )
        != 100.0
        or admissibility["finite_at_every_grid_point"] is not True
        or admissibility["nonincreasing_required_except_family"]
        != "early_activation_plus_power"
    ):
        raise SyntheticProtocolError("Truth admissibility policy changed")

    observation = _require_mapping(root["observation_model"], context="observation_model")
    _require_exact_keys(
        observation,
        {
            "primary_score_target",
            "secondary_score_target",
            "day_zero_observation_exact",
            "noise_process",
            "noise_recurrence",
            "rng_consumption_order",
            "sigma_pp",
            "ar1_correlation",
            "missingness_in_primary_experiment",
            "rounding_before_model_fit",
        },
        context="observation_model",
    )
    if (
        observation["primary_score_target"] != "latent_noise_free_truth"
        or observation["day_zero_observation_exact"] != 100.0
        or observation["missingness_in_primary_experiment"] is not False
        or observation["rounding_before_model_fit"] is not False
    ):
        raise SyntheticProtocolError("Observation model contract changed")
    if (
        "exactly 20 standard-normal innovations"
        not in observation["rng_consumption_order"]
        or "AR steps follow observation-grid index"
        not in observation["noise_recurrence"]
    ):
        raise SyntheticProtocolError("Frozen AR1 recurrence or RNG order changed")
    sigma = _validate_distribution(observation["sigma_pp"], context="observation_model.sigma_pp")
    rho = _validate_distribution(
        observation["ar1_correlation"], context="observation_model.ar1_correlation"
    )
    if sigma.kind != "uniform" or rho.kind != "uniform" or not 0.0 <= rho.minimum < rho.maximum < 1.0:
        raise SyntheticProtocolError("Frozen Gaussian AR1 ranges changed")

    partitions = _require_mapping(root["partitions"], context="partitions")
    _require_exact_keys(
        partitions,
        {
            "independent_unit",
            "within_cluster_timepoints_are_independent_units",
            "truth_family_is_a_reporting_stratum_not_an_independent_replication",
            "partition_overlap_allowed",
            "cluster_counts_per_truth_family",
            "total_cluster_counts",
            "partition_seed_roots",
            "generation_seed_derivation",
            "predictor_initialization_derivation",
            "predictor_identity_firewall",
            "seed_collision_policy",
            "test_truth_access",
        },
        context="partitions",
    )
    if (
        partitions["within_cluster_timepoints_are_independent_units"] is not False
        or partitions["truth_family_is_a_reporting_stratum_not_an_independent_replication"] is not True
        or partitions["partition_overlap_allowed"] is not False
    ):
        raise SyntheticProtocolError("Synthetic statistical-unit policy changed")
    counts = _require_mapping(
        partitions["cluster_counts_per_truth_family"],
        context="partitions.cluster_counts_per_truth_family",
    )
    _require_exact_keys(counts, set(PARTITION_NAMES), context="cluster counts")
    parsed_counts = tuple(
        (name, _positive_int(counts[name], context=f"cluster_counts.{name}"))
        for name in PARTITION_NAMES
    )
    totals = _require_mapping(
        partitions["total_cluster_counts"], context="partitions.total_cluster_counts"
    )
    _require_exact_keys(totals, set(PARTITION_NAMES), context="total cluster counts")
    for name, count in parsed_counts:
        if _positive_int(totals[name], context=f"total_cluster_counts.{name}") != (
            count * len(TRUTH_FAMILY_IDS)
        ):
            raise SyntheticProtocolError(f"Partition {name} total is inconsistent")
    roots = _require_mapping(
        partitions["partition_seed_roots"], context="partitions.partition_seed_roots"
    )
    expected_roots = {
        *PARTITION_NAMES,
        "matched_prefix_counterexamples",
        "random_rejection_comparator",
        "bootstrap",
    }
    _require_exact_keys(roots, expected_roots, context="partition seed roots")
    parsed_roots = tuple(
        (name, _positive_int(roots[name], context=f"partition_seed_roots.{name}"))
        for name in sorted(roots)
    )
    if len({value for _, value in parsed_roots}) != len(parsed_roots):
        raise SyntheticProtocolError("Partition seed roots must be unique")
    if (
        "partition_seed_root" not in partitions["generation_seed_derivation"]
        or "truth_parameters and measurement_noise only"
        not in partitions["generation_seed_derivation"]
        or "use no random seed"
        not in partitions["predictor_initialization_derivation"]
        or "cluster IDs are opaque labels"
        not in partitions["predictor_identity_firewall"]
    ):
        raise SyntheticProtocolError("Seed or predictor identity firewall changed")

    candidate_json = _validate_candidate(root["candidate"])
    firewall = _require_mapping(
        root["firewall_and_artifacts"], context="firewall_and_artifacts"
    )
    _require_exact_keys(
        firewall,
        {
            "generation_order",
            "required_artifacts",
            "exact_csv_column_allowlists",
            "unknown_column_policy",
            "bundle_key_contracts",
            "exact_json_key_allowlists",
            "scorer_freeze_contract",
            "hash_algorithm",
            "nonfinite_json_allowed",
            "post_generation_exclusions_allowed",
            "model_failures_are_recorded_as_abstentions_not_deleted",
        },
        context="firewall_and_artifacts",
    )
    allowlists = _require_mapping(
        firewall["exact_csv_column_allowlists"],
        context="firewall_and_artifacts.exact_csv_column_allowlists",
    )
    expected_allowlists = {
        "prefix_pack.csv": PREFIX_COLUMNS,
        "forecast_coordinates.csv": FORECAST_COORDINATE_COLUMNS,
        "prediction_bundle.csv": PREDICTION_COLUMNS,
        "decision_bundle.csv": DECISION_COLUMNS,
        "member_fit_diagnostics.csv": MEMBER_DIAGNOSTIC_COLUMNS,
        "truth_pack.csv": TRUTH_PACK_COLUMNS,
        "matched_prefix_pairs.csv": MATCHED_PAIR_COLUMNS,
    }
    _require_exact_keys(
        allowlists,
        set(expected_allowlists),
        context="firewall_and_artifacts.exact_csv_column_allowlists",
    )
    for filename, expected_columns in expected_allowlists.items():
        if tuple(allowlists[filename]) != expected_columns:
            raise SyntheticProtocolError(f"{filename} exact column allowlist changed")
    if firewall["unknown_column_policy"] != (
        "reject_before_fitting_or_scoring_for_every_allowlisted_bundle; "
        "aliases and extra oracle or truth fields are not ignored"
    ):
        raise SyntheticProtocolError("Unknown-column firewall weakened")
    if firewall.get("hash_algorithm") != "sha256" or firewall.get("nonfinite_json_allowed") is not False:
        raise SyntheticProtocolError("Prediction commitment policy changed")

    matched = _require_mapping(
        root["matched_prefix_counterexample_audit"],
        context="matched_prefix_counterexample_audit",
    )
    if (
        _finite_float(
            matched.get("latent_prefix_rmse_max_pp"),
            context="matched.latent_prefix_rmse_max_pp",
        )
        > 0.1
        or _finite_float(
            matched.get("latent_prefix_max_absolute_difference_pp"),
            context="matched.latent_prefix_max_absolute_difference_pp",
        )
        > 0.1
        or _finite_float(
            matched.get("minimum_25_year_truth_separation_pp"),
            context="matched.minimum_25_year_truth_separation_pp",
        )
        < 5.0
        or matched.get("matching_uses_candidate_predictions") is not False
    ):
        raise SyntheticProtocolError("Matched-prefix falsification thresholds weakened")

    return ValidatedSyntheticProtocol(
        protocol_id=FROZEN_PROTOCOL_ID,
        config_sha256=canonical_sha256,
        time_scale_days=time_scale,
        prefix_days=prefix_days,
        forecast_days=forecast_days,
        primary_prefix_end_day=prefix_end,
        primary_endpoint_day=endpoint,
        truth_families=truth_families,
        truth_retention_bounds_pct=truth_bounds,
        maximum_single_interval_change_pp=_finite_float(
            admissibility["maximum_single_interval_change_pp"],
            context="admissibility.maximum_single_interval_change_pp",
        ),
        maximum_draw_attempts_per_cluster=_positive_int(
            admissibility["maximum_draw_attempts_per_cluster"],
            context="admissibility.maximum_draw_attempts_per_cluster",
        ),
        noise_sigma_bounds_pp=(sigma.minimum, sigma.maximum),
        noise_ar1_bounds=(rho.minimum, rho.maximum),
        cluster_counts_per_truth_family=parsed_counts,
        partition_seed_roots=parsed_roots,
        candidate_config_json=candidate_json,
        matched_config_json=_canonical_json_bytes(matched).decode("ascii"),
        endpoint_config_json=_canonical_json_bytes(root["endpoints"]).decode("ascii"),
        decision_config_json=_canonical_json_bytes(root["decision_rules"]).decode(
            "ascii"
        ),
        bundle_columns_json=_canonical_json_bytes(
            {name: list(columns) for name, columns in expected_allowlists.items()}
        ).decode("ascii"),
        forbidden_prediction_fields=tuple(sorted(_FORBIDDEN_PREDICTION_FIELDS)),
    )


def stable_softplus(value: np.ndarray | Sequence[float] | float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if not np.isfinite(values).all():
        raise SyntheticProtocolError("Softplus input must be finite")
    return np.logaddexp(0.0, values)


def evaluate_truth_retention(
    family_id: str,
    parameters: Mapping[str, float],
    elapsed_days: Sequence[float] | np.ndarray,
    *,
    time_scale_days: float = 365.25,
) -> np.ndarray:
    """Evaluate a declared latent truth without observation noise."""
    if family_id not in _TRUTH_PARAMETERS:
        raise SyntheticProtocolError(f"Unknown truth family: {family_id}")
    if set(parameters) != set(_TRUTH_PARAMETERS[family_id]):
        raise SyntheticProtocolError(f"Truth parameters changed for {family_id}")
    values = {
        name: _finite_float(parameters[name], context=f"truth parameter {name}")
        for name in _TRUTH_PARAMETERS[family_id]
    }
    elapsed = np.asarray(elapsed_days, dtype=float)
    if elapsed.ndim != 1 or not np.isfinite(elapsed).all() or np.any(elapsed < 0.0):
        raise SyntheticProtocolError("Truth elapsed days must be finite and non-negative")
    scale = _finite_float(time_scale_days, context="time_scale_days")
    if scale <= 0.0:
        raise SyntheticProtocolError("time_scale_days must be positive")
    years = elapsed / scale

    if family_id == "single_power":
        loss = values["a"] * np.power(years, values["b"])
    elif family_id == "dual_power":
        loss = values["a1"] * np.power(years, values["b1"])
        loss += values["a2"] * np.power(years, values["b2"])
    elif family_id == "saturating_plus_slow":
        saturation = values["a_sat"] * (
            1.0
            - np.exp(
                -np.power(elapsed / values["tau_sat_days"], values["b_sat"])
            )
        )
        slow = values["a_slow"] * np.power(years, values["b_slow"])
        loss = saturation + slow
    elif family_id == "early_activation_plus_power":
        loss = values["a"] * np.power(years, values["b"])
        activation = values["activation_amplitude_pp"] * (
            1.0 - np.exp(-elapsed / values["tau_rise_days"])
        ) * np.exp(-elapsed / values["tau_decay_days"])
        loss -= activation
    else:
        base = values["a"] * np.power(years, values["b"])
        knee = values["k_pp_per_day"] * values["w_days"] * (
            stable_softplus((elapsed - values["t_knee_days"]) / values["w_days"])
            - stable_softplus(-values["t_knee_days"] / values["w_days"])
        )
        loss = base + knee
    retention = 100.0 - loss
    retention[elapsed == 0.0] = 100.0
    if not np.isfinite(retention).all():
        raise SyntheticProtocolError("Truth evaluation produced non-finite retention")
    return retention


def _seed_from_parts(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16) % (2**63 - 1)


def derive_truth_stream_seed(
    protocol_id: str,
    partition_seed_root: int,
    partition: str,
    family_id: str,
    zero_based_cluster_index: int,
    stream_name: str,
) -> int:
    """Derive truth-side seeds; this function must never seed model fitting."""
    if protocol_id != FROZEN_PROTOCOL_ID or partition not in PARTITION_NAMES:
        raise SyntheticProtocolError("Truth seed coordinates are invalid")
    root = _positive_int(partition_seed_root, context="partition_seed_root")
    if family_id not in TRUTH_FAMILY_IDS:
        raise SyntheticProtocolError("Truth seed family is invalid")
    if (
        isinstance(zero_based_cluster_index, (bool, np.bool_))
        or not isinstance(zero_based_cluster_index, (int, np.integer))
        or int(zero_based_cluster_index) < 0
    ):
        raise SyntheticProtocolError("Cluster index must be a non-negative integer")
    if stream_name not in {"truth_parameters", "measurement_noise"}:
        raise SyntheticProtocolError("Truth stream name is invalid")
    return _seed_from_parts(
        protocol_id,
        root,
        partition,
        family_id,
        int(zero_based_cluster_index),
        stream_name,
    )


def canonical_frame_sha256(frame: pd.DataFrame, *, key_columns: Sequence[str]) -> str:
    """Hash a finite dataframe after deterministic coordinate sorting."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SyntheticProtocolError("Cannot hash an empty non-dataframe bundle")
    missing = [name for name in key_columns if name not in frame.columns]
    if missing:
        raise SyntheticProtocolError(f"Hash key columns are missing: {missing}")
    if frame.duplicated(list(key_columns)).any():
        raise SyntheticProtocolError("Hash key coordinates must be unique")
    ordered = frame.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for record in ordered.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for name in sorted(record):
            value = record[name]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                raise SyntheticProtocolError("Committed bundles cannot contain non-finite values")
            clean[name] = value
        records.append(clean)
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def canonical_csv_bytes(frame: pd.DataFrame, *, columns: Sequence[str]) -> bytes:
    """Serialize one allowlisted bundle deterministically for a byte commitment."""
    if not isinstance(frame, pd.DataFrame):
        raise SyntheticProtocolError("Bundle must be a dataframe")
    if tuple(frame.columns) != tuple(columns):
        raise SyntheticProtocolError(
            "Bundle columns do not match the frozen allowlist: "
            f"observed={tuple(frame.columns)}, expected={tuple(columns)}"
        )
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")


def canonical_csv_sha256(frame: pd.DataFrame, *, columns: Sequence[str]) -> str:
    return hashlib.sha256(canonical_csv_bytes(frame, columns=columns)).hexdigest()


def _sample_parameter(
    rng: np.random.Generator, distribution: ParameterDistribution
) -> float:
    if distribution.kind == "uniform":
        return float(rng.uniform(distribution.minimum, distribution.maximum))
    return float(
        np.exp(
            rng.uniform(
                np.log(distribution.minimum), np.log(distribution.maximum)
            )
        )
    )


def _truth_is_admissible(
    family_id: str,
    retention: np.ndarray,
    protocol: ValidatedSyntheticProtocol,
) -> bool:
    if retention.shape != (len(protocol.prefix_days) + len(protocol.forecast_days),):
        return False
    if not np.isfinite(retention).all() or retention[0] != 100.0:
        return False
    lower, upper = protocol.truth_retention_bounds_pct
    if float(np.min(retention)) < lower or float(np.max(retention)) > upper:
        return False
    if float(np.max(np.abs(np.diff(retention)))) > (
        protocol.maximum_single_interval_change_pp
    ):
        return False
    if family_id != "early_activation_plus_power" and np.any(
        np.diff(retention) > 1e-12
    ):
        return False
    return True


def sample_truth_spec(
    protocol: ValidatedSyntheticProtocol,
    *,
    partition: str,
    family_id: str,
    zero_based_family_cluster_index: int,
    opaque_cluster_id: str,
) -> TruthSpec:
    """Sample one admissible truth using only its declared truth-side stream."""
    if partition not in PARTITION_NAMES:
        raise SyntheticProtocolError("Unknown synthetic partition")
    if family_id not in TRUTH_FAMILY_IDS:
        raise SyntheticProtocolError("Unknown synthetic truth family")
    if not isinstance(opaque_cluster_id, str) or not opaque_cluster_id.strip():
        raise SyntheticProtocolError("An externally assigned opaque cluster ID is required")
    if (
        isinstance(zero_based_family_cluster_index, (bool, np.bool_))
        or not isinstance(zero_based_family_cluster_index, (int, np.integer))
        or int(zero_based_family_cluster_index) < 0
    ):
        raise SyntheticProtocolError("Truth cluster index must be non-negative")
    roots = dict(protocol.partition_seed_roots)
    root = roots[partition]
    index = int(zero_based_family_cluster_index)
    truth_seed = derive_truth_stream_seed(
        protocol.protocol_id,
        root,
        partition,
        family_id,
        index,
        "truth_parameters",
    )
    measurement_seed = derive_truth_stream_seed(
        protocol.protocol_id,
        root,
        partition,
        family_id,
        index,
        "measurement_noise",
    )
    if truth_seed == measurement_seed:
        raise SyntheticProtocolError("Truth and measurement seeds collided")
    rng = np.random.Generator(np.random.PCG64DXSM(truth_seed))
    definition = protocol.truth_family_map()[family_id]
    grid = np.asarray(protocol.prefix_days + protocol.forecast_days, dtype=float)
    for _ in range(protocol.maximum_draw_attempts_per_cluster):
        parameters = tuple(
            (name, _sample_parameter(rng, distribution))
            for name, distribution in definition.parameters
        )
        retention = evaluate_truth_retention(
            family_id,
            dict(parameters),
            grid,
            time_scale_days=protocol.time_scale_days,
        )
        if _truth_is_admissible(family_id, retention, protocol):
            return TruthSpec(
                cluster_id=opaque_cluster_id,
                partition=partition,
                family_id=family_id,
                cluster_index=index,
                parameters=parameters,
                truth_seed=truth_seed,
                measurement_seed=measurement_seed,
            )
    raise RuntimeError(
        "Synthetic truth draw exhausted its frozen attempt limit; "
        "the cluster cannot be excluded or resampled under another seed"
    )


def _ar1_observation_noise(
    protocol: ValidatedSyntheticProtocol,
    *,
    measurement_seed: int,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.Generator(np.random.PCG64DXSM(int(measurement_seed)))
    sigma = float(rng.uniform(*protocol.noise_sigma_bounds_pp))
    rho = float(rng.uniform(*protocol.noise_ar1_bounds))
    count = len(protocol.prefix_days) + len(protocol.forecast_days)
    innovations = rng.standard_normal(count)
    error = np.empty(count, dtype=float)
    error[0] = sigma * innovations[0]
    innovation_scale = sigma * math.sqrt(1.0 - rho**2)
    for index in range(1, count):
        error[index] = rho * error[index - 1] + innovation_scale * innovations[index]
    normalized = error - error[0]
    normalized[0] = 0.0
    return normalized, sigma, rho


def generate_cluster_packs(
    protocol: ValidatedSyntheticProtocol,
    truth_spec: TruthSpec,
) -> SyntheticClusterPacks:
    """Generate one prefix/coordinate/sealed-truth triplet with strict schemas."""
    if truth_spec.partition not in PREDICTOR_PARTITION_NAMES:
        raise SyntheticProtocolError("Truth spec partition is invalid")
    if not truth_spec.cluster_id or truth_spec.family_id not in TRUTH_FAMILY_IDS:
        raise SyntheticProtocolError("Truth spec identity or family is invalid")
    if truth_spec.partition in PARTITION_NAMES:
        expected_spec = sample_truth_spec(
            protocol,
            partition=truth_spec.partition,
            family_id=truth_spec.family_id,
            zero_based_family_cluster_index=truth_spec.cluster_index,
            opaque_cluster_id=truth_spec.cluster_id,
        )
        if truth_spec != expected_spec:
            raise SyntheticProtocolError(
                "Ordinary truth spec differs from its frozen seed-derived draw"
            )
    else:
        index = truth_spec.cluster_index
        pair_seed = derive_matched_pair_seed(protocol, index)
        pair_id = derive_matched_pair_id(protocol, index)
        opaque_ids = derive_matched_opaque_cluster_ids(protocol, index)
        if truth_spec.cluster_id not in opaque_ids:
            raise SyntheticProtocolError("Matched truth spec has a non-frozen opaque ID")
        expected_family = (
            "single_power" if truth_spec.cluster_id == opaque_ids[0] else "late_knee"
        )
        rng = np.random.Generator(np.random.PCG64DXSM(pair_seed))
        a = float(rng.uniform(0.2, 0.9))
        b = float(rng.uniform(0.35, 0.75))
        t_knee = float(rng.uniform(1826.25, 3652.5))
        k = float(rng.uniform(0.0015, 0.0030))
        expected_parameters = (
            (("a", a), ("b", b))
            if expected_family == "single_power"
            else (
                ("a", a),
                ("b", b),
                ("k_pp_per_day", k),
                ("t_knee_days", t_knee),
                ("w_days", 30.0),
            )
        )
        if (
            truth_spec.family_id != expected_family
            or truth_spec.parameters != expected_parameters
            or truth_spec.truth_seed != pair_seed
            or truth_spec.measurement_seed
            != derive_matched_measurement_seed(protocol, pair_id)
        ):
            raise SyntheticProtocolError(
                "Matched truth spec differs from its frozen one-shot construction"
            )
    grid = np.asarray(protocol.prefix_days + protocol.forecast_days, dtype=float)
    latent = evaluate_truth_retention(
        truth_spec.family_id,
        truth_spec.parameter_map(),
        grid,
        time_scale_days=protocol.time_scale_days,
    )
    noise, _, _ = _ar1_observation_noise(
        protocol, measurement_seed=truth_spec.measurement_seed
    )
    observed = latent + noise
    observed[0] = 100.0
    prefix_count = len(protocol.prefix_days)
    prefix_pack = pd.DataFrame(
        {
            "protocol_id": protocol.protocol_id,
            "partition": truth_spec.partition,
            "cluster_id": truth_spec.cluster_id,
            "prefix_day": protocol.prefix_days,
            "observed_retention_pct": observed[:prefix_count],
        },
        columns=PREFIX_COLUMNS,
    )
    forecast_coordinates = pd.DataFrame(
        {
            "protocol_id": protocol.protocol_id,
            "partition": truth_spec.partition,
            "cluster_id": truth_spec.cluster_id,
            "forecast_day": protocol.forecast_days,
        },
        columns=FORECAST_COORDINATE_COLUMNS,
    )
    parameter_json = _canonical_json_bytes(truth_spec.parameter_map()).decode("ascii")
    truth_pack = pd.DataFrame(
        {
            "protocol_id": protocol.protocol_id,
            "partition": truth_spec.partition,
            "cluster_id": truth_spec.cluster_id,
            "truth_family": truth_spec.family_id,
            "truth_parameters_json": parameter_json,
            "forecast_day": protocol.forecast_days,
            "latent_retention_pct": latent[prefix_count:],
            "noisy_retention_pct": observed[prefix_count:],
        },
        columns=TRUTH_PACK_COLUMNS,
    )
    return SyntheticClusterPacks(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        truth_pack=truth_pack,
        truth_spec=truth_spec,
    )


def derive_matched_pair_seed(
    protocol: ValidatedSyntheticProtocol,
    zero_based_pair_index: int,
) -> int:
    """Derive the one frozen construction seed for a matched pair."""
    if (
        isinstance(zero_based_pair_index, (bool, np.bool_))
        or not isinstance(zero_based_pair_index, (int, np.integer))
        or int(zero_based_pair_index) < 0
    ):
        raise SyntheticProtocolError("Matched-pair index must be non-negative")
    root = dict(protocol.partition_seed_roots)[MATCHED_PARTITION]
    return _seed_from_parts(
        protocol.protocol_id,
        root,
        "single_power",
        "late_knee",
        int(zero_based_pair_index),
    )


def derive_matched_pair_id(
    protocol: ValidatedSyntheticProtocol,
    zero_based_pair_index: int,
) -> str:
    """Create a deterministic sealed pair label that carries no family label."""
    derive_matched_pair_seed(protocol, zero_based_pair_index)
    root = dict(protocol.partition_seed_roots)[MATCHED_PARTITION]
    material = (
        f"{protocol.protocol_id}|{root}|matched_pair_id|"
        f"{int(zero_based_pair_index)}"
    ).encode("utf-8")
    return "p_" + hashlib.sha256(material).hexdigest()[:32]


def derive_matched_opaque_cluster_ids(
    protocol: ValidatedSyntheticProtocol,
    zero_based_pair_index: int,
) -> tuple[str, str]:
    """Create an exchange-symmetric opaque assignment for the two members."""
    derive_matched_pair_seed(protocol, zero_based_pair_index)
    root = dict(protocol.partition_seed_roots)[MATCHED_PARTITION]
    identifiers = []
    pair_index = int(zero_based_pair_index)
    for global_ordinal in (2 * pair_index, 2 * pair_index + 1):
        material = (
            f"{protocol.protocol_id}|{root}|opaque_cluster_pool|{global_ordinal}"
        ).encode("utf-8")
        identifiers.append("c_" + hashlib.sha256(material).hexdigest()[:32])
    orientation_material = (
        f"{protocol.protocol_id}|{root}|opaque_exchange_orientation_v1"
    ).encode("utf-8")
    orientation_offset = int(hashlib.sha256(orientation_material).hexdigest(), 16) & 1
    swap = (pair_index + orientation_offset) & 1
    return (
        (identifiers[1], identifiers[0])
        if swap
        else (identifiers[0], identifiers[1])
    )


def derive_matched_measurement_seed(
    protocol: ValidatedSyntheticProtocol,
    pair_id: str,
) -> int:
    """Derive the shared noise stream without consulting either member label."""
    if not isinstance(pair_id, str) or not pair_id.strip() or pair_id != pair_id.strip():
        raise SyntheticProtocolError("A non-empty canonical pair_id is required")
    root = dict(protocol.partition_seed_roots)[MATCHED_PARTITION]
    return _seed_from_parts(
        protocol.protocol_id,
        root,
        pair_id,
        "measurement_noise",
    )


def generate_matched_pair_packs(
    protocol: ValidatedSyntheticProtocol,
    *,
    zero_based_pair_index: int,
) -> MatchedPairPacks:
    """Construct one predeclared smooth/knee counterexample with shared noise.

    Pair and cluster labels are opaque hashes derived without family or side text.
    """
    pair_seed = derive_matched_pair_seed(protocol, zero_based_pair_index)
    pair_id = derive_matched_pair_id(protocol, zero_based_pair_index)
    opaque_cluster_ids = derive_matched_opaque_cluster_ids(
        protocol, zero_based_pair_index
    )
    matched = protocol.matched_config()
    pair_count = int(matched["required_pairs_per_family_pair"])
    index = int(zero_based_pair_index)
    if index < 0 or index >= pair_count:
        raise SyntheticProtocolError("Matched-pair index is outside the frozen range")
    if int(matched["proposal_count_per_pair"]) != 1:
        raise SyntheticProtocolError("Matched construction must use exactly one proposal")

    rng = np.random.Generator(np.random.PCG64DXSM(pair_seed))
    # The order of these four draws is part of the frozen protocol.
    a = float(rng.uniform(0.2, 0.9))
    b = float(rng.uniform(0.35, 0.75))
    t_knee = float(rng.uniform(1826.25, 3652.5))
    k = float(rng.uniform(0.0015, 0.0030))
    width = 30.0
    measurement_seed = derive_matched_measurement_seed(protocol, pair_id)
    left_spec = TruthSpec(
        cluster_id=opaque_cluster_ids[0],
        partition=MATCHED_PARTITION,
        family_id="single_power",
        cluster_index=index,
        parameters=(("a", a), ("b", b)),
        truth_seed=pair_seed,
        measurement_seed=measurement_seed,
    )
    right_spec = TruthSpec(
        cluster_id=opaque_cluster_ids[1],
        partition=MATCHED_PARTITION,
        family_id="late_knee",
        cluster_index=index,
        parameters=(
            ("a", a),
            ("b", b),
            ("k_pp_per_day", k),
            ("t_knee_days", t_knee),
            ("w_days", width),
        ),
        truth_seed=pair_seed,
        measurement_seed=measurement_seed,
    )

    dense_prefix_days = np.arange(0.0, protocol.primary_prefix_end_day + 1.0, 1.0)
    left_prefix = evaluate_truth_retention(
        left_spec.family_id,
        left_spec.parameter_map(),
        dense_prefix_days,
        time_scale_days=protocol.time_scale_days,
    )
    right_prefix = evaluate_truth_retention(
        right_spec.family_id,
        right_spec.parameter_map(),
        dense_prefix_days,
        time_scale_days=protocol.time_scale_days,
    )
    prefix_difference = right_prefix - left_prefix
    prefix_rmse = float(np.sqrt(np.mean(np.square(prefix_difference))))
    prefix_max = float(np.max(np.abs(prefix_difference)))
    forecast_days = np.asarray(protocol.forecast_days, dtype=float)
    left_forecast = evaluate_truth_retention(
        left_spec.family_id,
        left_spec.parameter_map(),
        forecast_days,
        time_scale_days=protocol.time_scale_days,
    )
    right_forecast = evaluate_truth_retention(
        right_spec.family_id,
        right_spec.parameter_map(),
        forecast_days,
        time_scale_days=protocol.time_scale_days,
    )
    forecast_difference = np.abs(right_forecast - left_forecast)
    endpoint_index = int(
        np.flatnonzero(forecast_days == protocol.primary_endpoint_day)[0]
    )
    separation_25y = float(forecast_difference[endpoint_index])
    maximum_forecast_separation = float(np.max(forecast_difference))
    if (
        prefix_rmse > float(matched["latent_prefix_rmse_max_pp"])
        or prefix_max
        > float(matched["latent_prefix_max_absolute_difference_pp"])
        or separation_25y < float(matched["minimum_25_year_truth_separation_pp"])
        or maximum_forecast_separation
        < float(matched["minimum_maximum_forecast_grid_truth_separation_pp"])
    ):
        raise RuntimeError(
            "Frozen matched-pair construction failed its thresholds; "
            "resampling and threshold relaxation are forbidden"
        )

    left_pack = generate_cluster_packs(protocol, left_spec)
    right_pack = generate_cluster_packs(protocol, right_spec)
    prefix_pack, forecast_coordinates, truth_pack = concatenate_cluster_packs(
        (left_pack, right_pack)
    )
    pair_frame = pd.DataFrame(
        [
            {
                "protocol_id": protocol.protocol_id,
                "pair_id": pair_id,
                "left_cluster_id": opaque_cluster_ids[0],
                "right_cluster_id": opaque_cluster_ids[1],
                "left_family": "single_power",
                "right_family": "late_knee",
                "latent_prefix_rmse_pp": prefix_rmse,
                "latent_prefix_max_abs_difference_pp": prefix_max,
                "truth_separation_25y_pp": separation_25y,
                "max_forecast_truth_separation_pp": maximum_forecast_separation,
            }
        ],
        columns=MATCHED_PAIR_COLUMNS,
    )
    return MatchedPairPacks(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        truth_pack=truth_pack,
        matched_prefix_pairs=pair_frame,
        truth_specs=(left_spec, right_spec),
    )


def generate_all_matched_pair_packs(
    protocol: ValidatedSyntheticProtocol,
) -> MatchedPairPacks:
    """Construct the exact 200-pair frozen audit without selection or retries."""
    required = int(protocol.matched_config()["required_total_pair_clusters"])
    pairs = [
        generate_matched_pair_packs(
            protocol,
            zero_based_pair_index=index,
        )
        for index in range(required)
    ]
    prefix_pack = pd.concat([item.prefix_pack for item in pairs], ignore_index=True)
    forecast_coordinates = pd.concat(
        [item.forecast_coordinates for item in pairs], ignore_index=True
    )
    truth_pack = pd.concat([item.truth_pack for item in pairs], ignore_index=True)
    pair_frame = pd.concat(
        [item.matched_prefix_pairs for item in pairs], ignore_index=True
    )
    if pair_frame["pair_id"].duplicated().any():
        raise RuntimeError("Deterministic matched pair IDs collided")
    all_cluster_ids = pd.concat(
        [pair_frame["left_cluster_id"], pair_frame["right_cluster_id"]],
        ignore_index=True,
    )
    if all_cluster_ids.duplicated().any():
        raise RuntimeError("Deterministic matched opaque cluster IDs collided")
    return MatchedPairPacks(
        prefix_pack=prefix_pack.sort_values(
            ["partition", "cluster_id", "prefix_day"], kind="stable"
        ).reset_index(drop=True),
        forecast_coordinates=forecast_coordinates.sort_values(
            ["partition", "cluster_id", "forecast_day"], kind="stable"
        ).reset_index(drop=True),
        truth_pack=truth_pack.sort_values(
            ["partition", "cluster_id", "forecast_day"], kind="stable"
        ).reset_index(drop=True),
        matched_prefix_pairs=pair_frame.sort_values("pair_id", kind="stable").reset_index(
            drop=True
        ),
        truth_specs=tuple(spec for item in pairs for spec in item.truth_specs),
    )


def concatenate_cluster_packs(
    packs: Sequence[SyntheticClusterPacks],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not packs:
        raise SyntheticProtocolError("At least one synthetic cluster is required")
    prefix = pd.concat([item.prefix_pack for item in packs], ignore_index=True)
    coordinates = pd.concat(
        [item.forecast_coordinates for item in packs], ignore_index=True
    )
    truth = pd.concat([item.truth_pack for item in packs], ignore_index=True)
    return (
        prefix.sort_values(["partition", "cluster_id", "prefix_day"], kind="stable")
        .reset_index(drop=True)
        .loc[:, list(PREFIX_COLUMNS)],
        coordinates.sort_values(
            ["partition", "cluster_id", "forecast_day"], kind="stable"
        )
        .reset_index(drop=True)
        .loc[:, list(FORECAST_COORDINATE_COLUMNS)],
        truth.sort_values(
            ["partition", "cluster_id", "forecast_day"], kind="stable"
        )
        .reset_index(drop=True)
        .loc[:, list(TRUTH_PACK_COLUMNS)],
    )


def canonical_prefix_content_sha256(prefix: pd.DataFrame) -> str:
    """Hash only ordered prefix content, deliberately excluding identity labels."""
    required = {"prefix_day", "observed_retention_pct"}
    if tuple(prefix.columns) != PREFIX_COLUMNS:
        raise SyntheticProtocolError("Prefix bundle has unknown or missing columns")
    if prefix.empty or prefix["prefix_day"].duplicated().any():
        raise SyntheticProtocolError("One complete unique prefix is required")
    ordered = prefix.sort_values("prefix_day", kind="stable")
    values = []
    for row in ordered.itertuples(index=False):
        day = _finite_float(row.prefix_day, context="prefix_day")
        retention = _finite_float(
            row.observed_retention_pct, context="observed_retention_pct"
        )
        values.append([day, retention])
    if not required.issubset(prefix.columns):
        raise SyntheticProtocolError("Prefix content fields are missing")
    return hashlib.sha256(_canonical_json_bytes(values)).hexdigest()


def _candidate_specs(protocol: ValidatedSyntheticProtocol) -> dict[str, dict[str, Any]]:
    candidate = protocol.candidate_config()
    return {item["model_id"]: item for item in candidate["structure_member_specs"]}


def _candidate_bounds(
    spec: Mapping[str, Any], model_id: str
) -> tuple[np.ndarray, np.ndarray]:
    key = (
        "fitted_parameter_bounds"
        if model_id == "target_prefix_late_knee_prior_grid"
        else "parameter_bounds"
    )
    raw = spec[key]
    names = _CANDIDATE_PARAMETERS[model_id]
    lower = np.asarray([float(raw[name][0]) for name in names], dtype=float)
    upper = np.asarray([float(raw[name][1]) for name in names], dtype=float)
    return lower, upper


def _predict_structure(
    model_id: str,
    parameters: Mapping[str, float],
    elapsed_days: np.ndarray,
    *,
    time_scale_days: float,
    fixed: Mapping[str, float] | None = None,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    years = elapsed / time_scale_days
    if model_id == "target_prefix_persistence":
        return np.full_like(elapsed, parameters["last_retention_pct"], dtype=float)
    if model_id == "target_prefix_sqrt_time":
        return 100.0 - parameters["c"] * np.sqrt(years)
    if model_id == "target_prefix_bounded_power_law":
        return 100.0 - parameters["a"] * np.power(years, parameters["b"])
    if model_id == "target_prefix_saturating_plus_slow":
        saturated = parameters["a_sat"] * (
            1.0
            - np.exp(
                -np.power(
                    elapsed / parameters["tau_sat_days"], parameters["b_sat"]
                )
            )
        )
        slow = parameters["a_slow"] * np.power(years, parameters["b_slow"])
        return 100.0 - saturated - slow
    if model_id == "target_prefix_dual_power":
        loss = parameters["a1"] * np.power(years, parameters["b1"])
        loss += parameters["a2"] * np.power(years, parameters["b2"])
        return 100.0 - loss
    if model_id == "target_prefix_late_knee_prior_grid":
        if fixed is None or set(fixed) != {
            "k_pp_per_day",
            "t_knee_days",
            "w_days",
        }:
            raise SyntheticProtocolError("Late-knee prediction requires its fixed grid")
        base = parameters["a"] * np.power(years, parameters["b"])
        knee = fixed["k_pp_per_day"] * fixed["w_days"] * (
            stable_softplus(
                (elapsed - fixed["t_knee_days"]) / fixed["w_days"]
            )
            - stable_softplus(-fixed["t_knee_days"] / fixed["w_days"])
        )
        return 100.0 - base - knee
    raise SyntheticProtocolError(f"Unknown candidate structure: {model_id}")


def _sobol_starts(lower: np.ndarray, upper: np.ndarray, *, count: int) -> np.ndarray:
    if count < 1 or count & (count - 1):
        raise SyntheticProtocolError("Sobol multistart count must be a power of two")
    dimension = len(lower)
    if dimension < 1 or lower.shape != upper.shape or np.any(lower >= upper):
        raise SyntheticProtocolError("Candidate parameter bounds are invalid")
    sampler = qmc.Sobol(d=dimension, scramble=False)
    unit = sampler.random_base2(m=int(math.log2(count)))
    starts = lower + unit * (upper - lower)
    return np.minimum(np.maximum(starts, lower), upper)


def _fit_bounded_variant(
    model_id: str,
    variant_id: str,
    prefix_days: np.ndarray,
    observed: np.ndarray,
    forecast_days: np.ndarray,
    protocol: ValidatedSyntheticProtocol,
    spec: Mapping[str, Any],
    *,
    fixed: Mapping[str, float] | None = None,
) -> CandidateVariant:
    candidate = protocol.candidate_config()
    optimizer = candidate["optimizer"]
    names = _CANDIDATE_PARAMETERS[model_id]
    lower, upper = _candidate_bounds(spec, model_id)
    starts = _sobol_starts(
        lower,
        upper,
        count=int(optimizer["deterministic_multistart_count"]),
    )
    successes: list[tuple[float, tuple[float, ...], np.ndarray, np.ndarray]] = []

    def parameter_map(values: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(names, values)}

    def residual(values: np.ndarray) -> np.ndarray:
        return _predict_structure(
            model_id,
            parameter_map(values),
            prefix_days,
            time_scale_days=protocol.time_scale_days,
            fixed=fixed,
        ) - observed

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
            forecast = _predict_structure(
                model_id,
                parameter_map(values),
                forecast_days,
                time_scale_days=protocol.time_scale_days,
                fixed=fixed,
            )
            if (
                not fitted.success
                or not np.isfinite(values).all()
                or not np.isfinite(prefix_residual).all()
                or not np.isfinite(forecast).all()
            ):
                continue
            sse = float(prefix_residual @ prefix_residual)
            successes.append(
                (sse, tuple(float(value) for value in values), prefix_residual, forecast)
            )
        except (FloatingPointError, RuntimeError, ValueError):
            continue
    if not successes:
        return CandidateVariant(
            model_id=model_id,
            variant_id=variant_id,
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
    parameter_pairs = tuple((name, value) for name, value in zip(names, values))
    if fixed:
        parameter_pairs += tuple((name, float(fixed[name])) for name in sorted(fixed))
    return CandidateVariant(
        model_id=model_id,
        variant_id=variant_id,
        parameters=parameter_pairs,
        prefix_rmse_pp=float(np.sqrt(np.mean(np.square(prefix_residual)))),
        prefix_max_absolute_residual_pp=float(np.max(np.abs(prefix_residual))),
        forecast_retention_pct=tuple(float(value) for value in forecast),
        fit_succeeded=True,
    )


def _fit_closed_form_variants(
    prefix_days: np.ndarray,
    observed: np.ndarray,
    forecast_days: np.ndarray,
    protocol: ValidatedSyntheticProtocol,
    specs: Mapping[str, Mapping[str, Any]],
) -> list[CandidateVariant]:
    variants: list[CandidateVariant] = []
    last = float(observed[-1])
    persistence_prefix = np.full_like(observed, last, dtype=float)
    persistence_forecast = np.full_like(forecast_days, last, dtype=float)
    persistence_residual = persistence_prefix - observed
    variants.append(
        CandidateVariant(
            model_id="target_prefix_persistence",
            variant_id="persistence",
            parameters=(("last_retention_pct", last),),
            prefix_rmse_pp=float(
                np.sqrt(np.mean(np.square(persistence_residual)))
            ),
            prefix_max_absolute_residual_pp=float(
                np.max(np.abs(persistence_residual))
            ),
            forecast_retention_pct=tuple(float(value) for value in persistence_forecast),
            fit_succeeded=True,
        )
    )

    sqrt_spec = specs["target_prefix_sqrt_time"]
    lower, upper = _candidate_bounds(sqrt_spec, "target_prefix_sqrt_time")
    basis = np.sqrt(prefix_days / protocol.time_scale_days)
    denominator = float(basis @ basis)
    if denominator <= 0.0:
        coefficient = float(lower[0])
    else:
        coefficient = float(
            np.clip((basis @ (100.0 - observed)) / denominator, lower[0], upper[0])
        )
    sqrt_parameters = {"c": coefficient}
    sqrt_prefix = _predict_structure(
        "target_prefix_sqrt_time",
        sqrt_parameters,
        prefix_days,
        time_scale_days=protocol.time_scale_days,
    )
    sqrt_forecast = _predict_structure(
        "target_prefix_sqrt_time",
        sqrt_parameters,
        forecast_days,
        time_scale_days=protocol.time_scale_days,
    )
    sqrt_residual = sqrt_prefix - observed
    variants.append(
        CandidateVariant(
            model_id="target_prefix_sqrt_time",
            variant_id="sqrt_time",
            parameters=(("c", coefficient),),
            prefix_rmse_pp=float(np.sqrt(np.mean(np.square(sqrt_residual)))),
            prefix_max_absolute_residual_pp=float(np.max(np.abs(sqrt_residual))),
            forecast_retention_pct=tuple(float(value) for value in sqrt_forecast),
            fit_succeeded=True,
        )
    )
    return variants


def fit_structure_family_variants(
    prefix_days: Sequence[float],
    observed_retention_pct: Sequence[float],
    forecast_days: Sequence[float],
    protocol: ValidatedSyntheticProtocol,
) -> tuple[CandidateVariant, ...]:
    """Fit every frozen structure using prefix values only and seedless starts."""
    prefix = np.asarray(prefix_days, dtype=float)
    observed = np.asarray(observed_retention_pct, dtype=float)
    forecast = np.asarray(forecast_days, dtype=float)
    if (
        prefix.shape != observed.shape
        or prefix.ndim != 1
        or not np.isfinite(prefix).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(forecast).all()
        or np.any(prefix < 0.0)
        or np.any(forecast <= prefix.max())
    ):
        raise SyntheticProtocolError("Candidate fit inputs violate the prefix firewall")
    specs = _candidate_specs(protocol)
    variants = _fit_closed_form_variants(
        prefix, observed, forecast, protocol, specs
    )
    for model_id in (
        "target_prefix_bounded_power_law",
        "target_prefix_saturating_plus_slow",
        "target_prefix_dual_power",
    ):
        variants.append(
            _fit_bounded_variant(
                model_id,
                model_id,
                prefix,
                observed,
                forecast,
                protocol,
                specs[model_id],
            )
        )
    knee_spec = specs["target_prefix_late_knee_prior_grid"]
    grid = knee_spec["fixed_grid"]
    for k in grid["k_pp_per_day"]:
        for t_knee in grid["t_knee_days"]:
            for width in grid["w_days"]:
                fixed = {
                    "k_pp_per_day": float(k),
                    "t_knee_days": float(t_knee),
                    "w_days": float(width),
                }
                variant_id = f"k={k:g}|t={t_knee:g}|w={width:g}"
                variants.append(
                    _fit_bounded_variant(
                        "target_prefix_late_knee_prior_grid",
                        variant_id,
                        prefix,
                        observed,
                        forecast,
                        protocol,
                        knee_spec,
                        fixed=fixed,
                    )
                )
    return tuple(variants)


def _validate_exact_bundle_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    *,
    context: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise SyntheticProtocolError(f"{context} must be a dataframe")
    if tuple(frame.columns) != tuple(expected):
        raise SyntheticProtocolError(
            f"{context} has unknown or missing columns; "
            f"observed={tuple(frame.columns)}, expected={tuple(expected)}"
        )


def _validated_predictor_inputs(
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    protocol: ValidatedSyntheticProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_exact_bundle_columns(prefix_pack, PREFIX_COLUMNS, context="prefix_pack")
    _validate_exact_bundle_columns(
        forecast_coordinates,
        FORECAST_COORDINATE_COLUMNS,
        context="forecast_coordinates",
    )
    prefix = prefix_pack.copy()
    coordinates = forecast_coordinates.copy()
    if prefix.empty or coordinates.empty:
        raise SyntheticProtocolError("Predictor bundles cannot be empty")
    if prefix.duplicated(["partition", "cluster_id", "prefix_day"]).any():
        raise SyntheticProtocolError("Prefix coordinates must be unique")
    if coordinates.duplicated(
        ["partition", "cluster_id", "forecast_day"]
    ).any():
        raise SyntheticProtocolError("Forecast coordinates must be unique")
    for frame, day_column, value_column in (
        (prefix, "prefix_day", "observed_retention_pct"),
        (coordinates, "forecast_day", None),
    ):
        if not frame["protocol_id"].astype(str).eq(protocol.protocol_id).all():
            raise SyntheticProtocolError("Predictor protocol ID mismatch")
        if not frame["partition"].astype(str).isin(PREDICTOR_PARTITION_NAMES).all():
            raise SyntheticProtocolError("Predictor partition is invalid")
        if frame["cluster_id"].isna().any() or frame["cluster_id"].astype(str).eq("").any():
            raise SyntheticProtocolError("Predictor cluster IDs must be non-empty")
        frame[day_column] = pd.to_numeric(frame[day_column], errors="coerce")
        if frame[day_column].isna().any() or not np.isfinite(
            frame[day_column].to_numpy(dtype=float)
        ).all():
            raise SyntheticProtocolError("Predictor time coordinates must be finite")
        if value_column is not None:
            frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
            if frame[value_column].isna().any() or not np.isfinite(
                frame[value_column].to_numpy(dtype=float)
            ).all():
                raise SyntheticProtocolError("Prefix observations must be finite")
    prefix_clusters = set(
        zip(prefix["partition"].astype(str), prefix["cluster_id"].astype(str))
    )
    coordinate_clusters = set(
        zip(
            coordinates["partition"].astype(str),
            coordinates["cluster_id"].astype(str),
        )
    )
    if prefix_clusters != coordinate_clusters:
        raise SyntheticProtocolError("Prefix and forecast cluster sets differ")
    for key, group in prefix.groupby(["partition", "cluster_id"], sort=False):
        days = tuple(sorted(group["prefix_day"].to_numpy(dtype=float)))
        if days != protocol.prefix_days:
            raise SyntheticProtocolError(f"Cluster {key} prefix grid is incomplete")
    for key, group in coordinates.groupby(
        ["partition", "cluster_id"], sort=False
    ):
        days = tuple(sorted(group["forecast_day"].to_numpy(dtype=float)))
        if days != protocol.forecast_days:
            raise SyntheticProtocolError(f"Cluster {key} forecast grid is incomplete")
    return (
        prefix.sort_values(
            ["partition", "cluster_id", "prefix_day"], kind="stable"
        ).reset_index(drop=True),
        coordinates.sort_values(
            ["partition", "cluster_id", "forecast_day"], kind="stable"
        ).reset_index(drop=True),
    )


def _variant_diagnostic_record(
    variant: CandidateVariant,
    *,
    protocol_id: str,
    partition: str,
    cluster_id: str,
    prefix_sha256: str,
    credible: bool,
) -> dict[str, Any]:
    forecast = np.asarray(variant.forecast_retention_pct, dtype=float)
    return {
        "protocol_id": protocol_id,
        "partition": partition,
        "cluster_id": cluster_id,
        "model_id": variant.model_id,
        "variant_id": variant.variant_id,
        "fit_status": "succeeded" if variant.fit_succeeded else "failed",
        "credible_variant": bool(credible),
        "prefix_rmse_pp": (
            variant.prefix_rmse_pp if variant.fit_succeeded else None
        ),
        "prefix_max_abs_residual_pp": (
            variant.prefix_max_absolute_residual_pp
            if variant.fit_succeeded
            else None
        ),
        "forecast_min_pct": (
            float(np.min(forecast)) if variant.fit_succeeded else None
        ),
        "forecast_max_pct": (
            float(np.max(forecast)) if variant.fit_succeeded else None
        ),
        "canonical_prefix_content_sha256": prefix_sha256,
    }


def build_label_free_predictions(
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    protocol: ValidatedSyntheticProtocol,
) -> LabelFreePredictionResult:
    """Fit the frozen candidate from prefix values and future coordinates only."""
    prefix, coordinates = _validated_predictor_inputs(
        prefix_pack, forecast_coordinates, protocol
    )
    candidate = protocol.candidate_config()
    credible_rule = candidate["credible_member_rule"]
    rmse_slack = float(credible_rule["maximum_rmse_above_best_prefix_rmse_pp"])
    residual_cap = float(credible_rule["maximum_prefix_absolute_residual_pp"])
    forecast_lower, forecast_upper = map(
        float, credible_rule["forecast_retention_bounds_pct"]
    )
    minimum_families = int(
        credible_rule["minimum_structurally_distinct_family_ids"]
    )
    prediction_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []

    for (partition, cluster_id), prefix_group in prefix.groupby(
        ["partition", "cluster_id"], sort=True
    ):
        coordinate_group = coordinates.loc[
            coordinates["partition"].eq(partition)
            & coordinates["cluster_id"].eq(cluster_id)
        ].sort_values("forecast_day", kind="stable")
        ordered_prefix = prefix_group.sort_values("prefix_day", kind="stable")
        prefix_hash = canonical_prefix_content_sha256(
            ordered_prefix.loc[:, list(PREFIX_COLUMNS)]
        )
        forecast_days = coordinate_group["forecast_day"].to_numpy(dtype=float)
        variants = fit_structure_family_variants(
            ordered_prefix["prefix_day"].to_numpy(dtype=float),
            ordered_prefix["observed_retention_pct"].to_numpy(dtype=float),
            forecast_days,
            protocol,
        )
        successful = [item for item in variants if item.fit_succeeded]
        best_rmse = min(
            (item.prefix_rmse_pp for item in successful), default=math.inf
        )
        credibility: dict[tuple[str, str], bool] = {}
        for variant in variants:
            forecast = np.asarray(variant.forecast_retention_pct, dtype=float)
            credible = bool(
                variant.fit_succeeded
                and variant.prefix_rmse_pp <= best_rmse + rmse_slack
                and variant.prefix_max_absolute_residual_pp <= residual_cap
                and np.isfinite(forecast).all()
                and np.all(forecast >= forecast_lower)
                and np.all(forecast <= forecast_upper)
            )
            credibility[(variant.model_id, variant.variant_id)] = credible
            diagnostic_records.append(
                _variant_diagnostic_record(
                    variant,
                    protocol_id=protocol.protocol_id,
                    partition=str(partition),
                    cluster_id=str(cluster_id),
                    prefix_sha256=prefix_hash,
                    credible=credible,
                )
            )

        by_model: dict[str, list[np.ndarray]] = {}
        all_credible_forecasts: list[np.ndarray] = []
        for variant in variants:
            if credibility[(variant.model_id, variant.variant_id)]:
                values = np.asarray(variant.forecast_retention_pct, dtype=float)
                by_model.setdefault(variant.model_id, []).append(values)
                all_credible_forecasts.append(values)
        representatives = {
            model_id: np.median(np.vstack(values), axis=0)
            for model_id, values in by_model.items()
        }
        credible_family_count = len(representatives)
        baselines: dict[str, np.ndarray] = {}
        for model_id in (
            "target_prefix_persistence",
            "target_prefix_sqrt_time",
            "target_prefix_bounded_power_law",
        ):
            fitted = next(
                (item for item in variants if item.model_id == model_id and item.fit_succeeded),
                None,
            )
            baselines[model_id] = (
                np.asarray(fitted.forecast_retention_pct, dtype=float)
                if fitted is not None
                else np.full(len(forecast_days), np.nan)
            )
        bounded_fallback = baselines["target_prefix_bounded_power_law"]
        if credible_family_count >= minimum_families:
            point = np.median(np.vstack(list(representatives.values())), axis=0)
            stacked = np.vstack(all_credible_forecasts)
            envelope_low = np.min(stacked, axis=0)
            envelope_high = np.max(stacked, axis=0)
        else:
            point = bounded_fallback.copy()
            if all_credible_forecasts:
                stacked = np.vstack(all_credible_forecasts)
                envelope_low = np.min(stacked, axis=0)
                envelope_high = np.max(stacked, axis=0)
            else:
                envelope_low = bounded_fallback.copy()
                envelope_high = bounded_fallback.copy()
        for index, day in enumerate(forecast_days):
            prediction_records.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "partition": str(partition),
                    "cluster_id": str(cluster_id),
                    "forecast_day": float(day),
                    "candidate_point_forecast_pct": float(point[index]),
                    "persistence_forecast_pct": float(
                        baselines["target_prefix_persistence"][index]
                    ),
                    "sqrt_time_forecast_pct": float(
                        baselines["target_prefix_sqrt_time"][index]
                    ),
                    "bounded_power_forecast_pct": float(
                        baselines["target_prefix_bounded_power_law"][index]
                    ),
                    "structure_envelope_lower_pct": float(envelope_low[index]),
                    "structure_envelope_upper_pct": float(envelope_high[index]),
                    "canonical_prefix_content_sha256": prefix_hash,
                }
            )

    predictions = pd.DataFrame(prediction_records, columns=PREDICTION_COLUMNS)
    predictions = predictions.sort_values(
        ["partition", "cluster_id", "forecast_day"], kind="stable"
    ).reset_index(drop=True)
    diagnostics = pd.DataFrame(
        diagnostic_records, columns=MEMBER_DIAGNOSTIC_COLUMNS
    )
    diagnostics = diagnostics.sort_values(
        ["partition", "cluster_id", "model_id", "variant_id"], kind="stable"
    ).reset_index(drop=True)
    return LabelFreePredictionResult(
        prediction_bundle=predictions,
        member_diagnostics=diagnostics,
        prediction_sha256=canonical_csv_sha256(
            predictions, columns=PREDICTION_COLUMNS
        ),
        member_diagnostics_sha256=canonical_csv_sha256(
            diagnostics, columns=MEMBER_DIAGNOSTIC_COLUMNS
        ),
    )


def _sha256_string(value: str, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SyntheticProtocolError(f"{context} must be a lowercase SHA-256")
    return value


def build_disagreement_decisions(
    prediction_bundle: pd.DataFrame,
    member_diagnostics: pd.DataFrame,
    protocol: ValidatedSyntheticProtocol,
) -> DisagreementDecisionResult:
    """Rank hard-eligible trajectories without identity, family, or outcome inputs."""
    _validate_exact_bundle_columns(
        prediction_bundle, PREDICTION_COLUMNS, context="prediction_bundle"
    )
    _validate_exact_bundle_columns(
        member_diagnostics,
        MEMBER_DIAGNOSTIC_COLUMNS,
        context="member_fit_diagnostics",
    )
    if prediction_bundle.empty:
        raise SyntheticProtocolError("Prediction bundle cannot be empty")
    for frame, context in (
        (prediction_bundle, "prediction_bundle"),
        (member_diagnostics, "member_fit_diagnostics"),
    ):
        if not frame["protocol_id"].astype(str).eq(protocol.protocol_id).all():
            raise SyntheticProtocolError(f"{context} protocol ID mismatch")
        if not frame["partition"].astype(str).isin(PREDICTOR_PARTITION_NAMES).all():
            raise SyntheticProtocolError(f"{context} partition is invalid")
    if not member_diagnostics["model_id"].astype(str).isin(STRUCTURE_MEMBER_IDS).all():
        raise SyntheticProtocolError("Member diagnostics contain an undeclared model")
    if not member_diagnostics["fit_status"].astype(str).isin(
        {"succeeded", "failed"}
    ).all():
        raise SyntheticProtocolError("Member fit status is invalid")
    if not member_diagnostics["credible_variant"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise SyntheticProtocolError("Credible-variant flags must be booleans")
    if member_diagnostics.loc[
        member_diagnostics["fit_status"].eq("failed"), "credible_variant"
    ].any():
        raise SyntheticProtocolError("A failed member cannot be credible")
    if prediction_bundle.duplicated(
        ["partition", "cluster_id", "forecast_day"]
    ).any():
        raise SyntheticProtocolError("Prediction coordinates must be unique")
    if member_diagnostics.duplicated(
        ["partition", "cluster_id", "model_id", "variant_id"]
    ).any():
        raise SyntheticProtocolError("Member diagnostic coordinates must be unique")
    prediction_clusters = set(
        zip(
            prediction_bundle["partition"].astype(str),
            prediction_bundle["cluster_id"].astype(str),
        )
    )
    diagnostic_clusters = set(
        zip(
            member_diagnostics["partition"].astype(str),
            member_diagnostics["cluster_id"].astype(str),
        )
    )
    if prediction_clusters != diagnostic_clusters:
        raise SyntheticProtocolError("Prediction and diagnostic cluster sets differ")
    candidate = protocol.candidate_config()
    decision_rules = protocol.decision_config()
    test_target = int(
        candidate["primary_issuance_policy"][
            "required_eligible_test_cluster_count"
        ]
    )
    audit_target = int(decision_rules["minimum_eligible_audit_cluster_count"])
    decision_records: list[dict[str, Any]] = []
    target_issue_count = 0
    actual_issue_count = 0
    for partition, partition_predictions in prediction_bundle.groupby(
        "partition", sort=True
    ):
        partition_records: list[dict[str, Any]] = []
        for cluster_id, cluster_predictions in partition_predictions.groupby(
            "cluster_id", sort=True
        ):
            cluster_diagnostics = member_diagnostics.loc[
                member_diagnostics["partition"].eq(partition)
                & member_diagnostics["cluster_id"].eq(cluster_id)
            ]
            if set(cluster_diagnostics["model_id"].astype(str)) != set(
                STRUCTURE_MEMBER_IDS
            ):
                raise SyntheticProtocolError(
                    "Every cluster must record every declared structure family"
                )
            prefix_hashes = set(
                cluster_predictions["canonical_prefix_content_sha256"].astype(str)
            ) | set(
                cluster_diagnostics["canonical_prefix_content_sha256"].astype(str)
            )
            if len(prefix_hashes) != 1:
                raise SyntheticProtocolError("Canonical prefix commitment drifted")
            prefix_hash = _sha256_string(
                prefix_hashes.pop(), context="canonical_prefix_content_sha256"
            )
            if set(cluster_predictions["forecast_day"].astype(float)) != set(
                protocol.forecast_days
            ):
                raise SyntheticProtocolError("Prediction forecast grid is incomplete")
            credible = cluster_diagnostics.loc[
                cluster_diagnostics["credible_variant"].eq(True)  # noqa: E712
            ]
            credible_count = int(credible["model_id"].astype(str).nunique())
            failure_count = int(cluster_diagnostics["fit_status"].eq("failed").sum())
            finite_rmse = pd.to_numeric(
                cluster_diagnostics.loc[
                    cluster_diagnostics["fit_status"].eq("succeeded"),
                    "prefix_rmse_pp",
                ],
                errors="coerce",
            )
            best_rmse = (
                float(finite_rmse.min()) if finite_rmse.notna().any() else math.inf
            )
            low = pd.to_numeric(
                cluster_predictions["structure_envelope_lower_pct"], errors="coerce"
            ).to_numpy(dtype=float)
            high = pd.to_numeric(
                cluster_predictions["structure_envelope_upper_pct"], errors="coerce"
            ).to_numpy(dtype=float)
            point = pd.to_numeric(
                cluster_predictions["candidate_point_forecast_pct"], errors="coerce"
            ).to_numpy(dtype=float)
            if np.any(np.isfinite(low) & np.isfinite(high) & (low > high)):
                raise SyntheticProtocolError("Structure envelope lower exceeds upper")
            if credible_count < 2 or not np.isfinite(low).all() or not np.isfinite(high).all():
                disagreement = math.inf
            else:
                disagreement = float(np.max(high - low))
            hard_eligible = bool(
                credible_count >= 2
                and math.isfinite(disagreement)
                and np.isfinite(point).all()
            )
            reasons: list[str] = []
            if credible_count < 2:
                reasons.append("fewer_than_two_credible_structure_families")
            if not math.isfinite(disagreement):
                reasons.append("nonfinite_disagreement")
            if not np.isfinite(point).all():
                reasons.append("nonfinite_point_forecast")
            tie_hash = hashlib.sha256(
                f"{protocol.protocol_id}|{prefix_hash}".encode("utf-8")
            ).hexdigest()
            partition_records.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "partition": str(partition),
                    "cluster_id": str(cluster_id),
                    "canonical_prefix_content_sha256": prefix_hash,
                    "credible_structure_family_count": credible_count,
                    "fit_failure_count": failure_count,
                    "best_prefix_rmse_pp": best_rmse,
                    "disagreement_score_pp": disagreement,
                    "hard_eligible": hard_eligible,
                    "primary_issuance_rank": None,
                    "primary_issued": False,
                    "abstention_reasons": ";".join(reasons),
                    "_tie_hash": tie_hash,
                }
            )
        if partition == MATCHED_PARTITION:
            for item in partition_records:
                item["primary_issuance_rank"] = None
                item["primary_issued"] = False
                if item["hard_eligible"]:
                    item["abstention_reasons"] = (
                        "matched_pair_uses_frozen_calibration_threshold"
                    )
            decision_records.extend(partition_records)
            continue
        if partition == "development":
            for item in partition_records:
                item["primary_issuance_rank"] = None
                item["primary_issued"] = False
                if item["hard_eligible"]:
                    item["abstention_reasons"] = "development_partition_not_issued"
            decision_records.extend(partition_records)
            continue

        eligible = [item for item in partition_records if item["hard_eligible"]]
        tie_hashes = [item["_tie_hash"] for item in eligible]
        if len(tie_hashes) != len(set(tie_hashes)):
            raise SyntheticProtocolError(
                "Prefix-content tie-break collision prevents frozen ranking"
            )
        eligible.sort(
            key=lambda item: (item["disagreement_score_pp"], item["_tie_hash"])
        )
        if partition == "calibration":
            for rank, item in enumerate(eligible, start=1):
                item["primary_issuance_rank"] = rank
                item["primary_issued"] = False
                item["abstention_reasons"] = "calibration_partition_not_issued"
            decision_records.extend(partition_records)
            continue

        target = test_target if partition == "test" else audit_target
        target_issue_count += target
        for rank, item in enumerate(eligible, start=1):
            item["primary_issuance_rank"] = rank
            if rank <= target:
                item["primary_issued"] = True
                item["abstention_reasons"] = ""
                actual_issue_count += 1
            else:
                item["abstention_reasons"] = "ranked_above_issuance_quota"
        decision_records.extend(partition_records)
    for item in decision_records:
        item.pop("_tie_hash")
    decisions = pd.DataFrame(decision_records, columns=DECISION_COLUMNS)
    decisions = decisions.sort_values(
        ["partition", "cluster_id"], kind="stable"
    ).reset_index(drop=True)
    return DisagreementDecisionResult(
        decision_bundle=decisions,
        decision_sha256=canonical_csv_sha256(decisions, columns=DECISION_COLUMNS),
        target_issue_count=target_issue_count,
        actual_issue_count=actual_issue_count,
    )


def calibration_disagreement_threshold(
    decision_bundle: pd.DataFrame,
    protocol: ValidatedSyntheticProtocol,
) -> float:
    """Return the finite disagreement at frozen calibration rank 250."""
    _validate_exact_bundle_columns(
        decision_bundle, DECISION_COLUMNS, context="decision_bundle"
    )
    target_rank = int(
        protocol.decision_config()["minimum_eligible_calibration_cluster_count"]
    )
    calibration = decision_bundle.loc[
        decision_bundle["partition"].eq("calibration")
    ]
    if calibration.empty:
        raise SyntheticProtocolError("Calibration decisions are required")
    ranked = calibration.loc[calibration["hard_eligible"].eq(True)].copy()  # noqa: E712
    numeric_rank = pd.to_numeric(ranked["primary_issuance_rank"], errors="coerce")
    target = ranked.loc[numeric_rank.eq(float(target_rank))]
    if len(target) != 1 or len(ranked) < target_rank:
        raise SyntheticProtocolError(
            "At least 250 uniquely ranked hard-eligible calibration clusters are required"
        )
    threshold = _finite_float(
        target.iloc[0]["disagreement_score_pp"],
        context="calibration disagreement threshold",
    )
    if threshold < 0.0:
        raise SyntheticProtocolError("Calibration disagreement cannot be negative")
    return threshold


def evaluate_matched_pair_rejection(
    score_result: FrozenScoreResult,
    matched_prefix_pairs: pd.DataFrame,
    protocol: ValidatedSyntheticProtocol,
) -> MatchedPairAuditResult:
    """Apply the frozen threshold using only scorer-verified decisions."""
    if (
        not isinstance(score_result, FrozenScoreResult)
        or score_result._verification_token is not _VERIFIED_SCORE_TOKEN
    ):
        raise SyntheticProtocolError(
            "Matched audit requires a decision capability from the frozen scorer"
        )
    decision_bytes = score_result.verified_decision_bytes
    if hashlib.sha256(decision_bytes).hexdigest() != score_result.decision_sha256:
        raise SyntheticProtocolError("Verified decision capability was corrupted")
    try:
        decision_bundle = pd.read_csv(
            io.BytesIO(decision_bytes), float_precision="round_trip"
        )
    except Exception as exc:
        raise SyntheticProtocolError(
            "Verified decision capability is not a readable CSV"
        ) from exc
    _validate_exact_bundle_columns(
        decision_bundle, DECISION_COLUMNS, context="decision_bundle"
    )
    _validate_exact_bundle_columns(
        matched_prefix_pairs,
        MATCHED_PAIR_COLUMNS,
        context="matched_prefix_pairs",
    )
    matched = protocol.matched_config()
    required = int(matched["required_total_pair_clusters"])
    if len(matched_prefix_pairs) != required:
        raise SyntheticProtocolError(
            f"Matched-pair audit requires exactly {required} mapping rows"
        )
    if matched_prefix_pairs["pair_id"].duplicated().any():
        raise SyntheticProtocolError("Matched pair IDs must be unique")
    if not matched_prefix_pairs["protocol_id"].astype(str).eq(
        protocol.protocol_id
    ).all():
        raise SyntheticProtocolError("Matched-pair protocol ID mismatch")
    if not matched_prefix_pairs["left_family"].astype(str).eq(
        "single_power"
    ).all() or not matched_prefix_pairs["right_family"].astype(str).eq(
        "late_knee"
    ).all():
        raise SyntheticProtocolError("Matched-pair family mapping changed")
    expected_mapping = generate_all_matched_pair_packs(
        protocol
    ).matched_prefix_pairs.sort_values("pair_id", kind="stable").reset_index(drop=True)
    observed_mapping = matched_prefix_pairs.sort_values(
        "pair_id", kind="stable"
    ).reset_index(drop=True)
    identity_columns = [
        "protocol_id",
        "pair_id",
        "left_cluster_id",
        "right_cluster_id",
        "left_family",
        "right_family",
    ]
    if not observed_mapping.loc[:, identity_columns].equals(
        expected_mapping.loc[:, identity_columns]
    ):
        raise SyntheticProtocolError(
            "Matched-pair mapping differs from deterministic frozen construction"
        )
    metric_columns = list(MATCHED_PAIR_COLUMNS[len(identity_columns) :])
    observed_metrics = observed_mapping.loc[:, metric_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    expected_metrics = expected_mapping.loc[:, metric_columns].to_numpy(dtype=float)
    if not np.isfinite(observed_metrics).all() or not np.allclose(
        observed_metrics,
        expected_metrics,
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise SyntheticProtocolError(
            "Matched-pair metrics differ from deterministic frozen construction"
        )
    left_ids = matched_prefix_pairs["left_cluster_id"].astype(str)
    right_ids = matched_prefix_pairs["right_cluster_id"].astype(str)
    all_ids = list(left_ids) + list(right_ids)
    if len(set(all_ids)) != 2 * required:
        raise SyntheticProtocolError(
            "Every matched member must occur in exactly one sealed pair"
        )
    for name, limit, comparison in (
        (
            "latent_prefix_rmse_pp",
            float(matched["latent_prefix_rmse_max_pp"]),
            "maximum",
        ),
        (
            "latent_prefix_max_abs_difference_pp",
            float(matched["latent_prefix_max_absolute_difference_pp"]),
            "maximum",
        ),
        (
            "truth_separation_25y_pp",
            float(matched["minimum_25_year_truth_separation_pp"]),
            "minimum",
        ),
        (
            "max_forecast_truth_separation_pp",
            float(matched["minimum_maximum_forecast_grid_truth_separation_pp"]),
            "minimum",
        ),
    ):
        values = pd.to_numeric(matched_prefix_pairs[name], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise SyntheticProtocolError(f"Matched metric {name} must be finite")
        if comparison == "maximum" and np.any(values > limit):
            raise SyntheticProtocolError(f"Matched metric {name} exceeded its cap")
        if comparison == "minimum" and np.any(values < limit):
            raise SyntheticProtocolError(f"Matched metric {name} missed its floor")

    matched_decisions = decision_bundle.loc[
        decision_bundle["partition"].eq(MATCHED_PARTITION)
    ].copy()
    if matched_decisions["cluster_id"].astype(str).duplicated().any():
        raise SyntheticProtocolError("Matched decisions contain duplicate clusters")
    decision_ids = set(matched_decisions["cluster_id"].astype(str))
    if decision_ids != set(all_ids) or len(matched_decisions) != 2 * required:
        raise SyntheticProtocolError(
            "Matched decision cluster set differs from the sealed pair mapping"
        )
    target_rank = int(
        protocol.decision_config()["minimum_eligible_calibration_cluster_count"]
    )
    eligible_calibration_count = int(
        decision_bundle.loc[
            decision_bundle["partition"].eq("calibration")
            & decision_bundle["hard_eligible"].eq(True)  # noqa: E712
        ].shape[0]
    )
    threshold = (
        calibration_disagreement_threshold(decision_bundle, protocol)
        if eligible_calibration_count >= target_rank
        else None
    )
    score_by_id = {
        str(row.cluster_id): float(row.disagreement_score_pp)
        for row in matched_decisions.itertuples(index=False)
    }
    pair_records: list[dict[str, Any]] = []
    both_rejected_count = 0
    for row in matched_prefix_pairs.itertuples(index=False):
        left_score = score_by_id[str(row.left_cluster_id)]
        right_score = score_by_id[str(row.right_cluster_id)]
        left_rejected = bool(
            threshold is not None
            and math.isfinite(left_score)
            and left_score > threshold
        )
        right_rejected = bool(
            threshold is not None
            and math.isfinite(right_score)
            and right_score > threshold
        )
        both_rejected = bool(left_rejected and right_rejected)
        both_rejected_count += int(both_rejected)
        pair_records.append(
            {
                "pair_id": str(row.pair_id),
                "left_disagreement_score_pp": left_score,
                "right_disagreement_score_pp": right_score,
                "left_exceeds_threshold": left_rejected,
                "right_exceeds_threshold": right_rejected,
                "both_members_rejected": both_rejected,
            }
        )
    pair_scores = pd.DataFrame(pair_records).sort_values(
        "pair_id", kind="stable"
    ).reset_index(drop=True)
    return MatchedPairAuditResult(
        pair_scores=pair_scores,
        calibration_disagreement_threshold_pp=threshold,
        endpoint_available=threshold is not None,
        unavailable_reason=(
            None
            if threshold is not None
            else "fewer_than_250_hard_eligible_calibration_clusters"
        ),
        qualified_pair_count=required,
        both_rejected_pair_count=both_rejected_count,
        both_rejected_fraction=float(both_rejected_count / required),
    )


def _validate_complete_scoring_batch(
    prefix_pack: pd.DataFrame,
    prediction_bundle: pd.DataFrame,
    decision_bundle: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    member_diagnostics: pd.DataFrame,
    protocol: ValidatedSyntheticProtocol,
) -> None:
    """Reject truncation or cross-partition reuse before sealed truth is opened."""
    expected_cluster_counts = {
        partition: per_family * len(TRUTH_FAMILY_IDS)
        for partition, per_family in protocol.cluster_counts_per_truth_family
    }
    expected_cluster_counts[MATCHED_PARTITION] = (
        2 * int(protocol.matched_config()["required_total_pair_clusters"])
    )
    observed_partitions = set(decision_bundle["partition"].astype(str))
    if observed_partitions != set(PREDICTOR_PARTITION_NAMES):
        raise SyntheticProtocolError(
            "Frozen scorer requires every complete protocol partition"
        )
    if decision_bundle["cluster_id"].astype(str).duplicated().any():
        raise SyntheticProtocolError(
            "Opaque cluster IDs must be globally unique across partitions"
        )
    for partition, expected_clusters in expected_cluster_counts.items():
        partition_decisions = decision_bundle.loc[
            decision_bundle["partition"].eq(partition)
        ]
        if len(partition_decisions) != expected_clusters:
            raise SyntheticProtocolError(
                f"Partition {partition} must contain exactly "
                f"{expected_clusters} decision clusters"
            )
        expected_rows = expected_clusters * len(protocol.forecast_days)
        expected_prefix_rows = expected_clusters * len(protocol.prefix_days)
        if len(prefix_pack.loc[prefix_pack["partition"].eq(partition)]) != (
            expected_prefix_rows
        ):
            raise SyntheticProtocolError(
                f"Partition {partition} prefix-row cardinality is incomplete"
            )
        if len(
            prediction_bundle.loc[prediction_bundle["partition"].eq(partition)]
        ) != expected_rows or len(
            forecast_coordinates.loc[
                forecast_coordinates["partition"].eq(partition)
            ]
        ) != expected_rows:
            raise SyntheticProtocolError(
                f"Partition {partition} forecast-row cardinality is incomplete"
            )
        diagnostic_clusters = member_diagnostics.loc[
            member_diagnostics["partition"].eq(partition), "cluster_id"
        ].astype(str).nunique()
        if diagnostic_clusters != expected_clusters:
            raise SyntheticProtocolError(
                f"Partition {partition} member diagnostics are incomplete"
            )


def _validate_prediction_and_decision_freeze(
    prefix_pack: pd.DataFrame,
    prediction_bundle: pd.DataFrame,
    decision_bundle: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    member_diagnostics: pd.DataFrame,
    protocol: ValidatedSyntheticProtocol,
    *,
    expected_prefix_sha256: str,
    expected_prediction_sha256: str,
    expected_decision_sha256: str,
    expected_forecast_coordinates_sha256: str,
    expected_member_diagnostics_sha256: str,
) -> tuple[str, str, str, str, str]:
    """Verify frozen label-free bundles before a caller may expose truth."""
    _validate_exact_bundle_columns(prefix_pack, PREFIX_COLUMNS, context="prefix_pack")
    _validate_exact_bundle_columns(
        prediction_bundle, PREDICTION_COLUMNS, context="prediction_bundle"
    )
    _validate_exact_bundle_columns(
        decision_bundle, DECISION_COLUMNS, context="decision_bundle"
    )
    _validate_exact_bundle_columns(
        forecast_coordinates,
        FORECAST_COORDINATE_COLUMNS,
        context="forecast_coordinates",
    )
    _validate_exact_bundle_columns(
        member_diagnostics,
        MEMBER_DIAGNOSTIC_COLUMNS,
        context="member_fit_diagnostics",
    )
    expected_prefix = _sha256_string(
        expected_prefix_sha256, context="expected_prefix_sha256"
    )
    expected_prediction = _sha256_string(
        expected_prediction_sha256, context="expected_prediction_sha256"
    )
    expected_decision = _sha256_string(
        expected_decision_sha256, context="expected_decision_sha256"
    )
    expected_coordinates_hash = _sha256_string(
        expected_forecast_coordinates_sha256,
        context="expected_forecast_coordinates_sha256",
    )
    expected_diagnostics_hash = _sha256_string(
        expected_member_diagnostics_sha256,
        context="expected_member_diagnostics_sha256",
    )
    observed_prefix = canonical_csv_sha256(prefix_pack, columns=PREFIX_COLUMNS)
    observed_prediction = canonical_csv_sha256(
        prediction_bundle, columns=PREDICTION_COLUMNS
    )
    observed_decision = canonical_csv_sha256(
        decision_bundle, columns=DECISION_COLUMNS
    )
    observed_coordinates_hash = canonical_csv_sha256(
        forecast_coordinates, columns=FORECAST_COORDINATE_COLUMNS
    )
    observed_diagnostics_hash = canonical_csv_sha256(
        member_diagnostics, columns=MEMBER_DIAGNOSTIC_COLUMNS
    )
    if observed_prefix != expected_prefix:
        raise SyntheticProtocolError(
            "Prefix-pack bytes do not match the independently supplied commitment"
        )
    if observed_prediction != expected_prediction:
        raise SyntheticProtocolError(
            "Prediction bytes do not match the independently supplied freeze commitment"
        )
    if observed_decision != expected_decision:
        raise SyntheticProtocolError(
            "Decision bytes do not match the independently supplied freeze commitment"
        )
    if observed_coordinates_hash != expected_coordinates_hash:
        raise SyntheticProtocolError(
            "Forecast-coordinate bytes do not match the supplied commitment"
        )
    if observed_diagnostics_hash != expected_diagnostics_hash:
        raise SyntheticProtocolError(
            "Member-diagnostic bytes do not match the supplied commitment"
        )
    if (
        prefix_pack.empty
        or prediction_bundle.empty
        or decision_bundle.empty
        or forecast_coordinates.empty
        or member_diagnostics.empty
    ):
        raise SyntheticProtocolError("Frozen scoring bundles cannot be empty")
    for frame, context in (
        (prefix_pack, "prefix_pack"),
        (prediction_bundle, "prediction_bundle"),
        (decision_bundle, "decision_bundle"),
        (forecast_coordinates, "forecast_coordinates"),
        (member_diagnostics, "member_fit_diagnostics"),
    ):
        if not frame["protocol_id"].astype(str).eq(protocol.protocol_id).all():
            raise SyntheticProtocolError(f"{context} protocol ID mismatch")
        if not frame["partition"].astype(str).isin(PREDICTOR_PARTITION_NAMES).all():
            raise SyntheticProtocolError(f"{context} partition is invalid")
    _validated_predictor_inputs(prefix_pack, forecast_coordinates, protocol)
    prediction_keys = ["partition", "cluster_id", "forecast_day"]
    coordinate_keys = ["partition", "cluster_id", "forecast_day"]
    if prediction_bundle.duplicated(prediction_keys).any():
        raise SyntheticProtocolError("Prediction scoring keys must be unique")
    if forecast_coordinates.duplicated(coordinate_keys).any():
        raise SyntheticProtocolError("Forecast coordinate keys must be unique")
    expected_coordinates = set(
        forecast_coordinates.loc[:, coordinate_keys].itertuples(index=False, name=None)
    )
    observed_coordinates = set(
        prediction_bundle.loc[:, prediction_keys].itertuples(index=False, name=None)
    )
    if observed_coordinates != expected_coordinates or len(prediction_bundle) != len(
        forecast_coordinates
    ):
        raise SyntheticProtocolError(
            "Prediction coordinates are incomplete or differ from the frozen forecast pack"
        )
    decision_keys = ["partition", "cluster_id"]
    if decision_bundle.duplicated(decision_keys).any():
        raise SyntheticProtocolError("Decision scoring keys must be unique")
    prediction_clusters = set(
        prediction_bundle.loc[:, decision_keys].itertuples(index=False, name=None)
    )
    prefix_clusters = set(
        prefix_pack.loc[:, decision_keys].itertuples(index=False, name=None)
    )
    if prediction_clusters != prefix_clusters:
        raise SyntheticProtocolError("Prefix cluster set differs from predictions")
    decision_clusters = set(
        decision_bundle.loc[:, decision_keys].itertuples(index=False, name=None)
    )
    if prediction_clusters != decision_clusters:
        raise SyntheticProtocolError("Decision cluster set differs from predictions")
    diagnostic_clusters = set(
        member_diagnostics.loc[:, decision_keys].itertuples(index=False, name=None)
    )
    if prediction_clusters != diagnostic_clusters:
        raise SyntheticProtocolError("Diagnostic cluster set differs from predictions")
    for key, group in prediction_bundle.groupby(decision_keys, sort=False):
        if tuple(sorted(group["forecast_day"].astype(float))) != protocol.forecast_days:
            raise SyntheticProtocolError(f"Frozen prediction grid is incomplete for {key}")
        hashes = set(group["canonical_prefix_content_sha256"].astype(str))
        decision_hash = decision_bundle.loc[
            decision_bundle["partition"].eq(key[0])
            & decision_bundle["cluster_id"].eq(key[1]),
            "canonical_prefix_content_sha256",
        ].iloc[0]
        diagnostic_hashes = set(
            member_diagnostics.loc[
                member_diagnostics["partition"].eq(key[0])
                & member_diagnostics["cluster_id"].eq(key[1]),
                "canonical_prefix_content_sha256",
            ].astype(str)
        )
        prefix_group = prefix_pack.loc[
            prefix_pack["partition"].eq(key[0])
            & prefix_pack["cluster_id"].eq(key[1])
        ]
        recomputed_prefix_hash = canonical_prefix_content_sha256(prefix_group)
        if (
            len(hashes) != 1
            or len(diagnostic_hashes) != 1
            or hashes.pop() != recomputed_prefix_hash
            or diagnostic_hashes.pop() != recomputed_prefix_hash
            or str(decision_hash) != recomputed_prefix_hash
        ):
            raise SyntheticProtocolError(
                "Committed prefix hash differs across prefix, prediction, "
                "decision, or diagnostics"
            )
    for name in ("hard_eligible", "primary_issued"):
        if not decision_bundle[name].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise SyntheticProtocolError(f"{name} must contain strict booleans")
    if decision_bundle.loc[
        decision_bundle["primary_issued"].eq(True), "hard_eligible"  # noqa: E712
    ].ne(True).any():
        raise SyntheticProtocolError("An ineligible decision cannot be issued")
    matched = decision_bundle["partition"].eq(MATCHED_PARTITION)
    if decision_bundle.loc[matched, "primary_issued"].any() or decision_bundle.loc[
        matched, "primary_issuance_rank"
    ].notna().any():
        raise SyntheticProtocolError("Matched pairs cannot enter primary batch issuance")
    issued_keys = set(
        decision_bundle.loc[
            decision_bundle["primary_issued"].eq(True), decision_keys  # noqa: E712
        ].itertuples(index=False, name=None)
    )
    for key in issued_keys:
        values = prediction_bundle.loc[
            prediction_bundle["partition"].eq(key[0])
            & prediction_bundle["cluster_id"].eq(key[1]),
            "candidate_point_forecast_pct",
        ].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SyntheticProtocolError("An issued trajectory has a non-finite forecast")

    independently_rebuilt = build_disagreement_decisions(
        prediction_bundle,
        member_diagnostics,
        protocol,
    ).decision_bundle
    if canonical_csv_bytes(
        independently_rebuilt, columns=DECISION_COLUMNS
    ) != canonical_csv_bytes(decision_bundle, columns=DECISION_COLUMNS):
        raise SyntheticProtocolError(
            "Frozen decisions do not reproduce the declared gate and batch ranking"
        )
    _validate_complete_scoring_batch(
        prefix_pack,
        prediction_bundle,
        decision_bundle,
        forecast_coordinates,
        member_diagnostics,
        protocol,
    )
    return (
        observed_prefix,
        observed_prediction,
        observed_decision,
        observed_coordinates_hash,
        observed_diagnostics_hash,
    )


def _read_committed_csv(
    path: str | Path,
    *,
    expected_sha256: str,
    context: str,
) -> tuple[pd.DataFrame, str]:
    """Hash raw artifact bytes before parsing them as a dataframe."""
    expected = _sha256_string(expected_sha256, context=f"expected_{context}_sha256")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise SyntheticProtocolError(f"Cannot read committed {context} CSV") from exc
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise SyntheticProtocolError(
            f"{context} bytes do not match the independently supplied commitment"
        )
    try:
        frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except Exception as exc:
        raise SyntheticProtocolError(f"Committed {context} is not a readable CSV") from exc
    return frame, observed


def score_frozen_predictions(
    prefix_pack_path: str | Path,
    prediction_bundle_path: str | Path,
    decision_bundle_path: str | Path,
    forecast_coordinates_path: str | Path,
    member_diagnostics_path: str | Path,
    truth_pack_path: str | Path,
    protocol: ValidatedSyntheticProtocol,
    *,
    expected_prefix_sha256: str,
    expected_prediction_sha256: str,
    expected_decision_sha256: str,
    expected_forecast_coordinates_sha256: str,
    expected_member_diagnostics_sha256: str,
    expected_truth_sha256: str,
) -> FrozenScoreResult:
    """Verify all label-free commitments before reading sealed truth from disk."""
    prefix_pack, raw_prefix_sha256 = _read_committed_csv(
        prefix_pack_path,
        expected_sha256=expected_prefix_sha256,
        context="prefix_pack",
    )
    prediction_bundle, raw_prediction_sha256 = _read_committed_csv(
        prediction_bundle_path,
        expected_sha256=expected_prediction_sha256,
        context="prediction_bundle",
    )
    decision_bundle, raw_decision_sha256 = _read_committed_csv(
        decision_bundle_path,
        expected_sha256=expected_decision_sha256,
        context="decision_bundle",
    )
    forecast_coordinates, raw_coordinates_sha256 = _read_committed_csv(
        forecast_coordinates_path,
        expected_sha256=expected_forecast_coordinates_sha256,
        context="forecast_coordinates",
    )
    member_diagnostics, raw_diagnostics_sha256 = _read_committed_csv(
        member_diagnostics_path,
        expected_sha256=expected_member_diagnostics_sha256,
        context="member_diagnostics",
    )
    (
        observed_prefix,
        observed_prediction,
        observed_decision,
        observed_coordinates,
        observed_diagnostics,
    ) = _validate_prediction_and_decision_freeze(
        prefix_pack,
        prediction_bundle,
        decision_bundle,
        forecast_coordinates,
        member_diagnostics,
        protocol,
        expected_prefix_sha256=expected_prefix_sha256,
        expected_prediction_sha256=expected_prediction_sha256,
        expected_decision_sha256=expected_decision_sha256,
        expected_forecast_coordinates_sha256=expected_forecast_coordinates_sha256,
        expected_member_diagnostics_sha256=expected_member_diagnostics_sha256,
    )
    if (
        observed_prefix != raw_prefix_sha256
        or observed_prediction != raw_prediction_sha256
        or observed_decision != raw_decision_sha256
        or observed_coordinates != raw_coordinates_sha256
        or observed_diagnostics != raw_diagnostics_sha256
    ):
        raise SyntheticProtocolError(
            "Committed CSV bytes are not the frozen canonical serialization"
        )

    # No path access occurs until every label-free artifact and decision is verified.
    truth_pack, observed_truth = _read_committed_csv(
        truth_pack_path,
        expected_sha256=expected_truth_sha256,
        context="truth_pack",
    )
    _validate_exact_bundle_columns(truth_pack, TRUTH_PACK_COLUMNS, context="truth_pack")
    if truth_pack.empty:
        raise SyntheticProtocolError("Truth pack cannot be empty")
    if not truth_pack["protocol_id"].astype(str).eq(protocol.protocol_id).all():
        raise SyntheticProtocolError("Truth pack protocol ID mismatch")
    if not truth_pack["partition"].astype(str).isin(PREDICTOR_PARTITION_NAMES).all():
        raise SyntheticProtocolError("Truth pack partition is invalid")
    key_columns = ["partition", "cluster_id", "forecast_day"]
    if truth_pack.duplicated(key_columns).any():
        raise SyntheticProtocolError("Truth coordinates must be unique")
    truth_keys = set(
        truth_pack.loc[:, key_columns].itertuples(index=False, name=None)
    )
    prediction_keys = set(
        prediction_bundle.loc[:, key_columns].itertuples(index=False, name=None)
    )
    if truth_keys != prediction_keys or len(truth_pack) != len(prediction_bundle):
        raise SyntheticProtocolError("Truth coordinates do not exactly match predictions")
    truth_cluster_metadata = truth_pack.groupby(
        ["partition", "cluster_id"], sort=False
    )["truth_family"].nunique()
    if not truth_cluster_metadata.eq(1).all():
        raise SyntheticProtocolError("Each trajectory must have one truth family")
    for partition, per_family in protocol.cluster_counts_per_truth_family:
        partition_truth = truth_pack.loc[truth_pack["partition"].eq(partition)]
        family_counts = (
            partition_truth.loc[:, ["cluster_id", "truth_family"]]
            .drop_duplicates()
            .groupby("truth_family")["cluster_id"]
            .nunique()
            .to_dict()
        )
        if family_counts != {family: per_family for family in TRUTH_FAMILY_IDS}:
            raise SyntheticProtocolError(
                f"Partition {partition} truth-family strata are incomplete"
            )
    matched_truth = (
        truth_pack.loc[truth_pack["partition"].eq(MATCHED_PARTITION)]
        .loc[:, ["cluster_id", "truth_family"]]
        .drop_duplicates()
    )
    matched_family_counts = matched_truth.groupby("truth_family")[
        "cluster_id"
    ].nunique().to_dict()
    required_pairs = int(
        protocol.matched_config()["required_total_pair_clusters"]
    )
    if matched_family_counts != {
        "single_power": required_pairs,
        "late_knee": required_pairs,
    }:
        raise SyntheticProtocolError("Matched truth-family strata are incomplete")

    reconstructed_records: list[dict[str, Any]] = []
    for (partition, cluster_id), group in truth_pack.groupby(
        ["partition", "cluster_id"], sort=True
    ):
        families = set(group["truth_family"].astype(str))
        parameter_payloads = set(group["truth_parameters_json"].astype(str))
        if len(families) != 1 or len(parameter_payloads) != 1:
            raise SyntheticProtocolError("Truth metadata changed within a trajectory")
        family = families.pop()
        if family not in TRUTH_FAMILY_IDS:
            raise SyntheticProtocolError("Truth pack contains an undeclared family")
        try:
            parameters = json.loads(parameter_payloads.pop())
        except json.JSONDecodeError as exc:
            raise SyntheticProtocolError("Truth parameters are not valid JSON") from exc
        if not isinstance(parameters, dict):
            raise SyntheticProtocolError("Truth parameters must decode to an object")
        ordered = group.sort_values("forecast_day", kind="stable")
        days = ordered["forecast_day"].to_numpy(dtype=float)
        if tuple(days) != protocol.forecast_days:
            raise SyntheticProtocolError("Truth forecast grid is incomplete")
        latent = evaluate_truth_retention(
            family,
            parameters,
            days,
            time_scale_days=protocol.time_scale_days,
        )
        supplied_latent = pd.to_numeric(
            ordered["latent_retention_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        noisy = pd.to_numeric(
            ordered["noisy_retention_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(supplied_latent).all()
            or not np.isfinite(noisy).all()
            or not np.allclose(
                supplied_latent, latent, rtol=0.0, atol=1e-10, equal_nan=False
            )
        ):
            raise SyntheticProtocolError(
                "Truth pack failed independent latent-truth reconstruction"
            )
        for day, true_value, noisy_value in zip(days, latent, noisy):
            reconstructed_records.append(
                {
                    "partition": str(partition),
                    "cluster_id": str(cluster_id),
                    "forecast_day": float(day),
                    "truth_family": family,
                    "latent_retention_pct": float(true_value),
                    "noisy_retention_pct": float(noisy_value),
                }
            )
    reconstructed = pd.DataFrame(reconstructed_records)
    scored = prediction_bundle.merge(
        reconstructed,
        on=["partition", "cluster_id", "forecast_day"],
        how="left",
        validate="one_to_one",
    ).merge(
        decision_bundle,
        on=["protocol_id", "partition", "cluster_id", "canonical_prefix_content_sha256"],
        how="left",
        validate="many_to_one",
    )
    if len(scored) != len(prediction_bundle) or scored["truth_family"].isna().any():
        raise SyntheticProtocolError("Scoring join changed prediction cardinality")
    for forecast_name in (
        "candidate_point_forecast_pct",
        "persistence_forecast_pct",
        "sqrt_time_forecast_pct",
        "bounded_power_forecast_pct",
    ):
        scored[f"{forecast_name}_error_pp"] = (
            pd.to_numeric(scored[forecast_name], errors="coerce")
            - scored["latent_retention_pct"]
        )
        scored[f"{forecast_name}_absolute_error_pp"] = scored[
            f"{forecast_name}_error_pp"
        ].abs()

    catastrophe_threshold = 5.0
    trajectory_records: list[dict[str, Any]] = []
    for (partition, cluster_id), group in scored.groupby(
        ["partition", "cluster_id"], sort=True
    ):
        ordered = group.sort_values("forecast_day", kind="stable")
        days = ordered["forecast_day"].to_numpy(dtype=float)
        candidate_error = ordered[
            "candidate_point_forecast_pct_absolute_error_pp"
        ].to_numpy(dtype=float)
        finite = bool(np.isfinite(candidate_error).all())
        endpoint_error = float(candidate_error[-1]) if finite else math.inf
        iae = (
            float(np.trapezoid(candidate_error, days) / (days[-1] - days[0]))
            if finite
            else math.nan
        )
        decision = ordered.iloc[0]
        trajectory_records.append(
            {
                "partition": str(partition),
                "cluster_id": str(cluster_id),
                "truth_family": str(ordered["truth_family"].iloc[0]),
                "hard_eligible": bool(decision["hard_eligible"]),
                "primary_issued": bool(decision["primary_issued"]),
                "credible_structure_family_count": int(
                    decision["credible_structure_family_count"]
                ),
                "disagreement_score_pp": float(decision["disagreement_score_pp"]),
                "candidate_endpoint_absolute_error_pp": endpoint_error,
                "candidate_trajectory_iae_pp": iae,
                "catastrophic_error": bool(
                    not finite or endpoint_error >= catastrophe_threshold
                ),
            }
        )
    point_scores = scored.sort_values(
        ["partition", "cluster_id", "forecast_day"], kind="stable"
    ).reset_index(drop=True)
    trajectory_scores = pd.DataFrame(trajectory_records).sort_values(
        ["partition", "cluster_id"], kind="stable"
    ).reset_index(drop=True)
    return FrozenScoreResult(
        point_scores=point_scores,
        trajectory_scores=trajectory_scores,
        prediction_sha256=observed_prediction,
        decision_sha256=observed_decision,
        prefix_sha256=observed_prefix,
        forecast_coordinates_sha256=observed_coordinates,
        member_diagnostics_sha256=observed_diagnostics,
        truth_sha256=observed_truth,
        verified_decision_bytes=canonical_csv_bytes(
            decision_bundle, columns=DECISION_COLUMNS
        ),
        _verification_token=_VERIFIED_SCORE_TOKEN,
    )
