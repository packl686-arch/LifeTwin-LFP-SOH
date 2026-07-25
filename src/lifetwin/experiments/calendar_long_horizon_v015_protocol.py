from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FROZEN_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2"
FROZEN_CONFIG_COMMIT = "b8340f0"
FROZEN_CONFIG_BYTE_SHA256 = (
    "27dc7f89178f73779a52068c1878df26c9686faa7433686e60ba6496b6705796"
)
FROZEN_CONFIG_CANONICAL_SHA256 = (
    "704fe432c385b7e8223156f12a432afc264b695c45ced019bef55f529e694909"
)

CORE_FAMILY_IDS = (
    "single_power",
    "dual_power",
    "saturating_plus_slow",
    "early_activation_plus_power",
    "late_knee",
    "linear_drift_plus_power",
)
NOVEL_FAMILY_IDS = ("smooth_broken_power", "saturating_logistic_knee")
TRUTH_FAMILY_IDS = CORE_FAMILY_IDS + NOVEL_FAMILY_IDS
ORDINARY_PARTITIONS = (
    "center_development",
    "risk_development",
    "calibration",
    "test",
    "audit",
)
INTRINSIC_MATCHED_PARTITION = "intrinsic_matched_pairs"
STRESS_PLAN_MATCHED_PARTITION = "stress_plan_matched_pairs"
MATCHED_PARTITIONS = (
    INTRINSIC_MATCHED_PARTITION,
    STRESS_PLAN_MATCHED_PARTITION,
)

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
OPERATING_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    *REAL_OPERATING_FIELDS,
    *PLACEBO_FIELDS,
)
TRUTH_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "truth_family",
    "truth_parameters_json",
    "gamma",
    "forecast_day",
    "latent_retention_pct",
    "noisy_retention_pct",
)
MATCHED_PAIR_COLUMNS = (
    "protocol_id",
    "pair_partition",
    "pair_id",
    "left_cluster_id",
    "right_cluster_id",
    "construction_family",
    "left_side_code",
    "right_side_code",
    "latent_prefix_rmse_pp",
    "latent_prefix_max_abs_difference_pp",
    "truth_separation_25y_pp",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "freeze",
    "motivation",
    "scientific_questions",
    "evidence_role",
    "exposed_predecessor_data_policy",
    "time_grid",
    "design_partitions",
    "visible_information",
    "operating_covariate_generation",
    "truth_generation",
    "observation_model",
    "candidate_center",
    "risk_heads",
    "partial_identification",
    "comparators",
    "matched_audits",
    "endpoints",
    "decision_rules",
    "firewall_and_artifacts",
    "reporting",
    "claim_boundaries",
}
_SEED_ROOT_KEYS = {
    "center_development",
    "risk_development",
    "calibration",
    "test",
    "audit",
    "novel_mechanism_test",
    "novel_mechanism_audit",
    "intrinsic_matched_pairs",
    "stress_plan_matched_pairs",
    "random_rankings",
    "bootstrap",
    "stress_permutations",
    "placebo_covariate",
}
_EXPECTED_SEED_ROOTS = {
    "center_development": 202607230101,
    "risk_development": 202607230102,
    "calibration": 202607230103,
    "test": 202607230104,
    "audit": 202607230105,
    "novel_mechanism_test": 202607230106,
    "novel_mechanism_audit": 202607230107,
    "intrinsic_matched_pairs": 202607230108,
    "stress_plan_matched_pairs": 202607230109,
    "random_rankings": 202607230110,
    "bootstrap": 202607230111,
    "stress_permutations": 202607230112,
    "placebo_covariate": 202607230113,
}
_EXPECTED_TRUTH_FILE_ROWS = {
    "center_development_truth.csv": 4800,
    "risk_development_truth.csv": 4800,
    "calibration_truth.csv": 7200,
    "test_truth.csv": 15200,
    "audit_truth.csv": 7600,
    "intrinsic_matched_truth.csv": 4000,
    "stress_plan_matched_truth.csv": 4000,
}
_EXPECTED_PARAMETER_LAYOUT: dict[str, tuple[tuple[str, str, str], ...]] = {
    "single_power": (("a", "a", "uniform"), ("b", "b", "uniform")),
    "dual_power": (
        ("a1", "a1", "uniform"),
        ("b1", "b1", "uniform"),
        ("a2", "a2", "uniform"),
        ("b2", "b2", "uniform"),
    ),
    "saturating_plus_slow": (
        ("a_sat", "a_sat", "uniform"),
        ("tau_sat_days_log_uniform", "tau_sat_days", "log_uniform"),
        ("b_sat", "b_sat", "uniform"),
        ("a_slow", "a_slow", "uniform"),
        ("b_slow", "b_slow", "uniform"),
    ),
    "early_activation_plus_power": (
        ("a", "a", "uniform"),
        ("b", "b", "uniform"),
        ("activation_amplitude_pp", "activation_amplitude_pp", "uniform"),
        ("tau_rise_days_log_uniform", "tau_rise_days", "log_uniform"),
        ("tau_decay_days_log_uniform", "tau_decay_days", "log_uniform"),
    ),
    "late_knee": (
        ("a", "a", "uniform"),
        ("b", "b", "uniform"),
        ("k_pp_per_day_log_uniform", "k_pp_per_day", "log_uniform"),
        ("t_knee_days", "t_knee_days", "uniform"),
        ("w_days_log_uniform", "w_days", "log_uniform"),
    ),
    "linear_drift_plus_power": (
        ("a", "a", "uniform"),
        ("b", "b", "uniform"),
        ("c", "c", "uniform"),
    ),
    "smooth_broken_power": (
        ("a", "a", "uniform"),
        ("b_early", "b_early", "uniform"),
        ("b_late", "b_late", "uniform"),
        (
            "transition_tau_days_log_uniform",
            "transition_tau_days",
            "log_uniform",
        ),
        ("sharpness", "sharpness", "uniform"),
    ),
    "saturating_logistic_knee": (
        ("a", "a", "uniform"),
        ("b", "b", "uniform"),
        ("knee_amplitude_pp", "knee_amplitude_pp", "uniform"),
        ("t_knee_days", "t_knee_days", "uniform"),
        ("w_days_log_uniform", "w_days", "log_uniform"),
    ),
}
_EXPECTED_PARTITION_COUNTS: dict[str, dict[str, int]] = {
    "center_development": {family: 100 for family in CORE_FAMILY_IDS},
    "risk_development": {family: 100 for family in CORE_FAMILY_IDS},
    "calibration": {family: 150 for family in CORE_FAMILY_IDS},
    "test": {
        **{family: 250 for family in CORE_FAMILY_IDS},
        "smooth_broken_power": 200,
        "saturating_logistic_knee": 200,
    },
    "audit": {
        "single_power": 75,
        "dual_power": 100,
        "saturating_plus_slow": 125,
        "early_activation_plus_power": 150,
        "late_knee": 200,
        "linear_drift_plus_power": 100,
        "smooth_broken_power": 100,
        "saturating_logistic_knee": 100,
    },
}


class V015ProtocolError(ValueError):
    """Raised when the frozen V0.15 protocol or a generated object is invalid."""


@dataclass(frozen=True)
class ParameterDefinition:
    config_name: str
    parameter_name: str
    distribution: str
    minimum: float
    maximum: float


@dataclass(frozen=True)
class TruthFamilyDefinition:
    family_id: str
    loss_formula: str
    parameters: tuple[ParameterDefinition, ...]


@dataclass(frozen=True)
class ValidatedV015Protocol:
    protocol_id: str
    config_sha256: str
    time_scale_days: float
    prefix_days: tuple[float, ...]
    forecast_days: tuple[float, ...]
    family_definitions: tuple[TruthFamilyDefinition, ...]
    partition_family_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    seed_roots: tuple[tuple[str, int], ...]
    operating_support: tuple[tuple[str, tuple[float, float]], ...]
    truth_support_pct: tuple[float, float]
    maximum_single_interval_change_pp: float
    maximum_parameter_attempts: int
    standard_sigma_bounds: tuple[float, float]
    standard_rho_bounds: tuple[float, float]
    audit_sigma_bounds: tuple[float, float]
    audit_rho_bounds: tuple[float, float]
    config_json: str

    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)

    def family_map(self) -> dict[str, TruthFamilyDefinition]:
        return {item.family_id: item for item in self.family_definitions}

    def partition_count_map(self) -> dict[str, dict[str, int]]:
        return {
            partition: dict(counts)
            for partition, counts in self.partition_family_counts
        }

    def seed_root_map(self) -> dict[str, int]:
        return dict(self.seed_roots)

    def support_map(self) -> dict[str, tuple[float, float]]:
        return dict(self.operating_support)

    @property
    def combined_days(self) -> tuple[float, ...]:
        return self.prefix_days + self.forecast_days


@dataclass(frozen=True)
class OperatingCovariates:
    past_mean_temperature_c: float
    past_mean_soc_fraction: float
    past_mean_dod_fraction: float
    past_efc_per_year: float
    planned_mean_temperature_c: float
    planned_mean_soc_fraction: float
    planned_mean_dod_fraction: float
    planned_efc_per_year: float
    placebo_controls: tuple[float, ...] = (0.0,) * 8

    def __post_init__(self) -> None:
        values = (*self.real_values(), *self.placebo_controls)
        if len(self.placebo_controls) != 8 or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise V015ProtocolError(
                "Operating covariates require sixteen finite values"
            )

    def real_values(self) -> tuple[float, ...]:
        return (
            float(self.past_mean_temperature_c),
            float(self.past_mean_soc_fraction),
            float(self.past_mean_dod_fraction),
            float(self.past_efc_per_year),
            float(self.planned_mean_temperature_c),
            float(self.planned_mean_soc_fraction),
            float(self.planned_mean_dod_fraction),
            float(self.planned_efc_per_year),
        )

    def as_record(self) -> dict[str, float]:
        values = self.real_values()
        record = {
            name: float(value) for name, value in zip(REAL_OPERATING_FIELDS, values)
        }
        record.update(
            {
                name: float(value)
                for name, value in zip(PLACEBO_FIELDS, self.placebo_controls)
            }
        )
        return record


@dataclass(frozen=True)
class ObservationNoise:
    sigma_pp: float
    rho: float
    errors_pp: tuple[float, ...]


@dataclass(frozen=True)
class TruthSpec:
    cluster_id: str
    partition: str
    family_id: str
    cluster_index: int
    parameters: tuple[tuple[str, float], ...]
    gamma: float
    operating_seed: int
    placebo_seed: int
    truth_seed: int
    measurement_seed: int
    accepted_attempt: int

    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class GeneratedMemberPacks:
    prefix_pack: pd.DataFrame
    forecast_coordinates: pd.DataFrame
    operating_pack: pd.DataFrame
    truth_pack: pd.DataFrame


@dataclass(frozen=True)
class OrdinaryGeneratedCluster:
    truth_spec: TruthSpec
    operating: OperatingCovariates
    noise: ObservationNoise
    packs: GeneratedMemberPacks


@dataclass(frozen=True)
class MatchedPairPacks:
    prefix_pack: pd.DataFrame
    forecast_coordinates: pd.DataFrame
    operating_pack: pd.DataFrame
    truth_pack: pd.DataFrame
    matched_pairs: pd.DataFrame


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V015ProtocolError("Protocol values must be finite JSON") from exc
    return text.encode("ascii")


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V015ProtocolError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise V015ProtocolError(
            f"{context} keys changed; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise V015ProtocolError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise V015ProtocolError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise V015ProtocolError(f"{context} must be a finite number")
    return result


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise V015ProtocolError(f"{context} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise V015ProtocolError(f"{context} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise V015ProtocolError(f"{context} must be a positive integer")
    return int(numeric)


def _number_pair(value: Any, *, context: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise V015ProtocolError(f"{context} must be a two-number array")
    lower = _finite_float(value[0], context=f"{context}[0]")
    upper = _finite_float(value[1], context=f"{context}[1]")
    if not lower < upper:
        raise V015ProtocolError(f"{context} bounds must be strictly increasing")
    return lower, upper


def _strict_grid(value: Any, *, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise V015ProtocolError(f"{context} must be a non-empty array")
    grid = tuple(
        _finite_float(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if grid[0] < 0.0 or any(right <= left for left, right in zip(grid, grid[1:])):
        raise V015ProtocolError(f"{context} must be nonnegative and increasing")
    return grid


def _validate_family_definitions(
    truth_generation: Mapping[str, Any],
) -> tuple[TruthFamilyDefinition, ...]:
    if tuple(truth_generation["core_families"]) != CORE_FAMILY_IDS:
        raise V015ProtocolError("Frozen core truth-family order changed")
    if (
        tuple(
            truth_generation["novel_families_absent_from_center_risk_and_calibration"]
        )
        != NOVEL_FAMILY_IDS
    ):
        raise V015ProtocolError("Frozen novel truth-family order changed")
    raw_specs = truth_generation["base_family_specs"]
    if not isinstance(raw_specs, list):
        raise V015ProtocolError("truth_generation.base_family_specs must be an array")
    if tuple(item.get("family_id") for item in raw_specs) != TRUTH_FAMILY_IDS:
        raise V015ProtocolError("Frozen truth-family specification order changed")

    definitions: list[TruthFamilyDefinition] = []
    for raw in raw_specs:
        mapping = _require_mapping(raw, context="truth family")
        _require_exact_keys(
            mapping,
            {"family_id", "loss_formula", "parameters"},
            context=f"truth family {mapping.get('family_id')}",
        )
        family_id = str(mapping["family_id"])
        formula = mapping["loss_formula"]
        if not isinstance(formula, str) or not formula:
            raise V015ProtocolError(f"{family_id} loss formula must be non-empty")
        raw_parameters = _require_mapping(
            mapping["parameters"], context=f"{family_id}.parameters"
        )
        expected_layout = _EXPECTED_PARAMETER_LAYOUT[family_id]
        expected_config_names = tuple(item[0] for item in expected_layout)
        if tuple(raw_parameters) != expected_config_names:
            raise V015ProtocolError(
                f"{family_id} parameter order or membership changed"
            )
        parameters: list[ParameterDefinition] = []
        for config_name, parameter_name, distribution in expected_layout:
            minimum, maximum = _number_pair(
                raw_parameters[config_name],
                context=f"{family_id}.{config_name}",
            )
            if distribution == "log_uniform" and minimum <= 0.0:
                raise V015ProtocolError(
                    f"{family_id}.{config_name} log bounds must be positive"
                )
            parameters.append(
                ParameterDefinition(
                    config_name=config_name,
                    parameter_name=parameter_name,
                    distribution=distribution,
                    minimum=minimum,
                    maximum=maximum,
                )
            )
        definitions.append(
            TruthFamilyDefinition(
                family_id=family_id,
                loss_formula=formula,
                parameters=tuple(parameters),
            )
        )
    return tuple(definitions)


def validate_protocol_config(config: Mapping[str, Any]) -> ValidatedV015Protocol:
    """Validate the complete frozen design without generating any V2 outcome."""
    root = _require_mapping(config, context="protocol")
    _require_exact_keys(root, _TOP_LEVEL_KEYS, context="protocol")
    if root["schema_version"] != "2.0.0":
        raise V015ProtocolError("V0.15 protocol schema version changed")
    if root["protocol_id"] != FROZEN_PROTOCOL_ID:
        raise V015ProtocolError("V0.15 protocol ID changed")
    if root["status"] != "frozen_before_v2_implementation_or_simulation_execution":
        raise V015ProtocolError("V0.15 freeze status changed")

    time_grid = _require_mapping(root["time_grid"], context="time_grid")
    prefix_days = _strict_grid(time_grid["prefix_days"], context="prefix_days")
    forecast_days = _strict_grid(time_grid["forecast_days"], context="forecast_days")
    if (
        len(prefix_days) != 12
        or len(forecast_days) != 8
        or prefix_days[-1] != 730.0
        or forecast_days[-1] != 9131.25
        or forecast_days[0] <= prefix_days[-1]
    ):
        raise V015ProtocolError("Frozen V0.15 time grid changed")
    time_scale = _finite_float(
        time_grid["days_per_year_for_labels"], context="days_per_year_for_labels"
    )
    if time_scale != 365.25:
        raise V015ProtocolError("Frozen year length changed")

    truth_generation = _require_mapping(
        root["truth_generation"], context="truth_generation"
    )
    family_definitions = _validate_family_definitions(truth_generation)
    truth_support = _number_pair(
        truth_generation["truth_support_pct"], context="truth_support_pct"
    )
    admissibility = _require_mapping(
        truth_generation["admissibility"], context="truth_generation.admissibility"
    )
    maximum_change = _finite_float(
        admissibility["maximum_single_grid_interval_change_pp"],
        context="maximum_single_grid_interval_change_pp",
    )
    maximum_attempts = _positive_int(
        admissibility["maximum_parameter_attempts_with_frozen_covariates"],
        context="maximum_parameter_attempts_with_frozen_covariates",
    )

    design = _require_mapping(root["design_partitions"], context="design_partitions")
    partition_counts: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    ordinary_total = 0
    for partition in ORDINARY_PARTITIONS:
        partition_config = _require_mapping(
            design[partition], context=f"design_partitions.{partition}"
        )
        raw_counts = _require_mapping(
            partition_config["families"], context=f"{partition}.families"
        )
        observed_counts = {
            str(family): _positive_int(count, context=f"{partition}.families.{family}")
            for family, count in raw_counts.items()
        }
        expected_counts = _EXPECTED_PARTITION_COUNTS[partition]
        if observed_counts != expected_counts or tuple(raw_counts) != tuple(
            expected_counts
        ):
            raise V015ProtocolError(f"Frozen {partition} family counts changed")
        total = sum(observed_counts.values())
        if (
            _positive_int(
                partition_config["total_clusters"],
                context=f"{partition}.total_clusters",
            )
            != total
        ):
            raise V015ProtocolError(f"{partition} total does not match family counts")
        ordinary_total += total
        partition_counts.append(
            (partition, tuple((family, count) for family, count in raw_counts.items()))
        )
    if ordinary_total != 4950 or int(design["total_ordinary_clusters"]) != 4950:
        raise V015ProtocolError("Frozen ordinary-cluster total changed")
    for partition in MATCHED_PARTITIONS:
        item = _require_mapping(design[partition], context=partition)
        if int(item["pair_count"]) != 250 or int(item["member_count"]) != 500:
            raise V015ProtocolError(f"Frozen {partition} size changed")
    if (
        int(design["total_matched_members"]) != 1000
        or int(design["total_generated_members"]) != 5950
    ):
        raise V015ProtocolError("Frozen matched or generated total changed")

    raw_roots = _require_mapping(design["seed_roots"], context="seed_roots")
    _require_exact_keys(raw_roots, _SEED_ROOT_KEYS, context="seed_roots")
    seed_roots = tuple(
        (
            str(name),
            _positive_int(value, context=f"seed_roots.{name}"),
        )
        for name, value in raw_roots.items()
    )
    if len({root_value for _, root_value in seed_roots}) != len(seed_roots):
        raise V015ProtocolError("Frozen seed roots must be unique")
    if dict(seed_roots) != _EXPECTED_SEED_ROOTS:
        raise V015ProtocolError("Frozen seed-root values changed")

    operating_config = _require_mapping(
        root["operating_covariate_generation"],
        context="operating_covariate_generation",
    )
    raw_support = _require_mapping(
        operating_config["ordinary_support"], context="ordinary_support"
    )
    expected_support_keys = set(REAL_OPERATING_FIELDS) | {"placebo_control_1_through_8"}
    _require_exact_keys(raw_support, expected_support_keys, context="ordinary_support")
    support = tuple(
        (
            field,
            _number_pair(raw_support[field], context=f"ordinary_support.{field}"),
        )
        for field in REAL_OPERATING_FIELDS
    )

    observation = _require_mapping(
        root["observation_model"], context="observation_model"
    )
    standard_sigma = _number_pair(
        observation["standard_sigma_pp"], context="standard_sigma_pp"
    )
    standard_rho = _number_pair(observation["standard_rho"], context="standard_rho")
    audit_sigma = _number_pair(observation["audit_sigma_pp"], context="audit_sigma_pp")
    audit_rho = _number_pair(observation["audit_rho"], context="audit_rho")
    if not (
        0.0 <= standard_rho[0] < standard_rho[1] < 1.0
        and 0.0 <= audit_rho[0] < audit_rho[1] < 1.0
    ):
        raise V015ProtocolError("AR(1) rho bounds must lie in [0,1)")

    schemas = _require_mapping(
        _require_mapping(
            root["firewall_and_artifacts"], context="firewall_and_artifacts"
        )["artifact_schemas"],
        context="artifact_schemas",
    )
    expected_members = 5950
    expected_prefix_rows = expected_members * len(prefix_days)
    expected_forecast_rows = expected_members * len(forecast_days)
    if int(schemas["prefix_pack.csv"]["required_rows"]) != expected_prefix_rows:
        raise V015ProtocolError("Prefix artifact row count is inconsistent")
    if (
        int(schemas["forecast_coordinates.csv"]["required_rows"])
        != expected_forecast_rows
    ):
        raise V015ProtocolError("Forecast artifact row count is inconsistent")
    if int(schemas["prediction_bundle.csv"]["required_rows"]) != expected_forecast_rows:
        raise V015ProtocolError("Prediction artifact row count is inconsistent")
    if int(schemas["operating_pack.csv"]["required_rows"]) != expected_members:
        raise V015ProtocolError("Operating artifact row count is inconsistent")
    if tuple(schemas["prefix_pack.csv"]["columns"]) != PREFIX_COLUMNS:
        raise V015ProtocolError("Frozen prefix schema changed")
    if tuple(schemas["prefix_pack.csv"]["key"]) != (
        "partition",
        "cluster_id",
        "prefix_day",
    ):
        raise V015ProtocolError("Frozen prefix key changed")
    if tuple(schemas["forecast_coordinates.csv"]["columns"]) != (
        FORECAST_COORDINATE_COLUMNS
    ):
        raise V015ProtocolError("Frozen forecast-coordinate schema changed")
    if tuple(schemas["forecast_coordinates.csv"]["key"]) != (
        "partition",
        "cluster_id",
        "forecast_day",
    ):
        raise V015ProtocolError("Frozen forecast-coordinate key changed")
    if tuple(schemas["operating_pack.csv"]["columns"]) != OPERATING_COLUMNS:
        raise V015ProtocolError("Frozen operating schema changed")
    if tuple(schemas["operating_pack.csv"]["key"]) != (
        "partition",
        "cluster_id",
    ):
        raise V015ProtocolError("Frozen operating key changed")
    if tuple(schemas["truth_csv_family"]["columns"]) != TRUTH_COLUMNS:
        raise V015ProtocolError("Frozen truth schema changed")
    if tuple(schemas["truth_csv_family"]["key"]) != (
        "partition",
        "cluster_id",
        "forecast_day",
    ):
        raise V015ProtocolError("Frozen truth key changed")
    if tuple(schemas["matched_pair_csvs"]["columns"]) != MATCHED_PAIR_COLUMNS:
        raise V015ProtocolError("Frozen matched-pair schema changed")
    if tuple(schemas["matched_pair_csvs"]["key"]) != (
        "pair_partition",
        "pair_id",
    ):
        raise V015ProtocolError("Frozen matched-pair key changed")
    if int(schemas["matched_pair_csvs"]["required_rows_each"]) != 250:
        raise V015ProtocolError("Frozen matched-pair row count changed")
    truth_file_rows = schemas["truth_csv_family"]["exact_file_rows"]
    if {
        str(name): int(value) for name, value in truth_file_rows.items()
    } != _EXPECTED_TRUTH_FILE_ROWS:
        raise V015ProtocolError("Split truth-file row counts are inconsistent")

    canonical = _canonical_json_bytes(root)
    insertion_order_json = json.dumps(
        root,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return ValidatedV015Protocol(
        protocol_id=FROZEN_PROTOCOL_ID,
        config_sha256=hashlib.sha256(canonical).hexdigest(),
        time_scale_days=time_scale,
        prefix_days=prefix_days,
        forecast_days=forecast_days,
        family_definitions=family_definitions,
        partition_family_counts=tuple(partition_counts),
        seed_roots=seed_roots,
        operating_support=support,
        truth_support_pct=truth_support,
        maximum_single_interval_change_pp=maximum_change,
        maximum_parameter_attempts=maximum_attempts,
        standard_sigma_bounds=standard_sigma,
        standard_rho_bounds=standard_rho,
        audit_sigma_bounds=audit_sigma,
        audit_rho_bounds=audit_rho,
        config_json=insertion_order_json,
    )


def load_frozen_protocol_config(path: str | Path) -> ValidatedV015Protocol:
    """Load only the byte-exact V2 file committed by the preregistration freeze."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise V015ProtocolError("Cannot read frozen V0.15 protocol") from exc
    if hashlib.sha256(raw).hexdigest() != FROZEN_CONFIG_BYTE_SHA256:
        raise V015ProtocolError("Protocol file byte SHA-256 differs from frozen V2")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V015ProtocolError("Frozen V2 protocol is not valid UTF-8 JSON") from exc
    canonical = _canonical_json_bytes(payload)
    if hashlib.sha256(canonical).hexdigest() != FROZEN_CONFIG_CANONICAL_SHA256:
        raise V015ProtocolError("Protocol canonical SHA-256 differs from frozen V2")
    protocol = validate_protocol_config(payload)
    if protocol.config_sha256 != FROZEN_CONFIG_CANONICAL_SHA256:
        raise V015ProtocolError("Validated V2 protocol commitment drifted")
    return protocol


def derive_stream_seed(
    protocol_id: str,
    seed_root: int,
    partition: str,
    family_id: str,
    zero_based_index: int,
    stream_name: str,
) -> int:
    """Apply the frozen first-16-hex seed derivation to arbitrary identifiers."""
    if (
        not protocol_id
        or not partition
        or not family_id
        or not stream_name
        or isinstance(zero_based_index, bool)
        or int(zero_based_index) != zero_based_index
        or zero_based_index < 0
        or isinstance(seed_root, bool)
        or int(seed_root) != seed_root
        or seed_root < 1
    ):
        raise V015ProtocolError("Seed derivation inputs are invalid")
    material = (
        f"{protocol_id}|{int(seed_root)}|{partition}|{family_id}|"
        f"{int(zero_based_index)}|{stream_name}"
    )
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16) % (
        2**63 - 1
    )


def _ordinary_seed_root(
    protocol: ValidatedV015Protocol, partition: str, family_id: str
) -> int:
    roots = protocol.seed_root_map()
    if family_id in NOVEL_FAMILY_IDS:
        if partition not in {"test", "audit"}:
            raise V015ProtocolError("Novel families are allowed only in test and audit")
        return roots[f"novel_mechanism_{partition}"]
    if family_id not in CORE_FAMILY_IDS or partition not in ORDINARY_PARTITIONS:
        raise V015ProtocolError("Ordinary partition or family is invalid")
    return roots[partition]


def derive_ordinary_stream_seeds(
    protocol: ValidatedV015Protocol,
    *,
    partition: str,
    family_id: str,
    zero_based_index: int,
) -> dict[str, int]:
    """Derive the five declared ordinary streams without consuming any RNG."""
    counts = protocol.partition_count_map()
    if (
        partition not in counts
        or family_id not in counts[partition]
        or zero_based_index < 0
        or zero_based_index >= counts[partition][family_id]
    ):
        raise V015ProtocolError("Ordinary stream coordinate is outside the design")
    root = _ordinary_seed_root(protocol, partition, family_id)
    placebo_root = protocol.seed_root_map()["placebo_covariate"]
    result = {
        stream: derive_stream_seed(
            protocol.protocol_id,
            root,
            partition,
            family_id,
            zero_based_index,
            stream,
        )
        for stream in (
            "opaque_id",
            "operating_covariates",
            "truth_parameters",
            "measurement_noise",
        )
    }
    result["placebo_covariates"] = derive_stream_seed(
        protocol.protocol_id,
        placebo_root,
        partition,
        family_id,
        zero_based_index,
        "placebo_covariates",
    )
    if len(set(result.values())) != len(result):
        raise V015ProtocolError("Derived ordinary stream seed collision")
    return result


def derive_ordinary_cluster_id(
    protocol: ValidatedV015Protocol,
    *,
    partition: str,
    family_id: str,
    zero_based_index: int,
) -> str:
    seeds = derive_ordinary_stream_seeds(
        protocol,
        partition=partition,
        family_id=family_id,
        zero_based_index=zero_based_index,
    )
    material = f"{protocol.protocol_id}|{seeds['opaque_id']}|opaque_cluster_id"
    return "c_" + hashlib.sha256(material.encode("ascii")).hexdigest()[:32]


def validate_unique_stream_seeds(specs: Sequence[TruthSpec]) -> None:
    observed: set[int] = set()
    for spec in specs:
        for seed in (
            spec.operating_seed,
            spec.placebo_seed,
            spec.truth_seed,
            spec.measurement_seed,
        ):
            if seed in observed:
                raise V015ProtocolError("A generated stream seed collided")
            observed.add(seed)


def _generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(int(seed)))


def _draw_real_operating(
    protocol: ValidatedV015Protocol,
    rng: np.random.Generator,
    *,
    audit: bool,
) -> tuple[float, ...]:
    values: list[float] = []
    for field, (lower, upper) in protocol.operating_support:
        unit = float(rng.beta(0.7, 0.7)) if audit else float(rng.uniform())
        values.append(lower + unit * (upper - lower))
    return tuple(values)


def _operating_from_values(
    real_values: Sequence[float], placebo: Sequence[float]
) -> OperatingCovariates:
    if len(real_values) != 8 or len(placebo) != 8:
        raise V015ProtocolError("Operating construction requires eight real fields")
    return OperatingCovariates(
        past_mean_temperature_c=float(real_values[0]),
        past_mean_soc_fraction=float(real_values[1]),
        past_mean_dod_fraction=float(real_values[2]),
        past_efc_per_year=float(real_values[3]),
        planned_mean_temperature_c=float(real_values[4]),
        planned_mean_soc_fraction=float(real_values[5]),
        planned_mean_dod_fraction=float(real_values[6]),
        planned_efc_per_year=float(real_values[7]),
        placebo_controls=tuple(float(value) for value in placebo),
    )


def generate_operating_covariates(
    protocol: ValidatedV015Protocol,
    *,
    partition: str,
    family_id: str,
    zero_based_index: int,
) -> OperatingCovariates:
    """Generate the fixed ordinary covariates before any truth rejection draws."""
    seeds = derive_ordinary_stream_seeds(
        protocol,
        partition=partition,
        family_id=family_id,
        zero_based_index=zero_based_index,
    )
    real = _draw_real_operating(
        protocol,
        _generator(seeds["operating_covariates"]),
        audit=partition == "audit",
    )
    placebo_rng = _generator(seeds["placebo_covariates"])
    placebo = tuple(float(value) for value in placebo_rng.uniform(-1.0, 1.0, 8))
    return _operating_from_values(real, placebo)


def stress_index(
    temperature_c: float,
    soc_fraction: float,
    dod_fraction: float,
    efc_per_year: float,
) -> float:
    values = np.asarray(
        [temperature_c, soc_fraction, dod_fraction, efc_per_year], dtype=float
    )
    if not np.isfinite(values).all():
        raise V015ProtocolError("Stress-index inputs must be finite")
    z_temperature = (values[0] - 27.5) / 12.5
    z_soc = (values[1] - 0.55) / 0.35
    z_dod = (values[2] - 0.55) / 0.35
    z_efc = (values[3] - 275.0) / 175.0
    return float(0.35 * z_temperature + 0.25 * z_soc + 0.15 * z_dod + 0.25 * z_efc)


def operating_stress_indices(
    operating: OperatingCovariates,
) -> tuple[float, float]:
    values = operating.real_values()
    return (
        stress_index(*values[:4]),
        stress_index(*values[4:]),
    )


def stable_softplus(value: Sequence[float] | np.ndarray | float) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(value, dtype=float))


def stable_sigmoid(value: Sequence[float] | np.ndarray | float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    result = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _validated_parameter_map(
    family_id: str,
    parameters: Mapping[str, float],
) -> dict[str, float]:
    if family_id not in _EXPECTED_PARAMETER_LAYOUT:
        raise V015ProtocolError(f"Unknown V0.15 truth family: {family_id}")
    expected = tuple(item[1] for item in _EXPECTED_PARAMETER_LAYOUT[family_id])
    if set(parameters) != set(expected):
        raise V015ProtocolError(
            f"{family_id} parameters changed; expected={expected}, "
            f"observed={tuple(parameters)}"
        )
    result = {
        name: _finite_float(parameters[name], context=f"{family_id}.{name}")
        for name in expected
    }
    return result


def evaluate_base_loss(
    family_id: str,
    parameters: Mapping[str, float],
    elapsed_days: Sequence[float] | np.ndarray,
    *,
    time_scale_days: float = 365.25,
) -> np.ndarray:
    """Evaluate one of the eight frozen base-loss mechanisms."""
    elapsed = np.asarray(elapsed_days, dtype=float)
    if (
        elapsed.ndim != 1
        or not np.isfinite(elapsed).all()
        or np.any(elapsed < 0.0)
        or not math.isfinite(time_scale_days)
        or time_scale_days <= 0.0
    ):
        raise V015ProtocolError("Truth evaluation days are invalid")
    parameter = _validated_parameter_map(family_id, parameters)
    years = elapsed / time_scale_days
    if family_id == "single_power":
        loss = parameter["a"] * np.power(years, parameter["b"])
    elif family_id == "dual_power":
        loss = parameter["a1"] * np.power(years, parameter["b1"])
        loss += parameter["a2"] * np.power(years, parameter["b2"])
    elif family_id == "saturating_plus_slow":
        saturation = parameter["a_sat"] * (
            1.0
            - np.exp(-np.power(elapsed / parameter["tau_sat_days"], parameter["b_sat"]))
        )
        slow = parameter["a_slow"] * np.power(years, parameter["b_slow"])
        loss = saturation + slow
    elif family_id == "early_activation_plus_power":
        activation = (
            parameter["activation_amplitude_pp"]
            * (1.0 - np.exp(-elapsed / parameter["tau_rise_days"]))
            * np.exp(-elapsed / parameter["tau_decay_days"])
        )
        loss = parameter["a"] * np.power(years, parameter["b"]) - activation
    elif family_id == "late_knee":
        knee = (
            parameter["k_pp_per_day"]
            * parameter["w_days"]
            * (
                stable_softplus(
                    (elapsed - parameter["t_knee_days"]) / parameter["w_days"]
                )
                - stable_softplus(-parameter["t_knee_days"] / parameter["w_days"])
            )
        )
        loss = parameter["a"] * np.power(years, parameter["b"]) + knee
    elif family_id == "linear_drift_plus_power":
        loss = parameter["a"] * np.power(years, parameter["b"])
        loss += parameter["c"] * years
    elif family_id == "smooth_broken_power":
        transition = np.power(
            1.0
            + np.power(
                elapsed / parameter["transition_tau_days"],
                parameter["sharpness"],
            ),
            (parameter["b_late"] - parameter["b_early"]) / parameter["sharpness"],
        )
        loss = parameter["a"] * np.power(years, parameter["b_early"]) * transition
    elif family_id == "saturating_logistic_knee":
        knee = parameter["knee_amplitude_pp"] * (
            stable_sigmoid((elapsed - parameter["t_knee_days"]) / parameter["w_days"])
            - stable_sigmoid(-parameter["t_knee_days"] / parameter["w_days"])
        )
        loss = parameter["a"] * np.power(years, parameter["b"]) + knee
    else:  # pragma: no cover - guarded by _validated_parameter_map
        raise V015ProtocolError(f"Unknown V0.15 truth family: {family_id}")
    if not np.isfinite(loss).all() or loss.shape != elapsed.shape:
        raise V015ProtocolError("Truth family produced nonfinite loss")
    if len(loss) and elapsed[0] == 0.0:
        loss[0] = 0.0
    return loss


def apply_operating_scenario(
    base_loss: Sequence[float] | np.ndarray,
    elapsed_days: Sequence[float] | np.ndarray,
    *,
    base_loss_at_730: float,
    operating: OperatingCovariates,
    gamma: float,
    time_scale_days: float = 365.25,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    base = np.asarray(base_loss, dtype=float)
    gamma_value = _finite_float(gamma, context="gamma")
    if (
        elapsed.shape != base.shape
        or elapsed.ndim != 1
        or not np.isfinite(elapsed).all()
        or not np.isfinite(base).all()
        or np.any(elapsed < 0.0)
    ):
        raise V015ProtocolError("Operating-scenario inputs are invalid")
    past_stress, planned_stress = operating_stress_indices(operating)
    result = np.empty_like(base)
    prefix = elapsed <= 730.0
    result[prefix] = math.exp(0.20 * past_stress) * base[prefix]
    future = ~prefix
    future_years = (elapsed[future] - 730.0) / time_scale_days
    result[future] = (
        math.exp(0.20 * past_stress) * float(base_loss_at_730)
        + math.exp(0.25 * planned_stress) * (base[future] - float(base_loss_at_730))
        + gamma_value * max(planned_stress, 0.0) * np.power(future_years, 1.05)
    )
    return result


def evaluate_truth_retention(
    family_id: str,
    parameters: Mapping[str, float],
    operating: OperatingCovariates,
    gamma: float,
    elapsed_days: Sequence[float] | np.ndarray,
    *,
    time_scale_days: float = 365.25,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    base = evaluate_base_loss(
        family_id,
        parameters,
        elapsed,
        time_scale_days=time_scale_days,
    )
    base_730 = float(
        evaluate_base_loss(
            family_id,
            parameters,
            [730.0],
            time_scale_days=time_scale_days,
        )[0]
    )
    retention = 100.0 - apply_operating_scenario(
        base,
        elapsed,
        base_loss_at_730=base_730,
        operating=operating,
        gamma=gamma,
        time_scale_days=time_scale_days,
    )
    if len(retention) and elapsed[0] == 0.0:
        retention[0] = 100.0
    if not np.isfinite(retention).all():
        raise V015ProtocolError("Operating truth produced nonfinite retention")
    return retention


def truth_is_admissible(
    protocol: ValidatedV015Protocol,
    family_id: str,
    retention_pct: Sequence[float] | np.ndarray,
) -> bool:
    values = np.asarray(retention_pct, dtype=float)
    if (
        values.shape != (len(protocol.combined_days),)
        or not np.isfinite(values).all()
        or values[0] != 100.0
        or np.min(values) < protocol.truth_support_pct[0]
        or np.max(values) > protocol.truth_support_pct[1]
        or np.max(np.abs(np.diff(values))) > protocol.maximum_single_interval_change_pp
    ):
        return False
    if family_id != "early_activation_plus_power" and np.any(np.diff(values) > 1e-10):
        return False
    return True


def _inverse_parameter_cdf(definition: ParameterDefinition, unit: float) -> float:
    if not 0.0 <= unit <= 1.0:
        raise V015ProtocolError("Parameter CDF coordinate must lie in [0,1]")
    if definition.distribution == "uniform":
        return definition.minimum + unit * (definition.maximum - definition.minimum)
    return float(
        math.exp(
            math.log(definition.minimum)
            + unit * (math.log(definition.maximum) - math.log(definition.minimum))
        )
    )


def _draw_truth_parameters(
    family: TruthFamilyDefinition,
    rng: np.random.Generator,
    *,
    audit: bool,
) -> tuple[dict[str, float], float]:
    result: dict[str, float] = {}
    for definition in family.parameters:
        if audit:
            upper_quartile = bool(rng.binomial(1, 0.5))
            unit = float(
                rng.uniform(0.75, 1.0) if upper_quartile else rng.uniform(0.0, 0.25)
            )
        else:
            unit = float(rng.uniform())
        result[definition.parameter_name] = _inverse_parameter_cdf(definition, unit)
    if audit:
        upper_gamma = bool(rng.binomial(1, 0.5))
        gamma_unit = float(
            rng.uniform(0.75, 1.0) if upper_gamma else rng.uniform(0.0, 0.25)
        )
    else:
        gamma_unit = float(rng.uniform())
    gamma = 0.05 + gamma_unit * 0.20
    return result, gamma


def sample_truth_spec(
    protocol: ValidatedV015Protocol,
    *,
    partition: str,
    family_id: str,
    zero_based_index: int,
    operating: OperatingCovariates | None = None,
) -> tuple[TruthSpec, OperatingCovariates]:
    """Sample one formal ordinary V2 spec; callers must obey the freeze lifecycle."""
    seeds = derive_ordinary_stream_seeds(
        protocol,
        partition=partition,
        family_id=family_id,
        zero_based_index=zero_based_index,
    )
    fixed_operating = operating or generate_operating_covariates(
        protocol,
        partition=partition,
        family_id=family_id,
        zero_based_index=zero_based_index,
    )
    family = protocol.family_map()[family_id]
    truth_rng = _generator(seeds["truth_parameters"])
    for attempt in range(1, protocol.maximum_parameter_attempts + 1):
        parameters, gamma = _draw_truth_parameters(
            family, truth_rng, audit=partition == "audit"
        )
        retention = evaluate_truth_retention(
            family_id,
            parameters,
            fixed_operating,
            gamma,
            protocol.combined_days,
            time_scale_days=protocol.time_scale_days,
        )
        if truth_is_admissible(protocol, family_id, retention):
            return (
                TruthSpec(
                    cluster_id=derive_ordinary_cluster_id(
                        protocol,
                        partition=partition,
                        family_id=family_id,
                        zero_based_index=zero_based_index,
                    ),
                    partition=partition,
                    family_id=family_id,
                    cluster_index=zero_based_index,
                    parameters=tuple(parameters.items()),
                    gamma=gamma,
                    operating_seed=seeds["operating_covariates"],
                    placebo_seed=seeds["placebo_covariates"],
                    truth_seed=seeds["truth_parameters"],
                    measurement_seed=seeds["measurement_noise"],
                    accepted_attempt=attempt,
                ),
                fixed_operating,
            )
    raise V015ProtocolError(
        "Ordinary truth exceeded the frozen parameter-attempt limit"
    )


def ar1_observation_errors(
    sigma_pp: float,
    rho: float,
    innovations: Sequence[float] | np.ndarray,
) -> np.ndarray:
    sigma = _finite_float(sigma_pp, context="sigma_pp")
    correlation = _finite_float(rho, context="rho")
    values = np.asarray(innovations, dtype=float)
    if (
        sigma <= 0.0
        or not 0.0 <= correlation < 1.0
        or values.ndim != 1
        or not np.isfinite(values).all()
        or len(values) < 1
    ):
        raise V015ProtocolError("AR(1) inputs are invalid")
    errors = np.empty_like(values)
    errors[0] = sigma * values[0]
    innovation_scale = sigma * math.sqrt(1.0 - correlation**2)
    for index in range(1, len(values)):
        errors[index] = (
            correlation * errors[index - 1] + innovation_scale * values[index]
        )
    return errors


def sample_observation_noise(
    protocol: ValidatedV015Protocol,
    *,
    partition: str,
    seed: int,
) -> ObservationNoise:
    rng = _generator(seed)
    if partition == "audit":
        sigma_bounds = protocol.audit_sigma_bounds
        rho_bounds = protocol.audit_rho_bounds
    else:
        sigma_bounds = protocol.standard_sigma_bounds
        rho_bounds = protocol.standard_rho_bounds
    sigma = float(rng.uniform(*sigma_bounds))
    rho = float(rng.uniform(*rho_bounds))
    innovations = rng.standard_normal(len(protocol.combined_days))
    errors = ar1_observation_errors(sigma, rho, innovations)
    return ObservationNoise(
        sigma_pp=sigma,
        rho=rho,
        errors_pp=tuple(float(value) for value in errors),
    )


def apply_observation_noise(
    latent_retention_pct: Sequence[float] | np.ndarray,
    noise: ObservationNoise,
) -> np.ndarray:
    latent = np.asarray(latent_retention_pct, dtype=float)
    errors = np.asarray(noise.errors_pp, dtype=float)
    if (
        latent.shape != errors.shape
        or latent.ndim != 1
        or not np.isfinite(latent).all()
        or not np.isfinite(errors).all()
    ):
        raise V015ProtocolError("Latent trajectory and noise coordinates differ")
    observed = latent + errors - errors[0]
    observed[0] = 100.0
    return observed


def _parameter_json(parameters: Mapping[str, float]) -> str:
    return _canonical_json_bytes(
        {name: float(value) for name, value in parameters.items()}
    ).decode("ascii")


def _member_packs_from_curves(
    protocol: ValidatedV015Protocol,
    *,
    partition: str,
    cluster_id: str,
    family_label: str,
    parameters: Mapping[str, float],
    gamma: float,
    operating: OperatingCovariates,
    latent: np.ndarray,
    noisy: np.ndarray,
) -> GeneratedMemberPacks:
    prefix_count = len(protocol.prefix_days)
    if latent.shape != (len(protocol.combined_days),) or noisy.shape != latent.shape:
        raise V015ProtocolError("Generated trajectory does not match the frozen grid")
    common = {
        "protocol_id": protocol.protocol_id,
        "partition": partition,
        "cluster_id": cluster_id,
    }
    prefix = pd.DataFrame(
        [
            {
                **common,
                "prefix_day": float(day),
                "observed_retention_pct": float(noisy[index]),
            }
            for index, day in enumerate(protocol.prefix_days)
        ],
        columns=PREFIX_COLUMNS,
    )
    coordinates = pd.DataFrame(
        [{**common, "forecast_day": float(day)} for day in protocol.forecast_days],
        columns=FORECAST_COORDINATE_COLUMNS,
    )
    operating_frame = pd.DataFrame(
        [{**common, **operating.as_record()}],
        columns=OPERATING_COLUMNS,
    )
    parameters_json = _parameter_json(parameters)
    truth = pd.DataFrame(
        [
            {
                **common,
                "truth_family": family_label,
                "truth_parameters_json": parameters_json,
                "gamma": float(gamma),
                "forecast_day": float(day),
                "latent_retention_pct": float(latent[prefix_count + index]),
                "noisy_retention_pct": float(noisy[prefix_count + index]),
            }
            for index, day in enumerate(protocol.forecast_days)
        ],
        columns=TRUTH_COLUMNS,
    )
    return GeneratedMemberPacks(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        operating_pack=operating_frame,
        truth_pack=truth,
    )


def generate_cluster_packs(
    protocol: ValidatedV015Protocol,
    truth_spec: TruthSpec,
    operating: OperatingCovariates,
) -> OrdinaryGeneratedCluster:
    """Materialize one already sampled ordinary spec on the frozen grids."""
    latent = evaluate_truth_retention(
        truth_spec.family_id,
        truth_spec.parameter_map(),
        operating,
        truth_spec.gamma,
        protocol.combined_days,
        time_scale_days=protocol.time_scale_days,
    )
    if not truth_is_admissible(protocol, truth_spec.family_id, latent):
        raise V015ProtocolError("Truth spec is not admissible under frozen V2")
    noise = sample_observation_noise(
        protocol,
        partition=truth_spec.partition,
        seed=truth_spec.measurement_seed,
    )
    noisy = apply_observation_noise(latent, noise)
    packs = _member_packs_from_curves(
        protocol,
        partition=truth_spec.partition,
        cluster_id=truth_spec.cluster_id,
        family_label=truth_spec.family_id,
        parameters=truth_spec.parameter_map(),
        gamma=truth_spec.gamma,
        operating=operating,
        latent=latent,
        noisy=noisy,
    )
    return OrdinaryGeneratedCluster(
        truth_spec=truth_spec,
        operating=operating,
        noise=noise,
        packs=packs,
    )


def evaluate_intrinsic_pair_retention(
    base_parameters: Mapping[str, float],
    operating: OperatingCovariates,
    gamma: float,
    elapsed_days: Sequence[float] | np.ndarray,
    *,
    mechanism: str,
    mechanism_parameters: Mapping[str, float],
    time_scale_days: float = 365.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a hand-specified M0 ambiguity pair without sampling."""
    elapsed = np.asarray(elapsed_days, dtype=float)
    base = evaluate_base_loss(
        "single_power",
        base_parameters,
        elapsed,
        time_scale_days=time_scale_days,
    )
    if mechanism == "piecewise_linear_knee":
        expected = {"k_pp_per_day", "t_knee_days"}
        if set(mechanism_parameters) != expected:
            raise V015ProtocolError("Piecewise-knee parameters changed")
        k = _finite_float(mechanism_parameters["k_pp_per_day"], context="k_pp_per_day")
        knee_day = _finite_float(
            mechanism_parameters["t_knee_days"], context="t_knee_days"
        )
        addition = k * np.maximum(elapsed - knee_day, 0.0)
    elif mechanism == "compact_smoothstep":
        expected = {"amplitude_pp", "t_start_days", "duration_days"}
        if set(mechanism_parameters) != expected:
            raise V015ProtocolError("Compact-smoothstep parameters changed")
        amplitude = _finite_float(
            mechanism_parameters["amplitude_pp"], context="amplitude_pp"
        )
        start = _finite_float(
            mechanism_parameters["t_start_days"], context="t_start_days"
        )
        duration = _finite_float(
            mechanism_parameters["duration_days"], context="duration_days"
        )
        if duration <= 0.0:
            raise V015ProtocolError("Smoothstep duration must be positive")
        unit = (elapsed - start) / duration
        smooth = np.where(
            unit <= 0.0,
            0.0,
            np.where(unit >= 1.0, 1.0, 3.0 * unit**2 - 2.0 * unit**3),
        )
        addition = amplitude * smooth
    else:
        raise V015ProtocolError(f"Unknown intrinsic-pair mechanism: {mechanism}")
    base_730 = float(
        evaluate_base_loss(
            "single_power",
            base_parameters,
            [730.0],
            time_scale_days=time_scale_days,
        )[0]
    )
    left_loss = apply_operating_scenario(
        base,
        elapsed,
        base_loss_at_730=base_730,
        operating=operating,
        gamma=gamma,
        time_scale_days=time_scale_days,
    )
    right_loss = apply_operating_scenario(
        base + addition,
        elapsed,
        base_loss_at_730=base_730,
        operating=operating,
        gamma=gamma,
        time_scale_days=time_scale_days,
    )
    left = 100.0 - left_loss
    right = 100.0 - right_loss
    if len(elapsed) and elapsed[0] == 0.0:
        left[0] = 100.0
        right[0] = 100.0
    return left, right


def evaluate_stress_plan_pair_retention(
    family_id: str,
    parameters: Mapping[str, float],
    low_plan_operating: OperatingCovariates,
    high_plan_operating: OperatingCovariates,
    gamma: float,
    elapsed_days: Sequence[float] | np.ndarray,
    *,
    time_scale_days: float = 365.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a hand-specified M1 pair with a shared prefix and different plans."""
    low = evaluate_truth_retention(
        family_id,
        parameters,
        low_plan_operating,
        gamma,
        elapsed_days,
        time_scale_days=time_scale_days,
    )
    high = evaluate_truth_retention(
        family_id,
        parameters,
        high_plan_operating,
        gamma,
        elapsed_days,
        time_scale_days=time_scale_days,
    )
    return low, high


def _pair_stream_seed(
    protocol: ValidatedV015Protocol,
    partition: str,
    pair_index: int,
    stream_name: str,
) -> int:
    if partition not in MATCHED_PARTITIONS or not 0 <= pair_index < 250:
        raise V015ProtocolError("Matched-pair coordinate is outside the design")
    root = protocol.seed_root_map()[partition]
    return derive_stream_seed(
        protocol.protocol_id,
        root,
        partition,
        partition,
        pair_index,
        stream_name,
    )


def _pair_ids(
    protocol: ValidatedV015Protocol,
    partition: str,
    pair_index: int,
) -> tuple[str, str, str]:
    root = protocol.seed_root_map()[partition]
    ids = [
        "c_"
        + hashlib.sha256(
            f"{protocol.protocol_id}|{root}|{pair_index}|opaque_pool|{pool_index}".encode(
                "ascii"
            )
        ).hexdigest()[:32]
        for pool_index in range(2)
    ]
    swap_digest = hashlib.sha256(
        f"{protocol.protocol_id}|{root}|{pair_index}|opaque_swap".encode("ascii")
    ).digest()
    if swap_digest[-1] & 1:
        ids.reverse()
    pair_id = (
        "p_"
        + hashlib.sha256(
            f"{protocol.protocol_id}|{root}|{pair_index}|pair_id".encode("ascii")
        ).hexdigest()[:32]
    )
    return ids[0], ids[1], pair_id


def _draw_pair_real_operating(
    protocol: ValidatedV015Protocol, rng: np.random.Generator
) -> tuple[float, ...]:
    return tuple(
        lower + float(rng.uniform()) * (upper - lower)
        for _, (lower, upper) in protocol.operating_support
    )


def _draw_pair_noise(
    protocol: ValidatedV015Protocol,
    rng: np.random.Generator,
) -> ObservationNoise:
    sigma = float(rng.uniform(*protocol.standard_sigma_bounds))
    rho = float(rng.uniform(*protocol.standard_rho_bounds))
    innovations = rng.standard_normal(len(protocol.combined_days))
    errors = ar1_observation_errors(sigma, rho, innovations)
    return ObservationNoise(sigma, rho, tuple(float(value) for value in errors))


def _concatenate_member_packs(
    members: Sequence[GeneratedMemberPacks],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.concat([item.prefix_pack for item in members], ignore_index=True),
        pd.concat([item.forecast_coordinates for item in members], ignore_index=True),
        pd.concat([item.operating_pack for item in members], ignore_index=True),
        pd.concat([item.truth_pack for item in members], ignore_index=True),
    )


def generate_intrinsic_matched_pair(
    protocol: ValidatedV015Protocol,
    *,
    zero_based_pair_index: int,
) -> MatchedPairPacks:
    """Generate one formal M0 pair; do not call before the frozen execution."""
    partition = INTRINSIC_MATCHED_PARTITION
    truth_rng = _generator(
        _pair_stream_seed(protocol, partition, zero_based_pair_index, "shared_truth")
    )
    a = float(truth_rng.uniform(0.2, 0.8))
    b = float(truth_rng.uniform(0.35, 0.70))
    gamma = float(truth_rng.uniform(0.05, 0.25))
    if zero_based_pair_index < 125:
        mechanism = "piecewise_linear_knee"
        mechanism_parameters = {
            "k_pp_per_day": float(truth_rng.uniform(0.0015, 0.0025)),
            "t_knee_days": float(truth_rng.uniform(1461.0, 2922.0)),
        }
        right_family = "intrinsic_piecewise_linear_knee"
    else:
        mechanism = "compact_smoothstep"
        amplitude = float(truth_rng.uniform(6.5, 7.0))
        t_start = float(truth_rng.uniform(1095.75, 3652.5))
        duration = float(
            math.exp(truth_rng.uniform(math.log(365.25), math.log(1826.25)))
        )
        mechanism_parameters = {
            "amplitude_pp": amplitude,
            "t_start_days": t_start,
            "duration_days": duration,
        }
        right_family = "intrinsic_compact_smoothstep"
    real = _draw_pair_real_operating(
        protocol,
        _generator(
            _pair_stream_seed(
                protocol, partition, zero_based_pair_index, "shared_operating"
            )
        ),
    )
    placebo_rng = _generator(
        _pair_stream_seed(protocol, partition, zero_based_pair_index, "placebo")
    )
    operating = _operating_from_values(real, placebo_rng.uniform(-1.0, 1.0, 8))
    left, right = evaluate_intrinsic_pair_retention(
        {"a": a, "b": b},
        operating,
        gamma,
        protocol.combined_days,
        mechanism=mechanism,
        mechanism_parameters=mechanism_parameters,
        time_scale_days=protocol.time_scale_days,
    )
    prefix_count = len(protocol.prefix_days)
    separation = float(abs(left[-1] - right[-1]))
    prefix_difference = left[:prefix_count] - right[:prefix_count]
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or left[0] != 100.0
        or right[0] != 100.0
        or np.min(left) < 50.0
        or np.max(left) > 105.0
        or np.min(right) < 50.0
        or np.max(right) > 105.0
        or not np.array_equal(left[:prefix_count], right[:prefix_count])
        or separation < 5.0
    ):
        raise V015ProtocolError("One-shot intrinsic pair is invalid")
    noise = _draw_pair_noise(
        protocol,
        _generator(
            _pair_stream_seed(
                protocol, partition, zero_based_pair_index, "shared_measurement"
            )
        ),
    )
    left_noisy = apply_observation_noise(left, noise)
    right_noisy = apply_observation_noise(right, noise)
    left_id, right_id, pair_id = _pair_ids(protocol, partition, zero_based_pair_index)
    left_parameters = {"a": a, "b": b}
    right_parameters = {**left_parameters, **mechanism_parameters}
    left_packs = _member_packs_from_curves(
        protocol,
        partition=partition,
        cluster_id=left_id,
        family_label="intrinsic_single_power",
        parameters=left_parameters,
        gamma=gamma,
        operating=operating,
        latent=left,
        noisy=left_noisy,
    )
    right_packs = _member_packs_from_curves(
        protocol,
        partition=partition,
        cluster_id=right_id,
        family_label=right_family,
        parameters=right_parameters,
        gamma=gamma,
        operating=operating,
        latent=right,
        noisy=right_noisy,
    )
    prefix, coordinates, operating_pack, truth = _concatenate_member_packs(
        [left_packs, right_packs]
    )
    mapping = pd.DataFrame(
        [
            {
                "protocol_id": protocol.protocol_id,
                "pair_partition": partition,
                "pair_id": pair_id,
                "left_cluster_id": left_id,
                "right_cluster_id": right_id,
                "construction_family": mechanism,
                "left_side_code": "smooth_reference",
                "right_side_code": mechanism,
                "latent_prefix_rmse_pp": float(
                    np.sqrt(np.mean(np.square(prefix_difference)))
                ),
                "latent_prefix_max_abs_difference_pp": float(
                    np.max(np.abs(prefix_difference))
                ),
                "truth_separation_25y_pp": separation,
            }
        ],
        columns=MATCHED_PAIR_COLUMNS,
    )
    return MatchedPairPacks(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        operating_pack=operating_pack,
        truth_pack=truth,
        matched_pairs=mapping,
    )


def _draw_low_high_plan_operating(
    protocol: ValidatedV015Protocol,
    rng: np.random.Generator,
    placebo: Sequence[float],
) -> tuple[OperatingCovariates, OperatingCovariates]:
    support = protocol.support_map()
    past: list[float] = []
    for field in REAL_OPERATING_FIELDS[:4]:
        lower, upper = support[field]
        past.append(lower + float(rng.uniform()) * (upper - lower))
    low: list[float] = []
    for field in REAL_OPERATING_FIELDS[4:]:
        lower, upper = support[field]
        middle = (lower + upper) / 2.0
        low.append(lower + float(rng.uniform()) * (middle - lower))
    high: list[float] = []
    for field in REAL_OPERATING_FIELDS[4:]:
        lower, upper = support[field]
        middle = (lower + upper) / 2.0
        high.append(middle + float(rng.uniform()) * (upper - middle))
    return (
        _operating_from_values((*past, *low), placebo),
        _operating_from_values((*past, *high), placebo),
    )


def generate_stress_plan_matched_pair(
    protocol: ValidatedV015Protocol,
    *,
    zero_based_pair_index: int,
) -> MatchedPairPacks:
    """Generate one formal M1 pair; do not call before the frozen execution."""
    partition = STRESS_PLAN_MATCHED_PARTITION
    if not 0 <= zero_based_pair_index < 250:
        raise V015ProtocolError("Stress-plan pair index is outside the design")
    family_id = CORE_FAMILY_IDS[zero_based_pair_index % len(CORE_FAMILY_IDS)]
    placebo_rng = _generator(
        _pair_stream_seed(protocol, partition, zero_based_pair_index, "placebo")
    )
    placebo = tuple(float(value) for value in placebo_rng.uniform(-1.0, 1.0, 8))
    low_operating, high_operating = _draw_low_high_plan_operating(
        protocol,
        _generator(
            _pair_stream_seed(
                protocol, partition, zero_based_pair_index, "shared_operating"
            )
        ),
        placebo,
    )
    family = protocol.family_map()[family_id]
    truth_rng = _generator(
        _pair_stream_seed(protocol, partition, zero_based_pair_index, "shared_truth")
    )
    low = high = np.empty(0)
    parameters: dict[str, float] | None = None
    gamma = math.nan
    for _ in range(protocol.maximum_parameter_attempts):
        parameters, gamma = _draw_truth_parameters(family, truth_rng, audit=False)
        low, high = evaluate_stress_plan_pair_retention(
            family_id,
            parameters,
            low_operating,
            high_operating,
            gamma,
            protocol.combined_days,
            time_scale_days=protocol.time_scale_days,
        )
        if truth_is_admissible(protocol, family_id, low) and truth_is_admissible(
            protocol, family_id, high
        ):
            break
    else:
        raise V015ProtocolError(
            "Stress-plan pair exceeded the frozen parameter-attempt limit"
        )
    assert parameters is not None
    prefix_count = len(protocol.prefix_days)
    if not np.array_equal(low[:prefix_count], high[:prefix_count]):
        raise V015ProtocolError("Stress-plan pair prefix is not exactly shared")
    noise = _draw_pair_noise(
        protocol,
        _generator(
            _pair_stream_seed(
                protocol, partition, zero_based_pair_index, "shared_measurement"
            )
        ),
    )
    low_noisy = apply_observation_noise(low, noise)
    high_noisy = apply_observation_noise(high, noise)
    low_id, high_id, pair_id = _pair_ids(protocol, partition, zero_based_pair_index)
    low_packs = _member_packs_from_curves(
        protocol,
        partition=partition,
        cluster_id=low_id,
        family_label=family_id,
        parameters=parameters,
        gamma=gamma,
        operating=low_operating,
        latent=low,
        noisy=low_noisy,
    )
    high_packs = _member_packs_from_curves(
        protocol,
        partition=partition,
        cluster_id=high_id,
        family_label=family_id,
        parameters=parameters,
        gamma=gamma,
        operating=high_operating,
        latent=high,
        noisy=high_noisy,
    )
    prefix, coordinates, operating_pack, truth = _concatenate_member_packs(
        [low_packs, high_packs]
    )
    prefix_difference = low[:prefix_count] - high[:prefix_count]
    mapping = pd.DataFrame(
        [
            {
                "protocol_id": protocol.protocol_id,
                "pair_partition": partition,
                "pair_id": pair_id,
                "left_cluster_id": low_id,
                "right_cluster_id": high_id,
                "construction_family": family_id,
                "left_side_code": "low_plan",
                "right_side_code": "high_plan",
                "latent_prefix_rmse_pp": float(
                    np.sqrt(np.mean(np.square(prefix_difference)))
                ),
                "latent_prefix_max_abs_difference_pp": float(
                    np.max(np.abs(prefix_difference))
                ),
                "truth_separation_25y_pp": float(abs(low[-1] - high[-1])),
            }
        ],
        columns=MATCHED_PAIR_COLUMNS,
    )
    return MatchedPairPacks(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        operating_pack=operating_pack,
        truth_pack=truth,
        matched_pairs=mapping,
    )
