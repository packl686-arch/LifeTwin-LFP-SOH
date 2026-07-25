"""Committed-artifact-only scoring for the frozen V0.15 experiment.

The public entry point accepts in-memory artifacts which have already crossed
the phase-scoped firewall.  It does not accept a filesystem path, invoke an
optimizer, or call a truth generator.  Before using any outcome it validates
the complete prediction and sealed-truth bundles, decodes the canonical model
state, and independently recomputes every label-free prediction artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_synthetic import (
    canonical_csv_bytes as _v1_canonical_csv_bytes,
)
from lifetwin.experiments.calendar_long_horizon_v015_analysis import (
    AUDIT_ISSUE_COUNT,
    BOOTSTRAP_RESAMPLES,
    CORE_FAMILIES,
    RANDOM_RANKING_COUNT,
    REQUIRED_GATE_IDS,
    RISK_SCORE_IDS,
    STRESS_PERMUTATIONS,
    TEST_FAMILIES,
    TEST_ISSUE_COUNT,
    CoverageSummary,
    GateEvaluation,
    RiskReduction,
    StressPermutationSummary,
    V015AnalysisError,
    V015InconclusiveError,
    bootstrap_risk_reductions,
    common_pool_gate_evaluations,
    core_test_coverage_summary,
    deterministic_random_rankings,
    evaluate_audit_directional_gates,
    evaluate_intrinsic_pairs,
    evaluate_stress_plan_pairs,
    evaluate_test_safety_gates,
    issued_center_minus_baseline_iae,
    placebo_negative_control_gates,
    primary_gate_evaluations,
    rank_policy,
    resolve_result_status,
    score_trajectory_table,
    stress_permutation_metrics,
    validate_intrinsic_pair_construction,
    validate_intrinsic_output_invariance,
    validate_stress_plan_pair_construction,
    validate_stress_plan_arm_a_invariance,
)
from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEYS,
)
from lifetwin.experiments.calendar_long_horizon_v015_environment import (
    FormalEnvironmentIdentity,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FROZEN_PROTOCOL_ID,
    FrozenArtifactContract,
    V015ArtifactError,
    canonical_csv_bytes,
    canonical_json_bytes,
    load_artifact_contract,
    validate_prediction_artifact_bundle,
    validate_sealed_truth_bundle,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    LabelFreePipelineResult,
    V015PipelineError,
    recompute_label_free_pipeline,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_CANONICAL_SHA256,
    REAL_OPERATING_FIELDS,
    load_frozen_protocol_config,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import (
    DecodedModelState,
    V015TrainingError,
    build_model_state_payload,
    deserialize_model_state_json,
)


class V015ScoringError(ValueError):
    """Raised for a malformed scoring request or score artifact."""


@dataclass(frozen=True)
class ScoreFrameSchema:
    columns: tuple[str, ...]
    key: tuple[str, ...]


POINT_SCORE_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "forecast_day",
    "truth_family",
    "latent_retention_pct",
    "noisy_retention_pct",
    "center_forecast_pct",
    "sqrt_time_forecast_pct",
    "bounded_power_forecast_pct",
    "base_interval_lower_pct",
    "base_interval_upper_pct",
    "calibrated_interval_lower_pct",
    "calibrated_interval_upper_pct",
    "center_absolute_error_pp",
    "sqrt_absolute_error_pp",
    "bounded_power_absolute_error_pp",
    "interval_covers_truth",
    "interval_width_pp",
    "base_interval_covers_truth",
    "base_interval_width_pp",
    "canonical_prefix_content_sha256",
)
TRAJECTORY_SCORE_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "truth_family",
    "canonical_prefix_content_sha256",
    "center_endpoint_absolute_error_pp",
    "center_trajectory_iae_pp",
    "sqrt_trajectory_iae_pp",
    "bounded_power_trajectory_iae_pp",
    "persistence_trajectory_iae_pp",
    "catastrophic",
    "simultaneous_interval_covered",
    "max_interval_width_pp",
    "base_simultaneous_interval_covered",
    "base_max_interval_width_pp",
    *(f"risk_{score_id}" for score_id in RISK_SCORE_IDS),
    *(f"risk_hash_{score_id}" for score_id in RISK_SCORE_IDS),
    *(
        f"{field}_{arm}"
        for arm in ("prefix_only", "visible_stress")
        for field in (
            "hard_eligible",
            "issued",
            "issuance_rank",
            "raw_risk_score",
            "canonical_predictor_content_sha256",
        )
    ),
)
POLICY_COMPARISON_COLUMNS = (
    "partition",
    "score_id",
    "source_count",
    "eligible_count",
    "issued_count",
    "source_coverage",
    "eligible_coverage",
    "issued_catastrophic_rate",
    "mean_random_issued_catastrophic_rate",
    "relative_risk_reduction",
)
SUBSET_METRIC_COLUMNS = (
    "subset_id",
    "source_count",
    "eligible_count",
    "catastrophic_count",
    "issued_count",
    "prefix_only_issued_catastrophic_rate",
    "prefix_only_risk_reduction",
    "visible_stress_issued_catastrophic_rate",
    "visible_stress_risk_reduction",
    "visible_minus_prefix_increment",
)
COVERAGE_METRIC_COLUMNS = (
    "coverage_id",
    "n",
    "covered",
    "coverage",
    "one_sided_95_lower",
    "median_max_width_pp",
    "percentile_95_max_width_pp",
    "available",
    "unavailable_reason",
)
FAMILY_ERROR_COLUMNS = (
    "partition",
    "truth_family",
    "cluster_count",
    "catastrophic_count",
    "mean_center_endpoint_absolute_error_pp",
    "mean_center_trajectory_iae_pp",
    "mean_sqrt_trajectory_iae_pp",
    "mean_bounded_power_trajectory_iae_pp",
    "mean_persistence_trajectory_iae_pp",
)
FAMILY_METRIC_COLUMNS = (
    "metric_id",
    "partition",
    "truth_family",
    "source_count",
    "eligible_count",
    "catastrophic_count",
    "issued_count",
    "prefix_only_issued_catastrophic_rate",
    "prefix_only_risk_reduction",
    "visible_stress_issued_catastrophic_rate",
    "visible_stress_risk_reduction",
    "visible_minus_prefix_increment",
    "mean_center_endpoint_absolute_error_pp",
    "mean_center_trajectory_iae_pp",
    "mean_sqrt_trajectory_iae_pp",
    "mean_bounded_power_trajectory_iae_pp",
    "mean_persistence_trajectory_iae_pp",
)
INTRINSIC_PAIR_COLUMNS = (
    "pair_id",
    "left_cluster_id",
    "right_cluster_id",
    "arm_a_exact_equal",
    "arm_b_exact_equal",
    "interval_width_exact_equal",
    "both_futures_simultaneously_covered",
    "max_interval_width_pp",
)
STRESS_PAIR_COLUMNS = (
    "pair_id",
    "left_cluster_id",
    "right_cluster_id",
    "arm_a_exact_tie",
    "center_endpoint_error_delta_pp",
    "arm_b_risk_delta",
    "arm_b_correct_error_order",
)
MATCHED_PAIR_COLUMNS = (
    "experiment_id",
    "pair_id",
    "left_cluster_id",
    "right_cluster_id",
    "arm_a_exact_equal",
    "arm_b_exact_equal",
    "interval_width_exact_equal",
    "both_futures_simultaneously_covered",
    "max_interval_width_pp",
    "center_endpoint_error_delta_pp",
    "arm_b_risk_delta",
    "arm_b_correct_error_order",
)
RANDOM_RANKING_COLUMNS = (
    "partition",
    "ranking_index",
    "issued_count",
    "issued_catastrophic_rate",
    "analytic_random_expected_rate",
    "relative_risk_reduction",
)
BOOTSTRAP_COLUMNS = (
    "replicate_index",
    "defined",
    "eligible_count",
    "random_expected_catastrophic_rate",
    "prefix_only_risk_reduction",
    "visible_stress_risk_reduction",
    "visible_minus_prefix_increment",
    "placebo_minus_prefix_increment",
)
PERMUTATION_COLUMNS = (
    "permutation_index",
    "issued_count",
    "issued_catastrophic_rate",
    "visible_stress_risk_reduction",
    "visible_minus_prefix_increment",
)
GATE_COLUMNS = (
    "gate_id",
    "state",
    "estimate",
    "threshold",
    "reasons",
)

SCORE_FRAME_SCHEMAS: Mapping[str, ScoreFrameSchema] = {
    "point_scores.csv": ScoreFrameSchema(
        POINT_SCORE_COLUMNS,
        ("partition", "cluster_id", "forecast_day"),
    ),
    "trajectory_scores.csv": ScoreFrameSchema(
        TRAJECTORY_SCORE_COLUMNS,
        ("partition", "cluster_id"),
    ),
    "family_metrics.csv": ScoreFrameSchema(
        FAMILY_METRIC_COLUMNS,
        ("metric_id",),
    ),
    "matched_pair_scores.csv": ScoreFrameSchema(
        MATCHED_PAIR_COLUMNS,
        ("experiment_id", "pair_id"),
    ),
    "bootstrap_replicates.csv": ScoreFrameSchema(
        BOOTSTRAP_COLUMNS,
        ("replicate_index",),
    ),
    "random_ranking_metrics.csv": ScoreFrameSchema(
        RANDOM_RANKING_COLUMNS,
        ("partition", "ranking_index"),
    ),
    "stress_permutation_metrics.csv": ScoreFrameSchema(
        PERMUTATION_COLUMNS,
        ("permutation_index",),
    ),
}

REQUIRED_SCORE_ARTIFACTS = (
    "point_scores.csv",
    "trajectory_scores.csv",
    "family_metrics.csv",
    "matched_pair_scores.csv",
    "bootstrap_replicates.csv",
    "random_ranking_metrics.csv",
    "stress_permutation_metrics.csv",
    "negative_control_metrics.json",
    "score_report.json",
    "run_manifest.json",
)
REQUIRED_SCORE_CSV_ARTIFACTS = REQUIRED_SCORE_ARTIFACTS[:7]
NEGATIVE_CONTROL_METRICS_KEYS = frozenset(
    {
        "protocol_id",
        "available",
        "unavailable_reason",
        "placebo_point_increment",
        "placebo_bootstrap_two_sided_95_interval",
        "stress_permutation_summary",
        "gate_evaluations",
    }
)
SCORE_REPORT_KEYS = frozenset(
    {
        "protocol_id",
        "model_state_byte_sha256",
        "analysis_counts",
        "selected_mean_baseline",
        "test_primary_estimates",
        "policy_comparison",
        "risk_coverage_curves_secondary",
        "coverage_metrics",
        "isotonic_calibration_diagnostics",
        "structure_diagnostics",
        "test_audit_distribution_shift",
        "gate_evaluations",
        "stress_plan_summary",
        "stress_permutation_summary",
        "status",
        "protocol_deviations",
    }
)
RESULT_SUMMARY_KEYS = SCORE_REPORT_KEYS
RUN_MANIFEST_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "config_canonical_sha256",
        "model_state_byte_sha256",
        "analysis_counts",
        "required_score_artifacts",
        "manifest_hash_scope",
        "provenance",
        "artifacts",
        "protocol_deviations",
    }
)
RUN_PROVENANCE_KEYS = frozenset(
    {
        "environment_identity",
        "environment_identity_sha256",
        "source_tree_sha256",
        "implementation_freeze_record_sha256",
        "protocol_freeze_git_commit",
        "implementation_source_git_commit",
        "execution_git_commit",
        "seed_roots",
        "seed_derivation",
        "prediction_worker_count",
        "wall_time_seconds",
    }
)
ARTIFACT_METADATA_KEYS = frozenset({"path", "row_count", "byte_count", "byte_sha256"})
REQUIRED_FORMAL_NON_SCORE_ARTIFACTS = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "truth_commitments.json",
    "model_state.json",
    "model_state_commitment.json",
    "training_manifest.json",
    "calibration_manifest.json",
    "prediction_bundle.csv",
    "risk_bundle.csv",
    "decision_bundle.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
    "fit_commitment.json",
    "prediction_commitment.json",
    "center_state_checkpoint.json",
    "risk_state_checkpoint.json",
    "center_development_truth.csv",
    "risk_development_truth.csv",
    "calibration_truth.csv",
    "test_truth.csv",
    "audit_truth.csv",
    "intrinsic_matched_truth.csv",
    "stress_plan_matched_truth.csv",
    "intrinsic_matched_pairs.csv",
    "stress_plan_matched_pairs.csv",
    "exposure_log.jsonl",
)
_PROTOCOL_FREEZE_GIT_COMMIT = "b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91"
_PROVISIONAL_RUN_MANIFEST = {
    "protocol_id": FROZEN_PROTOCOL_ID,
    "finalized": False,
    "reason": "runner provenance and formal artifact metadata not supplied",
}


@dataclass(frozen=True)
class V015ScoringResult:
    point_scores: pd.DataFrame
    trajectory_scores: pd.DataFrame
    family_metrics: pd.DataFrame
    matched_pair_scores: pd.DataFrame
    bootstrap_replicates: pd.DataFrame
    random_ranking_metrics: pd.DataFrame
    stress_permutation_metrics: pd.DataFrame
    negative_control_metrics: Mapping[str, Any]
    score_report: Mapping[str, Any]
    run_manifest: Mapping[str, Any]

    @property
    def result_summary(self) -> Mapping[str, Any]:
        """Backward-compatible name for callers written before registry audit."""

        return self.score_report


@dataclass(frozen=True)
class _AnalysisCounts:
    random_rankings: int
    bootstrap_resamples: int
    stress_permutations: int


_FORMAL_ANALYSIS_COUNTS = _AnalysisCounts(
    random_rankings=RANDOM_RANKING_COUNT,
    bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    stress_permutations=STRESS_PERMUTATIONS,
)


def canonicalize_score_frame(
    frame: pd.DataFrame,
    filename: str,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Validate and deterministically sort one module-frozen score table."""

    try:
        schema = SCORE_FRAME_SCHEMAS[filename]
    except KeyError as exc:
        raise V015ScoringError(f"Unknown score artifact: {filename}") from exc
    if not isinstance(frame, pd.DataFrame):
        raise V015ScoringError(f"{filename} must be a dataframe")
    if tuple(frame.columns) != schema.columns:
        raise V015ScoringError(
            f"{filename} columns changed: observed={tuple(frame.columns)}"
        )
    if frame.empty and not allow_empty:
        raise V015ScoringError(f"{filename} cannot be empty")
    if not frame.empty:
        if frame.loc[:, list(schema.key)].isna().any().any():
            raise V015ScoringError(f"{filename} contains a missing key")
        if frame.duplicated(list(schema.key)).any():
            raise V015ScoringError(f"{filename} contains duplicate keys")
    return frame.sort_values(list(schema.key), kind="stable").reset_index(drop=True)


def canonical_score_csv_bytes(
    frame: pd.DataFrame,
    filename: str,
    *,
    allow_empty: bool = False,
) -> bytes:
    """Serialize a required score CSV, permitting header-only void outputs."""

    ordered = canonicalize_score_frame(frame, filename, allow_empty=allow_empty)
    schema = SCORE_FRAME_SCHEMAS[filename]
    try:
        return _v1_canonical_csv_bytes(ordered, columns=schema.columns)
    except ValueError as exc:
        raise V015ScoringError(str(exc)) from exc


def canonical_result_summary_bytes(payload: Mapping[str, Any]) -> bytes:
    """Backward-compatible canonicalizer for ``score_report.json``."""

    if set(payload) != SCORE_REPORT_KEYS:
        raise V015ScoringError("score_report.json keys changed")
    if payload.get("protocol_id") != FROZEN_PROTOCOL_ID:
        raise V015ScoringError("score_report.json protocol_id changed")
    return canonical_json_bytes(payload)


def _json_value(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _canonical_json_artifact_bytes(
    payload: Mapping[str, Any],
    *,
    filename: str,
    expected_keys: frozenset[str],
) -> bytes:
    if set(payload) != expected_keys:
        raise V015ScoringError(f"{filename} keys changed")
    if payload.get("protocol_id") != FROZEN_PROTOCOL_ID:
        raise V015ScoringError(f"{filename} protocol_id changed")
    return canonical_json_bytes(payload)


def _artifact_row_count(
    filename: str, value: pd.DataFrame | Mapping[str, Any]
) -> int | None:
    return len(value) if filename.endswith(".csv") else None


def _result_artifacts_without_manifest(
    result: V015ScoringResult,
) -> dict[str, pd.DataFrame | Mapping[str, Any]]:
    return {
        "point_scores.csv": result.point_scores,
        "trajectory_scores.csv": result.trajectory_scores,
        "family_metrics.csv": result.family_metrics,
        "matched_pair_scores.csv": result.matched_pair_scores,
        "bootstrap_replicates.csv": result.bootstrap_replicates,
        "random_ranking_metrics.csv": result.random_ranking_metrics,
        "stress_permutation_metrics.csv": result.stress_permutation_metrics,
        "negative_control_metrics.json": result.negative_control_metrics,
        "score_report.json": result.score_report,
    }


def _score_payloads_without_manifest(
    result: V015ScoringResult,
) -> dict[str, bytes]:
    status = result.score_report.get("status")
    terminal_statuses = {"void", "inconclusive_not_success"}
    allow_empty = (
        isinstance(status, Mapping) and status.get("status") in terminal_statuses
    )
    payloads: dict[str, bytes] = {}
    for filename, value in _result_artifacts_without_manifest(result).items():
        if filename.endswith(".csv"):
            if not isinstance(value, pd.DataFrame):
                raise V015ScoringError(f"{filename} must be a dataframe")
            payloads[filename] = canonical_score_csv_bytes(
                value, filename, allow_empty=allow_empty
            )
        elif filename == "negative_control_metrics.json":
            payloads[filename] = _canonical_json_artifact_bytes(
                value,
                filename=filename,
                expected_keys=NEGATIVE_CONTROL_METRICS_KEYS,
            )
        elif filename == "score_report.json":
            payloads[filename] = canonical_result_summary_bytes(value)
    if tuple(payloads) != REQUIRED_SCORE_ARTIFACTS[:-1]:
        raise V015ScoringError(
            "Pre-manifest score artifact registry differs from the freeze"
        )
    return payloads


def _sha256_text_value(value: object, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise V015ScoringError(f"{context} must be a lowercase SHA-256")
    return text


def _normalize_formal_artifact_metadata(
    value: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if set(value) != set(REQUIRED_FORMAL_NON_SCORE_ARTIFACTS):
        raise V015ScoringError("Formal non-score artifact metadata membership changed")
    records: list[dict[str, Any]] = []
    for filename in REQUIRED_FORMAL_NON_SCORE_ARTIFACTS:
        raw = value[filename]
        if not isinstance(raw, Mapping) or set(raw) != ARTIFACT_METADATA_KEYS:
            raise V015ScoringError(f"{filename} artifact metadata keys changed")
        if raw["path"] != filename:
            raise V015ScoringError(f"{filename} artifact metadata path changed")
        row_count = raw["row_count"]
        byte_count = raw["byte_count"]
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise V015ScoringError(f"{filename} artifact counts are invalid")
        records.append(
            {
                "path": filename,
                "row_count": row_count,
                "byte_count": byte_count,
                "byte_sha256": _sha256_text_value(
                    raw["byte_sha256"],
                    context=f"{filename} byte_sha256",
                ),
            }
        )
    return records


def _validate_manifest_artifact_records(
    records: object,
    *,
    expected_paths: Sequence[str],
) -> list[Mapping[str, Any]]:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise V015ScoringError("run_manifest.json artifact record count changed")
    normalized: list[Mapping[str, Any]] = []
    for expected_path, record in zip(expected_paths, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != ARTIFACT_METADATA_KEYS:
            raise V015ScoringError(
                f"run_manifest.json metadata keys changed for {expected_path}"
            )
        if record["path"] != expected_path:
            raise V015ScoringError("run_manifest.json formal artifact registry changed")
        row_count = record["row_count"]
        row_count_must_be_null = expected_path in {
            "negative_control_metrics.json",
            "score_report.json",
        }
        if row_count_must_be_null:
            if row_count is not None:
                raise V015ScoringError(
                    f"{expected_path} manifest row_count must be null"
                )
        elif (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
        ):
            raise V015ScoringError(f"{expected_path} manifest row_count is invalid")
        byte_count = record["byte_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise V015ScoringError(f"{expected_path} manifest byte_count is invalid")
        _sha256_text_value(
            record["byte_sha256"],
            context=f"{expected_path} manifest byte_sha256",
        )
        normalized.append(record)
    return normalized


def finalize_run_manifest(
    result: V015ScoringResult,
    *,
    environment_identity: FormalEnvironmentIdentity,
    implementation_freeze_record_sha256: str,
    protocol_freeze_git_commit: str,
    implementation_source_git_commit: str,
    seed_roots: Mapping[str, int],
    seed_derivation: str,
    prediction_worker_count: int,
    wall_time_seconds: float,
    formal_artifact_metadata: Mapping[str, Mapping[str, Any]],
) -> V015ScoringResult:
    """Bind runner provenance and every other formal artifact to the result."""

    if not isinstance(environment_identity, FormalEnvironmentIdentity):
        raise V015ScoringError(
            "environment_identity must be a verified formal identity"
        )
    if environment_identity.git_dirty:
        raise V015ScoringError("Formal run manifest cannot record a dirty git tree")
    _sha256_text_value(
        implementation_freeze_record_sha256,
        context="implementation_freeze_record_sha256",
    )
    protocol_commit = str(protocol_freeze_git_commit)
    implementation_commit = str(implementation_source_git_commit)
    execution_commit = environment_identity.git_commit
    for label, commit in (
        ("protocol_freeze_git_commit", protocol_commit),
        ("implementation_source_git_commit", implementation_commit),
        ("execution_git_commit", execution_commit),
    ):
        if len(commit) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise V015ScoringError(f"{label} is not a full git hash")
    if implementation_commit == execution_commit:
        raise V015ScoringError(
            "Implementation source and execution commits must be distinct"
        )
    if protocol_commit != _PROTOCOL_FREEZE_GIT_COMMIT:
        raise V015ScoringError("Protocol freeze git commit changed")
    if (
        environment_identity.config_byte_sha256
        != load_artifact_contract().config_byte_sha256
        or environment_identity.config_canonical_sha256
        != FROZEN_CONFIG_CANONICAL_SHA256
    ):
        raise V015ScoringError(
            "Environment identity has different frozen config hashes"
        )
    protocol = load_frozen_protocol_config(load_artifact_contract().config_path)
    expected_roots = protocol.seed_root_map()
    if dict(seed_roots) != expected_roots:
        raise V015ScoringError("Formal seed roots differ from the freeze")
    expected_derivation = protocol.config()["design_partitions"]["seed_derivation"]
    if seed_derivation != expected_derivation:
        raise V015ScoringError("Formal seed derivation differs from the freeze")
    if prediction_worker_count != 6:
        raise V015ScoringError("Formal prediction worker count must equal 6")
    if (
        isinstance(wall_time_seconds, bool)
        or not isinstance(wall_time_seconds, (int, float))
        or not math.isfinite(float(wall_time_seconds))
        or float(wall_time_seconds) < 0.0
    ):
        raise V015ScoringError("Formal wall time must be finite and nonnegative")

    environment_record = environment_identity.as_manifest_record()
    source_hashes = environment_record["source_byte_hashes"]
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise V015ScoringError("Formal source hash map is empty")
    provenance = {
        "environment_identity": environment_record,
        "environment_identity_sha256": hashlib.sha256(
            canonical_json_bytes(environment_record)
        ).hexdigest(),
        "source_tree_sha256": hashlib.sha256(
            canonical_json_bytes(dict(source_hashes))
        ).hexdigest(),
        "implementation_freeze_record_sha256": (implementation_freeze_record_sha256),
        "protocol_freeze_git_commit": protocol_commit,
        "implementation_source_git_commit": implementation_commit,
        "execution_git_commit": execution_commit,
        "seed_roots": dict(seed_roots),
        "seed_derivation": seed_derivation,
        "prediction_worker_count": prediction_worker_count,
        "wall_time_seconds": float(wall_time_seconds),
    }
    if set(provenance) != RUN_PROVENANCE_KEYS:
        raise V015ScoringError("Run provenance keys changed")

    score_payloads = _score_payloads_without_manifest(result)
    artifact_records = _normalize_formal_artifact_metadata(formal_artifact_metadata)
    model_entry = next(
        record for record in artifact_records if record["path"] == "model_state.json"
    )
    if model_entry["byte_sha256"] != result.score_report["model_state_byte_sha256"]:
        raise V015ScoringError(
            "model_state.json metadata differs from the scored model state"
        )
    score_values = _result_artifacts_without_manifest(result)
    artifact_records.extend(
        {
            "path": filename,
            "row_count": _artifact_row_count(filename, score_values[filename]),
            "byte_count": len(raw),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
        }
        for filename, raw in score_payloads.items()
    )
    manifest = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": environment_identity.config_byte_sha256,
        "config_canonical_sha256": (environment_identity.config_canonical_sha256),
        "model_state_byte_sha256": result.score_report["model_state_byte_sha256"],
        "analysis_counts": result.score_report["analysis_counts"],
        "required_score_artifacts": list(REQUIRED_SCORE_ARTIFACTS),
        "manifest_hash_scope": (
            "all required formal artifacts except run_manifest.json itself; "
            "exposure_log.jsonl is the exact runner-supplied pre-manifest snapshot"
        ),
        "provenance": provenance,
        "artifacts": artifact_records,
        "protocol_deviations": [],
    }
    _canonical_json_artifact_bytes(
        manifest,
        filename="run_manifest.json",
        expected_keys=RUN_MANIFEST_KEYS,
    )
    finalized = replace(result, run_manifest=manifest)
    required_score_artifact_payloads(finalized)
    return finalized


def required_score_artifact_payloads(
    result: V015ScoringResult,
) -> dict[str, bytes]:
    """Return the exact frozen score registry as canonical bytes."""

    if result.run_manifest.get("finalized") is False:
        raise V015ScoringError(
            "run_manifest.json must be finalized by the formal runner"
        )
    artifacts: dict[str, pd.DataFrame | Mapping[str, Any]] = {
        **_result_artifacts_without_manifest(result),
        "run_manifest.json": result.run_manifest,
    }
    if tuple(artifacts) != REQUIRED_SCORE_ARTIFACTS:
        raise V015ScoringError(
            "Score artifact registry differs from the frozen protocol"
        )
    payloads = _score_payloads_without_manifest(result)
    payloads["run_manifest.json"] = _canonical_json_artifact_bytes(
        result.run_manifest,
        filename="run_manifest.json",
        expected_keys=RUN_MANIFEST_KEYS,
    )
    manifest_entries = result.run_manifest.get("artifacts")
    expected_score_entries = [
        {
            "path": filename,
            "row_count": (
                len(artifacts[filename]) if filename.endswith(".csv") else None
            ),
            "byte_count": len(payloads[filename]),
            "byte_sha256": hashlib.sha256(payloads[filename]).hexdigest(),
        }
        for filename in REQUIRED_SCORE_ARTIFACTS[:-1]
    ]
    expected_paths = [
        *REQUIRED_FORMAL_NON_SCORE_ARTIFACTS,
        *REQUIRED_SCORE_ARTIFACTS[:-1],
    ]
    validated_entries = _validate_manifest_artifact_records(
        manifest_entries,
        expected_paths=expected_paths,
    )
    score_entries = validated_entries[len(REQUIRED_FORMAL_NON_SCORE_ARTIFACTS) :]
    if score_entries != expected_score_entries:
        raise V015ScoringError(
            "run_manifest.json does not bind the canonical score artifacts"
        )
    return payloads


def _canonical_committed_equal(
    committed: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    filename: str,
    contract: FrozenArtifactContract,
    formal: bool,
) -> None:
    schema = contract.csv_schema(filename)
    if canonical_csv_bytes(
        committed, schema, contract, formal=formal
    ) != canonical_csv_bytes(recomputed, schema, contract, formal=formal):
        raise V015ScoringError(
            f"{filename} differs byte-for-byte from label-free recomputation"
        )


def _model_state_bytes_from_decoded(state: DecodedModelState) -> bytes:
    payload = build_model_state_payload(
        state.training_state,
        center_development_input_hashes=state.input_byte_hashes["center_development"],
        risk_development_input_hashes=state.input_byte_hashes["risk_development"],
        calibration_input_hashes=state.input_byte_hashes["calibration"],
        software_versions=state.software_versions,
        created_utc=state.created_utc,
    )
    return canonical_json_bytes(payload)


def validate_and_recompute_committed_predictions(
    *,
    prediction_frames: Mapping[str, pd.DataFrame],
    model_state_bytes: bytes,
    formal: bool,
    decoded_model_state: DecodedModelState | None = None,
) -> tuple[DecodedModelState, LabelFreePipelineResult]:
    """Validate commitments and recompute outputs without fitting a model."""

    contract = load_artifact_contract()
    validate_prediction_artifact_bundle(
        prediction_frames,
        contract,
        formal=formal,
        expected_variant_keys=FROZEN_VARIANT_KEYS,
    )
    decoded = deserialize_model_state_json(model_state_bytes)
    if (
        decoded_model_state is not None
        and _model_state_bytes_from_decoded(decoded_model_state) != model_state_bytes
    ):
        raise V015ScoringError(
            "Caller-decoded model state differs from canonical model_state bytes"
        )
    recomputed = recompute_label_free_pipeline(
        prefix_pack=prediction_frames["prefix_pack.csv"],
        forecast_coordinates=prediction_frames["forecast_coordinates.csv"],
        operating_pack=prediction_frames["operating_pack.csv"],
        member_fit_diagnostics=prediction_frames["member_fit_diagnostics.csv"],
        member_forecast_bundle=prediction_frames["member_forecast_bundle.csv"],
        state=decoded.frozen_label_free_state,
    )
    for filename, observed, expected in (
        (
            "prediction_bundle.csv",
            prediction_frames["prediction_bundle.csv"],
            recomputed.prediction_bundle,
        ),
        (
            "risk_bundle.csv",
            prediction_frames["risk_bundle.csv"],
            recomputed.primary_risk_bundle,
        ),
        (
            "decision_bundle.csv",
            prediction_frames["decision_bundle.csv"],
            recomputed.decision_bundle,
        ),
    ):
        _canonical_committed_equal(
            observed,
            expected,
            filename=filename,
            contract=contract,
            formal=formal,
        )
    return decoded, recomputed


def _combine_truth(truth_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    truth_names = tuple(name for name in truth_frames if name.endswith("_truth.csv"))
    combined = pd.concat(
        [truth_frames[name] for name in sorted(truth_names)],
        ignore_index=True,
    )
    return combined.sort_values(
        ["partition", "cluster_id", "forecast_day"], kind="stable"
    ).reset_index(drop=True)


def _add_persistence_iae(
    points: pd.DataFrame,
    trajectories: pd.DataFrame,
    prefix_pack: pd.DataFrame,
) -> pd.DataFrame:
    endpoint = (
        prefix_pack.sort_values("prefix_day", kind="stable")
        .groupby(["protocol_id", "partition", "cluster_id"], sort=False)
        .tail(1)
        .loc[
            :,
            [
                "protocol_id",
                "partition",
                "cluster_id",
                "observed_retention_pct",
            ],
        ]
        .rename(columns={"observed_retention_pct": "_persistence_pct"})
    )
    working = points.merge(
        endpoint,
        on=["protocol_id", "partition", "cluster_id"],
        how="left",
        validate="many_to_one",
    )
    persistence = pd.to_numeric(working["_persistence_pct"], errors="coerce").to_numpy(
        float
    )
    if not np.isfinite(persistence).all():
        raise V015ScoringError("Persistence baseline is nonfinite")
    working["_persistence_error"] = np.abs(
        persistence - working["latent_retention_pct"].to_numpy(float)
    )
    records: list[dict[str, object]] = []
    for key, group in working.groupby(
        ["protocol_id", "partition", "cluster_id"], sort=False
    ):
        ordered = group.sort_values("forecast_day", kind="stable")
        days = ordered["forecast_day"].to_numpy(float)
        errors = ordered["_persistence_error"].to_numpy(float)
        records.append(
            {
                "protocol_id": key[0],
                "partition": key[1],
                "cluster_id": key[2],
                "persistence_trajectory_iae_pp": float(
                    np.trapezoid(errors, days) / (days[-1] - days[0])
                ),
            }
        )
    return trajectories.merge(
        pd.DataFrame(records),
        on=["protocol_id", "partition", "cluster_id"],
        how="left",
        validate="one_to_one",
    )


def _risk_reduction_from_mask(
    frame: pd.DataFrame,
    *,
    issued: pd.Series,
    random_mean: float,
) -> RiskReduction:
    issued_bool = issued.astype(bool)
    count = int(issued_bool.sum())
    if count < 1 or not math.isfinite(random_mean) or random_mean <= 0.0:
        raise V015ScoringError("Risk-reduction denominator is unavailable")
    catastrophic = frame["catastrophic"].astype(bool)
    issued_rate = float(catastrophic[issued_bool].mean())
    return RiskReduction(
        issued_count=count,
        issued_catastrophic_rate=issued_rate,
        random_expected_catastrophic_rate=random_mean,
        relative_risk_reduction=1.0 - issued_rate / random_mean,
    )


def _policy_comparison(
    frame: pd.DataFrame,
    random_rankings: pd.DataFrame,
    *,
    issue_count: int,
    partition: str,
) -> tuple[pd.DataFrame, dict[str, RiskReduction]]:
    if len(random_rankings) < 1:
        raise V015ScoringError("Random-ranking table is empty")
    random_mean = float(
        pd.to_numeric(
            random_rankings["issued_catastrophic_rate"], errors="coerce"
        ).mean()
    )
    if not math.isfinite(random_mean) or random_mean <= 0.0:
        raise V015ScoringError("Random-ranking mean risk is unavailable")
    eligible = frame["hard_eligible_visible_stress"].astype(bool)
    rows: list[dict[str, object]] = []
    reductions: dict[str, RiskReduction] = {}
    for score_id in RISK_SCORE_IDS:
        issued = rank_policy(
            frame,
            protocol_id=FROZEN_PROTOCOL_ID,
            arm=score_id,
            score_column=f"risk_{score_id}",
            predictor_hash_column=f"risk_hash_{score_id}",
            issue_count=issue_count,
        )
        reduction = _risk_reduction_from_mask(
            frame, issued=issued, random_mean=random_mean
        )
        reductions[score_id] = reduction
        rows.append(
            {
                "partition": partition,
                "score_id": score_id,
                "source_count": len(frame),
                "eligible_count": int(eligible.sum()),
                "issued_count": reduction.issued_count,
                "source_coverage": reduction.issued_count / len(frame),
                "eligible_coverage": reduction.issued_count / int(eligible.sum()),
                "issued_catastrophic_rate": (reduction.issued_catastrophic_rate),
                "mean_random_issued_catastrophic_rate": random_mean,
                "relative_risk_reduction": (reduction.relative_risk_reduction),
            }
        )
    return pd.DataFrame(rows), reductions


def _risk_coverage_curves(
    test: pd.DataFrame,
) -> list[dict[str, Any]]:
    source_count = len(test)
    eligible = test["hard_eligible_visible_stress"].astype(bool)
    eligible_count = int(eligible.sum())
    catastrophic = test["catastrophic"].astype(bool)
    random_rate = (
        float(catastrophic[eligible].mean()) if eligible_count > 0 else math.nan
    )
    records: list[dict[str, Any]] = []
    for score_id in RISK_SCORE_IDS:
        for fraction in (0.25, 0.50, 0.75):
            issue_count = math.floor(source_count * fraction)
            reason = ""
            issued_rate: float | None = None
            reduction: float | None = None
            if issue_count < 1:
                reason = "target issue count is zero"
            elif eligible_count < issue_count:
                reason = (
                    f"eligible_count={eligible_count} below "
                    f"target_issue_count={issue_count}"
                )
            elif not math.isfinite(random_rate) or random_rate <= 0.0:
                reason = "eligible-pool catastrophic prevalence is unavailable"
            else:
                issued = rank_policy(
                    test,
                    protocol_id=FROZEN_PROTOCOL_ID,
                    arm=score_id,
                    score_column=f"risk_{score_id}",
                    predictor_hash_column=f"risk_hash_{score_id}",
                    issue_count=issue_count,
                )
                issued_rate = float(catastrophic[issued].mean())
                reduction = 1.0 - issued_rate / random_rate
            records.append(
                {
                    "partition": "test",
                    "score_id": score_id,
                    "issuance_fraction": fraction,
                    "source_count": source_count,
                    "eligible_count": eligible_count,
                    "target_issue_count": issue_count,
                    "available": reason == "",
                    "unavailable_reason": reason,
                    "issued_catastrophic_rate": issued_rate,
                    "analytic_random_expected_rate": (
                        random_rate if math.isfinite(random_rate) else None
                    ),
                    "relative_risk_reduction": reduction,
                    "secondary_descriptive_only": True,
                }
            )
    return records


def _unavailable_risk_coverage_curves(reason: str) -> list[dict[str, Any]]:
    return [
        {
            "partition": "test",
            "score_id": score_id,
            "issuance_fraction": fraction,
            "source_count": 0,
            "eligible_count": 0,
            "target_issue_count": 0,
            "available": False,
            "unavailable_reason": reason,
            "issued_catastrophic_rate": None,
            "analytic_random_expected_rate": None,
            "relative_risk_reduction": None,
            "secondary_descriptive_only": True,
        }
        for score_id in RISK_SCORE_IDS
        for fraction in (0.25, 0.50, 0.75)
    ]


def _isotonic_calibration_diagnostics(
    trajectories: pd.DataFrame,
    risk_bundle: pd.DataFrame,
    decoded: DecodedModelState,
) -> list[dict[str, Any]]:
    calibration = trajectories.loc[
        trajectories["partition"].eq("calibration"),
        ["protocol_id", "partition", "cluster_id", "catastrophic"],
    ]
    records: list[dict[str, Any]] = []
    for score_id, threshold_count in (
        (
            "prefix_only",
            len(decoded.training_state.calibration.prefix_only_isotonic.x_thresholds),
        ),
        (
            "visible_stress",
            len(
                decoded.training_state.calibration.visible_stress_isotonic.x_thresholds
            ),
        ),
    ):
        risks = risk_bundle.loc[
            risk_bundle["partition"].eq("calibration")
            & risk_bundle["score_id"].eq(score_id),
            [
                "protocol_id",
                "partition",
                "cluster_id",
                "calibrated_catastrophic_probability",
            ],
        ]
        joined = calibration.merge(
            risks,
            on=["protocol_id", "partition", "cluster_id"],
            how="inner",
            validate="one_to_one",
        )
        probabilities = pd.to_numeric(
            joined["calibrated_catastrophic_probability"],
            errors="coerce",
        ).to_numpy(float)
        labels = joined["catastrophic"].astype(bool).to_numpy(float)
        reason = ""
        if len(joined) != 900:
            reason = f"calibration_count={len(joined)} expected=900"
        elif not np.isfinite(probabilities).all():
            reason = "calibrated probabilities are nonfinite"
        elif np.any((probabilities < 0.0) | (probabilities > 1.0)):
            reason = "calibrated probabilities are outside [0,1]"
        records.append(
            {
                "score_id": score_id,
                "n": len(joined),
                "positive_count": int(labels.sum()),
                "available": reason == "",
                "unavailable_reason": reason,
                "mean_predicted_probability": (
                    float(probabilities.mean()) if reason == "" else None
                ),
                "observed_catastrophic_prevalence": (
                    float(labels.mean()) if reason == "" else None
                ),
                "brier_score": (
                    float(np.mean(np.square(probabilities - labels)))
                    if reason == ""
                    else None
                ),
                "isotonic_threshold_count": threshold_count,
                "descriptive_only": True,
            }
        )
    return records


def _unavailable_isotonic_diagnostics(
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "score_id": score_id,
            "n": 0,
            "positive_count": 0,
            "available": False,
            "unavailable_reason": reason,
            "mean_predicted_probability": None,
            "observed_catastrophic_prevalence": None,
            "brier_score": None,
            "isotonic_threshold_count": None,
            "descriptive_only": True,
        }
        for score_id in ("prefix_only", "visible_stress")
    ]


def _numeric_summary(
    values: Sequence[object],
    *,
    unavailable_reason: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) == 0:
        return {
            "available": False,
            "unavailable_reason": unavailable_reason,
            "n": 0,
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "available": True,
        "unavailable_reason": "",
        "n": len(finite),
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite, ddof=0)),
        "median": float(np.quantile(finite, 0.5, method="linear")),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _structure_diagnostics(
    feature_bundle: pd.DataFrame,
    member_fit_diagnostics: pd.DataFrame,
) -> list[dict[str, Any]]:
    diagnostic_rows: list[dict[str, object]] = []
    for key, group in member_fit_diagnostics.groupby(
        ["protocol_id", "partition", "cluster_id"], sort=False
    ):
        credible_values = group["credible_variant"].tolist()
        if any(not isinstance(value, (bool, np.bool_)) for value in credible_values):
            raise V015ScoringError(
                "member_fit_diagnostics credible_variant is not boolean"
            )
        credible = group.loc[np.asarray(credible_values, dtype=bool)]
        diagnostic_rows.append(
            {
                "protocol_id": key[0],
                "partition": key[1],
                "cluster_id": key[2],
                "credible_variant_count": len(credible),
                "credible_variant_boundary_hit_fraction": (
                    float(
                        pd.to_numeric(
                            credible["parameter_boundary_hit_fraction"],
                            errors="coerce",
                        ).mean()
                    )
                    if not credible.empty
                    else math.nan
                ),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    working = feature_bundle.merge(
        diagnostics,
        on=["protocol_id", "partition", "cluster_id"],
        how="left",
        validate="one_to_one",
    )
    fields = (
        "successful_structure_family_count",
        "effective_unique_shape_count",
        "fit_failure_count",
        "parameter_boundary_hit_fraction",
        "credible_variant_count",
        "credible_variant_boundary_hit_fraction",
    )
    records: list[dict[str, Any]] = []
    for partition, group in working.groupby("partition", sort=True):
        records.append(
            {
                "partition": str(partition),
                "cluster_count": len(group),
                "available": True,
                "unavailable_reason": "",
                "metrics": {
                    field: _numeric_summary(
                        group[field],
                        unavailable_reason=(
                            f"{field} has no finite values in {partition}"
                        ),
                    )
                    for field in fields
                },
                "descriptive_only": True,
            }
        )
    return records


def _unavailable_structure_diagnostics(reason: str) -> list[dict[str, Any]]:
    return [
        {
            "partition": partition,
            "cluster_count": 0,
            "metrics": None,
            "available": False,
            "unavailable_reason": reason,
            "descriptive_only": True,
        }
        for partition in load_artifact_contract().partitions
    ]


def _test_audit_distribution_shift(
    operating_pack: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> dict[str, Any]:
    test = operating_pack.loc[operating_pack["partition"].eq("test")]
    audit = operating_pack.loc[operating_pack["partition"].eq("audit")]
    reason = ""
    if test.empty or audit.empty:
        reason = "test or audit operating cohort is empty"
    field_records: list[dict[str, Any]] = []
    for field in REAL_OPERATING_FIELDS:
        test_values = pd.to_numeric(test[field], errors="coerce").to_numpy(float)
        audit_values = pd.to_numeric(audit[field], errors="coerce").to_numpy(float)
        field_reason = reason
        if not field_reason and (
            not np.isfinite(test_values).all() or not np.isfinite(audit_values).all()
        ):
            field_reason = "operating values are nonfinite"
        test_mean = float(np.mean(test_values)) if field_reason == "" else None
        audit_mean = float(np.mean(audit_values)) if field_reason == "" else None
        test_std = float(np.std(test_values, ddof=0)) if field_reason == "" else None
        audit_std = float(np.std(audit_values, ddof=0)) if field_reason == "" else None
        pooled = (
            math.sqrt((test_std**2 + audit_std**2) / 2.0)
            if test_std is not None and audit_std is not None
            else math.nan
        )
        smd = (
            (audit_mean - test_mean) / pooled
            if (
                field_reason == ""
                and audit_mean is not None
                and test_mean is not None
                and pooled > 0.0
            )
            else None
        )
        if field_reason == "" and not pooled > 0.0:
            field_reason = "pooled standard deviation is zero"
        field_records.append(
            {
                "field": field,
                "test_mean": test_mean,
                "test_standard_deviation": test_std,
                "audit_mean": audit_mean,
                "audit_standard_deviation": audit_std,
                "standardized_mean_difference_audit_minus_test": smd,
                "available": field_reason == "",
                "unavailable_reason": field_reason,
            }
        )
    ordinary = trajectories.loc[
        trajectories["partition"].isin(("test", "audit")),
        ["partition", "truth_family"],
    ]
    family_records: list[dict[str, Any]] = []
    for family in TEST_FAMILIES:
        test_count = int(
            (
                ordinary["partition"].eq("test") & ordinary["truth_family"].eq(family)
            ).sum()
        )
        audit_count = int(
            (
                ordinary["partition"].eq("audit") & ordinary["truth_family"].eq(family)
            ).sum()
        )
        family_records.append(
            {
                "truth_family": family,
                "test_count": test_count,
                "test_proportion": (test_count / len(test) if len(test) else None),
                "audit_count": audit_count,
                "audit_proportion": (audit_count / len(audit) if len(audit) else None),
                "audit_minus_test_proportion": (
                    audit_count / len(audit) - test_count / len(test)
                    if len(test) and len(audit)
                    else None
                ),
            }
        )
    return {
        "available": reason == "",
        "unavailable_reason": reason,
        "test_count": len(test),
        "audit_count": len(audit),
        "operating_fields": field_records,
        "family_proportions": family_records,
        "descriptive_only": True,
    }


def _unavailable_distribution_shift(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "unavailable_reason": reason,
        "test_count": 0,
        "audit_count": 0,
        "operating_fields": [],
        "family_proportions": [],
        "descriptive_only": True,
    }


def _bootstrap_summary(
    replicates: pd.DataFrame,
    *,
    expected_count: int,
) -> Mapping[str, float] | None:
    if len(replicates) != expected_count:
        raise V015ScoringError("Bootstrap replicate count changed")
    defined = replicates["defined"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    if not defined.all():
        return None
    columns = {
        "visible_one_sided_95_lower": (
            "visible_stress_risk_reduction",
            0.05,
        ),
        "increment_one_sided_95_lower": (
            "visible_minus_prefix_increment",
            0.05,
        ),
        "placebo_two_sided_95_lower": (
            "placebo_minus_prefix_increment",
            0.025,
        ),
        "placebo_two_sided_95_upper": (
            "placebo_minus_prefix_increment",
            0.975,
        ),
    }
    result: dict[str, float] = {}
    for output, (column, probability) in columns.items():
        values = pd.to_numeric(replicates[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            return None
        result[output] = float(np.quantile(values, probability, method="linear"))
    return result


def _coverage_record(
    coverage_id: str,
    summary: CoverageSummary | None,
    *,
    reason: str = "",
) -> dict[str, object]:
    if summary is None:
        return {
            "coverage_id": coverage_id,
            "n": 0,
            "covered": 0,
            "coverage": math.nan,
            "one_sided_95_lower": math.nan,
            "median_max_width_pp": math.nan,
            "percentile_95_max_width_pp": math.nan,
            "available": False,
            "unavailable_reason": reason,
        }
    return {
        "coverage_id": coverage_id,
        "n": summary.n,
        "covered": summary.covered,
        "coverage": summary.coverage,
        "one_sided_95_lower": summary.one_sided_95_lower,
        "median_max_width_pp": summary.median_max_width_pp,
        "percentile_95_max_width_pp": summary.percentile_95_max_width_pp,
        "available": True,
        "unavailable_reason": "",
    }


def _family_error_table(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (partition, family), group in trajectories.groupby(
        ["partition", "truth_family"], sort=True
    ):
        rows.append(
            {
                "partition": partition,
                "truth_family": family,
                "cluster_count": len(group),
                "catastrophic_count": int(group["catastrophic"].sum()),
                "mean_center_endpoint_absolute_error_pp": float(
                    group["center_endpoint_absolute_error_pp"].mean()
                ),
                "mean_center_trajectory_iae_pp": float(
                    group["center_trajectory_iae_pp"].mean()
                ),
                "mean_sqrt_trajectory_iae_pp": float(
                    group["sqrt_trajectory_iae_pp"].mean()
                ),
                "mean_bounded_power_trajectory_iae_pp": float(
                    group["bounded_power_trajectory_iae_pp"].mean()
                ),
                "mean_persistence_trajectory_iae_pp": float(
                    group["persistence_trajectory_iae_pp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _family_metric_table(
    trajectories: pd.DataFrame,
    subset_metrics: pd.DataFrame,
) -> pd.DataFrame:
    errors = _family_error_table(trajectories)
    records: dict[str, dict[str, object]] = {}
    for row in errors.to_dict(orient="records"):
        partition = str(row["partition"])
        family = str(row["truth_family"])
        metric_id = f"{partition}_family_{family}"
        records[metric_id] = {
            "metric_id": metric_id,
            "partition": partition,
            "truth_family": family,
            "source_count": int(row["cluster_count"]),
            "eligible_count": math.nan,
            "catastrophic_count": int(row["catastrophic_count"]),
            "issued_count": math.nan,
            "prefix_only_issued_catastrophic_rate": math.nan,
            "prefix_only_risk_reduction": math.nan,
            "visible_stress_issued_catastrophic_rate": math.nan,
            "visible_stress_risk_reduction": math.nan,
            "visible_minus_prefix_increment": math.nan,
            "mean_center_endpoint_absolute_error_pp": row[
                "mean_center_endpoint_absolute_error_pp"
            ],
            "mean_center_trajectory_iae_pp": row["mean_center_trajectory_iae_pp"],
            "mean_sqrt_trajectory_iae_pp": row["mean_sqrt_trajectory_iae_pp"],
            "mean_bounded_power_trajectory_iae_pp": row[
                "mean_bounded_power_trajectory_iae_pp"
            ],
            "mean_persistence_trajectory_iae_pp": row[
                "mean_persistence_trajectory_iae_pp"
            ],
        }
    for row in subset_metrics.to_dict(orient="records"):
        metric_id = str(row["subset_id"])
        if metric_id.startswith("test_"):
            partition = "test"
        elif metric_id.startswith("audit_"):
            partition = "audit"
        else:
            raise V015ScoringError(f"Unknown frozen family-metric scope: {metric_id}")
        family = (
            metric_id.removeprefix(f"{partition}_family_")
            if metric_id.startswith(f"{partition}_family_")
            else ""
        )
        record = records.setdefault(
            metric_id,
            {
                "metric_id": metric_id,
                "partition": partition,
                "truth_family": family,
                "source_count": math.nan,
                "eligible_count": math.nan,
                "catastrophic_count": math.nan,
                "issued_count": math.nan,
                "prefix_only_issued_catastrophic_rate": math.nan,
                "prefix_only_risk_reduction": math.nan,
                "visible_stress_issued_catastrophic_rate": math.nan,
                "visible_stress_risk_reduction": math.nan,
                "visible_minus_prefix_increment": math.nan,
                "mean_center_endpoint_absolute_error_pp": math.nan,
                "mean_center_trajectory_iae_pp": math.nan,
                "mean_sqrt_trajectory_iae_pp": math.nan,
                "mean_bounded_power_trajectory_iae_pp": math.nan,
                "mean_persistence_trajectory_iae_pp": math.nan,
            },
        )
        for column in SUBSET_METRIC_COLUMNS:
            if column != "subset_id":
                record[column] = row[column]
    return pd.DataFrame(records.values(), columns=FAMILY_METRIC_COLUMNS)


def _matched_pair_table(
    intrinsic_scores: pd.DataFrame,
    stress_scores: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for experiment_id, frame in (
        ("intrinsic", intrinsic_scores),
        ("stress_plan", stress_scores),
    ):
        for row in frame.to_dict(orient="records"):
            records.append(
                {
                    "experiment_id": experiment_id,
                    "pair_id": row["pair_id"],
                    "left_cluster_id": row["left_cluster_id"],
                    "right_cluster_id": row["right_cluster_id"],
                    "arm_a_exact_equal": row.get(
                        "arm_a_exact_equal",
                        row.get("arm_a_exact_tie", pd.NA),
                    ),
                    "arm_b_exact_equal": row.get("arm_b_exact_equal", pd.NA),
                    "interval_width_exact_equal": row.get(
                        "interval_width_exact_equal", pd.NA
                    ),
                    "both_futures_simultaneously_covered": row.get(
                        "both_futures_simultaneously_covered", pd.NA
                    ),
                    "max_interval_width_pp": row.get("max_interval_width_pp", math.nan),
                    "center_endpoint_error_delta_pp": row.get(
                        "center_endpoint_error_delta_pp", math.nan
                    ),
                    "arm_b_risk_delta": row.get("arm_b_risk_delta", math.nan),
                    "arm_b_correct_error_order": row.get(
                        "arm_b_correct_error_order", pd.NA
                    ),
                }
            )
    return pd.DataFrame(records, columns=MATCHED_PAIR_COLUMNS)


def _gate_frame(gates: Sequence[GateEvaluation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id": gate.gate_id,
                "state": gate.state,
                "estimate": gate.estimate,
                "threshold": gate.threshold,
                "reasons": ";".join(gate.reasons),
            }
            for gate in gates
        ]
    )


def _inconclusive(
    gate_id: str,
    threshold: str,
    reason: str,
) -> GateEvaluation:
    return GateEvaluation(
        gate_id=gate_id,
        state="inconclusive",
        estimate=None,
        threshold=threshold,
        reasons=(reason,),
    )


def _primary_gates(
    *,
    visible: RiskReduction,
    prefix: RiskReduction,
    bootstrap: Mapping[str, float] | None,
    core: CoverageSummary | None,
    intrinsic: CoverageSummary | None,
    issued_iae_delta: float,
) -> tuple[GateEvaluation, ...]:
    increment = visible.relative_risk_reduction - prefix.relative_risk_reduction
    if bootstrap is not None and core is not None and intrinsic is not None:
        return primary_gate_evaluations(
            visible_reduction=visible.relative_risk_reduction,
            increment=increment,
            bootstrap_summary=bootstrap,
            core_coverage=core,
            intrinsic_coverage=intrinsic,
            issued_center_minus_baseline_iae_pp=issued_iae_delta,
        )
    gates: list[GateEvaluation] = []
    if bootstrap is None:
        gates.extend(
            (
                _inconclusive(
                    "visible_stress_catastrophic_risk_reduction",
                    "estimate >= 0.30 and bootstrap lower > 0",
                    "bootstrap is undefined",
                ),
                _inconclusive(
                    "visible_stress_increment_over_prefix_only",
                    "estimate >= 0.10 and paired-bootstrap lower > 0",
                    "bootstrap is undefined",
                ),
            )
        )
    else:
        placeholder = CoverageSummary(1, 1, 1.0, 1.0, 0.0, 0.0)
        gates.extend(
            primary_gate_evaluations(
                visible_reduction=visible.relative_risk_reduction,
                increment=increment,
                bootstrap_summary=bootstrap,
                core_coverage=placeholder,
                intrinsic_coverage=placeholder,
                issued_center_minus_baseline_iae_pp=issued_iae_delta,
            )[:2]
        )
    gates.append(
        (
            _inconclusive(
                "core_test_simultaneous_trajectory_coverage",
                "coverage/CP/width frozen gates",
                "one or more core intervals are legitimately unavailable",
            )
            if core is None
            else primary_gate_evaluations(
                visible_reduction=visible.relative_risk_reduction,
                increment=increment,
                bootstrap_summary=bootstrap
                or {
                    "visible_one_sided_95_lower": 0.0,
                    "increment_one_sided_95_lower": 0.0,
                },
                core_coverage=core,
                intrinsic_coverage=CoverageSummary(1, 1, 1.0, 1.0, 0.0, 0.0),
                issued_center_minus_baseline_iae_pp=issued_iae_delta,
            )[2]
        )
    )
    gates.append(
        (
            _inconclusive(
                "intrinsic_pair_simultaneous_both_future_coverage",
                "coverage/CP/width frozen gates",
                "one or more intrinsic-pair intervals are unavailable",
            )
            if intrinsic is None
            else primary_gate_evaluations(
                visible_reduction=visible.relative_risk_reduction,
                increment=increment,
                bootstrap_summary=bootstrap
                or {
                    "visible_one_sided_95_lower": 0.0,
                    "increment_one_sided_95_lower": 0.0,
                },
                core_coverage=CoverageSummary(1, 1, 1.0, 1.0, 0.0, 0.0),
                intrinsic_coverage=intrinsic,
                issued_center_minus_baseline_iae_pp=issued_iae_delta,
            )[3]
        )
    )
    gates.append(
        GateEvaluation(
            gate_id="issued_center_trajectory_iae_noninferiority",
            state="pass" if issued_iae_delta <= 0.10 else "fail",
            estimate=issued_iae_delta,
            threshold="<= +0.10 pp",
        )
    )
    return tuple(gates)


def _empty_frame(filename: str) -> pd.DataFrame:
    return pd.DataFrame(columns=SCORE_FRAME_SCHEMAS[filename].columns)


def _void_result(reason: str, model_state_bytes: bytes) -> V015ScoringResult:
    return _terminal_result(
        reason,
        model_state_bytes,
        status_kind="void",
    )


def _inconclusive_result(
    reason: str,
    model_state_bytes: bytes,
) -> V015ScoringResult:
    return _terminal_result(
        reason,
        model_state_bytes,
        status_kind="inconclusive_not_success",
    )


def _terminal_result(
    reason: str,
    model_state_bytes: bytes,
    *,
    status_kind: str,
) -> V015ScoringResult:
    if status_kind not in {"void", "inconclusive_not_success"}:
        raise V015ScoringError("Unknown terminal result status")
    unavailable_reason = (
        "protocol void" if status_kind == "void" else f"protocol inconclusive: {reason}"
    )
    gates = tuple(
        _inconclusive(gate_id, "not evaluated", unavailable_reason)
        for gate_id in REQUIRED_GATE_IDS
    )
    if status_kind == "void":
        status = resolve_result_status(gates, void_reasons=(reason,))
    else:
        status = resolve_result_status(
            gates,
            external_inconclusive_reasons=(reason,),
        )
    gate_frame = _gate_frame(gates)
    score_report = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "model_state_byte_sha256": hashlib.sha256(model_state_bytes).hexdigest(),
        "analysis_counts": {
            "random_rankings": RANDOM_RANKING_COUNT,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "stress_permutations": STRESS_PERMUTATIONS,
        },
        "selected_mean_baseline": None,
        "test_primary_estimates": None,
        "policy_comparison": [],
        "risk_coverage_curves_secondary": (
            _unavailable_risk_coverage_curves(unavailable_reason)
        ),
        "coverage_metrics": [],
        "isotonic_calibration_diagnostics": (
            _unavailable_isotonic_diagnostics(unavailable_reason)
        ),
        "structure_diagnostics": _unavailable_structure_diagnostics(unavailable_reason),
        "test_audit_distribution_shift": _unavailable_distribution_shift(
            unavailable_reason
        ),
        "gate_evaluations": _json_records(gate_frame),
        "stress_plan_summary": None,
        "stress_permutation_summary": None,
        "status": status,
        "protocol_deviations": [],
    }
    negative_controls = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "available": False,
        "unavailable_reason": unavailable_reason,
        "placebo_point_increment": None,
        "placebo_bootstrap_two_sided_95_interval": None,
        "stress_permutation_summary": None,
        "gate_evaluations": [],
    }
    artifacts_without_manifest: dict[str, pd.DataFrame | Mapping[str, Any]] = {
        "point_scores.csv": _empty_frame("point_scores.csv"),
        "trajectory_scores.csv": _empty_frame("trajectory_scores.csv"),
        "family_metrics.csv": _empty_frame("family_metrics.csv"),
        "matched_pair_scores.csv": _empty_frame("matched_pair_scores.csv"),
        "bootstrap_replicates.csv": _empty_frame("bootstrap_replicates.csv"),
        "random_ranking_metrics.csv": _empty_frame("random_ranking_metrics.csv"),
        "stress_permutation_metrics.csv": _empty_frame(
            "stress_permutation_metrics.csv"
        ),
        "negative_control_metrics.json": negative_controls,
        "score_report.json": score_report,
    }
    result = V015ScoringResult(
        point_scores=artifacts_without_manifest["point_scores.csv"],
        trajectory_scores=artifacts_without_manifest["trajectory_scores.csv"],
        family_metrics=artifacts_without_manifest["family_metrics.csv"],
        matched_pair_scores=artifacts_without_manifest["matched_pair_scores.csv"],
        bootstrap_replicates=artifacts_without_manifest["bootstrap_replicates.csv"],
        random_ranking_metrics=artifacts_without_manifest["random_ranking_metrics.csv"],
        stress_permutation_metrics=artifacts_without_manifest[
            "stress_permutation_metrics.csv"
        ],
        negative_control_metrics=negative_controls,
        score_report=score_report,
        run_manifest=dict(_PROVISIONAL_RUN_MANIFEST),
    )
    return result


def _score_committed_artifacts(
    *,
    prediction_frames: Mapping[str, pd.DataFrame],
    truth_frames: Mapping[str, pd.DataFrame],
    model_state_bytes: bytes,
    decoded_model_state: DecodedModelState | None,
    formal: bool,
    counts: _AnalysisCounts,
) -> V015ScoringResult:
    contract = load_artifact_contract()
    validate_sealed_truth_bundle(truth_frames, contract, formal=formal)
    decoded, recomputed = validate_and_recompute_committed_predictions(
        prediction_frames=prediction_frames,
        model_state_bytes=model_state_bytes,
        formal=formal,
        decoded_model_state=decoded_model_state,
    )
    truth = _combine_truth(truth_frames)
    points, trajectories = score_trajectory_table(
        prediction_frames["prediction_bundle.csv"],
        truth,
        prediction_frames["risk_bundle.csv"],
        prediction_frames["decision_bundle.csv"],
    )
    trajectories = _add_persistence_iae(
        points, trajectories, prediction_frames["prefix_pack.csv"]
    )

    intrinsic_mapping = truth_frames["intrinsic_matched_pairs.csv"]
    stress_mapping = truth_frames["stress_plan_matched_pairs.csv"]
    protocol = load_frozen_protocol_config(contract.config_path)
    validate_intrinsic_pair_construction(
        prediction_frames["prefix_pack.csv"],
        prediction_frames["operating_pack.csv"],
        truth_frames["intrinsic_matched_truth.csv"],
        intrinsic_mapping,
        protocol,
    )
    validate_stress_plan_pair_construction(
        prediction_frames["prefix_pack.csv"],
        prediction_frames["operating_pack.csv"],
        truth_frames["stress_plan_matched_truth.csv"],
        stress_mapping,
        protocol,
    )
    validate_intrinsic_output_invariance(points, trajectories, intrinsic_mapping)
    validate_stress_plan_arm_a_invariance(points, trajectories, stress_mapping)
    common_pool_gates = common_pool_gate_evaluations(trajectories)
    test = trajectories.loc[trajectories["partition"].eq("test")].copy()
    random_rankings = deterministic_random_rankings(
        test,
        issue_count=TEST_ISSUE_COUNT,
        rankings=counts.random_rankings,
    )
    random_rankings.insert(0, "partition", "test")
    policy_test, test_reductions = _policy_comparison(
        test,
        random_rankings,
        issue_count=TEST_ISSUE_COUNT,
        partition="test",
    )
    audit = trajectories.loc[trajectories["partition"].eq("audit")].copy()
    audit_random = deterministic_random_rankings(
        audit,
        issue_count=AUDIT_ISSUE_COUNT,
        rankings=counts.random_rankings,
    )
    audit_random.insert(0, "partition", "audit")
    policy_audit, _ = _policy_comparison(
        audit,
        audit_random,
        issue_count=AUDIT_ISSUE_COUNT,
        partition="audit",
    )
    policy = pd.concat((policy_test, policy_audit), ignore_index=True)
    random_ranking_metrics = pd.concat(
        (random_rankings, audit_random), ignore_index=True
    )

    bootstrap = bootstrap_risk_reductions(
        test,
        protocol_id=FROZEN_PROTOCOL_ID,
        issue_count=TEST_ISSUE_COUNT,
        resamples=counts.bootstrap_resamples,
    )
    bootstrap_summary = _bootstrap_summary(
        bootstrap, expected_count=counts.bootstrap_resamples
    )

    core_rows = test.loc[test["truth_family"].isin(CORE_FAMILIES)]
    if (
        core_rows[["simultaneous_interval_covered", "max_interval_width_pp"]]
        .isna()
        .any()
        .any()
    ):
        core_coverage = None
    else:
        core_coverage = core_test_coverage_summary(trajectories)
    if (
        core_rows[["base_simultaneous_interval_covered", "base_max_interval_width_pp"]]
        .isna()
        .any()
        .any()
    ):
        base_core_coverage = None
    else:
        base_core_coverage = core_test_coverage_summary(trajectories, calibrated=False)
    intrinsic_scores, intrinsic_coverage = evaluate_intrinsic_pairs(
        trajectories, intrinsic_mapping
    )
    stress_scores, stress_summary = evaluate_stress_plan_pairs(
        trajectories, stress_mapping
    )

    selected_baseline = decoded.training_state.calibration.selected_mean_baseline
    baseline_columns = {
        "target_prefix_persistence": "persistence_trajectory_iae_pp",
        "target_prefix_sqrt_time": "sqrt_trajectory_iae_pp",
        "target_prefix_bounded_power_law": ("bounded_power_trajectory_iae_pp"),
    }
    try:
        baseline_column = baseline_columns[selected_baseline]
    except KeyError as exc:
        raise V015ScoringError(
            "Frozen selected mean baseline is not a declared comparator"
        ) from exc
    test_iae_delta = issued_center_minus_baseline_iae(
        test,
        issued_column="issued_visible_stress",
        baseline_iae_column=baseline_column,
        expected_issue_count=TEST_ISSUE_COUNT,
    )
    audit_iae_delta = issued_center_minus_baseline_iae(
        audit,
        issued_column="issued_visible_stress",
        baseline_iae_column=baseline_column,
        expected_issue_count=AUDIT_ISSUE_COUNT,
    )

    test_subset, test_gates = evaluate_test_safety_gates(
        trajectories, protocol_id=FROZEN_PROTOCOL_ID
    )
    audit_subset, audit_gates = evaluate_audit_directional_gates(
        trajectories,
        protocol_id=FROZEN_PROTOCOL_ID,
        issued_center_minus_baseline_iae_pp=audit_iae_delta,
    )
    subset_metrics = pd.concat((test_subset, audit_subset), ignore_index=True)

    visible = test_reductions["visible_stress"]
    prefix = test_reductions["prefix_only"]
    placebo = test_reductions["placebo_8"]
    increment = visible.relative_risk_reduction - prefix.relative_risk_reduction
    placebo_increment = placebo.relative_risk_reduction - prefix.relative_risk_reduction
    primary_gates = _primary_gates(
        visible=visible,
        prefix=prefix,
        bootstrap=bootstrap_summary,
        core=core_coverage,
        intrinsic=intrinsic_coverage,
        issued_iae_delta=test_iae_delta,
    )
    if bootstrap_summary is None:
        placebo_gates = (
            GateEvaluation(
                gate_id="placebo_point_negative_control",
                state=("pass" if abs(placebo_increment) < 0.05 else "fail"),
                estimate=placebo_increment,
                threshold="abs(increment) < 0.05",
            ),
            _inconclusive(
                "placebo_interval_negative_control",
                "two-sided 95% interval contains 0",
                "bootstrap is undefined",
            ),
        )
    else:
        placebo_gates = placebo_negative_control_gates(
            placebo_minus_prefix_increment=placebo_increment,
            bootstrap_summary=bootstrap_summary,
        )

    permutation_input = trajectories.merge(
        recomputed.feature_bundle,
        on=["protocol_id", "partition", "cluster_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    permutations = stress_permutation_metrics(
        permutation_input,
        prediction_frames["operating_pack.csv"],
        prediction_frames["prefix_pack.csv"],
        prediction_frames["forecast_coordinates.csv"],
        visible_stress_state=decoded.frozen_label_free_state.visible_stress_risk,
        protocol_id=FROZEN_PROTOCOL_ID,
        random_expected_catastrophic_rate=(visible.random_expected_catastrophic_rate),
        observed_prefix_only_risk_reduction=(prefix.relative_risk_reduction),
        issue_count=TEST_ISSUE_COUNT,
        permutations=counts.stress_permutations,
    )
    permutation_values = pd.to_numeric(
        permutations["visible_minus_prefix_increment"], errors="coerce"
    ).to_numpy(float)
    if (
        len(permutations) != counts.stress_permutations
        or not np.isfinite(permutation_values).all()
    ):
        raise V015ScoringError("Stress-permutation output is incomplete")
    lower_count = int(np.sum(permutation_values < increment))
    permutation_summary = StressPermutationSummary(
        permutation_count=len(permutations),
        observed_visible_minus_prefix_increment=increment,
        strictly_lower_count=lower_count,
        strictly_lower_fraction=lower_count / len(permutations),
        gate_passed=(lower_count * STRESS_PERMUTATIONS >= 9900 * len(permutations)),
    )
    integrity_gates = (
        GateEvaluation(
            "stress_permutation_negative_control",
            "pass" if permutation_summary.gate_passed else "fail",
            permutation_summary.strictly_lower_fraction,
            "at least 9900 of 10000 strictly lower",
        ),
        GateEvaluation(
            "intrinsic_output_invariance",
            "pass",
            True,
            "all 250 normalized pair outputs exactly equal",
        ),
        GateEvaluation(
            "stress_plan_arm_a_invariance",
            "pass",
            True,
            "all 250 Arm-A normalized pair outputs exactly equal",
        ),
    )
    gates = (
        *common_pool_gates,
        *primary_gates,
        *test_gates,
        *placebo_gates,
        *integrity_gates,
        *audit_gates,
    )
    status = resolve_result_status(gates)

    coverage = pd.DataFrame(
        (
            _coverage_record(
                "core_test_calibrated",
                core_coverage,
                reason="one or more core intervals are unavailable",
            ),
            _coverage_record(
                "core_test_structural",
                base_core_coverage,
                reason="one or more core structural bands are unavailable",
            ),
            _coverage_record(
                "intrinsic_pairs_calibrated",
                intrinsic_coverage,
                reason="one or more intrinsic-pair intervals are unavailable",
            ),
        )
    )
    gate_frame = _gate_frame(gates).loc[:, GATE_COLUMNS]
    policy_frame = policy.loc[:, POLICY_COMPARISON_COLUMNS]
    coverage_frame = coverage.loc[:, COVERAGE_METRIC_COLUMNS]
    risk_coverage_curves = _risk_coverage_curves(test)
    isotonic_diagnostics = _isotonic_calibration_diagnostics(
        trajectories,
        prediction_frames["risk_bundle.csv"],
        decoded,
    )
    structure_diagnostics = _structure_diagnostics(
        recomputed.feature_bundle,
        prediction_frames["member_fit_diagnostics.csv"],
    )
    distribution_shift = _test_audit_distribution_shift(
        prediction_frames["operating_pack.csv"],
        trajectories,
    )
    permutation_summary_payload = {
        "permutation_count": permutation_summary.permutation_count,
        "observed_visible_minus_prefix_increment": (
            permutation_summary.observed_visible_minus_prefix_increment
        ),
        "strictly_lower_count": permutation_summary.strictly_lower_count,
        "strictly_lower_fraction": (permutation_summary.strictly_lower_fraction),
        "gate_passed": permutation_summary.gate_passed,
    }
    score_report = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "model_state_byte_sha256": hashlib.sha256(model_state_bytes).hexdigest(),
        "analysis_counts": {
            "random_rankings": counts.random_rankings,
            "bootstrap_resamples": counts.bootstrap_resamples,
            "stress_permutations": counts.stress_permutations,
        },
        "selected_mean_baseline": selected_baseline,
        "test_primary_estimates": {
            "visible_stress_risk_reduction": (visible.relative_risk_reduction),
            "prefix_only_risk_reduction": (prefix.relative_risk_reduction),
            "visible_minus_prefix_increment": increment,
            "issued_center_minus_baseline_iae_pp": test_iae_delta,
        },
        "policy_comparison": _json_records(policy_frame),
        "risk_coverage_curves_secondary": risk_coverage_curves,
        "coverage_metrics": _json_records(coverage_frame),
        "isotonic_calibration_diagnostics": isotonic_diagnostics,
        "structure_diagnostics": structure_diagnostics,
        "test_audit_distribution_shift": distribution_shift,
        "gate_evaluations": _json_records(gate_frame),
        "stress_plan_summary": {
            "pair_count": stress_summary.pair_count,
            "arm_a_exact_tie_count": stress_summary.arm_a_exact_tie_count,
            "arm_b_correct_order_count": (stress_summary.arm_b_correct_order_count),
            "arm_b_correct_order_fraction": (
                stress_summary.arm_b_correct_order_fraction
            ),
            "arm_b_two_sided_95_lower": (stress_summary.arm_b_two_sided_95_lower),
            "arm_b_two_sided_95_upper": (stress_summary.arm_b_two_sided_95_upper),
        },
        "stress_permutation_summary": permutation_summary_payload,
        "status": status,
        "protocol_deviations": [],
    }
    negative_controls = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "available": bootstrap_summary is not None,
        "unavailable_reason": (
            "" if bootstrap_summary is not None else "bootstrap is undefined"
        ),
        "placebo_point_increment": placebo_increment,
        "placebo_bootstrap_two_sided_95_interval": (
            None
            if bootstrap_summary is None
            else [
                bootstrap_summary["placebo_two_sided_95_lower"],
                bootstrap_summary["placebo_two_sided_95_upper"],
            ]
        ),
        "stress_permutation_summary": permutation_summary_payload,
        "gate_evaluations": _json_records(
            gate_frame.loc[
                gate_frame["gate_id"].isin(
                    (
                        "placebo_point_negative_control",
                        "placebo_interval_negative_control",
                        "stress_permutation_negative_control",
                        "intrinsic_output_invariance",
                        "stress_plan_arm_a_invariance",
                    )
                )
            ]
        ),
    }
    family_metrics = _family_metric_table(
        trajectories,
        subset_metrics.reindex(columns=SUBSET_METRIC_COLUMNS),
    )
    matched_pair_scores = _matched_pair_table(
        intrinsic_scores.loc[:, INTRINSIC_PAIR_COLUMNS],
        stress_scores.loc[:, STRESS_PAIR_COLUMNS],
    )
    result_frames = {
        "point_scores.csv": points.loc[:, POINT_SCORE_COLUMNS],
        "trajectory_scores.csv": trajectories.loc[:, TRAJECTORY_SCORE_COLUMNS],
        "family_metrics.csv": family_metrics,
        "matched_pair_scores.csv": matched_pair_scores,
        "bootstrap_replicates.csv": bootstrap.loc[:, BOOTSTRAP_COLUMNS],
        "random_ranking_metrics.csv": random_ranking_metrics.loc[
            :, RANDOM_RANKING_COLUMNS
        ],
        "stress_permutation_metrics.csv": permutations.loc[:, PERMUTATION_COLUMNS],
    }
    canonical = {
        name: canonicalize_score_frame(frame, name)
        for name, frame in result_frames.items()
    }
    canonical_result_summary_bytes(score_report)
    result = V015ScoringResult(
        point_scores=canonical["point_scores.csv"],
        trajectory_scores=canonical["trajectory_scores.csv"],
        family_metrics=canonical["family_metrics.csv"],
        matched_pair_scores=canonical["matched_pair_scores.csv"],
        bootstrap_replicates=canonical["bootstrap_replicates.csv"],
        random_ranking_metrics=canonical["random_ranking_metrics.csv"],
        stress_permutation_metrics=canonical["stress_permutation_metrics.csv"],
        negative_control_metrics=negative_controls,
        score_report=score_report,
        run_manifest=dict(_PROVISIONAL_RUN_MANIFEST),
    )
    return result


def score_committed_artifacts(
    *,
    prediction_frames: Mapping[str, pd.DataFrame],
    truth_frames: Mapping[str, pd.DataFrame],
    model_state_bytes: bytes,
    decoded_model_state: DecodedModelState | None = None,
) -> V015ScoringResult:
    """Run the frozen formal 10k/5k/10k committed-artifact analysis."""

    try:
        return _score_committed_artifacts(
            prediction_frames=prediction_frames,
            truth_frames=truth_frames,
            model_state_bytes=model_state_bytes,
            decoded_model_state=decoded_model_state,
            formal=True,
            counts=_FORMAL_ANALYSIS_COUNTS,
        )
    except V015InconclusiveError as exc:
        return _inconclusive_result(str(exc), model_state_bytes)
    except (
        V015ArtifactError,
        V015AnalysisError,
        V015PipelineError,
        V015TrainingError,
        V015ScoringError,
    ) as exc:
        return _void_result(str(exc), model_state_bytes)


def _run_stochastic_fixture_analyses(
    trajectories: pd.DataFrame,
    *,
    issue_count: int,
    counts: _AnalysisCounts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Private small-fixture hook; it cannot alter formal API counts."""

    random = deterministic_random_rankings(
        trajectories,
        issue_count=issue_count,
        rankings=counts.random_rankings,
    )
    bootstrap = bootstrap_risk_reductions(
        trajectories,
        protocol_id=FROZEN_PROTOCOL_ID,
        issue_count=issue_count,
        resamples=counts.bootstrap_resamples,
        families=tuple(
            family
            for family in TEST_FAMILIES
            if family in set(trajectories["truth_family"])
        ),
    )
    return random, bootstrap


__all__ = [
    "BOOTSTRAP_COLUMNS",
    "COVERAGE_METRIC_COLUMNS",
    "FAMILY_ERROR_COLUMNS",
    "FAMILY_METRIC_COLUMNS",
    "GATE_COLUMNS",
    "INTRINSIC_PAIR_COLUMNS",
    "MATCHED_PAIR_COLUMNS",
    "NEGATIVE_CONTROL_METRICS_KEYS",
    "PERMUTATION_COLUMNS",
    "POINT_SCORE_COLUMNS",
    "POLICY_COMPARISON_COLUMNS",
    "RANDOM_RANKING_COLUMNS",
    "REQUIRED_FORMAL_NON_SCORE_ARTIFACTS",
    "REQUIRED_SCORE_ARTIFACTS",
    "REQUIRED_SCORE_CSV_ARTIFACTS",
    "RESULT_SUMMARY_KEYS",
    "RUN_MANIFEST_KEYS",
    "RUN_PROVENANCE_KEYS",
    "SCORE_FRAME_SCHEMAS",
    "SCORE_REPORT_KEYS",
    "STRESS_PAIR_COLUMNS",
    "SUBSET_METRIC_COLUMNS",
    "TRAJECTORY_SCORE_COLUMNS",
    "ScoreFrameSchema",
    "V015ScoringError",
    "V015ScoringResult",
    "canonical_result_summary_bytes",
    "canonical_score_csv_bytes",
    "canonicalize_score_frame",
    "finalize_run_manifest",
    "required_score_artifact_payloads",
    "score_committed_artifacts",
    "validate_and_recompute_committed_predictions",
]
