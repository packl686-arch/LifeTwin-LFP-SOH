"""Committed-artifact scoring adapter for the frozen V2.1 protocol.

The V2 scientific endpoints, gates, and deterministic summaries remain the
sole scoring specification.  This adapter changes only protocol identity,
stochastic roots, and the V2.1 risk-calibration population.  Formal scoring
requires an IO-issued committed model capability and an envelope bound to
verified V2.1 prediction-commitment evidence; it never accepts a bare V2
decoded model state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments import calendar_long_horizon_v015_scoring as _v015
from lifetwin.experiments.calendar_long_horizon_v015_analysis import (
    AUDIT_ISSUE_COUNT,
    CORE_FAMILIES,
    REQUIRED_GATE_IDS,
    RISK_SCORE_IDS,
    TEST_FAMILIES,
    TEST_ISSUE_COUNT,
    GateEvaluation,
    RiskReduction,
    V015AnalysisError,
    V015InconclusiveError,
    common_pool_gate_evaluations,
    core_test_coverage_summary,
    evaluate_audit_directional_gates,
    evaluate_intrinsic_pairs,
    evaluate_stress_plan_pairs,
    evaluate_test_safety_gates,
    issued_center_minus_baseline_iae,
    placebo_negative_control_gates,
    resolve_result_status,
    score_trajectory_table,
    validate_intrinsic_output_invariance,
    validate_intrinsic_pair_construction,
    validate_stress_plan_arm_a_invariance,
    validate_stress_plan_pair_construction,
)
from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEYS,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    canonical_csv_bytes,
    canonical_json_bytes,
    validate_prediction_artifact_bundle,
    validate_sealed_truth_bundle,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    LabelFreePipelineResult,
    V015PipelineError,
)
from lifetwin.experiments.calendar_long_horizon_v016_analysis import (
    BOOTSTRAP_RESAMPLES,
    RANDOM_RANKING_COUNT,
    STRESS_PERMUTATIONS,
    V021AnalysisError,
    bootstrap_gate_summary,
    bootstrap_risk_reductions,
    deterministic_random_rankings,
    rank_policy,
    stress_permutation_metrics,
    summarize_stress_permutations,
)
from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    V021ContractView,
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_io import (
    V021IOError,
    V021PredictionCommitmentEvidence,
    _require_prediction_commitment_evidence_v021,
)
from lifetwin.experiments.calendar_long_horizon_v016_pipeline import (
    V021PipelineError,
    recompute_label_free_pipeline_v021,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_EXPECTED_SEED_ROOTS,
    V021_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v016_provenance import (
    V021CommittedModelStateEnvelope,
    V021ProvenanceError,
    _require_committed_model_state_envelope,
)


class V021ScoringError(ValueError):
    """Raised when committed V2.1 scoring cannot prove its boundaries."""


ScoreFrameSchema = _v015.ScoreFrameSchema
POINT_SCORE_COLUMNS = _v015.POINT_SCORE_COLUMNS
TRAJECTORY_SCORE_COLUMNS = _v015.TRAJECTORY_SCORE_COLUMNS
POLICY_COMPARISON_COLUMNS = _v015.POLICY_COMPARISON_COLUMNS
COVERAGE_METRIC_COLUMNS = _v015.COVERAGE_METRIC_COLUMNS
FAMILY_ERROR_COLUMNS = _v015.FAMILY_ERROR_COLUMNS
FAMILY_METRIC_COLUMNS = _v015.FAMILY_METRIC_COLUMNS
INTRINSIC_PAIR_COLUMNS = _v015.INTRINSIC_PAIR_COLUMNS
STRESS_PAIR_COLUMNS = _v015.STRESS_PAIR_COLUMNS
MATCHED_PAIR_COLUMNS = _v015.MATCHED_PAIR_COLUMNS
RANDOM_RANKING_COLUMNS = _v015.RANDOM_RANKING_COLUMNS
BOOTSTRAP_COLUMNS = _v015.BOOTSTRAP_COLUMNS
PERMUTATION_COLUMNS = _v015.PERMUTATION_COLUMNS
GATE_COLUMNS = _v015.GATE_COLUMNS
SUBSET_METRIC_COLUMNS = _v015.SUBSET_METRIC_COLUMNS
SCORE_FRAME_SCHEMAS = _v015.SCORE_FRAME_SCHEMAS
REQUIRED_SCORE_ARTIFACTS = _v015.REQUIRED_SCORE_ARTIFACTS
REQUIRED_SCORE_CSV_ARTIFACTS = _v015.REQUIRED_SCORE_CSV_ARTIFACTS
NEGATIVE_CONTROL_METRICS_KEYS = _v015.NEGATIVE_CONTROL_METRICS_KEYS
SCORE_REPORT_KEYS = _v015.SCORE_REPORT_KEYS
RESULT_SUMMARY_KEYS = SCORE_REPORT_KEYS
ARTIFACT_METADATA_KEYS = _v015.ARTIFACT_METADATA_KEYS

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_ATTEMPT_ID = re.compile(r"^v021-[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_PREDICTION_FRAME_FILENAMES = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
    "prediction_bundle.csv",
    "risk_bundle.csv",
    "decision_bundle.csv",
)
_RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "config_sha256",
        "attempt_id",
        "model_state_byte_sha256",
        "model_state_commitment_byte_sha256",
        "prediction_commitment_byte_sha256",
        "analysis_counts",
        "required_score_artifacts",
        "scored_artifacts",
        "protocol_deviations",
    }
)
RUN_MANIFEST_KEYS = _RUN_MANIFEST_KEYS
_PREDICTION_ENVELOPE_KEY = object()
_PREDICTION_ENVELOPE_DOMAIN = b"lifetwin-v021-scoring-prediction-envelope-v1\0"


@dataclass(frozen=True, slots=True)
class _AnalysisCounts:
    random_rankings: int
    bootstrap_resamples: int
    stress_permutations: int


_FORMAL_ANALYSIS_COUNTS = _AnalysisCounts(
    random_rankings=10_000,
    bootstrap_resamples=5_000,
    stress_permutations=10_000,
)
if _FORMAL_ANALYSIS_COUNTS != _AnalysisCounts(
    RANDOM_RANKING_COUNT,
    BOOTSTRAP_RESAMPLES,
    STRESS_PERMUTATIONS,
):
    raise RuntimeError("V2.1 formal analysis cardinalities drifted")
if (
    V021_EXPECTED_SEED_ROOTS["random_rankings"] != 202607260210
    or V021_EXPECTED_SEED_ROOTS["bootstrap"] != 202607260211
    or V021_EXPECTED_SEED_ROOTS["stress_permutations"] != 202607260212
):
    raise RuntimeError("V2.1 formal stochastic roots drifted")


@dataclass(frozen=True, slots=True)
class V021ScoringResult:
    """The ten-artifact V2.1 scoring registry in structured form."""

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
        return self.score_report


class V021PredictionCommitmentEnvelope:
    """Opaque scoring capability bound to verified V2.1 IO evidence."""

    __slots__ = (
        "_artifact_metadata",
        "_artifact_set_sha256",
        "_attempt_id",
        "_config_sha256",
        "_evidence",
        "_issuer_key",
        "_prediction_commitment_byte_sha256",
        "_protocol_id",
        "_provenance_sha256",
    )

    def __init__(
        self,
        *,
        _issuer_key: object,
        protocol_id: str,
        config_sha256: str,
        attempt_id: str,
        evidence: V021PredictionCommitmentEvidence,
        prediction_commitment_byte_sha256: str,
        artifact_set_sha256: str,
        artifact_metadata: tuple[tuple[str, int, int, str], ...],
        provenance_sha256: str,
    ) -> None:
        if (
            _issuer_key is not _PREDICTION_ENVELOPE_KEY
            or type(self) is not V021PredictionCommitmentEnvelope
        ):
            raise TypeError(
                "V021PredictionCommitmentEnvelope is issued only from an exact "
                "prediction commitment receipt"
            )
        object.__setattr__(self, "_issuer_key", _issuer_key)
        object.__setattr__(self, "_protocol_id", protocol_id)
        object.__setattr__(self, "_config_sha256", config_sha256)
        object.__setattr__(self, "_attempt_id", attempt_id)
        object.__setattr__(self, "_evidence", evidence)
        object.__setattr__(
            self,
            "_prediction_commitment_byte_sha256",
            prediction_commitment_byte_sha256,
        )
        object.__setattr__(self, "_artifact_set_sha256", artifact_set_sha256)
        object.__setattr__(self, "_artifact_metadata", artifact_metadata)
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("V2.1 prediction commitment envelopes are immutable")

    @property
    def protocol_id(self) -> str:
        return self._protocol_id

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def prediction_commitment_byte_sha256(self) -> str:
        return self._prediction_commitment_byte_sha256

    @property
    def artifact_set_sha256(self) -> str:
        return self._artifact_set_sha256

    @property
    def artifact_byte_sha256(self) -> dict[str, str]:
        return {filename: digest for filename, _, _, digest in self._artifact_metadata}

    @property
    def artifact_metadata(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "path": filename,
                "row_count": row_count,
                "byte_count": byte_count,
                "byte_sha256": digest,
            }
            for filename, row_count, byte_count, digest in self._artifact_metadata
        )

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256


def _sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V021ScoringError(f"{context} must be a lowercase SHA-256")
    return value


def _component(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _COMPONENT.fullmatch(value) is None:
        raise V021ScoringError(f"{context} must be a canonical component")
    return value


def _contract(view: V021ContractView | None = None) -> V021ContractView:
    contract = load_v021_contract_view() if view is None else view
    if type(contract) is not V021ContractView:
        raise V021ScoringError("The V2.1 contract view has an invalid type")
    if (
        contract.protocol.protocol_id != V021_PROTOCOL_ID
        or contract.artifacts.protocol_id != V021_PROTOCOL_ID
        or contract.protocol.config_sha256 != contract.artifacts.config_byte_sha256
    ):
        raise V021ScoringError("The active artifact contract is not exact V2.1")
    return contract


def _canonical_prediction_metadata(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    contract: FrozenArtifactContract,
    formal: bool,
) -> tuple[tuple[str, int, int, str], ...]:
    validate_prediction_artifact_bundle(
        prediction_frames,
        contract,
        formal=formal,
        expected_variant_keys=FROZEN_VARIANT_KEYS,
    )
    records: list[tuple[str, int, int, str]] = []
    for filename in _PREDICTION_FRAME_FILENAMES:
        raw = canonical_csv_bytes(
            prediction_frames[filename],
            contract.csv_schema(filename),
            contract,
            formal=formal,
        )
        records.append(
            (
                filename,
                len(prediction_frames[filename]),
                len(raw),
                hashlib.sha256(raw).hexdigest(),
            )
        )
    return tuple(records)


def _prediction_metadata_from_evidence(
    evidence: V021PredictionCommitmentEvidence,
) -> tuple[tuple[str, int, int, str], ...]:
    wanted = set(_PREDICTION_FRAME_FILENAMES)
    records = tuple(
        (
            str(entry["path"]),
            int(entry["row_count"]),
            int(entry["byte_count"]),
            str(entry["byte_sha256"]),
        )
        for entry in evidence.file_entries
        if entry["path"] in wanted
    )
    if tuple(name for name, _, _, _ in records) != _PREDICTION_FRAME_FILENAMES:
        raise V021ScoringError(
            "Prediction commitment evidence omits a prediction artifact"
        )
    return records


def _require_evidence_model_binding(
    evidence: V021PredictionCommitmentEvidence,
    model: V021CommittedModelStateEnvelope,
) -> None:
    entries = {
        str(entry["path"]): str(entry["byte_sha256"]) for entry in evidence.file_entries
    }
    training_commitments = (
        model.validated_model_state.training_provenance.commitment_byte_sha256
    )
    actual_analysis_hash = training_commitments.get("actual_analysis_hash_ledger")
    if (
        entries.get("model_state.json")
        != model.validated_model_state.model_state_byte_sha256
        or entries.get("model_state_commitment.json")
        != model.model_state_commitment_artifact_byte_sha256
        or entries.get("actual_analysis_hash_ledger_commitment.json")
        != evidence.actual_analysis_hash_ledger_commitment_byte_sha256
        or actual_analysis_hash
        != evidence.actual_analysis_hash_ledger_commitment_byte_sha256
    ):
        raise V021ScoringError(
            "Prediction commitment evidence differs from the committed model chain"
        )


def _prediction_envelope_digest(
    *,
    model_provenance_sha256: str,
    prediction_commitment_byte_sha256: str,
    artifact_set_sha256: str,
    artifact_metadata: tuple[tuple[str, int, int, str], ...],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_PREDICTION_ENVELOPE_DOMAIN)
    hasher.update(bytes.fromhex(model_provenance_sha256))
    hasher.update(bytes.fromhex(prediction_commitment_byte_sha256))
    hasher.update(bytes.fromhex(artifact_set_sha256))
    for filename, row_count, byte_count, digest in artifact_metadata:
        raw_name = filename.encode("ascii")
        hasher.update(len(raw_name).to_bytes(2, "little"))
        hasher.update(raw_name)
        hasher.update(row_count.to_bytes(8, "little"))
        hasher.update(byte_count.to_bytes(8, "little"))
        hasher.update(bytes.fromhex(digest))
    return hasher.hexdigest()


def _issue_prediction_commitment_envelope_v021(
    *,
    prediction_frames: Mapping[str, pd.DataFrame],
    model_state_envelope: V021CommittedModelStateEnvelope,
    prediction_commitment_evidence: V021PredictionCommitmentEvidence,
    formal: bool,
    contract_view: V021ContractView | None = None,
) -> V021PredictionCommitmentEnvelope:
    """Bind exact IO commitment evidence to the in-memory prediction tables."""

    view = _contract(contract_view)
    try:
        model = _require_committed_model_state_envelope(model_state_envelope)
    except V021ProvenanceError as exc:
        raise V021ScoringError(str(exc)) from exc
    if (
        model.protocol_id != V021_PROTOCOL_ID
        or model.config_sha256 != view.artifacts.config_byte_sha256
    ):
        raise V021ScoringError("Committed model identity differs from V2.1")
    try:
        evidence = _require_prediction_commitment_evidence_v021(
            prediction_commitment_evidence,
            require_ledger_committed=True,
        )
    except V021IOError as exc:
        raise V021ScoringError(str(exc)) from exc
    if evidence.attempt_id != model.attempt_id:
        raise V021ScoringError(
            "Prediction commitment attempt differs from the committed model"
        )
    _require_evidence_model_binding(evidence, model)
    commitment_hash = _sha256(
        evidence.byte_sha256,
        context="prediction commitment byte hash",
    )
    artifact_set_hash = _sha256(
        evidence.artifact_set_sha256,
        context="prediction commitment artifact-set hash",
    )
    metadata = _canonical_prediction_metadata(
        prediction_frames,
        contract=view.artifacts,
        formal=formal,
    )
    if metadata != _prediction_metadata_from_evidence(evidence):
        raise V021ScoringError(
            "Prediction frames differ from the exact committed IO artifacts"
        )
    provenance_hash = _prediction_envelope_digest(
        model_provenance_sha256=model.provenance_sha256,
        prediction_commitment_byte_sha256=commitment_hash,
        artifact_set_sha256=artifact_set_hash,
        artifact_metadata=metadata,
    )
    return V021PredictionCommitmentEnvelope(
        _issuer_key=_PREDICTION_ENVELOPE_KEY,
        protocol_id=V021_PROTOCOL_ID,
        config_sha256=view.artifacts.config_byte_sha256,
        attempt_id=model.attempt_id,
        evidence=evidence,
        prediction_commitment_byte_sha256=commitment_hash,
        artifact_set_sha256=artifact_set_hash,
        artifact_metadata=metadata,
        provenance_sha256=provenance_hash,
    )


def _require_prediction_envelope(
    value: object,
    *,
    model: V021CommittedModelStateEnvelope,
    view: V021ContractView,
) -> V021PredictionCommitmentEnvelope:
    if (
        type(value) is not V021PredictionCommitmentEnvelope
        or value._issuer_key is not _PREDICTION_ENVELOPE_KEY
    ):
        raise V021ScoringError("An exact V021PredictionCommitmentEnvelope is required")
    envelope = value
    if (
        envelope.protocol_id != V021_PROTOCOL_ID
        or envelope.config_sha256 != view.artifacts.config_byte_sha256
        or envelope.attempt_id != model.attempt_id
    ):
        raise V021ScoringError("Prediction commitment provenance identity changed")
    try:
        evidence = _require_prediction_commitment_evidence_v021(
            envelope._evidence,
            require_ledger_committed=True,
        )
    except V021IOError as exc:
        raise V021ScoringError(str(exc)) from exc
    if (
        evidence.attempt_id != envelope.attempt_id
        or evidence.byte_sha256 != envelope.prediction_commitment_byte_sha256
        or evidence.artifact_set_sha256 != envelope.artifact_set_sha256
    ):
        raise V021ScoringError("Prediction commitment IO evidence changed")
    _require_evidence_model_binding(evidence, model)
    commitment_hash = _sha256(
        envelope.prediction_commitment_byte_sha256,
        context="prediction commitment byte hash",
    )
    artifact_set_hash = _sha256(
        envelope.artifact_set_sha256,
        context="prediction commitment artifact-set hash",
    )
    metadata = tuple(envelope._artifact_metadata)
    if tuple(name for name, _, _, _ in metadata) != _PREDICTION_FRAME_FILENAMES:
        raise V021ScoringError("Prediction commitment artifact registry changed")
    for name, row_count, byte_count, digest in metadata:
        _component(name, context="prediction artifact name")
        if (
            type(row_count) is not int
            or row_count < 1
            or type(byte_count) is not int
            or byte_count < 1
        ):
            raise V021ScoringError(
                f"Prediction commitment metadata is invalid for {name}"
            )
        _sha256(digest, context=f"prediction artifact hash/{name}")
    if metadata != _prediction_metadata_from_evidence(evidence):
        raise V021ScoringError(
            "Prediction commitment envelope differs from IO evidence"
        )
    expected = _prediction_envelope_digest(
        model_provenance_sha256=model.provenance_sha256,
        prediction_commitment_byte_sha256=commitment_hash,
        artifact_set_sha256=artifact_set_hash,
        artifact_metadata=metadata,
    )
    if envelope.provenance_sha256 != expected:
        raise V021ScoringError("Prediction commitment envelope digest changed")
    return envelope


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
        committed,
        schema,
        contract,
        formal=formal,
    ) != canonical_csv_bytes(
        recomputed,
        schema,
        contract,
        formal=formal,
    ):
        raise V021ScoringError(
            f"{filename} differs byte-for-byte from V2.1 recomputation"
        )


def validate_and_recompute_committed_predictions_v021(
    *,
    prediction_frames: Mapping[str, pd.DataFrame],
    model_state_envelope: V021CommittedModelStateEnvelope,
    prediction_commitment_envelope: V021PredictionCommitmentEnvelope,
    formal: bool,
    contract_view: V021ContractView | None = None,
) -> tuple[Any, LabelFreePipelineResult]:
    """Validate prediction commitment and independently recompute every output."""

    view = _contract(contract_view)
    try:
        model = _require_committed_model_state_envelope(model_state_envelope)
    except V021ProvenanceError as exc:
        raise V021ScoringError(str(exc)) from exc
    receipt = _require_prediction_envelope(
        prediction_commitment_envelope,
        model=model,
        view=view,
    )
    observed_metadata = _canonical_prediction_metadata(
        prediction_frames,
        contract=view.artifacts,
        formal=formal,
    )
    if observed_metadata != receipt._artifact_metadata:
        raise V021ScoringError(
            "Prediction frames differ from their commitment envelope"
        )
    recomputed = recompute_label_free_pipeline_v021(
        prefix_pack=prediction_frames["prefix_pack.csv"],
        forecast_coordinates=prediction_frames["forecast_coordinates.csv"],
        operating_pack=prediction_frames["operating_pack.csv"],
        member_fit_diagnostics=prediction_frames["member_fit_diagnostics.csv"],
        member_forecast_bundle=prediction_frames["member_forecast_bundle.csv"],
        model_state_envelope=model,
        contract=view.artifacts,
        formal=formal,
    )
    for filename, committed, expected in (
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
            committed,
            expected,
            filename=filename,
            contract=view.artifacts,
            formal=formal,
        )
    return model.validated_model_state.decoded_model_state, recomputed


def _policy_comparison_v021(
    frame: pd.DataFrame,
    random_rankings: pd.DataFrame,
    *,
    issue_count: int,
    partition: str,
) -> tuple[pd.DataFrame, dict[str, RiskReduction]]:
    if len(random_rankings) < 1:
        raise V021ScoringError("Random-ranking table is empty")
    random_mean = float(
        pd.to_numeric(
            random_rankings["issued_catastrophic_rate"],
            errors="coerce",
        ).mean()
    )
    if not math.isfinite(random_mean) or random_mean <= 0.0:
        raise V021ScoringError("Random-ranking mean risk is unavailable")
    eligible = frame["hard_eligible_visible_stress"].astype(bool)
    catastrophic = frame["catastrophic"].astype(bool)
    rows: list[dict[str, object]] = []
    reductions: dict[str, RiskReduction] = {}
    for score_id in RISK_SCORE_IDS:
        issued = rank_policy(
            frame,
            protocol_id=V021_PROTOCOL_ID,
            arm=score_id,
            score_column=f"risk_{score_id}",
            predictor_hash_column=f"risk_hash_{score_id}",
            issue_count=issue_count,
        )
        issued_count = int(issued.sum())
        issued_rate = float(catastrophic[issued].mean())
        reduction = RiskReduction(
            issued_count=issued_count,
            issued_catastrophic_rate=issued_rate,
            random_expected_catastrophic_rate=random_mean,
            relative_risk_reduction=1.0 - issued_rate / random_mean,
        )
        reductions[score_id] = reduction
        rows.append(
            {
                "partition": partition,
                "score_id": score_id,
                "source_count": len(frame),
                "eligible_count": int(eligible.sum()),
                "issued_count": issued_count,
                "source_coverage": issued_count / len(frame),
                "eligible_coverage": issued_count / int(eligible.sum()),
                "issued_catastrophic_rate": issued_rate,
                "mean_random_issued_catastrophic_rate": random_mean,
                "relative_risk_reduction": reduction.relative_risk_reduction,
            }
        )
    return pd.DataFrame(rows), reductions


def _risk_coverage_curves_v021(test: pd.DataFrame) -> list[dict[str, Any]]:
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
                    protocol_id=V021_PROTOCOL_ID,
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


def _isotonic_calibration_diagnostics_v021(
    trajectories: pd.DataFrame,
    risk_bundle: pd.DataFrame,
    decoded: Any,
    *,
    source_calibration_count: int,
    risk_isotonic_eligible_count: int,
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
        supplied = joined["calibrated_catastrophic_probability"].notna().to_numpy()
        eligible = joined.loc[supplied]
        probabilities = pd.to_numeric(
            eligible["calibrated_catastrophic_probability"],
            errors="coerce",
        ).to_numpy(float)
        labels = eligible["catastrophic"].astype(bool).to_numpy(float)
        reason = ""
        if len(joined) != source_calibration_count:
            reason = (
                f"calibration_count={len(joined)} expected={source_calibration_count}"
            )
        elif len(eligible) != risk_isotonic_eligible_count:
            reason = (
                f"risk_isotonic_eligible_count={len(eligible)} "
                f"expected={risk_isotonic_eligible_count}"
            )
        elif not np.isfinite(probabilities).all():
            reason = "calibrated probabilities are nonfinite"
        elif np.any((probabilities < 0.0) | (probabilities > 1.0)):
            reason = "calibrated probabilities are outside [0,1]"
        records.append(
            {
                "score_id": score_id,
                "n": len(eligible),
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


def _bootstrap_summary_v021(
    replicates: pd.DataFrame,
    *,
    issue_count: int,
    expected_count: int,
) -> Mapping[str, float] | None:
    if len(replicates) != expected_count:
        raise V021ScoringError("Bootstrap replicate count changed")
    defined = replicates["defined"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    if not defined.all():
        return None
    return bootstrap_gate_summary(
        replicates,
        issue_count=issue_count,
        resamples=expected_count,
    )


def _empty_frame(filename: str) -> pd.DataFrame:
    return pd.DataFrame(columns=SCORE_FRAME_SCHEMAS[filename].columns)


def canonicalize_score_frame_v021(
    frame: pd.DataFrame,
    filename: str,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    try:
        result = _v015.canonicalize_score_frame(
            frame,
            filename,
            allow_empty=allow_empty,
        )
    except _v015.V015ScoringError as exc:
        raise V021ScoringError(str(exc)) from exc
    if "protocol_id" in result.columns and not result.empty:
        if set(result["protocol_id"].astype(str)) != {V021_PROTOCOL_ID}:
            raise V021ScoringError(f"{filename} protocol_id changed")
    return result


def canonical_score_csv_bytes_v021(
    frame: pd.DataFrame,
    filename: str,
    *,
    allow_empty: bool = False,
) -> bytes:
    ordered = canonicalize_score_frame_v021(
        frame,
        filename,
        allow_empty=allow_empty,
    )
    try:
        return _v015.canonical_score_csv_bytes(
            ordered,
            filename,
            allow_empty=allow_empty,
        )
    except _v015.V015ScoringError as exc:
        raise V021ScoringError(str(exc)) from exc


def canonical_result_summary_bytes_v021(payload: Mapping[str, Any]) -> bytes:
    if set(payload) != SCORE_REPORT_KEYS:
        raise V021ScoringError("score_report.json keys changed")
    if payload.get("protocol_id") != V021_PROTOCOL_ID:
        raise V021ScoringError("score_report.json protocol_id changed")
    return canonical_json_bytes(payload)


def _json_artifact_bytes(
    payload: Mapping[str, Any],
    *,
    filename: str,
    expected_keys: frozenset[str],
) -> bytes:
    if set(payload) != expected_keys:
        raise V021ScoringError(f"{filename} keys changed")
    if payload.get("protocol_id") != V021_PROTOCOL_ID:
        raise V021ScoringError(f"{filename} protocol_id changed")
    return canonical_json_bytes(payload)


def _artifacts_without_manifest(
    result: V021ScoringResult,
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


def _payloads_without_manifest(result: V021ScoringResult) -> dict[str, bytes]:
    status = result.score_report.get("status")
    allow_empty = isinstance(status, Mapping) and status.get("status") in {
        "void",
        "inconclusive_not_success",
    }
    payloads: dict[str, bytes] = {}
    for filename, value in _artifacts_without_manifest(result).items():
        if filename.endswith(".csv"):
            if not isinstance(value, pd.DataFrame):
                raise V021ScoringError(f"{filename} must be a dataframe")
            payloads[filename] = canonical_score_csv_bytes_v021(
                value,
                filename,
                allow_empty=allow_empty,
            )
        elif filename == "negative_control_metrics.json":
            payloads[filename] = _json_artifact_bytes(
                value,
                filename=filename,
                expected_keys=NEGATIVE_CONTROL_METRICS_KEYS,
            )
        else:
            payloads[filename] = canonical_result_summary_bytes_v021(value)
    if tuple(payloads) != REQUIRED_SCORE_ARTIFACTS[:-1]:
        raise V021ScoringError("V2.1 score artifact registry changed")
    return payloads


def _finalize_registry(
    result: V021ScoringResult,
    *,
    model: V021CommittedModelStateEnvelope,
    prediction: V021PredictionCommitmentEnvelope,
    view: V021ContractView,
) -> V021ScoringResult:
    payloads = _payloads_without_manifest(result)
    artifacts = _artifacts_without_manifest(result)
    records = [
        {
            "path": filename,
            "row_count": (
                len(artifacts[filename]) if filename.endswith(".csv") else None
            ),
            "byte_count": len(raw),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
        }
        for filename, raw in payloads.items()
    ]
    manifest = {
        "schema_version": "lifetwin.v021.scored_registry.v1",
        "protocol_id": V021_PROTOCOL_ID,
        "config_sha256": view.artifacts.config_byte_sha256,
        "attempt_id": model.attempt_id,
        "model_state_byte_sha256": model.validated_model_state.model_state_byte_sha256,
        "model_state_commitment_byte_sha256": (
            model.model_state_commitment_artifact_byte_sha256
        ),
        "prediction_commitment_byte_sha256": (
            prediction.prediction_commitment_byte_sha256
        ),
        "analysis_counts": dict(result.score_report["analysis_counts"]),
        "required_score_artifacts": list(REQUIRED_SCORE_ARTIFACTS),
        "scored_artifacts": records,
        "protocol_deviations": [],
    }
    if set(manifest) != _RUN_MANIFEST_KEYS:
        raise V021ScoringError("run_manifest.json keys changed")
    return replace(result, run_manifest=manifest)


def required_score_artifact_payloads_v021(
    result: V021ScoringResult,
) -> dict[str, bytes]:
    """Return all ten committed V2.1 score artifacts as canonical bytes."""

    if type(result) is not V021ScoringResult:
        raise V021ScoringError("An exact V021ScoringResult is required")
    manifest = result.run_manifest
    if not isinstance(manifest, Mapping) or set(manifest) != _RUN_MANIFEST_KEYS:
        raise V021ScoringError("run_manifest.json is not finalized")
    attempt_id = manifest.get("attempt_id")
    expected_counts = {
        "random_rankings": _FORMAL_ANALYSIS_COUNTS.random_rankings,
        "bootstrap_resamples": _FORMAL_ANALYSIS_COUNTS.bootstrap_resamples,
        "stress_permutations": _FORMAL_ANALYSIS_COUNTS.stress_permutations,
    }
    if (
        manifest.get("schema_version") != "lifetwin.v021.scored_registry.v1"
        or manifest.get("protocol_id") != V021_PROTOCOL_ID
        or manifest.get("config_sha256")
        != load_v021_contract_view().artifacts.config_byte_sha256
        or not isinstance(attempt_id, str)
        or _ATTEMPT_ID.fullmatch(attempt_id) is None
        or _sha256(
            manifest.get("model_state_byte_sha256"),
            context="run manifest model-state hash",
        )
        != result.score_report.get("model_state_byte_sha256")
        or _sha256(
            manifest.get("model_state_commitment_byte_sha256"),
            context="run manifest model-state commitment hash",
        )
        != manifest.get("model_state_commitment_byte_sha256")
        or _sha256(
            manifest.get("prediction_commitment_byte_sha256"),
            context="run manifest prediction commitment hash",
        )
        != manifest.get("prediction_commitment_byte_sha256")
        or manifest.get("analysis_counts") != expected_counts
        or manifest.get("required_score_artifacts") != list(REQUIRED_SCORE_ARTIFACTS)
        or manifest.get("protocol_deviations") != []
    ):
        raise V021ScoringError("run_manifest.json identity or registry changed")
    payloads = _payloads_without_manifest(result)
    artifacts = _artifacts_without_manifest(result)
    expected_records = [
        {
            "path": filename,
            "row_count": (
                len(artifacts[filename]) if filename.endswith(".csv") else None
            ),
            "byte_count": len(raw),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
        }
        for filename, raw in payloads.items()
    ]
    if manifest.get("scored_artifacts") != expected_records:
        raise V021ScoringError(
            "run_manifest.json does not bind the canonical score artifacts"
        )
    payloads["run_manifest.json"] = _json_artifact_bytes(
        manifest,
        filename="run_manifest.json",
        expected_keys=_RUN_MANIFEST_KEYS,
    )
    if tuple(payloads) != REQUIRED_SCORE_ARTIFACTS:
        raise V021ScoringError("The complete V2.1 score registry changed")
    return payloads


def _terminal_result(
    reason: str,
    *,
    status_kind: str,
    model: V021CommittedModelStateEnvelope,
    prediction: V021PredictionCommitmentEnvelope,
    view: V021ContractView,
) -> V021ScoringResult:
    if status_kind not in {"void", "inconclusive_not_success"}:
        raise V021ScoringError("Unknown terminal score status")
    unavailable = (
        "protocol void" if status_kind == "void" else f"protocol inconclusive: {reason}"
    )
    gates = tuple(
        _v015._inconclusive(gate_id, "not evaluated", unavailable)
        for gate_id in REQUIRED_GATE_IDS
    )
    status = (
        resolve_result_status(gates, void_reasons=(reason,))
        if status_kind == "void"
        else resolve_result_status(
            gates,
            external_inconclusive_reasons=(reason,),
        )
    )
    gate_frame = _v015._gate_frame(gates)
    score_report = {
        "protocol_id": V021_PROTOCOL_ID,
        "model_state_byte_sha256": model.validated_model_state.model_state_byte_sha256,
        "analysis_counts": {
            "random_rankings": 10_000,
            "bootstrap_resamples": 5_000,
            "stress_permutations": 10_000,
        },
        "selected_mean_baseline": None,
        "test_primary_estimates": None,
        "policy_comparison": [],
        "risk_coverage_curves_secondary": (
            _v015._unavailable_risk_coverage_curves(unavailable)
        ),
        "coverage_metrics": [],
        "isotonic_calibration_diagnostics": (
            _v015._unavailable_isotonic_diagnostics(unavailable)
        ),
        "structure_diagnostics": _v015._unavailable_structure_diagnostics(unavailable),
        "test_audit_distribution_shift": _v015._unavailable_distribution_shift(
            unavailable
        ),
        "gate_evaluations": _v015._json_records(gate_frame),
        "stress_plan_summary": None,
        "stress_permutation_summary": None,
        "status": status,
        "protocol_deviations": [],
    }
    negative_controls = {
        "protocol_id": V021_PROTOCOL_ID,
        "available": False,
        "unavailable_reason": unavailable,
        "placebo_point_increment": None,
        "placebo_bootstrap_two_sided_95_interval": None,
        "stress_permutation_summary": None,
        "gate_evaluations": [],
    }
    result = V021ScoringResult(
        point_scores=_empty_frame("point_scores.csv"),
        trajectory_scores=_empty_frame("trajectory_scores.csv"),
        family_metrics=_empty_frame("family_metrics.csv"),
        matched_pair_scores=_empty_frame("matched_pair_scores.csv"),
        bootstrap_replicates=_empty_frame("bootstrap_replicates.csv"),
        random_ranking_metrics=_empty_frame("random_ranking_metrics.csv"),
        stress_permutation_metrics=_empty_frame("stress_permutation_metrics.csv"),
        negative_control_metrics=negative_controls,
        score_report=score_report,
        run_manifest={},
    )
    return _finalize_registry(
        result,
        model=model,
        prediction=prediction,
        view=view,
    )


def _score_committed_artifacts_v021(
    *,
    prediction_frames: Mapping[str, pd.DataFrame],
    truth_frames: Mapping[str, pd.DataFrame],
    model_state_envelope: V021CommittedModelStateEnvelope,
    prediction_commitment_envelope: V021PredictionCommitmentEnvelope,
    formal: bool,
    counts: _AnalysisCounts,
    contract_view: V021ContractView | None = None,
) -> V021ScoringResult:
    view = _contract(contract_view)
    validate_sealed_truth_bundle(truth_frames, view.artifacts, formal=formal)
    decoded, recomputed = validate_and_recompute_committed_predictions_v021(
        prediction_frames=prediction_frames,
        model_state_envelope=model_state_envelope,
        prediction_commitment_envelope=prediction_commitment_envelope,
        formal=formal,
        contract_view=view,
    )
    truth = _v015._combine_truth(truth_frames)
    points, trajectories = score_trajectory_table(
        prediction_frames["prediction_bundle.csv"],
        truth,
        prediction_frames["risk_bundle.csv"],
        prediction_frames["decision_bundle.csv"],
    )
    trajectories = _v015._add_persistence_iae(
        points,
        trajectories,
        prediction_frames["prefix_pack.csv"],
    )

    intrinsic_mapping = truth_frames["intrinsic_matched_pairs.csv"]
    stress_mapping = truth_frames["stress_plan_matched_pairs.csv"]
    validate_intrinsic_pair_construction(
        prediction_frames["prefix_pack.csv"],
        prediction_frames["operating_pack.csv"],
        truth_frames["intrinsic_matched_truth.csv"],
        intrinsic_mapping,
        view.protocol,
    )
    validate_stress_plan_pair_construction(
        prediction_frames["prefix_pack.csv"],
        prediction_frames["operating_pack.csv"],
        truth_frames["stress_plan_matched_truth.csv"],
        stress_mapping,
        view.protocol,
    )
    validate_intrinsic_output_invariance(points, trajectories, intrinsic_mapping)
    validate_stress_plan_arm_a_invariance(points, trajectories, stress_mapping)
    common_pool_gates = common_pool_gate_evaluations(trajectories)

    test = trajectories.loc[trajectories["partition"].eq("test")].copy()
    random_rankings = deterministic_random_rankings(
        test,
        issue_count=TEST_ISSUE_COUNT,
        protocol_id=V021_PROTOCOL_ID,
        rankings=counts.random_rankings,
    )
    random_rankings.insert(0, "partition", "test")
    policy_test, test_reductions = _policy_comparison_v021(
        test,
        random_rankings,
        issue_count=TEST_ISSUE_COUNT,
        partition="test",
    )
    audit = trajectories.loc[trajectories["partition"].eq("audit")].copy()
    audit_random = deterministic_random_rankings(
        audit,
        issue_count=AUDIT_ISSUE_COUNT,
        protocol_id=V021_PROTOCOL_ID,
        rankings=counts.random_rankings,
    )
    audit_random.insert(0, "partition", "audit")
    policy_audit, _ = _policy_comparison_v021(
        audit,
        audit_random,
        issue_count=AUDIT_ISSUE_COUNT,
        partition="audit",
    )
    policy = pd.concat((policy_test, policy_audit), ignore_index=True)
    random_ranking_metrics = pd.concat(
        (random_rankings, audit_random),
        ignore_index=True,
    )

    bootstrap = bootstrap_risk_reductions(
        test,
        protocol_id=V021_PROTOCOL_ID,
        issue_count=TEST_ISSUE_COUNT,
        resamples=counts.bootstrap_resamples,
    )
    bootstrap_summary = _bootstrap_summary_v021(
        bootstrap,
        issue_count=TEST_ISSUE_COUNT,
        expected_count=counts.bootstrap_resamples,
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
        base_core_coverage = core_test_coverage_summary(
            trajectories,
            calibrated=False,
        )
    intrinsic_scores, intrinsic_coverage = evaluate_intrinsic_pairs(
        trajectories,
        intrinsic_mapping,
    )
    stress_scores, stress_summary = evaluate_stress_plan_pairs(
        trajectories,
        stress_mapping,
    )

    selected_baseline = decoded.training_state.calibration.selected_mean_baseline
    baseline_columns = {
        "target_prefix_persistence": "persistence_trajectory_iae_pp",
        "target_prefix_sqrt_time": "sqrt_trajectory_iae_pp",
        "target_prefix_bounded_power_law": "bounded_power_trajectory_iae_pp",
    }
    try:
        baseline_column = baseline_columns[selected_baseline]
    except KeyError as exc:
        raise V021ScoringError(
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
        trajectories,
        protocol_id=V021_PROTOCOL_ID,
    )
    audit_subset, audit_gates = evaluate_audit_directional_gates(
        trajectories,
        protocol_id=V021_PROTOCOL_ID,
        issued_center_minus_baseline_iae_pp=audit_iae_delta,
    )
    subset_metrics = pd.concat((test_subset, audit_subset), ignore_index=True)

    visible = test_reductions["visible_stress"]
    prefix = test_reductions["prefix_only"]
    placebo = test_reductions["placebo_8"]
    increment = visible.relative_risk_reduction - prefix.relative_risk_reduction
    placebo_increment = placebo.relative_risk_reduction - prefix.relative_risk_reduction
    primary_gates = _v015._primary_gates(
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
                state="pass" if abs(placebo_increment) < 0.05 else "fail",
                estimate=placebo_increment,
                threshold="abs(increment) < 0.05",
            ),
            _v015._inconclusive(
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
        protocol_id=V021_PROTOCOL_ID,
        random_expected_catastrophic_rate=(visible.random_expected_catastrophic_rate),
        observed_prefix_only_risk_reduction=prefix.relative_risk_reduction,
        issue_count=TEST_ISSUE_COUNT,
        permutations=counts.stress_permutations,
    )
    permutation_summary = summarize_stress_permutations(
        permutations,
        observed_visible_minus_prefix_increment=increment,
        issue_count=TEST_ISSUE_COUNT,
        permutations=counts.stress_permutations,
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
            _v015._coverage_record(
                "core_test_calibrated",
                core_coverage,
                reason="one or more core intervals are unavailable",
            ),
            _v015._coverage_record(
                "core_test_structural",
                base_core_coverage,
                reason="one or more core structural bands are unavailable",
            ),
            _v015._coverage_record(
                "intrinsic_pairs_calibrated",
                intrinsic_coverage,
                reason="one or more intrinsic-pair intervals are unavailable",
            ),
        )
    )
    gate_frame = _v015._gate_frame(gates).loc[:, GATE_COLUMNS]
    policy_frame = policy.loc[:, POLICY_COMPARISON_COLUMNS]
    coverage_frame = coverage.loc[:, COVERAGE_METRIC_COLUMNS]
    risk_coverage_curves = _risk_coverage_curves_v021(test)
    calibration_audit = (
        model_state_envelope.validated_model_state.training_provenance.calibration_audit
    )
    isotonic_diagnostics = _isotonic_calibration_diagnostics_v021(
        trajectories,
        prediction_frames["risk_bundle.csv"],
        decoded,
        source_calibration_count=calibration_audit.source_calibration_count,
        risk_isotonic_eligible_count=(calibration_audit.risk_isotonic_eligible_count),
    )
    structure_diagnostics = _v015._structure_diagnostics(
        recomputed.feature_bundle,
        prediction_frames["member_fit_diagnostics.csv"],
    )
    distribution_shift = _v015._test_audit_distribution_shift(
        prediction_frames["operating_pack.csv"],
        trajectories,
    )
    permutation_summary_payload = {
        "permutation_count": permutation_summary.permutation_count,
        "observed_visible_minus_prefix_increment": (
            permutation_summary.observed_visible_minus_prefix_increment
        ),
        "strictly_lower_count": permutation_summary.strictly_lower_count,
        "strictly_lower_fraction": permutation_summary.strictly_lower_fraction,
        "gate_passed": permutation_summary.gate_passed,
    }
    model_hash = model_state_envelope.validated_model_state.model_state_byte_sha256
    score_report = {
        "protocol_id": V021_PROTOCOL_ID,
        "model_state_byte_sha256": model_hash,
        "analysis_counts": {
            "random_rankings": counts.random_rankings,
            "bootstrap_resamples": counts.bootstrap_resamples,
            "stress_permutations": counts.stress_permutations,
        },
        "selected_mean_baseline": selected_baseline,
        "test_primary_estimates": {
            "visible_stress_risk_reduction": visible.relative_risk_reduction,
            "prefix_only_risk_reduction": prefix.relative_risk_reduction,
            "visible_minus_prefix_increment": increment,
            "issued_center_minus_baseline_iae_pp": test_iae_delta,
        },
        "policy_comparison": _v015._json_records(policy_frame),
        "risk_coverage_curves_secondary": risk_coverage_curves,
        "coverage_metrics": _v015._json_records(coverage_frame),
        "isotonic_calibration_diagnostics": isotonic_diagnostics,
        "structure_diagnostics": structure_diagnostics,
        "test_audit_distribution_shift": distribution_shift,
        "gate_evaluations": _v015._json_records(gate_frame),
        "stress_plan_summary": {
            "pair_count": stress_summary.pair_count,
            "arm_a_exact_tie_count": stress_summary.arm_a_exact_tie_count,
            "arm_b_correct_order_count": stress_summary.arm_b_correct_order_count,
            "arm_b_correct_order_fraction": (
                stress_summary.arm_b_correct_order_fraction
            ),
            "arm_b_two_sided_95_lower": stress_summary.arm_b_two_sided_95_lower,
            "arm_b_two_sided_95_upper": stress_summary.arm_b_two_sided_95_upper,
        },
        "stress_permutation_summary": permutation_summary_payload,
        "status": status,
        "protocol_deviations": [],
    }
    negative_controls = {
        "protocol_id": V021_PROTOCOL_ID,
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
        "gate_evaluations": _v015._json_records(
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
    family_metrics = _v015._family_metric_table(
        trajectories,
        subset_metrics.reindex(columns=SUBSET_METRIC_COLUMNS),
    )
    matched_pair_scores = _v015._matched_pair_table(
        intrinsic_scores.loc[:, INTRINSIC_PAIR_COLUMNS],
        stress_scores.loc[:, STRESS_PAIR_COLUMNS],
    )
    frames = {
        "point_scores.csv": points.loc[:, POINT_SCORE_COLUMNS],
        "trajectory_scores.csv": trajectories.loc[:, TRAJECTORY_SCORE_COLUMNS],
        "family_metrics.csv": family_metrics,
        "matched_pair_scores.csv": matched_pair_scores,
        "bootstrap_replicates.csv": bootstrap.loc[:, BOOTSTRAP_COLUMNS],
        "random_ranking_metrics.csv": random_ranking_metrics.loc[
            :,
            RANDOM_RANKING_COLUMNS,
        ],
        "stress_permutation_metrics.csv": permutations.loc[:, PERMUTATION_COLUMNS],
    }
    canonical = {
        name: canonicalize_score_frame_v021(frame, name)
        for name, frame in frames.items()
    }
    canonical_result_summary_bytes_v021(score_report)
    result = V021ScoringResult(
        point_scores=canonical["point_scores.csv"],
        trajectory_scores=canonical["trajectory_scores.csv"],
        family_metrics=canonical["family_metrics.csv"],
        matched_pair_scores=canonical["matched_pair_scores.csv"],
        bootstrap_replicates=canonical["bootstrap_replicates.csv"],
        random_ranking_metrics=canonical["random_ranking_metrics.csv"],
        stress_permutation_metrics=canonical["stress_permutation_metrics.csv"],
        negative_control_metrics=negative_controls,
        score_report=score_report,
        run_manifest={},
    )
    model = _require_committed_model_state_envelope(model_state_envelope)
    prediction = _require_prediction_envelope(
        prediction_commitment_envelope,
        model=model,
        view=view,
    )
    return _finalize_registry(
        result,
        model=model,
        prediction=prediction,
        view=view,
    )


def score_committed_artifacts(
    *,
    prediction_frames: Mapping[str, pd.DataFrame],
    truth_frames: Mapping[str, pd.DataFrame],
    model_state_envelope: V021CommittedModelStateEnvelope,
    prediction_commitment_envelope: V021PredictionCommitmentEnvelope,
) -> V021ScoringResult:
    """Run the formal V2.1 10k/5k/10k committed-artifact scorer."""

    view = _contract()
    try:
        model = _require_committed_model_state_envelope(model_state_envelope)
    except V021ProvenanceError as exc:
        raise V021ScoringError(str(exc)) from exc
    prediction = _require_prediction_envelope(
        prediction_commitment_envelope,
        model=model,
        view=view,
    )
    try:
        return _score_committed_artifacts_v021(
            prediction_frames=prediction_frames,
            truth_frames=truth_frames,
            model_state_envelope=model,
            prediction_commitment_envelope=prediction,
            formal=True,
            counts=_FORMAL_ANALYSIS_COUNTS,
            contract_view=view,
        )
    except V015InconclusiveError as exc:
        return _terminal_result(
            str(exc),
            status_kind="inconclusive_not_success",
            model=model,
            prediction=prediction,
            view=view,
        )
    except (
        V015ArtifactError,
        V015AnalysisError,
        V015PipelineError,
        V021AnalysisError,
        V021PipelineError,
        V021ScoringError,
        _v015.V015ScoringError,
    ) as exc:
        return _terminal_result(
            str(exc),
            status_kind="void",
            model=model,
            prediction=prediction,
            view=view,
        )


score_committed_artifacts_v021 = score_committed_artifacts
required_score_artifact_payloads = required_score_artifact_payloads_v021
canonicalize_score_frame = canonicalize_score_frame_v021
canonical_score_csv_bytes = canonical_score_csv_bytes_v021
canonical_result_summary_bytes = canonical_result_summary_bytes_v021


def _run_stochastic_fixture_analyses_v021(
    trajectories: pd.DataFrame,
    *,
    issue_count: int,
    counts: _AnalysisCounts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Private small-fixture hook that cannot alter the formal scorer."""

    random = deterministic_random_rankings(
        trajectories,
        issue_count=issue_count,
        protocol_id=V021_PROTOCOL_ID,
        rankings=counts.random_rankings,
    )
    families = tuple(
        family
        for family in TEST_FAMILIES
        if family in set(trajectories["truth_family"])
    )
    bootstrap = bootstrap_risk_reductions(
        trajectories,
        protocol_id=V021_PROTOCOL_ID,
        issue_count=issue_count,
        resamples=counts.bootstrap_resamples,
        families=families,
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
    "REQUIRED_SCORE_ARTIFACTS",
    "REQUIRED_SCORE_CSV_ARTIFACTS",
    "RESULT_SUMMARY_KEYS",
    "RUN_MANIFEST_KEYS",
    "SCORE_FRAME_SCHEMAS",
    "SCORE_REPORT_KEYS",
    "STRESS_PAIR_COLUMNS",
    "SUBSET_METRIC_COLUMNS",
    "TRAJECTORY_SCORE_COLUMNS",
    "ScoreFrameSchema",
    "V021PredictionCommitmentEnvelope",
    "V021ScoringError",
    "V021ScoringResult",
    "canonical_result_summary_bytes",
    "canonical_result_summary_bytes_v021",
    "canonical_score_csv_bytes",
    "canonical_score_csv_bytes_v021",
    "canonicalize_score_frame",
    "canonicalize_score_frame_v021",
    "required_score_artifact_payloads",
    "required_score_artifact_payloads_v021",
    "score_committed_artifacts",
    "score_committed_artifacts_v021",
    "validate_and_recompute_committed_predictions_v021",
]
