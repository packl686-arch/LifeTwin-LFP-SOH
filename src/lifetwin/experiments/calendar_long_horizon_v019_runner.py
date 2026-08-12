"""One-shot, phase-isolated orchestration for the frozen V2.4 experiment.

The runner owns process boundaries, exact artifact writes, exposure-ledger
transitions, and the development-only training sequence.  The prediction
subprocess receives only the label-free root and attempt identity.  It never
receives a sealed-truth root, score root, seed root, family mapping, or
collision-audit record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    canonical_json_bytes,
    read_canonical_csv,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    DECLARED_STRUCTURE_FAMILIES,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
    LabelFreePipelineResult,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    stress_index,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import (
    CalibrationDevelopmentState,
    CenterDevelopmentState,
    FrozenTrainingState,
    RiskDevelopmentState,
    center_state_sha256,
    default_software_versions,
    fit_center_development_state,
    fit_risk_development_state,
    make_probe_state,
    risk_state_sha256,
)
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractView,
    resolve_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v019_environment import (
    FormalEnvironmentIdentity,
    verify_formal_environment,
)
from lifetwin.experiments.calendar_long_horizon_v019_firewall import (
    FormalAttemptIdentity,
    append_formal_exposure_event,
    append_phase_error_without_masking,
    open_truth_for_phase,
    phase_commitment_message,
    validate_formal_exposure_log,
    verify_phase_artifact_commitment,
)
from lifetwin.experiments.calendar_long_horizon_v019_generation import (
    commit_frozen_v024_generation_plan,
    generate_frozen_v024_artifacts,
)
from lifetwin.experiments.calendar_long_horizon_v019_io import (
    V024CommittedLabelFreeBundle,
    create_prediction_commitment_v024,
    load_committed_label_free_bundle_v024,
    load_fresh_generation_bundle_v024,
    verify_prediction_commitment_v024,
)
from lifetwin.experiments.calendar_long_horizon_v019_pipeline import (
    recompute_validated_partition_with_state_v024,
)
from lifetwin.experiments.calendar_long_horizon_v019_partition import (
    WholeBundleValidated,
    consume_partition_frames,
    derive_partition_view,
    validate_whole_bundle_from_root,
)
from lifetwin.experiments.calendar_long_horizon_v019_prediction import (
    commit_validated_fit_result_v024,
    fit_verified_generation_bundle_v024,
    write_verified_fit_result_v024,
)
from lifetwin.experiments.calendar_long_horizon_v019_provenance import (
    V024CommittedModelStateEnvelope,
    _issue_v024_training_provenance_from_fresh_bytes,
)
from lifetwin.experiments.calendar_long_horizon_v019_scoring import (
    REQUIRED_SCORE_ARTIFACTS,
    _issue_prediction_commitment_envelope_v024,
    required_score_artifact_payloads_v024,
    score_committed_artifacts,
)
from lifetwin.experiments.calendar_long_horizon_v019_state import (
    deserialize_calibration_manifest_json_v024,
    deserialize_calibration_population_audit_json_v024,
    deserialize_model_state_json_v024,
    deserialize_training_manifest_json_v024,
    serialize_calibration_manifest_json_v024,
    serialize_calibration_mask_commitment_json_v024,
    serialize_calibration_population_audit_json_v024,
    serialize_model_state_json_v024,
    serialize_training_manifest_json_v024,
)
from lifetwin.experiments.calendar_long_horizon_v019_terminal import (
    TerminalContext,
    publish_terminal,
)
from lifetwin.experiments.calendar_long_horizon_v019_training import (
    V024CalibrationAudit,
    V024PretruthMaskCommitment,
    derive_calibration_mask_commitment_v024,
    fit_calibration_development_state_v024,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FORMAL_SCRIPT = _PROJECT_ROOT / "scripts" / "run_calendar_long_horizon_v019.py"
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
_MODEL_COMMITMENT_KEYS = frozenset(
    {"protocol_id", "config_sha256", "git_commit", "files", "created_utc"}
)
_FORMAL_ATTEMPT_ID = "v024-formal-20260810-a1"


class V024RunnerError(RuntimeError):
    """Raised when the frozen lifecycle cannot advance exactly."""


@dataclass(frozen=True, slots=True)
class V024RunPaths:
    repo_root: Path
    label_free_root: Path
    sealed_truth_root: Path
    score_root: Path
    termination_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        repo_root: str | Path,
        label_free_root: str | Path,
        sealed_truth_root: str | Path,
        score_root: str | Path,
        termination_root: str | Path,
    ) -> "V024RunPaths":
        artifact_values = (
            (label_free_root, "label-free"),
            (sealed_truth_root, "sealed-truth"),
            (score_root, "score"),
            (termination_root, "termination"),
        )
        for value, context in artifact_values:
            _reject_existing_reparse_traversal(value, context=context)
        repo = Path(repo_root).resolve()
        label, sealed, score, termination = (
            Path(value).resolve() for value, _ in artifact_values
        )
        artifact_roots = (label, sealed, score, termination)
        for left_index, left in enumerate(artifact_roots):
            for right in artifact_roots[left_index + 1 :]:
                try:
                    common = Path(os.path.commonpath((left, right)))
                except ValueError:
                    continue
                if common in {left, right}:
                    raise V024RunnerError(
                        "Label-free, sealed-truth, score, and termination roots "
                        "must be pairwise disjoint trees"
                    )
        return cls(repo, label, sealed, score, termination)

    @property
    def ledger_path(self) -> Path:
        return self.label_free_root / "exposure_log.jsonl"

    @property
    def truth_commitment_path(self) -> Path:
        return self.label_free_root / "truth_commitments.json"

    @property
    def prediction_commitment_path(self) -> Path:
        return self.label_free_root / "prediction_commitment.json"


@dataclass(frozen=True, slots=True)
class V024FormalRunResult:
    attempt_id: str
    git_commit: str
    truth_commitment_byte_sha256: str
    actual_analysis_hash_ledger_commitment_byte_sha256: str
    prediction_commitment_byte_sha256: str
    score_status: object
    wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class _TrainingArtifacts:
    center: CenterDevelopmentState
    risk: RiskDevelopmentState
    calibration: CalibrationDevelopmentState
    calibration_audit: V024CalibrationAudit
    mask_commitment: V024PretruthMaskCommitment
    center_input_bytes: Mapping[str, bytes]
    risk_input_bytes: Mapping[str, bytes]
    calibration_input_bytes: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class _CalibrationEvidence:
    kwargs: Mapping[str, object]
    cluster_ids: tuple[str, ...]
    forecast_days: np.ndarray


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_reparse(path: Path) -> bool:
    metadata = os.lstat(path)
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & flag)


def _reject_existing_reparse_traversal(
    raw: str | Path,
    *,
    context: str,
) -> None:
    lexical = Path(os.path.abspath(os.fspath(raw)))
    for candidate in (lexical, *lexical.parents):
        if os.path.lexists(candidate) and _is_reparse(candidate):
            raise V024RunnerError(f"{context} root traverses a reparse point")


def _require_physical_directory(root: Path, *, context: str) -> None:
    if not root.is_dir() or _is_reparse(root):
        raise V024RunnerError(f"{context} root is not a physical directory")
    for parent in root.parents:
        if parent.exists() and _is_reparse(parent):
            raise V024RunnerError(f"{context} root traverses a reparse point")


def _prepare_empty_physical_root(root: Path, *, context: str) -> None:
    if not root.exists():
        root.mkdir(parents=True)
    _require_physical_directory(root, context=context)
    if any(root.iterdir()):
        raise V024RunnerError(f"{context} root must be completely empty")


def _direct_child(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise V024RunnerError(f"Unsafe artifact filename: {filename!r}")
    child = (root / filename).resolve()
    if child.parent != root.resolve():
        raise V024RunnerError(f"Artifact escaped its root: {filename}")
    return child


def _exclusive_create(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise V024RunnerError(f"Formal artifact already exists: {path}") from exc
    except OSError as exc:
        raise V024RunnerError(f"Could not create formal artifact: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if path.read_bytes() != raw:
        raise V024RunnerError(f"Formal artifact changed after write: {path}")


def _file_hash(path: Path) -> str:
    return _sha256(path.read_bytes())


def _identity(
    attempt_id: str,
    environment: FormalEnvironmentIdentity,
) -> FormalAttemptIdentity:
    return FormalAttemptIdentity(
        attempt_id=attempt_id,
        git_commit=environment.git_commit,
        config_byte_sha256=environment.config_byte_sha256,
    )


def _append_phase(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    phase: str,
    exit_status: str,
    truth_hash: str | None,
    prediction_hash: str | None,
    message: str,
) -> None:
    append_formal_exposure_event(
        path=paths.ledger_path,
        identity=identity,
        contract=contract,
        created_utc=_utc_now(),
        phase=phase,
        exit_status=exit_status,
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        message=message,
    )


def _append_failure(
    *,
    error: BaseException,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    phase: str,
    truth_hash: str | None,
    prediction_hash: str | None,
) -> None:
    append_phase_error_without_masking(
        error=error,
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=contract,
        created_utc=_utc_now(),
        phase=phase,
        exit_status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        message=f"Formal {phase} stage did not complete.",
    )


def _write_json_checkpoint(
    path: Path,
    payload: Mapping[str, object],
    *,
    expected_keys: frozenset[str],
    view: V024ContractView,
) -> str:
    if (
        set(payload) != expected_keys
        or payload.get("protocol_id") != view.protocol.protocol_id
        or payload.get("config_sha256") != view.artifacts.config_byte_sha256
    ):
        raise V024RunnerError(f"{path.name} identity or schema changed")
    raw = canonical_json_bytes(payload)
    _exclusive_create(path, raw)
    return _sha256(raw)


def _read_label_frames(
    root: Path,
    contract: FrozenArtifactContract,
    *,
    include_predictions: bool = False,
) -> dict[str, pd.DataFrame]:
    names = (*_LABEL_INPUTS, *_FIT_OUTPUTS)
    if include_predictions:
        names = (*names, *_PREDICTION_OUTPUTS)
    return {
        name: read_canonical_csv(
            _direct_child(root, name),
            contract,
            formal=True,
        )
        for name in names
    }


def initialize_formal_attempt(
    *,
    paths: V024RunPaths,
    attempt_id: str,
    _contract_view: V024ContractView | None = None,
) -> tuple[FormalEnvironmentIdentity, V024ContractView, FormalAttemptIdentity]:
    """Attest the environment and create the sole initial ledger event."""

    if attempt_id != _FORMAL_ATTEMPT_ID:
        raise V024RunnerError(f"attempt_id must equal {_FORMAL_ATTEMPT_ID}")
    expected_parent = paths.repo_root / "artifacts"
    expected_roots = {
        "label-free": expected_parent / f"{attempt_id}-label-free",
        "sealed-truth": expected_parent / f"{attempt_id}-sealed-truth",
        "score": expected_parent / f"{attempt_id}-score",
        "termination": expected_parent / f"{attempt_id}-termination",
    }
    actual_roots = {
        "label-free": paths.label_free_root,
        "sealed-truth": paths.sealed_truth_root,
        "score": paths.score_root,
        "termination": paths.termination_root,
    }
    for context, expected in expected_roots.items():
        actual = actual_roots[context]
        if actual != expected.resolve():
            raise V024RunnerError(f"{context} root is not the frozen V2.4 path")
        if actual.exists():
            raise V024RunnerError(f"{context} root must not exist before launch")
    view = resolve_contract_view(_contract_view)
    environment = verify_formal_environment(paths.repo_root, contract_view=view)
    if environment.config_byte_sha256 != view.artifacts.config_byte_sha256:
        raise V024RunnerError("Environment and V2.4 contract hashes differ")
    for root, context in (
        (paths.label_free_root, "label-free"),
        (paths.sealed_truth_root, "sealed-truth"),
        (paths.score_root, "score"),
        (paths.termination_root, "termination"),
    ):
        _prepare_empty_physical_root(root, context=context)
    identity = _identity(attempt_id, environment)
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase="before_generation",
        exit_status="completed",
        truth_hash=None,
        prediction_hash=None,
        message="Clean frozen V2.4 environment verified before generation.",
    )
    return environment, view, identity


def run_isolated_generation_stage(
    *,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
    _contract_view: V024ContractView | None = None,
) -> None:
    """Run the RNG-free plan audit and one-shot generation in its subprocess."""

    view = resolve_contract_view(_contract_view)
    commit_frozen_v024_generation_plan(
        label_free_root=label_free_root,
        _contract_view=view,
    )
    generate_frozen_v024_artifacts(
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
        _contract_view=view,
    )


def _run_checked_process(arguments: Sequence[str], *, context: str) -> None:
    try:
        completed = subprocess.run(
            tuple(arguments),
            cwd=_PROJECT_ROOT,
            check=False,
        )
    except OSError as exc:
        raise V024RunnerError(f"Could not launch {context}") from exc
    if completed.returncode != 0:
        raise V024RunnerError(f"{context} exited with status {completed.returncode}")


def _launch_generation_process(paths: V024RunPaths) -> None:
    _run_checked_process(
        (
            sys.executable,
            str(_FORMAL_SCRIPT),
            "--internal-stage",
            "generation",
            "--label-free-root",
            str(paths.label_free_root),
            "--sealed-truth-root",
            str(paths.sealed_truth_root),
        ),
        context="isolated V2.4 generation process",
    )


def _launch_prediction_process(
    *,
    label_free_root: Path,
    attempt_id: str,
    repo_root: Path,
) -> None:
    _run_checked_process(
        (
            sys.executable,
            str(_FORMAL_SCRIPT),
            "--internal-stage",
            "prediction",
            "--label-free-root",
            str(label_free_root),
            "--attempt-id",
            attempt_id,
            "--repo-root",
            str(repo_root),
        ),
        context="truth-incapable V2.4 prediction process",
    )


def _commit_actual_analysis_hash_ledger(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    truth_hash: str,
) -> str:
    phase = "actual_analysis_hash_ledger_committed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message="Independent label-free actual-analysis hash reconstruction started.",
    )
    try:
        from lifetwin.experiments.calendar_long_horizon_v019_io import (  # noqa: PLC0415
            create_actual_analysis_hash_ledger_commitment_v024,
        )

        digest = create_actual_analysis_hash_ledger_commitment_v024(
            label_free_root=paths.label_free_root,
            attempt_id=identity.attempt_id,
            contract_view=view,
        )
        artifact = paths.label_free_root / "actual_analysis_hash_ledger_commitment.json"
        if _file_hash(artifact) != digest:
            raise V024RunnerError("Actual-analysis commitment changed after creation")
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(phase, digest),
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return digest


def _artifact_entry(
    path: Path,
    *,
    row_count: int,
) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.name,
        "row_count": row_count,
        "byte_count": len(raw),
        "byte_sha256": _sha256(raw),
    }


def _fit_structure_stage(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    truth_hash: str,
) -> tuple[WholeBundleValidated, str]:
    phase = "label_free_fit_committed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message="Six-worker fit-once stage started from verified label-free bytes.",
    )
    try:
        bundle = load_fresh_generation_bundle_v024(
            label_free_root=paths.label_free_root,
            attempt_id=identity.attempt_id,
            contract_view=view,
        )
        fitted = fit_verified_generation_bundle_v024(bundle)
        write_verified_fit_result_v024(
            bundle=bundle,
            fit_result=fitted,
        )
        del fitted
        whole = validate_whole_bundle_from_root(
            paths.label_free_root,
            view.artifacts,
        )
        digest = commit_validated_fit_result_v024(
            bundle=bundle,
            whole_bundle=whole,
            created_utc=_utc_now(),
        )
        del bundle
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(phase, digest),
        )
        progress = validate_formal_exposure_log(
            paths.ledger_path,
            view.artifacts,
        )[identity.attempt_id]
        verify_phase_artifact_commitment(
            progress,
            phase=phase,
            artifact_path=paths.label_free_root / "fit_commitment.json",
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return whole, digest


def _apply_partition(
    whole: WholeBundleValidated,
    *,
    partition: str,
    state: object,
    view: V024ContractView,
) -> tuple[LabelFreePipelineResult, Mapping[str, pd.DataFrame]]:
    partition_view = derive_partition_view(
        whole,
        partition=partition,
        contract=view,
    )
    pipeline = recompute_validated_partition_with_state_v024(
        partition_view,
        state=state,  # type: ignore[arg-type]
        contract=view,
    )
    return pipeline, consume_partition_frames(
        partition_view,
        contract=view,
    )


def _ordered_cluster_ids(
    pipeline: LabelFreePipelineResult,
    *,
    partition: str,
    include_operating_content: bool,
) -> tuple[str, ...]:
    content = pipeline.predictor_content_bundle.loc[
        pipeline.predictor_content_bundle["partition"].eq(partition)
    ]
    if content.empty or content["cluster_id"].duplicated().any():
        raise V024RunnerError(f"{partition} predictor-content rows are incomplete")
    keys: list[tuple[tuple[str, ...], str]] = []
    for row in content.itertuples(index=False):
        cluster_id = str(row.cluster_id)
        values = [str(row.arm_a_content_sha256)]
        if include_operating_content:
            values.extend(
                (
                    str(row.arm_b_content_sha256),
                    str(row.placebo_content_sha256),
                )
            )
        # Opaque IDs are used only to deterministically order byte-identical
        # label-free rows; no family or outcome value participates.
        values.append(cluster_id)
        keys.append((tuple(values), cluster_id))
    keys.sort(key=lambda item: item[0])
    return tuple(cluster_id for _, cluster_id in keys)


def _matrix_by_grid(
    frame: pd.DataFrame,
    *,
    partition: str,
    cluster_ids: Sequence[str],
    day_column: str,
    value_column: str,
    expected_days: Sequence[float],
) -> np.ndarray:
    selected = frame.loc[frame["partition"].eq(partition)]
    expected = tuple(float(value) for value in expected_days)
    records: list[np.ndarray] = []
    for cluster_id in cluster_ids:
        rows = selected.loc[selected["cluster_id"].eq(cluster_id)].sort_values(
            day_column,
            kind="stable",
        )
        days = tuple(pd.to_numeric(rows[day_column], errors="coerce").to_numpy(float))
        if days != expected:
            raise V024RunnerError(
                f"{partition}/{cluster_id} does not have the exact {day_column} grid"
            )
        values = pd.to_numeric(
            rows[value_column],
            errors="coerce",
        ).to_numpy(float)
        if values.shape != (len(expected),):
            raise V024RunnerError(
                f"{partition}/{cluster_id}/{value_column} has the wrong shape"
            )
        records.append(values)
    if set(selected["cluster_id"].astype(str)) != set(cluster_ids):
        raise V024RunnerError(
            f"{partition}/{value_column} contains unexpected opaque IDs"
        )
    return np.vstack(records)


def _forecast_matrix(
    frame: pd.DataFrame,
    *,
    partition: str,
    cluster_ids: Sequence[str],
    value_column: str,
) -> np.ndarray:
    return _matrix_by_grid(
        frame,
        partition=partition,
        cluster_ids=cluster_ids,
        day_column="forecast_day",
        value_column=value_column,
        expected_days=FORECAST_DAYS,
    )


def _latent_truth_matrix(
    truth_frame: pd.DataFrame,
    *,
    partition: str,
    cluster_ids: Sequence[str],
) -> np.ndarray:
    """Join development truth only by opaque ID and exact forecast day."""

    return _forecast_matrix(
        truth_frame,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="latent_retention_pct",
    )


def _feature_rows(
    feature_bundle: pd.DataFrame,
    *,
    partition: str,
    cluster_ids: Sequence[str],
) -> pd.DataFrame:
    rows = (
        feature_bundle.loc[feature_bundle["partition"].eq(partition)]
        .set_index("cluster_id")
        .reindex(cluster_ids)
    )
    if rows.index.has_duplicates or rows.isna().all(axis=1).any():
        raise V024RunnerError(f"{partition} feature rows are incomplete")
    return rows.reset_index()


def _phase_input_bytes(
    *,
    paths: V024RunPaths,
    label_filenames: Sequence[str],
    truth_filename: str,
    _input_filenames_by_stage: object | None = None,
    _stage: str | None = None,
) -> dict[str, bytes]:
    if _input_filenames_by_stage is not None:
        try:
            from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
                V020CheckpointRegistryError,
                require_input_filenames_by_stage_v020,
            )

            if _stage is None:
                raise V020CheckpointRegistryError("checkpoint stage is missing")
            registry = require_input_filenames_by_stage_v020(_input_filenames_by_stage)
            filenames = registry[_stage]
        except (KeyError, V020CheckpointRegistryError) as exc:
            raise V024RunnerError("V0.20 checkpoint registry is invalid") from exc
        if truth_filename not in filenames or any(
            name.endswith("_truth.csv") and name != truth_filename for name in filenames
        ):
            raise V024RunnerError("V0.20 checkpoint truth registry is invalid")
        return {
            name: _direct_child(
                paths.sealed_truth_root
                if name == truth_filename
                else paths.label_free_root,
                name,
            ).read_bytes()
            for name in filenames
        }
    result = {
        name: _direct_child(paths.label_free_root, name).read_bytes()
        for name in label_filenames
    }
    result[truth_filename] = _direct_child(
        paths.sealed_truth_root,
        truth_filename,
    ).read_bytes()
    return result


def _input_hashes(values: Mapping[str, bytes]) -> dict[str, str]:
    return {name: _sha256(raw) for name, raw in sorted(values.items())}


_COMMON_TRAINING_INPUTS = (
    "generation_plan_commitment.json",
    *_LABEL_INPUTS,
    "truth_commitments.json",
    "actual_analysis_hash_ledger_commitment.json",
    *_FIT_OUTPUTS,
    "fit_commitment.json",
)


def _fit_center_stage(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    frames: WholeBundleValidated,
    truth_hash: str,
    _input_filenames_by_stage: object | None = None,
) -> tuple[CenterDevelopmentState, Mapping[str, bytes], str]:
    partition = "center_development"
    probe, _ = _apply_partition(
        frames,
        partition=partition,
        state=make_probe_state(1.0),
        view=view,
    )
    center_truth = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=view,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="center_truth_opened",
        created_utc=_utc_now(),
    )["center_development_truth.csv"]
    phase = "center_state_committed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message="Center development fit started from its isolated truth slice.",
    )
    try:
        cluster_ids = _ordered_cluster_ids(
            probe,
            partition=partition,
            include_operating_content=False,
        )
        state = fit_center_development_state(
            library_forecasts_pct=_forecast_matrix(
                probe.prediction_bundle,
                partition=partition,
                cluster_ids=cluster_ids,
                value_column="center_forecast_pct",
            ),
            sqrt_forecasts_pct=_forecast_matrix(
                probe.prediction_bundle,
                partition=partition,
                cluster_ids=cluster_ids,
                value_column="sqrt_time_forecast_pct",
            ),
            latent_targets_pct=_latent_truth_matrix(
                center_truth,
                partition=partition,
                cluster_ids=cluster_ids,
            ),
        )
        input_bytes = _phase_input_bytes(
            paths=paths,
            label_filenames=_COMMON_TRAINING_INPUTS,
            truth_filename="center_development_truth.csv",
            _input_filenames_by_stage=_input_filenames_by_stage,
            _stage="center_development",
        )
        payload = {
            "protocol_id": view.protocol.protocol_id,
            "config_sha256": view.artifacts.config_byte_sha256,
            "state_kind": "center_development",
            "center_state_sha256": center_state_sha256(state),
            "center_beta": state.beta,
            "development_cluster_count": state.development_cluster_count,
            "forecast_horizon_count": state.forecast_horizon_count,
            "ridge_penalty": state.ridge_penalty,
            "completeness_rule": state.completeness_rule,
            "input_byte_hashes": _input_hashes(input_bytes),
            "created_utc": _utc_now(),
        }
        checkpoint_hash = _write_json_checkpoint(
            paths.label_free_root / "center_state_checkpoint.json",
            payload,
            expected_keys=_CENTER_CHECKPOINT_KEYS,
            view=view,
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(phase, checkpoint_hash),
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return state, input_bytes, checkpoint_hash


def _fit_risk_stage(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    frames: WholeBundleValidated,
    center: CenterDevelopmentState,
    center_input_bytes: Mapping[str, bytes],
    center_checkpoint_hash: str,
    truth_hash: str,
    _input_filenames_by_stage: object | None = None,
) -> tuple[RiskDevelopmentState, Mapping[str, bytes], str]:
    if (
        _file_hash(paths.label_free_root / "center_state_checkpoint.json")
        != center_checkpoint_hash
    ):
        raise V024RunnerError("Center checkpoint changed before risk reveal")
    partition = "risk_development"
    probe, _ = _apply_partition(
        frames,
        partition=partition,
        state=make_probe_state(center.beta),
        view=view,
    )
    risk_truth = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=view,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="risk_truth_opened",
        created_utc=_utc_now(),
        _input_filenames_by_stage=_input_filenames_by_stage,
    )["risk_development_truth.csv"]
    phase = "risk_state_committed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message="Risk development fit started from its isolated truth slice.",
    )
    try:
        cluster_ids = _ordered_cluster_ids(
            probe,
            partition=partition,
            include_operating_content=True,
        )
        features = _feature_rows(
            probe.feature_bundle,
            partition=partition,
            cluster_ids=cluster_ids,
        )
        planned = np.asarray(
            [
                stress_index(*(float(row[name]) for name in REAL_OPERATING_FIELDS[4:]))
                for _, row in features.iterrows()
            ],
            dtype=np.float64,
        )
        state = fit_risk_development_state(
            prefix_features=features.loc[
                :,
                list(PREFIX_FEATURE_NAMES),
            ].to_numpy(float),
            visible_stress_features=features.loc[
                :,
                list(REAL_OPERATING_FIELDS),
            ].to_numpy(float),
            placebo_features=features.loc[
                :,
                list(PLACEBO_FIELDS),
            ].to_numpy(float),
            planned_stress_index=planned,
            frozen_center_25y_pct=_forecast_matrix(
                probe.prediction_bundle,
                partition=partition,
                cluster_ids=cluster_ids,
                value_column="center_forecast_pct",
            )[:, -1],
            latent_target_25y_pct=_latent_truth_matrix(
                risk_truth,
                partition=partition,
                cluster_ids=cluster_ids,
            )[:, -1],
            common_pool_eligible=features["hard_eligible"].tolist(),
        )
        input_bytes = _phase_input_bytes(
            paths=paths,
            label_filenames=(
                *_COMMON_TRAINING_INPUTS,
                "center_state_checkpoint.json",
            ),
            truth_filename="risk_development_truth.csv",
            _input_filenames_by_stage=_input_filenames_by_stage,
            _stage="risk_development",
        )
        training_raw = serialize_training_manifest_json_v024(
            center_development_input_hashes=_input_hashes(center_input_bytes),
            risk_development_input_hashes=_input_hashes(input_bytes),
            center_state=center,
            risk_state=state,
            created_utc=_utc_now(),
            contract_view=view,
        )
        _exclusive_create(
            paths.label_free_root / "training_manifest.json",
            training_raw,
        )
        deserialize_training_manifest_json_v024(
            training_raw,
            center_state=center,
            risk_state=state,
            contract_view=view,
        )
        payload = {
            "protocol_id": view.protocol.protocol_id,
            "config_sha256": view.artifacts.config_byte_sha256,
            "state_kind": "risk_development",
            "center_checkpoint_byte_sha256": center_checkpoint_hash,
            "training_manifest_byte_sha256": _sha256(training_raw),
            "risk_state_sha256": risk_state_sha256(state),
            "development_cluster_count": state.development_cluster_count,
            "eligible_cluster_count": state.eligible_cluster_count,
            "positive_label_count": state.positive_label_count,
            "negative_label_count": state.negative_label_count,
            "input_byte_hashes": _input_hashes(input_bytes),
            "created_utc": _utc_now(),
        }
        checkpoint_hash = _write_json_checkpoint(
            paths.label_free_root / "risk_state_checkpoint.json",
            payload,
            expected_keys=_RISK_CHECKPOINT_KEYS,
            view=view,
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(phase, checkpoint_hash),
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return state, input_bytes, checkpoint_hash


def _calibration_probe_state(
    center: CenterDevelopmentState,
    risk: RiskDevelopmentState,
) -> object:
    probe = make_probe_state(center.beta)
    return replace(
        probe,
        prefix_only_risk=risk.prefix_only_risk,
        visible_stress_risk=risk.visible_stress_risk,
        placebo_risk=risk.placebo_risk,
        arm_a_plus_s_plan_risk=risk.arm_a_plus_s_plan_risk,
        strongest_single_feature_name=risk.strongest_single_feature_name,
        strongest_single_feature_orientation=(
            risk.strongest_single_feature_orientation
        ),
    )


def _persistence_matrix(
    prefix_pack: pd.DataFrame,
    *,
    partition: str,
    cluster_ids: Sequence[str],
) -> np.ndarray:
    endpoints = _matrix_by_grid(
        prefix_pack,
        partition=partition,
        cluster_ids=cluster_ids,
        day_column="prefix_day",
        value_column="observed_retention_pct",
        expected_days=PREFIX_DAYS,
    )[:, -1]
    return np.repeat(endpoints[:, None], len(FORECAST_DAYS), axis=1)


def _structural_supports(
    *,
    diagnostics: pd.DataFrame,
    forecasts: pd.DataFrame,
    partition: str,
    cluster_ids: Sequence[str],
) -> list[dict[str, list[list[float]]]]:
    diagnostic_partition = diagnostics.loc[diagnostics["partition"].eq(partition)]
    forecast_partition = forecasts.loc[forecasts["partition"].eq(partition)]
    result: list[dict[str, list[list[float]]]] = []
    expected_days = tuple(float(day) for day in FORECAST_DAYS)
    for cluster_id in cluster_ids:
        cluster_diagnostics = diagnostic_partition.loc[
            diagnostic_partition["cluster_id"].eq(cluster_id)
        ]
        support: dict[str, list[list[float]]] = {}
        for model_id in DECLARED_STRUCTURE_FAMILIES:
            credible = cluster_diagnostics.loc[
                cluster_diagnostics["model_id"].eq(model_id)
                & cluster_diagnostics["credible_variant"].eq(True)  # noqa: E712
            ]
            vectors: list[list[float]] = []
            for variant_id in credible["variant_id"].astype(str):
                rows = forecast_partition.loc[
                    forecast_partition["cluster_id"].eq(cluster_id)
                    & forecast_partition["model_id"].eq(model_id)
                    & forecast_partition["variant_id"].eq(variant_id)
                ].sort_values("forecast_day", kind="stable")
                days = tuple(
                    pd.to_numeric(
                        rows["forecast_day"],
                        errors="coerce",
                    ).to_numpy(float)
                )
                values = pd.to_numeric(
                    rows["raw_forecast_retention_pct"],
                    errors="coerce",
                ).to_numpy(float)
                if days != expected_days or values.shape != (len(expected_days),):
                    raise V024RunnerError(
                        f"{partition}/{cluster_id}/{model_id}/{variant_id} "
                        "does not have one exact forecast vector"
                    )
                vectors.append(values.tolist())
            if vectors:
                support[model_id] = vectors
        result.append(support)
    return result


def _calibration_evidence(
    *,
    frames: Mapping[str, pd.DataFrame],
    probe: LabelFreePipelineResult,
    partition: str,
    cluster_ids: Sequence[str],
) -> _CalibrationEvidence:
    identifiers = tuple(str(value) for value in cluster_ids)
    features = _feature_rows(
        probe.feature_bundle,
        partition=partition,
        cluster_ids=identifiers,
    )
    contents = (
        probe.predictor_content_bundle.loc[
            probe.predictor_content_bundle["partition"].eq(partition)
        ]
        .set_index("cluster_id")
        .reindex(identifiers)
    )
    if contents.index.has_duplicates or contents.isna().any(axis=None):
        raise V024RunnerError("Calibration predictor-content rows are incomplete")
    risks = (
        probe.primary_risk_bundle.loc[
            probe.primary_risk_bundle["partition"].eq(partition)
        ]
        .pivot(index="cluster_id", columns="score_id", values="raw_risk_score")
        .reindex(identifiers)
    )
    required_scores = {"prefix_only", "visible_stress"}
    if not required_scores.issubset(risks.columns) or risks.loc[
        :, list(required_scores)
    ].isna().any(axis=None):
        raise V024RunnerError("Calibration primary-risk rows are incomplete")
    center = _forecast_matrix(
        probe.prediction_bundle,
        partition=partition,
        cluster_ids=identifiers,
        value_column="center_forecast_pct",
    )
    prefix_days = _matrix_by_grid(
        frames["prefix_pack.csv"],
        partition=partition,
        cluster_ids=identifiers,
        day_column="prefix_day",
        value_column="prefix_day",
        expected_days=PREFIX_DAYS,
    )
    prefix_observations = _matrix_by_grid(
        frames["prefix_pack.csv"],
        partition=partition,
        cluster_ids=identifiers,
        day_column="prefix_day",
        value_column="observed_retention_pct",
        expected_days=PREFIX_DAYS,
    )
    forecast_days = _matrix_by_grid(
        frames["forecast_coordinates.csv"],
        partition=partition,
        cluster_ids=identifiers,
        day_column="forecast_day",
        value_column="forecast_day",
        expected_days=FORECAST_DAYS,
    )
    supports = _structural_supports(
        diagnostics=frames["member_fit_diagnostics.csv"],
        forecasts=frames["member_forecast_bundle.csv"],
        partition=partition,
        cluster_ids=identifiers,
    )
    counts = np.asarray([len(value) for value in supports], dtype=np.int64)
    prediction = probe.prediction_bundle
    kwargs: dict[str, object] = {
        "risk_state": None,
        "cluster_ids": identifiers,
        "arm_a_predictor_content_sha256": contents["arm_a_content_sha256"]
        .astype(str)
        .to_numpy(object),
        "arm_b_predictor_content_sha256": contents["arm_b_content_sha256"]
        .astype(str)
        .to_numpy(object),
        "placebo_predictor_content_sha256": contents["placebo_content_sha256"]
        .astype(str)
        .to_numpy(object),
        "prefix_days": prefix_days,
        "prefix_observations_pct": prefix_observations,
        "forecast_days": forecast_days,
        "real_operating_fields": features.loc[
            :,
            list(REAL_OPERATING_FIELDS),
        ].to_numpy(float),
        "placebo_operating_fields": features.loc[
            :,
            list(PLACEBO_FIELDS),
        ].to_numpy(float),
        "real_stress_features": features.loc[
            :,
            list(REAL_OPERATING_FIELDS),
        ].to_numpy(float),
        "placebo_features": features.loc[
            :,
            list(PLACEBO_FIELDS),
        ].to_numpy(float),
        "successful_structure_family_count": counts,
        "structural_family_supports_pct": supports,
        "frozen_center_forecasts_pct": center,
        "prefix_features": features.loc[
            :,
            list(PREFIX_FEATURE_NAMES),
        ].to_numpy(float),
        "raw_prefix_risk_scores": risks["prefix_only"].to_numpy(float),
        "raw_visible_risk_scores": risks["visible_stress"].to_numpy(float),
        "base_interval_lower_pct": _forecast_matrix(
            prediction,
            partition=partition,
            cluster_ids=identifiers,
            value_column="base_interval_lower_pct",
        ),
        "base_interval_upper_pct": _forecast_matrix(
            prediction,
            partition=partition,
            cluster_ids=identifiers,
            value_column="base_interval_upper_pct",
        ),
        "mean_baseline_forecasts_pct": {
            "target_prefix_persistence": _persistence_matrix(
                frames["prefix_pack.csv"],
                partition=partition,
                cluster_ids=identifiers,
            ),
            "target_prefix_sqrt_time": _forecast_matrix(
                prediction,
                partition=partition,
                cluster_ids=identifiers,
                value_column="sqrt_time_forecast_pct",
            ),
            "target_prefix_bounded_power_law": _forecast_matrix(
                prediction,
                partition=partition,
                cluster_ids=identifiers,
                value_column="bounded_power_forecast_pct",
            ),
        },
    }
    return _CalibrationEvidence(
        kwargs=kwargs,
        cluster_ids=identifiers,
        forecast_days=forecast_days,
    )


def _create_model_state_commitment(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
) -> str:
    payload = {
        "protocol_id": view.protocol.protocol_id,
        "config_sha256": view.artifacts.config_byte_sha256,
        "git_commit": identity.git_commit,
        "files": [
            _artifact_entry(
                _direct_child(paths.label_free_root, name),
                row_count=1,
            )
            for name in _MODEL_STATE_COMMITMENT_FILES
        ],
        "created_utc": _utc_now(),
    }
    return _write_json_checkpoint(
        paths.label_free_root / "model_state_commitment.json",
        payload,
        expected_keys=_MODEL_COMMITMENT_KEYS,
        view=view,
    )


def _fit_calibration_stage(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    frames: WholeBundleValidated,
    center: CenterDevelopmentState,
    risk: RiskDevelopmentState,
    center_input_bytes: Mapping[str, bytes],
    risk_input_bytes: Mapping[str, bytes],
    risk_checkpoint_hash: str,
    truth_hash: str,
    _input_filenames_by_stage: object | None = None,
) -> tuple[
    CalibrationDevelopmentState,
    V024CalibrationAudit,
    V024PretruthMaskCommitment,
    Mapping[str, bytes],
    str,
]:
    if (
        _file_hash(paths.label_free_root / "risk_state_checkpoint.json")
        != risk_checkpoint_hash
    ):
        raise V024RunnerError("Risk checkpoint changed before mask derivation")
    partition = "calibration"
    probe, partition_frames = _apply_partition(
        frames,
        partition=partition,
        state=_calibration_probe_state(center, risk),
        view=view,
    )
    cluster_ids = _ordered_cluster_ids(
        probe,
        partition=partition,
        include_operating_content=True,
    )
    evidence = _calibration_evidence(
        frames=partition_frames,
        probe=probe,
        partition=partition,
        cluster_ids=cluster_ids,
    )
    evidence_kwargs = dict(evidence.kwargs)
    evidence_kwargs["risk_state"] = risk
    evidence_kwargs["protocol_id"] = view.protocol.protocol_id

    mask_phase = "calibration_mask_committed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view,
        phase=mask_phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message="Outcome-free calibration eligibility derivation started.",
    )
    try:
        mask = derive_calibration_mask_commitment_v024(**evidence_kwargs)
        mask_raw = serialize_calibration_mask_commitment_json_v024(mask)
        _exclusive_create(
            paths.label_free_root / "calibration_mask_commitment.json",
            mask_raw,
        )
        mask_hash = _sha256(mask_raw)
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=mask_phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(mask_phase, mask_hash),
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=mask_phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise

    calibration_truth = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=view,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="calibration_truth_opened",
        created_utc=_utc_now(),
        calibration_mask_commitment_path=(
            paths.label_free_root / "calibration_mask_commitment.json"
        ),
        _input_filenames_by_stage=_input_filenames_by_stage,
    )["calibration_truth.csv"]
    model_phase = "model_state_committed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=model_phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message="Calibration and model-state serialization started.",
    )
    try:
        latent = _latent_truth_matrix(
            calibration_truth,
            partition=partition,
            cluster_ids=evidence.cluster_ids,
        )
        calibration, audit = fit_calibration_development_state_v024(
            pretruth_commitment=mask,
            **evidence_kwargs,
            latent_target_cluster_ids=evidence.cluster_ids,
            latent_target_forecast_days=evidence.forecast_days,
            latent_targets_pct=latent,
        )
        calibration_input_bytes = _phase_input_bytes(
            paths=paths,
            label_filenames=(
                *_COMMON_TRAINING_INPUTS,
                "center_state_checkpoint.json",
                "risk_state_checkpoint.json",
                "training_manifest.json",
                "calibration_mask_commitment.json",
            ),
            truth_filename="calibration_truth.csv",
            _input_filenames_by_stage=_input_filenames_by_stage,
            _stage="calibration",
        )
        calibration_raw = serialize_calibration_manifest_json_v024(
            calibration_input_hashes=_input_hashes(calibration_input_bytes),
            calibration_state=calibration,
            created_utc=_utc_now(),
            mask_commitment=mask,
            contract_view=view,
        )
        _exclusive_create(
            paths.label_free_root / "calibration_manifest.json",
            calibration_raw,
        )
        deserialize_calibration_manifest_json_v024(
            calibration_raw,
            calibration_state=calibration,
            mask_commitment=mask,
            contract_view=view,
        )
        training_state = FrozenTrainingState(center, risk, calibration)
        provenance = _issue_v024_training_provenance_from_fresh_bytes(
            contract_view=view,
            attempt_id=identity.attempt_id,
            training_state=training_state,
            calibration_audit=audit,
            mask_commitment=mask,
            generation_plan_commitment_bytes=(
                paths.label_free_root / "generation_plan_commitment.json"
            ).read_bytes(),
            truth_commitment_bytes=paths.truth_commitment_path.read_bytes(),
            actual_analysis_hash_ledger_commitment_bytes=(
                paths.label_free_root / "actual_analysis_hash_ledger_commitment.json"
            ).read_bytes(),
            label_free_fit_commitment_bytes=(
                paths.label_free_root / "fit_commitment.json"
            ).read_bytes(),
            center_state_commitment_bytes=(
                paths.label_free_root / "center_state_checkpoint.json"
            ).read_bytes(),
            risk_state_commitment_bytes=(
                paths.label_free_root / "risk_state_checkpoint.json"
            ).read_bytes(),
            calibration_state_commitment_bytes=calibration_raw,
            center_development_input_bytes=center_input_bytes,
            risk_development_input_bytes=risk_input_bytes,
            calibration_input_bytes=calibration_input_bytes,
        )
        audit_raw = serialize_calibration_population_audit_json_v024(
            provenance_envelope=provenance,
            created_utc=_utc_now(),
        )
        _exclusive_create(
            paths.label_free_root / "calibration_population_audit.json",
            audit_raw,
        )
        deserialize_calibration_population_audit_json_v024(
            audit_raw,
            provenance_envelope=provenance,
        )
        model_raw = serialize_model_state_json_v024(
            provenance,
            software_versions=default_software_versions(),
            created_utc=_utc_now(),
        )
        _exclusive_create(paths.label_free_root / "model_state.json", model_raw)
        deserialize_model_state_json_v024(
            model_raw,
            provenance_envelope=provenance,
        )
        model_hash = _create_model_state_commitment(
            paths=paths,
            identity=identity,
            view=view,
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=model_phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(model_phase, model_hash),
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=model_phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return calibration, audit, mask, calibration_input_bytes, model_hash


def _fit_training_stages(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    frames: WholeBundleValidated,
    truth_hash: str,
    _input_filenames_by_stage: object | None = None,
) -> _TrainingArtifacts:
    center, center_inputs, center_hash = _fit_center_stage(
        paths=paths,
        identity=identity,
        view=view,
        frames=frames,
        truth_hash=truth_hash,
        _input_filenames_by_stage=_input_filenames_by_stage,
    )
    risk, risk_inputs, risk_hash = _fit_risk_stage(
        paths=paths,
        identity=identity,
        view=view,
        frames=frames,
        center=center,
        center_input_bytes=center_inputs,
        center_checkpoint_hash=center_hash,
        truth_hash=truth_hash,
        _input_filenames_by_stage=_input_filenames_by_stage,
    )
    calibration, audit, mask, calibration_inputs, _ = _fit_calibration_stage(
        paths=paths,
        identity=identity,
        view=view,
        frames=frames,
        center=center,
        risk=risk,
        center_input_bytes=center_inputs,
        risk_input_bytes=risk_inputs,
        risk_checkpoint_hash=risk_hash,
        truth_hash=truth_hash,
        _input_filenames_by_stage=_input_filenames_by_stage,
    )
    return _TrainingArtifacts(
        center=center,
        risk=risk,
        calibration=calibration,
        calibration_audit=audit,
        mask_commitment=mask,
        center_input_bytes=center_inputs,
        risk_input_bytes=risk_inputs,
        calibration_input_bytes=calibration_inputs,
    )


def _prediction_and_commitment(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    truth_hash: str,
) -> tuple[
    str,
    V024CommittedModelStateEnvelope,
    Mapping[str, pd.DataFrame],
    object,
]:
    phase = "prediction_started"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message=(
            "Truth-incapable prediction subprocess started from committed "
            "label-free state."
        ),
    )
    try:
        bundle: V024CommittedLabelFreeBundle = load_committed_label_free_bundle_v024(
            label_free_root=paths.label_free_root,
            attempt_id=identity.attempt_id,
            contract_view=view,
        )
        model = bundle.model_state
        _launch_prediction_process(
            label_free_root=paths.label_free_root,
            attempt_id=identity.attempt_id,
            repo_root=paths.repo_root,
        )
        precommit_evidence = create_prediction_commitment_v024(
            bundle,
            created_utc=_utc_now(),
        )
        prediction_hash = precommit_evidence.byte_sha256
        if _file_hash(paths.prediction_commitment_path) != prediction_hash:
            raise V024RunnerError(
                "Prediction commitment changed before ledger publication"
            )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message="Truth-incapable prediction outputs were written exactly once.",
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise

    commit_phase = "prediction_committed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=commit_phase,
        exit_status="completed",
        truth_hash=truth_hash,
        prediction_hash=prediction_hash,
        message=(
            "All predictor artifacts were committed before held-out truth access."
        ),
    )
    evidence = verify_prediction_commitment_v024(
        label_free_root=paths.label_free_root,
        attempt_id=identity.attempt_id,
        contract_view=view,
        require_ledger_committed=True,
    )
    frames = _read_label_frames(
        paths.label_free_root,
        view.artifacts,
        include_predictions=True,
    )
    prediction_envelope = _issue_prediction_commitment_envelope_v024(
        prediction_frames=frames,
        model_state_envelope=model,
        prediction_commitment_evidence=evidence,
        formal=True,
        contract_view=view,
    )
    return prediction_hash, model, frames, prediction_envelope


def _require_empty_score_root(root: Path) -> None:
    try:
        _require_physical_directory(root, context="score")
    except V024RunnerError as exc:
        raise V024RunnerError(
            "Score root must remain a physical empty directory until scoring"
        ) from exc
    if any(root.iterdir()):
        raise V024RunnerError(
            "Score root must remain a physical empty directory until scoring"
        )


def _score_and_write(
    *,
    paths: V024RunPaths,
    identity: FormalAttemptIdentity,
    view: V024ContractView,
    truth_hash: str,
    prediction_hash: str,
    prediction_frames: Mapping[str, pd.DataFrame],
    model: V024CommittedModelStateEnvelope,
    prediction_envelope: object,
) -> object:
    truth_frames = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=view,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="scoring_truth_opened",
        created_utc=_utc_now(),
        calibration_mask_commitment_path=(
            paths.label_free_root / "calibration_mask_commitment.json"
        ),
        prediction_commitment_path=paths.prediction_commitment_path,
    )
    phase = "scoring_completed"
    _append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=prediction_hash,
        message="Committed-artifact-only scoring started.",
    )
    try:
        result = score_committed_artifacts(
            prediction_frames=prediction_frames,
            truth_frames=truth_frames,
            model_state_envelope=model,
            prediction_commitment_envelope=prediction_envelope,
            _contract_view=view,
        )
        payloads = required_score_artifact_payloads_v024(
            result,
            contract_view=view,
        )
        if tuple(payloads) != tuple(REQUIRED_SCORE_ARTIFACTS):
            raise V024RunnerError("Scorer returned a changed artifact registry")
        _require_empty_score_root(paths.score_root)
        for name in REQUIRED_SCORE_ARTIFACTS:
            _exclusive_create(paths.score_root / name, payloads[name])
        if tuple(sorted(path.name for path in paths.score_root.iterdir())) != (
            tuple(sorted(REQUIRED_SCORE_ARTIFACTS))
        ):
            raise V024RunnerError("Persisted score registry is incomplete")
        _append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=prediction_hash,
            message="All frozen score artifacts were written without tuning.",
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            truth_hash=truth_hash,
            prediction_hash=prediction_hash,
        )
        raise
    return result.score_report.get("status")


def _publish_preprediction_failure(
    *,
    error: BaseException,
    paths: V024RunPaths,
    view: V024ContractView,
    attempt_id: str,
    attempt_created_utc: str,
) -> None:
    """Publish the exact terminal registry only before prediction commitment."""

    try:
        progress = validate_formal_exposure_log(
            paths.ledger_path,
            view.artifacts,
        ).get(attempt_id)
    except BaseException as ledger_error:
        error.add_note(
            "Terminal publication was unavailable because the exposure ledger "
            f"could not be validated: {ledger_error!r}"
        )
        return
    if progress is None or progress.prediction_commitment_byte_sha256 is not None:
        return
    try:
        _prepare_empty_physical_root(
            paths.termination_root,
            context="termination",
        )
        publish_terminal(
            termination_root=paths.termination_root,
            label_free_artifact_root=paths.label_free_root,
            context=TerminalContext.from_progress(
                progress,
                created_utc=attempt_created_utc,
                terminated_utc=_utc_now(),
                protocol_id=view.protocol.protocol_id,
            ),
            error=error,
            repo_root=paths.repo_root,
            _contract_view=view,
        )
    except BaseException as publication_error:
        error.add_note(
            "The original lifecycle error was preserved, but pre-prediction "
            f"terminal publication failed: {publication_error!r}"
        )
        raise error from publication_error


def run_formal_attempt(
    *,
    attempt_id: str,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
    score_root: str | Path,
    termination_root: str | Path,
    repo_root: str | Path = _PROJECT_ROOT,
    _contract_view: V024ContractView | None = None,
) -> V024FormalRunResult:
    """Execute the fixed V2.4 lifecycle without a scientific override surface."""

    started = time.monotonic()
    attempt_created_utc = _utc_now()
    paths = V024RunPaths.resolve(
        repo_root=repo_root,
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
        score_root=score_root,
        termination_root=termination_root,
    )
    environment, view, identity = initialize_formal_attempt(
        paths=paths,
        attempt_id=attempt_id,
        _contract_view=_contract_view,
    )
    try:
        if view.design_status != "implementation_frozen":
            raise V024RunnerError("V2.4 design is not implementation_frozen")
        _launch_generation_process(paths)
        progress = validate_formal_exposure_log(
            paths.ledger_path,
            view.artifacts,
        ).get(attempt_id)
        if (
            progress is None
            or progress.completed_phase != "truth_committed"
            or progress.pending_phase is not None
            or progress.truth_commitments_byte_sha256 is None
        ):
            raise V024RunnerError("Isolated generation did not commit all truth")
        truth_hash = progress.truth_commitments_byte_sha256
        if _file_hash(paths.truth_commitment_path) != truth_hash:
            raise V024RunnerError("Truth commitment differs from the exposure ledger")
        actual_hash = _commit_actual_analysis_hash_ledger(
            paths=paths,
            identity=identity,
            view=view,
            truth_hash=truth_hash,
        )
        frames, _ = _fit_structure_stage(
            paths=paths,
            identity=identity,
            view=view,
            truth_hash=truth_hash,
        )
        _fit_training_stages(
            paths=paths,
            identity=identity,
            view=view,
            frames=frames,
            truth_hash=truth_hash,
        )
        prediction_hash, model, prediction_frames, prediction_envelope = (
            _prediction_and_commitment(
                paths=paths,
                identity=identity,
                view=view,
                truth_hash=truth_hash,
            )
        )
        score_status = _score_and_write(
            paths=paths,
            identity=identity,
            view=view,
            truth_hash=truth_hash,
            prediction_hash=prediction_hash,
            prediction_frames=prediction_frames,
            model=model,
            prediction_envelope=prediction_envelope,
        )
        final = validate_formal_exposure_log(
            paths.ledger_path,
            view.artifacts,
        )[attempt_id]
        if (
            final.completed_phase != "scoring_completed"
            or final.pending_phase is not None
            or final.terminal_failed
        ):
            raise V024RunnerError("Formal attempt did not reach scoring_completed")
    except BaseException as exc:
        _publish_preprediction_failure(
            error=exc,
            paths=paths,
            view=view,
            attempt_id=attempt_id,
            attempt_created_utc=attempt_created_utc,
        )
        raise
    return V024FormalRunResult(
        attempt_id=attempt_id,
        git_commit=environment.git_commit,
        truth_commitment_byte_sha256=truth_hash,
        actual_analysis_hash_ledger_commitment_byte_sha256=actual_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        score_status=score_status,
        wall_time_seconds=float(time.monotonic() - started),
    )


__all__ = [
    "V024FormalRunResult",
    "V024RunPaths",
    "V024RunnerError",
    "initialize_formal_attempt",
    "run_formal_attempt",
    "run_isolated_generation_stage",
]
