"""Strict process orchestration for the one-shot frozen V0.15 experiment.

The formal runner is intentionally narrower than the scientific components it
coordinates.  It owns phase order, exclusive artifact creation, checkpoint
binding, and the process boundary that keeps the prediction stage from
receiving a sealed-truth path.  The frozen generator and scorer remain the
only components that generate truth and evaluate outcomes, respectively.

This entrypoint is deliberately one-shot.  An interrupted ledger is retained
for publication, but this module does not silently reuse partial artifacts.
Any protocol-authorized restart must use a new empty root set and remain
traceable alongside the interrupted attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_environment import (
    FormalEnvironmentIdentity,
    verify_formal_environment,
)
from lifetwin.experiments.calendar_long_horizon_v015_firewall import (
    FormalAttemptIdentity,
    append_formal_exposure_event,
    open_truth_for_phase,
    phase_commitment_message,
    validate_formal_exposure_log,
    verify_phase_artifact_commitment,
)
from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEYS,
)
from lifetwin.experiments.calendar_long_horizon_v015_generation import (
    generate_frozen_v015_artifacts,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    DEFAULT_V2_CONFIG_PATH,
    FROZEN_PROTOCOL_ID,
    FrozenArtifactContract,
    V015ArtifactError,
    assert_separate_truth_roots,
    canonical_csv_bytes,
    canonical_json_bytes,
    create_prediction_commitment,
    load_artifact_contract,
    read_canonical_csv,
    read_prediction_artifact_bundle,
    verify_prediction_commitment,
    write_canonical_csv,
    write_canonical_json,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_FEATURE_NAMES,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    PLACEBO_FEATURE_NAMES,
    REAL_OPERATING_FIELDS,
    LabelFreePipelineResult,
)
from lifetwin.experiments.calendar_long_horizon_v015_prediction import (
    fit_structure_library_formal,
    run_pipeline_from_fitted,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256,
    load_frozen_protocol_config,
    stress_index,
)
from lifetwin.experiments.calendar_long_horizon_v015_scoring import (
    REQUIRED_FORMAL_NON_SCORE_ARTIFACTS,
    REQUIRED_SCORE_ARTIFACTS,
    finalize_run_manifest,
    required_score_artifact_payloads,
    score_committed_artifacts,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import (
    CalibrationDevelopmentState,
    CenterDevelopmentState,
    DecodedModelState,
    FrozenTrainingState,
    RiskDevelopmentState,
    build_calibration_manifest,
    build_training_manifest,
    center_state_sha256,
    default_software_versions,
    deserialize_model_state_json,
    fit_calibration_development_state,
    fit_center_development_state,
    fit_risk_development_state,
    make_probe_state,
    risk_state_sha256,
    serialize_model_state_json,
    verify_calibration_manifest_state_hashes,
    verify_training_manifest_state_hashes,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FORMAL_SCRIPT = _PROJECT_ROOT / "scripts" / "run_calendar_long_horizon_v015.py"
_FREEZE_RECORD = (
    _PROJECT_ROOT
    / "reports"
    / "synthetic_long_horizon_identifiability_freeze_record_v2.json"
)
_PROTOCOL_FREEZE_GIT_COMMIT = "b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91"
_LABEL_INPUT_FILENAMES = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
)
_FIT_FILENAMES = (
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
)
_PREDICTION_OUTPUT_FILENAMES = (
    "prediction_bundle.csv",
    "risk_bundle.csv",
    "decision_bundle.csv",
)
_STATE_FILENAMES = (
    "fit_commitment.json",
    "center_state_checkpoint.json",
    "risk_state_checkpoint.json",
    "training_manifest.json",
    "calibration_manifest.json",
    "model_state.json",
    "model_state_commitment.json",
)
_POST_GENERATION_LABEL_FILES = frozenset(
    {
        *_LABEL_INPUT_FILENAMES,
        "truth_commitments.json",
        "exposure_log.jsonl",
    }
)
_POST_FIT_LABEL_FILES = frozenset(
    {
        *_POST_GENERATION_LABEL_FILES,
        *_FIT_FILENAMES,
        "fit_commitment.json",
    }
)
_POST_CENTER_LABEL_FILES = frozenset(
    {*_POST_FIT_LABEL_FILES, "center_state_checkpoint.json"}
)
_POST_RISK_LABEL_FILES = frozenset(
    {
        *_POST_CENTER_LABEL_FILES,
        "risk_state_checkpoint.json",
        "training_manifest.json",
    }
)
_PRE_PREDICTION_LABEL_FILES = frozenset(
    {
        *_POST_GENERATION_LABEL_FILES,
        *_FIT_FILENAMES,
        *_STATE_FILENAMES,
    }
)
_PREDICTION_NO_COMMITMENT_LABEL_FILES = frozenset(
    {*_PRE_PREDICTION_LABEL_FILES, *_PREDICTION_OUTPUT_FILENAMES}
)
_POST_PREDICTION_LABEL_FILES = frozenset(
    {
        *_PRE_PREDICTION_LABEL_FILES,
        *_PREDICTION_OUTPUT_FILENAMES,
        "prediction_commitment.json",
    }
)
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
_FIT_COMMITMENT_FILENAMES = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "truth_commitments.json",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
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
_MODEL_STATE_COMMITMENT_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "git_commit",
        "files",
        "created_utc",
    }
)
_MODEL_STATE_COMMITMENT_FILENAMES = (
    "fit_commitment.json",
    "center_state_checkpoint.json",
    "risk_state_checkpoint.json",
    "training_manifest.json",
    "calibration_manifest.json",
    "model_state.json",
)


class V015RunnerError(RuntimeError):
    """Raised when the formal lifecycle cannot advance without a deviation."""


@dataclass(frozen=True)
class FormalRunPaths:
    repo_root: Path
    label_free_root: Path
    sealed_truth_root: Path
    score_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        repo_root: str | Path,
        label_free_root: str | Path,
        sealed_truth_root: str | Path,
        score_root: str | Path,
    ) -> FormalRunPaths:
        repo = Path(repo_root).resolve()
        try:
            label, sealed = assert_separate_truth_roots(
                label_free_root, sealed_truth_root
            )
        except V015ArtifactError as exc:
            raise V015RunnerError(
                "Label-free and sealed-truth roots must be disjoint trees"
            ) from exc
        score = Path(score_root).resolve()
        for other, label_name in (
            (label, "label-free"),
            (sealed, "sealed-truth"),
        ):
            try:
                common = Path(os.path.commonpath((score, other)))
            except ValueError:
                continue
            if common in {score, other}:
                raise V015RunnerError(
                    f"Score and {label_name} roots must be disjoint trees"
                )
        return cls(repo, label, sealed, score)

    @property
    def ledger_path(self) -> Path:
        return self.label_free_root / "exposure_log.jsonl"

    @property
    def truth_commitment_path(self) -> Path:
        return self.label_free_root / "truth_commitments.json"

    @property
    def prediction_commitment_path(self) -> Path:
        return self.label_free_root / "prediction_commitment.json"


@dataclass(frozen=True)
class FormalRunResult:
    attempt_id: str
    git_commit: str
    truth_commitment_byte_sha256: str
    prediction_commitment_byte_sha256: str
    score_status: object
    wall_time_seconds: float


@dataclass(frozen=True)
class _TrainingArtifacts:
    center: CenterDevelopmentState
    risk: RiskDevelopmentState
    calibration: CalibrationDevelopmentState
    decoded_model: DecodedModelState
    center_input_hashes: Mapping[str, str]
    risk_input_hashes: Mapping[str, str]
    calibration_input_hashes: Mapping[str, str]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _byte_count_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _sha256_path(path: Path) -> str:
    return _byte_count_and_sha256(path)[1]


def _freeze_record_commits() -> tuple[str, str]:
    try:
        payload = json.loads(_FREEZE_RECORD.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V015RunnerError("Implementation freeze record is unreadable") from exc
    protocol_commit = payload.get("freeze_commit")
    implementation_commit = payload.get("implementation_source_commit")
    if (
        protocol_commit != _PROTOCOL_FREEZE_GIT_COMMIT
        or not isinstance(implementation_commit, str)
        or len(implementation_commit) not in {40, 64}
        or any(
            character not in "0123456789abcdef" for character in implementation_commit
        )
    ):
        raise V015RunnerError(
            "Implementation freeze record commit identities are invalid"
        )
    return protocol_commit, implementation_commit


def _exclusive_create_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise V015RunnerError(f"Formal artifact already exists: {path}") from exc
    except OSError as exc:
        raise V015RunnerError(
            f"Could not atomically create formal artifact: {path}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.read_bytes() != raw:
        raise V015RunnerError(f"Formal artifact bytes changed after write: {path}")


def _safe_direct_child(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise V015RunnerError(f"Unsafe artifact filename: {filename!r}")
    child = (root.resolve() / filename).resolve()
    if child.parent != root.resolve():
        raise V015RunnerError(f"Artifact escaped its root: {filename}")
    return child


def _require_empty_score_root(score_root: Path) -> None:
    if score_root.exists() and not score_root.is_dir():
        raise V015RunnerError("Score root exists but is not a directory")
    if score_root.exists() and any(score_root.iterdir()):
        raise V015RunnerError("Score root must be absent or empty before scoring")


def _is_reparse_entry(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise V015RunnerError(f"Cannot inspect formal path: {path}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _require_fresh_root(root: Path, *, context: str) -> None:
    if root.exists():
        if _is_reparse_entry(root):
            raise V015RunnerError(f"{context} root cannot be a reparse point")
        if not root.is_dir():
            raise V015RunnerError(f"{context} root is not a directory")
        if any(root.iterdir()):
            raise V015RunnerError(f"{context} root must be absent or completely empty")


def _require_exact_root_files(
    root: Path,
    expected: Sequence[str] | frozenset[str],
    *,
    context: str,
) -> None:
    if not root.is_dir() or _is_reparse_entry(root):
        raise V015RunnerError(f"{context} root is not a physical directory")
    entries = tuple(root.iterdir())
    observed = {entry.name for entry in entries}
    if observed != set(expected):
        raise V015RunnerError(
            f"{context} root membership changed: "
            f"missing={sorted(set(expected) - observed)}, "
            f"unexpected={sorted(observed - set(expected))}"
        )
    for entry in entries:
        if _is_reparse_entry(entry) or not entry.is_file():
            raise V015RunnerError(
                f"{context} artifact is not a direct regular file: {entry}"
            )


def _write_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_keys: frozenset[str],
) -> str:
    if set(payload) != expected_keys:
        raise V015RunnerError(f"{path.name} checkpoint keys changed")
    if (
        payload.get("protocol_id") != FROZEN_PROTOCOL_ID
        or payload.get("config_sha256") != FROZEN_CONFIG_BYTE_SHA256
    ):
        raise V015RunnerError(f"{path.name} checkpoint identity changed")
    raw = canonical_json_bytes(payload)
    _exclusive_create_bytes(path, raw)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V015RunnerError(f"{path.name} checkpoint is invalid JSON") from exc
    if set(decoded) != expected_keys or canonical_json_bytes(decoded) != raw:
        raise V015RunnerError(f"{path.name} checkpoint is not canonical")
    return _sha256_bytes(raw)


def _hash_named_files(root: Path, filenames: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in filenames:
        path = _safe_direct_child(root, filename)
        if not path.is_file():
            raise V015RunnerError(f"Required formal input is absent: {path}")
        hashes[filename] = _sha256_path(path)
    return hashes


def _fit_commitment_entry(
    path: Path,
    *,
    contract: FrozenArtifactContract,
) -> dict[str, object]:
    if path.suffix != ".csv":
        row_count = 1
    else:
        schema = contract.csv_schema(path.name)
        if schema.required_rows is not None:
            row_count = schema.required_rows
        else:
            member_count = sum(contract.partition_member_counts.values())
            variant_count = len(FROZEN_VARIANT_KEYS)
            if path.name == "member_fit_diagnostics.csv":
                row_count = member_count * variant_count
            elif path.name == "member_forecast_bundle.csv":
                row_count = member_count * variant_count * len(contract.forecast_days)
            else:
                raise V015RunnerError(
                    f"No frozen row-count rule exists for {path.name}"
                )
    byte_count, byte_sha256 = _byte_count_and_sha256(path)
    return {
        "path": path.name,
        "row_count": row_count,
        "byte_count": byte_count,
        "byte_sha256": byte_sha256,
    }


def _create_fit_commitment(
    *,
    paths: FormalRunPaths,
    contract: FrozenArtifactContract,
    identity: FormalAttemptIdentity,
) -> str:
    payload = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "git_commit": identity.git_commit,
        "worker_count": 6,
        "files": [
            _fit_commitment_entry(
                _safe_direct_child(paths.label_free_root, filename),
                contract=contract,
            )
            for filename in _FIT_COMMITMENT_FILENAMES
        ],
        "created_utc": _utc_now(),
    }
    return _write_checkpoint(
        paths.label_free_root / "fit_commitment.json",
        payload,
        expected_keys=_FIT_COMMITMENT_KEYS,
    )


def _verify_fit_commitment(
    *,
    label_free_root: Path,
    contract: FrozenArtifactContract,
    identity: FormalAttemptIdentity,
    expected_byte_sha256: str | None = None,
) -> str:
    path = label_free_root / "fit_commitment.json"
    raw = path.read_bytes()
    observed_hash = _sha256_bytes(raw)
    if expected_byte_sha256 is not None and observed_hash != expected_byte_sha256:
        raise V015RunnerError("Fit commitment bytes changed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V015RunnerError("fit_commitment.json is invalid JSON") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _FIT_COMMITMENT_KEYS
        or canonical_json_bytes(payload) != raw
        or payload.get("protocol_id") != FROZEN_PROTOCOL_ID
        or payload.get("config_sha256") != FROZEN_CONFIG_BYTE_SHA256
        or payload.get("git_commit") != identity.git_commit
        or payload.get("worker_count") != 6
    ):
        raise V015RunnerError("fit_commitment.json identity changed")
    expected_entries = [item for item in payload.get("files", ())]
    if not isinstance(payload.get("files"), list) or len(expected_entries) != len(
        _FIT_COMMITMENT_FILENAMES
    ):
        raise V015RunnerError("fit_commitment.json file registry changed")
    for filename, declared in zip(
        _FIT_COMMITMENT_FILENAMES, expected_entries, strict=True
    ):
        if (
            not isinstance(declared, Mapping)
            or set(declared) != {"path", "row_count", "byte_count", "byte_sha256"}
            or declared.get("path") != filename
            or isinstance(declared.get("row_count"), bool)
            or not isinstance(declared.get("row_count"), int)
            or int(declared["row_count"]) < 1
        ):
            raise V015RunnerError("fit_commitment.json contains invalid file metadata")
        committed_path = _safe_direct_child(label_free_root, filename)
        byte_count, byte_sha256 = _byte_count_and_sha256(committed_path)
        if (
            declared.get("byte_count") != byte_count
            or declared.get("byte_sha256") != byte_sha256
        ):
            raise V015RunnerError(
                "A label-free fit input or fitted table changed after commitment"
            )
    if tuple(item["path"] for item in expected_entries) != (_FIT_COMMITMENT_FILENAMES):
        raise V015RunnerError("fit_commitment.json file order changed")
    return observed_hash


def _state_file_entry(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.name,
        "row_count": 1,
        "byte_count": len(raw),
        "byte_sha256": _sha256_bytes(raw),
    }


def _create_model_state_commitment(
    *,
    paths: FormalRunPaths,
    identity: FormalAttemptIdentity,
) -> str:
    payload = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "git_commit": identity.git_commit,
        "files": [
            _state_file_entry(_safe_direct_child(paths.label_free_root, filename))
            for filename in _MODEL_STATE_COMMITMENT_FILENAMES
        ],
        "created_utc": _utc_now(),
    }
    return _write_checkpoint(
        paths.label_free_root / "model_state_commitment.json",
        payload,
        expected_keys=_MODEL_STATE_COMMITMENT_KEYS,
    )


def _verify_model_state_commitment(
    *,
    label_free_root: Path,
    identity: FormalAttemptIdentity,
    expected_byte_sha256: str | None = None,
) -> str:
    path = label_free_root / "model_state_commitment.json"
    raw = path.read_bytes()
    observed_hash = _sha256_bytes(raw)
    if expected_byte_sha256 is not None and observed_hash != expected_byte_sha256:
        raise V015RunnerError("Model-state commitment bytes changed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V015RunnerError("model_state_commitment.json is invalid JSON") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _MODEL_STATE_COMMITMENT_KEYS
        or canonical_json_bytes(payload) != raw
        or payload.get("protocol_id") != FROZEN_PROTOCOL_ID
        or payload.get("config_sha256") != FROZEN_CONFIG_BYTE_SHA256
        or payload.get("git_commit") != identity.git_commit
        or not isinstance(payload.get("files"), list)
    ):
        raise V015RunnerError("Model-state commitment identity changed")
    expected_entries = [
        _state_file_entry(_safe_direct_child(label_free_root, filename))
        for filename in _MODEL_STATE_COMMITMENT_FILENAMES
    ]
    if payload["files"] != expected_entries:
        raise V015RunnerError(
            "A frozen state artifact changed after model-state commitment"
        )
    return observed_hash


def _verify_ledger_artifact(
    *,
    ledger_path: Path,
    contract: FrozenArtifactContract,
    identity: FormalAttemptIdentity,
    phase: str,
    artifact_path: Path,
) -> str:
    states = validate_formal_exposure_log(ledger_path, contract)
    try:
        progress = states[identity.attempt_id]
    except KeyError as exc:
        raise V015RunnerError("Attempt is absent from exposure ledger") from exc
    if progress.identity != identity:
        raise V015RunnerError("Ledger artifact identity changed")
    return verify_phase_artifact_commitment(
        progress,
        phase=phase,
        artifact_path=artifact_path,
    )


def _identity(
    attempt_id: str, environment: FormalEnvironmentIdentity
) -> FormalAttemptIdentity:
    return FormalAttemptIdentity(
        attempt_id=attempt_id,
        git_commit=environment.git_commit,
        config_byte_sha256=environment.config_byte_sha256,
    )


def _append_phase(
    *,
    paths: FormalRunPaths,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    phase: str,
    exit_status: str,
    truth_hash: str | None,
    prediction_hash: str | None,
    message: str,
    created_utc: str | None = None,
) -> None:
    append_formal_exposure_event(
        path=paths.ledger_path,
        identity=identity,
        contract=contract,
        created_utc=created_utc or _utc_now(),
        phase=phase,
        exit_status=exit_status,
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        message=message,
    )


def _terminal_phase(
    *,
    error: BaseException,
    paths: FormalRunPaths,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    phase: str,
    truth_hash: str | None,
    prediction_hash: str | None,
) -> None:
    status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    try:
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase=phase,
            exit_status=status,
            truth_hash=truth_hash,
            prediction_hash=prediction_hash,
            message=f"Formal {phase} stage {status}.",
        )
    except BaseException as checkpoint_error:
        error.add_note(
            f"The {phase} error was preserved, but its terminal ledger "
            f"checkpoint failed: {checkpoint_error!r}"
        )
        raise error from checkpoint_error


def initialize_formal_attempt(
    *,
    paths: FormalRunPaths,
    attempt_id: str,
) -> tuple[
    FormalEnvironmentIdentity,
    FrozenArtifactContract,
    FormalAttemptIdentity,
]:
    """Verify the implementation freeze and create the first ledger event."""

    environment = verify_formal_environment(paths.repo_root)
    contract = load_artifact_contract()
    identity = _identity(attempt_id, environment)
    for root, context in (
        (paths.label_free_root, "label-free"),
        (paths.sealed_truth_root, "sealed-truth"),
        (paths.score_root, "score"),
    ):
        _require_fresh_root(root, context=context)
    if paths.ledger_path.exists():
        raise V015RunnerError(
            "A fresh formal attempt cannot overwrite an exposure ledger"
        )
    _require_empty_score_root(paths.score_root)
    _append_phase(
        paths=paths,
        identity=identity,
        contract=contract,
        phase="before_generation",
        exit_status="completed",
        truth_hash=None,
        prediction_hash=None,
        message="Clean frozen implementation verified before V0.15 generation.",
    )
    return environment, contract, identity


def _run_checked_process(arguments: Sequence[str], *, context: str) -> None:
    try:
        completed = subprocess.run(
            tuple(arguments),
            cwd=_PROJECT_ROOT,
            check=False,
        )
    except OSError as exc:
        raise V015RunnerError(f"Could not launch {context} process") from exc
    if completed.returncode != 0:
        raise V015RunnerError(
            f"{context} process exited with status {completed.returncode}"
        )


def _launch_generation_process(paths: FormalRunPaths) -> None:
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
        context="isolated generation",
    )


def _launch_prediction_process(*, label_free_root: Path, attempt_id: str) -> None:
    # Deliberately no sealed-truth argument exists in this process capability.
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
        ),
        context="truth-incapable prediction",
    )


def run_isolated_generation_stage(
    *, label_free_root: str | Path, sealed_truth_root: str | Path
) -> None:
    """Internal CLI capability for the frozen isolated generator."""

    generate_frozen_v015_artifacts(
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
    )


def _read_label_inputs(
    label_free_root: Path,
    contract: FrozenArtifactContract,
) -> dict[str, pd.DataFrame]:
    return {
        filename: read_canonical_csv(
            _safe_direct_child(label_free_root, filename),
            contract,
            formal=True,
        )
        for filename in (*_LABEL_INPUT_FILENAMES, *_FIT_FILENAMES)
    }


def _subset_partition(
    frames: Mapping[str, pd.DataFrame], partition: str
) -> dict[str, pd.DataFrame]:
    subset: dict[str, pd.DataFrame] = {}
    for filename, frame in frames.items():
        selected = frame.loc[frame["partition"].eq(partition)].reset_index(drop=True)
        if selected.empty:
            raise V015RunnerError(f"{filename} has no rows for partition {partition!r}")
        subset[filename] = selected
    return subset


def _apply_partition(
    frames: Mapping[str, pd.DataFrame],
    *,
    partition: str,
    state: object,
) -> LabelFreePipelineResult:
    selected = _subset_partition(frames, partition)
    return run_pipeline_from_fitted(
        prefix_pack=selected["prefix_pack.csv"],
        forecast_coordinates=selected["forecast_coordinates.csv"],
        operating_pack=selected["operating_pack.csv"],
        member_fit_diagnostics=selected["member_fit_diagnostics.csv"],
        member_forecast_bundle=selected["member_forecast_bundle.csv"],
        state=state,  # type: ignore[arg-type]
    )


def _plain_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _truth_content_digest(
    truth_frame: pd.DataFrame, *, partition: str, cluster_id: str
) -> str:
    columns = (
        "truth_family",
        "truth_parameters_json",
        "gamma",
        "forecast_day",
        "latent_retention_pct",
        "noisy_retention_pct",
    )
    rows = truth_frame.loc[
        truth_frame["partition"].eq(partition)
        & truth_frame["cluster_id"].eq(cluster_id),
        list(columns),
    ].sort_values("forecast_day", kind="stable")
    if len(rows) != len(FORECAST_DAYS):
        raise V015RunnerError(f"{partition}/{cluster_id} truth grid is incomplete")
    records = [
        [_plain_json_value(value) for value in row]
        for row in rows.itertuples(index=False, name=None)
    ]
    return _sha256_bytes(canonical_json_bytes({"truth_records": records}))


def _ordered_training_cluster_ids(
    pipeline: LabelFreePipelineResult,
    truth_frame: pd.DataFrame,
    *,
    partition: str,
    include_operating_content: bool,
) -> tuple[str, ...]:
    content = pipeline.predictor_content_bundle.loc[
        pipeline.predictor_content_bundle["partition"].eq(partition)
    ].set_index("cluster_id")
    if content.empty or not content.index.is_unique:
        raise V015RunnerError(f"{partition} predictor-content rows are incomplete")
    keys: list[tuple[tuple[str, ...], str]] = []
    for raw_cluster_id, row in content.iterrows():
        cluster_id = str(raw_cluster_id)
        ordering = [str(row["arm_a_content_sha256"])]
        if include_operating_content:
            ordering.extend(
                (
                    str(row["arm_b_content_sha256"]),
                    str(row["placebo_content_sha256"]),
                )
            )
        ordering.append(
            _truth_content_digest(
                truth_frame,
                partition=partition,
                cluster_id=cluster_id,
            )
        )
        keys.append((tuple(ordering), cluster_id))
    # The scientific sort key contains no opaque identity.  Exact ties have
    # byte-identical predictor and truth rows, so their internal order cannot
    # change any fitted numeric input.
    keys.sort(key=lambda item: item[0])
    return tuple(cluster_id for _, cluster_id in keys)


def _forecast_matrix(
    frame: pd.DataFrame,
    *,
    partition: str,
    cluster_ids: Sequence[str],
    value_column: str,
) -> np.ndarray:
    selected = frame.loc[frame["partition"].eq(partition)]
    records: list[np.ndarray] = []
    expected_days = tuple(float(value) for value in FORECAST_DAYS)
    for cluster_id in cluster_ids:
        rows = selected.loc[selected["cluster_id"].eq(cluster_id)].sort_values(
            "forecast_day", kind="stable"
        )
        days = tuple(
            pd.to_numeric(rows["forecast_day"], errors="coerce").to_numpy(float)
        )
        if days != expected_days:
            raise V015RunnerError(
                f"{partition}/{cluster_id} lacks the exact forecast grid"
            )
        values = pd.to_numeric(rows[value_column], errors="coerce").to_numpy(float)
        if values.shape != (len(FORECAST_DAYS),):
            raise V015RunnerError(
                f"{partition}/{cluster_id}/{value_column} has wrong shape"
            )
        records.append(values)
    if set(selected["cluster_id"].astype(str)) != set(cluster_ids):
        raise V015RunnerError(
            f"{partition}/{value_column} contains unexpected clusters"
        )
    return np.vstack(records)


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
        raise V015RunnerError(f"{partition} feature rows are incomplete")
    return rows.reset_index()


def _input_hashes(
    *,
    paths: FormalRunPaths,
    label_filenames: Sequence[str],
    truth_filename: str,
    checkpoint_filenames: Sequence[str] = (),
) -> dict[str, str]:
    hashes = _hash_named_files(paths.label_free_root, label_filenames)
    hashes[truth_filename] = _sha256_path(
        _safe_direct_child(paths.sealed_truth_root, truth_filename)
    )
    hashes.update(_hash_named_files(paths.label_free_root, checkpoint_filenames))
    return hashes


def _fit_and_commit_structure_library(
    *, paths: FormalRunPaths, contract: FrozenArtifactContract
) -> dict[str, pd.DataFrame]:
    prefix = read_canonical_csv(
        paths.label_free_root / "prefix_pack.csv", contract, formal=True
    )
    coordinates = read_canonical_csv(
        paths.label_free_root / "forecast_coordinates.csv",
        contract,
        formal=True,
    )
    fitted = fit_structure_library_formal(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
    )
    write_canonical_csv(
        paths.label_free_root / "member_fit_diagnostics.csv",
        fitted.member_fit_diagnostics,
        contract,
        formal=True,
    )
    write_canonical_csv(
        paths.label_free_root / "member_forecast_bundle.csv",
        fitted.member_forecast_bundle,
        contract,
        formal=True,
    )
    # The formal bundle has 4,093,600 rows. Release its in-memory construction
    # before the canonical committed files are read back and verified.
    del fitted, prefix, coordinates
    return _read_label_inputs(paths.label_free_root, contract)


def _fit_center_stage(
    *,
    paths: FormalRunPaths,
    frames: Mapping[str, pd.DataFrame],
    center_truth: pd.DataFrame,
) -> tuple[CenterDevelopmentState, Mapping[str, str], str]:
    partition = "center_development"
    probe = _apply_partition(
        frames,
        partition=partition,
        state=make_probe_state(1.0),
    )
    cluster_ids = _ordered_training_cluster_ids(
        probe,
        center_truth,
        partition=partition,
        include_operating_content=False,
    )
    library = _forecast_matrix(
        probe.prediction_bundle,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="center_forecast_pct",
    )
    sqrt = _forecast_matrix(
        probe.prediction_bundle,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="sqrt_time_forecast_pct",
    )
    latent = _forecast_matrix(
        center_truth,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="latent_retention_pct",
    )
    state = fit_center_development_state(
        library_forecasts_pct=library,
        sqrt_forecasts_pct=sqrt,
        latent_targets_pct=latent,
    )
    input_hashes = _input_hashes(
        paths=paths,
        label_filenames=(*_LABEL_INPUT_FILENAMES, *_FIT_FILENAMES),
        truth_filename="center_development_truth.csv",
        checkpoint_filenames=("fit_commitment.json",),
    )
    payload = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "state_kind": "center_development",
        "center_state_sha256": center_state_sha256(state),
        "center_beta": state.beta,
        "development_cluster_count": state.development_cluster_count,
        "forecast_horizon_count": state.forecast_horizon_count,
        "ridge_penalty": state.ridge_penalty,
        "completeness_rule": state.completeness_rule,
        "input_byte_hashes": dict(sorted(input_hashes.items())),
        "created_utc": _utc_now(),
    }
    checkpoint_hash = _write_checkpoint(
        paths.label_free_root / "center_state_checkpoint.json",
        payload,
        expected_keys=_CENTER_CHECKPOINT_KEYS,
    )
    if payload["center_state_sha256"] != center_state_sha256(state):
        raise V015RunnerError("Center state changed after checkpoint creation")
    return state, input_hashes, checkpoint_hash


def _fit_risk_stage(
    *,
    paths: FormalRunPaths,
    contract: FrozenArtifactContract,
    frames: Mapping[str, pd.DataFrame],
    center: CenterDevelopmentState,
    center_input_hashes: Mapping[str, str],
    center_checkpoint_hash: str,
    risk_truth: pd.DataFrame,
) -> tuple[RiskDevelopmentState, Mapping[str, str], str]:
    center_checkpoint_path = paths.label_free_root / "center_state_checkpoint.json"
    if _sha256_path(center_checkpoint_path) != center_checkpoint_hash:
        raise V015RunnerError("Center checkpoint changed before risk reveal")
    partition = "risk_development"
    probe = _apply_partition(
        frames,
        partition=partition,
        state=make_probe_state(center.beta),
    )
    cluster_ids = _ordered_training_cluster_ids(
        probe,
        risk_truth,
        partition=partition,
        include_operating_content=True,
    )
    features = _feature_rows(
        probe.feature_bundle,
        partition=partition,
        cluster_ids=cluster_ids,
    )
    predictions = _forecast_matrix(
        probe.prediction_bundle,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="center_forecast_pct",
    )
    targets = _forecast_matrix(
        risk_truth,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="latent_retention_pct",
    )
    planned = np.asarray(
        [
            stress_index(*(float(row[name]) for name in REAL_OPERATING_FIELDS[4:]))
            for _, row in features.iterrows()
        ],
        dtype=np.float64,
    )
    state = fit_risk_development_state(
        prefix_features=features.loc[:, list(PREFIX_FEATURE_NAMES)].to_numpy(float),
        visible_stress_features=features.loc[:, list(REAL_OPERATING_FIELDS)].to_numpy(
            float
        ),
        placebo_features=features.loc[
            :,
            [
                name
                for name in PLACEBO_FEATURE_NAMES
                if name not in PREFIX_FEATURE_NAMES
            ],
        ].to_numpy(float),
        planned_stress_index=planned,
        frozen_center_25y_pct=predictions[:, -1],
        latent_target_25y_pct=targets[:, -1],
        common_pool_eligible=features["hard_eligible"].tolist(),
    )
    input_hashes = _input_hashes(
        paths=paths,
        label_filenames=(*_LABEL_INPUT_FILENAMES, *_FIT_FILENAMES),
        truth_filename="risk_development_truth.csv",
        checkpoint_filenames=(
            "fit_commitment.json",
            "center_state_checkpoint.json",
        ),
    )
    manifest = build_training_manifest(
        center_development_input_hashes=center_input_hashes,
        risk_development_input_hashes=input_hashes,
        center_state=center,
        risk_state=state,
        created_utc=_utc_now(),
    )
    manifest_path = paths.label_free_root / "training_manifest.json"
    write_canonical_json(manifest_path, manifest, contract)
    verify_training_manifest_state_hashes(
        manifest, center_state=center, risk_state=state
    )
    manifest_hash = _sha256_path(manifest_path)
    payload = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "state_kind": "risk_development",
        "center_checkpoint_byte_sha256": center_checkpoint_hash,
        "training_manifest_byte_sha256": manifest_hash,
        "risk_state_sha256": risk_state_sha256(state),
        "development_cluster_count": state.development_cluster_count,
        "eligible_cluster_count": state.eligible_cluster_count,
        "positive_label_count": state.positive_label_count,
        "negative_label_count": state.negative_label_count,
        "input_byte_hashes": dict(sorted(input_hashes.items())),
        "created_utc": _utc_now(),
    }
    checkpoint_hash = _write_checkpoint(
        paths.label_free_root / "risk_state_checkpoint.json",
        payload,
        expected_keys=_RISK_CHECKPOINT_KEYS,
    )
    if payload["risk_state_sha256"] != risk_state_sha256(state):
        raise V015RunnerError("Risk state changed after checkpoint creation")
    return state, input_hashes, checkpoint_hash


def _calibration_probe_state(
    center: CenterDevelopmentState, risk: RiskDevelopmentState
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
    selected = prefix_pack.loc[prefix_pack["partition"].eq(partition)]
    values: list[float] = []
    for cluster_id in cluster_ids:
        rows = selected.loc[selected["cluster_id"].eq(cluster_id)].sort_values(
            "prefix_day", kind="stable"
        )
        if len(rows) != 12:
            raise V015RunnerError(
                f"{partition}/{cluster_id} prefix cardinality changed"
            )
        endpoint = float(rows.iloc[-1]["observed_retention_pct"])
        if not math.isfinite(endpoint):
            raise V015RunnerError("Persistence endpoint is nonfinite")
        values.append(endpoint)
    return np.repeat(
        np.asarray(values, dtype=np.float64)[:, None],
        len(FORECAST_DAYS),
        axis=1,
    )


def _fit_calibration_stage(
    *,
    paths: FormalRunPaths,
    contract: FrozenArtifactContract,
    environment: FormalEnvironmentIdentity,
    frames: Mapping[str, pd.DataFrame],
    center: CenterDevelopmentState,
    risk: RiskDevelopmentState,
    center_input_hashes: Mapping[str, str],
    risk_input_hashes: Mapping[str, str],
    risk_checkpoint_hash: str,
    calibration_truth: pd.DataFrame,
) -> tuple[CalibrationDevelopmentState, Mapping[str, str], DecodedModelState]:
    risk_checkpoint_path = paths.label_free_root / "risk_state_checkpoint.json"
    if _sha256_path(risk_checkpoint_path) != risk_checkpoint_hash:
        raise V015RunnerError("Risk checkpoint changed before calibration reveal")
    partition = "calibration"
    probe = _apply_partition(
        frames,
        partition=partition,
        state=_calibration_probe_state(center, risk),
    )
    cluster_ids = _ordered_training_cluster_ids(
        probe,
        calibration_truth,
        partition=partition,
        include_operating_content=True,
    )
    features = _feature_rows(
        probe.feature_bundle,
        partition=partition,
        cluster_ids=cluster_ids,
    )
    predictions = probe.prediction_bundle
    center_forecasts = _forecast_matrix(
        predictions,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="center_forecast_pct",
    )
    targets = _forecast_matrix(
        calibration_truth,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="latent_retention_pct",
    )
    lower = _forecast_matrix(
        predictions,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="base_interval_lower_pct",
    )
    upper = _forecast_matrix(
        predictions,
        partition=partition,
        cluster_ids=cluster_ids,
        value_column="base_interval_upper_pct",
    )
    mean_baselines = {
        "target_prefix_persistence": _persistence_matrix(
            frames["prefix_pack.csv"],
            partition=partition,
            cluster_ids=cluster_ids,
        ),
        "target_prefix_sqrt_time": _forecast_matrix(
            predictions,
            partition=partition,
            cluster_ids=cluster_ids,
            value_column="sqrt_time_forecast_pct",
        ),
        "target_prefix_bounded_power_law": _forecast_matrix(
            predictions,
            partition=partition,
            cluster_ids=cluster_ids,
            value_column="bounded_power_forecast_pct",
        ),
    }
    calibration = fit_calibration_development_state(
        risk_state=risk,
        prefix_features=features.loc[:, list(PREFIX_FEATURE_NAMES)].to_numpy(float),
        visible_stress_features=features.loc[:, list(REAL_OPERATING_FIELDS)].to_numpy(
            float
        ),
        frozen_center_25y_pct=center_forecasts[:, -1],
        latent_targets_pct=targets,
        base_interval_lower_pct=lower,
        base_interval_upper_pct=upper,
        mean_baseline_forecasts_pct=mean_baselines,
    )
    calibration_input_hashes = _input_hashes(
        paths=paths,
        label_filenames=(*_LABEL_INPUT_FILENAMES, *_FIT_FILENAMES),
        truth_filename="calibration_truth.csv",
        checkpoint_filenames=(
            "fit_commitment.json",
            "center_state_checkpoint.json",
            "risk_state_checkpoint.json",
            "training_manifest.json",
        ),
    )
    calibration_manifest = build_calibration_manifest(
        calibration_input_hashes=calibration_input_hashes,
        calibration_state=calibration,
        created_utc=_utc_now(),
    )
    calibration_manifest_path = paths.label_free_root / "calibration_manifest.json"
    write_canonical_json(calibration_manifest_path, calibration_manifest, contract)
    verify_calibration_manifest_state_hashes(
        calibration_manifest, calibration_state=calibration
    )
    training_state = FrozenTrainingState(center, risk, calibration)
    model_raw = serialize_model_state_json(
        training_state,
        center_development_input_hashes=center_input_hashes,
        risk_development_input_hashes=risk_input_hashes,
        calibration_input_hashes=calibration_input_hashes,
        software_versions=default_software_versions(),
        created_utc=_utc_now(),
    )
    model_path = paths.label_free_root / "model_state.json"
    _exclusive_create_bytes(model_path, model_raw)
    decoded = deserialize_model_state_json(model_path.read_bytes())
    if decoded.training_state != training_state:
        raise V015RunnerError(
            "Serialized model state differs after mandatory deserialization"
        )
    return calibration, calibration_input_hashes, decoded


def _fit_training_stages(
    *,
    paths: FormalRunPaths,
    environment: FormalEnvironmentIdentity,
    contract: FrozenArtifactContract,
    identity: FormalAttemptIdentity,
    truth_hash: str,
) -> _TrainingArtifacts:
    _append_phase(
        paths=paths,
        identity=identity,
        contract=contract,
        phase="label_free_fit_committed",
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message=(
            "Six-worker fit-once structure-library stage started without opening truth."
        ),
    )
    try:
        frames = _fit_and_commit_structure_library(paths=paths, contract=contract)
        fit_commitment_hash = _create_fit_commitment(
            paths=paths,
            contract=contract,
            identity=identity,
        )
        _verify_fit_commitment(
            label_free_root=paths.label_free_root,
            contract=contract,
            identity=identity,
            expected_byte_sha256=fit_commitment_hash,
        )
        _require_exact_root_files(
            paths.label_free_root,
            _POST_FIT_LABEL_FILES,
            context="post-fit label-free",
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase="label_free_fit_committed",
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(
                "label_free_fit_committed", fit_commitment_hash
            ),
        )
        if (
            _verify_ledger_artifact(
                ledger_path=paths.ledger_path,
                contract=contract,
                identity=identity,
                phase="label_free_fit_committed",
                artifact_path=(paths.label_free_root / "fit_commitment.json"),
            )
            != fit_commitment_hash
        ):
            raise V015RunnerError("Ledger fit commitment differs from committed bytes")
    except BaseException as exc:
        _terminal_phase(
            error=exc,
            paths=paths,
            identity=identity,
            contract=contract,
            phase="label_free_fit_committed",
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise

    _verify_fit_commitment(
        label_free_root=paths.label_free_root,
        contract=contract,
        identity=identity,
        expected_byte_sha256=fit_commitment_hash,
    )
    center_truth = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=contract,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="center_truth_opened",
        created_utc=_utc_now(),
    )["center_development_truth.csv"]
    try:
        center, center_hashes, center_checkpoint_hash = _fit_center_stage(
            paths=paths,
            frames=frames,
            center_truth=center_truth,
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase="center_state_committed",
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(
                "center_state_committed", center_checkpoint_hash
            ),
        )
        _require_exact_root_files(
            paths.label_free_root,
            _POST_CENTER_LABEL_FILES,
            context="post-center label-free",
        )
    except BaseException as exc:
        _terminal_phase(
            error=exc,
            paths=paths,
            identity=identity,
            contract=contract,
            phase="center_state_committed",
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise

    _verify_fit_commitment(
        label_free_root=paths.label_free_root,
        contract=contract,
        identity=identity,
        expected_byte_sha256=fit_commitment_hash,
    )
    center_ledger_hash = _verify_ledger_artifact(
        ledger_path=paths.ledger_path,
        contract=contract,
        identity=identity,
        phase="center_state_committed",
        artifact_path=(paths.label_free_root / "center_state_checkpoint.json"),
    )
    if (
        _sha256_path(paths.label_free_root / "center_state_checkpoint.json")
        != center_ledger_hash
    ):
        raise V015RunnerError("Center checkpoint differs from its ledger commitment")
    risk_truth = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=contract,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="risk_truth_opened",
        created_utc=_utc_now(),
    )["risk_development_truth.csv"]
    try:
        risk, risk_hashes, risk_checkpoint_hash = _fit_risk_stage(
            paths=paths,
            contract=contract,
            frames=frames,
            center=center,
            center_input_hashes=center_hashes,
            center_checkpoint_hash=center_checkpoint_hash,
            risk_truth=risk_truth,
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase="risk_state_committed",
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(
                "risk_state_committed", risk_checkpoint_hash
            ),
        )
        _require_exact_root_files(
            paths.label_free_root,
            _POST_RISK_LABEL_FILES,
            context="post-risk label-free",
        )
    except BaseException as exc:
        _terminal_phase(
            error=exc,
            paths=paths,
            identity=identity,
            contract=contract,
            phase="risk_state_committed",
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise

    _verify_fit_commitment(
        label_free_root=paths.label_free_root,
        contract=contract,
        identity=identity,
        expected_byte_sha256=fit_commitment_hash,
    )
    risk_ledger_hash = _verify_ledger_artifact(
        ledger_path=paths.ledger_path,
        contract=contract,
        identity=identity,
        phase="risk_state_committed",
        artifact_path=(paths.label_free_root / "risk_state_checkpoint.json"),
    )
    if (
        _sha256_path(paths.label_free_root / "risk_state_checkpoint.json")
        != risk_ledger_hash
    ):
        raise V015RunnerError("Risk checkpoint differs from its ledger commitment")
    calibration_truth = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=contract,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="calibration_truth_opened",
        created_utc=_utc_now(),
    )["calibration_truth.csv"]
    try:
        calibration, calibration_hashes, decoded = _fit_calibration_stage(
            paths=paths,
            contract=contract,
            environment=environment,
            frames=frames,
            center=center,
            risk=risk,
            center_input_hashes=center_hashes,
            risk_input_hashes=risk_hashes,
            risk_checkpoint_hash=risk_checkpoint_hash,
            calibration_truth=calibration_truth,
        )
        _verify_fit_commitment(
            label_free_root=paths.label_free_root,
            contract=contract,
            identity=identity,
            expected_byte_sha256=fit_commitment_hash,
        )
        model_commitment_hash = _create_model_state_commitment(
            paths=paths,
            identity=identity,
        )
        _verify_model_state_commitment(
            label_free_root=paths.label_free_root,
            identity=identity,
            expected_byte_sha256=model_commitment_hash,
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase="model_state_committed",
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=phase_commitment_message(
                "model_state_committed", model_commitment_hash
            ),
        )
        _require_exact_root_files(
            paths.label_free_root,
            _PRE_PREDICTION_LABEL_FILES,
            context="post-model label-free",
        )
    except BaseException as exc:
        _terminal_phase(
            error=exc,
            paths=paths,
            identity=identity,
            contract=contract,
            phase="model_state_committed",
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return _TrainingArtifacts(
        center=center,
        risk=risk,
        calibration=calibration,
        decoded_model=decoded,
        center_input_hashes=center_hashes,
        risk_input_hashes=risk_hashes,
        calibration_input_hashes=calibration_hashes,
    )


def run_formal_prediction_stage(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    repo_root: str | Path = _PROJECT_ROOT,
) -> None:
    """Apply committed fits/state in a process with no sealed-truth capability."""

    root = Path(label_free_root).resolve()
    environment = verify_formal_environment(repo_root)
    contract = load_artifact_contract()
    identity = _identity(attempt_id, environment)
    states = validate_formal_exposure_log(root / "exposure_log.jsonl", contract)
    try:
        progress = states[attempt_id]
    except KeyError as exc:
        raise V015RunnerError("Prediction attempt is absent from ledger") from exc
    if (
        progress.identity != identity
        or progress.completed_phase != "model_state_committed"
        or progress.pending_phase != "prediction_started"
        or progress.terminal_failed
    ):
        raise V015RunnerError(
            "Prediction process lacks the exact pending ledger capability"
        )
    _require_exact_root_files(
        root,
        _PRE_PREDICTION_LABEL_FILES,
        context="prediction-capability label-free",
    )
    _verify_fit_commitment(
        label_free_root=root,
        contract=contract,
        identity=identity,
        expected_byte_sha256=_verify_ledger_artifact(
            ledger_path=root / "exposure_log.jsonl",
            contract=contract,
            identity=identity,
            phase="label_free_fit_committed",
            artifact_path=root / "fit_commitment.json",
        ),
    )
    _verify_model_state_commitment(
        label_free_root=root,
        identity=identity,
        expected_byte_sha256=_verify_ledger_artifact(
            ledger_path=root / "exposure_log.jsonl",
            contract=contract,
            identity=identity,
            phase="model_state_committed",
            artifact_path=root / "model_state_commitment.json",
        ),
    )
    frames = _read_label_inputs(root, contract)
    decoded = deserialize_model_state_json((root / "model_state.json").read_bytes())
    output = run_pipeline_from_fitted(
        prefix_pack=frames["prefix_pack.csv"],
        forecast_coordinates=frames["forecast_coordinates.csv"],
        operating_pack=frames["operating_pack.csv"],
        member_fit_diagnostics=frames["member_fit_diagnostics.csv"],
        member_forecast_bundle=frames["member_forecast_bundle.csv"],
        state=decoded.frozen_label_free_state,
    )
    for filename, frame in (
        ("prediction_bundle.csv", output.prediction_bundle),
        ("risk_bundle.csv", output.primary_risk_bundle),
        ("decision_bundle.csv", output.decision_bundle),
    ):
        path = root / filename
        schema = contract.csv_schema(filename)
        _exclusive_create_bytes(
            path,
            canonical_csv_bytes(frame, schema, contract, formal=True),
        )
    del decoded, frame, frames, output
    _require_exact_root_files(
        root,
        _PREDICTION_NO_COMMITMENT_LABEL_FILES,
        context="completed prediction label-free",
    )
    read_prediction_artifact_bundle(root, contract, formal=True)


def _prediction_and_commitment(
    *,
    paths: FormalRunPaths,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    truth_hash: str,
) -> str:
    _verify_model_state_commitment(
        label_free_root=paths.label_free_root,
        identity=identity,
        expected_byte_sha256=_verify_ledger_artifact(
            ledger_path=paths.ledger_path,
            contract=contract,
            identity=identity,
            phase="model_state_committed",
            artifact_path=(paths.label_free_root / "model_state_commitment.json"),
        ),
    )
    _require_exact_root_files(
        paths.label_free_root,
        _PRE_PREDICTION_LABEL_FILES,
        context="pre-prediction label-free",
    )
    _append_phase(
        paths=paths,
        identity=identity,
        contract=contract,
        phase="prediction_started",
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message=(
            "Truth-incapable prediction process started from committed fits "
            "and deserialized model state."
        ),
    )
    try:
        _launch_prediction_process(
            label_free_root=paths.label_free_root,
            attempt_id=identity.attempt_id,
        )
        read_prediction_artifact_bundle(paths.label_free_root, contract, formal=True)
        _verify_fit_commitment(
            label_free_root=paths.label_free_root,
            contract=contract,
            identity=identity,
        )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase="prediction_started",
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=(
                "Prediction artifacts completed without optimizer or sealed "
                "truth capability."
            ),
        )
    except BaseException as exc:
        _terminal_phase(
            error=exc,
            paths=paths,
            identity=identity,
            contract=contract,
            phase="prediction_started",
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise

    try:
        _verify_fit_commitment(
            label_free_root=paths.label_free_root,
            contract=contract,
            identity=identity,
        )
        _verify_model_state_commitment(
            label_free_root=paths.label_free_root,
            identity=identity,
            expected_byte_sha256=_verify_ledger_artifact(
                ledger_path=paths.ledger_path,
                contract=contract,
                identity=identity,
                phase="model_state_committed",
                artifact_path=(paths.label_free_root / "model_state_commitment.json"),
            ),
        )
        create_prediction_commitment(
            label_free_root=paths.label_free_root,
            commitment_path=paths.prediction_commitment_path,
            contract=contract,
            created_utc=_utc_now(),
            formal=True,
        )
        verify_prediction_commitment(
            commitment_path=paths.prediction_commitment_path,
            label_free_root=paths.label_free_root,
            contract=contract,
            formal=True,
        )
        _require_exact_root_files(
            paths.label_free_root,
            _POST_PREDICTION_LABEL_FILES,
            context="post-prediction label-free",
        )
        prediction_hash = _sha256_path(paths.prediction_commitment_path)
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase="prediction_committed",
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=prediction_hash,
            message=(
                "All predictor artifacts were verified and committed before "
                "test, audit, or matched truth access."
            ),
        )
    except BaseException as exc:
        _terminal_phase(
            error=exc,
            paths=paths,
            identity=identity,
            contract=contract,
            phase="prediction_committed",
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return prediction_hash


def _formal_artifact_metadata(
    *,
    paths: FormalRunPaths,
    contract: FrozenArtifactContract,
) -> dict[str, dict[str, object]]:
    label_names = {
        *_LABEL_INPUT_FILENAMES,
        *_FIT_FILENAMES,
        *_PREDICTION_OUTPUT_FILENAMES,
        *_STATE_FILENAMES,
        "truth_commitments.json",
        "prediction_commitment.json",
        "exposure_log.jsonl",
    }
    sealed_names = set(contract.sealed_filenames)
    expected = set(REQUIRED_FORMAL_NON_SCORE_ARTIFACTS)
    unknown = expected.difference(label_names | sealed_names)
    if unknown:
        raise V015RunnerError(
            "Scorer requires unknown formal artifact metadata: "
            + ", ".join(sorted(unknown))
        )
    fit_payload = json.loads(
        (paths.label_free_root / "fit_commitment.json").read_text(encoding="utf-8")
    )
    fit_rows = {
        str(item["path"]): int(item["row_count"])
        for item in fit_payload["files"]
        if str(item["path"]) in _FIT_FILENAMES
    }
    records: dict[str, dict[str, object]] = {}
    for filename in REQUIRED_FORMAL_NON_SCORE_ARTIFACTS:
        root = (
            paths.sealed_truth_root
            if filename in sealed_names
            else paths.label_free_root
        )
        path = _safe_direct_child(root, filename)
        if not path.is_file():
            raise V015RunnerError(f"Formal provenance input is absent: {path}")
        byte_count, byte_sha256 = _byte_count_and_sha256(path)
        if filename.endswith(".csv"):
            schema = contract.csv_schema(filename)
            if schema.required_rows is not None:
                row_count = schema.required_rows
            else:
                try:
                    row_count = fit_rows[filename]
                except KeyError as exc:
                    raise V015RunnerError(
                        f"No committed row count exists for {filename}"
                    ) from exc
        elif filename == "exposure_log.jsonl":
            raw = path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise V015RunnerError("Exposure ledger has a truncated line")
            row_count = len(raw.splitlines())
        else:
            row_count = 1
        records[filename] = {
            "path": filename,
            "row_count": row_count,
            "byte_count": byte_count,
            "byte_sha256": byte_sha256,
        }
    if set(records) != expected:
        raise V015RunnerError("Formal provenance artifact membership changed")
    return records


def _score_and_write(
    *,
    paths: FormalRunPaths,
    environment: FormalEnvironmentIdentity,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    truth_hash: str,
    prediction_hash: str,
    started_monotonic: float,
) -> object:
    _verify_fit_commitment(
        label_free_root=paths.label_free_root,
        contract=contract,
        identity=identity,
        expected_byte_sha256=_verify_ledger_artifact(
            ledger_path=paths.ledger_path,
            contract=contract,
            identity=identity,
            phase="label_free_fit_committed",
            artifact_path=paths.label_free_root / "fit_commitment.json",
        ),
    )
    _verify_model_state_commitment(
        label_free_root=paths.label_free_root,
        identity=identity,
        expected_byte_sha256=_verify_ledger_artifact(
            ledger_path=paths.ledger_path,
            contract=contract,
            identity=identity,
            phase="model_state_committed",
            artifact_path=(paths.label_free_root / "model_state_commitment.json"),
        ),
    )
    truth_frames = open_truth_for_phase(
        ledger_path=paths.ledger_path,
        identity=identity,
        contract=contract,
        commitment_path=paths.truth_commitment_path,
        sealed_truth_root=paths.sealed_truth_root,
        label_free_root=paths.label_free_root,
        phase="scoring_truth_opened",
        created_utc=_utc_now(),
        prediction_commitment_path=paths.prediction_commitment_path,
    )
    _append_phase(
        paths=paths,
        identity=identity,
        contract=contract,
        phase="scoring_completed",
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=prediction_hash,
        message="Committed-artifact-only one-shot scoring started.",
    )
    try:
        prediction_frames = read_prediction_artifact_bundle(
            paths.label_free_root, contract, formal=True
        )
        model_raw = (paths.label_free_root / "model_state.json").read_bytes()
        decoded = deserialize_model_state_json(model_raw)
        result = score_committed_artifacts(
            prediction_frames=prediction_frames,
            truth_frames=truth_frames,
            model_state_bytes=model_raw,
            decoded_model_state=decoded,
        )
        protocol = load_frozen_protocol_config(DEFAULT_V2_CONFIG_PATH)
        config = protocol.config()
        protocol_commit, implementation_commit = _freeze_record_commits()
        result = finalize_run_manifest(
            result,
            environment_identity=environment,
            implementation_freeze_record_sha256=_sha256_path(_FREEZE_RECORD),
            protocol_freeze_git_commit=protocol_commit,
            implementation_source_git_commit=implementation_commit,
            seed_roots=protocol.seed_root_map(),
            seed_derivation=config["design_partitions"]["seed_derivation"],
            prediction_worker_count=6,
            wall_time_seconds=float(time.monotonic() - started_monotonic),
            formal_artifact_metadata=_formal_artifact_metadata(
                paths=paths,
                contract=contract,
            ),
        )
        payloads = required_score_artifact_payloads(result)
        if tuple(payloads) != tuple(REQUIRED_SCORE_ARTIFACTS):
            raise V015RunnerError(
                "Scorer returned a missing, extra, or reordered artifact"
            )
        _require_empty_score_root(paths.score_root)
        for filename in REQUIRED_SCORE_ARTIFACTS:
            _exclusive_create_bytes(paths.score_root / filename, payloads[filename])
        observed = tuple(sorted(path.name for path in paths.score_root.iterdir()))
        if observed != tuple(sorted(REQUIRED_SCORE_ARTIFACTS)):
            raise V015RunnerError(
                "Persisted score artifact registry differs from the freeze"
            )
        _append_phase(
            paths=paths,
            identity=identity,
            contract=contract,
            phase="scoring_completed",
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=prediction_hash,
            message=(
                "All exact required score artifacts were canonically written "
                "without tuning."
            ),
        )
        return result.score_report.get("status")
    except BaseException as exc:
        _terminal_phase(
            error=exc,
            paths=paths,
            identity=identity,
            contract=contract,
            phase="scoring_completed",
            truth_hash=truth_hash,
            prediction_hash=prediction_hash,
        )
        raise


def run_formal_attempt(
    *,
    attempt_id: str,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
    score_root: str | Path,
    repo_root: str | Path = _PROJECT_ROOT,
) -> FormalRunResult:
    """Execute the frozen lifecycle once without a protocol override surface."""

    started = time.monotonic()
    paths = FormalRunPaths.resolve(
        repo_root=repo_root,
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
        score_root=score_root,
    )
    environment, contract, identity = initialize_formal_attempt(
        paths=paths,
        attempt_id=attempt_id,
    )
    _launch_generation_process(paths)
    _require_exact_root_files(
        paths.label_free_root,
        _POST_GENERATION_LABEL_FILES,
        context="post-generation label-free",
    )
    _require_exact_root_files(
        paths.sealed_truth_root,
        contract.sealed_filenames,
        context="post-generation sealed-truth",
    )
    states = validate_formal_exposure_log(paths.ledger_path, contract)
    progress = states.get(attempt_id)
    if (
        progress is None
        or progress.completed_phase != "truth_committed"
        or progress.pending_phase is not None
        or progress.truth_commitments_byte_sha256 is None
    ):
        raise V015RunnerError(
            "Isolated generator did not complete the truth commitment phase"
        )
    truth_hash = progress.truth_commitments_byte_sha256
    if _sha256_path(paths.truth_commitment_path) != truth_hash:
        raise V015RunnerError("Truth commitment differs after generation")
    _fit_training_stages(
        paths=paths,
        environment=environment,
        contract=contract,
        identity=identity,
        truth_hash=truth_hash,
    )
    prediction_hash = _prediction_and_commitment(
        paths=paths,
        identity=identity,
        contract=contract,
        truth_hash=truth_hash,
    )
    status = _score_and_write(
        paths=paths,
        environment=environment,
        identity=identity,
        contract=contract,
        truth_hash=truth_hash,
        prediction_hash=prediction_hash,
        started_monotonic=started,
    )
    final = validate_formal_exposure_log(paths.ledger_path, contract)[attempt_id]
    if (
        final.completed_phase != "scoring_completed"
        or final.pending_phase is not None
        or final.terminal_failed
    ):
        raise V015RunnerError("Formal attempt did not reach scoring_completed")
    return FormalRunResult(
        attempt_id=attempt_id,
        git_commit=environment.git_commit,
        truth_commitment_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        score_status=status,
        wall_time_seconds=float(time.monotonic() - started),
    )


__all__ = [
    "FormalRunPaths",
    "FormalRunResult",
    "V015RunnerError",
    "initialize_formal_attempt",
    "run_formal_attempt",
    "run_formal_prediction_stage",
    "run_isolated_generation_stage",
]
