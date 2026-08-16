"""Strict V2.4 label-free IO and commitment capabilities.

This module is the only filesystem boundary used by V2.4 fitting and
prediction.  It accepts a label-free root, never a sealed-truth or score root,
requires direct physical files, validates the shared canonical exposure
ledger, and turns verified bytes into sealed in-memory capabilities.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd

from lifetwin.experiments import calendar_long_horizon_v015_training as _v015
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    ArtifactMetadata,
    FrozenArtifactContract,
    V015ArtifactError,
    canonical_csv_bytes,
    canonical_json_bytes,
    read_canonical_csv,
    validate_prediction_artifact_bundle,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256 as V2_CONFIG_BYTE_SHA256,
    FROZEN_PROTOCOL_ID as V2_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractView,
    resolve_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v019_ledger import (
    AttemptProgress,
    V024LedgerError,
    read_exposure_log,
)
from lifetwin.experiments.calendar_long_horizon_v019_provenance import (
    V024CommittedModelStateEnvelope,
    V024ProvenanceError,
    _issue_committed_model_state_envelope_v024,
    _rehydrate_v024_training_provenance_after_strict_io,
)
from lifetwin.experiments.calendar_long_horizon_v019_state import (
    V024StateCodecError,
    _AUDIT_KEYS,
    _REQUIRED_AUDIT_COUNT_FIELDS,
    _validate_calibration_audit,
    deserialize_calibration_mask_commitment_json_v024,
    deserialize_model_state_json_v024,
)
from lifetwin.experiments.calendar_long_horizon_v019_training import (
    V024CalibrationAudit,
)


class V024IOError(ValueError):
    """Raised when a V2.4 filesystem capability cannot be proven."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_SEAL = object()
_EVIDENCE_SEAL = object()
_ARTIFACT_SET_DOMAIN = b"lifetwin-v024-prediction-artifact-set-v1\0"
_EVIDENCE_DOMAIN = b"lifetwin-v024-prediction-evidence-v1\0"
_ACTUAL_ANALYSIS_HASH_FILENAME = "actual_analysis_hash_ledger_commitment.json"

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
_POST_TRUTH_FILES = frozenset(
    {
        "generation_plan_commitment.json",
        *_LABEL_INPUTS,
        "truth_commitments.json",
        "exposure_log.jsonl",
    }
)
_GENERATION_FILES = frozenset({*_POST_TRUTH_FILES, _ACTUAL_ANALYSIS_HASH_FILENAME})
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
        *_GENERATION_FILES,
        *_FIT_OUTPUTS,
        *_MODEL_STATE_COMMITMENT_FILES,
        "model_state_commitment.json",
    }
)
_PREDICTION_FILES = frozenset({*_PRE_PREDICTION_FILES, *_PREDICTION_OUTPUTS})
_POST_PREDICTION_FILES = frozenset({*_PREDICTION_FILES, "prediction_commitment.json"})
_FIT_COMMITMENT_FILES = (
    "generation_plan_commitment.json",
    *_LABEL_INPUTS,
    "truth_commitments.json",
    _ACTUAL_ANALYSIS_HASH_FILENAME,
    *_FIT_OUTPUTS,
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
_POST_FIT_FILES = frozenset({*_GENERATION_FILES, *_FIT_OUTPUTS, "fit_commitment.json"})
_FIT_OUTPUT_FILES = frozenset({*_GENERATION_FILES, *_FIT_OUTPUTS})
_CENTER_KEYS = frozenset(
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
_RISK_KEYS = frozenset(
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
_FILE_ENTRY_KEYS = frozenset({"path", "row_count", "byte_count", "byte_sha256"})
_PREDICTION_COMMITMENT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "config_sha256",
        "attempt_id",
        "git_commit",
        "model_state_commitment_byte_sha256",
        "actual_analysis_hash_ledger_commitment_byte_sha256",
        "files",
        "row_counts",
        "artifact_set_sha256",
        "ledger_prefix_event_count",
        "ledger_prefix_byte_sha256",
        "ledger_phase",
        "created_utc",
        "sealed_truth_opened_before_commitment",
    }
)
_COMMITMENT_FILE_REGISTRY = (
    "generation_plan_commitment.json",
    *_LABEL_INPUTS,
    "truth_commitments.json",
    _ACTUAL_ANALYSIS_HASH_FILENAME,
    *_FIT_OUTPUTS,
    *_MODEL_STATE_COMMITMENT_FILES,
    "model_state_commitment.json",
    *_PREDICTION_OUTPUTS,
)


class V024PredictionCommitmentEvidence:
    """Sealed receipt issued only after strict prediction-chain verification."""

    __slots__ = (
        "_actual_analysis_hash_ledger_commitment_byte_sha256",
        "_artifact_set_sha256",
        "_attempt_id",
        "_byte_sha256",
        "_file_entries",
        "_ledger_committed",
        "_provenance_sha256",
        "_seal",
    )

    def __init__(
        self,
        *,
        _seal: object,
        attempt_id: str,
        byte_sha256: str,
        artifact_set_sha256: str,
        actual_analysis_hash_ledger_commitment_byte_sha256: str,
        file_entries: tuple[tuple[str, int, int, str], ...],
        ledger_committed: bool,
        provenance_sha256: str,
    ) -> None:
        if (
            _seal is not _EVIDENCE_SEAL
            or type(self) is not V024PredictionCommitmentEvidence
        ):
            raise TypeError(
                "Prediction commitment evidence is issued only by strict V2.4 IO"
            )
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_attempt_id", attempt_id)
        object.__setattr__(self, "_byte_sha256", byte_sha256)
        object.__setattr__(self, "_artifact_set_sha256", artifact_set_sha256)
        object.__setattr__(
            self,
            "_actual_analysis_hash_ledger_commitment_byte_sha256",
            actual_analysis_hash_ledger_commitment_byte_sha256,
        )
        object.__setattr__(self, "_file_entries", file_entries)
        object.__setattr__(self, "_ledger_committed", ledger_committed)
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Prediction commitment evidence is immutable")

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def byte_sha256(self) -> str:
        return self._byte_sha256

    @property
    def artifact_set_sha256(self) -> str:
        return self._artifact_set_sha256

    @property
    def actual_analysis_hash_ledger_commitment_byte_sha256(self) -> str:
        return self._actual_analysis_hash_ledger_commitment_byte_sha256

    @property
    def file_entries(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "path": path,
                "row_count": row_count,
                "byte_count": byte_count,
                "byte_sha256": byte_sha256,
            }
            for path, row_count, byte_count, byte_sha256 in self._file_entries
        )

    @property
    def ledger_committed(self) -> bool:
        return self._ledger_committed


class V024FreshGenerationBundle:
    """Sealed fresh-generation capability accepted by the formal fit wrapper."""

    __slots__ = (
        "_contract_view",
        "_file_hashes",
        "_frames",
        "_identity",
        "_ledger_prefix",
        "_root",
        "_seal",
    )

    def __init__(
        self,
        *,
        _seal: object,
        root: Path,
        contract_view: V024ContractView,
        identity: object,
        frames: Mapping[str, pd.DataFrame],
        file_hashes: Mapping[str, str],
        ledger_prefix: bytes,
    ) -> None:
        if _seal is not _SEAL or type(self) is not V024FreshGenerationBundle:
            raise TypeError("Fresh-generation bundles are issued only by V2.4 IO")
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_contract_view", contract_view)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(
            self,
            "_frames",
            tuple((name, frame.copy(deep=True)) for name, frame in frames.items()),
        )
        object.__setattr__(self, "_file_hashes", tuple(sorted(file_hashes.items())))
        object.__setattr__(self, "_ledger_prefix", ledger_prefix)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Fresh-generation bundles are immutable")

    @property
    def attempt_id(self) -> str:
        return self._identity.attempt_id

    @property
    def protocol_id(self) -> str:
        return self._contract_view.protocol.protocol_id


class V024CommittedLabelFreeBundle:
    """Sealed pre-prediction files plus their committed model capability."""

    __slots__ = (
        "_artifact_contract",
        "_contract_view",
        "_config_sha256",
        "_design_status",
        "_file_hashes",
        "_frames",
        "_identity",
        "_ledger_event_count",
        "_ledger_prefix",
        "_model_state",
        "_root",
        "_seal",
    )

    def __init__(
        self,
        *,
        _seal: object,
        root: Path,
        artifact_contract: FrozenArtifactContract,
        contract_view: V024ContractView,
        design_status: str,
        config_sha256: str,
        identity: object,
        frames: Mapping[str, pd.DataFrame],
        file_hashes: Mapping[str, str],
        ledger_prefix: bytes,
        ledger_event_count: int,
        model_state: V024CommittedModelStateEnvelope,
    ) -> None:
        if _seal is not _SEAL or type(self) is not V024CommittedLabelFreeBundle:
            raise TypeError("Committed label-free bundles are issued only by V2.4 IO")
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_artifact_contract", artifact_contract)
        object.__setattr__(self, "_contract_view", contract_view)
        object.__setattr__(self, "_design_status", design_status)
        object.__setattr__(self, "_config_sha256", config_sha256)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(
            self,
            "_frames",
            tuple((name, frame.copy(deep=True)) for name, frame in frames.items()),
        )
        object.__setattr__(self, "_file_hashes", tuple(sorted(file_hashes.items())))
        object.__setattr__(self, "_ledger_prefix", ledger_prefix)
        object.__setattr__(self, "_ledger_event_count", ledger_event_count)
        object.__setattr__(self, "_model_state", model_state)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Committed label-free bundles are immutable")

    @property
    def attempt_id(self) -> str:
        return self._identity.attempt_id

    @property
    def model_state(self) -> V024CommittedModelStateEnvelope:
        return self._model_state


def _prediction_artifact_contract_snapshot(
    contract: FrozenArtifactContract,
) -> FrozenArtifactContract:
    """Remove config-file reachability from a prediction capability."""

    return FrozenArtifactContract(
        protocol_id=contract.protocol_id,
        schema_version=contract.schema_version,
        config_path=Path("__v024_prediction_config_withheld__"),
        config_byte_sha256=contract.config_byte_sha256,
        csv_schemas=MappingProxyType(dict(contract.csv_schemas)),
        json_key_allowlists=MappingProxyType(dict(contract.json_key_allowlists)),
        exposure_keys=contract.exposure_keys,
        partitions=contract.partitions,
        partition_member_counts=MappingProxyType(
            dict(contract.partition_member_counts)
        ),
        prefix_days=contract.prefix_days,
        forecast_days=contract.forecast_days,
        truth_filenames=contract.truth_filenames,
        matched_pair_filenames=contract.matched_pair_filenames,
    )


def _bundle_artifact_contract(
    value: V024FreshGenerationBundle | V024CommittedLabelFreeBundle,
) -> FrozenArtifactContract:
    if type(value) is V024FreshGenerationBundle:
        return value._contract_view.artifacts
    if type(value) is V024CommittedLabelFreeBundle:
        return value._artifact_contract
    raise V024IOError("Unknown sealed bundle type")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verify_stored_bundle_frames(
    value: V024FreshGenerationBundle | V024CommittedLabelFreeBundle,
    *,
    expected_filenames: Sequence[str],
) -> None:
    contract = _bundle_artifact_contract(value)
    frames = tuple(value._frames)
    if tuple(name for name, _ in frames) != tuple(expected_filenames):
        raise V024IOError("Sealed in-memory frame registry changed")
    file_hashes = dict(value._file_hashes)
    for filename, frame in frames:
        if type(frame) is not pd.DataFrame:
            raise V024IOError(f"Sealed frame type changed: {filename}")
        try:
            raw = canonical_csv_bytes(
                frame,
                contract.csv_schema(filename),
                contract,
                formal=True,
            )
        except V015ArtifactError as exc:
            raise V024IOError(f"Sealed in-memory frame changed: {filename}") from exc
        if file_hashes.get(filename) != _sha256(raw):
            raise V024IOError(f"Sealed in-memory frame changed: {filename}")


def _evidence_digest(
    *,
    attempt_id: str,
    byte_sha256: str,
    artifact_set_sha256: str,
    actual_analysis_hash_ledger_commitment_byte_sha256: str,
    file_entries: tuple[tuple[str, int, int, str], ...],
    ledger_committed: bool,
) -> str:
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise V024IOError("Prediction evidence attempt ID is invalid")
    if type(ledger_committed) is not bool:
        raise V024IOError("Prediction evidence ledger status is invalid")
    hasher = hashlib.sha256()
    hasher.update(_EVIDENCE_DOMAIN)
    for value in (
        attempt_id.encode("ascii"),
        bytes.fromhex(byte_sha256),
        bytes.fromhex(artifact_set_sha256),
        bytes.fromhex(actual_analysis_hash_ledger_commitment_byte_sha256),
        b"\x01" if ledger_committed else b"\x00",
    ):
        hasher.update(struct.pack("<Q", len(value)))
        hasher.update(value)
    for path, row_count, byte_count, digest in file_entries:
        raw_path = path.encode("ascii")
        hasher.update(struct.pack("<Q", len(raw_path)))
        hasher.update(raw_path)
        hasher.update(struct.pack("<Q", row_count))
        hasher.update(struct.pack("<Q", byte_count))
        hasher.update(bytes.fromhex(digest))
    return hasher.hexdigest()


def _validated_evidence_file_entries(
    value: object,
) -> tuple[tuple[str, int, int, str], ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise V024IOError("Prediction evidence file entries must be a sequence")
    result: list[tuple[str, int, int, str]] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != _FILE_ENTRY_KEYS:
            raise V024IOError("Prediction evidence contains an invalid file entry")
        path = entry.get("path")
        row_count = entry.get("row_count")
        byte_count = entry.get("byte_count")
        if (
            not isinstance(path, str)
            or Path(path).name != path
            or type(row_count) is not int
            or row_count < 1
            or type(byte_count) is not int
            or byte_count < 1
        ):
            raise V024IOError("Prediction evidence file metadata is invalid")
        result.append(
            (
                path,
                row_count,
                byte_count,
                _digest(
                    entry.get("byte_sha256"),
                    context=f"prediction evidence hash/{path}",
                ),
            )
        )
    if not result or tuple(item[0] for item in result) != _COMMITMENT_FILE_REGISTRY:
        raise V024IOError("Prediction evidence file registry changed")
    return tuple(result)


def _issue_prediction_commitment_evidence(
    *,
    attempt_id: str,
    byte_sha256: str,
    artifact_set_sha256: str,
    actual_analysis_hash_ledger_commitment_byte_sha256: str,
    file_entries: Sequence[Mapping[str, object]],
    ledger_committed: bool,
) -> V024PredictionCommitmentEvidence:
    for value, context in (
        (byte_sha256, "prediction commitment hash"),
        (artifact_set_sha256, "prediction artifact-set hash"),
        (
            actual_analysis_hash_ledger_commitment_byte_sha256,
            "actual-analysis hash-ledger commitment hash",
        ),
    ):
        _digest(value, context=context)
    if type(ledger_committed) is not bool:
        raise V024IOError("ledger_committed must be an exact boolean")
    entries = _validated_evidence_file_entries(file_entries)
    provenance = _evidence_digest(
        attempt_id=attempt_id,
        byte_sha256=byte_sha256,
        artifact_set_sha256=artifact_set_sha256,
        actual_analysis_hash_ledger_commitment_byte_sha256=(
            actual_analysis_hash_ledger_commitment_byte_sha256
        ),
        file_entries=entries,
        ledger_committed=ledger_committed,
    )
    return V024PredictionCommitmentEvidence(
        _seal=_EVIDENCE_SEAL,
        attempt_id=attempt_id,
        byte_sha256=byte_sha256,
        artifact_set_sha256=artifact_set_sha256,
        actual_analysis_hash_ledger_commitment_byte_sha256=(
            actual_analysis_hash_ledger_commitment_byte_sha256
        ),
        file_entries=entries,
        ledger_committed=ledger_committed,
        provenance_sha256=provenance,
    )


def _require_prediction_commitment_evidence_v024(
    value: object,
    *,
    require_ledger_committed: bool,
) -> V024PredictionCommitmentEvidence:
    """Validate the issuer seal and digest of an IO prediction receipt."""

    if (
        type(value) is not V024PredictionCommitmentEvidence
        or value._seal is not _EVIDENCE_SEAL
    ):
        raise V024IOError(
            "Exact prediction commitment evidence issued by V2.4 IO is required"
        )
    expected = _evidence_digest(
        attempt_id=value.attempt_id,
        byte_sha256=_digest(
            value.byte_sha256,
            context="prediction commitment hash",
        ),
        artifact_set_sha256=_digest(
            value.artifact_set_sha256,
            context="prediction artifact-set hash",
        ),
        actual_analysis_hash_ledger_commitment_byte_sha256=_digest(
            value.actual_analysis_hash_ledger_commitment_byte_sha256,
            context="actual-analysis hash-ledger commitment hash",
        ),
        file_entries=_validated_evidence_file_entries(value.file_entries),
        ledger_committed=value.ledger_committed,
    )
    if value._provenance_sha256 != expected:
        raise V024IOError("Prediction commitment evidence digest changed")
    if require_ledger_committed and not value.ledger_committed:
        raise V024IOError("Prediction commitment evidence is not ledger committed")
    return value


def _is_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise V024IOError(f"Cannot inspect physical path: {path}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & flag)


def _physical_root(raw: str | Path) -> Path:
    root = Path(os.path.abspath(os.fspath(raw)))
    if not root.is_dir() or _is_reparse(root):
        raise V024IOError("Label-free root must be a direct physical directory")
    for parent in root.parents:
        if parent.exists() and _is_reparse(parent):
            raise V024IOError("Label-free root traverses a reparse point")
    return root


def _direct_file(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise V024IOError(f"Unsafe label-free filename: {filename!r}")
    path = root / filename
    if not path.exists() or _is_reparse(path) or not path.is_file():
        raise V024IOError(f"{filename} is not a direct physical file")
    return path


def _require_membership(root: Path, expected: frozenset[str], *, context: str) -> None:
    entries = tuple(root.iterdir())
    observed = {entry.name for entry in entries}
    if observed != set(expected):
        raise V024IOError(
            f"{context} root membership changed: "
            f"missing={sorted(set(expected) - observed)}, "
            f"unexpected={sorted(observed - set(expected))}"
        )
    for entry in entries:
        if _is_reparse(entry) or not entry.is_file():
            raise V024IOError(f"{context} contains a nonphysical artifact")


def _require_contract(view: object) -> V024ContractView:
    try:
        validated = resolve_contract_view(view)
    except (TypeError, ValueError) as exc:
        raise V024IOError("contract_view is invalid") from exc
    if validated.design_status != "implementation_frozen":
        raise V024IOError("V2.4 formal IO requires the implementation-frozen contract")
    return validated


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V024IOError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V024IOError(f"Nonfinite JSON constant is forbidden: {token}")


def _strict_json(raw: bytes, *, filename: str, compact: bool = False) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise V024IOError(f"{filename} must be exact bytes")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V024IOError(f"{filename} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise V024IOError(f"{filename} must be a JSON object")
    canonical = (
        (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        if compact
        else canonical_json_bytes(payload)
    )
    if canonical != raw:
        raise V024IOError(f"{filename} is not canonical JSON")
    return payload


def _utc(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise V024IOError(f"{context} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise V024IOError(f"{context} is not a valid timestamp") from exc
    if parsed.utcoffset() is None:
        raise V024IOError(f"{context} lacks a timezone")
    return value


def _digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V024IOError(f"{context} must be lowercase SHA256")
    return value


def _positive_int(value: object, *, context: str) -> int:
    if type(value) is not int or value < 1:
        raise V024IOError(f"{context} must be a positive integer")
    return value


def _identity_json(
    payload: Mapping[str, Any],
    *,
    contract: FrozenArtifactContract,
    filename: str,
) -> None:
    if (
        payload.get("protocol_id") != contract.protocol_id
        or payload.get("config_sha256") != contract.config_byte_sha256
    ):
        raise V024IOError(f"{filename} identity changed")


def _load_ledger(
    root: Path,
    *,
    view: V024ContractView,
    attempt_id: str,
) -> tuple[AttemptProgress, bytes, int]:
    ledger = _direct_file(root, "exposure_log.jsonl")
    try:
        events, states, raw = read_exposure_log(
            ledger,
            expected_config_sha256=view.artifacts.config_byte_sha256,
            sealed_filenames=view.artifacts.sealed_filenames,
        )
    except V024LedgerError as exc:
        raise V024IOError(str(exc)) from exc
    try:
        progress = states[attempt_id]
    except KeyError as exc:
        raise V024IOError("Attempt is absent from the canonical ledger") from exc
    if progress.identity.attempt_id != attempt_id or progress.terminal_failed:
        raise V024IOError("Attempt ledger identity is unusable")
    return progress, raw, len(events)


def _require_phase_hash(
    progress: AttemptProgress,
    *,
    field: str,
    raw: bytes,
    context: str,
) -> str:
    expected = getattr(progress, field)
    observed = _sha256(raw)
    if expected is None or expected != observed:
        raise V024IOError(f"{context} differs from its ledger commitment")
    return observed


def _verify_generation_and_truth(
    root: Path,
    *,
    view: V024ContractView,
    progress: AttemptProgress,
    require_semantic_plan_recompute: bool,
) -> tuple[bytes, bytes, dict[str, str]]:
    plan_raw = _direct_file(root, "generation_plan_commitment.json").read_bytes()
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
    if require_semantic_plan_recompute:
        from lifetwin.experiments.calendar_long_horizon_v019_actual_ledger_io import (  # noqa: PLC0415
            V024ActualLedgerIOError,
            recompute_generation_plan_commitment_bytes_v024,
        )

        try:
            expected_plan_raw = recompute_generation_plan_commitment_bytes_v024(view)
        except V024ActualLedgerIOError as exc:
            raise V024IOError("Formal generation-plan recomputation failed") from exc
        if plan_raw != expected_plan_raw:
            raise V024IOError(
                "Generation plan is ledger-bound but not the frozen formal plan"
            )

    truth_path = _direct_file(root, "truth_commitments.json")
    truth_raw = truth_path.read_bytes()
    if progress.truth_commitments_byte_sha256 != _sha256(truth_raw):
        raise V024IOError("Truth commitment differs from the ledger")
    truth_payload = _strict_json(
        truth_raw,
        filename="truth_commitments.json",
    )
    if (
        set(truth_payload) != view.artifacts.json_keys("truth_commitments.json")
        or truth_payload.get("protocol_id") != view.protocol.protocol_id
        or truth_payload.get("config_sha256") != view.artifacts.config_byte_sha256
        or truth_payload.get("truth_values_withheld_by_physical_path") is not True
    ):
        raise V024IOError("Truth commitment identity or schema changed")
    _utc(
        truth_payload.get("created_utc"),
        context="truth commitment created_utc",
    )
    entries = truth_payload.get("files")
    if (
        not isinstance(entries, list)
        or len(entries) != 9
        or len(entries) != len(view.artifacts.sealed_filenames)
    ):
        raise V024IOError("Truth commitment file registry changed")
    truth_hashes: dict[str, str] = {}
    for filename, entry in zip(
        view.artifacts.sealed_filenames,
        entries,
        strict=True,
    ):
        schema = view.artifacts.csv_schema(filename)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != _FILE_ENTRY_KEYS
            or entry.get("path") != filename
            or type(entry.get("row_count")) is not int
            or entry.get("row_count") != schema.required_rows
            or type(entry.get("byte_count")) is not int
            or entry["byte_count"] < 1
        ):
            raise V024IOError(f"Truth commitment metadata changed for {filename}")
        truth_hashes[filename] = _digest(
            entry.get("byte_sha256"),
            context=f"truth commitment hash/{filename}",
        )
    return (
        plan_raw,
        truth_raw,
        truth_hashes,
    )


def _recompute_actual_analysis_hash_ledger_bytes(
    frames: Mapping[str, pd.DataFrame],
    *,
    view: V024ContractView,
) -> bytes:
    from lifetwin.experiments.calendar_long_horizon_v019_actual_ledger_io import (  # noqa: PLC0415
        V024ActualLedgerIOError,
        recompute_actual_analysis_hash_ledger_bytes_v024,
    )

    try:
        return recompute_actual_analysis_hash_ledger_bytes_v024(
            frames,
            view=view,
        )
    except V024ActualLedgerIOError as exc:
        raise V024IOError(
            "Formal actual-analysis hash-ledger recomputation failed"
        ) from exc


def _ordinary_family_lookup(
    view: V024ContractView,
) -> dict[tuple[str, str], str]:
    from lifetwin.experiments.calendar_long_horizon_v019_actual_ledger_io import (  # noqa: PLC0415
        V024ActualLedgerIOError,
        ordinary_family_lookup_v024,
    )

    try:
        return ordinary_family_lookup_v024(view)
    except V024ActualLedgerIOError as exc:
        raise V024IOError("Formal ordinary identity derivation failed") from exc


def _verify_actual_analysis_hash_ledger(
    root: Path,
    *,
    view: V024ContractView,
    progress: AttemptProgress,
    frames: Mapping[str, pd.DataFrame],
    require_semantic_recompute: bool,
) -> bytes:
    raw = _direct_file(root, _ACTUAL_ANALYSIS_HASH_FILENAME).read_bytes()
    expected_hash = _require_phase_hash(
        progress,
        field="actual_analysis_hash_ledger_commitment_byte_sha256",
        raw=raw,
        context="Actual-analysis hash ledger",
    )
    payload = _strict_json(
        raw,
        filename=_ACTUAL_ANALYSIS_HASH_FILENAME,
        compact=True,
    )
    if require_semantic_recompute:
        from lifetwin.experiments.calendar_long_horizon_v019_actual_ledger_io import (  # noqa: PLC0415
            V024ActualLedgerIOError,
            verify_actual_analysis_hash_ledger_payload_v024,
        )

        try:
            verify_actual_analysis_hash_ledger_payload_v024(
                payload,
                expected_byte_sha256=expected_hash,
                view=view,
            )
        except V024ActualLedgerIOError as exc:
            raise V024IOError(
                "Actual-analysis hash-ledger structure is invalid"
            ) from exc
        recomputed = _recompute_actual_analysis_hash_ledger_bytes(
            frames,
            view=view,
        )
        if raw != recomputed:
            raise V024IOError(
                "Actual-analysis hash ledger does not match the canonical inputs"
            )
    return raw


def _read_frames(
    root: Path,
    *,
    contract: FrozenArtifactContract,
    filenames: Sequence[str],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for filename in filenames:
        try:
            frames[filename] = read_canonical_csv(
                _direct_file(root, filename),
                contract,
                formal=True,
            )
        except V015ArtifactError as exc:
            raise V024IOError(str(exc)) from exc
    return frames


def _verify_fit_commitment(
    root: Path,
    *,
    contract: FrozenArtifactContract,
    progress: AttemptProgress,
    frames: Mapping[str, pd.DataFrame],
) -> bytes:
    raw = _direct_file(root, "fit_commitment.json").read_bytes()
    _require_phase_hash(
        progress,
        field="fit_commitment_byte_sha256",
        raw=raw,
        context="Fit commitment",
    )
    payload = _strict_json(raw, filename="fit_commitment.json")
    if (
        set(payload) != _FIT_COMMITMENT_KEYS
        or payload.get("git_commit") != progress.identity.git_commit
        or payload.get("worker_count") != 6
    ):
        raise V024IOError("fit_commitment.json semantics changed")
    _identity_json(payload, contract=contract, filename="fit_commitment.json")
    _utc(payload.get("created_utc"), context="fit commitment created_utc")
    entries = payload.get("files")
    if not isinstance(entries, list) or len(entries) != len(_FIT_COMMITMENT_FILES):
        raise V024IOError("Fit commitment file registry changed")
    for filename, entry in zip(_FIT_COMMITMENT_FILES, entries, strict=True):
        if not isinstance(entry, Mapping) or set(entry) != _FILE_ENTRY_KEYS:
            raise V024IOError("Fit commitment contains an invalid file entry")
        path = _direct_file(root, filename)
        file_raw = path.read_bytes()
        expected_rows = len(frames[filename]) if filename.endswith(".csv") else 1
        if (
            entry.get("path") != filename
            or entry.get("row_count") != expected_rows
            or entry.get("byte_count") != len(file_raw)
            or entry.get("byte_sha256") != _sha256(file_raw)
        ):
            raise V024IOError(f"Fit commitment does not bind {filename}")
    return raw


def _resolve_input_hashes(
    value: object,
    *,
    root: Path,
    truth_hashes: Mapping[str, str],
    context: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise V024IOError(f"{context} must be a nonempty hash mapping")
    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        if not isinstance(raw_name, str):
            raise V024IOError(f"{context} contains a non-string filename")
        digest = _digest(raw_digest, context=f"{context}.{raw_name}")
        if raw_name in truth_hashes:
            actual = truth_hashes[raw_name]
        else:
            actual = _sha256(_direct_file(root, raw_name).read_bytes())
        if actual != digest:
            raise V024IOError(f"{context} input changed: {raw_name}")
        result[raw_name] = digest
    return result


def _resolve_stage_input_hashes(
    value: object,
    *,
    root: Path,
    truth_hashes: Mapping[str, str],
    context: str,
    stage: str,
    _input_filenames_by_stage: object | None,
) -> dict[str, str]:
    expected_filenames: tuple[str, ...] | None = None
    if _input_filenames_by_stage is not None:
        try:
            from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
                V020CheckpointRegistryError,
                require_input_filenames_by_stage_v020,
            )

            expected_filenames = require_input_filenames_by_stage_v020(
                _input_filenames_by_stage
            )[stage]
        except (KeyError, V020CheckpointRegistryError) as exc:
            raise V024IOError("V0.20 checkpoint registry is invalid") from exc
        if not isinstance(value, Mapping) or set(value) != set(expected_filenames):
            raise V024IOError(f"{context} input registry changed")
    return _resolve_input_hashes(
        value,
        root=root,
        truth_hashes=truth_hashes,
        context=context,
    )


def _require_model_state_input_hashes(
    value: object,
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    if value != expected:
        raise V024IOError("Model-state input hashes differ from verified artifacts")


def _decode_model_state(
    raw: bytes,
    *,
    view: V024ContractView,
) -> _v015.DecodedModelState:
    payload = _strict_json(raw, filename="model_state.json")
    if set(payload) != view.artifacts.json_keys("model_state.json"):
        raise V024IOError("model_state.json schema changed")
    _identity_json(payload, contract=view.artifacts, filename="model_state.json")
    inherited = dict(payload)
    inherited["protocol_id"] = V2_PROTOCOL_ID
    inherited["config_sha256"] = V2_CONFIG_BYTE_SHA256
    try:
        return _v015.validate_model_state_payload(inherited)
    except (TypeError, ValueError) as exc:
        raise V024IOError("V2.4 model state is numerically invalid") from exc


def _verify_calibration_audit(
    raw: bytes,
    *,
    view: V024ContractView,
    decoded: _v015.DecodedModelState,
    mask: object,
) -> V024CalibrationAudit:
    payload = _strict_json(raw, filename="calibration_population_audit.json")
    if set(payload) != _AUDIT_KEYS:
        raise V024IOError("Calibration population audit schema changed")
    _identity_json(
        payload,
        contract=view.artifacts,
        filename="calibration_population_audit.json",
    )
    try:
        audit = V024CalibrationAudit(
            **{name: payload[name] for name in _REQUIRED_AUDIT_COUNT_FIELDS},
            eligibility_mask_sha256=payload["eligibility_mask_sha256"],
        )
        _validate_calibration_audit(
            audit,
            mask_commitment=mask,
            calibration_state=decoded.training_state.calibration,
        )
    except (TypeError, ValueError, V024StateCodecError) as exc:
        raise V024IOError("Calibration population audit is invalid") from exc
    expected_ids = [row.cluster_id for row in mask.rows]
    expected_mask = [row.eligible for row in mask.rows]
    if (
        payload.get("eligibility_mask_cluster_ids") != expected_ids
        or payload.get("eligibility_mask") != expected_mask
        or payload.get("calibration_mask_commitment_byte_sha256")
        != mask.canonical_byte_sha256
        or payload.get("isotonic_state_sha256")
        != _v015.isotonic_state_sha256(decoded.training_state.calibration)
        or payload.get("conformal_state_sha256")
        != _v015.conformal_state_sha256(decoded.training_state.calibration)
        or payload.get("selected_mean_baseline")
        != decoded.training_state.calibration.selected_mean_baseline
    ):
        raise V024IOError("Calibration population audit bindings changed")
    _utc(payload.get("created_utc"), context="calibration audit created_utc")
    return audit


def _translate_manifest_to_v2(
    payload: Mapping[str, Any],
    *,
    view: V024ContractView,
    filename: str,
) -> dict[str, Any]:
    if set(payload) != view.artifacts.json_keys(filename):
        raise V024IOError(f"{filename} schema changed")
    _identity_json(payload, contract=view.artifacts, filename=filename)
    inherited = dict(payload)
    inherited["protocol_id"] = V2_PROTOCOL_ID
    inherited["config_sha256"] = V2_CONFIG_BYTE_SHA256
    return inherited


def _verify_training_chain(
    root: Path,
    *,
    view: V024ContractView,
    progress: AttemptProgress,
    truth_hashes: Mapping[str, str],
    decoded: _v015.DecodedModelState,
    _input_filenames_by_stage: object | None = None,
) -> tuple[bytes, bytes, bytes, dict[str, dict[str, str]]]:
    contract = view.artifacts
    center_raw = _direct_file(root, "center_state_checkpoint.json").read_bytes()
    risk_raw = _direct_file(root, "risk_state_checkpoint.json").read_bytes()
    _require_phase_hash(
        progress,
        field="center_state_checkpoint_byte_sha256",
        raw=center_raw,
        context="Center checkpoint",
    )
    _require_phase_hash(
        progress,
        field="risk_state_checkpoint_byte_sha256",
        raw=risk_raw,
        context="Risk checkpoint",
    )
    center = _strict_json(center_raw, filename="center_state_checkpoint.json")
    risk = _strict_json(risk_raw, filename="risk_state_checkpoint.json")
    if (
        set(center) != _CENTER_KEYS
        or center.get("state_kind") != "center_development"
        or center.get("center_state_sha256")
        != _v015.center_state_sha256(decoded.training_state.center)
    ):
        raise V024IOError("Center checkpoint semantics changed")
    if (
        set(risk) != _RISK_KEYS
        or risk.get("state_kind") != "risk_development"
        or risk.get("center_checkpoint_byte_sha256") != _sha256(center_raw)
        or risk.get("risk_state_sha256")
        != _v015.risk_state_sha256(decoded.training_state.risk)
    ):
        raise V024IOError("Risk checkpoint semantics changed")
    for filename, payload in (
        ("center_state_checkpoint.json", center),
        ("risk_state_checkpoint.json", risk),
    ):
        _identity_json(payload, contract=contract, filename=filename)
        _utc(payload.get("created_utc"), context=f"{filename} created_utc")
    center_inputs = _resolve_stage_input_hashes(
        center.get("input_byte_hashes"),
        root=root,
        truth_hashes=truth_hashes,
        context="center checkpoint",
        stage="center_development",
        _input_filenames_by_stage=_input_filenames_by_stage,
    )
    risk_inputs = _resolve_stage_input_hashes(
        risk.get("input_byte_hashes"),
        root=root,
        truth_hashes=truth_hashes,
        context="risk checkpoint",
        stage="risk_development",
        _input_filenames_by_stage=_input_filenames_by_stage,
    )

    training_raw = _direct_file(root, "training_manifest.json").read_bytes()
    if risk.get("training_manifest_byte_sha256") != _sha256(training_raw):
        raise V024IOError("Risk checkpoint does not bind training manifest")
    training = _strict_json(training_raw, filename="training_manifest.json")
    training_v2 = _translate_manifest_to_v2(
        training,
        view=view,
        filename="training_manifest.json",
    )
    try:
        _v015.validate_training_manifest(training_v2)
        _v015.verify_training_manifest_state_hashes(
            training_v2,
            center_state=decoded.training_state.center,
            risk_state=decoded.training_state.risk,
        )
    except (TypeError, ValueError) as exc:
        raise V024IOError("Training manifest state bindings changed") from exc
    if (
        training.get("center_development_input_hashes") != center_inputs
        or training.get("risk_development_input_hashes") != risk_inputs
    ):
        raise V024IOError("Training manifest input hashes changed")

    calibration_raw = _direct_file(root, "calibration_manifest.json").read_bytes()
    calibration = _strict_json(
        calibration_raw,
        filename="calibration_manifest.json",
    )
    calibration_v2 = _translate_manifest_to_v2(
        calibration,
        view=view,
        filename="calibration_manifest.json",
    )
    try:
        _v015.validate_calibration_manifest(calibration_v2)
        _v015.verify_calibration_manifest_state_hashes(
            calibration_v2,
            calibration_state=decoded.training_state.calibration,
        )
    except (TypeError, ValueError) as exc:
        raise V024IOError("Calibration manifest state bindings changed") from exc
    calibration_inputs = _resolve_stage_input_hashes(
        calibration.get("calibration_input_hashes"),
        root=root,
        truth_hashes=truth_hashes,
        context="calibration manifest",
        stage="calibration",
        _input_filenames_by_stage=_input_filenames_by_stage,
    )
    expected_inputs = {
        "center_development": center_inputs,
        "risk_development": risk_inputs,
        "calibration": calibration_inputs,
    }
    _require_model_state_input_hashes(decoded.input_byte_hashes, expected_inputs)
    return center_raw, risk_raw, calibration_raw, expected_inputs


def _verify_model_commitment(
    root: Path,
    *,
    view: V024ContractView,
    progress: AttemptProgress,
) -> tuple[bytes, dict[str, bytes]]:
    raw = _direct_file(root, "model_state_commitment.json").read_bytes()
    _require_phase_hash(
        progress,
        field="model_state_commitment_byte_sha256",
        raw=raw,
        context="Model-state commitment",
    )
    payload = _strict_json(raw, filename="model_state_commitment.json")
    if (
        set(payload) != _MODEL_COMMITMENT_KEYS
        or payload.get("git_commit") != progress.identity.git_commit
        or not isinstance(payload.get("files"), list)
    ):
        raise V024IOError("Model-state commitment identity changed")
    _identity_json(
        payload,
        contract=view.artifacts,
        filename="model_state_commitment.json",
    )
    _utc(payload.get("created_utc"), context="model-state commitment created_utc")
    entries = payload["files"]
    if len(entries) != len(_MODEL_STATE_COMMITMENT_FILES):
        raise V024IOError("Model-state commitment registry changed")
    artifacts: dict[str, bytes] = {}
    for filename, entry in zip(
        _MODEL_STATE_COMMITMENT_FILES,
        entries,
        strict=True,
    ):
        if not isinstance(entry, Mapping) or set(entry) != _FILE_ENTRY_KEYS:
            raise V024IOError("Model-state commitment file entry changed")
        artifact = _direct_file(root, filename).read_bytes()
        if (
            entry.get("path") != filename
            or entry.get("row_count") != 1
            or entry.get("byte_count") != len(artifact)
            or entry.get("byte_sha256") != _sha256(artifact)
        ):
            raise V024IOError(f"Model-state commitment does not bind {filename}")
        artifacts[filename] = artifact
    return raw, artifacts


def create_actual_analysis_hash_ledger_commitment_v024(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    contract_view: V024ContractView | None = None,
) -> str:
    """Recompute and exclusively persist the formal post-generation ledger."""

    view = _require_contract(contract_view)
    root = _physical_root(label_free_root)
    _require_membership(
        root,
        _POST_TRUTH_FILES,
        context="Pre-actual-analysis commitment",
    )
    progress, ledger_raw, _ = _load_ledger(
        root,
        view=view,
        attempt_id=attempt_id,
    )
    if (
        progress.completed_phase != "truth_committed"
        or progress.pending_phase != "actual_analysis_hash_ledger_committed"
    ):
        raise V024IOError(
            "Ledger is not at the pending actual-analysis commitment boundary"
        )
    _verify_generation_and_truth(
        root,
        view=view,
        progress=progress,
        require_semantic_plan_recompute=True,
    )
    frames = _read_frames(
        root,
        contract=view.artifacts,
        filenames=_LABEL_INPUTS,
    )
    raw = _recompute_actual_analysis_hash_ledger_bytes(frames, view=view)
    if _direct_file(root, "exposure_log.jsonl").read_bytes() != ledger_raw:
        raise V024IOError(
            "Ledger changed while deriving the actual-analysis commitment"
        )
    _exclusive_create(root / _ACTUAL_ANALYSIS_HASH_FILENAME, raw)
    _require_membership(
        root,
        _GENERATION_FILES,
        context="Actual-analysis commitment",
    )
    return _sha256(raw)


def load_fresh_generation_bundle_v024(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    contract_view: V024ContractView | None = None,
) -> V024FreshGenerationBundle:
    """Verify the exact post-generation root and issue the sole formal fit input."""

    view = _require_contract(contract_view)
    root = _physical_root(label_free_root)
    _require_membership(root, _GENERATION_FILES, context="Fresh generation")
    progress, ledger_raw, _ = _load_ledger(
        root,
        view=view,
        attempt_id=attempt_id,
    )
    if (
        progress.completed_phase != "actual_analysis_hash_ledger_committed"
        or progress.pending_phase != "label_free_fit_committed"
    ):
        raise V024IOError("Ledger is not at the pending formal-fit boundary")
    _verify_generation_and_truth(
        root,
        view=view,
        progress=progress,
        require_semantic_plan_recompute=True,
    )
    frames = _read_frames(
        root,
        contract=view.artifacts,
        filenames=_LABEL_INPUTS,
    )
    _verify_actual_analysis_hash_ledger(
        root,
        view=view,
        progress=progress,
        frames=frames,
        require_semantic_recompute=True,
    )
    hashes = {
        filename: _sha256(_direct_file(root, filename).read_bytes())
        for filename in _GENERATION_FILES
    }
    return V024FreshGenerationBundle(
        _seal=_SEAL,
        root=root,
        contract_view=view,
        identity=progress.identity,
        frames=frames,
        file_hashes=hashes,
        ledger_prefix=ledger_raw,
    )


def _load_committed_bundle(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    contract_view: V024ContractView,
    expected_membership: frozenset[str],
    _input_filenames_by_stage: object | None = None,
) -> tuple[V024CommittedLabelFreeBundle, AttemptProgress]:
    view = _require_contract(contract_view)
    root = _physical_root(label_free_root)
    _require_membership(root, expected_membership, context="Committed label-free")
    progress, ledger_raw, event_count = _load_ledger(
        root,
        view=view,
        attempt_id=attempt_id,
    )
    allowed = {
        ("model_state_committed", "prediction_started"),
        ("prediction_committed", None),
    }
    if (progress.completed_phase, progress.pending_phase) not in allowed:
        raise V024IOError("Ledger is not at a committed prediction boundary")
    plan_raw, truth_raw, truth_hashes = _verify_generation_and_truth(
        root,
        view=view,
        progress=progress,
        require_semantic_plan_recompute=False,
    )
    frames = _read_frames(
        root,
        contract=view.artifacts,
        filenames=(*_LABEL_INPUTS, *_FIT_OUTPUTS),
    )
    actual_analysis_raw = _verify_actual_analysis_hash_ledger(
        root,
        view=view,
        progress=progress,
        frames={name: frames[name] for name in _LABEL_INPUTS},
        require_semantic_recompute=False,
    )
    fit_raw = _verify_fit_commitment(
        root,
        contract=view.artifacts,
        progress=progress,
        frames={
            **frames,
            "truth_commitments.json": pd.DataFrame(index=[0]),
        },
    )
    model_raw = _direct_file(root, "model_state.json").read_bytes()
    decoded = _decode_model_state(model_raw, view=view)
    mask_raw = _direct_file(root, "calibration_mask_commitment.json").read_bytes()
    try:
        mask = deserialize_calibration_mask_commitment_json_v024(mask_raw)
    except (TypeError, ValueError) as exc:
        raise V024IOError("Calibration mask commitment is invalid") from exc
    _require_phase_hash(
        progress,
        field="calibration_mask_commitment_byte_sha256",
        raw=mask_raw,
        context="Calibration mask",
    )
    center_raw, risk_raw, calibration_raw, input_hashes = _verify_training_chain(
        root,
        view=view,
        progress=progress,
        truth_hashes=truth_hashes,
        decoded=decoded,
        _input_filenames_by_stage=_input_filenames_by_stage,
    )
    audit_raw = _direct_file(root, "calibration_population_audit.json").read_bytes()
    audit = _verify_calibration_audit(
        audit_raw,
        view=view,
        decoded=decoded,
        mask=mask,
    )
    try:
        training_provenance = _rehydrate_v024_training_provenance_after_strict_io(
            contract_view=view,
            attempt_id=attempt_id,
            training_state=decoded.training_state,
            calibration_audit=audit,
            mask_commitment=mask,
            generation_plan_commitment_bytes=plan_raw,
            truth_commitment_bytes=truth_raw,
            actual_analysis_hash_ledger_commitment_bytes=(actual_analysis_raw),
            label_free_fit_commitment_bytes=fit_raw,
            center_state_commitment_bytes=center_raw,
            risk_state_commitment_bytes=risk_raw,
            calibration_state_commitment_bytes=calibration_raw,
            verified_input_byte_hashes=input_hashes,
        )
        validated_model = deserialize_model_state_json_v024(
            model_raw,
            provenance_envelope=training_provenance,
        )
    except (V024ProvenanceError, V024StateCodecError) as exc:
        raise V024IOError("Model-state provenance rehydration failed") from exc
    model_commitment_raw, state_artifacts = _verify_model_commitment(
        root,
        view=view,
        progress=progress,
    )
    try:
        committed_model = _issue_committed_model_state_envelope_v024(
            validated_model_state=validated_model,
            model_state_commitment_bytes=model_commitment_raw,
            ledger_model_state_commitment_byte_sha256=_sha256(model_commitment_raw),
            committed_state_artifact_bytes=state_artifacts,
        )
    except V024ProvenanceError as exc:
        raise V024IOError("Committed model-state envelope issuance failed") from exc
    hashes = {
        filename: _sha256(_direct_file(root, filename).read_bytes())
        for filename in _PRE_PREDICTION_FILES
    }
    bundle = V024CommittedLabelFreeBundle(
        _seal=_SEAL,
        root=root,
        artifact_contract=_prediction_artifact_contract_snapshot(view.artifacts),
        contract_view=view,
        design_status=view.design_status,
        config_sha256=view.artifacts.config_byte_sha256,
        identity=progress.identity,
        frames=frames,
        file_hashes=hashes,
        ledger_prefix=ledger_raw,
        ledger_event_count=event_count,
        model_state=committed_model,
    )
    return bundle, progress


def load_committed_label_free_bundle_v024(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    contract_view: V024ContractView | None = None,
    _input_filenames_by_stage: object | None = None,
) -> V024CommittedLabelFreeBundle:
    """Validate the complete pre-prediction chain without any truth path."""

    view = _require_contract(contract_view)
    bundle, progress = _load_committed_bundle(
        label_free_root=label_free_root,
        attempt_id=attempt_id,
        contract_view=view,
        expected_membership=_PRE_PREDICTION_FILES,
        _input_filenames_by_stage=_input_filenames_by_stage,
    )
    if (
        progress.completed_phase != "model_state_committed"
        or progress.pending_phase != "prediction_started"
    ):
        raise V024IOError("Prediction has already advanced past its sole write window")
    return bundle


def _require_fresh_bundle(value: object) -> V024FreshGenerationBundle:
    if type(value) is not V024FreshGenerationBundle or value._seal is not _SEAL:
        raise V024IOError("An exact IO-issued fresh-generation bundle is required")
    return _require_fresh_bundle_unchanged(
        value,
        expected_membership=_GENERATION_FILES,
    )


def _require_fresh_bundle_unchanged(
    value: V024FreshGenerationBundle,
    *,
    expected_membership: frozenset[str],
) -> V024FreshGenerationBundle:
    if type(value) is not V024FreshGenerationBundle or value._seal is not _SEAL:
        raise V024IOError("An exact IO-issued fresh-generation bundle is required")
    _require_membership(
        value._root,
        expected_membership,
        context="Fresh-generation operation",
    )
    for filename, digest in value._file_hashes:
        if _sha256(_direct_file(value._root, filename).read_bytes()) != digest:
            raise V024IOError("Fresh-generation artifact changed after issuance")
    if _direct_file(value._root, "exposure_log.jsonl").read_bytes() != (
        value._ledger_prefix
    ):
        raise V024IOError("Fresh-generation ledger changed after issuance")
    _verify_stored_bundle_frames(value, expected_filenames=_LABEL_INPUTS)
    return value


def _extract_fresh_generation_frames_for_formal_fit_v024(
    value: object,
) -> tuple[pd.DataFrame, pd.DataFrame, FrozenArtifactContract]:
    bundle = _require_fresh_bundle(value)
    frames = dict(bundle._frames)
    return (
        frames["prefix_pack.csv"].copy(deep=True),
        frames["forecast_coordinates.csv"].copy(deep=True),
        bundle._contract_view.artifacts,
    )


def _require_committed_bundle(value: object) -> V024CommittedLabelFreeBundle:
    if type(value) is not V024CommittedLabelFreeBundle or value._seal is not _SEAL:
        raise V024IOError("An exact IO-issued committed label-free bundle is required")
    return value


def _require_committed_bundle_unchanged(
    value: object,
    *,
    expected_membership: frozenset[str],
) -> V024CommittedLabelFreeBundle:
    bundle = _require_committed_bundle(value)
    _require_membership(
        bundle._root,
        expected_membership,
        context="Committed label-free operation",
    )
    for filename, digest in bundle._file_hashes:
        if _sha256(_direct_file(bundle._root, filename).read_bytes()) != digest:
            raise V024IOError("A committed label-free artifact changed")
    if _direct_file(bundle._root, "exposure_log.jsonl").read_bytes() != (
        bundle._ledger_prefix
    ):
        raise V024IOError("Committed prediction ledger changed")
    _verify_stored_bundle_frames(
        bundle,
        expected_filenames=(*_LABEL_INPUTS, *_FIT_OUTPUTS),
    )
    return bundle


def _extract_prediction_inputs_v024(
    value: object,
    *,
    model_state: V024CommittedModelStateEnvelope,
) -> tuple[dict[str, pd.DataFrame], V024ContractView]:
    bundle = _require_committed_bundle(value)
    if (
        type(model_state) is not V024CommittedModelStateEnvelope
        or model_state.provenance_sha256 != bundle._model_state.provenance_sha256
    ):
        raise V024IOError("Prediction model does not belong to the label-free bundle")
    bundle = _require_committed_bundle_unchanged(
        bundle,
        expected_membership=_PRE_PREDICTION_FILES,
    )
    return (
        {name: frame.copy(deep=True) for name, frame in bundle._frames},
        bundle._contract_view,
    )


def _exclusive_create(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise V024IOError(f"Formal artifact already exists: {path.name}") from exc
    try:
        written = os.write(descriptor, raw)
        if written != len(raw):
            raise OSError("exclusive write was incomplete")
        os.fsync(descriptor)
    except OSError as exc:
        raise V024IOError(f"Could not write {path.name} exclusively") from exc
    finally:
        os.close(descriptor)
    if path.read_bytes() != raw:
        raise V024IOError(f"{path.name} changed after its exclusive write")


def _write_verified_fit_outputs_v024(
    bundle: V024FreshGenerationBundle,
    *,
    frames: Mapping[str, pd.DataFrame],
) -> None:
    value = _require_fresh_bundle(bundle)
    if set(frames) != set(_FIT_OUTPUTS):
        raise V024IOError("Formal fit output registry changed")
    raw_by_name: dict[str, bytes] = {}
    for filename in _FIT_OUTPUTS:
        try:
            raw_by_name[filename] = canonical_csv_bytes(
                frames[filename],
                value._contract_view.artifacts.csv_schema(filename),
                value._contract_view.artifacts,
                formal=True,
            )
        except V015ArtifactError as exc:
            raise V024IOError(str(exc)) from exc
    for filename in (*_FIT_OUTPUTS, "fit_commitment.json"):
        if (value._root / filename).exists():
            raise V024IOError(f"Formal fit artifact already exists: {filename}")

    for filename in _FIT_OUTPUTS:
        _exclusive_create(value._root / filename, raw_by_name[filename])
    _require_fresh_bundle_unchanged(
        value,
        expected_membership=_FIT_OUTPUT_FILES,
    )

    # A physical fresh read-back is mandatory before the whole-bundle gate.
    for filename in _FIT_OUTPUTS:
        path = _direct_file(value._root, filename)
        try:
            reread = read_canonical_csv(
                path,
                value._contract_view.artifacts,
                formal=True,
            )
            reread_raw = canonical_csv_bytes(
                reread,
                value._contract_view.artifacts.csv_schema(filename),
                value._contract_view.artifacts,
                formal=True,
            )
        except V015ArtifactError as exc:
            raise V024IOError(str(exc)) from exc
        if reread_raw != raw_by_name[filename] or path.read_bytes() != reread_raw:
            raise V024IOError(f"{filename} failed its fresh read-back")


def _create_validated_fit_commitment_v024(
    bundle: V024FreshGenerationBundle,
    *,
    validated_source_hashes: Mapping[str, str],
    created_utc: str,
) -> str:
    value = _require_fresh_bundle_unchanged(
        bundle,
        expected_membership=_FIT_OUTPUT_FILES,
    )
    _utc(created_utc, context="fit commitment created_utc")
    validated_names = (*_LABEL_INPUTS, *_FIT_OUTPUTS)
    if set(validated_source_hashes) != set(validated_names):
        raise V024IOError("Validated whole-bundle source registry changed")
    for filename in validated_names:
        observed = _sha256(_direct_file(value._root, filename).read_bytes())
        if validated_source_hashes[filename] != observed:
            raise V024IOError("Validated whole-bundle source hash changed")
    if (value._root / "fit_commitment.json").exists():
        raise V024IOError("Formal fit artifact already exists: fit_commitment.json")

    entries: list[dict[str, object]] = []
    for filename in _FIT_COMMITMENT_FILES:
        raw = _direct_file(value._root, filename).read_bytes()
        if filename in (*_FIT_OUTPUTS, *_LABEL_INPUTS):
            try:
                row_count = len(
                    read_canonical_csv(
                        value._root / filename,
                        value._contract_view.artifacts,
                        formal=True,
                    )
                )
            except V015ArtifactError as exc:
                raise V024IOError(str(exc)) from exc
        else:
            row_count = 1
        entries.append(
            {
                "path": filename,
                "row_count": row_count,
                "byte_count": len(raw),
                "byte_sha256": _sha256(raw),
            }
        )
    _require_fresh_bundle_unchanged(
        value,
        expected_membership=_FIT_OUTPUT_FILES,
    )
    payload = {
        "protocol_id": value._contract_view.protocol.protocol_id,
        "config_sha256": value._contract_view.artifacts.config_byte_sha256,
        "git_commit": value._identity.git_commit,
        "worker_count": 6,
        "files": entries,
        "created_utc": created_utc,
    }
    if set(payload) != _FIT_COMMITMENT_KEYS:
        raise V024IOError("Fit commitment schema changed")
    raw = canonical_json_bytes(payload)
    _exclusive_create(value._root / "fit_commitment.json", raw)
    _require_fresh_bundle_unchanged(
        value,
        expected_membership=_POST_FIT_FILES,
    )
    return _sha256(raw)


def _write_prediction_outputs_v024(
    bundle: V024CommittedLabelFreeBundle,
    *,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[ArtifactMetadata, ...]:
    value = _require_committed_bundle_unchanged(
        bundle,
        expected_membership=_PRE_PREDICTION_FILES,
    )
    if set(frames) != set(_PREDICTION_OUTPUTS):
        raise V024IOError("Prediction output file registry changed")
    raw_by_name: dict[str, bytes] = {}
    for filename in _PREDICTION_OUTPUTS:
        try:
            raw_by_name[filename] = canonical_csv_bytes(
                frames[filename],
                value._artifact_contract.csv_schema(filename),
                value._artifact_contract,
                formal=True,
            )
        except V015ArtifactError as exc:
            raise V024IOError(str(exc)) from exc
        if (value._root / filename).exists():
            raise V024IOError(f"Prediction artifact already exists: {filename}")
    _require_committed_bundle_unchanged(
        value,
        expected_membership=_PRE_PREDICTION_FILES,
    )
    metadata: list[ArtifactMetadata] = []
    for filename in _PREDICTION_OUTPUTS:
        raw = raw_by_name[filename]
        _exclusive_create(value._root / filename, raw)
        metadata.append(
            ArtifactMetadata(
                path=filename,
                row_count=len(frames[filename]),
                byte_count=len(raw),
                byte_sha256=_sha256(raw),
            )
        )
    _require_committed_bundle_unchanged(
        value,
        expected_membership=_PREDICTION_FILES,
    )
    return tuple(metadata)


def _file_metadata(
    root: Path,
    *,
    contract: FrozenArtifactContract,
) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    entries: list[dict[str, object]] = []
    row_counts: dict[str, int] = {}
    for filename in _COMMITMENT_FILE_REGISTRY:
        path = _direct_file(root, filename)
        raw = path.read_bytes()
        if filename.endswith(".csv"):
            try:
                rows = len(read_canonical_csv(path, contract, formal=True))
            except V015ArtifactError as exc:
                raise V024IOError(str(exc)) from exc
            row_counts[filename] = rows
        else:
            rows = 1
        entries.append(
            {
                "path": filename,
                "row_count": rows,
                "byte_count": len(raw),
                "byte_sha256": _sha256(raw),
            }
        )
    return tuple(entries), row_counts


def _artifact_set_digest(entries: Sequence[Mapping[str, object]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(_ARTIFACT_SET_DOMAIN)
    for entry in entries:
        name = str(entry["path"]).encode("ascii")
        hasher.update(struct.pack("<Q", len(name)))
        hasher.update(name)
        hasher.update(struct.pack("<Q", int(entry["row_count"])))
        hasher.update(struct.pack("<Q", int(entry["byte_count"])))
        hasher.update(bytes.fromhex(str(entry["byte_sha256"])))
    return hasher.hexdigest()


def _verify_file_entries_current(
    root: Path,
    entries: Sequence[Mapping[str, object]],
) -> None:
    if tuple(entry.get("path") for entry in entries) != (_COMMITMENT_FILE_REGISTRY):
        raise V024IOError("Prediction commitment file registry changed")
    for entry in entries:
        filename = str(entry["path"])
        raw = _direct_file(root, filename).read_bytes()
        if entry.get("byte_count") != len(raw) or entry.get("byte_sha256") != _sha256(
            raw
        ):
            raise V024IOError(f"Prediction commitment input changed: {filename}")


def create_prediction_commitment_v024(
    bundle: V024CommittedLabelFreeBundle,
    *,
    created_utc: str,
) -> V024PredictionCommitmentEvidence:
    """Exclusively commit all fit, state, and prediction bytes."""

    value = _require_committed_bundle_unchanged(
        bundle,
        expected_membership=_PREDICTION_FILES,
    )
    _utc(created_utc, context="prediction commitment created_utc")
    ledger_raw = _direct_file(value._root, "exposure_log.jsonl").read_bytes()
    all_frames = {
        **{name: frame.copy(deep=True) for name, frame in value._frames},
        **_read_frames(
            value._root,
            contract=value._artifact_contract,
            filenames=_PREDICTION_OUTPUTS,
        ),
    }
    try:
        validate_prediction_artifact_bundle(
            all_frames,
            value._artifact_contract,
            formal=True,
        )
    except V015ArtifactError as exc:
        raise V024IOError(str(exc)) from exc
    entries, row_counts = _file_metadata(
        value._root,
        contract=value._artifact_contract,
    )
    artifact_set = _artifact_set_digest(entries)
    actual_analysis_hash = _sha256(
        _direct_file(
            value._root,
            _ACTUAL_ANALYSIS_HASH_FILENAME,
        ).read_bytes()
    )
    provenance_commitments = value._model_state.validated_model_state.training_provenance.commitment_byte_sha256
    if (
        provenance_commitments.get("actual_analysis_hash_ledger")
        != actual_analysis_hash
    ):
        raise V024IOError(
            "Model provenance does not bind the actual-analysis hash ledger"
        )
    payload = {
        "schema_version": "1.0.0",
        "protocol_id": value._artifact_contract.protocol_id,
        "config_sha256": value._config_sha256,
        "attempt_id": value.attempt_id,
        "git_commit": value._identity.git_commit,
        "model_state_commitment_byte_sha256": (
            value._model_state.model_state_commitment_artifact_byte_sha256
        ),
        "actual_analysis_hash_ledger_commitment_byte_sha256": (actual_analysis_hash),
        "files": list(entries),
        "row_counts": row_counts,
        "artifact_set_sha256": artifact_set,
        "ledger_prefix_event_count": value._ledger_event_count,
        "ledger_prefix_byte_sha256": _sha256(ledger_raw),
        "ledger_phase": {
            "completed_phase": "model_state_committed",
            "pending_phase": "prediction_started",
        },
        "created_utc": created_utc,
        "sealed_truth_opened_before_commitment": False,
    }
    if set(payload) != _PREDICTION_COMMITMENT_KEYS:
        raise V024IOError("Prediction commitment schema changed")
    raw = canonical_json_bytes(payload)
    _require_committed_bundle_unchanged(
        value,
        expected_membership=_PREDICTION_FILES,
    )
    _verify_file_entries_current(value._root, entries)
    target = value._root / "prediction_commitment.json"
    _exclusive_create(target, raw)
    _require_committed_bundle_unchanged(
        value,
        expected_membership=_POST_PREDICTION_FILES,
    )
    _verify_file_entries_current(value._root, entries)
    return _issue_prediction_commitment_evidence(
        attempt_id=value.attempt_id,
        byte_sha256=_sha256(raw),
        artifact_set_sha256=artifact_set,
        actual_analysis_hash_ledger_commitment_byte_sha256=(actual_analysis_hash),
        file_entries=entries,
        ledger_committed=False,
    )


def verify_prediction_commitment_v024(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    contract_view: V024ContractView | None = None,
    require_ledger_committed: bool,
) -> V024PredictionCommitmentEvidence:
    """Verify prediction bytes and their precommit/current ledger bindings."""

    view = _require_contract(contract_view)
    full_chain_bundle, progress = _load_committed_bundle(
        label_free_root=label_free_root,
        attempt_id=attempt_id,
        contract_view=view,
        expected_membership=_POST_PREDICTION_FILES,
    )
    root = full_chain_bundle._root
    ledger_raw = _direct_file(root, "exposure_log.jsonl").read_bytes()
    event_count = len(ledger_raw.splitlines())
    commitment_raw = _direct_file(root, "prediction_commitment.json").read_bytes()
    payload = _strict_json(
        commitment_raw,
        filename="prediction_commitment.json",
    )
    if (
        set(payload) != _PREDICTION_COMMITMENT_KEYS
        or payload.get("schema_version") != "1.0.0"
        or payload.get("protocol_id") != view.protocol.protocol_id
        or payload.get("config_sha256") != view.artifacts.config_byte_sha256
        or payload.get("attempt_id") != attempt_id
        or payload.get("git_commit") != progress.identity.git_commit
        or payload.get("sealed_truth_opened_before_commitment") is not False
    ):
        raise V024IOError("Prediction commitment identity changed")
    _utc(payload.get("created_utc"), context="prediction commitment created_utc")
    prefix_count = _positive_int(
        payload.get("ledger_prefix_event_count"),
        context="ledger prefix event count",
    )
    lines = ledger_raw.splitlines(keepends=True)
    if prefix_count > len(lines):
        raise V024IOError("Prediction commitment ledger prefix is unavailable")
    prefix = b"".join(lines[:prefix_count])
    if payload.get("ledger_prefix_byte_sha256") != _sha256(prefix):
        raise V024IOError("Prediction commitment ledger prefix changed")
    if payload.get("ledger_phase") != {
        "completed_phase": "model_state_committed",
        "pending_phase": "prediction_started",
    }:
        raise V024IOError("Prediction commitment precommit phase changed")

    entries, row_counts = _file_metadata(root, contract=view.artifacts)
    if payload.get("files") != list(entries) or payload.get("row_counts") != row_counts:
        raise V024IOError("Prediction commitment artifact metadata changed")
    artifact_set = _artifact_set_digest(entries)
    if payload.get("artifact_set_sha256") != artifact_set:
        raise V024IOError("Prediction commitment artifact-set digest changed")
    model_commitment_hash = _sha256(
        _direct_file(root, "model_state_commitment.json").read_bytes()
    )
    if payload.get("model_state_commitment_byte_sha256") != model_commitment_hash:
        raise V024IOError("Prediction commitment model-state binding changed")
    actual_analysis_hash = _sha256(
        _direct_file(root, _ACTUAL_ANALYSIS_HASH_FILENAME).read_bytes()
    )
    if (
        payload.get("actual_analysis_hash_ledger_commitment_byte_sha256")
        != actual_analysis_hash
        or progress.actual_analysis_hash_ledger_commitment_byte_sha256
        != actual_analysis_hash
    ):
        raise V024IOError("Prediction commitment actual-analysis binding changed")
    commitment_hash = _sha256(commitment_raw)
    ledger_committed = (
        progress.completed_phase == "prediction_committed"
        and progress.pending_phase is None
        and progress.prediction_commitment_byte_sha256 == commitment_hash
    )
    if require_ledger_committed and not ledger_committed:
        raise V024IOError("Prediction commitment is absent from the completed ledger")
    if not ledger_committed and (
        progress.completed_phase != "model_state_committed"
        or progress.pending_phase != "prediction_started"
        or event_count != prefix_count
    ):
        raise V024IOError("Prediction commitment and ledger phase disagree")
    return _issue_prediction_commitment_evidence(
        attempt_id=attempt_id,
        byte_sha256=commitment_hash,
        artifact_set_sha256=artifact_set,
        actual_analysis_hash_ledger_commitment_byte_sha256=(actual_analysis_hash),
        file_entries=entries,
        ledger_committed=ledger_committed,
    )


__all__ = [
    "V024CommittedLabelFreeBundle",
    "V024FreshGenerationBundle",
    "V024IOError",
    "V024PredictionCommitmentEvidence",
    "create_actual_analysis_hash_ledger_commitment_v024",
    "create_prediction_commitment_v024",
    "load_committed_label_free_bundle_v024",
    "load_fresh_generation_bundle_v024",
    "verify_prediction_commitment_v024",
]
