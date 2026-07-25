"""Formal-attempt state machine and scoped truth access for V0.15.

The protocol-level artifact helpers validate bytes.  This module adds the
temporal guarantees needed by the formal run: one immutable implementation
identity per attempt, monotone commitments, exact phase transitions, and
phase-scoped access to committed truth files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    append_exposure_event,
    assert_separate_truth_roots,
    read_canonical_csv,
    read_exposure_log,
    read_truth_commitments,
    verify_prediction_commitment,
)


class V015FirewallError(ValueError):
    """Raised when a formal attempt crosses a frozen firewall boundary."""


PHASES = (
    "before_generation",
    "truth_committed",
    "label_free_fit_committed",
    "center_truth_opened",
    "center_state_committed",
    "risk_truth_opened",
    "risk_state_committed",
    "calibration_truth_opened",
    "model_state_committed",
    "prediction_started",
    "prediction_committed",
    "scoring_truth_opened",
    "scoring_completed",
)
EXIT_STATUSES = frozenset({"started", "completed", "interrupted", "failed"})

_CENTER_TRUTH = ("center_development_truth.csv",)
_RISK_TRUTH = _CENTER_TRUTH + ("risk_development_truth.csv",)
_CALIBRATION_TRUTH = _RISK_TRUTH + ("calibration_truth.csv",)
_SCORING_TRUTH = _CALIBRATION_TRUTH + (
    "test_truth.csv",
    "audit_truth.csv",
    "intrinsic_matched_truth.csv",
    "stress_plan_matched_truth.csv",
    "intrinsic_matched_pairs.csv",
    "stress_plan_matched_pairs.csv",
)
_OPENED_BY_PHASE = {
    "before_generation": (),
    "truth_committed": (),
    "label_free_fit_committed": (),
    "center_truth_opened": _CENTER_TRUTH,
    "center_state_committed": _CENTER_TRUTH,
    "risk_truth_opened": _RISK_TRUTH,
    "risk_state_committed": _RISK_TRUTH,
    "calibration_truth_opened": _CALIBRATION_TRUTH,
    "model_state_committed": _CALIBRATION_TRUTH,
    "prediction_started": _CALIBRATION_TRUTH,
    "prediction_committed": _CALIBRATION_TRUTH,
    "scoring_truth_opened": _SCORING_TRUTH,
    "scoring_completed": _SCORING_TRUTH,
}
_OPEN_PHASE_FILES = {
    "center_truth_opened": ("center_development_truth.csv",),
    "risk_truth_opened": ("risk_development_truth.csv",),
    "calibration_truth_opened": ("calibration_truth.csv",),
    # The final scorer rereads all nine committed sealed files so the complete
    # cross-file truth/mapping contract can be revalidated after commitment.
    "scoring_truth_opened": _SCORING_TRUTH,
}
_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHASE_COMMITMENT_FIELDS = {
    "label_free_fit_committed": "fit_commitment_byte_sha256",
    "center_state_committed": "center_state_checkpoint_byte_sha256",
    "risk_state_committed": "risk_state_checkpoint_byte_sha256",
    "model_state_committed": "model_state_commitment_byte_sha256",
}


@dataclass(frozen=True)
class FormalAttemptIdentity:
    """Implementation identity that may not change within one attempt."""

    attempt_id: str
    git_commit: str
    config_byte_sha256: str

    def __post_init__(self) -> None:
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise V015FirewallError("attempt_id is not a safe stable identifier")
        if _HEX_COMMIT.fullmatch(self.git_commit) is None:
            raise V015FirewallError("git_commit must be a full lowercase hash")
        if re.fullmatch(r"[0-9a-f]{64}", self.config_byte_sha256) is None:
            raise V015FirewallError("config_byte_sha256 is invalid")


@dataclass(frozen=True)
class AttemptProgress:
    """Validated state recovered solely from the append-only ledger."""

    identity: FormalAttemptIdentity
    completed_phase: str
    pending_phase: str | None
    truth_commitments_byte_sha256: str | None
    prediction_commitment_byte_sha256: str | None
    opened_truth_files: tuple[str, ...]
    terminal_failed: bool
    fit_commitment_byte_sha256: str | None = None
    center_state_checkpoint_byte_sha256: str | None = None
    risk_state_checkpoint_byte_sha256: str | None = None
    model_state_commitment_byte_sha256: str | None = None


def phase_commitment_message(phase: str, byte_sha256: str) -> str:
    """Return the only valid completed-message encoding for a checkpoint."""

    try:
        field = _PHASE_COMMITMENT_FIELDS[phase]
    except KeyError as exc:
        raise V015FirewallError(
            f"{phase!r} is not a phase-specific commitment"
        ) from exc
    if not isinstance(byte_sha256, str) or _SHA256.fullmatch(byte_sha256) is None:
        raise V015FirewallError("Phase commitment is not lowercase SHA256")
    return f"{field}={byte_sha256}"


def _parse_phase_commitment_message(phase: str, message: str) -> str:
    field = _PHASE_COMMITMENT_FIELDS[phase]
    match = re.fullmatch(rf"{re.escape(field)}=([0-9a-f]{{64}})", message)
    if match is None:
        raise V015FirewallError(
            f"Completed {phase} lacks its exact machine-readable commitment"
        )
    return match.group(1)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_formal_attempt_environment(
    identity: FormalAttemptIdentity,
) -> None:
    """Bind a capability call to the clean implementation in its ledger."""

    from lifetwin.experiments.calendar_long_horizon_v015_environment import (
        verify_formal_environment,
    )

    project_root = Path(__file__).resolve().parents[3]
    environment = verify_formal_environment(project_root)
    if (
        environment.git_dirty
        or environment.git_commit != identity.git_commit
        or environment.config_byte_sha256 != identity.config_byte_sha256
    ):
        raise V015FirewallError(
            "Current formal environment differs from the attempt identity"
        )


def _validate_event_shape(
    event: Mapping[str, Any], *, contract: FrozenArtifactContract
) -> None:
    if set(event) != contract.exposure_keys:
        raise V015FirewallError("Exposure event keys differ from the freeze")
    if (
        not isinstance(event.get("attempt_id"), str)
        or _ATTEMPT_ID.fullmatch(event["attempt_id"]) is None
        or not isinstance(event.get("git_commit"), str)
        or _HEX_COMMIT.fullmatch(event["git_commit"]) is None
        or not isinstance(event.get("git_dirty"), bool)
        or event["git_dirty"] is not False
        or event.get("config_byte_sha256") != contract.config_byte_sha256
    ):
        raise V015FirewallError(
            "Exposure event has an invalid formal implementation identity"
        )
    timestamp = event.get("created_utc")
    if not isinstance(timestamp, str):
        raise V015FirewallError("Exposure event timestamp is invalid")
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise V015FirewallError("Exposure event timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise V015FirewallError("Exposure event timestamp lacks a timezone")
    for key in (
        "truth_commitments_byte_sha256",
        "prediction_commitment_byte_sha256",
    ):
        value = event.get(key)
        if value is not None and (
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
        ):
            raise V015FirewallError(f"Exposure event {key} is invalid")
    opened = event.get("opened_truth_files")
    if (
        not isinstance(opened, list)
        or any(not isinstance(item, str) for item in opened)
        or opened != sorted(set(opened))
        or any(item not in contract.sealed_filenames for item in opened)
    ):
        raise V015FirewallError("Exposure event opened-truth list is invalid")
    for key in ("phase", "exit_status", "message"):
        if not isinstance(event.get(key), str):
            raise V015FirewallError(f"Exposure event {key} is invalid")


def _validate_attempt_events(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: FrozenArtifactContract,
) -> AttemptProgress:
    if not events:
        raise V015FirewallError("A formal attempt has no exposure events")
    for event in events:
        _validate_event_shape(event, contract=contract)
    first = events[0]
    identity = FormalAttemptIdentity(
        attempt_id=str(first["attempt_id"]),
        git_commit=str(first["git_commit"]),
        config_byte_sha256=str(first["config_byte_sha256"]),
    )
    if identity.config_byte_sha256 != contract.config_byte_sha256:
        raise V015FirewallError("Attempt config hash differs from frozen V2")

    completed_index = -1
    pending_phase: str | None = None
    truth_hash: str | None = None
    prediction_hash: str | None = None
    opened: tuple[str, ...] = ()
    terminal_failed = False
    phase_commitments: dict[str, str] = {}

    for position, event in enumerate(events):
        if str(event["attempt_id"]) != identity.attempt_id:
            raise V015FirewallError("Attempt event was grouped under the wrong ID")
        if (
            str(event["git_commit"]) != identity.git_commit
            or str(event["config_byte_sha256"]) != identity.config_byte_sha256
            or event["git_dirty"] is not False
        ):
            raise V015FirewallError(
                "Formal attempt implementation identity changed or was dirty"
            )
        if terminal_failed:
            raise V015FirewallError("A failed formal attempt has later events")

        phase = str(event["phase"])
        status = str(event["exit_status"])
        if phase not in PHASES or status not in EXIT_STATUSES:
            raise V015FirewallError("Exposure phase or exit_status is not frozen")
        expected_index = completed_index + 1
        if expected_index >= len(PHASES):
            raise V015FirewallError("A completed attempt has later events")
        expected_phase = PHASES[expected_index]
        if phase != expected_phase:
            raise V015FirewallError(
                f"Illegal phase transition to {phase!r}; expected {expected_phase!r}"
            )
        if pending_phase is not None and phase != pending_phase:
            raise V015FirewallError("An interrupted or started phase was skipped")

        event_opened = tuple(event["opened_truth_files"])
        expected_opened = tuple(sorted(_OPENED_BY_PHASE[phase]))
        if event_opened != expected_opened:
            raise V015FirewallError(
                f"{phase} opened-truth set differs from the frozen phase capability"
            )
        if not set(opened).issubset(event_opened):
            raise V015FirewallError("A formal attempt forgot an opened truth file")
        opened = event_opened

        observed_truth_hash = event["truth_commitments_byte_sha256"]
        if phase == "before_generation":
            if observed_truth_hash is not None:
                raise V015FirewallError(
                    "Truth commitment cannot exist before generation"
                )
        elif status == "completed" or truth_hash is not None:
            if not isinstance(observed_truth_hash, str):
                raise V015FirewallError(
                    "Completed post-generation phase lacks truth commitment"
                )
        if observed_truth_hash is not None:
            if truth_hash is None:
                truth_hash = str(observed_truth_hash)
            elif observed_truth_hash != truth_hash:
                raise V015FirewallError("Truth commitment changed within an attempt")

        observed_prediction_hash = event["prediction_commitment_byte_sha256"]
        prediction_phase_index = PHASES.index("prediction_committed")
        if expected_index < prediction_phase_index:
            if observed_prediction_hash is not None:
                raise V015FirewallError(
                    "Prediction commitment appeared before its frozen phase"
                )
        elif status == "completed" or prediction_hash is not None:
            if not isinstance(observed_prediction_hash, str):
                raise V015FirewallError(
                    "Completed post-prediction phase lacks prediction commitment"
                )
        if observed_prediction_hash is not None:
            if prediction_hash is None:
                prediction_hash = str(observed_prediction_hash)
            elif observed_prediction_hash != prediction_hash:
                raise V015FirewallError(
                    "Prediction commitment changed within an attempt"
                )

        if position == 0 and (phase != "before_generation" or status != "completed"):
            raise V015FirewallError("The first event must checkpoint before generation")
        if status == "completed" and phase in _PHASE_COMMITMENT_FIELDS:
            phase_commitments[phase] = _parse_phase_commitment_message(
                phase, str(event["message"])
            )
        if status == "completed":
            completed_index = expected_index
            pending_phase = None
        elif status in {"started", "interrupted"}:
            pending_phase = phase
        else:
            terminal_failed = True

    completed_phase = PHASES[completed_index] if completed_index >= 0 else ""
    return AttemptProgress(
        identity=identity,
        completed_phase=completed_phase,
        pending_phase=pending_phase,
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        opened_truth_files=opened,
        terminal_failed=terminal_failed,
        fit_commitment_byte_sha256=phase_commitments.get("label_free_fit_committed"),
        center_state_checkpoint_byte_sha256=phase_commitments.get(
            "center_state_committed"
        ),
        risk_state_checkpoint_byte_sha256=phase_commitments.get("risk_state_committed"),
        model_state_commitment_byte_sha256=phase_commitments.get(
            "model_state_committed"
        ),
    )


def validate_formal_exposure_events(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: FrozenArtifactContract,
) -> Mapping[str, AttemptProgress]:
    """Validate all interleaved attempts in one publication ledger."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        _validate_event_shape(event, contract=contract)
        attempt_id = str(event["attempt_id"])
        grouped.setdefault(attempt_id, []).append(event)
    return {
        attempt_id: _validate_attempt_events(items, contract=contract)
        for attempt_id, items in grouped.items()
    }


def validate_formal_exposure_log(
    path: str | Path,
    contract: FrozenArtifactContract,
) -> Mapping[str, AttemptProgress]:
    """Read the canonical JSONL ledger and enforce the formal state machine."""

    events = read_exposure_log(path, contract)
    return validate_formal_exposure_events(events, contract=contract)


def append_formal_exposure_event(
    *,
    path: str | Path,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    created_utc: str,
    phase: str,
    exit_status: str,
    truth_commitments_byte_sha256: str | None,
    prediction_commitment_byte_sha256: str | None,
    message: str,
) -> AttemptProgress:
    """Validate a prospective transition before appending its canonical bytes."""

    event = {
        "attempt_id": identity.attempt_id,
        "created_utc": created_utc,
        "git_commit": identity.git_commit,
        "git_dirty": False,
        "config_byte_sha256": identity.config_byte_sha256,
        "phase": phase,
        "truth_commitments_byte_sha256": truth_commitments_byte_sha256,
        "prediction_commitment_byte_sha256": prediction_commitment_byte_sha256,
        "opened_truth_files": list(sorted(_OPENED_BY_PHASE.get(phase, ()))),
        "exit_status": exit_status,
        "message": message,
    }
    existing = read_exposure_log(path, contract)
    prospective = (*existing, event)
    states = validate_formal_exposure_events(prospective, contract=contract)
    try:
        append_exposure_event(path, event, contract)
    except V015ArtifactError as exc:
        raise V015FirewallError(str(exc)) from exc
    return states[identity.attempt_id]


def append_phase_error_without_masking(
    *,
    error: BaseException,
    ledger_path: str | Path,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    created_utc: str,
    phase: str,
    exit_status: str,
    truth_commitments_byte_sha256: str | None,
    prediction_commitment_byte_sha256: str | None,
    message: str,
) -> None:
    try:
        append_formal_exposure_event(
            path=ledger_path,
            identity=identity,
            contract=contract,
            created_utc=created_utc,
            phase=phase,
            exit_status=exit_status,
            truth_commitments_byte_sha256=(truth_commitments_byte_sha256),
            prediction_commitment_byte_sha256=(prediction_commitment_byte_sha256),
            message=message,
        )
    except BaseException as checkpoint_error:
        error.add_note(
            "The original phase error was preserved, but its terminal "
            f"{exit_status!r} exposure checkpoint also failed: "
            f"{checkpoint_error!r}"
        )
        raise error from checkpoint_error


def open_truth_for_phase(
    *,
    ledger_path: str | Path,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    commitment_path: str | Path,
    sealed_truth_root: str | Path,
    label_free_root: str | Path,
    phase: str,
    created_utc: str,
    prediction_commitment_path: str | Path | None = None,
    formal: bool = True,
) -> Mapping[str, pd.DataFrame]:
    """Open only the truth files authorized by one frozen reveal phase.

    The exposure checkpoint is appended before any sealed file is read.  Test,
    audit, and matched truth additionally require the exact prediction
    commitment bytes to have been checkpointed already.
    """

    if phase not in _OPEN_PHASE_FILES:
        raise V015FirewallError("This phase has no truth-access capability")
    if formal:
        verify_formal_attempt_environment(identity)
    label_root, sealed_root = assert_separate_truth_roots(
        label_free_root, sealed_truth_root
    )
    commitment = Path(commitment_path).resolve()
    if commitment.parent != label_root:
        raise V015FirewallError(
            "truth_commitments.json must live in the label-free root"
        )
    states = validate_formal_exposure_log(ledger_path, contract)
    try:
        prior = states[identity.attempt_id]
    except KeyError as exc:
        raise V015FirewallError(
            "Attempt has not checkpointed before generation"
        ) from exc
    if prior.identity != identity:
        raise V015FirewallError("Truth capability identity differs from the ledger")
    truth_hash = prior.truth_commitments_byte_sha256
    if truth_hash is None:
        raise V015FirewallError(
            "Truth capability lacks the checkpointed truth commitment"
        )

    prediction_hash: str | None = None
    if phase == "scoring_truth_opened":
        if prediction_commitment_path is None:
            raise V015FirewallError(
                "Scoring truth requires a prediction commitment path"
            )
        prediction_path = Path(prediction_commitment_path).resolve()
        if (
            prediction_path.parent != label_root
            or prediction_path.name != "prediction_commitment.json"
        ):
            raise V015FirewallError(
                "Prediction commitment must live in the label-free root"
            )
        prediction_hash = prior.prediction_commitment_byte_sha256
        if prediction_hash is None:
            raise V015FirewallError(
                "Scoring capability lacks a checkpointed prediction"
            )
    elif prediction_commitment_path is not None:
        raise V015FirewallError(
            "Development reveal cannot receive a prediction commitment path"
        )

    append_formal_exposure_event(
        path=ledger_path,
        identity=identity,
        contract=contract,
        created_utc=created_utc,
        phase=phase,
        exit_status="started",
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        message=f"Started the exact truth capability for {phase}.",
    )

    try:
        if _sha256_file(commitment) != truth_hash:
            raise V015FirewallError(
                "Truth capability does not match its ledger commitment"
            )
        if phase == "scoring_truth_opened":
            assert prediction_commitment_path is not None
            prediction_path = Path(prediction_commitment_path).resolve()
            if _sha256_file(prediction_path) != prediction_hash:
                raise V015FirewallError(
                    "Scoring capability does not match the checkpointed prediction"
                )
            try:
                verify_prediction_commitment(
                    commitment_path=prediction_path,
                    label_free_root=label_root,
                    contract=contract,
                    formal=formal,
                )
            except V015ArtifactError as exc:
                raise V015FirewallError(
                    "Scoring capability failed prediction commitment verification"
                ) from exc

        payload = read_truth_commitments(commitment, contract, formal=formal)
        entries = {str(item["path"]): item for item in payload["files"]}
        opened: dict[str, pd.DataFrame] = {}
        for filename in _OPEN_PHASE_FILES[phase]:
            entry = entries[filename]
            path = (sealed_root / filename).resolve()
            if path.parent != sealed_root:
                raise V015FirewallError("Sealed truth path escaped its root")
            frame = read_canonical_csv(path, contract, formal=formal)
            raw = path.read_bytes()
            if (
                len(frame) != int(entry["row_count"])
                or len(raw) != int(entry["byte_count"])
                or hashlib.sha256(raw).hexdigest() != entry["byte_sha256"]
            ):
                raise V015FirewallError(
                    f"{filename} differs from its pre-reveal commitment"
                )
            opened[filename] = frame
    except KeyboardInterrupt as exc:
        append_phase_error_without_masking(
            error=exc,
            ledger_path=ledger_path,
            identity=identity,
            contract=contract,
            created_utc=created_utc,
            phase=phase,
            exit_status="interrupted",
            truth_commitments_byte_sha256=truth_hash,
            prediction_commitment_byte_sha256=prediction_hash,
            message=f"Truth capability for {phase} was interrupted.",
        )
        raise
    except BaseException as exc:
        append_phase_error_without_masking(
            error=exc,
            ledger_path=ledger_path,
            identity=identity,
            contract=contract,
            created_utc=created_utc,
            phase=phase,
            exit_status="failed",
            truth_commitments_byte_sha256=truth_hash,
            prediction_commitment_byte_sha256=prediction_hash,
            message=f"Truth capability for {phase} failed.",
        )
        raise

    append_formal_exposure_event(
        path=ledger_path,
        identity=identity,
        contract=contract,
        created_utc=created_utc,
        phase=phase,
        exit_status="completed",
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=prediction_hash,
        message=f"Opened the exact truth capability for {phase}.",
    )
    return opened


def reopen_authorized_truth_for_recovery(
    *,
    ledger_path: str | Path,
    identity: FormalAttemptIdentity,
    contract: FrozenArtifactContract,
    commitment_path: str | Path,
    sealed_truth_root: str | Path,
    label_free_root: str | Path,
    filenames: Sequence[str],
    formal: bool = True,
) -> Mapping[str, pd.DataFrame]:
    """Reread only truth already exposed for a pending recovery phase.

    The caller must first append ``started`` for the interrupted/incomplete
    state-commit or scoring-completion phase.  This helper creates no new
    capability and therefore does not advance the ledger.
    """

    requested = tuple(filenames)
    if (
        not requested
        or any(not isinstance(name, str) for name in requested)
        or len(set(requested)) != len(requested)
        or any(name not in contract.sealed_filenames for name in requested)
    ):
        raise V015FirewallError("Recovery truth-file request is invalid")
    if formal:
        verify_formal_attempt_environment(identity)
    label_root, sealed_root = assert_separate_truth_roots(
        label_free_root, sealed_truth_root
    )
    commitment = Path(commitment_path).resolve()
    if commitment.parent != label_root or commitment.name != "truth_commitments.json":
        raise V015FirewallError(
            "truth_commitments.json must live in the label-free root"
        )
    states = validate_formal_exposure_log(ledger_path, contract)
    try:
        progress = states[identity.attempt_id]
    except KeyError as exc:
        raise V015FirewallError("Recovery attempt is absent from the ledger") from exc
    if progress.identity != identity or progress.terminal_failed:
        raise V015FirewallError(
            "Recovery identity is invalid or the attempt is terminal"
        )
    if progress.pending_phase not in {
        "center_state_committed",
        "risk_state_committed",
        "model_state_committed",
        "scoring_completed",
    }:
        raise V015FirewallError(
            "Truth recovery requires a pending deterministic state/score phase"
        )
    if not set(requested).issubset(progress.opened_truth_files):
        raise V015FirewallError("Recovery requested truth that was not already exposed")
    truth_hash = progress.truth_commitments_byte_sha256
    if truth_hash is None or _sha256_file(commitment) != truth_hash:
        raise V015FirewallError(
            "Recovery truth commitment differs from the attempt ledger"
        )

    payload = read_truth_commitments(commitment, contract, formal=formal)
    entries = {str(item["path"]): item for item in payload["files"]}
    opened: dict[str, pd.DataFrame] = {}
    for filename in requested:
        entry = entries[filename]
        path = (sealed_root / filename).resolve()
        if path.parent != sealed_root:
            raise V015FirewallError("Recovery truth path escaped its root")
        frame = read_canonical_csv(path, contract, formal=formal)
        raw = path.read_bytes()
        if (
            len(frame) != int(entry["row_count"])
            or len(raw) != int(entry["byte_count"])
            or hashlib.sha256(raw).hexdigest() != entry["byte_sha256"]
        ):
            raise V015FirewallError(f"{filename} differs from its recovery commitment")
        opened[filename] = frame
    return opened


def verify_phase_artifact_commitment(
    progress: AttemptProgress,
    *,
    phase: str,
    artifact_path: str | Path,
) -> str:
    """Verify one checkpoint file against its ledger-bound byte digest."""

    try:
        field = _PHASE_COMMITMENT_FIELDS[phase]
    except KeyError as exc:
        raise V015FirewallError(
            f"{phase!r} is not a phase-specific commitment"
        ) from exc
    expected = getattr(progress, field)
    if expected is None:
        raise V015FirewallError(f"{phase} has no completed commitment")
    observed = _sha256_file(artifact_path)
    if observed != expected:
        raise V015FirewallError(f"{phase} artifact differs from its ledger commitment")
    return observed


__all__ = [
    "AttemptProgress",
    "EXIT_STATUSES",
    "FormalAttemptIdentity",
    "PHASES",
    "V015FirewallError",
    "append_phase_error_without_masking",
    "append_formal_exposure_event",
    "open_truth_for_phase",
    "phase_commitment_message",
    "reopen_authorized_truth_for_recovery",
    "verify_formal_attempt_environment",
    "validate_formal_exposure_events",
    "validate_formal_exposure_log",
    "verify_phase_artifact_commitment",
]
