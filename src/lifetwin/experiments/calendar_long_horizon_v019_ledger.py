"""Strict, capability-free V2.4 exposure-ledger parsing and locking.

This module intentionally has no dataframe, artifact, truth, model, or scoring
imports.  Both the information firewall and terminal publisher use this single
parser so that an evidence path cannot quietly accept a weaker JSONL dialect.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence


class V024LedgerError(ValueError):
    """Raised when the V2.4 append-only ledger is not exact."""


PHASES = (
    "before_generation",
    "generation_plan_committed",
    "truth_committed",
    "actual_analysis_hash_ledger_committed",
    "label_free_fit_committed",
    "center_truth_opened",
    "center_state_committed",
    "risk_truth_opened",
    "risk_state_committed",
    "calibration_mask_committed",
    "calibration_truth_opened",
    "model_state_committed",
    "prediction_started",
    "prediction_committed",
    "scoring_truth_opened",
    "scoring_completed",
)
EXIT_STATUSES = frozenset({"started", "completed", "interrupted", "failed"})

EXPOSURE_EVENT_KEYS = frozenset(
    {
        "attempt_id",
        "created_utc",
        "git_commit",
        "git_dirty",
        "config_byte_sha256",
        "phase",
        "truth_commitments_byte_sha256",
        "prediction_commitment_byte_sha256",
        "opened_truth_files",
        "exit_status",
        "message",
    }
)

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
OPENED_BY_PHASE = {
    "before_generation": (),
    "generation_plan_committed": (),
    "truth_committed": (),
    "actual_analysis_hash_ledger_committed": (),
    "label_free_fit_committed": (),
    "center_truth_opened": _CENTER_TRUTH,
    "center_state_committed": _CENTER_TRUTH,
    "risk_truth_opened": _RISK_TRUTH,
    "risk_state_committed": _RISK_TRUTH,
    "calibration_mask_committed": _RISK_TRUTH,
    "calibration_truth_opened": _CALIBRATION_TRUTH,
    "model_state_committed": _CALIBRATION_TRUTH,
    "prediction_started": _CALIBRATION_TRUTH,
    "prediction_committed": _CALIBRATION_TRUTH,
    "scoring_truth_opened": _SCORING_TRUTH,
    "scoring_completed": _SCORING_TRUTH,
}
PHASE_COMMITMENT_FIELDS = {
    "generation_plan_committed": "generation_plan_commitment_byte_sha256",
    "actual_analysis_hash_ledger_committed": (
        "actual_analysis_hash_ledger_commitment_byte_sha256"
    ),
    "label_free_fit_committed": "fit_commitment_byte_sha256",
    "center_state_committed": "center_state_checkpoint_byte_sha256",
    "risk_state_committed": "risk_state_checkpoint_byte_sha256",
    "calibration_mask_committed": "calibration_mask_commitment_byte_sha256",
    "model_state_committed": "model_state_commitment_byte_sha256",
}

_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BINARY_FLAG = getattr(os, "O_BINARY", 0)


@dataclass(frozen=True)
class FormalAttemptIdentity:
    """Immutable implementation identity for one V2.4 attempt."""

    attempt_id: str
    git_commit: str
    config_byte_sha256: str

    def __post_init__(self) -> None:
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise V024LedgerError("attempt_id is not a safe stable identifier")
        if _HEX_COMMIT.fullmatch(self.git_commit) is None:
            raise V024LedgerError("git_commit must be a full lowercase hash")
        if _SHA256.fullmatch(self.config_byte_sha256) is None:
            raise V024LedgerError("config_byte_sha256 is invalid")


@dataclass(frozen=True)
class AttemptProgress:
    """State reconstructed only from strict canonical exposure events."""

    identity: FormalAttemptIdentity
    completed_phase: str
    pending_phase: str | None
    truth_commitments_byte_sha256: str | None
    prediction_commitment_byte_sha256: str | None
    opened_truth_files: tuple[str, ...]
    terminal_failed: bool
    generation_plan_commitment_byte_sha256: str | None = None
    actual_analysis_hash_ledger_commitment_byte_sha256: str | None = None
    fit_commitment_byte_sha256: str | None = None
    center_state_checkpoint_byte_sha256: str | None = None
    risk_state_checkpoint_byte_sha256: str | None = None
    calibration_mask_commitment_byte_sha256: str | None = None
    model_state_commitment_byte_sha256: str | None = None


def phase_commitment_message(phase: str, byte_sha256: str) -> str:
    """Encode a phase artifact hash in the only accepted ledger message."""

    try:
        field = PHASE_COMMITMENT_FIELDS[phase]
    except KeyError as exc:
        raise V024LedgerError(f"{phase!r} is not a phase-specific commitment") from exc
    if not isinstance(byte_sha256, str) or _SHA256.fullmatch(byte_sha256) is None:
        raise V024LedgerError("Phase commitment is not lowercase SHA256")
    return f"{field}={byte_sha256}"


def _parse_phase_commitment_message(phase: str, message: str) -> str:
    field = PHASE_COMMITMENT_FIELDS[phase]
    match = re.fullmatch(rf"{re.escape(field)}=([0-9a-f]{{64}})", message)
    if match is None:
        raise V024LedgerError(
            f"Completed {phase} lacks its exact machine-readable commitment"
        )
    return match.group(1)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V024LedgerError(f"Exposure JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V024LedgerError(f"Exposure JSON contains nonfinite token {token}")


def canonical_json_line_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the sole accepted ASCII JSONL representation."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise V024LedgerError("Exposure row is not finite canonical JSON") from exc
    return encoded + b"\n"


def _decode_canonical_line(line: bytes, *, line_number: int) -> dict[str, Any]:
    if not line.endswith(b"\n"):
        raise V024LedgerError("Exposure log has a truncated final line")
    try:
        value = json.loads(
            line.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V024LedgerError(
            f"Exposure line {line_number} is not strict ASCII JSON"
        ) from exc
    if not isinstance(value, dict):
        raise V024LedgerError(f"Exposure line {line_number} is not an object")
    if canonical_json_line_bytes(value) != line:
        raise V024LedgerError(f"Exposure line {line_number} is not canonical JSONL")
    return value


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise V024LedgerError("Exposure event timestamp is invalid")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise V024LedgerError("Exposure event timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise V024LedgerError("Exposure event timestamp lacks a timezone")
    return parsed


def _validate_event_shape(
    event: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    sealed_filenames: frozenset[str],
) -> None:
    if set(event) != EXPOSURE_EVENT_KEYS:
        raise V024LedgerError("Exposure event keys differ from the V2.4 freeze")
    if (
        not isinstance(event.get("attempt_id"), str)
        or _ATTEMPT_ID.fullmatch(event["attempt_id"]) is None
        or not isinstance(event.get("git_commit"), str)
        or _HEX_COMMIT.fullmatch(event["git_commit"]) is None
        or event.get("git_dirty") is not False
        or event.get("config_byte_sha256") != expected_config_sha256
    ):
        raise V024LedgerError(
            "Exposure event has an invalid formal implementation identity"
        )
    _parse_timestamp(event.get("created_utc"))
    for key in (
        "truth_commitments_byte_sha256",
        "prediction_commitment_byte_sha256",
    ):
        value = event.get(key)
        if value is not None and (
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
        ):
            raise V024LedgerError(f"Exposure event {key} is invalid")
    opened = event.get("opened_truth_files")
    if (
        not isinstance(opened, list)
        or any(not isinstance(item, str) for item in opened)
        or opened != sorted(set(opened))
        or any(item not in sealed_filenames for item in opened)
    ):
        raise V024LedgerError("Exposure event opened-truth list is invalid")
    for key in ("phase", "exit_status", "message"):
        if not isinstance(event.get(key), str):
            raise V024LedgerError(f"Exposure event {key} is invalid")


def _validate_attempt_events(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_config_sha256: str,
    sealed_filenames: frozenset[str],
) -> AttemptProgress:
    if not events:
        raise V024LedgerError("A formal attempt has no exposure events")
    for event in events:
        _validate_event_shape(
            event,
            expected_config_sha256=expected_config_sha256,
            sealed_filenames=sealed_filenames,
        )

    first = events[0]
    identity = FormalAttemptIdentity(
        attempt_id=str(first["attempt_id"]),
        git_commit=str(first["git_commit"]),
        config_byte_sha256=str(first["config_byte_sha256"]),
    )
    completed_index = -1
    pending_phase: str | None = None
    truth_hash: str | None = None
    prediction_hash: str | None = None
    opened: tuple[str, ...] = ()
    terminal_failed = False
    phase_commitments: dict[str, str] = {}

    for position, event in enumerate(events):
        if str(event["attempt_id"]) != identity.attempt_id:
            raise V024LedgerError("Attempt event was grouped under the wrong ID")
        if (
            str(event["git_commit"]) != identity.git_commit
            or str(event["config_byte_sha256"]) != identity.config_byte_sha256
            or event["git_dirty"] is not False
        ):
            raise V024LedgerError(
                "Formal attempt implementation identity changed or was dirty"
            )
        if terminal_failed:
            raise V024LedgerError("A failed formal attempt has later events")

        phase = str(event["phase"])
        status = str(event["exit_status"])
        if phase not in PHASES or status not in EXIT_STATUSES:
            raise V024LedgerError("Exposure phase or exit_status is not frozen")
        expected_index = completed_index + 1
        if expected_index >= len(PHASES):
            raise V024LedgerError("A completed attempt has later events")
        expected_phase = PHASES[expected_index]
        if phase != expected_phase:
            raise V024LedgerError(
                f"Illegal phase transition to {phase!r}; expected {expected_phase!r}"
            )
        if pending_phase is not None and phase != pending_phase:
            raise V024LedgerError("An interrupted or started phase was skipped")

        event_opened = tuple(event["opened_truth_files"])
        expected_opened = tuple(sorted(OPENED_BY_PHASE[phase]))
        if event_opened != expected_opened:
            raise V024LedgerError(
                f"{phase} opened-truth set differs from the frozen phase capability"
            )
        if not set(opened).issubset(event_opened):
            raise V024LedgerError("A formal attempt forgot an opened truth file")
        opened = event_opened

        observed_truth_hash = event["truth_commitments_byte_sha256"]
        truth_phase_index = PHASES.index("truth_committed")
        if expected_index < truth_phase_index:
            if observed_truth_hash is not None:
                raise V024LedgerError(
                    "Truth commitment appeared before generation completed"
                )
        elif status == "completed" or truth_hash is not None:
            if not isinstance(observed_truth_hash, str):
                raise V024LedgerError(
                    "Completed post-generation phase lacks truth commitment"
                )
        if observed_truth_hash is not None:
            if truth_hash is None:
                truth_hash = str(observed_truth_hash)
            elif observed_truth_hash != truth_hash:
                raise V024LedgerError("Truth commitment changed within an attempt")

        observed_prediction_hash = event["prediction_commitment_byte_sha256"]
        prediction_phase_index = PHASES.index("prediction_committed")
        if expected_index < prediction_phase_index:
            if observed_prediction_hash is not None:
                raise V024LedgerError(
                    "Prediction commitment appeared before its frozen phase"
                )
        elif status == "completed" or prediction_hash is not None:
            if not isinstance(observed_prediction_hash, str):
                raise V024LedgerError(
                    "Completed post-prediction phase lacks prediction commitment"
                )
        if observed_prediction_hash is not None:
            if prediction_hash is None:
                prediction_hash = str(observed_prediction_hash)
            elif observed_prediction_hash != prediction_hash:
                raise V024LedgerError("Prediction commitment changed within an attempt")

        if position == 0 and (phase != "before_generation" or status != "completed"):
            raise V024LedgerError("The first event must checkpoint before generation")
        if status == "completed" and phase in PHASE_COMMITMENT_FIELDS:
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
        generation_plan_commitment_byte_sha256=phase_commitments.get(
            "generation_plan_committed"
        ),
        actual_analysis_hash_ledger_commitment_byte_sha256=(
            phase_commitments.get("actual_analysis_hash_ledger_committed")
        ),
        fit_commitment_byte_sha256=phase_commitments.get("label_free_fit_committed"),
        center_state_checkpoint_byte_sha256=phase_commitments.get(
            "center_state_committed"
        ),
        risk_state_checkpoint_byte_sha256=phase_commitments.get("risk_state_committed"),
        calibration_mask_commitment_byte_sha256=phase_commitments.get(
            "calibration_mask_committed"
        ),
        model_state_commitment_byte_sha256=phase_commitments.get(
            "model_state_committed"
        ),
    )


def validate_exposure_events(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_config_sha256: str,
    sealed_filenames: Sequence[str],
) -> Mapping[str, AttemptProgress]:
    """Validate already-decoded events with the same strict state machine."""

    if (
        not isinstance(expected_config_sha256, str)
        or _SHA256.fullmatch(expected_config_sha256) is None
    ):
        raise V024LedgerError("Expected config hash is invalid")
    sealed = frozenset(sealed_filenames)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        _validate_event_shape(
            event,
            expected_config_sha256=expected_config_sha256,
            sealed_filenames=sealed,
        )
        grouped.setdefault(str(event["attempt_id"]), []).append(event)
    return {
        attempt_id: _validate_attempt_events(
            items,
            expected_config_sha256=expected_config_sha256,
            sealed_filenames=sealed,
        )
        for attempt_id, items in grouped.items()
    }


def parse_exposure_log_bytes(
    raw: bytes,
    *,
    expected_config_sha256: str,
    sealed_filenames: Sequence[str],
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, AttemptProgress]]:
    """Decode exact canonical JSONL bytes and reconstruct all attempts."""

    if type(raw) is not bytes:
        raise V024LedgerError("Exposure log input must be exact bytes")
    if raw and not raw.endswith(b"\n"):
        raise V024LedgerError("Exposure log has a truncated final line")
    events = tuple(
        _decode_canonical_line(line, line_number=index)
        for index, line in enumerate(raw.splitlines(keepends=True), start=1)
    )
    previous: datetime | None = None
    for event in events:
        observed = _parse_timestamp(event.get("created_utc"))
        if previous is not None and observed < previous:
            raise V024LedgerError("Exposure timestamps moved backwards")
        previous = observed
    states = validate_exposure_events(
        events,
        expected_config_sha256=expected_config_sha256,
        sealed_filenames=sealed_filenames,
    )
    return events, states


def read_exposure_log(
    path: str | Path,
    *,
    expected_config_sha256: str,
    sealed_filenames: Sequence[str],
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, AttemptProgress], bytes]:
    """Read one strict canonical exposure log."""

    target = Path(path)
    try:
        raw = target.read_bytes() if target.exists() else b""
    except OSError as exc:
        raise V024LedgerError(f"Cannot read exposure log: {target}") from exc
    events, states = parse_exposure_log_bytes(
        raw,
        expected_config_sha256=expected_config_sha256,
        sealed_filenames=sealed_filenames,
    )
    return events, states, raw


@contextmanager
def locked_ledger(path: str | Path) -> Iterator[None]:
    """Hold the one OS-level lock shared by every V2.4 ledger writer."""

    target = Path(os.path.abspath(os.fspath(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep synchronization metadata outside the frozen artifact inventory.
    # Generation and terminal inventory checks intentionally reject any extra
    # file inside the label-free root.
    lock_path = target.parent.parent / (f".{target.parent.name}.{target.name}.lock")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | _BINARY_FLAG,
        0o600,
    )
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def append_exposure_event_cas(
    path: str | Path,
    event: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    sealed_filenames: Sequence[str],
) -> AttemptProgress:
    """Atomically validate and append one event under the shared ledger lock."""

    target = Path(path)
    with locked_ledger(target):
        existing, _, prefix = read_exposure_log(
            target,
            expected_config_sha256=expected_config_sha256,
            sealed_filenames=sealed_filenames,
        )
        states = validate_exposure_events(
            (*existing, event),
            expected_config_sha256=expected_config_sha256,
            sealed_filenames=sealed_filenames,
        )
        raw = canonical_json_line_bytes(event)
        current = target.read_bytes() if target.exists() else b""
        if current != prefix:
            raise V024LedgerError("Exposure ledger changed before atomic append")
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | _BINARY_FLAG,
            0o600,
        )
        try:
            written = os.write(descriptor, raw)
            if written != len(raw):
                raise OSError("Exposure ledger append was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        observed = target.read_bytes()
        if observed != prefix + raw:
            raise V024LedgerError("Exposure ledger append was not exact")
    return states[str(event["attempt_id"])]


__all__ = [
    "AttemptProgress",
    "EXIT_STATUSES",
    "EXPOSURE_EVENT_KEYS",
    "FormalAttemptIdentity",
    "OPENED_BY_PHASE",
    "PHASES",
    "PHASE_COMMITMENT_FIELDS",
    "V024LedgerError",
    "append_exposure_event_cas",
    "canonical_json_line_bytes",
    "locked_ledger",
    "parse_exposure_log_bytes",
    "phase_commitment_message",
    "read_exposure_log",
    "validate_exposure_events",
]
