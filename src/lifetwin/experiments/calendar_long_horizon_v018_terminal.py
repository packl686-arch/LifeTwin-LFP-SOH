"""Fail-closed V2.3 pre-prediction terminal publication.

The writer emits exactly the three terminal artifacts frozen by the V2.3
amendment. It can inspect only a strict label-free artifact root and has no
sealed-truth, prediction, numeric-model-decoding, or scoring capability.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any, Iterator, Mapping, Sequence

from lifetwin.experiments.calendar_long_horizon_v018_environment import (
    verify_formal_environment,
)
from lifetwin.experiments.calendar_long_horizon_v018_protocol import (
    V023_PROTOCOL_ID,
    load_v023_design,
)
from lifetwin.experiments.calendar_long_horizon_v018_partition import (
    V023PartitionCapabilityError,
    V023PartitionContractError,
    V023WholeBundleContractError,
)
from lifetwin.experiments.calendar_long_horizon_v018_numeric_contract import (
    V023NumericContractError,
)
from lifetwin.experiments.calendar_long_horizon_v018_ledger import (
    AttemptProgress,
    PHASES,
    PHASE_COMMITMENT_FIELDS,
    V023LedgerError,
    locked_ledger,
    parse_exposure_log_bytes,
)
from lifetwin.experiments.calendar_long_horizon_v018_signals import (
    V023CalibrationTerminalInconclusive,
)


V023_AMENDMENT_BYTE_SHA256 = load_v023_design().config_byte_sha256
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class V018TerminationError(ValueError):
    """Raised when terminal evidence is invalid, unsafe, or conflicting."""


class TerminalReason(str, Enum):
    """The exact reason-code registry frozen by the V2.3 amendment."""

    CENTER_FIT_UNDEFINED = "CENTER_FIT_UNDEFINED"
    RISK_FIT_UNDEFINED = "RISK_FIT_UNDEFINED"
    CALIBRATION_SOURCE_COUNT_NOT_900 = "CALIBRATION_SOURCE_COUNT_NOT_900"
    CALIBRATION_RISK_ELIGIBLE_BELOW_855 = "CALIBRATION_RISK_ELIGIBLE_BELOW_855"
    CALIBRATION_RISK_POSITIVE_BELOW_60 = "CALIBRATION_RISK_POSITIVE_BELOW_60"
    CALIBRATION_RISK_NEGATIVE_BELOW_60 = "CALIBRATION_RISK_NEGATIVE_BELOW_60"
    CALIBRATION_RISK_SCORE_NONFINITE = "CALIBRATION_RISK_SCORE_NONFINITE"
    CALIBRATION_ISOTONIC_FIT_UNDEFINED = "CALIBRATION_ISOTONIC_FIT_UNDEFINED"
    CALIBRATION_BASELINE_INCOMPLETE = "CALIBRATION_BASELINE_INCOMPLETE"
    CALIBRATION_ZERO_FAMILY_NO_BAND = "CALIBRATION_ZERO_FAMILY_NO_BAND"
    CALIBRATION_BAND_NONFINITE_OR_UNORDERED = "CALIBRATION_BAND_NONFINITE_OR_UNORDERED"
    CALIBRATION_CONFORMAL_COUNT_NOT_900 = "CALIBRATION_CONFORMAL_COUNT_NOT_900"
    CALIBRATION_CONFORMAL_SCORE_NONFINITE = "CALIBRATION_CONFORMAL_SCORE_NONFINITE"
    CALIBRATION_CONFORMAL_FIT_UNDEFINED = "CALIBRATION_CONFORMAL_FIT_UNDEFINED"

    INTEGRITY_CONFIG_OR_SOURCE_HASH_MISMATCH = (
        "INTEGRITY_CONFIG_OR_SOURCE_HASH_MISMATCH"
    )
    INTEGRITY_ENVIRONMENT_MISMATCH = "INTEGRITY_ENVIRONMENT_MISMATCH"
    INTEGRITY_ARTIFACT_HASH_MISMATCH = "INTEGRITY_ARTIFACT_HASH_MISMATCH"
    INTEGRITY_PARTITION_SEED_ID_OR_CONTENT_COLLISION = (
        "INTEGRITY_PARTITION_SEED_ID_OR_CONTENT_COLLISION"
    )
    INTEGRITY_FORBIDDEN_PREDECESSOR_REUSE = "INTEGRITY_FORBIDDEN_PREDECESSOR_REUSE"
    INTEGRITY_FORBIDDEN_TRUTH_ACCESS = "INTEGRITY_FORBIDDEN_TRUTH_ACCESS"
    INTEGRITY_INFORMATION_LEAK = "INTEGRITY_INFORMATION_LEAK"
    INTEGRITY_MISSING_COMMITMENT = "INTEGRITY_MISSING_COMMITMENT"
    INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH = (
        "INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH"
    )
    INTEGRITY_PARTITION_CONTRACT_MISMATCH = (
        "INTEGRITY_PARTITION_CONTRACT_MISMATCH"
    )
    INTEGRITY_PARTITION_CAPABILITY_MISMATCH = (
        "INTEGRITY_PARTITION_CAPABILITY_MISMATCH"
    )
    INTEGRITY_NUMERIC_OUTPUT_CONTRACT_MISMATCH = (
        "INTEGRITY_NUMERIC_OUTPUT_CONTRACT_MISMATCH"
    )
    INTEGRITY_PATH_ISOLATION_FAILURE = "INTEGRITY_PATH_ISOLATION_FAILURE"
    INTEGRITY_PREDECESSOR_HISTORY_DRIFT = "INTEGRITY_PREDECESSOR_HISTORY_DRIFT"

    INTERRUPTED_BY_OPERATOR = "INTERRUPTED_BY_OPERATOR"
    INTERRUPTED_BY_PLATFORM = "INTERRUPTED_BY_PLATFORM"
    UNKNOWN_PRE_PREDICTION_EXCEPTION = "UNKNOWN_PRE_PREDICTION_EXCEPTION"


_SCIENTIFIC_MESSAGES: Mapping[TerminalReason, str] = {
    TerminalReason.CENTER_FIT_UNDEFINED: (
        "The frozen center fit was mathematically undefined."
    ),
    TerminalReason.RISK_FIT_UNDEFINED: (
        "The frozen risk fit was mathematically undefined."
    ),
    TerminalReason.CALIBRATION_SOURCE_COUNT_NOT_900: (
        "The calibration source count was not exactly 900."
    ),
    TerminalReason.CALIBRATION_RISK_ELIGIBLE_BELOW_855: (
        "Fewer than 855 calibration rows met the frozen risk mask."
    ),
    TerminalReason.CALIBRATION_RISK_POSITIVE_BELOW_60: (
        "The frozen risk mask contained fewer than 60 positive labels."
    ),
    TerminalReason.CALIBRATION_RISK_NEGATIVE_BELOW_60: (
        "The frozen risk mask contained fewer than 60 negative labels."
    ),
    TerminalReason.CALIBRATION_RISK_SCORE_NONFINITE: (
        "A required calibration risk score was nonfinite."
    ),
    TerminalReason.CALIBRATION_ISOTONIC_FIT_UNDEFINED: (
        "The frozen isotonic calibration was mathematically undefined."
    ),
    TerminalReason.CALIBRATION_BASELINE_INCOMPLETE: (
        "The frozen full-pool mean-baseline comparison was incomplete."
    ),
    TerminalReason.CALIBRATION_ZERO_FAMILY_NO_BAND: (
        "A calibration member had no successful structural family."
    ),
    TerminalReason.CALIBRATION_BAND_NONFINITE_OR_UNORDERED: (
        "A required calibration structural band was nonfinite or unordered."
    ),
    TerminalReason.CALIBRATION_CONFORMAL_COUNT_NOT_900: (
        "The conformal calibration count was not exactly 900."
    ),
    TerminalReason.CALIBRATION_CONFORMAL_SCORE_NONFINITE: (
        "A required simultaneous conformal score was nonfinite."
    ),
    TerminalReason.CALIBRATION_CONFORMAL_FIT_UNDEFINED: (
        "The frozen simultaneous conformal fit was undefined."
    ),
}
_INTEGRITY_MESSAGES: Mapping[TerminalReason, str] = {
    TerminalReason.INTEGRITY_CONFIG_OR_SOURCE_HASH_MISMATCH: (
        "A frozen config or source hash did not match."
    ),
    TerminalReason.INTEGRITY_ENVIRONMENT_MISMATCH: (
        "The frozen execution environment did not match."
    ),
    TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH: (
        "A committed artifact hash did not match."
    ),
    TerminalReason.INTEGRITY_PARTITION_SEED_ID_OR_CONTENT_COLLISION: (
        "A partition seed, identifier, or content collision was detected."
    ),
    TerminalReason.INTEGRITY_FORBIDDEN_PREDECESSOR_REUSE: (
        "Forbidden predecessor evidence reuse was detected."
    ),
    TerminalReason.INTEGRITY_FORBIDDEN_TRUTH_ACCESS: (
        "Forbidden truth access was detected."
    ),
    TerminalReason.INTEGRITY_INFORMATION_LEAK: (
        "An information-boundary violation was detected."
    ),
    TerminalReason.INTEGRITY_MISSING_COMMITMENT: (
        "A required pre-prediction commitment was absent."
    ),
    TerminalReason.INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH: (
        "The complete label-free bundle failed its frozen formal contract."
    ),
    TerminalReason.INTEGRITY_PARTITION_CONTRACT_MISMATCH: (
        "A capability-derived partition failed its exact frozen contract."
    ),
    TerminalReason.INTEGRITY_PARTITION_CAPABILITY_MISMATCH: (
        "A partition validation capability was forged, mutated, or misused."
    ),
    TerminalReason.INTEGRITY_NUMERIC_OUTPUT_CONTRACT_MISMATCH: (
        "A label-free numeric output violated its exact finite/structural-NaN contract."
    ),
    TerminalReason.INTEGRITY_PATH_ISOLATION_FAILURE: (
        "The frozen V2.3 path-isolation contract failed."
    ),
    TerminalReason.INTEGRITY_PREDECESSOR_HISTORY_DRIFT: (
        "A predecessor history anchor changed after preflight."
    ),
}
_INTERRUPTION_MESSAGES: Mapping[TerminalReason, str] = {
    TerminalReason.INTERRUPTED_BY_OPERATOR: (
        "The formal attempt was interrupted by the operator."
    ),
    TerminalReason.INTERRUPTED_BY_PLATFORM: (
        "The formal attempt was interrupted by the execution platform."
    ),
}
_UNKNOWN_MESSAGE = "An unclassified exception terminated the attempt before prediction."
_CONSERVATIVE_VOID_MESSAGE = (
    "An unclassified pre-prediction exception could not be proven free of "
    "integrity implications."
)
_REASON_MESSAGES = {
    **_SCIENTIFIC_MESSAGES,
    **_INTEGRITY_MESSAGES,
    **_INTERRUPTION_MESSAGES,
    TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION: _UNKNOWN_MESSAGE,
}
SCIENTIFIC_TERMINAL_REASONS = frozenset(_SCIENTIFIC_MESSAGES)
INTEGRITY_TERMINAL_REASONS = frozenset(_INTEGRITY_MESSAGES)
INTERRUPTION_TERMINAL_REASONS = frozenset(_INTERRUPTION_MESSAGES)


def _parse_reason(reason: TerminalReason | str) -> TerminalReason:
    if isinstance(reason, TerminalReason):
        return reason
    if not isinstance(reason, str):
        raise TypeError("Terminal reason must be a registered string or enum")
    try:
        return TerminalReason(reason)
    except ValueError as exc:
        raise ValueError("Terminal reason is not registered") from exc


class V018TerminalInconclusive(RuntimeError):
    """A preregistered scientific condition with no scored result."""

    def __init__(self, reason: TerminalReason | str) -> None:
        parsed = _parse_reason(reason)
        if parsed not in SCIENTIFIC_TERMINAL_REASONS:
            raise ValueError("Terminal-inconclusive reason is not scientific")
        self.reason = parsed
        super().__init__(_SCIENTIFIC_MESSAGES[parsed])


class V018IntegrityError(RuntimeError):
    """A typed proven-integrity condition that voids the attempt."""

    def __init__(
        self,
        reason: TerminalReason | str = (
            TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH
        ),
    ) -> None:
        parsed = _parse_reason(reason)
        if parsed not in INTEGRITY_TERMINAL_REASONS:
            raise ValueError("Integrity reason is not registered")
        self.reason = parsed
        super().__init__(_INTEGRITY_MESSAGES[parsed])


class V018Interrupted(RuntimeError):
    """A typed interruption that may resume only under the frozen rule."""

    def __init__(self, reason: TerminalReason | str) -> None:
        parsed = _parse_reason(reason)
        if parsed not in INTERRUPTION_TERMINAL_REASONS:
            raise ValueError("Interruption reason is not registered")
        self.reason = parsed
        super().__init__(_INTERRUPTION_MESSAGES[parsed])


class V018ConservativeVoid(RuntimeError):
    """An explicit typed choice to publish an unknown exception as void."""

    def __init__(self) -> None:
        self.reason = TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION
        super().__init__(_CONSERVATIVE_VOID_MESSAGE)


class TerminalDisposition(str, Enum):
    SCIENTIFIC_INCONCLUSIVE = "inconclusive_not_success"
    INTEGRITY_FAILURE = "void"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unclassified_terminal_not_success"


class ClassificationMode(str, Enum):
    DECLARED_SCIENTIFIC = "declared_scientific"
    PROVEN_INTEGRITY = "proven_integrity"
    TYPED_INTERRUPTION = "typed_interruption"
    UNKNOWN_DEFAULT = "unknown_default"
    UNKNOWN_CONSERVATIVE_VOID = "unknown_conservative_void"


@dataclass(frozen=True)
class TerminalClassification:
    disposition: TerminalDisposition
    mode: ClassificationMode
    reason: TerminalReason
    scientific_status: str
    exception_class: str
    safe_message: str

    @property
    def is_scientific_terminal(self) -> bool:
        return self.mode is ClassificationMode.DECLARED_SCIENTIFIC


def _safe_exception_class(error: BaseException) -> str:
    name = type(error).__name__
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name) is None:
        return "Exception"
    return name


def _classification(
    *,
    error: BaseException,
    disposition: TerminalDisposition,
    mode: ClassificationMode,
    reason: TerminalReason,
    safe_message: str | None = None,
) -> TerminalClassification:
    return TerminalClassification(
        disposition=disposition,
        mode=mode,
        reason=reason,
        scientific_status=disposition.value,
        exception_class=_safe_exception_class(error),
        safe_message=(
            _REASON_MESSAGES[reason] if safe_message is None else safe_message
        ),
    )


def _v023_training_reason(error: BaseException) -> TerminalReason | None:
    """Recognize only the shared nominal calibration signal."""

    if type(error) is not V023CalibrationTerminalInconclusive:
        return None
    raw_reason = getattr(error, "reason_code", None)
    if not isinstance(raw_reason, str):
        return None
    try:
        reason = TerminalReason(raw_reason)
    except ValueError:
        return None
    if reason not in SCIENTIFIC_TERMINAL_REASONS:
        return None
    return reason


def classify_terminal_exception(error: BaseException) -> TerminalClassification:
    """Classify from types and registered codes, never raw exception text."""

    if isinstance(error, V018IntegrityError):
        return _classification(
            error=error,
            disposition=TerminalDisposition.INTEGRITY_FAILURE,
            mode=ClassificationMode.PROVEN_INTEGRITY,
            reason=error.reason,
        )
    partition_reason = {
        V023WholeBundleContractError: (
            TerminalReason.INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH
        ),
        V023PartitionContractError: (
            TerminalReason.INTEGRITY_PARTITION_CONTRACT_MISMATCH
        ),
        V023PartitionCapabilityError: (
            TerminalReason.INTEGRITY_PARTITION_CAPABILITY_MISMATCH
        ),
    }.get(type(error))
    if partition_reason is not None:
        return _classification(
            error=error,
            disposition=TerminalDisposition.INTEGRITY_FAILURE,
            mode=ClassificationMode.PROVEN_INTEGRITY,
            reason=partition_reason,
        )
    if type(error) is V023NumericContractError:
        return _classification(
            error=error,
            disposition=TerminalDisposition.INTEGRITY_FAILURE,
            mode=ClassificationMode.PROVEN_INTEGRITY,
            reason=TerminalReason.INTEGRITY_NUMERIC_OUTPUT_CONTRACT_MISMATCH,
        )
    if isinstance(error, V018ConservativeVoid):
        return _classification(
            error=error,
            disposition=TerminalDisposition.INTEGRITY_FAILURE,
            mode=ClassificationMode.UNKNOWN_CONSERVATIVE_VOID,
            reason=TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION,
            safe_message=_CONSERVATIVE_VOID_MESSAGE,
        )
    if isinstance(error, V018TerminalInconclusive):
        return _classification(
            error=error,
            disposition=TerminalDisposition.SCIENTIFIC_INCONCLUSIVE,
            mode=ClassificationMode.DECLARED_SCIENTIFIC,
            reason=error.reason,
        )
    training_reason = _v023_training_reason(error)
    if training_reason is not None:
        return _classification(
            error=error,
            disposition=TerminalDisposition.SCIENTIFIC_INCONCLUSIVE,
            mode=ClassificationMode.DECLARED_SCIENTIFIC,
            reason=training_reason,
        )
    if isinstance(error, V018Interrupted):
        return _classification(
            error=error,
            disposition=TerminalDisposition.INTERRUPTED,
            mode=ClassificationMode.TYPED_INTERRUPTION,
            reason=error.reason,
        )
    if isinstance(error, KeyboardInterrupt):
        return _classification(
            error=error,
            disposition=TerminalDisposition.INTERRUPTED,
            mode=ClassificationMode.TYPED_INTERRUPTION,
            reason=TerminalReason.INTERRUPTED_BY_OPERATOR,
        )
    return _classification(
        error=error,
        disposition=TerminalDisposition.UNKNOWN,
        mode=ClassificationMode.UNKNOWN_DEFAULT,
        reason=TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION,
    )


_ADDRESS = re.compile(r"0[xX][0-9A-Fa-f]+")
_HEX64 = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"']+")
_UNC_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:\\\\|//)[^\s\"']+")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_.])/(?:[^\s\"']+/)*[^\s\"']+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_STDERR_REDACTION_POLICY = (
    "utf8_replace_normalize_newlines_redact_paths_addresses_sha256_controls_v1"
)


def _sanitize_text(value: str, *, maximum_length: int = 8192) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _UNC_PATH.sub("<path>", text)
    text = _WINDOWS_PATH.sub("<path>", text)
    text = _POSIX_PATH.sub("<path>", text)
    text = _ADDRESS.sub("<address>", text)
    text = _HEX64.sub("<digest>", text)
    if len(text) > maximum_length:
        text = f"{text[:maximum_length]}\n<truncated>"
    return text


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise V018TerminationError("Terminal JSON is not canonicalizable") from exc
    return f"{text}\n".encode("ascii")


def _safe_frame_path(filename: str, repo_root: Path | None) -> str:
    path = Path(filename)
    if repo_root is not None:
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except (OSError, ValueError):
            relative = None
        if relative is not None:
            return _sanitize_text(relative.as_posix(), maximum_length=512)
    return _sanitize_text(f"<external>/{path.name}", maximum_length=512)


def sanitized_structural_traceback(
    error: BaseException,
    *,
    repo_root: str | Path | None = None,
) -> bytes:
    """Return deterministic traceback structure without locals or raw text."""

    root = None if repo_root is None else Path(repo_root)
    frames: list[dict[str, object]] = []
    current = error.__traceback__
    while current is not None:
        code = current.tb_frame.f_code
        frames.append(
            {
                "file": _safe_frame_path(code.co_filename, root),
                "function": _sanitize_text(code.co_name, maximum_length=256),
                "line": int(current.tb_lineno),
            }
        )
        current = current.tb_next
    classification = classify_terminal_exception(error)
    payload = {
        "schema_version": "1.0.0",
        "exception_class": classification.exception_class,
        "classification_mode": classification.mode.value,
        "attempt_disposition": classification.disposition.value,
        "reason_code": classification.reason.value,
        "safe_message": classification.safe_message,
        "frames": frames,
    }
    raw = _canonical_json_bytes(payload)
    text = raw.decode("ascii")
    if (
        _ADDRESS.search(text)
        or _HEX64.search(text)
        or _WINDOWS_PATH.search(text)
        or _UNC_PATH.search(text)
        or re.search(r'"file":"(?:[A-Za-z]:|/)', text)
    ):
        raise V018TerminationError("Structural traceback sanitization failed")
    return raw


def sanitized_stderr_bytes(stderr: str | bytes) -> bytes:
    """Normalize and redact stderr for metadata-only commitment."""

    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    else:
        raise TypeError("stderr must be text or bytes")
    sanitized = _sanitize_text(text)
    return sanitized.encode("utf-8") + (b"" if sanitized.endswith("\n") else b"\n")


_HEX40_OR_64 = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_UTC_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_HELD_OUT_TRUTH = frozenset(
    {
        "test_truth.csv",
        "audit_truth.csv",
        "intrinsic_matched_truth.csv",
        "stress_plan_matched_truth.csv",
        "intrinsic_matched_pairs.csv",
        "stress_plan_matched_pairs.csv",
    }
)
_DEVELOPMENT_TRUTH = frozenset(
    {
        "center_development_truth.csv",
        "risk_development_truth.csv",
        "calibration_truth.csv",
    }
)
_ALLOWED_PRETERMINAL_ARTIFACTS = frozenset(
    {
        "generation_plan_commitment.json",
        "prefix_pack.csv",
        "forecast_coordinates.csv",
        "operating_pack.csv",
        "truth_commitments.json",
        "actual_analysis_hash_ledger_commitment.json",
        "exposure_log.jsonl",
        "member_fit_diagnostics.csv",
        "member_forecast_bundle.csv",
        "fit_commitment.json",
        "center_state_checkpoint.json",
        "risk_state_checkpoint.json",
        "training_manifest.json",
        "calibration_mask_commitment.json",
        "calibration_manifest.json",
        "calibration_population_audit.json",
        "model_state.json",
        "model_state_commitment.json",
        "prediction_bundle.csv",
        "risk_bundle.csv",
        "decision_bundle.csv",
        "prediction_commitment.json",
    }
)
_FORBIDDEN_PREDICTION_SCORE_ARTIFACTS = frozenset(
    {
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
    }
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
_PREDICTION_STAGE_FILES = frozenset(
    {
        "prediction_bundle.csv",
        "risk_bundle.csv",
        "decision_bundle.csv",
        "prediction_commitment.json",
    }
)
_COMMITMENT_PHASE_BY_ARTIFACT = {
    "generation_plan_commitment.json": "generation_plan_committed",
    "truth_commitments.json": "truth_committed",
    "actual_analysis_hash_ledger_commitment.json": (
        "actual_analysis_hash_ledger_committed"
    ),
    "fit_commitment.json": "label_free_fit_committed",
    "center_state_checkpoint.json": "center_state_committed",
    "risk_state_checkpoint.json": "risk_state_committed",
    "calibration_mask_commitment.json": "calibration_mask_committed",
    "model_state_commitment.json": "model_state_committed",
}
_TERMINAL_FILENAMES = frozenset(
    {
        "terminal_attempt_record.json",
        "terminal_artifact_manifest.json",
        "terminal_exposure_log_snapshot.jsonl",
    }
)
_PUBLICATION_ORDER = (
    "terminal_attempt_record.json",
    "terminal_exposure_log_snapshot.jsonl",
    "terminal_artifact_manifest.json",
)
_LOCK_FILENAME = ".terminal_publish.lock"


@dataclass(frozen=True)
class TerminalContext:
    protocol_id: str
    attempt_id: str
    git_commit: str
    git_dirty: bool
    config_byte_sha256: str
    created_utc: str
    terminated_utc: str
    attempted_phase: str
    last_completed_phase: str
    truth_commitments_byte_sha256: str | None
    generation_plan_commitment_byte_sha256: str | None = None
    actual_analysis_hash_ledger_commitment_byte_sha256: str | None = None
    fit_commitment_byte_sha256: str | None = None
    center_state_checkpoint_byte_sha256: str | None = None
    risk_state_checkpoint_byte_sha256: str | None = None
    calibration_mask_commitment_byte_sha256: str | None = None
    model_state_commitment_byte_sha256: str | None = None

    @classmethod
    def from_progress(
        cls,
        progress: AttemptProgress,
        *,
        created_utc: str,
        terminated_utc: str,
    ) -> "TerminalContext":
        """Derive every lifecycle and commitment field from the strict ledger."""

        if progress.pending_phase is not None:
            attempted = progress.pending_phase
        else:
            try:
                completed_index = PHASES.index(progress.completed_phase)
                attempted = PHASES[completed_index + 1]
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    "Completed attempt has no pre-prediction terminal phase"
                ) from exc
        return cls(
            protocol_id=V023_PROTOCOL_ID,
            attempt_id=progress.identity.attempt_id,
            git_commit=progress.identity.git_commit,
            git_dirty=False,
            config_byte_sha256=progress.identity.config_byte_sha256,
            created_utc=created_utc,
            terminated_utc=terminated_utc,
            attempted_phase=attempted,
            last_completed_phase=progress.completed_phase,
            truth_commitments_byte_sha256=(progress.truth_commitments_byte_sha256),
            generation_plan_commitment_byte_sha256=(
                progress.generation_plan_commitment_byte_sha256
            ),
            actual_analysis_hash_ledger_commitment_byte_sha256=(
                progress.actual_analysis_hash_ledger_commitment_byte_sha256
            ),
            fit_commitment_byte_sha256=progress.fit_commitment_byte_sha256,
            center_state_checkpoint_byte_sha256=(
                progress.center_state_checkpoint_byte_sha256
            ),
            risk_state_checkpoint_byte_sha256=(
                progress.risk_state_checkpoint_byte_sha256
            ),
            calibration_mask_commitment_byte_sha256=(
                progress.calibration_mask_commitment_byte_sha256
            ),
            model_state_commitment_byte_sha256=(
                progress.model_state_commitment_byte_sha256
            ),
        )

    def __post_init__(self) -> None:
        if self.protocol_id != V023_PROTOCOL_ID:
            raise ValueError("Terminal protocol_id is not the validated V2.3 ID")
        if self.config_byte_sha256 != V023_AMENDMENT_BYTE_SHA256:
            raise ValueError("Terminal config hash is not the validated V2.3 amendment")
        if (
            not isinstance(self.attempt_id, str)
            or _SAFE_ID.fullmatch(self.attempt_id) is None
        ):
            raise ValueError("attempt_id is invalid")
        if (
            not isinstance(self.git_commit, str)
            or _HEX40_OR_64.fullmatch(self.git_commit) is None
        ):
            raise ValueError("git_commit is invalid")
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be a strict boolean")
        if self.truth_commitments_byte_sha256 is not None and (
            not isinstance(self.truth_commitments_byte_sha256, str)
            or _SHA256.fullmatch(self.truth_commitments_byte_sha256) is None
        ):
            raise ValueError("truth_commitments_byte_sha256 is invalid")
        for name in (
            "generation_plan_commitment_byte_sha256",
            "actual_analysis_hash_ledger_commitment_byte_sha256",
            "fit_commitment_byte_sha256",
            "center_state_checkpoint_byte_sha256",
            "risk_state_checkpoint_byte_sha256",
            "calibration_mask_commitment_byte_sha256",
            "model_state_commitment_byte_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
            ):
                raise ValueError(f"{name} is invalid")
        for value in (self.created_utc, self.terminated_utc):
            if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
                raise ValueError("Terminal timestamps must be UTC seconds")
        for value in (self.attempted_phase, self.last_completed_phase):
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise ValueError("Terminal phase is invalid")


@dataclass(frozen=True)
class LedgerSnapshot:
    byte_count: int
    row_count: int
    byte_sha256: str
    opened_truth_files: tuple[str, ...]


@dataclass(frozen=True)
class PublishedTermination:
    manifest_path: Path
    manifest_byte_sha256: str
    attempt_record_byte_sha256: str
    ledger_snapshot_byte_sha256: str
    ledger_record_appended: bool
    disposition: TerminalDisposition
    reason: TerminalReason


_ATTEMPT_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "attempt_id",
        "created_utc",
        "git_commit",
        "git_dirty",
        "config_byte_sha256",
        "phase",
        "last_completed_phase",
        "attempt_disposition",
        "classification_mode",
        "reason_code",
        "scientific_status",
        "truth_commitments_byte_sha256",
        "prediction_commitment_byte_sha256",
        "opened_truth_files",
        "message",
        "preterminal_artifacts_byte_sha256",
        "terminal_exposure_log_snapshot_byte_sha256",
        "diagnostics",
    }
)
_DIAGNOSTIC_KEYS = frozenset({"structural_traceback", "sanitized_stderr"})
_DIAGNOSTIC_METADATA_KEYS = frozenset({"byte_count", "byte_sha256", "redaction_policy"})
_TRACEBACK_KEYS = frozenset(
    {
        "schema_version",
        "exception_class",
        "classification_mode",
        "attempt_disposition",
        "reason_code",
        "safe_message",
        "frames",
    }
)
_FRAME_KEYS = frozenset({"file", "function", "line"})
_FILE_RECORD_KEYS = frozenset({"path", "byte_count", "byte_sha256"})
_TERMINAL_FILE_RECORD_KEYS = frozenset({"path", "role", "byte_count", "byte_sha256"})
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "attempt_id",
        "created_utc",
        "git_commit",
        "config_byte_sha256",
        "registry",
        "terminal_files",
        "preterminal_artifacts",
        "preterminal_commitments",
        "artifact_policy",
        "content_boundary_attestation",
    }
)
_COMMITMENT_KEYS = frozenset(
    {
        "truth_commitments_byte_sha256",
        "generation_plan_commitment_byte_sha256",
        "actual_analysis_hash_ledger_commitment_byte_sha256",
        "fit_commitment_byte_sha256",
        "center_state_checkpoint_byte_sha256",
        "risk_state_checkpoint_byte_sha256",
        "calibration_mask_commitment_byte_sha256",
        "model_state_commitment_byte_sha256",
        "prediction_commitment_byte_sha256",
    }
)
_LEDGER_PHASE_COMMITMENT_FIELDS = {
    phase: field for phase, field in PHASE_COMMITMENT_FIELDS.items()
}
_ARTIFACT_POLICY_KEYS = frozenset(
    {
        "termination_root_was_independent_and_empty",
        "terminal_registry_was_exclusive",
        "forbidden_artifacts_observed",
        "unknown_artifacts_observed",
        "links_or_directories_observed",
        "model_artifacts_fabricated",
        "prediction_artifacts_fabricated",
        "score_artifacts_fabricated",
    }
)
_TERMINAL_LEDGER_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "attempt_id",
        "git_commit",
        "git_dirty",
        "config_byte_sha256",
        "created_utc",
        "phase",
        "terminal_disposition",
        "classification_mode",
        "scientific_status",
        "reason_code",
        "terminal_attempt_record_byte_sha256",
        "terminal_exposure_log_snapshot_byte_sha256",
        "terminal_artifact_manifest_byte_sha256",
    }
)
_BINARY_FLAG = getattr(os, "O_BINARY", 0)
_PROCESS_PUBLICATION_LOCK = threading.RLock()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_exact_keys(
    value: object,
    keys: frozenset[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise V018TerminationError(f"{context} keys are not exact")
    return value


def _validate_identity(payload: Mapping[str, Any], *, context: str) -> None:
    if (
        payload["protocol_id"] != V023_PROTOCOL_ID
        or payload["config_byte_sha256"] != V023_AMENDMENT_BYTE_SHA256
        or not isinstance(payload["attempt_id"], str)
        or _SAFE_ID.fullmatch(payload["attempt_id"]) is None
        or not isinstance(payload["git_commit"], str)
        or _HEX40_OR_64.fullmatch(payload["git_commit"]) is None
        or not isinstance(payload["created_utc"], str)
        or _UTC_SECONDS.fullmatch(payload["created_utc"]) is None
    ):
        raise V018TerminationError(f"{context} identity is invalid")


def _expected_classification(
    reason: TerminalReason,
    mode: ClassificationMode,
) -> tuple[TerminalDisposition, str]:
    if (
        mode is ClassificationMode.DECLARED_SCIENTIFIC
        and reason in SCIENTIFIC_TERMINAL_REASONS
    ):
        return (
            TerminalDisposition.SCIENTIFIC_INCONCLUSIVE,
            _SCIENTIFIC_MESSAGES[reason],
        )
    if (
        mode is ClassificationMode.PROVEN_INTEGRITY
        and reason in INTEGRITY_TERMINAL_REASONS
    ):
        return TerminalDisposition.INTEGRITY_FAILURE, _INTEGRITY_MESSAGES[reason]
    if (
        mode is ClassificationMode.TYPED_INTERRUPTION
        and reason in INTERRUPTION_TERMINAL_REASONS
    ):
        return TerminalDisposition.INTERRUPTED, _INTERRUPTION_MESSAGES[reason]
    if (
        mode is ClassificationMode.UNKNOWN_DEFAULT
        and reason is TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION
    ):
        return TerminalDisposition.UNKNOWN, _UNKNOWN_MESSAGE
    if (
        mode is ClassificationMode.UNKNOWN_CONSERVATIVE_VOID
        and reason is TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION
    ):
        return TerminalDisposition.INTEGRITY_FAILURE, _CONSERVATIVE_VOID_MESSAGE
    raise V018TerminationError("Disposition and reason classification conflict")


def _validate_structural_traceback(
    value: object,
    *,
    expected_reason: TerminalReason,
    expected_mode: ClassificationMode,
    expected_disposition: TerminalDisposition,
    expected_message: str,
) -> None:
    trace = _validate_exact_keys(
        value,
        _TRACEBACK_KEYS,
        context="structural traceback",
    )
    if (
        trace["schema_version"] != "1.0.0"
        or not isinstance(trace["exception_class"], str)
        or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,127}",
            trace["exception_class"],
        )
        is None
        or trace["reason_code"] != expected_reason.value
        or trace["classification_mode"] != expected_mode.value
        or trace["attempt_disposition"] != expected_disposition.value
        or trace["safe_message"] != expected_message
        or not isinstance(trace["frames"], list)
    ):
        raise V018TerminationError("Structural traceback identity is invalid")
    for value in trace["frames"]:
        frame = _validate_exact_keys(value, _FRAME_KEYS, context="traceback frame")
        if (
            not isinstance(frame["file"], str)
            or not isinstance(frame["function"], str)
            or not isinstance(frame["line"], int)
            or isinstance(frame["line"], bool)
            or frame["line"] < 0
            or Path(frame["file"]).is_absolute()
        ):
            raise V018TerminationError("Structural traceback frame is invalid")
    serialized = _canonical_json_bytes(trace).decode("ascii")
    if (
        _ADDRESS.search(serialized)
        or _HEX64.search(serialized)
        or _WINDOWS_PATH.search(serialized)
        or _UNC_PATH.search(serialized)
        or re.search(r'"file":"(?:[A-Za-z]:|/)', serialized)
    ):
        raise V018TerminationError("Structural traceback contains unsafe text")


def validate_terminal_attempt_record(payload: Mapping[str, Any]) -> None:
    """Validate the exact canonical terminal-attempt record schema."""

    record = _validate_exact_keys(
        payload,
        _ATTEMPT_RECORD_KEYS,
        context="terminal attempt record",
    )
    _validate_identity(record, context="Terminal attempt record")
    if not isinstance(record["git_dirty"], bool):
        raise V018TerminationError("Terminal attempt git_dirty is invalid")
    for phase_name in ("phase", "last_completed_phase"):
        if (
            not isinstance(record[phase_name], str)
            or _SAFE_ID.fullmatch(record[phase_name]) is None
        ):
            raise V018TerminationError("Terminal attempt phase is invalid")
    try:
        reason = TerminalReason(record["reason_code"])
        mode = ClassificationMode(record["classification_mode"])
    except (TypeError, ValueError) as exc:
        raise V018TerminationError("Terminal classification code is invalid") from exc
    disposition, message = _expected_classification(reason, mode)
    if (
        record["attempt_disposition"] != disposition.value
        or record["scientific_status"] != disposition.value
        or record["message"] != message
        or record["prediction_commitment_byte_sha256"] is not None
        or (
            record["truth_commitments_byte_sha256"] is not None
            and (
                not isinstance(record["truth_commitments_byte_sha256"], str)
                or _SHA256.fullmatch(record["truth_commitments_byte_sha256"]) is None
            )
        )
        or not isinstance(record["preterminal_artifacts_byte_sha256"], str)
        or _SHA256.fullmatch(record["preterminal_artifacts_byte_sha256"]) is None
        or not isinstance(record["terminal_exposure_log_snapshot_byte_sha256"], str)
        or _SHA256.fullmatch(record["terminal_exposure_log_snapshot_byte_sha256"])
        is None
    ):
        raise V018TerminationError("Terminal disposition is invalid")
    opened = record["opened_truth_files"]
    if (
        not isinstance(opened, list)
        or any(not isinstance(item, str) for item in opened)
        or opened != sorted(set(opened))
        or not set(opened).issubset(_DEVELOPMENT_TRUTH)
    ):
        raise V018TerminationError("Terminal truth exposure is invalid")
    diagnostics = _validate_exact_keys(
        record["diagnostics"],
        _DIAGNOSTIC_KEYS,
        context="terminal diagnostics",
    )
    _validate_structural_traceback(
        diagnostics["structural_traceback"],
        expected_reason=reason,
        expected_mode=mode,
        expected_disposition=disposition,
        expected_message=message,
    )
    stderr = _validate_exact_keys(
        diagnostics["sanitized_stderr"],
        _DIAGNOSTIC_METADATA_KEYS,
        context="sanitized stderr metadata",
    )
    if (
        not isinstance(stderr["byte_count"], int)
        or isinstance(stderr["byte_count"], bool)
        or stderr["byte_count"] < 1
        or not isinstance(stderr["byte_sha256"], str)
        or _SHA256.fullmatch(stderr["byte_sha256"]) is None
        or stderr["redaction_policy"] != _STDERR_REDACTION_POLICY
    ):
        raise V018TerminationError("Sanitized stderr metadata is invalid")


def _validate_preterminal_file_record(value: object) -> Mapping[str, Any]:
    record = _validate_exact_keys(
        value,
        _FILE_RECORD_KEYS,
        context="preterminal artifact record",
    )
    if (
        not isinstance(record["path"], str)
        or record["path"] not in _ALLOWED_PRETERMINAL_ARTIFACTS
        or not isinstance(record["byte_count"], int)
        or isinstance(record["byte_count"], bool)
        or record["byte_count"] < 1
        or not isinstance(record["byte_sha256"], str)
        or _SHA256.fullmatch(record["byte_sha256"]) is None
    ):
        raise V018TerminationError("Preterminal artifact record is invalid")
    return record


def _validate_terminal_file_record(
    value: object,
    *,
    expected_path: str,
    expected_role: str,
) -> None:
    record = _validate_exact_keys(
        value,
        _TERMINAL_FILE_RECORD_KEYS,
        context=f"{expected_path} file record",
    )
    if (
        record["path"] != expected_path
        or record["role"] != expected_role
        or not isinstance(record["byte_count"], int)
        or isinstance(record["byte_count"], bool)
        or record["byte_count"] < 1
        or not isinstance(record["byte_sha256"], str)
        or _SHA256.fullmatch(record["byte_sha256"]) is None
    ):
        raise V018TerminationError(f"{expected_path} file record is invalid")


def validate_terminal_artifact_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the exact canonical terminal-artifact manifest schema."""

    manifest = _validate_exact_keys(
        payload,
        _MANIFEST_KEYS,
        context="terminal artifact manifest",
    )
    _validate_identity(manifest, context="Terminal artifact manifest")
    files = manifest["terminal_files"]
    if not isinstance(files, list) or len(files) != 2:
        raise V018TerminationError("Terminal file registry is invalid")
    expected = (
        ("terminal_attempt_record.json", "terminal_attempt_record"),
        (
            "terminal_exposure_log_snapshot.jsonl",
            "preterminal_exposure_log_snapshot",
        ),
    )
    for observed, (path, role) in zip(files, expected, strict=True):
        _validate_terminal_file_record(
            observed,
            expected_path=path,
            expected_role=role,
        )
    preterminal = manifest["preterminal_artifacts"]
    if not isinstance(preterminal, list) or not preterminal:
        raise V018TerminationError("Preterminal artifact registry is invalid")
    validated = [_validate_preterminal_file_record(item) for item in preterminal]
    paths = [str(item["path"]) for item in validated]
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or "exposure_log.jsonl" not in paths
    ):
        raise V018TerminationError("Preterminal artifact registry is incomplete")
    commitments = _validate_exact_keys(
        manifest["preterminal_commitments"],
        _COMMITMENT_KEYS,
        context="preterminal commitments",
    )
    for key, value in commitments.items():
        if key == "prediction_commitment_byte_sha256":
            if value is not None:
                raise V018TerminationError("Prediction commitment is forbidden")
        elif value is not None and (
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
        ):
            raise V018TerminationError("Preterminal commitment is invalid")
    by_path = {str(item["path"]): item for item in validated}
    commitment_paths = {
        "truth_commitments_byte_sha256": "truth_commitments.json",
        "generation_plan_commitment_byte_sha256": ("generation_plan_commitment.json"),
        "actual_analysis_hash_ledger_commitment_byte_sha256": (
            "actual_analysis_hash_ledger_commitment.json"
        ),
        "fit_commitment_byte_sha256": "fit_commitment.json",
        "center_state_checkpoint_byte_sha256": "center_state_checkpoint.json",
        "risk_state_checkpoint_byte_sha256": "risk_state_checkpoint.json",
        "calibration_mask_commitment_byte_sha256": ("calibration_mask_commitment.json"),
        "model_state_commitment_byte_sha256": "model_state_commitment.json",
    }
    for commitment_name, path in commitment_paths.items():
        committed = commitments[commitment_name]
        observed = by_path.get(path)
        if committed is not None and observed is None:
            raise V018TerminationError("Committed preterminal artifact is absent")
        if committed is not None and observed["byte_sha256"] != committed:
            raise V018TerminationError("Preterminal artifact commitment hash differs")
    policy = _validate_exact_keys(
        manifest["artifact_policy"],
        _ARTIFACT_POLICY_KEYS,
        context="terminal artifact policy",
    )
    if policy != {
        "termination_root_was_independent_and_empty": True,
        "terminal_registry_was_exclusive": True,
        "forbidden_artifacts_observed": False,
        "unknown_artifacts_observed": False,
        "links_or_directories_observed": False,
        "model_artifacts_fabricated": False,
        "prediction_artifacts_fabricated": False,
        "score_artifacts_fabricated": False,
    }:
        raise V018TerminationError("Terminal artifact policy is false")
    if (
        manifest["schema_version"] != "1.0.0"
        or manifest["registry"] != "terminal_pre_prediction"
        or manifest["content_boundary_attestation"]
        != "strict_label_free_inventory_no_truth_or_scoring_capability"
    ):
        raise V018TerminationError("Terminal manifest boundary is invalid")


def validate_termination_manifest(payload: Mapping[str, Any]) -> None:
    """Compatibility alias for the terminal-artifact manifest validator."""

    validate_terminal_artifact_manifest(payload)


def _parse_ledger_prefix(raw: bytes, context: TerminalContext) -> LedgerSnapshot:
    if not raw or not raw.endswith(b"\n"):
        raise V018TerminationError("Preterminal ledger is empty or truncated")
    try:
        events, states = parse_exposure_log_bytes(
            raw,
            expected_config_sha256=context.config_byte_sha256,
            sealed_filenames=tuple(sorted(_DEVELOPMENT_TRUTH | _HELD_OUT_TRUTH)),
        )
    except V023LedgerError as exc:
        raise V018TerminationError(
            "Preterminal ledger failed the strict canonical V2.3 parser"
        ) from exc
    try:
        progress = states[context.attempt_id]
    except KeyError:
        raise V018TerminationError("Attempt is absent from the preterminal ledger")
    if (
        progress.identity.git_commit != context.git_commit
        or progress.identity.config_byte_sha256 != context.config_byte_sha256
        or context.git_dirty is not False
    ):
        raise V018IntegrityError(
            TerminalReason.INTEGRITY_CONFIG_OR_SOURCE_HASH_MISMATCH
        )
    opened = set(progress.opened_truth_files)
    if opened & _HELD_OUT_TRUTH:
        raise V018IntegrityError(TerminalReason.INTEGRITY_FORBIDDEN_TRUTH_ACCESS)
    if opened - _DEVELOPMENT_TRUTH:
        raise V018IntegrityError(TerminalReason.INTEGRITY_INFORMATION_LEAK)
    if progress.prediction_commitment_byte_sha256 is not None:
        raise V018IntegrityError(TerminalReason.INTEGRITY_INFORMATION_LEAK)
    if progress.truth_commitments_byte_sha256 != context.truth_commitments_byte_sha256:
        raise V018IntegrityError(TerminalReason.INTEGRITY_MISSING_COMMITMENT)
    for field in _LEDGER_PHASE_COMMITMENT_FIELDS.values():
        if getattr(progress, field) != getattr(context, field):
            raise V018IntegrityError(TerminalReason.INTEGRITY_MISSING_COMMITMENT)
    derived = TerminalContext.from_progress(
        progress,
        created_utc=context.created_utc,
        terminated_utc=context.terminated_utc,
    )
    if (
        derived.last_completed_phase != context.last_completed_phase
        or derived.attempted_phase != context.attempted_phase
    ):
        raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)
    return LedgerSnapshot(
        byte_count=len(raw),
        row_count=len(events),
        byte_sha256=_sha256(raw),
        opened_truth_files=tuple(sorted(opened)),
    )


def _artifact_record(path: str, raw: bytes) -> Mapping[str, object]:
    return {
        "path": path,
        "byte_count": len(raw),
        "byte_sha256": _sha256(raw),
    }


def _artifact_record_from_file(path: Path) -> Mapping[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_count += len(chunk)
            digest.update(chunk)
    return {
        "path": path.name,
        "byte_count": byte_count,
        "byte_sha256": digest.hexdigest(),
    }


def _is_reparse(path: Path) -> bool:
    metadata = os.lstat(path)
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & flag)


def _verify_model_state_inventory(
    *,
    label_free_root: Path,
    by_path: Mapping[str, Mapping[str, object]],
    context: TerminalContext,
) -> None:
    """Verify the completed model registry without decoding numeric state."""

    model_only = {
        "calibration_population_audit.json",
        "model_state.json",
        "model_state_commitment.json",
    }
    observed_model = model_only.intersection(by_path)
    if context.model_state_commitment_byte_sha256 is None:
        if observed_model and context.attempted_phase != "model_state_committed":
            raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)
        return
    if observed_model != model_only:
        raise V018IntegrityError(TerminalReason.INTEGRITY_MISSING_COMMITMENT)
    raw = (label_free_root / "model_state_commitment.json").read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V018IntegrityError(
            TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {"protocol_id", "config_sha256", "git_commit", "files", "created_utc"}
        or payload.get("protocol_id") != context.protocol_id
        or payload.get("config_sha256") != context.config_byte_sha256
        or payload.get("git_commit") != context.git_commit
        or not isinstance(payload.get("files"), list)
        or len(payload["files"]) != len(_MODEL_STATE_COMMITMENT_FILES)
    ):
        raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)
    for name, entry in zip(
        _MODEL_STATE_COMMITMENT_FILES,
        payload["files"],
        strict=True,
    ):
        observed = by_path.get(name)
        if (
            observed is None
            or not isinstance(entry, Mapping)
            or set(entry) != {"path", "row_count", "byte_count", "byte_sha256"}
            or entry.get("path") != name
            or entry.get("row_count") != 1
            or entry.get("byte_count") != observed["byte_count"]
            or entry.get("byte_sha256") != observed["byte_sha256"]
        ):
            raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)


def _scan_preterminal_artifacts(
    *,
    label_free_root: Path,
    ledger_snapshot_bytes: bytes,
    snapshot: LedgerSnapshot,
    context: TerminalContext,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for path in sorted(label_free_root.iterdir(), key=lambda value: value.name):
        name = path.name
        if _is_reparse(path) or not path.is_file():
            raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)
        if name in _HELD_OUT_TRUTH or name in _DEVELOPMENT_TRUTH:
            raise V018IntegrityError(TerminalReason.INTEGRITY_FORBIDDEN_TRUTH_ACCESS)
        if name in _FORBIDDEN_PREDICTION_SCORE_ARTIFACTS:
            raise V018IntegrityError(TerminalReason.INTEGRITY_INFORMATION_LEAK)
        if name not in _ALLOWED_PRETERMINAL_ARTIFACTS:
            raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)
        record = (
            _artifact_record(name, ledger_snapshot_bytes)
            if name == "exposure_log.jsonl"
            else _artifact_record_from_file(path)
        )
        if record["byte_count"] == 0:
            raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)
        records.append(record)

    by_path = {str(record["path"]): record for record in records}
    expected_commitments = {
        "truth_commitments.json": context.truth_commitments_byte_sha256,
        "generation_plan_commitment.json": (
            context.generation_plan_commitment_byte_sha256
        ),
        "actual_analysis_hash_ledger_commitment.json": (
            context.actual_analysis_hash_ledger_commitment_byte_sha256
        ),
        "fit_commitment.json": context.fit_commitment_byte_sha256,
        "center_state_checkpoint.json": (context.center_state_checkpoint_byte_sha256),
        "risk_state_checkpoint.json": (context.risk_state_checkpoint_byte_sha256),
        "calibration_mask_commitment.json": (
            context.calibration_mask_commitment_byte_sha256
        ),
        "model_state_commitment.json": (context.model_state_commitment_byte_sha256),
    }
    for path, expected_hash in expected_commitments.items():
        observed = by_path.get(path)
        if expected_hash is None and observed is not None:
            allowed_partial_phase = _COMMITMENT_PHASE_BY_ARTIFACT.get(path)
            if context.attempted_phase != allowed_partial_phase:
                raise V018IntegrityError(
                    TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH
                )
        if expected_hash is not None and observed is None:
            raise V018IntegrityError(TerminalReason.INTEGRITY_MISSING_COMMITMENT)
        if (
            expected_hash is not None
            and observed is not None
            and observed["byte_sha256"] != expected_hash
        ):
            raise V018IntegrityError(TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH)
    if "exposure_log.jsonl" not in by_path:
        raise V018IntegrityError(TerminalReason.INTEGRITY_MISSING_COMMITMENT)
    if "calibration_truth.csv" in snapshot.opened_truth_files and (
        context.calibration_mask_commitment_byte_sha256 is None
        or "calibration_mask_commitment.json" not in by_path
    ):
        raise V018IntegrityError(TerminalReason.INTEGRITY_MISSING_COMMITMENT)
    _verify_model_state_inventory(
        label_free_root=label_free_root,
        by_path=by_path,
        context=context,
    )
    observed_prediction = _PREDICTION_STAGE_FILES.intersection(by_path)
    if observed_prediction and context.attempted_phase != "prediction_started":
        raise V018IntegrityError(TerminalReason.INTEGRITY_INFORMATION_LEAK)
    return tuple(records)


def _preterminal_registry_sha256(
    artifacts: Sequence[Mapping[str, object]],
) -> str:
    return _sha256(_canonical_json_bytes({"preterminal_artifacts": list(artifacts)}))


def _build_attempt_record(
    *,
    context: TerminalContext,
    classification: TerminalClassification,
    snapshot: LedgerSnapshot,
    preterminal_artifacts: Sequence[Mapping[str, object]],
    traceback_bytes: bytes,
    stderr_bytes: bytes,
) -> Mapping[str, Any]:
    try:
        traceback_payload = json.loads(traceback_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V018TerminationError("Sanitized structural traceback is invalid") from exc
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "protocol_id": context.protocol_id,
        "attempt_id": context.attempt_id,
        "created_utc": context.terminated_utc,
        "git_commit": context.git_commit,
        "git_dirty": context.git_dirty,
        "config_byte_sha256": context.config_byte_sha256,
        "phase": context.attempted_phase,
        "last_completed_phase": context.last_completed_phase,
        "attempt_disposition": classification.disposition.value,
        "classification_mode": classification.mode.value,
        "reason_code": classification.reason.value,
        "scientific_status": classification.scientific_status,
        "truth_commitments_byte_sha256": (context.truth_commitments_byte_sha256),
        "prediction_commitment_byte_sha256": None,
        "opened_truth_files": list(snapshot.opened_truth_files),
        "message": classification.safe_message,
        "preterminal_artifacts_byte_sha256": (
            _preterminal_registry_sha256(preterminal_artifacts)
        ),
        "terminal_exposure_log_snapshot_byte_sha256": snapshot.byte_sha256,
        "diagnostics": {
            "structural_traceback": traceback_payload,
            "sanitized_stderr": {
                "byte_count": len(stderr_bytes),
                "byte_sha256": _sha256(stderr_bytes),
                "redaction_policy": _STDERR_REDACTION_POLICY,
            },
        },
    }
    validate_terminal_attempt_record(payload)
    return payload


def _terminal_file_record(
    path: str,
    role: str,
    raw: bytes,
) -> Mapping[str, object]:
    return {
        "path": path,
        "role": role,
        "byte_count": len(raw),
        "byte_sha256": _sha256(raw),
    }


def build_terminal_artifact_manifest(
    *,
    context: TerminalContext,
    attempt_record_bytes: bytes,
    ledger_snapshot_bytes: bytes,
    preterminal_artifacts: Sequence[Mapping[str, object]],
) -> Mapping[str, Any]:
    """Build the manifest committing terminal and prior label-free artifacts."""

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "protocol_id": context.protocol_id,
        "attempt_id": context.attempt_id,
        "created_utc": context.terminated_utc,
        "git_commit": context.git_commit,
        "config_byte_sha256": context.config_byte_sha256,
        "registry": "terminal_pre_prediction",
        "terminal_files": [
            _terminal_file_record(
                "terminal_attempt_record.json",
                "terminal_attempt_record",
                attempt_record_bytes,
            ),
            _terminal_file_record(
                "terminal_exposure_log_snapshot.jsonl",
                "preterminal_exposure_log_snapshot",
                ledger_snapshot_bytes,
            ),
        ],
        "preterminal_artifacts": list(preterminal_artifacts),
        "preterminal_commitments": {
            "truth_commitments_byte_sha256": (context.truth_commitments_byte_sha256),
            "generation_plan_commitment_byte_sha256": (
                context.generation_plan_commitment_byte_sha256
            ),
            "actual_analysis_hash_ledger_commitment_byte_sha256": (
                context.actual_analysis_hash_ledger_commitment_byte_sha256
            ),
            "fit_commitment_byte_sha256": (context.fit_commitment_byte_sha256),
            "center_state_checkpoint_byte_sha256": (
                context.center_state_checkpoint_byte_sha256
            ),
            "risk_state_checkpoint_byte_sha256": (
                context.risk_state_checkpoint_byte_sha256
            ),
            "calibration_mask_commitment_byte_sha256": (
                context.calibration_mask_commitment_byte_sha256
            ),
            "model_state_commitment_byte_sha256": (
                context.model_state_commitment_byte_sha256
            ),
            "prediction_commitment_byte_sha256": None,
        },
        "artifact_policy": {
            "termination_root_was_independent_and_empty": True,
            "terminal_registry_was_exclusive": True,
            "forbidden_artifacts_observed": False,
            "unknown_artifacts_observed": False,
            "links_or_directories_observed": False,
            "model_artifacts_fabricated": False,
            "prediction_artifacts_fabricated": False,
            "score_artifacts_fabricated": False,
        },
        "content_boundary_attestation": (
            "strict_label_free_inventory_no_truth_or_scoring_capability"
        ),
    }
    validate_terminal_artifact_manifest(payload)
    return payload


def build_termination_manifest(
    *,
    context: TerminalContext,
    attempt_record_bytes: bytes,
    ledger_snapshot_bytes: bytes,
    preterminal_artifacts: Sequence[Mapping[str, object]],
) -> Mapping[str, Any]:
    """Compatibility alias for the exact V2.3 artifact-manifest builder."""

    return build_terminal_artifact_manifest(
        context=context,
        attempt_record_bytes=attempt_record_bytes,
        ledger_snapshot_bytes=ledger_snapshot_bytes,
        preterminal_artifacts=preterminal_artifacts,
    )


def _atomic_exclusive_write(path: Path, raw: bytes) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_FLAG,
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("Terminal artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _is_reparse(path) or path.read_bytes() != raw:
                raise V018TerminationError(
                    f"Terminal artifact conflicts: {path.name}"
                ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _publication_lock(root: Path) -> Iterator[None]:
    with _PROCESS_PUBLICATION_LOCK:
        lock_path = root / _LOCK_FILENAME
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY_FLAG,
                0o600,
            )
        except FileExistsError as exc:
            raise V018TerminationError(
                "Terminal publication is already locked"
            ) from exc
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            yield
        finally:
            os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _terminal_ledger_record(
    *,
    context: TerminalContext,
    classification: TerminalClassification,
    attempt_record_sha256: str,
    snapshot_sha256: str,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "terminal_artifact_manifest_commitment",
        "attempt_id": context.attempt_id,
        "git_commit": context.git_commit,
        "git_dirty": context.git_dirty,
        "config_byte_sha256": context.config_byte_sha256,
        "created_utc": context.terminated_utc,
        "phase": context.attempted_phase,
        "terminal_disposition": classification.disposition.value,
        "classification_mode": classification.mode.value,
        "scientific_status": classification.scientific_status,
        "reason_code": classification.reason.value,
        "terminal_attempt_record_byte_sha256": attempt_record_sha256,
        "terminal_exposure_log_snapshot_byte_sha256": snapshot_sha256,
        "terminal_artifact_manifest_byte_sha256": manifest_sha256,
    }


def _validate_terminal_ledger_record(
    value: object,
    *,
    expected: Mapping[str, Any],
) -> None:
    record = _validate_exact_keys(
        value,
        _TERMINAL_LEDGER_KEYS,
        context="terminal ledger record",
    )
    if record != expected:
        raise V018TerminationError("Terminal ledger commitment conflicts")


def _finish_terminal_ledger(
    path: Path,
    *,
    expected_prefix: bytes,
    expected_record: Mapping[str, Any],
) -> bool:
    record_raw = _canonical_json_bytes(expected_record)
    current = path.read_bytes()
    if current == expected_prefix:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | _BINARY_FLAG)
        try:
            written = os.write(descriptor, record_raw)
            if written != len(record_raw):
                raise OSError("Terminal ledger append was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if path.read_bytes() != expected_prefix + record_raw:
            raise V018TerminationError("Terminal ledger append was not exact")
        return True
    if not current.startswith(expected_prefix):
        raise V018TerminationError("Live ledger prefix changed after snapshot")
    suffix = current[len(expected_prefix) :]
    if not suffix.endswith(b"\n") or len(suffix.splitlines()) != 1:
        raise V018TerminationError("Terminal ledger has later or malformed records")
    observed_record = _canonical_mapping_from_bytes(
        suffix,
        context="terminal ledger record",
    )
    _validate_terminal_ledger_record(observed_record, expected=expected_record)
    return False


def _canonical_mapping_from_bytes(
    raw: bytes,
    *,
    context: str,
) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V018TerminationError(f"Existing {context} is invalid") from exc
    if not isinstance(value, Mapping):
        raise V018TerminationError(f"Existing {context} is not an object")
    if _canonical_json_bytes(value) != raw:
        raise V018TerminationError(f"Existing {context} is noncanonical")
    return value


def _publication_bytes(
    *,
    context: TerminalContext,
    classification: TerminalClassification,
    label_free_root: Path,
    ledger_snapshot_bytes: bytes,
    error: BaseException,
    stderr: str | bytes,
    repo_root: str | Path | None,
) -> tuple[
    bytes,
    bytes,
    bytes,
    LedgerSnapshot,
    tuple[Mapping[str, object], ...],
]:
    snapshot = _parse_ledger_prefix(ledger_snapshot_bytes, context)
    artifacts = _scan_preterminal_artifacts(
        label_free_root=label_free_root,
        ledger_snapshot_bytes=ledger_snapshot_bytes,
        snapshot=snapshot,
        context=context,
    )
    traceback_bytes = sanitized_structural_traceback(
        error,
        repo_root=repo_root,
    )
    stderr_bytes = sanitized_stderr_bytes(stderr)
    attempt_record = _build_attempt_record(
        context=context,
        classification=classification,
        snapshot=snapshot,
        preterminal_artifacts=artifacts,
        traceback_bytes=traceback_bytes,
        stderr_bytes=stderr_bytes,
    )
    attempt_raw = _canonical_json_bytes(attempt_record)
    manifest = build_terminal_artifact_manifest(
        context=context,
        attempt_record_bytes=attempt_raw,
        ledger_snapshot_bytes=ledger_snapshot_bytes,
        preterminal_artifacts=artifacts,
    )
    manifest_raw = _canonical_json_bytes(manifest)
    return attempt_raw, ledger_snapshot_bytes, manifest_raw, snapshot, artifacts


def _resolve_publication_roots(
    *,
    termination_root: str | Path,
    label_free_artifact_root: str | Path,
) -> tuple[Path, Path, Path]:
    def physical_root(raw: str | Path, *, context: str) -> Path:
        lexical = Path(os.path.abspath(os.fspath(raw)))
        for candidate in (lexical, *lexical.parents):
            if os.path.lexists(candidate) and _is_reparse(candidate):
                raise V018TerminationError(f"{context} root traverses a reparse point")
        resolved = lexical.resolve()
        if not resolved.is_dir():
            raise V018TerminationError("Terminal publication roots must exist")
        for parent in (resolved, *resolved.parents):
            if os.path.lexists(parent) and _is_reparse(parent):
                raise V018TerminationError(f"{context} root traverses a reparse point")
        return resolved

    terminal = physical_root(termination_root, context="Termination")
    label_free = physical_root(
        label_free_artifact_root,
        context="Label-free",
    )
    if (
        terminal == label_free
        or terminal in label_free.parents
        or label_free in terminal.parents
    ):
        raise V018TerminationError("Termination and label-free roots must be disjoint")
    ledger = label_free / "exposure_log.jsonl"
    if not ledger.exists() or _is_reparse(ledger) or not ledger.is_file():
        raise V018IntegrityError(TerminalReason.INTEGRITY_MISSING_COMMITMENT)
    return terminal, label_free, ledger


def _publish_terminal_classification(
    *,
    termination_root: str | Path,
    label_free_artifact_root: str | Path,
    context: TerminalContext,
    error: BaseException,
    classification: TerminalClassification,
    stderr: str | bytes,
    repo_root: str | Path | None,
) -> PublishedTermination:
    root, label_free, ledger = _resolve_publication_roots(
        termination_root=termination_root,
        label_free_artifact_root=label_free_artifact_root,
    )
    with _publication_lock(root), locked_ledger(ledger):
        observed_paths = {
            path.name: path for path in root.iterdir() if path.name != _LOCK_FILENAME
        }
        if any(
            _is_reparse(path) or not path.is_file() for path in observed_paths.values()
        ):
            raise V018TerminationError("Terminal registry contains unsafe entries")
        observed_names = set(observed_paths)
        allowed_prefixes = {
            frozenset(_PUBLICATION_ORDER[:count])
            for count in range(len(_PUBLICATION_ORDER) + 1)
        }
        if frozenset(observed_names) not in allowed_prefixes:
            raise V018TerminationError(
                "Termination root is not an exact publication prefix"
            )

        snapshot_path = root / "terminal_exposure_log_snapshot.jsonl"
        ledger_snapshot_raw = (
            snapshot_path.read_bytes()
            if snapshot_path.exists()
            else ledger.read_bytes()
        )
        (
            attempt_raw,
            snapshot_raw,
            manifest_raw,
            snapshot,
            artifacts,
        ) = _publication_bytes(
            context=context,
            classification=classification,
            label_free_root=label_free,
            ledger_snapshot_bytes=ledger_snapshot_raw,
            error=error,
            stderr=stderr,
            repo_root=repo_root,
        )
        expected_bytes = {
            "terminal_attempt_record.json": attempt_raw,
            "terminal_exposure_log_snapshot.jsonl": snapshot_raw,
            "terminal_artifact_manifest.json": manifest_raw,
        }
        for name, path in observed_paths.items():
            if path.read_bytes() != expected_bytes[name]:
                raise V018TerminationError(f"Terminal artifact conflicts: {name}")
        for name in _PUBLICATION_ORDER:
            if name not in observed_names:
                _atomic_exclusive_write(root / name, expected_bytes[name])

        attempt_payload = _canonical_mapping_from_bytes(
            attempt_raw,
            context="terminal attempt record",
        )
        manifest_payload = _canonical_mapping_from_bytes(
            manifest_raw,
            context="terminal artifact manifest",
        )
        validate_terminal_attempt_record(attempt_payload)
        validate_terminal_artifact_manifest(manifest_payload)

        rescanned = _scan_preterminal_artifacts(
            label_free_root=label_free,
            ledger_snapshot_bytes=snapshot_raw,
            snapshot=snapshot,
            context=context,
        )
        if rescanned != artifacts:
            raise V018TerminationError(
                "Label-free artifacts changed during terminal publication"
            )

        attempt_sha = _sha256(attempt_raw)
        snapshot_sha = _sha256(snapshot_raw)
        manifest_sha = _sha256(manifest_raw)
        expected_record = _terminal_ledger_record(
            context=context,
            classification=classification,
            attempt_record_sha256=attempt_sha,
            snapshot_sha256=snapshot_sha,
            manifest_sha256=manifest_sha,
        )
        appended = _finish_terminal_ledger(
            ledger,
            expected_prefix=snapshot_raw,
            expected_record=expected_record,
        )
        return PublishedTermination(
            manifest_path=root / "terminal_artifact_manifest.json",
            manifest_byte_sha256=manifest_sha,
            attempt_record_byte_sha256=attempt_sha,
            ledger_snapshot_byte_sha256=snapshot_sha,
            ledger_record_appended=appended,
            disposition=classification.disposition,
            reason=classification.reason,
        )


def _environment_guarded_error(
    *,
    context: TerminalContext,
    error: BaseException,
    repo_root: str | Path | None,
) -> BaseException:
    """Convert any failed fresh environment attestation into an integrity void."""

    try:
        environment = verify_formal_environment(repo_root or _PROJECT_ROOT)
    except BaseException:
        return V018IntegrityError(TerminalReason.INTEGRITY_ENVIRONMENT_MISMATCH)
    if (
        environment.protocol_id != context.protocol_id
        or environment.git_commit != context.git_commit
        or environment.git_dirty is not context.git_dirty
        or environment.config_byte_sha256 != context.config_byte_sha256
    ):
        return V018IntegrityError(TerminalReason.INTEGRITY_ENVIRONMENT_MISMATCH)
    return error


def publish_terminal(
    *,
    termination_root: str | Path,
    label_free_artifact_root: str | Path,
    context: TerminalContext,
    error: BaseException,
    stderr: str | bytes = b"",
    repo_root: str | Path | None = None,
) -> PublishedTermination:
    """Publish any typed/default V2.3 pre-prediction terminal disposition."""

    guarded_error = _environment_guarded_error(
        context=context,
        error=error,
        repo_root=repo_root,
    )
    classification = classify_terminal_exception(guarded_error)
    return _publish_terminal_classification(
        termination_root=termination_root,
        label_free_artifact_root=label_free_artifact_root,
        context=context,
        error=guarded_error,
        classification=classification,
        stderr=stderr,
        repo_root=repo_root,
    )


def publish_terminal_inconclusive(
    *,
    termination_root: str | Path,
    label_free_artifact_root: str | Path,
    context: TerminalContext,
    error: BaseException,
    stderr: str | bytes = b"",
    repo_root: str | Path | None = None,
) -> PublishedTermination:
    """Publish only a declared scientific terminal-inconclusive condition."""

    guarded_error = _environment_guarded_error(
        context=context,
        error=error,
        repo_root=repo_root,
    )
    classification = classify_terminal_exception(guarded_error)
    if not classification.is_scientific_terminal:
        if classification.reason is TerminalReason.INTEGRITY_ENVIRONMENT_MISMATCH:
            return _publish_terminal_classification(
                termination_root=termination_root,
                label_free_artifact_root=label_free_artifact_root,
                context=context,
                error=guarded_error,
                classification=classification,
                stderr=stderr,
                repo_root=repo_root,
            )
        raise V018TerminationError(
            "A non-scientific exception cannot publish an inconclusive registry"
        )
    return _publish_terminal_classification(
        termination_root=termination_root,
        label_free_artifact_root=label_free_artifact_root,
        context=context,
        error=guarded_error,
        classification=classification,
        stderr=stderr,
        repo_root=repo_root,
    )


__all__ = [
    "ClassificationMode",
    "INTEGRITY_TERMINAL_REASONS",
    "INTERRUPTION_TERMINAL_REASONS",
    "LedgerSnapshot",
    "PublishedTermination",
    "SCIENTIFIC_TERMINAL_REASONS",
    "TerminalClassification",
    "TerminalContext",
    "TerminalDisposition",
    "TerminalReason",
    "V018ConservativeVoid",
    "V018IntegrityError",
    "V018Interrupted",
    "V018TerminalInconclusive",
    "V018TerminationError",
    "V023_AMENDMENT_BYTE_SHA256",
    "build_terminal_artifact_manifest",
    "build_termination_manifest",
    "classify_terminal_exception",
    "publish_terminal",
    "publish_terminal_inconclusive",
    "sanitized_stderr_bytes",
    "sanitized_structural_traceback",
    "validate_terminal_artifact_manifest",
    "validate_terminal_attempt_record",
    "validate_termination_manifest",
]
