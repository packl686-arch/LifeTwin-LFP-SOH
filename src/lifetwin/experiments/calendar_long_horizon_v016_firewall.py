"""V2.1 formal-attempt state machine and phase-scoped truth capabilities.

V2.1 inserts two outcome-free commitments into the inherited V2 lifecycle:
the complete generation-coordinate plan before generation and the exact
calibration eligibility mask before calibration truth is opened.  This module
owns those temporal guarantees without broadening any inherited truth
capability.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    assert_separate_truth_roots,
    canonical_json_bytes,
    read_canonical_csv,
    read_truth_commitments,
    verify_prediction_commitment,
)
from lifetwin.experiments.calendar_long_horizon_v016_ledger import (
    AttemptProgress,
    EXIT_STATUSES,
    FormalAttemptIdentity,
    OPENED_BY_PHASE,
    PHASES,
    PHASE_COMMITMENT_FIELDS,
    V021LedgerError,
    append_exposure_event_cas,
    phase_commitment_message,
    read_exposure_log as read_v021_exposure_log,
    validate_exposure_events,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_EXPECTED_SEED_ROOTS,
    V021_PROTOCOL_ID,
)


class V021FirewallError(ValueError):
    """Raised when a V2.1 attempt crosses a frozen information boundary."""


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
_OPEN_PHASE_FILES = {
    "center_truth_opened": ("center_development_truth.csv",),
    "risk_truth_opened": ("risk_development_truth.csv",),
    "calibration_truth_opened": ("calibration_truth.csv",),
    "scoring_truth_opened": _SCORING_TRUTH,
}
_RECOVERABLE_TRUTH_PHASES = frozenset(
    {
        "center_state_committed",
        "risk_state_committed",
        "model_state_committed",
        "scoring_completed",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIT_COMMITMENT_FILENAMES = (
    "generation_plan_commitment.json",
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "truth_commitments.json",
    "actual_analysis_hash_ledger_commitment.json",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
)
_CENTER_INPUT_FILENAMES = (
    "center_development_truth.csv",
    "fit_commitment.json",
    "forecast_coordinates.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
    "operating_pack.csv",
    "prefix_pack.csv",
)
_RISK_INPUT_FILENAMES = (
    "center_state_checkpoint.json",
    "fit_commitment.json",
    "forecast_coordinates.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
    "operating_pack.csv",
    "prefix_pack.csv",
    "risk_development_truth.csv",
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


def _sha256_file(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise V021FirewallError(f"Cannot read committed artifact: {path}") from exc


def _is_reparse_entry(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise V021FirewallError(f"Cannot inspect formal path: {path}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & flag)


def _physical_root(raw_path: str | Path, *, context: str) -> Path:
    path = Path(os.path.abspath(os.fspath(raw_path)))
    for component in (*reversed(path.parents), path):
        if os.path.lexists(component) and _is_reparse_entry(component):
            raise V021FirewallError(f"{context} root has a reparse-point ancestor")
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise V021FirewallError(
            f"{context} root must be a pre-created physical directory"
        )
    return path


def _direct_physical_child(
    root: Path,
    raw_path: str | Path,
    *,
    filename: str,
    context: str,
    must_exist: bool = True,
) -> Path:
    candidate = Path(os.path.abspath(os.fspath(raw_path)))
    expected = root / filename
    if candidate != expected:
        raise V021FirewallError(f"{filename} must be a direct label-free artifact")
    if not os.path.lexists(candidate):
        if must_exist:
            raise V021FirewallError(f"{context} is absent")
        return candidate
    if _is_reparse_entry(candidate) or not candidate.is_file():
        raise V021FirewallError(f"{context} is not a direct regular file")
    try:
        if candidate.resolve(strict=True) != candidate:
            raise V021FirewallError(f"{context} changed during resolution")
    except OSError as exc:
        raise V021FirewallError(f"Cannot bind {context}") from exc
    return candidate


def _require_ledger_path(label_root: Path, ledger_path: str | Path) -> Path:
    return _direct_physical_child(
        label_root,
        ledger_path,
        filename="exposure_log.jsonl",
        context="Exposure ledger",
        must_exist=False,
    )


def _strict_json_object(
    raw: bytes,
    *,
    filename: str,
    compact: bool,
) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise V021FirewallError(f"{filename} contains duplicate key {key!r}")
            decoded[key] = value
        return decoded

    def reject_constant(token: str) -> None:
        raise V021FirewallError(f"{filename} contains nonfinite token {token}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V021FirewallError(f"{filename} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise V021FirewallError(f"{filename} must contain one JSON object")
    if compact:
        try:
            canonical = (
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                + b"\n"
            )
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise V021FirewallError(f"{filename} is not finite ASCII JSON") from exc
    else:
        try:
            canonical = canonical_json_bytes(payload)
        except V015ArtifactError as exc:
            raise V021FirewallError(f"{filename} is not finite JSON") from exc
    if raw != canonical:
        raise V021FirewallError(f"{filename} is not canonical JSON")
    return payload


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V021FirewallError(f"{context} is not lowercase SHA256")
    return value


def _artifact_path(
    *,
    label_root: Path,
    sealed_root: Path,
    filename: str,
) -> Path:
    root = (
        sealed_root
        if filename in {"center_development_truth.csv", "risk_development_truth.csv"}
        else label_root
    )
    candidate = root / filename
    if not os.path.lexists(candidate):
        raise V021FirewallError(f"Committed input is absent: {filename}")
    if _is_reparse_entry(candidate) or not candidate.is_file():
        raise V021FirewallError(
            f"Committed input is not a direct regular file: {filename}"
        )
    return candidate


def verify_formal_attempt_environment(identity: FormalAttemptIdentity) -> None:
    """Bind a truth-capability call to the frozen V2.1 implementation."""

    from lifetwin.experiments.calendar_long_horizon_v016_environment import (
        verify_formal_environment,
    )

    project_root = Path(__file__).resolve().parents[3]
    environment = verify_formal_environment(project_root)
    if (
        environment.git_dirty
        or environment.git_commit != identity.git_commit
        or environment.config_byte_sha256 != identity.config_byte_sha256
        or environment.protocol_id != V021_PROTOCOL_ID
    ):
        raise V021FirewallError(
            "Current formal environment differs from the V2.1 attempt identity"
        )


def validate_formal_exposure_events(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: FrozenArtifactContract,
) -> Mapping[str, AttemptProgress]:
    """Validate all interleaved V2.1 attempts in one publication ledger."""

    if contract.protocol_id != V021_PROTOCOL_ID:
        raise V021FirewallError("Firewall requires the V2.1 artifact contract")
    try:
        return validate_exposure_events(
            events,
            expected_config_sha256=contract.config_byte_sha256,
            sealed_filenames=contract.sealed_filenames,
        )
    except V021LedgerError as exc:
        raise V021FirewallError(str(exc)) from exc


def validate_formal_exposure_log(
    path: str | Path,
    contract: FrozenArtifactContract,
) -> Mapping[str, AttemptProgress]:
    """Read canonical JSONL and enforce the complete V2.1 state machine."""

    if contract.protocol_id != V021_PROTOCOL_ID:
        raise V021FirewallError("Firewall requires the V2.1 artifact contract")
    try:
        _, states, _ = read_v021_exposure_log(
            path,
            expected_config_sha256=contract.config_byte_sha256,
            sealed_filenames=contract.sealed_filenames,
        )
    except V021LedgerError as exc:
        raise V021FirewallError(str(exc)) from exc
    return states


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
    """Validate a prospective V2.1 transition before appending its bytes."""

    event = {
        "attempt_id": identity.attempt_id,
        "created_utc": created_utc,
        "git_commit": identity.git_commit,
        "git_dirty": False,
        "config_byte_sha256": identity.config_byte_sha256,
        "phase": phase,
        "truth_commitments_byte_sha256": truth_commitments_byte_sha256,
        "prediction_commitment_byte_sha256": prediction_commitment_byte_sha256,
        "opened_truth_files": list(sorted(OPENED_BY_PHASE.get(phase, ()))),
        "exit_status": exit_status,
        "message": message,
    }
    if contract.protocol_id != V021_PROTOCOL_ID:
        raise V021FirewallError("Firewall requires the V2.1 artifact contract")
    try:
        return append_exposure_event_cas(
            path,
            event,
            expected_config_sha256=contract.config_byte_sha256,
            sealed_filenames=contract.sealed_filenames,
        )
    except V021LedgerError as exc:
        raise V021FirewallError(str(exc)) from exc


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
    """Append a terminal phase event while preserving the original error."""

    try:
        append_formal_exposure_event(
            path=ledger_path,
            identity=identity,
            contract=contract,
            created_utc=created_utc,
            phase=phase,
            exit_status=exit_status,
            truth_commitments_byte_sha256=truth_commitments_byte_sha256,
            prediction_commitment_byte_sha256=prediction_commitment_byte_sha256,
            message=message,
        )
    except BaseException as checkpoint_error:
        error.add_note(
            "The original phase error was preserved, but its terminal "
            f"{exit_status!r} exposure checkpoint also failed: "
            f"{checkpoint_error!r}"
        )
        raise error from checkpoint_error


def _require_direct_commitment(
    *,
    label_root: Path,
    path: str | Path | None,
    filename: str,
    expected_sha256: str | None,
    context: str,
) -> Path:
    if path is None or expected_sha256 is None:
        raise V021FirewallError(f"{context} lacks its checkpointed commitment")
    resolved = _direct_physical_child(
        label_root,
        path,
        filename=filename,
        context=context,
    )
    if _sha256_file(resolved) != expected_sha256:
        raise V021FirewallError(f"{context} differs from its ledger commitment")
    return resolved


def _verify_generation_plan(
    *,
    label_root: Path,
    progress: AttemptProgress,
    contract: FrozenArtifactContract,
) -> None:
    expected = progress.generation_plan_commitment_byte_sha256
    path = _require_direct_commitment(
        label_root=label_root,
        path=label_root / "generation_plan_commitment.json",
        filename="generation_plan_commitment.json",
        expected_sha256=expected,
        context="Generation-plan capability",
    )
    payload = _strict_json_object(
        path.read_bytes(),
        filename=path.name,
        compact=True,
    )
    try:
        from lifetwin.experiments.calendar_long_horizon_v016_collision import (
            audit_formal_v021_generation_plan,
            verify_generation_plan_commitment,
        )
        from lifetwin.experiments.calendar_long_horizon_v016_contract import (
            load_v021_contract_view,
        )

        view = load_v021_contract_view()
        if view.artifacts != contract:
            raise V021FirewallError(
                "Generation plan uses a different V2.1 artifact contract"
            )
        assert expected is not None
        verify_generation_plan_commitment(
            payload,
            expected_byte_sha256=expected,
            expected_current_protocol_id=V021_PROTOCOL_ID,
            expected_current_protocol_byte_sha256=contract.config_byte_sha256,
            expected_predecessor_protocol_id=(
                "synthetic_long_horizon_identifiability_v2"
            ),
            expected_predecessor_protocol_byte_sha256=view.base_config_byte_sha256,
        )
        recomputed = audit_formal_v021_generation_plan(view)
    except V021FirewallError:
        raise
    except (TypeError, ValueError) as exc:
        raise V021FirewallError("Generation-plan semantic verification failed") from exc
    if recomputed.canonical_bytes != path.read_bytes() or (
        recomputed.byte_sha256 != expected
    ):
        raise V021FirewallError(
            "Generation plan differs from independent recomputation"
        )


def _verify_actual_analysis_hash_ledger(
    *,
    label_root: Path,
    progress: AttemptProgress,
) -> None:
    path = _require_direct_commitment(
        label_root=label_root,
        path=label_root / "actual_analysis_hash_ledger_commitment.json",
        filename="actual_analysis_hash_ledger_commitment.json",
        expected_sha256=(progress.actual_analysis_hash_ledger_commitment_byte_sha256),
        context="Actual-analysis hash-ledger capability",
    )
    payload = _strict_json_object(
        path.read_bytes(),
        filename=path.name,
        compact=True,
    )
    try:
        from lifetwin.experiments.calendar_long_horizon_v016_collision import (
            verify_actual_analysis_hash_ledger_commitment,
        )

        expected = progress.actual_analysis_hash_ledger_commitment_byte_sha256
        assert expected is not None
        verify_actual_analysis_hash_ledger_commitment(
            payload,
            expected_byte_sha256=expected,
            expected_protocol_id=V021_PROTOCOL_ID,
            expected_random_ranking_root=V021_EXPECTED_SEED_ROOTS["random_rankings"],
            expected_stress_permutation_root=V021_EXPECTED_SEED_ROOTS[
                "stress_permutations"
            ],
        )
    except (TypeError, ValueError) as exc:
        raise V021FirewallError(
            "Actual-analysis hash ledger failed semantic verification"
        ) from exc


def _verify_fit_commitment(
    *,
    label_root: Path,
    progress: AttemptProgress,
    contract: FrozenArtifactContract,
    formal: bool,
) -> None:
    path = _require_direct_commitment(
        label_root=label_root,
        path=label_root / "fit_commitment.json",
        filename="fit_commitment.json",
        expected_sha256=progress.fit_commitment_byte_sha256,
        context="Label-free fit capability",
    )
    payload = _strict_json_object(
        path.read_bytes(),
        filename=path.name,
        compact=False,
    )
    if (
        set(payload) != _FIT_COMMITMENT_KEYS
        or payload.get("protocol_id") != V021_PROTOCOL_ID
        or payload.get("config_sha256") != contract.config_byte_sha256
        or payload.get("git_commit") != progress.identity.git_commit
        or payload.get("worker_count") != 6
        or not isinstance(payload.get("created_utc"), str)
    ):
        raise V021FirewallError("fit_commitment.json identity changed")
    entries = payload.get("files")
    if not isinstance(entries, list) or len(entries) != len(_FIT_COMMITMENT_FILENAMES):
        raise V021FirewallError("fit_commitment.json file registry changed")
    for filename, entry in zip(
        _FIT_COMMITMENT_FILENAMES,
        entries,
        strict=True,
    ):
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "row_count", "byte_count", "byte_sha256"}
            or entry.get("path") != filename
            or isinstance(entry.get("row_count"), bool)
            or not isinstance(entry.get("row_count"), int)
            or int(entry["row_count"]) < 1
            or isinstance(entry.get("byte_count"), bool)
            or not isinstance(entry.get("byte_count"), int)
            or int(entry["byte_count"]) < 1
        ):
            raise V021FirewallError(
                "fit_commitment.json contains invalid file metadata"
            )
        committed = _direct_physical_child(
            label_root,
            label_root / filename,
            filename=filename,
            context=f"Fit input {filename}",
        )
        raw = committed.read_bytes()
        if filename.endswith(".csv"):
            try:
                row_count = len(read_canonical_csv(committed, contract, formal=formal))
            except V015ArtifactError as exc:
                raise V021FirewallError(
                    f"Fit input {filename} is not a canonical frozen table"
                ) from exc
        else:
            row_count = 1
        if (
            entry["row_count"] != row_count
            or entry["byte_count"] != len(raw)
            or entry["byte_sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise V021FirewallError("A label-free fit input changed after commitment")


def _verify_input_hashes(
    value: object,
    *,
    expected_filenames: tuple[str, ...],
    label_root: Path,
    sealed_root: Path,
    context: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_filenames):
        raise V021FirewallError(f"{context} input registry changed")
    observed: dict[str, str] = {}
    for filename in expected_filenames:
        expected = _require_sha256(
            value[filename],
            context=f"{context}.{filename}",
        )
        path = _artifact_path(
            label_root=label_root,
            sealed_root=sealed_root,
            filename=filename,
        )
        actual = _sha256_file(path)
        if actual != expected:
            raise V021FirewallError(
                f"{context} input changed after checkpoint: {filename}"
            )
        observed[filename] = actual
    return observed


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise V021FirewallError(f"{context} must be a positive integer")
    return value


def _verify_center_checkpoint(
    *,
    label_root: Path,
    sealed_root: Path,
    progress: AttemptProgress,
    contract: FrozenArtifactContract,
) -> dict[str, Any]:
    path = _require_direct_commitment(
        label_root=label_root,
        path=label_root / "center_state_checkpoint.json",
        filename="center_state_checkpoint.json",
        expected_sha256=progress.center_state_checkpoint_byte_sha256,
        context="Center-state capability",
    )
    payload = _strict_json_object(
        path.read_bytes(),
        filename=path.name,
        compact=False,
    )
    beta = payload.get("center_beta")
    if (
        set(payload) != _CENTER_CHECKPOINT_KEYS
        or payload.get("protocol_id") != V021_PROTOCOL_ID
        or payload.get("config_sha256") != contract.config_byte_sha256
        or payload.get("state_kind") != "center_development"
        or not isinstance(beta, (int, float))
        or isinstance(beta, bool)
        or not pd.notna(beta)
        or not isinstance(payload.get("completeness_rule"), str)
        or not payload["completeness_rule"]
        or not isinstance(payload.get("created_utc"), str)
    ):
        raise V021FirewallError("center_state_checkpoint.json is invalid")
    _require_sha256(
        payload.get("center_state_sha256"),
        context="center state hash",
    )
    _positive_int(
        payload.get("development_cluster_count"),
        context="center development_cluster_count",
    )
    _positive_int(
        payload.get("forecast_horizon_count"),
        context="center forecast_horizon_count",
    )
    ridge = payload.get("ridge_penalty")
    if (
        not isinstance(ridge, (int, float))
        or isinstance(ridge, bool)
        or not pd.notna(ridge)
        or float(ridge) < 0.0
    ):
        raise V021FirewallError("center ridge_penalty is invalid")
    _verify_input_hashes(
        payload.get("input_byte_hashes"),
        expected_filenames=_CENTER_INPUT_FILENAMES,
        label_root=label_root,
        sealed_root=sealed_root,
        context="center checkpoint",
    )
    return payload


def _verify_training_manifest(
    *,
    label_root: Path,
    sealed_root: Path,
    contract: FrozenArtifactContract,
    center_payload: Mapping[str, Any],
    risk_payload: Mapping[str, Any],
) -> None:
    path = _direct_physical_child(
        label_root,
        label_root / "training_manifest.json",
        filename="training_manifest.json",
        context="Training manifest",
    )
    if _sha256_file(path) != risk_payload.get("training_manifest_byte_sha256"):
        raise V021FirewallError("Training manifest differs from the risk checkpoint")
    payload = _strict_json_object(
        path.read_bytes(),
        filename=path.name,
        compact=False,
    )
    try:
        expected_keys = contract.json_keys("training_manifest.json")
    except V015ArtifactError as exc:
        raise V021FirewallError(
            "V2.1 contract lacks the training-manifest schema"
        ) from exc
    if (
        set(payload) != expected_keys
        or payload.get("protocol_id") != V021_PROTOCOL_ID
        or payload.get("config_sha256") != contract.config_byte_sha256
        or payload.get("opened_truth_files")
        != ["center_development_truth.csv", "risk_development_truth.csv"]
        or payload.get("forbidden_v1_evidence_matches") != []
        or payload.get("center_state_sha256")
        != center_payload.get("center_state_sha256")
        or payload.get("risk_state_sha256") != risk_payload.get("risk_state_sha256")
        or payload.get("center_development_input_hashes")
        != center_payload.get("input_byte_hashes")
        or payload.get("risk_development_input_hashes")
        != risk_payload.get("input_byte_hashes")
        or not isinstance(payload.get("created_utc"), str)
    ):
        raise V021FirewallError("training_manifest.json semantics changed")


def _verify_risk_checkpoint(
    *,
    label_root: Path,
    sealed_root: Path,
    progress: AttemptProgress,
    contract: FrozenArtifactContract,
) -> None:
    center_payload = _verify_center_checkpoint(
        label_root=label_root,
        sealed_root=sealed_root,
        progress=progress,
        contract=contract,
    )
    path = _require_direct_commitment(
        label_root=label_root,
        path=label_root / "risk_state_checkpoint.json",
        filename="risk_state_checkpoint.json",
        expected_sha256=progress.risk_state_checkpoint_byte_sha256,
        context="Risk-state capability",
    )
    payload = _strict_json_object(
        path.read_bytes(),
        filename=path.name,
        compact=False,
    )
    if (
        set(payload) != _RISK_CHECKPOINT_KEYS
        or payload.get("protocol_id") != V021_PROTOCOL_ID
        or payload.get("config_sha256") != contract.config_byte_sha256
        or payload.get("state_kind") != "risk_development"
        or payload.get("center_checkpoint_byte_sha256")
        != progress.center_state_checkpoint_byte_sha256
        or not isinstance(payload.get("created_utc"), str)
    ):
        raise V021FirewallError("risk_state_checkpoint.json is invalid")
    _require_sha256(
        payload.get("training_manifest_byte_sha256"),
        context="training manifest hash",
    )
    _require_sha256(
        payload.get("risk_state_sha256"),
        context="risk state hash",
    )
    development = _positive_int(
        payload.get("development_cluster_count"),
        context="risk development_cluster_count",
    )
    eligible = _positive_int(
        payload.get("eligible_cluster_count"),
        context="risk eligible_cluster_count",
    )
    positive = _positive_int(
        payload.get("positive_label_count"),
        context="risk positive_label_count",
    )
    negative = _positive_int(
        payload.get("negative_label_count"),
        context="risk negative_label_count",
    )
    if eligible > development or positive + negative != eligible:
        raise V021FirewallError("Risk checkpoint counts are inconsistent")
    _verify_input_hashes(
        payload.get("input_byte_hashes"),
        expected_filenames=_RISK_INPUT_FILENAMES,
        label_root=label_root,
        sealed_root=sealed_root,
        context="risk checkpoint",
    )
    _verify_training_manifest(
        label_root=label_root,
        sealed_root=sealed_root,
        contract=contract,
        center_payload=center_payload,
        risk_payload=payload,
    )


def _verify_mask_commitment(
    *,
    label_root: Path,
    progress: AttemptProgress,
    path: str | Path | None,
    context: str = "Calibration mask capability",
) -> None:
    committed = _require_direct_commitment(
        label_root=label_root,
        path=path,
        filename="calibration_mask_commitment.json",
        expected_sha256=progress.calibration_mask_commitment_byte_sha256,
        context=context,
    )
    try:
        from lifetwin.experiments.calendar_long_horizon_v016_state import (
            deserialize_calibration_mask_commitment_json_v021,
        )

        deserialize_calibration_mask_commitment_json_v021(committed.read_bytes())
    except (TypeError, ValueError) as exc:
        raise V021FirewallError(
            "Calibration mask failed its canonical semantic decoder"
        ) from exc


def _verify_reveal_prerequisites(
    *,
    phase: str,
    label_root: Path,
    sealed_root: Path,
    progress: AttemptProgress,
    contract: FrozenArtifactContract,
    formal: bool,
    calibration_mask_commitment_path: str | Path | None,
) -> None:
    if phase in {"calibration_truth_opened", "scoring_truth_opened"}:
        _verify_mask_commitment(
            label_root=label_root,
            progress=progress,
            path=calibration_mask_commitment_path,
        )
    _verify_generation_plan(
        label_root=label_root,
        progress=progress,
        contract=contract,
    )
    _verify_actual_analysis_hash_ledger(
        label_root=label_root,
        progress=progress,
    )
    _verify_fit_commitment(
        label_root=label_root,
        progress=progress,
        contract=contract,
        formal=formal,
    )
    if phase in {
        "risk_truth_opened",
        "calibration_truth_opened",
        "scoring_truth_opened",
    }:
        _verify_center_checkpoint(
            label_root=label_root,
            sealed_root=sealed_root,
            progress=progress,
            contract=contract,
        )
    if phase in {"calibration_truth_opened", "scoring_truth_opened"}:
        _verify_risk_checkpoint(
            label_root=label_root,
            sealed_root=sealed_root,
            progress=progress,
            contract=contract,
        )


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
    calibration_mask_commitment_path: str | Path | None = None,
    prediction_commitment_path: str | Path | None = None,
    formal: bool = True,
) -> Mapping[str, pd.DataFrame]:
    """Open only truth authorized by one V2.1 reveal phase.

    Calibration access additionally requires the immutable, outcome-free mask
    commitment.  Scoring access requires both that mask commitment and the
    prediction commitment.  The exposure checkpoint is appended before any
    sealed file is read.
    """

    if phase not in _OPEN_PHASE_FILES:
        raise V021FirewallError("This phase has no truth-access capability")
    if formal:
        verify_formal_attempt_environment(identity)
    physical_label = _physical_root(label_free_root, context="Label-free")
    physical_sealed = _physical_root(sealed_truth_root, context="Sealed-truth")
    try:
        label_root, sealed_root = assert_separate_truth_roots(
            physical_label, physical_sealed
        )
    except V015ArtifactError as exc:
        raise V021FirewallError(str(exc)) from exc
    ledger = _require_ledger_path(label_root, ledger_path)
    commitment = _direct_physical_child(
        label_root,
        commitment_path,
        filename="truth_commitments.json",
        context="Truth commitment",
    )
    states = validate_formal_exposure_log(ledger, contract)
    try:
        prior = states[identity.attempt_id]
    except KeyError as exc:
        raise V021FirewallError(
            "Attempt has not checkpointed before generation"
        ) from exc
    if prior.identity != identity:
        raise V021FirewallError("Truth capability identity differs from the ledger")
    truth_hash = prior.truth_commitments_byte_sha256
    if truth_hash is None:
        raise V021FirewallError(
            "Truth capability lacks the checkpointed truth commitment"
        )

    if (
        phase not in {"calibration_truth_opened", "scoring_truth_opened"}
        and calibration_mask_commitment_path is not None
    ):
        raise V021FirewallError(
            "Pre-calibration reveal cannot receive a mask commitment path"
        )
    _verify_reveal_prerequisites(
        phase=phase,
        label_root=label_root,
        sealed_root=sealed_root,
        progress=prior,
        contract=contract,
        formal=formal,
        calibration_mask_commitment_path=calibration_mask_commitment_path,
    )

    prediction_hash: str | None = None
    if phase == "scoring_truth_opened":
        prediction_path = _require_direct_commitment(
            label_root=label_root,
            path=prediction_commitment_path,
            filename="prediction_commitment.json",
            expected_sha256=prior.prediction_commitment_byte_sha256,
            context="Scoring capability",
        )
        prediction_hash = prior.prediction_commitment_byte_sha256
        try:
            verify_prediction_commitment(
                commitment_path=prediction_path,
                label_free_root=label_root,
                contract=contract,
                formal=formal,
            )
        except V015ArtifactError as exc:
            raise V021FirewallError(
                "Scoring capability failed prediction commitment verification"
            ) from exc
    elif prediction_commitment_path is not None:
        raise V021FirewallError(
            "Development reveal cannot receive a prediction commitment path"
        )

    if _sha256_file(commitment) != truth_hash:
        raise V021FirewallError("Truth capability does not match its ledger commitment")
    try:
        payload = read_truth_commitments(commitment, contract, formal=formal)
    except V015ArtifactError as exc:
        raise V021FirewallError(
            "Truth commitment failed its canonical semantic decoder"
        ) from exc

    append_formal_exposure_event(
        path=ledger,
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
        entries = {str(item["path"]): item for item in payload["files"]}
        opened: dict[str, pd.DataFrame] = {}
        for filename in _OPEN_PHASE_FILES[phase]:
            entry = entries[filename]
            path = (sealed_root / filename).resolve()
            if path.parent != sealed_root:
                raise V021FirewallError("Sealed truth path escaped its root")
            frame = read_canonical_csv(path, contract, formal=formal)
            raw = path.read_bytes()
            if (
                len(frame) != int(entry["row_count"])
                or len(raw) != int(entry["byte_count"])
                or hashlib.sha256(raw).hexdigest() != entry["byte_sha256"]
            ):
                raise V021FirewallError(
                    f"{filename} differs from its pre-reveal commitment"
                )
            opened[filename] = frame
    except KeyboardInterrupt as exc:
        append_phase_error_without_masking(
            error=exc,
            ledger_path=ledger,
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
            ledger_path=ledger,
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
        path=ledger,
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
    calibration_mask_commitment_path: str | Path | None = None,
    formal: bool = True,
) -> Mapping[str, pd.DataFrame]:
    """Reread only truth already exposed for a pending deterministic phase."""

    requested = tuple(filenames)
    if (
        not requested
        or any(not isinstance(name, str) for name in requested)
        or len(set(requested)) != len(requested)
        or any(name not in contract.sealed_filenames for name in requested)
    ):
        raise V021FirewallError("Recovery truth-file request is invalid")
    if formal:
        verify_formal_attempt_environment(identity)
    physical_label = _physical_root(label_free_root, context="Label-free")
    physical_sealed = _physical_root(sealed_truth_root, context="Sealed-truth")
    try:
        label_root, sealed_root = assert_separate_truth_roots(
            physical_label, physical_sealed
        )
    except V015ArtifactError as exc:
        raise V021FirewallError(str(exc)) from exc
    ledger = _require_ledger_path(label_root, ledger_path)
    commitment = _direct_physical_child(
        label_root,
        commitment_path,
        filename="truth_commitments.json",
        context="Truth commitment",
    )
    states = validate_formal_exposure_log(ledger, contract)
    try:
        progress = states[identity.attempt_id]
    except KeyError as exc:
        raise V021FirewallError("Recovery attempt is absent from the ledger") from exc
    if progress.identity != identity or progress.terminal_failed:
        raise V021FirewallError(
            "Recovery identity is invalid or the attempt is terminal"
        )
    if progress.pending_phase not in _RECOVERABLE_TRUTH_PHASES:
        raise V021FirewallError(
            "Truth recovery requires a pending deterministic state/score phase"
        )
    if not set(requested).issubset(progress.opened_truth_files):
        raise V021FirewallError("Recovery requested truth that was not already exposed")
    if "calibration_truth.csv" in requested:
        _verify_mask_commitment(
            label_root=label_root,
            progress=progress,
            path=calibration_mask_commitment_path,
            context="Calibration mask recovery capability",
        )
    elif calibration_mask_commitment_path is not None:
        raise V021FirewallError(
            "Non-calibration recovery cannot receive a mask commitment path"
        )
    _verify_generation_plan(
        label_root=label_root,
        progress=progress,
        contract=contract,
    )
    _verify_actual_analysis_hash_ledger(
        label_root=label_root,
        progress=progress,
    )
    _verify_fit_commitment(
        label_root=label_root,
        progress=progress,
        contract=contract,
        formal=formal,
    )
    if any(
        filename in requested
        for filename in ("risk_development_truth.csv", "calibration_truth.csv")
    ):
        _verify_center_checkpoint(
            label_root=label_root,
            sealed_root=sealed_root,
            progress=progress,
            contract=contract,
        )
    if "calibration_truth.csv" in requested:
        _verify_risk_checkpoint(
            label_root=label_root,
            sealed_root=sealed_root,
            progress=progress,
            contract=contract,
        )
    truth_hash = progress.truth_commitments_byte_sha256
    if truth_hash is None or _sha256_file(commitment) != truth_hash:
        raise V021FirewallError(
            "Recovery truth commitment differs from the attempt ledger"
        )

    payload = read_truth_commitments(commitment, contract, formal=formal)
    entries = {str(item["path"]): item for item in payload["files"]}
    opened: dict[str, pd.DataFrame] = {}
    for filename in requested:
        entry = entries[filename]
        path = (sealed_root / filename).resolve()
        if path.parent != sealed_root:
            raise V021FirewallError("Recovery truth path escaped its root")
        frame = read_canonical_csv(path, contract, formal=formal)
        raw = path.read_bytes()
        if (
            len(frame) != int(entry["row_count"])
            or len(raw) != int(entry["byte_count"])
            or hashlib.sha256(raw).hexdigest() != entry["byte_sha256"]
        ):
            raise V021FirewallError(f"{filename} differs from its recovery commitment")
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
        field = PHASE_COMMITMENT_FIELDS[phase]
    except KeyError as exc:
        raise V021FirewallError(
            f"{phase!r} is not a phase-specific commitment"
        ) from exc
    expected = getattr(progress, field)
    if expected is None:
        raise V021FirewallError(f"{phase} has no completed commitment")
    observed = _sha256_file(artifact_path)
    if observed != expected:
        raise V021FirewallError(f"{phase} artifact differs from its ledger commitment")
    return observed


__all__ = [
    "AttemptProgress",
    "EXIT_STATUSES",
    "FormalAttemptIdentity",
    "PHASES",
    "V021FirewallError",
    "append_formal_exposure_event",
    "append_phase_error_without_masking",
    "open_truth_for_phase",
    "phase_commitment_message",
    "reopen_authorized_truth_for_recovery",
    "validate_formal_exposure_events",
    "validate_formal_exposure_log",
    "verify_formal_attempt_environment",
    "verify_phase_artifact_commitment",
]
