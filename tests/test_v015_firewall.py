from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from lifetwin.experiments.calendar_long_horizon_v015_firewall import (
    AttemptProgress,
    FormalAttemptIdentity,
    PHASES,
    V015FirewallError,
    phase_commitment_message,
    reopen_authorized_truth_for_recovery,
    validate_formal_exposure_events,
    verify_phase_artifact_commitment,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    load_artifact_contract,
)


GIT_COMMIT = "1" * 40
TRUTH_HASH = "2" * 64
PREDICTION_HASH = "3" * 64
ATTEMPT_ID = "hand-fixture-attempt"
FIT_HASH = "a" * 64
CENTER_STATE_HASH = "b" * 64
RISK_STATE_HASH = "c" * 64
MODEL_STATE_HASH = "d" * 64
PHASE_MESSAGES = {
    "label_free_fit_committed": phase_commitment_message(
        "label_free_fit_committed", FIT_HASH
    ),
    "center_state_committed": phase_commitment_message(
        "center_state_committed", CENTER_STATE_HASH
    ),
    "risk_state_committed": phase_commitment_message(
        "risk_state_committed", RISK_STATE_HASH
    ),
    "model_state_committed": phase_commitment_message(
        "model_state_committed", MODEL_STATE_HASH
    ),
}

CENTER = ("center_development_truth.csv",)
RISK = CENTER + ("risk_development_truth.csv",)
CALIBRATION = RISK + ("calibration_truth.csv",)
ALL_TRUTH = CALIBRATION + (
    "test_truth.csv",
    "audit_truth.csv",
    "intrinsic_matched_truth.csv",
    "stress_plan_matched_truth.csv",
    "intrinsic_matched_pairs.csv",
    "stress_plan_matched_pairs.csv",
)
OPENED = {
    "before_generation": (),
    "truth_committed": (),
    "label_free_fit_committed": (),
    "center_truth_opened": CENTER,
    "center_state_committed": CENTER,
    "risk_truth_opened": RISK,
    "risk_state_committed": RISK,
    "calibration_truth_opened": CALIBRATION,
    "model_state_committed": CALIBRATION,
    "prediction_started": CALIBRATION,
    "prediction_committed": CALIBRATION,
    "scoring_truth_opened": ALL_TRUTH,
    "scoring_completed": ALL_TRUTH,
}


def _event(phase: str, index: int, *, status: str = "completed") -> dict[str, object]:
    contract = load_artifact_contract()
    phase_index = PHASES.index(phase)
    return {
        "attempt_id": ATTEMPT_ID,
        "created_utc": f"2026-07-23T00:{index:02d}:00+00:00",
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
        "config_byte_sha256": contract.config_byte_sha256,
        "phase": phase,
        "truth_commitments_byte_sha256": (
            None if phase_index < PHASES.index("truth_committed") else TRUTH_HASH
        ),
        "prediction_commitment_byte_sha256": (
            None
            if phase_index < PHASES.index("prediction_committed")
            else PREDICTION_HASH
        ),
        "opened_truth_files": list(sorted(OPENED[phase])),
        "exit_status": status,
        "message": PHASE_MESSAGES.get(phase, "hand fixture"),
    }


def _complete_attempt() -> list[dict[str, object]]:
    return [_event(phase, index) for index, phase in enumerate(PHASES)]


def test_formal_identity_requires_full_clean_commit_material() -> None:
    contract = load_artifact_contract()
    identity = FormalAttemptIdentity(
        ATTEMPT_ID, GIT_COMMIT, contract.config_byte_sha256
    )
    assert identity.git_commit == GIT_COMMIT
    with pytest.raises(V015FirewallError, match="full lowercase hash"):
        FormalAttemptIdentity(ATTEMPT_ID, "abc", contract.config_byte_sha256)
    with pytest.raises(V015FirewallError, match="safe stable"):
        FormalAttemptIdentity("../escape", GIT_COMMIT, contract.config_byte_sha256)


def test_complete_frozen_phase_sequence_is_accepted() -> None:
    contract = load_artifact_contract()
    states = validate_formal_exposure_events(_complete_attempt(), contract=contract)
    state = states[ATTEMPT_ID]
    assert state.completed_phase == "scoring_completed"
    assert state.pending_phase is None
    assert state.truth_commitments_byte_sha256 == TRUTH_HASH
    assert state.prediction_commitment_byte_sha256 == PREDICTION_HASH
    assert state.opened_truth_files == tuple(sorted(ALL_TRUTH))
    assert state.fit_commitment_byte_sha256 == FIT_HASH
    assert state.center_state_checkpoint_byte_sha256 == CENTER_STATE_HASH
    assert state.risk_state_checkpoint_byte_sha256 == RISK_STATE_HASH
    assert state.model_state_commitment_byte_sha256 == MODEL_STATE_HASH


def test_phase_cannot_be_skipped() -> None:
    contract = load_artifact_contract()
    events = _complete_attempt()
    del events[3]
    with pytest.raises(V015FirewallError, match="Illegal phase transition"):
        validate_formal_exposure_events(events, contract=contract)


def test_started_and_interrupted_phase_must_resume_same_checkpoint() -> None:
    contract = load_artifact_contract()
    events = [
        _event("before_generation", 0),
        _event("truth_committed", 1, status="started"),
        _event("truth_committed", 2, status="interrupted"),
        _event("truth_committed", 3),
    ]
    state = validate_formal_exposure_events(events, contract=contract)[ATTEMPT_ID]
    assert state.completed_phase == "truth_committed"
    assert state.pending_phase is None

    broken = events[:2] + [_event("center_truth_opened", 2)]
    with pytest.raises(V015FirewallError, match="Illegal phase transition"):
        validate_formal_exposure_events(broken, contract=contract)


def test_long_label_free_fit_has_a_resumable_logged_phase() -> None:
    contract = load_artifact_contract()
    events = [
        _event("before_generation", 0),
        _event("truth_committed", 1),
        _event("label_free_fit_committed", 2, status="started"),
        _event("label_free_fit_committed", 3, status="interrupted"),
    ]
    state = validate_formal_exposure_events(events, contract=contract)[ATTEMPT_ID]
    assert state.completed_phase == "truth_committed"
    assert state.pending_phase == "label_free_fit_committed"
    assert state.opened_truth_files == ()

    events.append(_event("label_free_fit_committed", 4))
    resumed = validate_formal_exposure_events(events, contract=contract)[ATTEMPT_ID]
    assert resumed.completed_phase == "label_free_fit_committed"
    assert resumed.pending_phase is None


def test_recovery_cannot_expand_an_existing_truth_capability(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_artifact_contract()
    identity = FormalAttemptIdentity(
        ATTEMPT_ID, GIT_COMMIT, contract.config_byte_sha256
    )
    progress = AttemptProgress(
        identity=identity,
        completed_phase="center_truth_opened",
        pending_phase="center_state_committed",
        truth_commitments_byte_sha256=TRUTH_HASH,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=CENTER,
        terminal_failed=False,
    )
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v015_firewall."
        "validate_formal_exposure_log",
        lambda *_: {ATTEMPT_ID: progress},
    )
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v015_firewall."
        "verify_formal_attempt_environment",
        lambda *_: None,
    )
    label = tmp_path / "label"
    sealed = tmp_path / "sealed"
    label.mkdir()
    sealed.mkdir()
    with pytest.raises(V015FirewallError, match="not already exposed"):
        reopen_authorized_truth_for_recovery(
            ledger_path=label / "exposure_log.jsonl",
            identity=identity,
            contract=contract,
            commitment_path=label / "truth_commitments.json",
            sealed_truth_root=sealed,
            label_free_root=label,
            filenames=("risk_development_truth.csv",),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("git_commit", "4" * 40, "implementation identity"),
        ("git_dirty", True, "implementation identity"),
        ("truth_commitments_byte_sha256", "5" * 64, "Truth commitment changed"),
    ],
)
def test_attempt_identity_and_truth_commitment_are_immutable(
    field: str, replacement: object, message: str
) -> None:
    contract = load_artifact_contract()
    events = _complete_attempt()[:5]
    events[-1][field] = replacement
    with pytest.raises(V015FirewallError, match=message):
        validate_formal_exposure_events(events, contract=contract)


def test_prediction_commitment_cannot_appear_early_or_change() -> None:
    contract = load_artifact_contract()
    early = _complete_attempt()[:4]
    early[-1]["prediction_commitment_byte_sha256"] = PREDICTION_HASH
    with pytest.raises(V015FirewallError, match="appeared before"):
        validate_formal_exposure_events(early, contract=contract)

    changed = _complete_attempt()
    changed[-1]["prediction_commitment_byte_sha256"] = "6" * 64
    with pytest.raises(V015FirewallError, match="Prediction commitment changed"):
        validate_formal_exposure_events(changed, contract=contract)


def test_opened_truth_capability_is_exact_for_each_phase() -> None:
    contract = load_artifact_contract()
    events = _complete_attempt()
    events[3]["opened_truth_files"] = []
    with pytest.raises(V015FirewallError, match="phase capability"):
        validate_formal_exposure_events(events, contract=contract)

    events = _complete_attempt()
    events[8]["opened_truth_files"] = list(sorted(RISK))
    with pytest.raises(V015FirewallError, match="phase capability"):
        validate_formal_exposure_events(events, contract=contract)


def test_failed_attempt_is_terminal() -> None:
    contract = load_artifact_contract()
    events = _complete_attempt()[:3]
    failed = deepcopy(_event("center_truth_opened", 3, status="failed"))
    events.extend((failed, _event("center_truth_opened", 4)))
    with pytest.raises(V015FirewallError, match="later events"):
        validate_formal_exposure_events(events, contract=contract)


def test_checkpoint_completed_message_must_be_exact_and_machine_readable() -> None:
    contract = load_artifact_contract()
    events = _complete_attempt()[:3]
    events[-1]["message"] = f"fit complete; hash={FIT_HASH}"
    with pytest.raises(V015FirewallError, match="machine-readable commitment"):
        validate_formal_exposure_events(events, contract=contract)


def test_phase_artifact_commitment_rejects_changed_bytes(tmp_path: Path) -> None:
    contract = load_artifact_contract()
    progress = validate_formal_exposure_events(_complete_attempt(), contract=contract)[
        ATTEMPT_ID
    ]
    artifact = tmp_path / "fit_commitment.json"
    artifact.write_bytes(b"committed bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    progress = replace(
        progress,
        fit_commitment_byte_sha256=digest,
    )
    assert (
        verify_phase_artifact_commitment(
            progress,
            phase="label_free_fit_committed",
            artifact_path=artifact,
        )
        == digest
    )
    artifact.write_bytes(b"changed bytes")
    with pytest.raises(V015FirewallError, match="differs"):
        verify_phase_artifact_commitment(
            progress,
            phase="label_free_fit_committed",
            artifact_path=artifact,
        )
