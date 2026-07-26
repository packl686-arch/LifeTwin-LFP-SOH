from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_firewall import (
    AttemptProgress,
    FormalAttemptIdentity,
    PHASES,
    V021FirewallError,
    open_truth_for_phase,
    phase_commitment_message,
    reopen_authorized_truth_for_recovery,
    validate_formal_exposure_events,
    verify_phase_artifact_commitment,
)


GIT_COMMIT = "1" * 40
TRUTH_HASH = "2" * 64
PREDICTION_HASH = "3" * 64
ATTEMPT_ID = "v021-hand-fixture"
GENERATION_PLAN_HASH = "9" * 64
ACTUAL_ANALYSIS_HASH = "8" * 64
FIT_HASH = "a" * 64
CENTER_STATE_HASH = "b" * 64
RISK_STATE_HASH = "c" * 64
MASK_HASH = "d" * 64
MODEL_STATE_HASH = "e" * 64
PHASE_HASHES = {
    "generation_plan_committed": GENERATION_PLAN_HASH,
    "actual_analysis_hash_ledger_committed": ACTUAL_ANALYSIS_HASH,
    "label_free_fit_committed": FIT_HASH,
    "center_state_committed": CENTER_STATE_HASH,
    "risk_state_committed": RISK_STATE_HASH,
    "calibration_mask_committed": MASK_HASH,
    "model_state_committed": MODEL_STATE_HASH,
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
    "generation_plan_committed": (),
    "truth_committed": (),
    "actual_analysis_hash_ledger_committed": (),
    "label_free_fit_committed": (),
    "center_truth_opened": CENTER,
    "center_state_committed": CENTER,
    "risk_truth_opened": RISK,
    "risk_state_committed": RISK,
    "calibration_mask_committed": RISK,
    "calibration_truth_opened": CALIBRATION,
    "model_state_committed": CALIBRATION,
    "prediction_started": CALIBRATION,
    "prediction_committed": CALIBRATION,
    "scoring_truth_opened": ALL_TRUTH,
    "scoring_completed": ALL_TRUTH,
}


def _contract():
    return load_v021_contract_view().artifacts


def _event(
    phase: str,
    index: int,
    *,
    status: str = "completed",
) -> dict[str, object]:
    phase_index = PHASES.index(phase)
    contract = _contract()
    message = (
        phase_commitment_message(phase, PHASE_HASHES[phase])
        if phase in PHASE_HASHES
        else "hand fixture"
    )
    return {
        "attempt_id": ATTEMPT_ID,
        "created_utc": f"2026-07-26T00:{index:02d}:00+00:00",
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
        "message": message,
    }


def _complete_attempt() -> list[dict[str, object]]:
    return [_event(phase, index) for index, phase in enumerate(PHASES)]


def test_complete_v021_phase_sequence_binds_both_new_commitments() -> None:
    state = validate_formal_exposure_events(_complete_attempt(), contract=_contract())[
        ATTEMPT_ID
    ]
    assert state.completed_phase == "scoring_completed"
    assert state.generation_plan_commitment_byte_sha256 == GENERATION_PLAN_HASH
    assert state.calibration_mask_commitment_byte_sha256 == MASK_HASH
    assert state.fit_commitment_byte_sha256 == FIT_HASH
    assert state.prediction_commitment_byte_sha256 == PREDICTION_HASH
    assert state.opened_truth_files == tuple(sorted(ALL_TRUTH))


def test_generation_plan_and_mask_phases_cannot_be_skipped() -> None:
    missing_plan = _complete_attempt()
    del missing_plan[1]
    with pytest.raises(V021FirewallError, match="Illegal phase transition"):
        validate_formal_exposure_events(missing_plan, contract=_contract())

    missing_mask = _complete_attempt()
    del missing_mask[8]
    with pytest.raises(V021FirewallError, match="Illegal phase transition"):
        validate_formal_exposure_events(missing_mask, contract=_contract())


@pytest.mark.parametrize(
    "phase",
    [
        "generation_plan_committed",
        "label_free_fit_committed",
        "center_state_committed",
        "risk_state_committed",
        "calibration_mask_committed",
        "model_state_committed",
    ],
)
def test_all_commitment_messages_are_exact(phase: str) -> None:
    events = _complete_attempt()[: PHASES.index(phase) + 1]
    events[-1]["message"] = f"free text {PHASE_HASHES[phase]}"
    with pytest.raises(V021FirewallError, match="machine-readable commitment"):
        validate_formal_exposure_events(events, contract=_contract())


def test_calibration_truth_cannot_be_declared_before_mask_commitment() -> None:
    events = _complete_attempt()[:8]
    forged = _event("calibration_truth_opened", 8)
    events.append(forged)
    with pytest.raises(V021FirewallError, match="expected"):
        validate_formal_exposure_events(events, contract=_contract())


def test_mask_phase_does_not_open_calibration_truth() -> None:
    events = _complete_attempt()[:9]
    events[-1]["opened_truth_files"] = list(sorted(CALIBRATION))
    with pytest.raises(V021FirewallError, match="phase capability"):
        validate_formal_exposure_events(events, contract=_contract())


def test_truth_and_prediction_commitments_cannot_appear_early() -> None:
    truth_early = _complete_attempt()[:2]
    truth_early[-1]["truth_commitments_byte_sha256"] = TRUTH_HASH
    with pytest.raises(V021FirewallError, match="before generation"):
        validate_formal_exposure_events(truth_early, contract=_contract())

    prediction_early = _complete_attempt()[:12]
    prediction_early[-1]["prediction_commitment_byte_sha256"] = PREDICTION_HASH
    with pytest.raises(V021FirewallError, match="appeared before"):
        validate_formal_exposure_events(prediction_early, contract=_contract())


def test_started_or_interrupted_phase_must_resume_in_place() -> None:
    events = [
        _event("before_generation", 0),
        _event("generation_plan_committed", 1, status="started"),
        _event("generation_plan_committed", 2, status="interrupted"),
        _event("generation_plan_committed", 3),
    ]
    state = validate_formal_exposure_events(events, contract=_contract())[ATTEMPT_ID]
    assert state.completed_phase == "generation_plan_committed"
    assert state.pending_phase is None

    broken = events[:2] + [_event("truth_committed", 2)]
    with pytest.raises(V021FirewallError, match="Illegal phase transition"):
        validate_formal_exposure_events(broken, contract=_contract())


def test_failed_attempt_is_terminal() -> None:
    phase = "calibration_mask_committed"
    phase_index = PHASES.index(phase)
    events = _complete_attempt()[:phase_index]
    events.append(_event(phase, phase_index, status="failed"))
    events.append(_event(phase, phase_index + 1))
    with pytest.raises(V021FirewallError, match="later events"):
        validate_formal_exposure_events(events, contract=_contract())


def test_v2_contract_is_rejected() -> None:
    from lifetwin.experiments.calendar_long_horizon_v015_io import (
        load_artifact_contract,
    )

    event = _complete_attempt()[0]
    v2 = load_artifact_contract()
    event["config_byte_sha256"] = v2.config_byte_sha256
    with pytest.raises(V021FirewallError, match="V2.1 artifact contract"):
        validate_formal_exposure_events([event], contract=v2)


def test_phase_artifact_verification_rejects_changed_mask_bytes(
    tmp_path: Path,
) -> None:
    state = validate_formal_exposure_events(_complete_attempt(), contract=_contract())[
        ATTEMPT_ID
    ]
    path = tmp_path / "calibration_mask_commitment.json"
    path.write_bytes(b"mask")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state = replace(
        state,
        calibration_mask_commitment_byte_sha256=digest,
    )
    assert (
        verify_phase_artifact_commitment(
            state,
            phase="calibration_mask_committed",
            artifact_path=path,
        )
        == digest
    )
    path.write_bytes(b"changed")
    with pytest.raises(V021FirewallError, match="differs"):
        verify_phase_artifact_commitment(
            state,
            phase="calibration_mask_committed",
            artifact_path=path,
        )


def test_calibration_capability_requires_exact_mask_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    identity = FormalAttemptIdentity(
        ATTEMPT_ID, GIT_COMMIT, contract.config_byte_sha256
    )
    progress = AttemptProgress(
        identity=identity,
        completed_phase="calibration_mask_committed",
        pending_phase=None,
        truth_commitments_byte_sha256=TRUTH_HASH,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=tuple(sorted(RISK)),
        terminal_failed=False,
        calibration_mask_commitment_byte_sha256=MASK_HASH,
    )
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v016_firewall."
        "validate_formal_exposure_log",
        lambda *_: {ATTEMPT_ID: progress},
    )
    label = tmp_path / "label"
    sealed = tmp_path / "sealed"
    label.mkdir()
    sealed.mkdir()
    (label / "truth_commitments.json").write_bytes(b"truth commitment")
    with pytest.raises(V021FirewallError, match="mask capability"):
        open_truth_for_phase(
            ledger_path=label / "exposure_log.jsonl",
            identity=identity,
            contract=contract,
            commitment_path=label / "truth_commitments.json",
            sealed_truth_root=sealed,
            label_free_root=label,
            phase="calibration_truth_opened",
            created_utc="2026-07-26T00:00:00+00:00",
            formal=False,
        )
    wrong = label / "wrong.json"
    wrong.write_bytes(b"mask")
    with pytest.raises(V021FirewallError, match="direct label-free"):
        open_truth_for_phase(
            ledger_path=label / "exposure_log.jsonl",
            identity=identity,
            contract=contract,
            commitment_path=label / "truth_commitments.json",
            sealed_truth_root=sealed,
            label_free_root=label,
            phase="calibration_truth_opened",
            created_utc="2026-07-26T00:00:00+00:00",
            calibration_mask_commitment_path=wrong,
            formal=False,
        )


def test_recovery_cannot_expand_truth_or_bypass_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    identity = FormalAttemptIdentity(
        ATTEMPT_ID, GIT_COMMIT, contract.config_byte_sha256
    )
    progress = AttemptProgress(
        identity=identity,
        completed_phase="calibration_truth_opened",
        pending_phase="model_state_committed",
        truth_commitments_byte_sha256=TRUTH_HASH,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=tuple(sorted(CALIBRATION)),
        terminal_failed=False,
        calibration_mask_commitment_byte_sha256=MASK_HASH,
    )
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v016_firewall."
        "validate_formal_exposure_log",
        lambda *_: {ATTEMPT_ID: progress},
    )
    label = tmp_path / "label"
    sealed = tmp_path / "sealed"
    label.mkdir()
    sealed.mkdir()
    truth = label / "truth_commitments.json"
    truth.write_bytes(b"truth")

    with pytest.raises(V021FirewallError, match="mask recovery capability"):
        reopen_authorized_truth_for_recovery(
            ledger_path=label / "exposure_log.jsonl",
            identity=identity,
            contract=contract,
            commitment_path=truth,
            sealed_truth_root=sealed,
            label_free_root=label,
            filenames=("calibration_truth.csv",),
            formal=False,
        )

    restricted = replace(
        progress,
        completed_phase="center_truth_opened",
        pending_phase="center_state_committed",
        opened_truth_files=CENTER,
    )
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v016_firewall."
        "validate_formal_exposure_log",
        lambda *_: {ATTEMPT_ID: restricted},
    )
    with pytest.raises(V021FirewallError, match="not already exposed"):
        reopen_authorized_truth_for_recovery(
            ledger_path=label / "exposure_log.jsonl",
            identity=identity,
            contract=contract,
            commitment_path=truth,
            sealed_truth_root=sealed,
            label_free_root=label,
            filenames=("risk_development_truth.csv",),
            formal=False,
        )


def test_identity_and_commitment_hashes_are_immutable() -> None:
    events = _complete_attempt()[:10]
    changed = deepcopy(events)
    changed[-1]["truth_commitments_byte_sha256"] = "f" * 64
    with pytest.raises(V021FirewallError, match="Truth commitment changed"):
        validate_formal_exposure_events(changed, contract=_contract())

    changed = deepcopy(events)
    changed[-1]["git_commit"] = "f" * 40
    with pytest.raises(V021FirewallError, match="implementation identity"):
        validate_formal_exposure_events(changed, contract=_contract())
