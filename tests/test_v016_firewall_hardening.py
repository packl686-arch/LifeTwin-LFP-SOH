from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_firewall import (
    AttemptProgress,
    FormalAttemptIdentity,
    V021FirewallError,
    open_truth_for_phase,
    validate_formal_exposure_log,
)
from lifetwin.experiments.calendar_long_horizon_v016_ledger import (
    canonical_json_line_bytes,
)


_ATTEMPT = "strict-ledger"
_GIT = "1" * 40


def _contract():
    return load_v021_contract_view().artifacts


def _before_generation_event() -> dict[str, object]:
    contract = _contract()
    return {
        "attempt_id": _ATTEMPT,
        "created_utc": "2026-07-26T00:00:00+00:00",
        "git_commit": _GIT,
        "git_dirty": False,
        "config_byte_sha256": contract.config_byte_sha256,
        "phase": "before_generation",
        "truth_commitments_byte_sha256": None,
        "prediction_commitment_byte_sha256": None,
        "opened_truth_files": [],
        "exit_status": "completed",
        "message": "before generation",
    }


@pytest.mark.parametrize("mutation", ["duplicate", "noncanonical", "missing", "extra"])
def test_exposure_log_has_one_strict_canonical_dialect(
    tmp_path: Path,
    mutation: str,
) -> None:
    event = _before_generation_event()
    canonical = canonical_json_line_bytes(event)
    if mutation == "duplicate":
        raw = b'{"attempt_id":"strict-ledger",' + canonical.removeprefix(b"{")
    elif mutation == "noncanonical":
        raw = canonical.replace(b":", b": ", 1)
    elif mutation == "missing":
        event.pop("message")
        raw = canonical_json_line_bytes(event)
    else:
        event["unexpected"] = None
        raw = canonical_json_line_bytes(event)
    path = tmp_path / "exposure_log.jsonl"
    path.write_bytes(raw)

    with pytest.raises(V021FirewallError):
        validate_formal_exposure_log(path, _contract())


def _progress(*, mask_hash: str | None = None) -> AttemptProgress:
    contract = _contract()
    identity = FormalAttemptIdentity(
        attempt_id=_ATTEMPT,
        git_commit=_GIT,
        config_byte_sha256=contract.config_byte_sha256,
    )
    return AttemptProgress(
        identity=identity,
        completed_phase=(
            "calibration_mask_committed"
            if mask_hash is not None
            else "label_free_fit_committed"
        ),
        pending_phase=None,
        truth_commitments_byte_sha256="2" * 64,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(),
        terminal_failed=False,
        generation_plan_commitment_byte_sha256="9" * 64,
        fit_commitment_byte_sha256="a" * 64,
        calibration_mask_commitment_byte_sha256=mask_hash,
    )


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    label = tmp_path / "label"
    sealed = tmp_path / "sealed"
    label.mkdir()
    sealed.mkdir()
    (label / "truth_commitments.json").write_bytes(b"truth")
    return label, sealed


def test_truth_capability_rejects_ledger_outside_label_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label, sealed = _roots(tmp_path)
    progress = _progress()
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v016_firewall."
        "validate_formal_exposure_log",
        lambda *_: {_ATTEMPT: progress},
    )
    with pytest.raises(V021FirewallError, match="direct label-free"):
        open_truth_for_phase(
            ledger_path=tmp_path / "elsewhere.jsonl",
            identity=progress.identity,
            contract=_contract(),
            commitment_path=label / "truth_commitments.json",
            sealed_truth_root=sealed,
            label_free_root=label,
            phase="center_truth_opened",
            created_utc="2026-07-26T00:01:00+00:00",
            formal=False,
        )


def test_truth_capability_does_not_trust_opaque_generation_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label, sealed = _roots(tmp_path)
    progress = _progress()
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v016_firewall."
        "validate_formal_exposure_log",
        lambda *_: {_ATTEMPT: progress},
    )
    with pytest.raises(V021FirewallError, match="Generation-plan capability"):
        open_truth_for_phase(
            ledger_path=label / "exposure_log.jsonl",
            identity=progress.identity,
            contract=_contract(),
            commitment_path=label / "truth_commitments.json",
            sealed_truth_root=sealed,
            label_free_root=label,
            phase="center_truth_opened",
            created_utc="2026-07-26T00:01:00+00:00",
            formal=False,
        )


def test_empty_object_cannot_masquerade_as_calibration_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label, sealed = _roots(tmp_path)
    raw = b"{}\n"
    mask = label / "calibration_mask_commitment.json"
    mask.write_bytes(raw)
    progress = _progress(mask_hash=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        "lifetwin.experiments.calendar_long_horizon_v016_firewall."
        "validate_formal_exposure_log",
        lambda *_: {_ATTEMPT: progress},
    )
    with pytest.raises(V021FirewallError, match="semantic decoder"):
        open_truth_for_phase(
            ledger_path=label / "exposure_log.jsonl",
            identity=progress.identity,
            contract=_contract(),
            commitment_path=label / "truth_commitments.json",
            sealed_truth_root=sealed,
            label_free_root=label,
            phase="calibration_truth_opened",
            created_utc="2026-07-26T00:01:00+00:00",
            calibration_mask_commitment_path=mask,
            formal=False,
        )
