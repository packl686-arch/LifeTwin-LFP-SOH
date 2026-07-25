from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts import verify_v015_terminal_evidence as verifier


def test_public_terminal_evidence_verifies() -> None:
    result = verifier.verify_terminal_evidence(verifier.DEFAULT_EVIDENCE_DIR)
    assert result["status"] == "passed"
    assert result["protocol_scientific_status"] == "inconclusive_not_success"
    assert result["endpoint_evaluation_status"] == "not_computed"
    assert result["prediction_commitment_present"] is False
    assert result["last_completed_phase"] == "calibration_truth_opened"


def test_tampered_commitment_copy_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verifier._sha256

    def tampered_hash(path: object) -> str:
        if getattr(path, "name", "") == "training_manifest.json":
            return "0" * 64
        return original(path)  # type: ignore[arg-type]

    monkeypatch.setattr(verifier, "_sha256", tampered_hash)
    with pytest.raises(verifier.TerminalEvidenceError, match="Hash mismatch"):
        verifier.verify_terminal_evidence(verifier.DEFAULT_EVIDENCE_DIR)


def test_false_scored_result_claim_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verifier._json_object

    def false_claim(path: object) -> object:
        payload = original(path)  # type: ignore[arg-type]
        if getattr(path, "name", "") != verifier.MANIFEST_FILENAME:
            return payload
        modified = deepcopy(payload)
        modified["endpoint_evaluation_status"] = "computed"
        return modified

    monkeypatch.setattr(verifier, "_json_object", false_claim)
    with pytest.raises(verifier.TerminalEvidenceError, match="not_computed"):
        verifier.verify_terminal_evidence(verifier.DEFAULT_EVIDENCE_DIR)


def test_false_truth_exposure_boundary_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verifier._json_object

    def false_boundary(path: object) -> object:
        payload = original(path)  # type: ignore[arg-type]
        if getattr(path, "name", "") != verifier.MANIFEST_FILENAME:
            return payload
        modified = json.loads(json.dumps(payload))
        modified["truth_exposure"]["opened_truth_files"].append("test_truth.csv")
        return modified

    monkeypatch.setattr(verifier, "_json_object", false_boundary)
    with pytest.raises(verifier.TerminalEvidenceError, match="boundary"):
        verifier.verify_terminal_evidence(verifier.DEFAULT_EVIDENCE_DIR)


def test_tampered_ledger_is_rejected_before_state_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verifier._sha256

    def tampered_hash(path: object) -> str:
        if getattr(path, "name", "") == "exposure_log.jsonl":
            return "f" * 64
        return original(path)  # type: ignore[arg-type]

    monkeypatch.setattr(verifier, "_sha256", tampered_hash)
    with pytest.raises(verifier.TerminalEvidenceError, match="Ledger hash mismatch"):
        verifier.verify_terminal_evidence(verifier.DEFAULT_EVIDENCE_DIR)
