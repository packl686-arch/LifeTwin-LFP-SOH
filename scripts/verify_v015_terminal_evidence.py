#!/usr/bin/env python3
"""Verify the public evidence for the terminated V0.15 formal attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from lifetwin.experiments.calendar_long_horizon_v015_firewall import (
    validate_formal_exposure_log,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    DEFAULT_V2_CONFIG_PATH,
    load_artifact_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT
    / "showcase"
    / "evidence_v015"
    / "synthetic_long_horizon_identifiability_v2"
)
MANIFEST_FILENAME = "formal_attempt_termination_manifest.json"
EXPECTED_COMMITMENT_FILES = {
    "truth_commitments.json",
    "fit_commitment.json",
    "center_state_checkpoint.json",
    "training_manifest.json",
    "risk_state_checkpoint.json",
}
EXPECTED_OPENED_TRUTH = (
    "calibration_truth.csv",
    "center_development_truth.csv",
    "risk_development_truth.csv",
)


class TerminalEvidenceError(ValueError):
    """Raised when a public terminal-attempt record is internally inconsistent."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalEvidenceError(f"Cannot read canonical JSON: {path.name}") from exc
    if not isinstance(value, Mapping):
        raise TerminalEvidenceError(f"{path.name} must contain a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalEvidenceError(message)


def verify_terminal_evidence(evidence_dir: str | Path) -> dict[str, Any]:
    """Validate copied commitments, ledger state, and the termination summary."""

    root = Path(evidence_dir).resolve()
    manifest_path = root / MANIFEST_FILENAME
    manifest = _json_object(manifest_path)
    attempt_id = manifest.get("attempt_id")
    _require(
        isinstance(attempt_id, str) and bool(attempt_id),
        "Termination manifest lacks attempt_id",
    )
    _require(
        manifest.get("attempt_disposition") == "terminated_pre_prediction",
        "Attempt disposition changed",
    )
    _require(
        manifest.get("protocol_scientific_status") == "inconclusive_not_success",
        "Scientific status changed",
    )
    _require(
        manifest.get("endpoint_evaluation_status") == "not_computed",
        "Endpoint status must remain not_computed",
    )
    _require(
        manifest.get("complete_conforming_score_package") is False,
        "A terminated attempt cannot claim a complete score package",
    )

    exposure = manifest.get("truth_exposure")
    _require(isinstance(exposure, Mapping), "truth_exposure must be an object")
    _require(
        exposure.get("prediction_commitment_present") is False,
        "Prediction commitment must remain absent",
    )
    _require(
        exposure.get("evaluation_truth_opened") is False,
        "Held-out truth exposure claim changed",
    )
    _require(
        tuple(sorted(exposure.get("opened_truth_files", ()))) == EXPECTED_OPENED_TRUTH,
        "Opened truth-file boundary changed",
    )

    commitment_rows = manifest.get("commitment_chain")
    _require(isinstance(commitment_rows, list), "commitment_chain must be an array")
    observed_names: set[str] = set()
    verified_hashes: dict[str, str] = {}
    for row in commitment_rows:
        _require(isinstance(row, Mapping), "Commitment row must be an object")
        name = row.get("artifact")
        _require(
            isinstance(name, str) and name in EXPECTED_COMMITMENT_FILES,
            "Unknown commitment artifact",
        )
        _require(name not in observed_names, "Duplicate commitment artifact")
        path = root / name
        _require(path.is_file(), f"Missing commitment artifact: {name}")
        actual_hash = _sha256(path)
        actual_size = path.stat().st_size
        _require(actual_hash == row.get("byte_sha256"), f"Hash mismatch: {name}")
        _require(actual_size == row.get("byte_count"), f"Byte-count mismatch: {name}")
        _require(row.get("verified") is True, f"Unverified commitment row: {name}")
        observed_names.add(name)
        verified_hashes[name] = actual_hash
    _require(
        observed_names == EXPECTED_COMMITMENT_FILES,
        "Commitment artifact set changed",
    )

    ledger = manifest.get("ledger")
    _require(isinstance(ledger, Mapping), "ledger must be an object")
    ledger_path = root / str(ledger.get("artifact"))
    _require(ledger_path.name == "exposure_log.jsonl", "Ledger filename changed")
    _require(ledger_path.is_file(), "Public exposure ledger is missing")
    _require(_sha256(ledger_path) == ledger.get("byte_sha256"), "Ledger hash mismatch")
    _require(
        ledger_path.stat().st_size == ledger.get("byte_count"), "Ledger size mismatch"
    )
    _require(
        ledger_path.read_bytes().endswith(b"\n"),
        "Exposure ledger must be newline terminated",
    )

    contract = load_artifact_contract(DEFAULT_V2_CONFIG_PATH)
    progress_by_attempt = validate_formal_exposure_log(ledger_path, contract)
    _require(set(progress_by_attempt) == {attempt_id}, "Ledger attempt set changed")
    progress = progress_by_attempt[str(attempt_id)]
    _require(progress.terminal_failed, "Ledger is not terminal")
    _require(
        progress.completed_phase == "calibration_truth_opened",
        "Last completed phase changed",
    )
    _require(progress.pending_phase is None, "Terminal ledger cannot remain pending")
    _require(
        progress.prediction_commitment_byte_sha256 is None,
        "Ledger unexpectedly contains a prediction commitment",
    )
    _require(
        progress.opened_truth_files == EXPECTED_OPENED_TRUTH,
        "Ledger truth-exposure boundary changed",
    )
    _require(
        progress.truth_commitments_byte_sha256
        == verified_hashes["truth_commitments.json"],
        "Truth commitment does not bind the copied file",
    )
    _require(
        progress.fit_commitment_byte_sha256 == verified_hashes["fit_commitment.json"],
        "Fit commitment does not bind the copied file",
    )
    _require(
        progress.center_state_checkpoint_byte_sha256
        == verified_hashes["center_state_checkpoint.json"],
        "Center checkpoint does not bind the copied file",
    )
    _require(
        progress.risk_state_checkpoint_byte_sha256
        == verified_hashes["risk_state_checkpoint.json"],
        "Risk checkpoint does not bind the copied file",
    )
    _require(
        progress.model_state_commitment_byte_sha256 is None,
        "Model-state commitment must remain absent",
    )

    fit_commitment = _json_object(root / "fit_commitment.json")
    fit_files = fit_commitment.get("files")
    _require(isinstance(fit_files, list), "fit_commitment.files must be an array")
    truth_rows = [
        row
        for row in fit_files
        if isinstance(row, Mapping) and row.get("path") == "truth_commitments.json"
    ]
    _require(len(truth_rows) == 1, "Fit commitment truth reference changed")
    _require(
        truth_rows[0].get("byte_sha256") == verified_hashes["truth_commitments.json"],
        "Fit commitment references a different truth commitment",
    )

    training = _json_object(root / "training_manifest.json")
    center = _json_object(root / "center_state_checkpoint.json")
    risk = _json_object(root / "risk_state_checkpoint.json")
    _require(
        training.get("center_state_sha256") == center.get("center_state_sha256"),
        "Center state hash cross-reference changed",
    )
    _require(
        training.get("risk_state_sha256") == risk.get("risk_state_sha256"),
        "Risk state hash cross-reference changed",
    )
    _require(
        risk.get("training_manifest_byte_sha256")
        == verified_hashes["training_manifest.json"],
        "Risk checkpoint does not bind the training manifest",
    )

    return {
        "status": "passed",
        "protocol_id": manifest.get("protocol_id"),
        "attempt_id": attempt_id,
        "attempt_disposition": manifest.get("attempt_disposition"),
        "protocol_scientific_status": manifest.get("protocol_scientific_status"),
        "endpoint_evaluation_status": manifest.get("endpoint_evaluation_status"),
        "last_completed_phase": progress.completed_phase,
        "opened_truth_files": list(progress.opened_truth_files),
        "prediction_commitment_present": False,
        "verified_commitment_files": sorted(verified_hashes),
        "ledger_sha256": ledger.get("byte_sha256"),
        "termination_manifest_sha256": _sha256(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
    )
    args = parser.parse_args(argv)
    try:
        result = verify_terminal_evidence(args.evidence_dir)
    except TerminalEvidenceError as exc:
        result = {"status": "failed", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
