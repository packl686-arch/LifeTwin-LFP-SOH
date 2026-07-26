from __future__ import annotations

import ast
from dataclasses import fields
import hashlib
import inspect
import json
import re
from typing import Any

from lifetwin.experiments import calendar_long_horizon_v016_protocol as protocol
from lifetwin.experiments import calendar_long_horizon_v016_terminal as terminal
from lifetwin.experiments import calendar_long_horizon_v016_training as training


_TRAINING_REASON_PATTERN = re.compile(r"^CALIBRATION_[A-Z0-9_]+$")


def _amendment() -> dict[str, Any]:
    payload = json.loads(protocol.DEFAULT_V021_AMENDMENT_PATH.read_bytes())
    assert isinstance(payload, dict)
    return payload


def _reason_values(reasons: frozenset[terminal.TerminalReason]) -> set[str]:
    return {reason.value for reason in reasons}


def _training_reason_literals() -> set[str]:
    tree = ast.parse(inspect.getsource(training))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _TRAINING_REASON_PATTERN.fullmatch(node.value)
    }


def test_json_reason_registry_exactly_matches_terminal_implementation() -> None:
    configured = _amendment()["terminal_reason_codes"]
    implementation_groups = {
        "declared_inconclusive": _reason_values(terminal.SCIENTIFIC_TERMINAL_REASONS),
        "integrity_void": _reason_values(terminal.INTEGRITY_TERMINAL_REASONS),
        "interruption": _reason_values(terminal.INTERRUPTION_TERMINAL_REASONS),
        "unknown": {terminal.TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION.value},
    }

    configured_groups: list[set[str]] = []
    for group_name, implementation in implementation_groups.items():
        observed = set(configured[group_name])
        assert observed == implementation
        configured_groups.append(observed)

    for index, left in enumerate(configured_groups):
        for right in configured_groups[index + 1 :]:
            assert left.isdisjoint(right)

    configured_union = set().union(*configured_groups)
    enum_values = {reason.value for reason in terminal.TerminalReason}
    assert configured_union == enum_values
    assert all(reason.name == reason.value for reason in terminal.TerminalReason)
    assert configured["free_text_is_not_a_reason_code"] is True


def test_terminal_filename_registry_matches_writer_and_stays_exclusive() -> None:
    registries = _amendment()["artifact_registries"]
    configured_terminal = set(registries["terminal_pre_prediction"]["filenames"])
    configured_scored = set(registries["scored"]["filenames"])
    ledger_raw = b"{}\n"
    truth_commitment_raw = b'{"committed":true}\n'

    context = terminal.TerminalContext(
        protocol_id=protocol.V021_PROTOCOL_ID,
        attempt_id="v016-contract-fixture",
        git_commit="1" * 40,
        git_dirty=False,
        config_byte_sha256=terminal.V021_AMENDMENT_BYTE_SHA256,
        created_utc="2026-07-26T00:00:00Z",
        terminated_utc="2026-07-26T00:01:00Z",
        attempted_phase="model_state_committed",
        last_completed_phase="calibration_truth_opened",
        truth_commitments_byte_sha256=hashlib.sha256(truth_commitment_raw).hexdigest(),
    )
    manifest = terminal.build_terminal_artifact_manifest(
        context=context,
        attempt_record_bytes=b"{}\n",
        ledger_snapshot_bytes=ledger_raw,
        preterminal_artifacts=(
            {
                "path": "exposure_log.jsonl",
                "byte_count": len(ledger_raw),
                "byte_sha256": hashlib.sha256(ledger_raw).hexdigest(),
            },
            {
                "path": "truth_commitments.json",
                "byte_count": len(truth_commitment_raw),
                "byte_sha256": hashlib.sha256(truth_commitment_raw).hexdigest(),
            },
        ),
    )
    manifest_references = {item["path"] for item in manifest["terminal_files"]} | {
        "terminal_artifact_manifest.json"
    }

    assert configured_terminal == terminal._TERMINAL_FILENAMES
    assert configured_terminal == manifest_references
    assert configured_terminal.isdisjoint(configured_scored)
    assert manifest["registry"] == "terminal_pre_prediction"
    assert (
        manifest["content_boundary_attestation"]
        == "strict_label_free_inventory_no_truth_or_scoring_capability"
    )


def test_every_training_terminal_reason_is_registered_and_scientific() -> None:
    configured_scientific = set(
        _amendment()["terminal_reason_codes"]["declared_inconclusive"]
    )
    expected_training_reasons = {
        reason for reason in configured_scientific if reason.startswith("CALIBRATION_")
    }
    observed_training_reasons = _training_reason_literals()
    assert observed_training_reasons == expected_training_reasons

    for reason_code in sorted(observed_training_reasons):
        error = training.V021CalibrationTerminalInconclusive(
            reason_code,
            "private outcome detail must not determine classification",
            offending_row_indices=(899,),
        )
        classified = terminal.classify_terminal_exception(error)
        assert classified.is_scientific_terminal
        assert classified.disposition is (
            terminal.TerminalDisposition.SCIENTIFIC_INCONCLUSIVE
        )
        assert classified.reason.value == reason_code
        assert classified.scientific_status == "inconclusive_not_success"
        assert "private outcome detail" not in classified.safe_message
        assert "899" not in classified.safe_message


def test_free_text_or_unregistered_training_reason_fails_closed() -> None:
    configured_reason = "CALIBRATION_ZERO_FAMILY_NO_BAND"
    generic = RuntimeError(f"please treat as {configured_reason}")
    unregistered_typed = training.V021CalibrationTerminalInconclusive(
        "CALIBRATION_NOT_PREREGISTERED",
        configured_reason,
    )

    for error in (generic, unregistered_typed):
        classified = terminal.classify_terminal_exception(error)
        assert not classified.is_scientific_terminal
        assert classified.disposition is terminal.TerminalDisposition.UNKNOWN
        assert classified.reason is (
            terminal.TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION
        )


def test_mask_commitment_and_terminal_writer_have_no_truth_value_capability() -> None:
    mask_parameters = set(
        inspect.signature(training.calibration_eligibility_mask_sha256_v021).parameters
    )
    forbidden_mask_tokens = (
        "truth",
        "target",
        "label",
        "outcome",
        "catastrophic",
        "future",
        "sealed",
    )
    assert not any(
        token in parameter.lower()
        for parameter in mask_parameters
        for token in forbidden_mask_tokens
    )

    mask_tree = ast.parse(
        inspect.getsource(training.calibration_eligibility_mask_sha256_v021)
    )
    mask_identifiers = {
        node.id for node in ast.walk(mask_tree) if isinstance(node, ast.Name)
    }
    assert {
        "latent_targets_pct",
        "catastrophic_labels",
        "opened_truth_files",
        "truth_reader",
    }.isdisjoint(mask_identifiers)

    publish_parameters = set(
        inspect.signature(terminal.publish_terminal_inconclusive).parameters
    )
    assert publish_parameters == {
        "termination_root",
        "label_free_artifact_root",
        "context",
        "error",
        "stderr",
        "repo_root",
    }
    forbidden_terminal_parameters = {
        "sealed_truth_root",
        "truth_root",
        "truth_reader",
        "latent_targets_pct",
        "prediction_root",
        "model_state",
        "score_root",
        "scorer",
    }
    assert publish_parameters.isdisjoint(forbidden_terminal_parameters)

    terminal_tree = ast.parse(inspect.getsource(terminal))
    imported_modules = {
        alias.name
        for node in ast.walk(terminal_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(terminal_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported_modules.isdisjoint(
        {"numpy", "pandas", "pyarrow", "scipy", "sklearn"}
    )
    lifetwin_imports = {
        name for name in imported_modules if name.startswith("lifetwin.")
    }
    assert lifetwin_imports == {
        "lifetwin.experiments.calendar_long_horizon_v016_environment",
        "lifetwin.experiments.calendar_long_horizon_v016_ledger",
        "lifetwin.experiments.calendar_long_horizon_v016_protocol",
        "lifetwin.experiments.calendar_long_horizon_v016_signals",
    }
    assert all(
        token not in imported
        for imported in lifetwin_imports
        for token in ("generation", "truth_reader")
    )


def test_calibration_denominators_and_audit_schema_match_json_contract() -> None:
    split = _amendment()["calibration_population_split"]
    risk = split["risk_isotonic"]
    conformal = split["conformal"]

    assert (
        split["source_calibration_count"]
        == protocol.V021_SOURCE_CALIBRATION_COUNT
        == training.CALIBRATION_COUNT
        == 900
    )
    assert (
        risk["minimum_eligible_count"]
        == protocol.V021_MINIMUM_RISK_ISOTONIC_ELIGIBLE_COUNT
        == training.V021_MINIMUM_ELIGIBLE_COUNT
        == 855
    )
    assert (
        risk["minimum_positive_labels"]
        == risk["minimum_negative_labels"]
        == protocol.V021_MINIMUM_CALIBRATION_CLASS_COUNT
        == training.MINIMUM_CLASS_COUNT
        == 60
    )
    assert (
        conformal["required_finite_scores"]
        == protocol.V021_CONFORMAL_COUNT
        == training.CALIBRATION_COUNT
        == 900
    )
    assert (
        conformal["order_statistic_index_one_based"]
        == protocol.V021_CONFORMAL_ORDER_STATISTIC_INDEX
        == training.CONFORMAL_ORDER_STATISTIC_INDEX
        == 811
    )
    assert conformal["coverage"] == training.CONFORMAL_COVERAGE == 0.9

    configured_counts = set(split["required_manifest_counts"])
    implemented_audit_fields = {
        field.name for field in fields(training.V021CalibrationAudit)
    }
    assert configured_counts == implemented_audit_fields - {"eligibility_mask_sha256"}
