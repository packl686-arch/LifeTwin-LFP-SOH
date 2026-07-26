from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Callable

import pytest

from lifetwin.experiments import calendar_long_horizon_v016_terminal as terminal
from lifetwin.experiments import calendar_long_horizon_v016_training as training
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_PROTOCOL_ID,
    load_v021_design,
)
from lifetwin.experiments.calendar_long_horizon_v016_training import (
    V021CalibrationTerminalInconclusive,
)


_GIT_COMMIT = "1" * 40
_PLAN_RAW = b'{"kind":"generation_plan_commitment_fixture"}\n'
_TRUTH_RAW = b'{"kind":"truth_commitment_fixture"}\n'
_ACTUAL_RAW = b'{"kind":"actual_analysis_hash_ledger_fixture"}\n'
_FIT_RAW = b'{"kind":"fit_commitment_fixture"}\n'
_CENTER_RAW = b'{"kind":"center_checkpoint_fixture"}\n'
_RISK_RAW = b'{"kind":"risk_checkpoint_fixture"}\n'
_MASK_RAW = b'{"kind":"calibration_mask_commitment_fixture"}\n'
_TRAINING_RAW = b'{"kind":"training_manifest_fixture"}\n'
_PREFIX_RAW = b"cluster_id,prefix_feature\nfixture,1.0\n"
_PLAN_SHA = hashlib.sha256(_PLAN_RAW).hexdigest()
_TRUTH_SHA = hashlib.sha256(_TRUTH_RAW).hexdigest()
_ACTUAL_SHA = hashlib.sha256(_ACTUAL_RAW).hexdigest()
_FIT_SHA = hashlib.sha256(_FIT_RAW).hexdigest()
_CENTER_SHA = hashlib.sha256(_CENTER_RAW).hexdigest()
_RISK_SHA = hashlib.sha256(_RISK_RAW).hexdigest()
_MASK_SHA = hashlib.sha256(_MASK_RAW).hexdigest()
_TRAINING_TERMINAL_REASONS = frozenset(
    {
        "CALIBRATION_SOURCE_COUNT_NOT_900",
        "CALIBRATION_RISK_ELIGIBLE_BELOW_855",
        "CALIBRATION_RISK_POSITIVE_BELOW_60",
        "CALIBRATION_RISK_NEGATIVE_BELOW_60",
        "CALIBRATION_RISK_SCORE_NONFINITE",
        "CALIBRATION_ISOTONIC_FIT_UNDEFINED",
        "CALIBRATION_BASELINE_INCOMPLETE",
        "CALIBRATION_ZERO_FAMILY_NO_BAND",
        "CALIBRATION_BAND_NONFINITE_OR_UNORDERED",
        "CALIBRATION_CONFORMAL_COUNT_NOT_900",
        "CALIBRATION_CONFORMAL_SCORE_NONFINITE",
        "CALIBRATION_CONFORMAL_FIT_UNDEFINED",
    }
)
_REQUIRED_ATTEMPT_FIELDS = frozenset(
    {
        "protocol_id",
        "attempt_id",
        "created_utc",
        "git_commit",
        "git_dirty",
        "config_byte_sha256",
        "phase",
        "reason_code",
        "scientific_status",
        "truth_commitments_byte_sha256",
        "prediction_commitment_byte_sha256",
        "opened_truth_files",
        "message",
    }
)


def _context() -> terminal.TerminalContext:
    return terminal.TerminalContext(
        protocol_id=V021_PROTOCOL_ID,
        attempt_id="v016-fixture",
        git_commit=_GIT_COMMIT,
        git_dirty=False,
        config_byte_sha256=terminal.V021_AMENDMENT_BYTE_SHA256,
        created_utc="2026-07-26T00:00:00Z",
        terminated_utc="2026-07-26T00:10:00Z",
        attempted_phase="model_state_committed",
        last_completed_phase="calibration_truth_opened",
        truth_commitments_byte_sha256=_TRUTH_SHA,
        generation_plan_commitment_byte_sha256=_PLAN_SHA,
        actual_analysis_hash_ledger_commitment_byte_sha256=_ACTUAL_SHA,
        fit_commitment_byte_sha256=_FIT_SHA,
        center_state_checkpoint_byte_sha256=_CENTER_SHA,
        risk_state_checkpoint_byte_sha256=_RISK_SHA,
        calibration_mask_commitment_byte_sha256=_MASK_SHA,
    )


@pytest.fixture(autouse=True)
def _formal_environment_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        terminal,
        "verify_formal_environment",
        lambda _root: SimpleNamespace(
            protocol_id=V021_PROTOCOL_ID,
            git_commit=_GIT_COMMIT,
            git_dirty=False,
            config_byte_sha256=terminal.V021_AMENDMENT_BYTE_SHA256,
        ),
    )


def _ledger_row(
    *,
    phase: str = "calibration_truth_opened",
    message: str = "fixture",
    truth_sha: str | None = _TRUTH_SHA,
    opened: list[str] | None = None,
    prediction_sha: str | None = None,
) -> dict[str, object]:
    return {
        "attempt_id": "v016-fixture",
        "config_byte_sha256": terminal.V021_AMENDMENT_BYTE_SHA256,
        "created_utc": "2026-07-26T00:09:00Z",
        "exit_status": "completed",
        "git_commit": _GIT_COMMIT,
        "git_dirty": False,
        "message": message,
        "opened_truth_files": (
            [
                "calibration_truth.csv",
                "center_development_truth.csv",
                "risk_development_truth.csv",
            ]
            if opened is None
            else opened
        ),
        "phase": phase,
        "prediction_commitment_byte_sha256": prediction_sha,
        "truth_commitments_byte_sha256": truth_sha,
    }


def _canonical_line(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _make_roots(
    tmp_path: Path,
    *,
    opened: list[str] | None = None,
    prediction_sha: str | None = None,
) -> tuple[Path, Path, bytes]:
    label_root = tmp_path / "label_free"
    termination_root = tmp_path / "termination"
    label_root.mkdir()
    termination_root.mkdir()
    rows = (
        _ledger_row(
            phase="before_generation",
            truth_sha=None,
            opened=[],
        ),
        _ledger_row(
            phase="generation_plan_committed",
            message=f"generation_plan_commitment_byte_sha256={_PLAN_SHA}",
            truth_sha=None,
            opened=[],
        ),
        _ledger_row(phase="truth_committed", opened=[]),
        _ledger_row(
            phase="actual_analysis_hash_ledger_committed",
            message=(
                f"actual_analysis_hash_ledger_commitment_byte_sha256={_ACTUAL_SHA}"
            ),
            opened=[],
        ),
        _ledger_row(
            phase="label_free_fit_committed",
            message=f"fit_commitment_byte_sha256={_FIT_SHA}",
            opened=[],
        ),
        _ledger_row(
            phase="center_truth_opened",
            opened=["center_development_truth.csv"],
        ),
        _ledger_row(
            phase="center_state_committed",
            message=f"center_state_checkpoint_byte_sha256={_CENTER_SHA}",
            opened=["center_development_truth.csv"],
        ),
        _ledger_row(
            phase="risk_truth_opened",
            opened=[
                "center_development_truth.csv",
                "risk_development_truth.csv",
            ],
        ),
        _ledger_row(
            phase="risk_state_committed",
            message=f"risk_state_checkpoint_byte_sha256={_RISK_SHA}",
            opened=[
                "center_development_truth.csv",
                "risk_development_truth.csv",
            ],
        ),
        _ledger_row(
            phase="calibration_mask_committed",
            message=f"calibration_mask_commitment_byte_sha256={_MASK_SHA}",
            opened=[
                "center_development_truth.csv",
                "risk_development_truth.csv",
            ],
        ),
        _ledger_row(opened=opened, prediction_sha=prediction_sha),
    )
    prefix = b"".join(_canonical_line(row) for row in rows)
    (label_root / "exposure_log.jsonl").write_bytes(prefix)
    for name, raw in {
        "generation_plan_commitment.json": _PLAN_RAW,
        "truth_commitments.json": _TRUTH_RAW,
        "actual_analysis_hash_ledger_commitment.json": _ACTUAL_RAW,
        "fit_commitment.json": _FIT_RAW,
        "center_state_checkpoint.json": _CENTER_RAW,
        "risk_state_checkpoint.json": _RISK_RAW,
        "calibration_mask_commitment.json": _MASK_RAW,
        "training_manifest.json": _TRAINING_RAW,
        "prefix_pack.csv": _PREFIX_RAW,
    }.items():
        (label_root / name).write_bytes(raw)
    return termination_root, label_root, prefix


def _advance_to_model_committed(
    label_root: Path,
    prefix: bytes,
) -> tuple[bytes, str]:
    for name, raw in {
        "calibration_manifest.json": b'{"kind":"calibration_manifest_fixture"}\n',
        "calibration_population_audit.json": (
            b'{"kind":"calibration_population_audit_fixture"}\n'
        ),
        "model_state.json": b'{"kind":"model_state_fixture"}\n',
    }.items():
        (label_root / name).write_bytes(raw)
    committed_names = (
        "fit_commitment.json",
        "center_state_checkpoint.json",
        "risk_state_checkpoint.json",
        "training_manifest.json",
        "calibration_mask_commitment.json",
        "calibration_manifest.json",
        "calibration_population_audit.json",
        "model_state.json",
    )
    entries = []
    for name in committed_names:
        raw = (label_root / name).read_bytes()
        entries.append(
            {
                "path": name,
                "row_count": 1,
                "byte_count": len(raw),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    model_commitment_raw = (
        json.dumps(
            {
                "protocol_id": V021_PROTOCOL_ID,
                "config_sha256": terminal.V021_AMENDMENT_BYTE_SHA256,
                "git_commit": _GIT_COMMIT,
                "files": entries,
                "created_utc": "2026-07-26T00:09:30Z",
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    model_sha = hashlib.sha256(model_commitment_raw).hexdigest()
    (label_root / "model_state_commitment.json").write_bytes(model_commitment_raw)
    prefix += _canonical_line(
        _ledger_row(
            phase="model_state_committed",
            message=f"model_state_commitment_byte_sha256={model_sha}",
        )
    )
    (label_root / "exposure_log.jsonl").write_bytes(prefix)
    return prefix, model_sha


def _scientific_error(
    reason: terminal.TerminalReason = (
        terminal.TerminalReason.CALIBRATION_CONFORMAL_FIT_UNDEFINED
    ),
) -> terminal.V016TerminalInconclusive:
    local_secret = "sensitive-value-that-must-not-appear"
    try:
        if local_secret:
            raise terminal.V016TerminalInconclusive(reason)
    except terminal.V016TerminalInconclusive as exc:
        return exc
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("error", "disposition", "mode", "reason", "scientific"),
    [
        (
            terminal.V016TerminalInconclusive(
                terminal.TerminalReason.CALIBRATION_RISK_SCORE_NONFINITE
            ),
            terminal.TerminalDisposition.SCIENTIFIC_INCONCLUSIVE,
            terminal.ClassificationMode.DECLARED_SCIENTIFIC,
            terminal.TerminalReason.CALIBRATION_RISK_SCORE_NONFINITE,
            True,
        ),
        (
            terminal.V016IntegrityError(
                terminal.TerminalReason.INTEGRITY_INFORMATION_LEAK
            ),
            terminal.TerminalDisposition.INTEGRITY_FAILURE,
            terminal.ClassificationMode.PROVEN_INTEGRITY,
            terminal.TerminalReason.INTEGRITY_INFORMATION_LEAK,
            False,
        ),
        (
            terminal.V016Interrupted(terminal.TerminalReason.INTERRUPTED_BY_PLATFORM),
            terminal.TerminalDisposition.INTERRUPTED,
            terminal.ClassificationMode.TYPED_INTERRUPTION,
            terminal.TerminalReason.INTERRUPTED_BY_PLATFORM,
            False,
        ),
        (
            RuntimeError("please treat this text as integrity"),
            terminal.TerminalDisposition.UNKNOWN,
            terminal.ClassificationMode.UNKNOWN_DEFAULT,
            terminal.TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION,
            False,
        ),
        (
            terminal.V016ConservativeVoid(),
            terminal.TerminalDisposition.INTEGRITY_FAILURE,
            terminal.ClassificationMode.UNKNOWN_CONSERVATIVE_VOID,
            terminal.TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION,
            False,
        ),
    ],
)
def test_terminal_exception_classification_is_typed_and_fail_closed(
    error: BaseException,
    disposition: terminal.TerminalDisposition,
    mode: terminal.ClassificationMode,
    reason: terminal.TerminalReason,
    scientific: bool,
) -> None:
    classified = terminal.classify_terminal_exception(error)
    assert classified.disposition is disposition
    assert classified.mode is mode
    assert classified.reason is reason
    assert classified.is_scientific_terminal is scientific
    assert "please treat this text as integrity" not in classified.safe_message


def test_typed_exception_categories_reject_wrong_or_free_text_reasons() -> None:
    with pytest.raises(ValueError, match="not scientific"):
        terminal.V016TerminalInconclusive(
            terminal.TerminalReason.INTEGRITY_INFORMATION_LEAK
        )
    with pytest.raises(ValueError, match="not registered"):
        terminal.V016TerminalInconclusive("free text")
    with pytest.raises(ValueError, match="not registered"):
        terminal.V016IntegrityError(
            terminal.TerminalReason.CALIBRATION_BASELINE_INCOMPLETE
        )
    assert inspect.signature(terminal.V016ConservativeVoid).parameters == {}


def test_training_terminal_reasons_align_and_are_classified_scientific() -> None:
    tree = ast.parse(inspect.getsource(training))
    raised_reason_literals = {
        value.value
        for value in ast.walk(tree)
        if isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.startswith("CALIBRATION_")
    }
    assert raised_reason_literals == _TRAINING_TERMINAL_REASONS
    assert _TRAINING_TERMINAL_REASONS.issubset(
        {reason.value for reason in terminal.SCIENTIFIC_TERMINAL_REASONS}
    )
    for reason in sorted(_TRAINING_TERMINAL_REASONS):
        error = V021CalibrationTerminalInconclusive(reason, "private detail")
        classified = terminal.classify_terminal_exception(error)
        assert classified.is_scientific_terminal
        assert classified.reason.value == reason
        assert "private detail" not in classified.safe_message


def test_forged_training_exception_name_cannot_claim_scientific_status() -> None:
    forged_type = type(
        "V021CalibrationTerminalInconclusive",
        (RuntimeError,),
        {"__module__": ("lifetwin.experiments.calendar_long_horizon_v016_training")},
    )
    forged = forged_type("forged")
    forged.reason_code = "CALIBRATION_ZERO_FAMILY_NO_BAND"
    classified = terminal.classify_terminal_exception(forged)
    assert classified.disposition is terminal.TerminalDisposition.UNKNOWN
    assert classified.reason is (
        terminal.TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION
    )


def test_context_is_bound_to_validated_v021_identity() -> None:
    design = load_v021_design()
    assert terminal.V021_AMENDMENT_BYTE_SHA256 == design.config_byte_sha256
    assert _context().protocol_id == V021_PROTOCOL_ID
    with pytest.raises(ValueError, match="protocol_id"):
        replace(_context(), protocol_id="synthetic_other_protocol")
    with pytest.raises(ValueError, match="amendment"):
        replace(_context(), config_byte_sha256="9" * 64)


def test_structural_traceback_and_stderr_are_deterministically_sanitized(
    tmp_path: Path,
) -> None:
    error = _scientific_error()
    first = terminal.sanitized_structural_traceback(error, repo_root=tmp_path)
    second = terminal.sanitized_structural_traceback(error, repo_root=tmp_path)
    assert first == second
    assert b"sensitive-value-that-must-not-appear" not in first
    assert str(Path(__file__).resolve()).encode() not in first
    assert re.search(rb"0[xX][0-9A-Fa-f]+", first) is None
    assert re.search(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", first) is None
    decoded = json.loads(first)
    assert decoded["reason_code"] == "CALIBRATION_CONFORMAL_FIT_UNDEFINED"
    assert decoded["frames"]
    assert all(not Path(row["file"]).is_absolute() for row in decoded["frames"])

    digest = "a" * 64
    stderr = terminal.sanitized_stderr_bytes(
        "C:\\Users\\name\\secret.txt "
        "\\\\server\\private\\secret 0XDEADBEEF "
        f"{digest}\r\n"
    )
    assert b"C:\\Users" not in stderr
    assert b"\\\\server" not in stderr
    assert b"0XDEADBEEF" not in stderr
    assert digest.encode() not in stderr
    assert stderr.endswith(b"\n")


@pytest.mark.parametrize(
    ("error_factory", "disposition", "mode", "reason"),
    [
        (
            _scientific_error,
            "inconclusive_not_success",
            "declared_scientific",
            "CALIBRATION_CONFORMAL_FIT_UNDEFINED",
        ),
        (
            lambda: terminal.V016IntegrityError(
                terminal.TerminalReason.INTEGRITY_ENVIRONMENT_MISMATCH
            ),
            "void",
            "proven_integrity",
            "INTEGRITY_ENVIRONMENT_MISMATCH",
        ),
        (
            lambda: terminal.V016Interrupted(
                terminal.TerminalReason.INTERRUPTED_BY_PLATFORM
            ),
            "interrupted",
            "typed_interruption",
            "INTERRUPTED_BY_PLATFORM",
        ),
        (
            lambda: RuntimeError("unknown private detail"),
            "unclassified_terminal_not_success",
            "unknown_default",
            "UNKNOWN_PRE_PREDICTION_EXCEPTION",
        ),
        (
            terminal.V016ConservativeVoid,
            "void",
            "unknown_conservative_void",
            "UNKNOWN_PRE_PREDICTION_EXCEPTION",
        ),
    ],
)
def test_generic_publisher_emits_each_exact_terminal_disposition(
    tmp_path: Path,
    error_factory: Callable[[], BaseException],
    disposition: str,
    mode: str,
    reason: str,
) -> None:
    termination_root, label_root, _ = _make_roots(tmp_path)
    result = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=error_factory(),
    )
    attempt = json.loads(
        (termination_root / "terminal_attempt_record.json").read_bytes()
    )
    terminal.validate_terminal_attempt_record(attempt)
    assert attempt["attempt_disposition"] == disposition
    assert attempt["scientific_status"] == disposition
    assert attempt["classification_mode"] == mode
    assert attempt["reason_code"] == reason
    assert result.disposition.value == disposition
    assert result.reason.value == reason


def test_pre_generation_collision_can_publish_without_truth_commitment(
    tmp_path: Path,
) -> None:
    label_root = tmp_path / "label_free"
    termination_root = tmp_path / "termination"
    label_root.mkdir()
    termination_root.mkdir()
    ledger_raw = _canonical_line(
        _ledger_row(
            phase="before_generation",
            truth_sha=None,
            opened=[],
        )
    )
    (label_root / "exposure_log.jsonl").write_bytes(ledger_raw)
    context = replace(
        _context(),
        attempted_phase="generation_plan_committed",
        last_completed_phase="before_generation",
        truth_commitments_byte_sha256=None,
        generation_plan_commitment_byte_sha256=None,
        actual_analysis_hash_ledger_commitment_byte_sha256=None,
        fit_commitment_byte_sha256=None,
        center_state_checkpoint_byte_sha256=None,
        risk_state_checkpoint_byte_sha256=None,
        calibration_mask_commitment_byte_sha256=None,
        model_state_commitment_byte_sha256=None,
    )
    result = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=context,
        error=terminal.V016IntegrityError(
            terminal.TerminalReason.INTEGRITY_PARTITION_SEED_ID_OR_CONTENT_COLLISION
        ),
    )
    attempt = json.loads(
        (termination_root / "terminal_attempt_record.json").read_bytes()
    )
    manifest = json.loads(
        (termination_root / "terminal_artifact_manifest.json").read_bytes()
    )
    assert attempt["truth_commitments_byte_sha256"] is None
    assert manifest["preterminal_commitments"]["truth_commitments_byte_sha256"] is None
    assert [item["path"] for item in manifest["preterminal_artifacts"]] == [
        "exposure_log.jsonl"
    ]
    assert result.disposition is terminal.TerminalDisposition.INTEGRITY_FAILURE


def test_inconclusive_wrapper_remains_scientific_only(tmp_path: Path) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    with pytest.raises(terminal.V016TerminationError, match="non-scientific"):
        terminal.publish_terminal_inconclusive(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=RuntimeError("unknown"),
        )
    assert list(termination_root.iterdir()) == []
    assert (label_root / "exposure_log.jsonl").read_bytes() == prefix

    result = terminal.publish_terminal_inconclusive(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=_scientific_error(),
    )
    assert result.disposition is terminal.TerminalDisposition.SCIENTIFIC_INCONCLUSIVE


def test_publish_hashes_exact_label_free_inventory_and_is_idempotent(
    tmp_path: Path,
) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    error = _scientific_error()
    stderr_secret = "do-not-publish-this-diagnostic-content"
    result = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=error,
        stderr=stderr_secret,
        repo_root=Path(__file__).resolve().parents[1],
    )
    assert result.ledger_record_appended is True
    assert {path.name for path in termination_root.iterdir()} == {
        "terminal_attempt_record.json",
        "terminal_artifact_manifest.json",
        "terminal_exposure_log_snapshot.jsonl",
    }
    snapshot_path = termination_root / "terminal_exposure_log_snapshot.jsonl"
    assert snapshot_path.read_bytes() == prefix

    attempt_raw = (termination_root / "terminal_attempt_record.json").read_bytes()
    attempt = json.loads(attempt_raw)
    assert _REQUIRED_ATTEMPT_FIELDS.issubset(attempt)
    stderr_metadata = attempt["diagnostics"]["sanitized_stderr"]
    sanitized_stderr = terminal.sanitized_stderr_bytes(stderr_secret)
    assert stderr_metadata["byte_count"] == len(sanitized_stderr)
    assert (
        stderr_metadata["byte_sha256"] == hashlib.sha256(sanitized_stderr).hexdigest()
    )
    assert "redaction_policy" in stderr_metadata
    public_bytes = b"".join(
        path.read_bytes() for path in sorted(termination_root.iterdir())
    )
    assert stderr_secret.encode() not in public_bytes

    manifest_raw = result.manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    terminal.validate_terminal_artifact_manifest(manifest)
    paths = [item["path"] for item in manifest["preterminal_artifacts"]]
    assert paths == sorted(path.name for path in label_root.iterdir())
    assert "calibration_mask_commitment.json" in paths
    assert "actual_analysis_hash_ledger_commitment.json" in paths
    by_path = {item["path"]: item for item in manifest["preterminal_artifacts"]}
    assert by_path["calibration_mask_commitment.json"]["byte_sha256"] == _MASK_SHA
    assert (
        by_path["exposure_log.jsonl"]["byte_sha256"]
        == hashlib.sha256(prefix).hexdigest()
    )
    for path in label_root.iterdir():
        if path.name == "exposure_log.jsonl":
            continue
        assert (
            by_path[path.name]["byte_sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )

    ledger = label_root / "exposure_log.jsonl"
    ledger_after = ledger.read_bytes()
    record = json.loads(ledger_after[len(prefix) :])
    assert (
        record["terminal_artifact_manifest_byte_sha256"] == result.manifest_byte_sha256
    )
    repeated = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=error,
        stderr=stderr_secret,
        repo_root=Path(__file__).resolve().parents[1],
    )
    assert repeated.ledger_record_appended is False
    assert repeated.manifest_byte_sha256 == result.manifest_byte_sha256
    assert ledger.read_bytes() == ledger_after


@pytest.mark.parametrize(
    ("filename", "expected_reason"),
    [
        ("test_truth.csv", terminal.TerminalReason.INTEGRITY_FORBIDDEN_TRUTH_ACCESS),
        (
            "prediction_commitment.json",
            terminal.TerminalReason.INTEGRITY_INFORMATION_LEAK,
        ),
        ("score_report.json", terminal.TerminalReason.INTEGRITY_INFORMATION_LEAK),
        (
            "undeclared_artifact.bin",
            terminal.TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH,
        ),
    ],
)
def test_strict_label_free_inventory_rejects_forbidden_or_unknown_files(
    tmp_path: Path,
    filename: str,
    expected_reason: terminal.TerminalReason,
) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    (label_root / filename).write_bytes(b"forbidden\n")
    with pytest.raises(terminal.V016IntegrityError) as raised:
        terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=_scientific_error(),
        )
    assert raised.value.reason is expected_reason
    assert list(termination_root.iterdir()) == []
    assert (label_root / "exposure_log.jsonl").read_bytes() == prefix


def test_strict_label_free_inventory_rejects_links_and_directories(
    tmp_path: Path,
) -> None:
    termination_root, label_root, _ = _make_roots(tmp_path)
    (label_root / "unexpected_directory").mkdir()
    with pytest.raises(terminal.V016IntegrityError) as directory_error:
        terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=_scientific_error(),
        )
    assert (
        directory_error.value.reason
        is terminal.TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH
    )

    (label_root / "unexpected_directory").rmdir()
    link = label_root / "operating_pack.csv"
    try:
        os.symlink(label_root / "prefix_pack.csv", link)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this Windows setup")
    with pytest.raises(terminal.V016IntegrityError) as link_error:
        terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=_scientific_error(),
        )
    assert (
        link_error.value.reason
        is terminal.TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH
    )


@pytest.mark.parametrize("mutation", ["missing", "tampered", "uncommitted"])
def test_calibration_mask_commitment_is_evidence_backed(
    tmp_path: Path,
    mutation: str,
) -> None:
    termination_root, label_root, _ = _make_roots(tmp_path)
    mask_path = label_root / "calibration_mask_commitment.json"
    context = _context()
    if mutation == "missing":
        mask_path.unlink()
    elif mutation == "tampered":
        mask_path.write_bytes(b"tampered\n")
    else:
        mask_path.unlink()
        context = replace(
            context,
            calibration_mask_commitment_byte_sha256=None,
        )
    with pytest.raises(terminal.V016IntegrityError):
        terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=context,
            error=_scientific_error(),
        )
    assert list(termination_root.iterdir()) == []


def test_completed_model_phase_requires_and_binds_model_inventory(
    tmp_path: Path,
) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    prefix, model_sha = _advance_to_model_committed(label_root, prefix)
    context = replace(
        _context(),
        attempted_phase="prediction_started",
        last_completed_phase="model_state_committed",
        model_state_commitment_byte_sha256=model_sha,
    )
    missing_root = tmp_path / "label_free_missing"
    missing_root.mkdir()
    for path in label_root.iterdir():
        if path.name != "calibration_population_audit.json":
            (missing_root / path.name).write_bytes(path.read_bytes())

    result = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=context,
        error=RuntimeError("prediction was not started"),
    )
    assert result.disposition is terminal.TerminalDisposition.UNKNOWN

    empty_termination = tmp_path / "termination_missing"
    empty_termination.mkdir()
    with pytest.raises(terminal.V016IntegrityError) as raised:
        terminal.publish_terminal(
            termination_root=empty_termination,
            label_free_artifact_root=missing_root,
            context=context,
            error=RuntimeError("prediction was not started"),
        )
    assert raised.value.reason is terminal.TerminalReason.INTEGRITY_MISSING_COMMITMENT


def test_model_state_stage_hashes_partial_direct_artifacts(tmp_path: Path) -> None:
    termination_root, label_root, _ = _make_roots(tmp_path)
    partial = b'{"kind":"partial_model_state"}\n'
    (label_root / "model_state.json").write_bytes(partial)
    result = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=terminal.V016Interrupted(terminal.TerminalReason.INTERRUPTED_BY_PLATFORM),
    )
    manifest = json.loads(result.manifest_path.read_bytes())
    inventory = {item["path"]: item for item in manifest["preterminal_artifacts"]}
    assert (
        inventory["model_state.json"]["byte_sha256"]
        == hashlib.sha256(partial).hexdigest()
    )


@pytest.mark.parametrize("partial_count", [1, 2])
def test_prediction_started_terminal_hashes_partial_outputs(
    tmp_path: Path,
    partial_count: int,
) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    prefix, model_sha = _advance_to_model_committed(label_root, prefix)
    started = _ledger_row(
        phase="prediction_started",
        message="truth-incapable prediction fixture started",
    )
    started["exit_status"] = "started"
    prefix += _canonical_line(started)
    (label_root / "exposure_log.jsonl").write_bytes(prefix)
    for name in ("prediction_bundle.csv", "risk_bundle.csv")[:partial_count]:
        (label_root / name).write_bytes(f"{name},partial\n".encode("ascii"))
    context = replace(
        _context(),
        attempted_phase="prediction_started",
        last_completed_phase="model_state_committed",
        model_state_commitment_byte_sha256=model_sha,
    )
    result = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=context,
        error=terminal.V016Interrupted(terminal.TerminalReason.INTERRUPTED_BY_PLATFORM),
    )
    manifest = json.loads(result.manifest_path.read_bytes())
    inventory = {item["path"]: item for item in manifest["preterminal_artifacts"]}
    for name in ("prediction_bundle.csv", "risk_bundle.csv")[:partial_count]:
        assert (
            inventory[name]["byte_sha256"]
            == hashlib.sha256((label_root / name).read_bytes()).hexdigest()
        )
    assert "decision_bundle.csv" not in inventory
    assert result.disposition is terminal.TerminalDisposition.INTERRUPTED


def test_environment_mismatch_overrides_scientific_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    termination_root, label_root, _ = _make_roots(tmp_path)
    monkeypatch.setattr(
        terminal,
        "verify_formal_environment",
        lambda _root: (_ for _ in ()).throw(
            TypeError("fixture verifier could not establish identity")
        ),
    )
    result = terminal.publish_terminal_inconclusive(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=_scientific_error(),
    )
    assert result.disposition is terminal.TerminalDisposition.INTEGRITY_FAILURE
    assert result.reason is terminal.TerminalReason.INTEGRITY_ENVIRONMENT_MISMATCH
    attempt = json.loads(
        (termination_root / "terminal_attempt_record.json").read_bytes()
    )
    assert attempt["reason_code"] == "INTEGRITY_ENVIRONMENT_MISMATCH"


@pytest.mark.parametrize(
    ("failure_point", "expected_prefix"),
    [
        ("before_attempt", frozenset()),
        ("after_attempt", frozenset({"terminal_attempt_record.json"})),
        (
            "after_snapshot",
            frozenset(
                {
                    "terminal_attempt_record.json",
                    "terminal_exposure_log_snapshot.jsonl",
                }
            ),
        ),
        (
            "before_ledger",
            frozenset(
                {
                    "terminal_attempt_record.json",
                    "terminal_exposure_log_snapshot.jsonl",
                    "terminal_artifact_manifest.json",
                }
            ),
        ),
    ],
)
def test_partial_publication_recovers_at_every_write_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    expected_prefix: frozenset[str],
) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    error = _scientific_error()
    original_write = terminal._atomic_exclusive_write
    original_finish = terminal._finish_terminal_ledger
    fail_index = {
        "before_attempt": 0,
        "after_attempt": 1,
        "after_snapshot": 2,
    }.get(failure_point)
    write_count = 0

    def flaky_write(path: Path, raw: bytes) -> None:
        nonlocal write_count
        if write_count == fail_index:
            raise RuntimeError("simulated interruption")
        write_count += 1
        original_write(path, raw)

    def flaky_finish(
        path: Path,
        *,
        expected_prefix: bytes,
        expected_record: dict[str, object],
    ) -> bool:
        del path, expected_prefix, expected_record
        raise RuntimeError("simulated interruption")

    if failure_point == "before_ledger":
        monkeypatch.setattr(terminal, "_finish_terminal_ledger", flaky_finish)
    else:
        monkeypatch.setattr(terminal, "_atomic_exclusive_write", flaky_write)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=error,
        )
    assert {path.name for path in termination_root.iterdir()} == expected_prefix
    assert (label_root / "exposure_log.jsonl").read_bytes() == prefix

    monkeypatch.setattr(terminal, "_atomic_exclusive_write", original_write)
    monkeypatch.setattr(terminal, "_finish_terminal_ledger", original_finish)
    recovered = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=error,
    )
    assert recovered.ledger_record_appended is True
    assert {path.name for path in termination_root.iterdir()} == {
        "terminal_attempt_record.json",
        "terminal_exposure_log_snapshot.jsonl",
        "terminal_artifact_manifest.json",
    }
    assert (
        len(
            (label_root / "exposure_log.jsonl").read_bytes()[len(prefix) :].splitlines()
        )
        == 1
    )


def test_partial_publication_conflict_is_rejected_without_overwrite(
    tmp_path: Path,
) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    attempt_path = termination_root / "terminal_attempt_record.json"
    attempt_path.write_bytes(b'{"forged":true}\n')
    before = attempt_path.read_bytes()
    with pytest.raises(terminal.V016TerminationError, match="conflicts"):
        terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=_scientific_error(),
        )
    assert attempt_path.read_bytes() == before
    assert (label_root / "exposure_log.jsonl").read_bytes() == prefix


def test_process_lock_serializes_threads_and_prevents_double_append(
    tmp_path: Path,
) -> None:
    termination_root, label_root, prefix = _make_roots(tmp_path)
    error = _scientific_error()

    def publish() -> terminal.PublishedTermination:
        return terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=error,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: publish(), range(2)))
    assert sorted(result.ledger_record_appended for result in results) == [
        False,
        True,
    ]
    ledger = (label_root / "exposure_log.jsonl").read_bytes()
    assert len(ledger[len(prefix) :].splitlines()) == 1
    assert not (termination_root / ".terminal_publish.lock").exists()


def test_validator_rejects_disposition_reason_or_inventory_forgery(
    tmp_path: Path,
) -> None:
    termination_root, label_root, _ = _make_roots(tmp_path)
    result = terminal.publish_terminal(
        termination_root=termination_root,
        label_free_artifact_root=label_root,
        context=_context(),
        error=RuntimeError("unknown"),
    )
    attempt_path = termination_root / "terminal_attempt_record.json"
    manifest = json.loads(result.manifest_path.read_bytes())

    forged_mode = json.loads(attempt_path.read_bytes())
    forged_mode["classification_mode"] = "proven_integrity"
    with pytest.raises(terminal.V016TerminationError, match="classification"):
        terminal.validate_terminal_attempt_record(forged_mode)

    forged_message = json.loads(attempt_path.read_bytes())
    forged_message["message"] = "free-text override"
    with pytest.raises(terminal.V016TerminationError, match="disposition"):
        terminal.validate_terminal_attempt_record(forged_message)

    forged_inventory = json.loads(result.manifest_path.read_bytes())
    forged_inventory["preterminal_artifacts"].append(
        {
            "path": "model_state.json",
            "byte_count": 1,
            "byte_sha256": "9" * 64,
        }
    )
    with pytest.raises(
        terminal.V016TerminationError,
        match="Preterminal artifact registry",
    ):
        terminal.validate_terminal_artifact_manifest(forged_inventory)

    extra = dict(manifest)
    extra["undeclared"] = True
    with pytest.raises(terminal.V016TerminationError, match="keys are not exact"):
        terminal.validate_terminal_artifact_manifest(extra)


def test_roots_must_be_strict_disjoint_label_free_directories(
    tmp_path: Path,
) -> None:
    _, label_root, prefix = _make_roots(tmp_path)
    with pytest.raises(terminal.V016TerminationError, match="disjoint"):
        terminal.publish_terminal(
            termination_root=label_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=_scientific_error(),
        )
    assert (label_root / "exposure_log.jsonl").read_bytes() == prefix


def test_publication_rejects_reparse_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    termination_root, label_root, _ = _make_roots(tmp_path)
    parent = label_root.parent
    original = terminal._is_reparse
    monkeypatch.setattr(
        terminal,
        "_is_reparse",
        lambda path: path == parent or original(path),
    )
    with pytest.raises(
        terminal.V016TerminationError,
        match="traverses a reparse point",
    ):
        terminal.publish_terminal(
            termination_root=termination_root,
            label_free_artifact_root=label_root,
            context=_context(),
            error=_scientific_error(),
        )


def test_terminal_writer_has_no_sealed_truth_or_scoring_capability() -> None:
    signature = inspect.signature(terminal.publish_terminal)
    assert "label_free_artifact_root" in signature.parameters
    assert "sealed_truth_root" not in signature.parameters
    assert "ledger_path" not in signature.parameters
    assert "score_root" not in signature.parameters
    assert "model_state" not in signature.parameters
    source = inspect.getsource(terminal)
    assert "open_truth_for_phase" not in source
    assert "read_canonical_csv" not in source
    assert "pandas" not in source
