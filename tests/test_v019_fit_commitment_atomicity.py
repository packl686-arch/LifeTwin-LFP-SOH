from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lifetwin.experiments import calendar_long_horizon_v019_runner as runner
from lifetwin.experiments import calendar_long_horizon_v019_partition as partition
from lifetwin.experiments.calendar_long_horizon_v019_numeric_contract import (
    V024MemberFitNumericContractError,
)
from lifetwin.experiments.calendar_long_horizon_v020_contract import (
    load_v025_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v019_terminal import (
    ClassificationMode,
    TerminalDisposition,
    TerminalReason,
    classify_terminal_exception,
)


FIXTURE_ID = "development-fixture-not-formal-attempt"


def _exercise_fit_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: str | None,
) -> tuple[list[str], Path]:
    events: list[str] = []
    commitment = tmp_path / "fit_commitment.json"

    def append_phase(**kwargs: object) -> None:
        status = str(kwargs["exit_status"])
        if status == "completed" and fail_at == "phase_append":
            raise OSError("injected completed-phase append failure")
        events.append(f"phase:{status}")

    def write_fit(**kwargs: object) -> None:
        del kwargs
        events.append("write_and_readback")
        (tmp_path / "member_fit_diagnostics.csv").write_text(
            "partial diagnostics\n", encoding="utf-8"
        )
        if fail_at == "write_readback":
            raise ValueError("injected write/read-back failure")
        (tmp_path / "member_forecast_bundle.csv").write_text(
            "partial forecasts\n", encoding="utf-8"
        )

    def validate_whole(*args: object, **kwargs: object) -> object:
        del args, kwargs
        events.append("whole_validate")
        if fail_at == "whole_validate":
            raise ValueError("injected whole-bundle validation failure")
        return object()

    def commit_fit(**kwargs: object) -> str:
        del kwargs
        events.append("commit")
        if fail_at == "commit":
            raise ValueError("injected commitment failure")
        commitment.write_text("validated commitment\n", encoding="utf-8")
        return "d" * 64

    monkeypatch.setattr(runner, "_append_phase", append_phase)
    monkeypatch.setattr(
        runner,
        "_append_failure",
        lambda **kwargs: events.append("failure"),
    )
    monkeypatch.setattr(
        runner,
        "load_fresh_generation_bundle_v024",
        lambda **kwargs: events.append("load") or object(),
    )
    monkeypatch.setattr(
        runner,
        "fit_verified_generation_bundle_v024",
        lambda bundle: events.append("fit") or object(),
    )
    monkeypatch.setattr(runner, "write_verified_fit_result_v024", write_fit)
    monkeypatch.setattr(runner, "validate_whole_bundle_from_root", validate_whole)
    monkeypatch.setattr(runner, "commit_validated_fit_result_v024", commit_fit)
    monkeypatch.setattr(
        runner,
        "validate_formal_exposure_log",
        lambda *args, **kwargs: {FIXTURE_ID: object()},
    )
    monkeypatch.setattr(
        runner,
        "verify_phase_artifact_commitment",
        lambda *args, **kwargs: events.append("verify"),
    )

    paths = SimpleNamespace(
        label_free_root=tmp_path,
        ledger_path=tmp_path / "development-ledger-not-written.jsonl",
    )
    identity = SimpleNamespace(attempt_id=FIXTURE_ID)
    view = SimpleNamespace(artifacts=object())
    if fail_at is None:
        runner._fit_structure_stage(
            paths=paths,
            identity=identity,
            view=view,
            truth_hash="0" * 64,
        )
    else:
        with pytest.raises((OSError, ValueError), match="injected"):
            runner._fit_structure_stage(
                paths=paths,
                identity=identity,
                view=view,
                truth_hash="0" * 64,
            )
    return events, commitment


def test_fit_commitment_follows_readback_whole_validation_and_precedes_phase() -> None:
    # This source-order guard complements the injected runtime ordering checks.
    source = Path(runner.__file__).read_text(encoding="utf-8")
    body = source[
        source.index("def _fit_structure_stage") : source.index("def _apply_partition")
    ]
    assert body.index("write_verified_fit_result_v024(") < body.index(
        "validate_whole_bundle_from_root("
    )
    assert body.index("validate_whole_bundle_from_root(") < body.index(
        "commit_validated_fit_result_v024("
    )
    assert body.index("commit_validated_fit_result_v024(") < body.index(
        'exit_status="completed"'
    )


def test_v025_view_reaches_whole_validator_and_fit_stage_passes_that_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = load_v025_contract_view()
    missing_root = tmp_path / "fresh-label-free-root"

    with pytest.raises(partition.V024WholeBundleContractError, match="physical"):
        partition.validate_whole_bundle_from_root(missing_root, view)
    with pytest.raises(partition.V024PartitionCapabilityError, match="invalid"):
        partition.validate_whole_bundle_from_root(missing_root, view.artifacts)

    events: list[str] = []
    observed: list[object] = []
    paths = SimpleNamespace(label_free_root=tmp_path, ledger_path=tmp_path / "ledger")
    identity = SimpleNamespace(attempt_id=FIXTURE_ID)
    monkeypatch.setattr(
        runner,
        "_append_phase",
        lambda **kwargs: events.append(f"phase:{kwargs['exit_status']}"),
    )
    monkeypatch.setattr(
        runner, "_append_failure", lambda **kwargs: events.append("failure")
    )
    monkeypatch.setattr(
        runner, "load_fresh_generation_bundle_v024", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        runner, "fit_verified_generation_bundle_v024", lambda _: object()
    )
    monkeypatch.setattr(
        runner,
        "write_verified_fit_result_v024",
        lambda **kwargs: events.append("write"),
    )
    monkeypatch.setattr(
        runner,
        "validate_whole_bundle_from_root",
        lambda root, contract: observed.append(contract)
        or events.append("whole")
        or object(),
    )
    monkeypatch.setattr(
        runner,
        "commit_validated_fit_result_v024",
        lambda **kwargs: events.append("commit") or "d" * 64,
    )
    monkeypatch.setattr(
        runner, "validate_formal_exposure_log", lambda *args: {FIXTURE_ID: object()}
    )
    monkeypatch.setattr(
        runner, "verify_phase_artifact_commitment", lambda *args, **kwargs: None
    )

    runner._fit_structure_stage(
        paths=paths,
        identity=identity,
        view=view,
        truth_hash="0" * 64,
    )
    assert observed == [view]
    assert events == ["phase:started", "write", "whole", "commit", "phase:completed"]


def test_member_fit_contract_error_has_a_typed_integrity_disposition() -> None:
    classified = classify_terminal_exception(
        V024MemberFitNumericContractError("fixture details remain private")
    )
    assert classified.disposition is TerminalDisposition.INTEGRITY_FAILURE
    assert classified.mode is ClassificationMode.PROVEN_INTEGRITY
    assert (
        classified.reason
        is TerminalReason.INTEGRITY_MEMBER_FIT_NUMERIC_CONTRACT_MISMATCH
    )


def test_valid_fit_commitment_is_unique_and_registered_later(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, commitment = _exercise_fit_stage(tmp_path, monkeypatch, fail_at=None)
    assert events == [
        "phase:started",
        "load",
        "fit",
        "write_and_readback",
        "whole_validate",
        "commit",
        "phase:completed",
        "verify",
    ]
    assert commitment.is_file()
    assert events.count("commit") == 1


@pytest.mark.parametrize("fail_at", ["write_readback", "whole_validate", "commit"])
def test_precommit_failures_leave_partial_evidence_without_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
) -> None:
    events, commitment = _exercise_fit_stage(tmp_path, monkeypatch, fail_at=fail_at)
    assert not commitment.exists()
    assert "phase:completed" not in events
    assert events[-1] == "failure"


def test_completed_phase_append_failure_leaves_unregistered_partial_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, commitment = _exercise_fit_stage(
        tmp_path,
        monkeypatch,
        fail_at="phase_append",
    )
    assert commitment.is_file()
    assert "phase:completed" not in events
    assert events[-1] == "failure"
    # The file is evidence of the attempted phase; without a completed ledger
    # event it is not a registered fit commitment.
    registered_fit_commitment = None
    assert registered_fit_commitment is None
