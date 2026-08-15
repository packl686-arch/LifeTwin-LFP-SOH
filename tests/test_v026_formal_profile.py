from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lifetwin.experiments import calendar_long_horizon_v021_runner as runner
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractError,
)
from lifetwin.experiments import calendar_long_horizon_v019_partition as partition
from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
    INPUT_FILENAMES_BY_STAGE,
    V020CheckpointRegistryError,
)
from lifetwin.experiments.calendar_long_horizon_v020_contract import (
    load_v025_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v021_contract import (
    V026ContractError,
    load_v026_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v021_environment import (
    verify_formal_environment_v026,
    verify_prediction_environment_v026,
)
from lifetwin.experiments.calendar_long_horizon_v021_protocol import (
    V026_DESIGN_FREEZE_COMMIT,
    V026_EXPECTED_SEED_ROOTS,
    V026_ONLY_ATTEMPT_ID,
    V026_PROTOCOL_ID,
    V026ProtocolError,
    load_v026_design,
)
from lifetwin.experiments.calendar_long_horizon_v019_terminal import (
    ClassificationMode,
    TerminalDisposition,
    TerminalReason,
    classify_terminal_exception,
)
from scripts.verify_historical_freezes import current_checkout_is_freeze


ROOT = Path(__file__).resolve().parents[1]
FREEZE_RECORD = (
    ROOT / "reports/synthetic_long_horizon_identifiability_freeze_record_v2_6.json"
)


def _rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_v026_design_and_contract_load_without_consuming_rng() -> None:
    before = np.random.get_state()
    design = load_v026_design()
    view = load_v026_contract_view()
    after = np.random.get_state()

    assert _rng_equal(before, after)
    assert design.protocol_id == V026_PROTOCOL_ID
    assert design.raw["attempt_registry"]["only_attempt_id"] == V026_ONLY_ATTEMPT_ID
    assert design.raw["base_contract"]["fixed_core_commit"] != V026_DESIGN_FREEZE_COMMIT
    assert view.protocol.protocol_id == V026_PROTOCOL_ID
    assert view.protocol.seed_roots == tuple(V026_EXPECTED_SEED_ROOTS.items())
    assert view.design_status == "implementation_frozen"


def test_v026_only_adapts_authenticated_identity_fields() -> None:
    base = load_v025_contract_view()
    view = load_v026_contract_view()
    restored = replace(
        view.protocol,
        protocol_id=base.protocol.protocol_id,
        config_sha256=base.protocol.config_sha256,
        seed_roots=base.protocol.seed_roots,
        config_json=base.protocol.config_json,
    )
    restored_artifacts = replace(
        view.artifacts,
        protocol_id=base.artifacts.protocol_id,
        config_path=base.artifacts.config_path,
        config_byte_sha256=base.artifacts.config_byte_sha256,
    )
    assert restored == base.protocol
    assert restored_artifacts == base.artifacts
    assert view.whole_rows == base.whole_rows
    assert view.partition_rows == base.partition_rows


def test_v026_loader_rejects_any_config_byte_or_identity_drift(tmp_path: Path) -> None:
    source = load_v026_design().config_path
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(V026ProtocolError):
        load_v026_design(changed)
    with pytest.raises((V024ContractError, V026ProtocolError, V026ContractError)):
        load_v026_contract_view(changed)


def test_v026_authenticated_view_reaches_whole_validator_and_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    view = load_v026_contract_view()
    missing_root = tmp_path / "fresh-label-free-root"
    with pytest.raises(partition.V024WholeBundleContractError, match="physical"):
        partition.validate_whole_bundle_from_root(missing_root, view)
    with pytest.raises(partition.V024PartitionCapabilityError, match="invalid"):
        partition.validate_whole_bundle_from_root(missing_root, view.artifacts)


def test_v026_formal_profile_passes_one_fixed_identity_to_generic_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(attempt_id=V026_ONLY_ATTEMPT_ID)

    monkeypatch.setattr(runner, "run_formal_attempt", capture)
    result = runner.run_formal_attempt_v026(
        attempt_id=V026_ONLY_ATTEMPT_ID,
        label_free_root=tmp_path / "label-free",
        sealed_truth_root=tmp_path / "sealed-truth",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
        repo_root=ROOT,
    )
    assert result.attempt_id == V026_ONLY_ATTEMPT_ID
    assert observed["_formal_attempt_id"] == V026_ONLY_ATTEMPT_ID
    assert observed["_contract_view"].protocol.protocol_id == V026_PROTOCOL_ID
    assert observed["_environment_verifier"] is verify_formal_environment_v026
    assert observed["_input_filenames_by_stage"] is INPUT_FILENAMES_BY_STAGE
    assert Path(observed["_formal_script"]).name == (
        "run_calendar_long_horizon_v021.py"
    )
    assert not any(tmp_path.iterdir())


def test_v026_generation_and_prediction_profiles_bind_attesters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation: dict[str, object] = {}
    prediction: dict[str, object] = {}

    monkeypatch.setattr(
        runner,
        "run_isolated_generation_stage",
        lambda **kwargs: generation.update(kwargs),
    )
    monkeypatch.setattr(
        runner,
        "run_isolated_prediction_process_v024",
        lambda **kwargs: prediction.update(kwargs) or SimpleNamespace(artifacts={}),
    )
    runner.run_isolated_generation_stage_v026(
        label_free_root=tmp_path / "label-free",
        sealed_truth_root=tmp_path / "sealed-truth",
    )
    runner.run_isolated_prediction_process_v026(
        label_free_root=tmp_path / "label-free",
        attempt_id=V026_ONLY_ATTEMPT_ID,
        repo_root=ROOT,
    )

    assert generation["_contract_view"].protocol.protocol_id == V026_PROTOCOL_ID
    assert generation["_environment_verifier"] is verify_formal_environment_v026
    assert prediction["_environment_verifier"] is (verify_prediction_environment_v026)
    assert prediction["_input_filenames_by_stage"] is INPUT_FILENAMES_BY_STAGE
    assert not any(tmp_path.iterdir())


def test_v026_profile_rejects_other_attempt_before_prediction_capsule(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fixed V2.6 identity"):
        runner.run_isolated_prediction_process_v026(
            label_free_root=tmp_path,
            attempt_id="v026-formal-20260812-a2",
            repo_root=ROOT,
        )


def test_v026_formal_roots_remain_absent_before_authorization() -> None:
    isolation = load_v026_design().raw["path_isolation"]
    assert isinstance(isolation, Mapping)
    for role in ("label_free", "sealed_truth", "score", "termination"):
        assert not (ROOT / isolation[f"{role}_root"]).exists()


def test_v026_cli_exposes_no_scientific_or_identity_override() -> None:
    text = (ROOT / "scripts/run_calendar_long_horizon_v021.py").read_text("utf-8")
    forbidden = (
        "--protocol-id",
        "--config",
        "--seed",
        "--threshold",
        "--partition",
        "--success-condition",
        "--checkpoint-registry",
    )
    assert all(flag not in text for flag in forbidden)


def test_v026_cli_rejects_a2_before_creating_a_root(tmp_path: Path) -> None:
    label_free = tmp_path / "label-free"
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/run_calendar_long_horizon_v021.py"),
            "--attempt-id",
            "v026-formal-20260812-a2",
            "--label-free-root",
            str(label_free),
            "--sealed-truth-root",
            str(tmp_path / "sealed-truth"),
            "--score-root",
            str(tmp_path / "score"),
            "--termination-root",
            str(tmp_path / "termination"),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "attempt ID must equal v026-formal-20260812-a1" in completed.stderr
    assert not any(tmp_path.iterdir())


def test_v026_seed_registry_is_unique_and_declared_exactly() -> None:
    payload = json.loads(load_v026_design().config_path.read_text("utf-8"))
    roots = payload["fresh_generation"]["seed_roots"]
    assert roots == dict(V026_EXPECTED_SEED_ROOTS)
    assert len(set(roots.values())) == 13


def test_v026_checkpoint_registry_drift_is_a_proven_integrity_void() -> None:
    try:
        try:
            raise V020CheckpointRegistryError("fixture")
        except V020CheckpointRegistryError as cause:
            raise RuntimeError("boundary") from cause
    except RuntimeError as error:
        classified = classify_terminal_exception(error)
    assert classified.disposition is TerminalDisposition.INTEGRITY_FAILURE
    assert classified.mode is ClassificationMode.PROVEN_INTEGRITY
    assert classified.reason is TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH


@pytest.mark.skipif(not FREEZE_RECORD.is_file(), reason="freeze commit not created")
def test_v026_formal_and_prediction_attesters_bind_or_reject_head() -> None:
    view = load_v026_contract_view()
    if not current_checkout_is_freeze(ROOT, FREEZE_RECORD):
        with pytest.raises(RuntimeError):
            verify_formal_environment_v026(ROOT, view)
        with pytest.raises(RuntimeError):
            verify_prediction_environment_v026(ROOT)
        return
    formal = verify_formal_environment_v026(ROOT, view)
    prediction = verify_prediction_environment_v026(ROOT)
    assert formal.git_commit == prediction.git_commit
    assert formal.protocol_id == prediction.protocol_id == V026_PROTOCOL_ID
    assert formal.config_byte_sha256 == prediction.config_byte_sha256
    assert formal.git_dirty is False
