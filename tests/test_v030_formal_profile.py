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

from lifetwin.experiments import calendar_long_horizon_v025_runner as runner
from lifetwin.experiments import (
    calendar_long_horizon_v019_prediction_capsule as prediction_capsule,
)
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractError,
)
from lifetwin.experiments import calendar_long_horizon_v019_partition as partition
from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
    INPUT_FILENAMES_BY_STAGE,
    V020CheckpointRegistryError,
)
from lifetwin.experiments.calendar_long_horizon_v024_contract import (
    load_v029_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v025_contract import (
    V030ContractError,
    load_v030_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v025_environment import (
    verify_formal_environment_v030,
    verify_prediction_environment_v030,
)
from lifetwin.experiments.calendar_long_horizon_v025_protocol import (
    V030_DESIGN_FREEZE_COMMIT,
    V030_EXPECTED_SEED_ROOTS,
    V030_FIXED_CORE_COMMIT,
    V030_ONLY_ATTEMPT_ID,
    V030_PROTOCOL_ID,
    V030ProtocolError,
    load_v030_design,
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
    ROOT / "reports/synthetic_long_horizon_identifiability_freeze_record_v2_10.json"
)


def _rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_v030_design_and_contract_load_without_consuming_rng() -> None:
    before = np.random.get_state()
    design = load_v030_design()
    view = load_v030_contract_view()
    after = np.random.get_state()

    assert _rng_equal(before, after)
    assert design.protocol_id == V030_PROTOCOL_ID
    assert design.raw["attempt_registry"]["only_attempt_id"] == V030_ONLY_ATTEMPT_ID
    assert (
        design.raw["base_contract"]["fixed_core_commit"]
        == V030_FIXED_CORE_COMMIT
        != V030_DESIGN_FREEZE_COMMIT
    )
    assert view.protocol.protocol_id == V030_PROTOCOL_ID
    assert view.protocol.seed_roots == tuple(V030_EXPECTED_SEED_ROOTS.items())
    assert view.design_status == "implementation_frozen"


def test_v030_only_adapts_authenticated_identity_fields() -> None:
    base = load_v029_contract_view()
    view = load_v030_contract_view()
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


def test_v030_p_to_i_changes_only_amendment_status() -> None:
    path = "configs/experiments/synthetic_long_horizon_identifiability_v2_10_amendment.json"
    committed = subprocess.run(
        ("git", "show", f"{V030_DESIGN_FREEZE_COMMIT}:{path}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    before = json.loads(committed.stdout)
    after = json.loads((ROOT / path).read_text("utf-8"))
    differences = {key for key in before if before[key] != after[key]}
    assert differences == {"status"}
    assert before["status"] == "preregistered_post_fix_pre_formalization"
    assert after["status"] == "implementation_frozen"


def test_v030_fixed_core_to_preregistration_history_is_linear() -> None:
    parent = subprocess.run(
        ("git", "rev-parse", f"{V030_DESIGN_FREEZE_COMMIT}^"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert parent == V030_FIXED_CORE_COMMIT
    changed = subprocess.run(
        (
            "git",
            "diff",
            "--name-only",
            V030_FIXED_CORE_COMMIT,
            V030_DESIGN_FREEZE_COMMIT,
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    assert set(changed) == {
        "configs/experiments/synthetic_long_horizon_identifiability_v2_10_amendment.json",
        "reports/synthetic_long_horizon_identifiability_prereg_v2_10.md",
        "requirements/v030-formal.txt",
    }


def test_v030_amendment_binds_the_result_blind_handoff_fix() -> None:
    design = load_v030_design()
    handoff = design.raw["prediction_capsule_identity_handoff_fix"]
    assert isinstance(handoff, Mapping)
    assert handoff["required_constructor_keyword"] == "protocol_id"
    assert handoff["development_fix_commit"] == V030_FIXED_CORE_COMMIT
    assert handoff["scientific_outputs_unchanged"] is True
    assert handoff["file_hash_and_membership_checks_unchanged"] is True
    assert handoff["truth_capability_added"] is False


def test_v030_truth_rows_match_the_inherited_artifact_contract() -> None:
    artifacts = load_v030_contract_view().artifacts
    assert {
        filename: prediction_capsule._TRUTH_REQUIRED_ROWS[filename]
        for filename in artifacts.sealed_filenames
    } == {
        filename: artifacts.csv_schema(filename).required_rows
        for filename in artifacts.sealed_filenames
    }
    for filename in artifacts.matched_pair_filenames:
        partition = filename.removesuffix(".csv")
        assert prediction_capsule._TRUTH_REQUIRED_ROWS[filename] == 250
        assert artifacts.partition_member_counts[partition] == 500


def test_v030_loader_rejects_any_config_byte_or_identity_drift(tmp_path: Path) -> None:
    source = load_v030_design().config_path
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(V030ProtocolError):
        load_v030_design(changed)
    with pytest.raises((V024ContractError, V030ProtocolError, V030ContractError)):
        load_v030_contract_view(changed)


def test_v030_authenticated_view_reaches_whole_validator_and_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    view = load_v030_contract_view()
    missing_root = tmp_path / "fresh-label-free-root"
    with pytest.raises(partition.V024WholeBundleContractError, match="physical"):
        partition.validate_whole_bundle_from_root(missing_root, view)
    with pytest.raises(partition.V024PartitionCapabilityError, match="invalid"):
        partition.validate_whole_bundle_from_root(missing_root, view.artifacts)


def test_v030_formal_profile_passes_one_fixed_identity_to_generic_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(attempt_id=V030_ONLY_ATTEMPT_ID)

    monkeypatch.setattr(runner, "run_formal_attempt", capture)
    result = runner.run_formal_attempt_v030(
        attempt_id=V030_ONLY_ATTEMPT_ID,
        label_free_root=tmp_path / "label-free",
        sealed_truth_root=tmp_path / "sealed-truth",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
        repo_root=ROOT,
    )
    assert result.attempt_id == V030_ONLY_ATTEMPT_ID
    assert observed["_formal_attempt_id"] == V030_ONLY_ATTEMPT_ID
    assert observed["_contract_view"].protocol.protocol_id == V030_PROTOCOL_ID
    assert observed["_environment_verifier"] is verify_formal_environment_v030
    assert observed["_input_filenames_by_stage"] is INPUT_FILENAMES_BY_STAGE
    assert Path(observed["_formal_script"]).name == (
        "run_calendar_long_horizon_v025.py"
    )
    assert not any(tmp_path.iterdir())


def test_v030_generation_and_prediction_profiles_bind_attesters(
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
    runner.run_isolated_generation_stage_v030(
        label_free_root=tmp_path / "label-free",
        sealed_truth_root=tmp_path / "sealed-truth",
    )
    runner.run_isolated_prediction_process_v030(
        label_free_root=tmp_path / "label-free",
        attempt_id=V030_ONLY_ATTEMPT_ID,
        repo_root=ROOT,
    )

    assert generation["_contract_view"].protocol.protocol_id == V030_PROTOCOL_ID
    assert generation["_environment_verifier"] is verify_formal_environment_v030
    assert prediction["_environment_verifier"] is (verify_prediction_environment_v030)
    assert prediction["_input_filenames_by_stage"] is INPUT_FILENAMES_BY_STAGE
    assert not any(tmp_path.iterdir())


def test_v030_profile_rejects_other_attempt_before_prediction_capsule(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fixed V2.10 identity"):
        runner.run_isolated_prediction_process_v030(
            label_free_root=tmp_path,
            attempt_id="v030-formal-20260814-a2",
            repo_root=ROOT,
        )


def test_v030_formal_roots_follow_freeze_lifecycle() -> None:
    isolation = load_v030_design().raw["path_isolation"]
    assert isinstance(isolation, Mapping)
    roots = tuple(
        ROOT / isolation[f"{role}_root"]
        for role in ("label_free", "sealed_truth", "score", "termination")
    )
    assert len({path.resolve() for path in roots}) == 4
    if current_checkout_is_freeze(ROOT, FREEZE_RECORD):
        assert not any(path.exists() for path in roots)
    else:
        record = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
        assert record["formal_roots_created_before_implementation_freeze"] is False


def test_v030_cli_exposes_no_scientific_or_identity_override() -> None:
    text = (ROOT / "scripts/run_calendar_long_horizon_v025.py").read_text("utf-8")
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


def test_v030_cli_rejects_a2_before_creating_a_root(tmp_path: Path) -> None:
    label_free = tmp_path / "label-free"
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/run_calendar_long_horizon_v025.py"),
            "--attempt-id",
            "v030-formal-20260814-a2",
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
    assert "attempt ID must equal v030-formal-20260814-a1" in completed.stderr
    assert not any(tmp_path.iterdir())


def test_v030_seed_registry_is_unique_and_declared_exactly() -> None:
    payload = json.loads(load_v030_design().config_path.read_text("utf-8"))
    roots = payload["fresh_generation"]["seed_roots"]
    assert roots == dict(V030_EXPECTED_SEED_ROOTS)
    assert len(set(roots.values())) == 13


def test_v030_checkpoint_registry_drift_is_a_proven_integrity_void() -> None:
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
def test_v030_formal_and_prediction_attesters_bind_or_reject_head() -> None:
    view = load_v030_contract_view()
    if not current_checkout_is_freeze(ROOT, FREEZE_RECORD):
        with pytest.raises(RuntimeError):
            verify_formal_environment_v030(ROOT, view)
        with pytest.raises(RuntimeError):
            verify_prediction_environment_v030(ROOT)
        return
    formal = verify_formal_environment_v030(ROOT, view)
    prediction = verify_prediction_environment_v030(ROOT)
    assert formal.git_commit == prediction.git_commit
    assert formal.protocol_id == prediction.protocol_id == V030_PROTOCOL_ID
    assert formal.config_byte_sha256 == prediction.config_byte_sha256
    assert formal.git_dirty is False
