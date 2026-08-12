from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lifetwin.experiments import calendar_long_horizon_v020_runner as runner
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractError,
    load_v024_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
    INPUT_FILENAMES_BY_STAGE,
    V020CheckpointRegistryError,
)
from lifetwin.experiments.calendar_long_horizon_v020_contract import (
    V025ContractError,
    load_v025_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v020_environment import (
    verify_formal_environment_v025,
    verify_prediction_environment_v025,
)
from lifetwin.experiments.calendar_long_horizon_v020_protocol import (
    V025_DESIGN_FREEZE_COMMIT,
    V025_EXPECTED_SEED_ROOTS,
    V025_ONLY_ATTEMPT_ID,
    V025_PROTOCOL_ID,
    V025ProtocolError,
    load_v025_design,
)
from lifetwin.experiments.calendar_long_horizon_v019_terminal import (
    ClassificationMode,
    TerminalDisposition,
    TerminalReason,
    classify_terminal_exception,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE_RECORD = (
    ROOT / "reports/synthetic_long_horizon_identifiability_freeze_record_v2_5.json"
)


def _rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_v025_design_and_contract_load_without_consuming_rng() -> None:
    before = np.random.get_state()
    design = load_v025_design()
    view = load_v025_contract_view()
    after = np.random.get_state()

    assert _rng_equal(before, after)
    assert design.protocol_id == V025_PROTOCOL_ID
    assert design.raw["attempt_registry"]["only_attempt_id"] == V025_ONLY_ATTEMPT_ID
    assert (
        design.raw["base_contract"]["generic_core_commit"] != V025_DESIGN_FREEZE_COMMIT
    )
    assert view.protocol.protocol_id == V025_PROTOCOL_ID
    assert view.protocol.seed_roots == tuple(V025_EXPECTED_SEED_ROOTS.items())
    assert view.design_status == "implementation_frozen"


def test_v025_only_adapts_authenticated_identity_fields() -> None:
    base = load_v024_contract_view()
    view = load_v025_contract_view()
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


def test_v025_loader_rejects_any_config_byte_or_identity_drift(tmp_path: Path) -> None:
    source = load_v025_design().config_path
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(V025ProtocolError):
        load_v025_design(changed)
    with pytest.raises((V024ContractError, V025ProtocolError, V025ContractError)):
        load_v025_contract_view(changed)


def test_v025_formal_profile_passes_one_fixed_identity_to_generic_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(attempt_id=V025_ONLY_ATTEMPT_ID)

    monkeypatch.setattr(runner, "run_formal_attempt", capture)
    result = runner.run_formal_attempt_v025(
        attempt_id=V025_ONLY_ATTEMPT_ID,
        label_free_root=tmp_path / "label-free",
        sealed_truth_root=tmp_path / "sealed-truth",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
        repo_root=ROOT,
    )
    assert result.attempt_id == V025_ONLY_ATTEMPT_ID
    assert observed["_formal_attempt_id"] == V025_ONLY_ATTEMPT_ID
    assert observed["_contract_view"].protocol.protocol_id == V025_PROTOCOL_ID
    assert observed["_environment_verifier"] is verify_formal_environment_v025
    assert observed["_input_filenames_by_stage"] is INPUT_FILENAMES_BY_STAGE
    assert Path(observed["_formal_script"]).name == (
        "run_calendar_long_horizon_v020.py"
    )
    assert not any(tmp_path.iterdir())


def test_v025_generation_and_prediction_profiles_bind_attesters(
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
    runner.run_isolated_generation_stage_v025(
        label_free_root=tmp_path / "label-free",
        sealed_truth_root=tmp_path / "sealed-truth",
    )
    runner.run_isolated_prediction_process_v025(
        label_free_root=tmp_path / "label-free",
        attempt_id=V025_ONLY_ATTEMPT_ID,
        repo_root=ROOT,
    )

    assert generation["_contract_view"].protocol.protocol_id == V025_PROTOCOL_ID
    assert generation["_environment_verifier"] is verify_formal_environment_v025
    assert prediction["_environment_verifier"] is (verify_prediction_environment_v025)
    assert prediction["_input_filenames_by_stage"] is INPUT_FILENAMES_BY_STAGE
    assert not any(tmp_path.iterdir())


def test_v025_profile_rejects_other_attempt_before_prediction_capsule(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fixed V2.5 identity"):
        runner.run_isolated_prediction_process_v025(
            label_free_root=tmp_path,
            attempt_id="v025-formal-20260812-a2",
            repo_root=ROOT,
        )


def test_v025_formal_roots_remain_absent_before_authorization() -> None:
    isolation = load_v025_design().raw["path_isolation"]
    assert isinstance(isolation, Mapping)
    for role in ("label_free", "sealed_truth", "score", "termination"):
        assert not (ROOT / isolation[f"{role}_root"]).exists()


def test_v025_cli_exposes_no_scientific_or_identity_override() -> None:
    text = (ROOT / "scripts/run_calendar_long_horizon_v020.py").read_text("utf-8")
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


def test_v025_seed_registry_is_unique_and_declared_exactly() -> None:
    payload = json.loads(load_v025_design().config_path.read_text("utf-8"))
    roots = payload["fresh_generation"]["seed_roots"]
    assert roots == dict(V025_EXPECTED_SEED_ROOTS)
    assert len(set(roots.values())) == 13


def test_v025_checkpoint_registry_drift_is_a_proven_integrity_void() -> None:
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
def test_v025_formal_and_prediction_attesters_bind_the_same_freeze() -> None:
    view = load_v025_contract_view()
    formal = verify_formal_environment_v025(ROOT, view)
    prediction = verify_prediction_environment_v025(ROOT)
    assert formal.git_commit == prediction.git_commit
    assert formal.protocol_id == prediction.protocol_id == V025_PROTOCOL_ID
    assert formal.config_byte_sha256 == prediction.config_byte_sha256
    assert formal.git_dirty is False
