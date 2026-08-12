from __future__ import annotations

from pathlib import Path

import pytest

from lifetwin.experiments import calendar_long_horizon_v019_firewall as firewall_v019
from lifetwin.experiments import calendar_long_horizon_v019_io as io_v019
from lifetwin.experiments import calendar_long_horizon_v019_runner as runner_v019
from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
    CALIBRATION_INPUT_FILENAMES,
    CENTER_INPUT_FILENAMES,
    INPUT_FILENAMES_BY_STAGE,
    REVEAL_PREREQUISITES,
    RISK_INPUT_FILENAMES,
    V020CheckpointRegistryError,
    registered_input_hashes_v020,
    verify_registered_input_hashes_v020,
)


_MISSING_V019_FIREWALL_INPUTS = {
    "actual_analysis_hash_ledger_commitment.json",
    "generation_plan_commitment.json",
    "truth_commitments.json",
}


def _write_registered_inputs(tmp_path: Path) -> tuple[Path, Path]:
    label_root = tmp_path / "label-free"
    sealed_root = tmp_path / "sealed"
    label_root.mkdir()
    sealed_root.mkdir()
    filenames = set().union(*map(set, INPUT_FILENAMES_BY_STAGE.values()))
    for index, filename in enumerate(sorted(filenames)):
        root = sealed_root if filename.endswith("_truth.csv") else label_root
        (root / filename).write_bytes(f"fixture/{index}/{filename}\n".encode())
    return label_root, sealed_root


def test_v019_audit_finds_center_and_risk_registry_drift_only_in_allowlists() -> None:
    common = set(runner_v019._COMMON_TRAINING_INPUTS)
    center_producer = common | {"center_development_truth.csv"}
    risk_producer = common | {
        "center_state_checkpoint.json",
        "risk_development_truth.csv",
    }
    assert center_producer - set(firewall_v019._CENTER_INPUT_FILENAMES) == (
        _MISSING_V019_FIREWALL_INPUTS
    )
    assert risk_producer - set(firewall_v019._RISK_INPUT_FILENAMES) == (
        _MISSING_V019_FIREWALL_INPUTS
    )
    assert not set(firewall_v019._CENTER_INPUT_FILENAMES) - center_producer
    assert not set(firewall_v019._RISK_INPUT_FILENAMES) - risk_producer


@pytest.mark.parametrize(
    ("stage", "v019_allowlist"),
    [
        ("center_development", firewall_v019._CENTER_INPUT_FILENAMES),
        ("risk_development", firewall_v019._RISK_INPUT_FILENAMES),
    ],
)
def test_synthetic_v019_drift_is_reproduced_and_v020_passes(
    tmp_path: Path,
    stage: str,
    v019_allowlist: tuple[str, ...],
) -> None:
    label_root, sealed_root = _write_registered_inputs(tmp_path)
    produced = registered_input_hashes_v020(
        stage,
        label_root=label_root,
        sealed_root=sealed_root,
    )
    with pytest.raises(firewall_v019.V024FirewallError, match="registry changed"):
        firewall_v019._verify_input_hashes(
            produced,
            expected_filenames=v019_allowlist,
            label_root=label_root,
            sealed_root=sealed_root,
            context=f"synthetic {stage}",
        )
    assert (
        verify_registered_input_hashes_v020(
            stage,
            produced,
            label_root=label_root,
            sealed_root=sealed_root,
        )
        == produced
    )


def test_v020_registry_matrix_covers_all_downstream_bindings() -> None:
    assert len(CENTER_INPUT_FILENAMES) == 10
    assert len(RISK_INPUT_FILENAMES) == 11
    assert len(CALIBRATION_INPUT_FILENAMES) == 14
    assert INPUT_FILENAMES_BY_STAGE == {
        "center_development": CENTER_INPUT_FILENAMES,
        "risk_development": RISK_INPUT_FILENAMES,
        "calibration": CALIBRATION_INPUT_FILENAMES,
    }
    assert "calibration_mask_commitment.json" in CALIBRATION_INPUT_FILENAMES
    assert runner_v019._MODEL_STATE_COMMITMENT_FILES == (
        io_v019._MODEL_STATE_COMMITMENT_FILES
    )
    assert set(REVEAL_PREREQUISITES) == {
        "risk_truth_opened",
        "calibration_truth_opened",
        "scoring_truth_opened",
    }
    assert "center_state_checkpoint.json" in REVEAL_PREREQUISITES["risk_truth_opened"]
    assert (
        "risk_state_checkpoint.json" in REVEAL_PREREQUISITES["calibration_truth_opened"]
    )
    assert "prediction_commitment.json" in REVEAL_PREREQUISITES["scoring_truth_opened"]


def test_legal_center_registry_passes(tmp_path: Path) -> None:
    label_root, sealed_root = _write_registered_inputs(tmp_path)
    committed = registered_input_hashes_v020(
        "center_development",
        label_root=label_root,
        sealed_root=sealed_root,
    )
    assert set(committed) == set(CENTER_INPUT_FILENAMES)
    assert (
        verify_registered_input_hashes_v020(
            "center_development",
            committed,
            label_root=label_root,
            sealed_root=sealed_root,
        )
        == committed
    )


@pytest.mark.parametrize("missing", CENTER_INPUT_FILENAMES)
def test_every_required_center_key_is_fail_closed(
    tmp_path: Path,
    missing: str,
) -> None:
    label_root, sealed_root = _write_registered_inputs(tmp_path)
    committed = registered_input_hashes_v020(
        "center_development",
        label_root=label_root,
        sealed_root=sealed_root,
    )
    del committed[missing]
    with pytest.raises(V020CheckpointRegistryError, match="registry changed"):
        verify_registered_input_hashes_v020(
            "center_development",
            committed,
            label_root=label_root,
            sealed_root=sealed_root,
        )


@pytest.mark.parametrize("mutation", ["unknown", "renamed"])
def test_unknown_or_renamed_center_key_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    label_root, sealed_root = _write_registered_inputs(tmp_path)
    committed = registered_input_hashes_v020(
        "center_development",
        label_root=label_root,
        sealed_root=sealed_root,
    )
    if mutation == "unknown":
        committed["unexpected.json"] = "0" * 64
    else:
        digest = committed.pop("truth_commitments.json")
        committed["truth_commitment.json"] = digest
    with pytest.raises(V020CheckpointRegistryError, match="registry changed"):
        verify_registered_input_hashes_v020(
            "center_development",
            committed,
            label_root=label_root,
            sealed_root=sealed_root,
        )


@pytest.mark.parametrize("filename", CENTER_INPUT_FILENAMES)
def test_every_registered_byte_change_is_rejected(
    tmp_path: Path,
    filename: str,
) -> None:
    label_root, sealed_root = _write_registered_inputs(tmp_path)
    committed = registered_input_hashes_v020(
        "center_development",
        label_root=label_root,
        sealed_root=sealed_root,
    )
    root = sealed_root if filename.endswith("_truth.csv") else label_root
    (root / filename).write_bytes(b"changed\n")
    with pytest.raises(V020CheckpointRegistryError, match="input changed"):
        verify_registered_input_hashes_v020(
            "center_development",
            committed,
            label_root=label_root,
            sealed_root=sealed_root,
        )


def test_center_checkpoint_precedes_single_risk_truth_reveal(tmp_path: Path) -> None:
    label_root, sealed_root = _write_registered_inputs(tmp_path)
    committed = registered_input_hashes_v020(
        "center_development",
        label_root=label_root,
        sealed_root=sealed_root,
    )
    opened: list[str] = []

    def open_risk_truth() -> bytes:
        opened.append("risk_development_truth.csv")
        return (sealed_root / opened[-1]).read_bytes()

    verify_registered_input_hashes_v020(
        "center_development",
        committed,
        label_root=label_root,
        sealed_root=sealed_root,
    )
    assert open_risk_truth()
    assert opened == ["risk_development_truth.csv"]


@pytest.mark.parametrize("stage", tuple(INPUT_FILENAMES_BY_STAGE))
def test_each_stage_producer_and_consumer_share_one_registry(
    tmp_path: Path,
    stage: str,
) -> None:
    label_root, sealed_root = _write_registered_inputs(tmp_path)
    produced = registered_input_hashes_v020(
        stage,
        label_root=label_root,
        sealed_root=sealed_root,
    )
    consumed = verify_registered_input_hashes_v020(
        stage,
        produced,
        label_root=label_root,
        sealed_root=sealed_root,
    )
    assert tuple(produced) == INPUT_FILENAMES_BY_STAGE[stage]
    assert consumed == produced
