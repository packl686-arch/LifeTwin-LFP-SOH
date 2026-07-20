from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import pytest

from lifetwin.data.geisbauer_calendar import (
    GEISBAUER_CALENDAR_OBSERVATIONS_SHA256,
)
from lifetwin.experiments.calendar_landmark_readiness import (
    EXPECTED_CANONICAL_OUTCOME_SHA256,
)
from scripts import run_calendar_v4_hybrid_development as v4_runner
from scripts import run_geisbauer_external_stress as external_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAUMANN_PATH = PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv"
V4_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/naumann_calendar_v4_hybrid_development.json"
)
GEISBAUER_PATH = PROJECT_ROOT / "data/external/geisbauer_2022/LFP_Data.csv"
GEISBAUER_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs/experiments/geisbauer_lfp_calendar_external_stress.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_nonfinite_json(value: str) -> None:
    raise AssertionError(f"non-finite JSON token emitted: {value}")


def _scratch_root() -> Path:
    base = Path(
        os.environ.get(
            "LIFETWIN_TEST_SCRATCH",
            str(PROJECT_ROOT / "artifacts/test-scratch"),
        )
    )
    root = base / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


@pytest.fixture
def writable_root() -> Path:
    root = _scratch_root()
    try:
        yield root
    finally:
        shutil.rmtree(root)


@pytest.fixture(scope="module")
def runner_outputs() -> dict[str, object]:
    root = _scratch_root()
    v4_dir = root / "v4"
    external_dir = root / "external"
    try:
        v4_result = v4_runner.run(NAUMANN_PATH, V4_CONFIG_PATH, v4_dir)
        external_result = external_runner.run(
            NAUMANN_PATH,
            GEISBAUER_PATH,
            GEISBAUER_PROTOCOL_PATH,
            external_dir,
        )
        yield {
            "v4_dir": v4_dir,
            "v4_result": v4_result,
            "external_dir": external_dir,
            "external_result": external_result,
        }
    finally:
        shutil.rmtree(root)


def test_v4_runner_writes_frozen_auditable_artifacts(
    runner_outputs: dict[str, object],
) -> None:
    artifact_dir = runner_outputs["v4_dir"]
    result = runner_outputs["v4_result"]
    assert isinstance(artifact_dir, Path)
    assert isinstance(result, dict)
    expected_rows = {
        "label_free_predictions": 300,
        "training_residual_crossfit": 175,
        "calibration_condition_scores": 6,
        "calibration_quantiles": 6,
        "condition_metrics": 12,
        "condition_splits": 17,
    }
    assert set(result["artifacts"]) == set(expected_rows)
    for name, expected_count in expected_rows.items():
        metadata = result["artifacts"][name]
        path = Path(metadata["path"])
        assert path.is_file()
        assert metadata["row_count"] == expected_count
        assert metadata["sha256"] == _sha256(path)

    provenance = result["provenance"]
    assert provenance["input_file_sha256"] == v4_runner.EXPECTED_INPUT_SHA256
    assert provenance["canonical_outcome_sha256"] == (
        EXPECTED_CANONICAL_OUTCOME_SHA256
    )
    assert provenance["config_file_sha256"] == _sha256(V4_CONFIG_PATH)
    parsed = json.loads(
        (artifact_dir / "result.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    assert parsed["prediction_pack_sha256"] == result["prediction_pack_sha256"]
    assert parsed["calibration"]["operational_issued_trajectory_count"] == 0
    assert parsed["confirmation"]["15_to_25_year_claim_allowed"] is False


def test_external_runner_writes_frozen_auditable_artifacts(
    runner_outputs: dict[str, object],
) -> None:
    artifact_dir = runner_outputs["external_dir"]
    result = runner_outputs["external_result"]
    assert isinstance(artifact_dir, Path)
    assert isinstance(result, dict)
    expected_rows = {
        "label_free_predictions": 120,
        "cell_metrics": 60,
        "condition_summary": 12,
        "comparison_summary": 4,
    }
    assert set(result["artifacts"]) == set(expected_rows)
    for name, expected_count in expected_rows.items():
        metadata = result["artifacts"][name]
        path = Path(metadata["path"])
        assert path.is_file()
        assert metadata["row_count"] == expected_count
        assert metadata["sha256"] == _sha256(path)

    provenance = result["provenance"]
    assert provenance["source_input_file_sha256"] == (
        external_runner.EXPECTED_SOURCE_INPUT_SHA256
    )
    assert provenance["source_canonical_outcome_sha256"] == (
        EXPECTED_CANONICAL_OUTCOME_SHA256
    )
    assert provenance["target_canonical_observations_sha256"] == (
        GEISBAUER_CALENDAR_OBSERVATIONS_SHA256
    )
    assert provenance["target_adapter_audit"]["canonical_output_sha256"] == (
        GEISBAUER_CALENDAR_OBSERVATIONS_SHA256
    )
    assert provenance["protocol_file_sha256"] == _sha256(
        GEISBAUER_PROTOCOL_PATH
    )
    parsed = json.loads(
        (artifact_dir / "result.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    assert parsed["descriptive_signal_status"] == (
        "primary_candidate_did_not_outperform_comparator"
    )
    assert parsed["decision"][
        "independent_long_term_validation_claim_allowed"
    ] is False


def test_runners_refuse_to_overwrite_any_existing_evidence_artifact(
    runner_outputs: dict[str, object],
) -> None:
    with pytest.raises(FileExistsError, match="never overwrites"):
        v4_runner.run(
            NAUMANN_PATH,
            V4_CONFIG_PATH,
            runner_outputs["v4_dir"],
        )
    with pytest.raises(FileExistsError, match="never overwrites"):
        external_runner.run(
            NAUMANN_PATH,
            GEISBAUER_PATH,
            GEISBAUER_PROTOCOL_PATH,
            runner_outputs["external_dir"],
        )


def test_v4_runner_rejects_changed_input_bytes_before_writing(
    writable_root: Path,
) -> None:
    changed = writable_root / "changed_naumann.csv"
    changed.write_bytes(NAUMANN_PATH.read_bytes() + b"\n")
    artifact_dir = writable_root / "v4_artifacts"
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        v4_runner.run(changed, V4_CONFIG_PATH, artifact_dir)
    assert not artifact_dir.exists()


def test_external_runner_rejects_changed_target_bytes_before_adapter_use(
    writable_root: Path,
) -> None:
    changed = writable_root / "changed_geisbauer.csv"
    changed.write_bytes(GEISBAUER_PATH.read_bytes() + b"\n")
    artifact_dir = writable_root / "external_artifacts"
    with pytest.raises(ValueError, match="target input SHA-256 mismatch"):
        external_runner.run(
            NAUMANN_PATH,
            changed,
            GEISBAUER_PROTOCOL_PATH,
            artifact_dir,
        )
    assert not artifact_dir.exists()


def test_external_runner_rechecks_adapter_canonical_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    writable_root: Path,
) -> None:
    monkeypatch.setattr(
        external_runner,
        "geisbauer_calendar_observations_sha256",
        lambda _observations: "0" * 64,
    )
    artifact_dir = writable_root / "external_artifacts"
    with pytest.raises(ValueError, match="adapter canonical fingerprint mismatch"):
        external_runner.run(
            NAUMANN_PATH,
            GEISBAUER_PATH,
            GEISBAUER_PROTOCOL_PATH,
            artifact_dir,
        )
    assert not artifact_dir.exists()
