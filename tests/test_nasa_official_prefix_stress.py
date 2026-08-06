from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import lifetwin.experiments.nasa_official_prefix_stress as nasa_stress
from lifetwin.experiments.nasa_official_prefix_stress import (
    NasaOfficialPrefixStressError,
    canonical_json_sha256,
    ensure_execution_authorized,
    execute_score_once,
    load_nasa_official_prefix_stress_config,
    predict_prefix_baselines,
    prepare_prefix_and_future_labels,
    score_prefix_baselines,
    validate_nasa_official_prefix_stress_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/nasa_pcoe_official_bundle_prefix_stress_v1.json"
)


@pytest.fixture
def config() -> dict[str, object]:
    return load_nasa_official_prefix_stress_config(CONFIG_PATH)


@pytest.fixture
def algorithm_config(
    config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    monkeypatch.setattr(nasa_stress, "ensure_execution_authorized", lambda _config: None)
    return config


def _cycles() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for physical_id, shift, bundle in (
        ("B0005", 0.0, "bundle_1"),
        ("B0049", 0.4, "bundle_5"),
    ):
        for cycle_index in range(1, 151):
            retention = 1.0 - 0.0008 * cycle_index - 0.004 * np.sqrt(cycle_index)
            rows.append(
                {
                    "physical_battery_id": physical_id,
                    "cycle_index": cycle_index,
                    "capacity_ah": (2.0 + shift) * retention,
                    "source_bundle": bundle,
                    "experiment_batch": bundle,
                    "temperature_c": 24.0,
                    "discharge_cutoff_v": 2.7,
                    "dynamic_condition": "constant_current",
                }
            )
    return pd.DataFrame(rows)


def test_rights_gate_blocks_official_outcome_execution(config: dict[str, object]) -> None:
    with pytest.raises(NasaOfficialPrefixStressError, match="rights review"):
        ensure_execution_authorized(config)


def test_formal_library_entry_points_all_block_before_input_processing(
    config: dict[str, object], tmp_path: Path
) -> None:
    sentinel = object()
    with pytest.raises(NasaOfficialPrefixStressError, match="rights review"):
        prepare_prefix_and_future_labels(sentinel, config)
    with pytest.raises(NasaOfficialPrefixStressError, match="rights review"):
        predict_prefix_baselines(sentinel, config)
    with pytest.raises(NasaOfficialPrefixStressError, match="rights review"):
        score_prefix_baselines(sentinel, sentinel, {}, config)
    with pytest.raises(NasaOfficialPrefixStressError, match="rights review"):
        execute_score_once(sentinel, sentinel, {}, config, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_production_test_authorization_bypasses_do_not_exist(
    config: dict[str, object],
) -> None:
    assert not hasattr(nasa_stress, "synthetic_test_authorized_config")
    attacked = copy.deepcopy(config)
    attacked["semantic_sha256"] = None
    attacked["synthetic_test_authorization_only"] = True
    attacked["rights_gate"]["execution_allowed"] = True
    with pytest.raises(NasaOfficialPrefixStressError, match="rights review"):
        ensure_execution_authorized(attacked)


def test_deep_copy_cannot_enable_execution_without_license(
    config: dict[str, object],
) -> None:
    attacked = copy.deepcopy(config)
    attacked["semantic_sha256"] = None
    attacked["rights_gate"]["execution_allowed"] = True
    with pytest.raises(NasaOfficialPrefixStressError, match="rights review"):
        ensure_execution_authorized(attacked)


def test_formal_config_semantic_hash_is_unchanged(config: dict[str, object]) -> None:
    assert config["semantic_sha256"] == (
        "037d6d090cf42714ff638d876fdbe7149d60da48e33fd847351bf8597ed9ce9a"
    )
    assert canonical_json_sha256(config) == config["semantic_sha256"]


def test_cli_commands_block_before_reading_input_files(
    tmp_path: Path,
) -> None:
    script = PROJECT_ROOT / "scripts/run_nasa_pcoe_official_prefix_stress.py"
    missing = tmp_path / "must-not-be-opened.csv"
    commands = [
        ["prepare", str(missing), str(tmp_path / "prepare-output")],
        ["predict", str(missing), str(tmp_path / "predict-output")],
        [
            "score",
            str(missing),
            str(missing),
            str(tmp_path / "missing-manifest.json"),
            str(tmp_path / "score-output"),
        ],
    ]
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    for arguments in commands:
        completed = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 1
        payload = json.loads(completed.stdout)
        assert payload["status"] == "blocked"
        assert "rights review" in payload["reason"]


def test_illegal_nasa_chemistry_claim_is_rejected(config: dict[str, object]) -> None:
    attacked = copy.deepcopy(config)
    attacked["dataset"]["chemistry"] = "LFP"
    with pytest.raises(NasaOfficialPrefixStressError, match="chemistry claim"):
        validate_nasa_official_prefix_stress_config(attacked)


def test_physical_battery_partitions_are_disjoint(config: dict[str, object]) -> None:
    groups = [set(config["partitions"][name]) for name in ("training", "validation", "locked_test")]
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
    assert {"B0025", "B0026", "B0027", "B0028"} <= groups[1]

    attacked = copy.deepcopy(config)
    attacked["partitions"]["locked_test"].append("B0025")
    with pytest.raises(NasaOfficialPrefixStressError, match="crosses partitions"):
        validate_nasa_official_prefix_stress_config(attacked)


def test_future_suffix_mutation_changes_scores_not_predictions(
    algorithm_config: dict[str, object],
    tmp_path: Path,
) -> None:
    config = algorithm_config
    cycles = _cycles()
    prefixes, labels, _ = prepare_prefix_and_future_labels(cycles, config)
    predictions, manifest = predict_prefix_baselines(prefixes, config)
    scores, summary, _ = execute_score_once(
        labels, predictions, manifest, config, tmp_path / "first-score"
    )

    attacked_cycles = cycles.copy()
    suffix = (attacked_cycles["physical_battery_id"] == "B0049") & (
        attacked_cycles["cycle_index"] > 100
    )
    attacked_cycles.loc[suffix, "capacity_ah"] *= 0.85
    attacked_prefixes, attacked_labels, _ = prepare_prefix_and_future_labels(
        attacked_cycles, config
    )
    attacked_predictions, attacked_manifest = predict_prefix_baselines(
        attacked_prefixes, config
    )
    attacked_scores, attacked_summary, _ = execute_score_once(
        attacked_labels,
        attacked_predictions,
        attacked_manifest,
        config,
        tmp_path / "mutated-score",
    )
    pd.testing.assert_frame_equal(predictions, attacked_predictions)
    assert manifest == attacked_manifest
    assert not scores.equals(attacked_scores)
    assert summary["score_sha256"] != attacked_summary["score_sha256"]


def test_prediction_and_manifest_are_row_order_invariant(
    algorithm_config: dict[str, object],
) -> None:
    config = algorithm_config
    prefixes, _, _ = prepare_prefix_and_future_labels(_cycles(), config)
    first, first_manifest = predict_prefix_baselines(prefixes, config)
    shuffled = prefixes.sample(frac=1.0, random_state=17).reset_index(drop=True)
    second, second_manifest = predict_prefix_baselines(shuffled, config)
    pd.testing.assert_frame_equal(first, second)
    assert first_manifest == second_manifest


def test_predict_rejects_future_label_columns(
    algorithm_config: dict[str, object],
) -> None:
    config = algorithm_config
    prefixes, _, _ = prepare_prefix_and_future_labels(_cycles(), config)
    attacked = prefixes.copy()
    attacked["future_capacity_label"] = 0.0
    with pytest.raises(NasaOfficialPrefixStressError, match="future-label columns"):
        predict_prefix_baselines(attacked, config)


def test_scorer_requires_preexisting_prediction_manifest(
    algorithm_config: dict[str, object],
    tmp_path: Path,
) -> None:
    config = algorithm_config
    prefixes, labels, _ = prepare_prefix_and_future_labels(_cycles(), config)
    predictions, manifest = predict_prefix_baselines(prefixes, config)
    assert manifest["score_executed"] is False
    assert manifest["prediction_sha256"]
    attacked = predictions.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] += 0.01
    with pytest.raises(NasaOfficialPrefixStressError, match="hash mismatch"):
        execute_score_once(labels, attacked, manifest, config, tmp_path / "failed")
    receipt = tmp_path / "failed" / "score_attempt_receipt.jsonl"
    assert receipt.is_file()
    events = [json.loads(line) for line in receipt.read_text("utf-8").splitlines()]
    assert events[-1]["status"] == "failed"


def test_primary_metric_is_locked_test_equal_cell_equal_prefix(
    algorithm_config: dict[str, object],
    tmp_path: Path,
) -> None:
    config = algorithm_config
    prefixes, labels, _ = prepare_prefix_and_future_labels(_cycles(), config)
    predictions, manifest = predict_prefix_baselines(prefixes, config)
    scores, summary, receipt = execute_score_once(
        labels, predictions, manifest, config, tmp_path / "score"
    )
    assert receipt["status"] == "completed"
    assert set(scores["model_id"]) == {
        "target_prefix_persistence",
        "nonpositive_linear_trend",
        "constrained_sqrt_loss_trend",
    }
    assert summary["primary_scope"] == "locked_test"
    assert summary["primary_metric"] == (
        "trajectory_mae_pp_equal_cell_then_equal_prefix"
    )
    assert summary["prediction_interval_coverage_fraction"] is None
    assert summary["inference"] == "descriptive_only_no_significance_test"


def test_single_score_attempt_and_output_overwrite_are_rejected(
    algorithm_config: dict[str, object],
    tmp_path: Path,
) -> None:
    prefixes, labels, _ = prepare_prefix_and_future_labels(
        _cycles(), algorithm_config
    )
    predictions, manifest = predict_prefix_baselines(prefixes, algorithm_config)
    execute_score_once(
        labels, predictions, manifest, algorithm_config, tmp_path / "first"
    )
    with pytest.raises(NasaOfficialPrefixStressError, match="Duplicate"):
        execute_score_once(
            labels, predictions, manifest, algorithm_config, tmp_path / "second"
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(NasaOfficialPrefixStressError, match="already exists"):
        execute_score_once(
            labels, predictions, manifest, algorithm_config, existing
        )


@pytest.mark.parametrize("replacement", [None, 0, 2])
def test_locked_test_score_limit_is_frozen(
    config: dict[str, object],
    replacement: int | None,
) -> None:
    attacked = copy.deepcopy(config)
    attacked["semantic_sha256"] = None
    if replacement is None:
        del attacked["partitions"]["locked_test_score_limit"]
    else:
        attacked["partitions"]["locked_test_score_limit"] = replacement
    with pytest.raises(NasaOfficialPrefixStressError, match="score limit"):
        validate_nasa_official_prefix_stress_config(attacked)
