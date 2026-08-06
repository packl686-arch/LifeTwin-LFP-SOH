from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.nasa_official_prefix_stress import (
    NasaOfficialPrefixStressError,
    ensure_execution_authorized,
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
    config: dict[str, object],
) -> None:
    cycles = _cycles()
    prefixes, labels, _ = prepare_prefix_and_future_labels(cycles, config)
    predictions, manifest = predict_prefix_baselines(prefixes, config)
    scores, summary = score_prefix_baselines(labels, predictions, manifest, config)

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
    attacked_scores, attacked_summary = score_prefix_baselines(
        attacked_labels, attacked_predictions, attacked_manifest, config
    )
    pd.testing.assert_frame_equal(predictions, attacked_predictions)
    assert manifest == attacked_manifest
    assert not scores.equals(attacked_scores)
    assert summary["score_sha256"] != attacked_summary["score_sha256"]


def test_prediction_and_manifest_are_row_order_invariant(
    config: dict[str, object],
) -> None:
    prefixes, _, _ = prepare_prefix_and_future_labels(_cycles(), config)
    first, first_manifest = predict_prefix_baselines(prefixes, config)
    shuffled = prefixes.sample(frac=1.0, random_state=17).reset_index(drop=True)
    second, second_manifest = predict_prefix_baselines(shuffled, config)
    pd.testing.assert_frame_equal(first, second)
    assert first_manifest == second_manifest


def test_predict_rejects_future_label_columns(config: dict[str, object]) -> None:
    prefixes, _, _ = prepare_prefix_and_future_labels(_cycles(), config)
    attacked = prefixes.copy()
    attacked["future_capacity_label"] = 0.0
    with pytest.raises(NasaOfficialPrefixStressError, match="future-label columns"):
        predict_prefix_baselines(attacked, config)


def test_scorer_requires_preexisting_prediction_manifest(
    config: dict[str, object],
) -> None:
    prefixes, labels, _ = prepare_prefix_and_future_labels(_cycles(), config)
    predictions, manifest = predict_prefix_baselines(prefixes, config)
    assert manifest["score_executed"] is False
    assert manifest["prediction_sha256"]
    attacked = predictions.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] += 0.01
    with pytest.raises(NasaOfficialPrefixStressError, match="hash mismatch"):
        score_prefix_baselines(labels, attacked, manifest, config)


def test_primary_metric_is_locked_test_equal_cell_equal_prefix(
    config: dict[str, object],
) -> None:
    prefixes, labels, _ = prepare_prefix_and_future_labels(_cycles(), config)
    predictions, manifest = predict_prefix_baselines(prefixes, config)
    scores, summary = score_prefix_baselines(labels, predictions, manifest, config)
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
