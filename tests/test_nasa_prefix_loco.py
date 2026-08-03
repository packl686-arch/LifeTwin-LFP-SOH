from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.nasa_prefix_loco import (
    CELL_CUTOFFS,
    DATASET_ID,
    MODEL_IDS,
    NasaPrefixLocoError,
    PREDICTION_COLUMNS,
    PREFIX_CYCLES,
    PREFIX_TABLE_COLUMNS,
    SCORE_END_CYCLE,
    build_nasa_prefix_table,
    canonical_frame_sha256,
    load_nasa_prefix_loco_config,
    predict_nasa_prefix_loco,
    score_nasa_prefix_loco,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/nasa_prefix_loco_v1.json"


@pytest.fixture
def config() -> dict[str, object]:
    return load_nasa_prefix_loco_config(CONFIG_PATH)


@pytest.fixture
def cycles() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cell_parameters = {
        "B0005": (1.86, 0.29),
        "B0006": (2.04, 0.41),
        "B0007": (1.90, 0.25),
        "B0018": (1.86, 0.31),
    }
    for cell_id, (initial_capacity, terminal_loss) in cell_parameters.items():
        for cycle_index in range(1, SCORE_END_CYCLE + 1):
            progress = np.sqrt((cycle_index - 1) / (SCORE_END_CYCLE - 1))
            capacity = initial_capacity * (1.0 - terminal_loss * progress)
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "discharge_capacity_ah": float(capacity),
                    "discharge_cutoff_voltage_v": CELL_CUTOFFS[cell_id],
                    "ignored_adapter_column": "not_a_model_input",
                }
            )
    return pd.DataFrame(rows)


def _prediction_bundle(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    prefix_table = build_nasa_prefix_table(cycles, config)
    predictions, manifest = predict_nasa_prefix_loco(prefix_table, config)
    return predictions, manifest, prefix_table


def test_frozen_config_marks_nasa_as_descriptive_non_lfp_stress_only(
    config: dict[str, object],
) -> None:
    assert config["status"] == "descriptive_stress_test_frozen"
    assert config["dataset"]["chemistry"] == "unspecified_li_ion_not_lfp_evidence"
    assert config["metrics"]["inference"] == "descriptive_only_no_significance_test"
    prohibited = set(config["claim_boundaries"]["prohibited_claims"])
    assert {
        "lfp_chemistry_validation",
        "calendar_aging_validation",
        "fifteen_to_twenty_five_year_accuracy",
        "hithium_product_accuracy",
        "formal_uncertainty_coverage",
    }.issubset(prohibited)


def test_prefix_table_is_exactly_truncated_and_drops_unapproved_columns(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    prefix_table = build_nasa_prefix_table(cycles, config)
    assert tuple(prefix_table.columns) == PREFIX_TABLE_COLUMNS
    assert len(prefix_table) == len(CELL_CUTOFFS) * sum(PREFIX_CYCLES)
    counts = prefix_table.groupby(["cell_id", "prefix_cycle"]).size()
    for cell_id in CELL_CUTOFFS:
        for prefix_cycle in PREFIX_CYCLES:
            assert counts.loc[(cell_id, prefix_cycle)] == prefix_cycle
            group = prefix_table.loc[
                (prefix_table["cell_id"] == cell_id)
                & (prefix_table["prefix_cycle"] == prefix_cycle)
            ]
            assert group["cycle_index"].max() == prefix_cycle


def test_target_future_mutation_cannot_change_prediction_bundle(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, manifest, _ = _prediction_bundle(cycles, config)
    mutated = cycles.copy()
    target_future = (mutated["cell_id"] == "B0005") & (
        mutated["cycle_index"] > max(PREFIX_CYCLES)
    )
    mutated.loc[target_future, "discharge_capacity_ah"] *= 0.91

    attacked_predictions, attacked_manifest, _ = _prediction_bundle(mutated, config)

    pd.testing.assert_frame_equal(predictions, attacked_predictions)
    assert manifest == attacked_manifest
    assert manifest["target_future_outcomes_used"] is False
    assert manifest["training_cell_histories_used"] is False


def test_each_landmark_is_invariant_to_its_own_target_suffix(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, _, _ = _prediction_bundle(cycles, config)
    mutated = cycles.copy()
    target_suffix = (mutated["cell_id"] == "B0005") & (mutated["cycle_index"] > 40)
    mutated.loc[target_suffix, "discharge_capacity_ah"] *= 0.93
    attacked, _, _ = _prediction_bundle(mutated, config)

    frozen_landmarks = (predictions["held_out_cell_id"] == "B0005") & (
        predictions["prefix_cycle"] <= 40
    )
    attacked_landmarks = (attacked["held_out_cell_id"] == "B0005") & (
        attacked["prefix_cycle"] <= 40
    )
    pd.testing.assert_frame_equal(
        predictions.loc[frozen_landmarks].reset_index(drop=True),
        attacked.loc[attacked_landmarks].reset_index(drop=True),
    )


def test_prediction_rejects_any_future_row_in_prefix_input(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    prefix_table = build_nasa_prefix_table(cycles, config)
    future = cycles.loc[
        (cycles["cell_id"] == "B0005") & (cycles["cycle_index"] == 21),
        PREFIX_TABLE_COLUMNS[:-1],
    ].copy()
    future["prefix_cycle"] = 20
    attacked = pd.concat(
        [prefix_table, future.loc[:, PREFIX_TABLE_COLUMNS]], ignore_index=True
    )

    with pytest.raises(NasaPrefixLocoError, match="not exactly truncated"):
        predict_nasa_prefix_loco(attacked, config)


def test_prediction_coordinates_and_hash_are_deterministic(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    first, first_manifest, prefix_table = _prediction_bundle(cycles, config)
    second, second_manifest = predict_nasa_prefix_loco(prefix_table, config)

    pd.testing.assert_frame_equal(first, second)
    assert first_manifest == second_manifest
    expected_rows = (
        len(CELL_CUTOFFS)
        * len(MODEL_IDS)
        * sum(SCORE_END_CYCLE - prefix_cycle for prefix_cycle in PREFIX_CYCLES)
    )
    assert len(first) == expected_rows
    assert tuple(first.columns) == PREDICTION_COLUMNS
    assert first_manifest["prediction_sha256"] == canonical_frame_sha256(
        first, PREDICTION_COLUMNS
    )


def test_independent_scorer_returns_four_fold_descriptive_metrics(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, manifest, _ = _prediction_bundle(cycles, config)
    scores, summary = score_nasa_prefix_loco(
        cycles,
        predictions,
        manifest,
        config,
    )

    assert len(scores) == len(CELL_CUTOFFS) * len(PREFIX_CYCLES) * len(MODEL_IDS)
    assert len(summary["aggregate_metrics"]) == len(PREFIX_CYCLES) * len(MODEL_IDS)
    assert summary["fold_count"] == 4
    assert summary["inference_scope"] == "descriptive_only_no_significance_test"
    numeric = scores.select_dtypes(include=["number"]).to_numpy(dtype=float)
    assert np.isfinite(numeric).all()
    expected_future_counts = {
        prefix_cycle: SCORE_END_CYCLE - prefix_cycle for prefix_cycle in PREFIX_CYCLES
    }
    for row in scores.itertuples(index=False):
        assert row.future_observation_count == expected_future_counts[row.prefix_cycle]


def test_score_changes_when_linked_suffix_changes_but_prediction_does_not(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, manifest, _ = _prediction_bundle(cycles, config)
    original_scores, _ = score_nasa_prefix_loco(cycles, predictions, manifest, config)
    mutated = cycles.copy()
    target_future = (mutated["cell_id"] == "B0005") & (
        mutated["cycle_index"] > max(PREFIX_CYCLES)
    )
    mutated.loc[target_future, "discharge_capacity_ah"] *= 0.91
    attacked_predictions, attacked_manifest, _ = _prediction_bundle(mutated, config)
    attacked_scores, _ = score_nasa_prefix_loco(
        mutated,
        attacked_predictions,
        attacked_manifest,
        config,
    )

    pd.testing.assert_frame_equal(predictions, attacked_predictions)
    assert not original_scores.equals(attacked_scores)


def test_score_rejects_prediction_value_tampering(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, manifest, _ = _prediction_bundle(cycles, config)
    attacked = predictions.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] += 0.01

    with pytest.raises(NasaPrefixLocoError, match="artifact hash mismatch"):
        score_nasa_prefix_loco(cycles, attacked, manifest, config)


def test_prediction_csv_round_trip_remains_scorable(
    tmp_path: Path,
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, manifest, _ = _prediction_bundle(cycles, config)
    path = tmp_path / "predictions.csv"
    predictions.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    restored = pd.read_csv(path, float_precision="round_trip")

    scores, summary = score_nasa_prefix_loco(
        cycles,
        restored,
        manifest,
        config,
    )

    assert len(scores) == len(CELL_CUTOFFS) * len(PREFIX_CYCLES) * len(MODEL_IDS)
    assert summary["prediction_sha256"] == manifest["prediction_sha256"]
