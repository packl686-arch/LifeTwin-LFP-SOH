from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.nasa_dynamic_gate_v2 import (
    BASE_MODEL_IDS,
    FOLD_TABLE_COLUMNS,
    GATED_MODEL_IDS,
    MODEL_IDS,
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    NasaDynamicGateError,
    build_nasa_dynamic_gate_fold_table,
    load_nasa_dynamic_gate_config,
    predict_nasa_dynamic_gate,
    score_nasa_dynamic_gate,
)
from lifetwin.experiments.nasa_prefix_loco import (
    CELL_CUTOFFS,
    DATASET_ID,
    PREFIX_CYCLES,
    SCORE_END_CYCLE,
    canonical_frame_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/nasa_dynamic_gate_v2.json"


@pytest.fixture
def config() -> dict[str, object]:
    return load_nasa_dynamic_gate_config(CONFIG_PATH)


@pytest.fixture
def cycles() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    parameters = {
        "B0005": (1.88, 0.22, 0.035),
        "B0006": (2.02, 0.28, 0.055),
        "B0007": (1.91, 0.18, 0.025),
        "B0018": (1.87, 0.30, 0.060),
    }
    for cell_id, (initial, terminal_loss, curvature) in parameters.items():
        cutoff = CELL_CUTOFFS[cell_id]
        for cycle_index in range(1, SCORE_END_CYCLE + 1):
            progress = (cycle_index - 1) / (SCORE_END_CYCLE - 1)
            loss = terminal_loss * progress + curvature * np.sqrt(progress)
            recovery = 0.004 * np.sin(cycle_index / 7.0)
            capacity = initial * (1.0 - loss + recovery)
            integrated_capacity = capacity * 1.008
            duration = 1_800.0 * capacity / initial
            voltage_1ah = 3.48 - 0.10 * progress + 0.025 * (cutoff - 2.5)
            voltage_0p5ah = voltage_1ah + 0.18
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "discharge_capacity_ah": float(capacity),
                    "discharge_cutoff_voltage_v": cutoff,
                    "common_window_3p8_to_3p4_duration_s": float(duration),
                    "voltage_at_0p5_ah_v": float(voltage_0p5ah),
                    "voltage_at_1p0_ah_v": float(voltage_1ah),
                    "mean_dv_dq_0p5_to_1p0_v_per_ah": float(
                        (voltage_1ah - voltage_0p5ah) / 0.5
                    ),
                    "temperature_rise_c": float(14.0 + 3.0 * progress),
                    "integrated_discharge_capacity_ah": float(integrated_capacity),
                    "integrated_discharge_energy_wh": float(integrated_capacity * 3.55),
                    "capacity_integration_ratio": 1.008,
                    "ignored_adapter_column": "not_a_model_input",
                }
            )
    return pd.DataFrame(rows)


def _prediction_bundle(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, config)
    predictions, manifest = predict_nasa_dynamic_gate(fold_table, config)
    return predictions, manifest, fold_table


def test_frozen_v2_config_is_honest_about_development_and_claim_limits(
    config: dict[str, object],
) -> None:
    assert config["status"] == "post_v1_development_protocol_frozen_before_v2_run"
    assert config["design"]["target_future_outcomes_available_to_prediction"] is False
    assert (
        config["design"]["training_cell_histories_available_to_nested_selection"]
        is True
    )
    assert config["evidence_band"]["formal_coverage_claim"] is False
    prohibited = set(config["claim_boundaries"]["prohibited_claims"])
    assert {
        "independent_outcome_blind_confirmation",
        "lfp_chemistry_validation",
        "calendar_aging_validation",
        "formal_uncertainty_coverage",
    }.issubset(prohibited)


def test_fold_table_exposes_only_target_prefix_and_outer_training_histories(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, config)
    assert tuple(fold_table.columns) == FOLD_TABLE_COLUMNS
    assert "ignored_adapter_column" not in fold_table
    for held_out_cell_id in CELL_CUTOFFS:
        for prefix_cycle in PREFIX_CYCLES:
            fold = fold_table.loc[
                (fold_table["held_out_cell_id"] == held_out_cell_id)
                & (fold_table["prefix_cycle"] == prefix_cycle)
            ]
            target = fold.loc[fold["row_role"] == "target_prefix"]
            training = fold.loc[fold["row_role"] == "training_history"]
            assert set(target["cell_id"]) == {held_out_cell_id}
            assert target["cycle_index"].max() == prefix_cycle
            assert len(target) == prefix_cycle
            assert held_out_cell_id not in set(training["cell_id"])
            assert training.groupby("cell_id").size().eq(SCORE_END_CYCLE).all()


def test_predictions_are_deterministic_gated_and_manifest_bound(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    first, first_manifest, fold_table = _prediction_bundle(cycles, config)
    second, second_manifest = predict_nasa_dynamic_gate(fold_table, config)

    pd.testing.assert_frame_equal(first, second)
    assert first_manifest == second_manifest
    assert tuple(first.columns) == PREDICTION_COLUMNS
    assert set(first["model_id"]) == set(MODEL_IDS)
    assert set(first["selected_base_model_id"]) <= set(BASE_MODEL_IDS)
    assert set(
        first.loc[first["model_id"].isin(GATED_MODEL_IDS), "gate_feature_set"]
    ) == {
        "capacity_only",
        "capacity_plus_curve",
    }
    expected_rows = (
        len(CELL_CUTOFFS)
        * len(MODEL_IDS)
        * sum(SCORE_END_CYCLE - prefix_cycle for prefix_cycle in PREFIX_CYCLES)
    )
    assert len(first) == expected_rows
    assert first_manifest["target_future_outcomes_used"] is False
    assert first_manifest["outer_fold_training_histories_used"] is True
    assert first_manifest["prediction_sha256"] == canonical_frame_sha256(
        first,
        PREDICTION_COLUMNS,
    )


def test_each_held_out_cell_is_invariant_to_its_own_unseen_suffix(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, _, _ = _prediction_bundle(cycles, config)
    mutated = cycles.copy()
    suffix = (mutated["cell_id"] == "B0005") & (
        mutated["cycle_index"] > max(PREFIX_CYCLES)
    )
    mutated.loc[suffix, "discharge_capacity_ah"] *= 0.82
    mutated.loc[suffix, "common_window_3p8_to_3p4_duration_s"] *= 0.77
    mutated.loc[suffix, "voltage_at_1p0_ah_v"] -= 0.15
    attacked, _, _ = _prediction_bundle(mutated, config)

    original_target = predictions.loc[
        predictions["held_out_cell_id"] == "B0005"
    ].reset_index(drop=True)
    attacked_target = attacked.loc[attacked["held_out_cell_id"] == "B0005"].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(original_target, attacked_target)
    assert (
        not predictions.loc[predictions["held_out_cell_id"] != "B0005"]
        .reset_index(drop=True)
        .equals(
            attacked.loc[attacked["held_out_cell_id"] != "B0005"].reset_index(drop=True)
        )
    )


def test_prediction_rejects_target_future_injection(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, config)
    source = fold_table.loc[
        (fold_table["held_out_cell_id"] == "B0005")
        & (fold_table["prefix_cycle"] == 20)
        & (fold_table["row_role"] == "training_history")
        & (fold_table["cell_id"] == "B0006")
        & (fold_table["cycle_index"] == 21)
    ].copy()
    source["cell_id"] = "B0005"
    source["row_role"] = "target_prefix"
    source["discharge_cutoff_voltage_v"] = CELL_CUTOFFS["B0005"]
    attacked = pd.concat([fold_table, source], ignore_index=True)

    with pytest.raises(NasaDynamicGateError, match="not exactly truncated"):
        predict_nasa_dynamic_gate(attacked, config)


def test_score_reports_ablation_and_descriptive_evidence_bands(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, manifest, _ = _prediction_bundle(cycles, config)
    scores, summary = score_nasa_dynamic_gate(
        cycles,
        predictions,
        manifest,
        config,
    )

    assert tuple(scores.columns) == SCORE_COLUMNS
    assert len(scores) == len(CELL_CUTOFFS) * len(PREFIX_CYCLES) * len(MODEL_IDS)
    assert scores["empirical_evidence_band_coverage_fraction"].between(0, 1).all()
    assert (scores["mean_evidence_band_width_pp"] > 0).all()
    assert len(summary["aggregate_metrics"]) == len(PREFIX_CYCLES) * len(MODEL_IDS)
    assert summary["evidence_band_scope"] == "descriptive_not_formal_coverage"
    assert "capacity_only_versus_curve_aware_gate_ablation" in summary["allowed_claims"]
    assert "independent_outcome_blind_confirmation" in summary["prohibited_claims"]


def test_csv_round_trip_preserves_prediction_commitment(
    tmp_path: Path,
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> None:
    predictions, manifest, fold_table = _prediction_bundle(cycles, config)
    fold_path = tmp_path / "fold.csv"
    prediction_path = tmp_path / "predictions.csv"
    fold_table.to_csv(fold_path, index=False, float_format="%.17g")
    predictions.to_csv(prediction_path, index=False, float_format="%.17g")

    replayed_fold = pd.read_csv(fold_path, float_precision="round_trip")
    replayed_predictions, replayed_manifest = predict_nasa_dynamic_gate(
        replayed_fold,
        config,
    )
    pd.testing.assert_frame_equal(predictions, replayed_predictions)
    assert manifest == replayed_manifest

    persisted_predictions = pd.read_csv(
        prediction_path,
        float_precision="round_trip",
    )
    scores, _ = score_nasa_dynamic_gate(
        cycles,
        persisted_predictions,
        manifest,
        config,
    )
    assert len(scores) == len(CELL_CUTOFFS) * len(PREFIX_CYCLES) * len(MODEL_IDS)
