from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.nasa_dynamic_gate_v2 import (
    BASE_MODEL_IDS,
    build_nasa_dynamic_gate_fold_table,
    load_nasa_dynamic_gate_config,
)
from lifetwin.experiments.nasa_evidence_weighted_moe_v3 import (
    MODEL_IDS,
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    V2_COMPARISON_GATE_MODEL_ID,
    V3_MODEL_ID,
    NasaEvidenceWeightedMoeError,
    load_nasa_evidence_weighted_moe_config,
    predict_nasa_evidence_weighted_moe,
    score_nasa_evidence_weighted_moe,
    validate_nasa_evidence_weighted_moe_config,
)
from lifetwin.experiments.nasa_prefix_loco import (
    CELL_CUTOFFS,
    DATASET_ID,
    PREFIX_CYCLES,
    SCORE_END_CYCLE,
    canonical_frame_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/nasa_dynamic_gate_v2.json"
V3_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/nasa_evidence_weighted_moe_v3.json"


@pytest.fixture(scope="module")
def v2_config() -> dict[str, object]:
    return load_nasa_dynamic_gate_config(V2_CONFIG_PATH)


@pytest.fixture(scope="module")
def v3_config() -> dict[str, object]:
    return load_nasa_evidence_weighted_moe_config(V3_CONFIG_PATH)


@pytest.fixture(scope="module")
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
            duration = 1_800.0 * capacity / initial
            voltage_1ah = 3.48 - 0.10 * progress + 0.025 * (cutoff - 2.5)
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "discharge_capacity_ah": float(capacity),
                    "discharge_cutoff_voltage_v": cutoff,
                    "common_window_3p8_to_3p4_duration_s": float(duration),
                    "voltage_at_1p0_ah_v": float(voltage_1ah),
                    "temperature_rise_c": float(14.0 + 3.0 * progress),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def prediction_bundle(
    cycles: pd.DataFrame,
    v2_config: dict[str, object],
    v3_config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, v2_config)
    predictions, manifest = predict_nasa_evidence_weighted_moe(
        fold_table,
        v2_config,
        v3_config,
    )
    return predictions, manifest, fold_table


def test_v3_freeze_records_exposed_outcomes_and_claim_boundaries(
    v3_config: dict[str, object],
) -> None:
    assert (
        v3_config["status"]
        == "post_outcome_development_frozen_before_any_new_external_dataset"
    )
    exposed = {entry["dataset_id"] for entry in v3_config["outcome_exposure_registry"]}
    assert exposed == {
        "NASA_PCOE_LI_ION_AGING_DERIVED_CSV_V1",
        "ATTIA_2020_VALIDATION45",
    }
    prohibited = set(v3_config["claim_boundaries"]["prohibited_claims"])
    assert {
        "nasa_v3_is_preregistered_outcome_blind_confirmation",
        "attia_2020_is_still_an_unseen_external_test",
        "formal_uncertainty_coverage",
        "hithium_product_accuracy",
    }.issubset(prohibited)


def test_v3_config_rejects_post_freeze_changes(
    v3_config: dict[str, object],
) -> None:
    attacked = copy.deepcopy(v3_config)
    attacked["mixture"]["risk_inverse_power"] = 3.0
    with pytest.raises(NasaEvidenceWeightedMoeError, match="config changed"):
        validate_nasa_evidence_weighted_moe_config(attacked)


def test_predictions_are_deterministic_and_weights_are_valid(
    prediction_bundle: tuple[pd.DataFrame, dict[str, object], pd.DataFrame],
    v2_config: dict[str, object],
    v3_config: dict[str, object],
) -> None:
    predictions, manifest, fold_table = prediction_bundle
    replayed, replayed_manifest = predict_nasa_evidence_weighted_moe(
        fold_table,
        v2_config,
        v3_config,
    )
    pd.testing.assert_frame_equal(predictions, replayed)
    assert manifest == replayed_manifest
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert set(predictions["model_id"]) == set(MODEL_IDS)
    assert manifest["target_future_outcomes_used"] is False
    assert manifest["outcomes_previously_exposed_for_development"] is True
    assert manifest["prediction_sha256"] == canonical_frame_sha256(
        predictions,
        PREDICTION_COLUMNS,
    )

    decisions = (
        predictions.loc[predictions["model_id"] == V3_MODEL_ID]
        .groupby(["held_out_cell_id", "prefix_cycle"], sort=True)
        .first()
    )
    for value in decisions["expert_weights_json"]:
        weights = json.loads(value)
        assert set(weights) == set(BASE_MODEL_IDS)
        assert all(weight >= 0.0 for weight in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert (
        decisions["evidence_band_lower_pct"]
        <= decisions["predicted_capacity_retention_pct"]
    ).all()
    assert (
        decisions["predicted_capacity_retention_pct"]
        <= decisions["evidence_band_upper_pct"]
    ).all()


def test_target_predictions_ignore_the_same_cells_unseen_suffix(
    cycles: pd.DataFrame,
    prediction_bundle: tuple[pd.DataFrame, dict[str, object], pd.DataFrame],
    v2_config: dict[str, object],
    v3_config: dict[str, object],
) -> None:
    original, _, _ = prediction_bundle
    attacked_cycles = cycles.copy()
    suffix = (attacked_cycles["cell_id"] == "B0005") & (
        attacked_cycles["cycle_index"] > max(PREFIX_CYCLES)
    )
    attacked_cycles.loc[suffix, "discharge_capacity_ah"] *= 0.80
    attacked_cycles.loc[suffix, "common_window_3p8_to_3p4_duration_s"] *= 0.75
    attacked_cycles.loc[suffix, "voltage_at_1p0_ah_v"] -= 0.2
    attacked_fold = build_nasa_dynamic_gate_fold_table(
        attacked_cycles,
        v2_config,
    )
    attacked, _ = predict_nasa_evidence_weighted_moe(
        attacked_fold,
        v2_config,
        v3_config,
    )

    original_target = original.loc[original["held_out_cell_id"] == "B0005"].reset_index(
        drop=True
    )
    attacked_target = attacked.loc[attacked["held_out_cell_id"] == "B0005"].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(original_target, attacked_target)
    assert (
        not original.loc[original["held_out_cell_id"] != "B0005"]
        .reset_index(drop=True)
        .equals(
            attacked.loc[attacked["held_out_cell_id"] != "B0005"].reset_index(drop=True)
        )
    )


def test_out_of_domain_prefix_triggers_refusal_recommendation(
    cycles: pd.DataFrame,
    v2_config: dict[str, object],
    v3_config: dict[str, object],
) -> None:
    attacked = cycles.copy()
    target = attacked["cell_id"] == "B0005"
    index = attacked.loc[target, "cycle_index"].to_numpy(dtype=float)
    attacked.loc[target, "common_window_3p8_to_3p4_duration_s"] *= 1.0 + index
    attacked.loc[target, "voltage_at_1p0_ah_v"] -= 0.10 * index
    attacked.loc[target, "temperature_rise_c"] += 8.0 * index
    fold_table = build_nasa_dynamic_gate_fold_table(attacked, v2_config)
    predictions, _ = predict_nasa_evidence_weighted_moe(
        fold_table,
        v2_config,
        v3_config,
    )
    decisions = (
        predictions.loc[
            (predictions["held_out_cell_id"] == "B0005")
            & (predictions["model_id"] == V3_MODEL_ID)
        ]
        .groupby("prefix_cycle", sort=True)
        .first()
    )
    assert (decisions["mean_neighbor_distance"] >= 6.0).all()
    assert set(decisions["evidence_status"]) == {"out_of_domain"}
    assert set(decisions["operational_action"]) == {"refuse_recommended"}


def test_scorer_replays_commitment_and_rejects_tampering(
    cycles: pd.DataFrame,
    prediction_bundle: tuple[pd.DataFrame, dict[str, object], pd.DataFrame],
    v2_config: dict[str, object],
    v3_config: dict[str, object],
) -> None:
    predictions, manifest, _ = prediction_bundle
    scores, summary = score_nasa_evidence_weighted_moe(
        cycles,
        predictions,
        manifest,
        v2_config,
        v3_config,
    )
    assert tuple(scores.columns) == SCORE_COLUMNS
    assert len(scores) == len(CELL_CUTOFFS) * len(PREFIX_CYCLES) * len(MODEL_IDS)
    assert summary["development_status"].startswith("post_nasa_v2")
    assert summary["evidence_band_scope"] == "descriptive_not_formal_coverage"
    assert summary["nasa_development_promotion_gate"]["status"] in {
        "passed",
        "failed",
    }
    assert "formal_uncertainty_coverage" in summary["prohibited_claims"]

    attacked = predictions.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] -= 0.01
    with pytest.raises(
        NasaEvidenceWeightedMoeError,
        match="deterministic frozen replay",
    ):
        score_nasa_evidence_weighted_moe(
            cycles,
            attacked,
            manifest,
            v2_config,
            v3_config,
        )

    attacked_manifest = dict(manifest)
    attacked_manifest["inference_scope"] = "independent_confirmation"
    with pytest.raises(NasaEvidenceWeightedMoeError, match="manifest mismatch"):
        score_nasa_evidence_weighted_moe(
            cycles,
            predictions,
            attacked_manifest,
            v2_config,
            v3_config,
        )


def test_csv_round_trip_preserves_prediction_commitment(
    tmp_path: Path,
    cycles: pd.DataFrame,
    prediction_bundle: tuple[pd.DataFrame, dict[str, object], pd.DataFrame],
    v2_config: dict[str, object],
    v3_config: dict[str, object],
) -> None:
    predictions, manifest, fold_table = prediction_bundle
    fold_path = tmp_path / "fold.csv"
    prediction_path = tmp_path / "predictions.csv"
    fold_table.to_csv(fold_path, index=False, float_format="%.17g")
    predictions.to_csv(prediction_path, index=False, float_format="%.17g")
    replayed_fold = pd.read_csv(fold_path, float_precision="round_trip")
    replayed_predictions, replayed_manifest = predict_nasa_evidence_weighted_moe(
        replayed_fold,
        v2_config,
        v3_config,
    )
    pd.testing.assert_frame_equal(predictions, replayed_predictions)
    assert manifest == replayed_manifest

    persisted = pd.read_csv(prediction_path, float_precision="round_trip")
    scores, _ = score_nasa_evidence_weighted_moe(
        cycles,
        persisted,
        manifest,
        v2_config,
        v3_config,
    )
    assert set(scores["model_id"]) == set(MODEL_IDS)
    assert V2_COMPARISON_GATE_MODEL_ID in set(scores["model_id"])
