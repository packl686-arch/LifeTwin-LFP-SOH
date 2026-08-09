from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.data.snl import DATASET_ID, RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
import lifetwin.experiments.snl_rpt_loco as snl_loco
from lifetwin.experiments.snl_rpt_post_outcome_audit import (
    CELL_DIAGNOSTIC_COLUMNS,
    CONDITION_DIAGNOSTIC_COLUMNS,
    EXTRACTION_SENSITIVITY_COLUMNS,
    MODEL_METRIC_COLUMNS,
    SELECTOR_CHOICE_COLUMNS,
    SNLRPTPostOutcomeAuditError,
    audit_snl_rpt_loco_result,
    summarize_snl_rpt_extraction_sensitivity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/snl_lfp_rpt_loco_v1.json"


def _synthetic_trajectories() -> pd.DataFrame:
    records = []
    conditions = [
        ("C1", 15.0, 1.0, 1.0, 2.2),
        ("C2", 25.0, 0.6, 1.0, 2.8),
        ("C3", 35.0, 1.0, 2.0, 4.0),
        ("C4", 25.0, 0.2, 3.0, 3.3),
    ]
    for condition_id, temperature, dod, rate, fade_rate in conditions:
        for visit, efc in enumerate((0.0, 100.0, 200.0, 300.0, 400.0, 500.0)):
            retention = 100.0 - fade_rate * (efc / 1000.0) ** 0.5
            records.append(
                {
                    "dataset_id": DATASET_ID,
                    "cell_id": f"CELL_{condition_id}",
                    "condition_id": condition_id,
                    "temperature_c": temperature,
                    "min_soc_pct": 0.0,
                    "max_soc_pct": dod * 100.0,
                    "dod_fraction": dod,
                    "charge_c_rate": 0.5,
                    "discharge_c_rate": rate,
                    "visit_index": visit,
                    "elapsed_days": efc / 4.0,
                    "equivalent_full_cycles": efc,
                    "capacity_ah": 1.1 * retention / 100.0,
                    "capacity_retention_pct": retention,
                    "rpt_cycle_count": 3,
                }
            )
    return pd.DataFrame(records, columns=RPT_TRAJECTORY_COLUMNS).sort_values(
        ["condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )


def _synthetic_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    config = deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    trajectories = _synthetic_trajectories()
    config["dataset"].update(
        {
            "rpt_trajectory_sha256": canonical_frame_sha256(
                trajectories, RPT_TRAJECTORY_COLUMNS
            ),
            "physical_cell_count": 4,
            "condition_cluster_count": 4,
            "rpt_trajectory_row_count": 24,
            "minimum_rpt_visit_count": 6,
        }
    )
    config["outer_validation"]["expected_outer_folds"] = 4
    config["dynamic_landmarks"].update(
        {
            "prefix_visit_counts": [3],
            "primary_prefix_visit_count": 3,
            "minimum_future_visits_by_landmark": {"3": 3},
            "score_end_equivalent_full_cycles": 500.0,
            "forecast_grid_step_equivalent_full_cycles": 25.0,
        }
    )
    config["base_experts"]["nearest_reference_count"] = 2
    config["safe_hard_selector"]["local_neighbor_count"] = 2
    config["evaluation"]["primary_landmark_visit_count"] = 3
    semantic_hash = canonical_json_sha256(config)
    monkeypatch.setattr(snl_loco, "CONFIG_SEMANTIC_SHA256", semantic_hash)
    return config


def test_snl_loco_prediction_firewall_and_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _synthetic_config(monkeypatch)
    trajectories = _synthetic_trajectories()
    references, prefixes, truth, audit = snl_loco.build_snl_rpt_loco_inputs(
        trajectories, config
    )
    assert audit["target_suffix_rows_in_prefix_input"] is False
    for outer, frame in references.groupby("outer_condition_id"):
        assert outer not in set(frame["condition_id"])

    predictions, decisions, manifest = snl_loco.predict_snl_rpt_loco(
        references, prefixes, config
    )
    assert manifest["target_truth_argument_accepted"] is False
    assert len(decisions) == 4
    assert set(predictions["model_id"]).issuperset(set(snl_loco.BASE_MODEL_IDS))

    attacked_truth = truth.copy()
    attacked_truth.loc[attacked_truth["visit_index"] >= 3, "capacity_retention_pct"] -= 20.0
    replay, replay_decisions, replay_manifest = snl_loco.predict_snl_rpt_loco(
        references, prefixes, config
    )
    pd.testing.assert_frame_equal(predictions, replay)
    pd.testing.assert_frame_equal(decisions, replay_decisions)
    assert manifest == replay_manifest

    scores, summary = snl_loco.score_snl_rpt_loco(
        truth, predictions, decisions, manifest, config
    )
    attacked_scores, _ = snl_loco.score_snl_rpt_loco(
        attacked_truth, predictions, decisions, manifest, config
    )
    assert not scores["trajectory_iae_pp"].equals(
        attacked_scores["trajectory_iae_pp"]
    )
    assert summary["evidence_role"] == (
        "retrospective_grouped_cross_condition_development"
    )


def test_snl_loco_scorer_rejects_prediction_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _synthetic_config(monkeypatch)
    references, prefixes, truth, _ = snl_loco.build_snl_rpt_loco_inputs(
        _synthetic_trajectories(), config
    )
    predictions, decisions, manifest = snl_loco.predict_snl_rpt_loco(
        references, prefixes, config
    )
    attacked = predictions.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] += 1.0
    with pytest.raises(snl_loco.SNLRPTLOCOError, match="changed after freeze"):
        snl_loco.score_snl_rpt_loco(
            truth, attacked, decisions, manifest, config
        )


def test_snl_post_outcome_audit_replays_and_labels_hindsight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _synthetic_config(monkeypatch)
    references, prefixes, truth, _ = snl_loco.build_snl_rpt_loco_inputs(
        _synthetic_trajectories(), config
    )
    predictions, decisions, manifest = snl_loco.predict_snl_rpt_loco(
        references, prefixes, config
    )
    scores, score_summary = snl_loco.score_snl_rpt_loco(
        truth, predictions, decisions, manifest, config
    )
    cells, conditions, models, choices, summary = audit_snl_rpt_loco_result(
        truth,
        predictions,
        decisions,
        manifest,
        scores,
        score_summary,
        config,
    )

    assert tuple(cells.columns) == CELL_DIAGNOSTIC_COLUMNS
    assert tuple(conditions.columns) == CONDITION_DIAGNOSTIC_COLUMNS
    assert tuple(models.columns) == MODEL_METRIC_COLUMNS
    assert tuple(choices.columns) == SELECTOR_CHOICE_COLUMNS
    assert len(cells) == 4
    assert len(conditions) == 4
    assert len(models) == len(snl_loco.MODEL_IDS)
    assert summary["evidence_role"] == "post_outcome_diagnostic_not_model_selection"
    assert "hindsight" in " ".join(summary["claim_boundary"]).lower()

    attacked_scores = scores.copy()
    attacked_scores.loc[0, "trajectory_iae_pp"] += 0.01
    with pytest.raises(SNLRPTPostOutcomeAuditError, match="score table changed"):
        audit_snl_rpt_loco_result(
            truth,
            predictions,
            decisions,
            manifest,
            attacked_scores,
            score_summary,
            config,
        )


def test_snl_extraction_sensitivity_requires_exact_primary_replay() -> None:
    primary = _synthetic_trajectories()
    primary_hash = canonical_frame_sha256(primary, RPT_TRAJECTORY_COLUMNS)
    alternate = primary.loc[
        ~(
            (primary["cell_id"] == "CELL_C4")
            & (primary["visit_index"] == primary["visit_index"].max())
        )
    ].reset_index(drop=True)
    alternate_hash = canonical_frame_sha256(alternate, RPT_TRAJECTORY_COLUMNS)
    table, summary = summarize_snl_rpt_extraction_sensitivity(
        primary,
        [
            (
                1.0,
                10.0,
                primary,
                {"canonical_rpt_trajectory_sha256": primary_hash},
            ),
            (
                2.0,
                10.0,
                alternate,
                {"canonical_rpt_trajectory_sha256": alternate_hash},
            ),
        ],
    )

    assert tuple(table.columns) == EXTRACTION_SENSITIVITY_COLUMNS
    assert len(table) == 2
    assert bool(table.loc[table["is_primary_setting"]].iloc[0][
        "exact_primary_trajectory_match"
    ])
    alternate_row = table.loc[~table["is_primary_setting"]].iloc[0]
    assert alternate_row["visit_count_changed_cell_count"] == 1
    assert summary["setting_count"] == 2
