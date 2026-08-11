from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import fastcharge_v5_landmark as landmark
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiments/v5_dynamic_landmark_online_update_v1.json"


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _synthetic_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_index, cell_id in enumerate(("cell_a", "cell_b", "cell_c")):
        for prefix in (20, 40, 60, 100):
            for cycle in range(prefix + 1, 121):
                truth = 100.0 - 0.01 * cycle - 0.02 * cell_index
                rows.append(
                    {
                        "cell_id": cell_id,
                        "prefix_cycle": prefix,
                        "forecast_cycle": cycle,
                        "observed_retention_pct": truth,
                        "candidate_prediction_pct": truth + 0.2,
                    }
                )
    return pd.DataFrame(rows)


def test_dynamic_landmark_protocol_is_frozen_and_training_only() -> None:
    protocol = _protocol()
    firewall = protocol["data_firewall"]
    selection = protocol["selection"]

    assert "frozen_before_formal" in protocol["status"]
    assert firewall["candidate_selection_uses_training_cells_only"] is True
    assert firewall["future_target_outcomes_available_to_update"] is False
    assert (
        firewall["public_evaluation_outcomes_available_to_candidate_selection"] is False
    )
    assert selection["evaluation_split_retuning_permitted"] is False
    assert (
        protocol["stopping_rules"][
            "same_exposed_evaluation_cohort_second_round_permitted"
        ]
        is False
    )


def test_registered_candidate_library_is_small_and_deterministic() -> None:
    rules = landmark.candidate_rules(_protocol())
    ids = [rule.candidate_id for rule in rules]

    assert len(rules) == 20
    assert len(set(ids)) == len(ids)
    assert ids[0] == "no_update"
    assert sum(rule.family == "fixed_gaussian_process" for rule in rules) == 3


def test_residual_correction_rejects_future_history_and_caps_gp() -> None:
    protocol = _protocol()
    gp_rule = next(
        rule
        for rule in landmark.candidate_rules(protocol)
        if rule.family == "fixed_gaussian_process"
    )
    correction = landmark.residual_correction(
        np.arange(21, 41),
        np.full(20, 100.0),
        np.arange(41, 61),
        previous_prefix=20,
        score_end_cycle=300,
        rule=gp_rule,
        config=protocol,
    )
    assert np.max(np.abs(correction)) <= 1.5

    with pytest.raises(FastChargeV5PairwiseError, match="landmark firewall"):
        landmark.residual_correction(
            np.arange(21, 42),
            np.zeros(21),
            np.arange(41, 61),
            previous_prefix=20,
            score_end_cycle=300,
            rule=gp_rule,
            config=protocol,
        )


def test_training_only_selector_activates_only_eligible_rules() -> None:
    protocol = _protocol()
    scores = landmark.score_residual_candidates(_synthetic_predictions(), protocol)
    summary = landmark.summarize_candidate_scores(scores, protocol)
    selected = landmark.select_rules(summary)

    assert set(selected) == {40, 60, 100}
    assert all(candidate_id != "no_update" for candidate_id in selected.values())
    for current, candidate_id in selected.items():
        row = summary.loc[
            (summary["current_prefix_cycle"] == current)
            & (summary["candidate_id"] == candidate_id)
        ]
        assert len(row) == 1
        assert bool(row.iloc[0]["eligible"])


def test_nested_selector_and_reissue_keep_physical_cell_as_unit() -> None:
    protocol = _protocol()
    frame = _synthetic_predictions()
    scores = landmark.score_residual_candidates(frame, protocol)
    nested = landmark.nested_selector_audit(scores, protocol)
    reissues = landmark.score_base_reissues(frame, protocol)

    assert len(nested) == 3 * 3
    assert nested["selection_fit_cell_count"].eq(2).all()
    assert len(reissues) == 3 * 3
    assert reissues["cell_id"].nunique() == 3
    assert set(reissues["current_prefix_cycle"]) == {40, 60, 100}


def test_published_landmark_decision_matches_frozen_protocol_and_gates() -> None:
    decision = json.loads(
        (ROOT / "showcase/evidence_v5/dynamic_landmark_decision.json").read_text(
            encoding="utf-8"
        )
    )
    protocol_hash = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    evaluation = decision["public_evaluation_base_reissue"]
    overall = evaluation["overall_transition_equal"]

    assert decision["protocol_sha256"] == protocol_hash
    assert decision["selected_residual_rule_by_current_prefix"] == {
        "40": "offset_median_all_a0p25",
        "60": "no_update",
        "100": "no_update",
    }
    assert decision["interval_subgate_passed"] is True
    assert decision["full_H2_gp_online_landmark_gate_passed"] is False
    assert decision["gp_branch_activated"] is False
    assert overall["mean_previous_trajectory_mae_pp"] == pytest.approx(0.2644217248)
    assert overall["mean_current_trajectory_mae_pp"] == pytest.approx(0.1738385239)
    bootstrap = overall["physical_cell_clustered_bootstrap"]
    assert bootstrap["lower_delta_mae_pp"] == pytest.approx(-0.1195073450)
    assert bootstrap["upper_delta_mae_pp"] == pytest.approx(-0.0658987551)
    p60 = next(
        row for row in evaluation["transitions"] if row["current_prefix_cycle"] == 60
    )
    assert p60["paired_cell_bootstrap"]["lower_delta_mae_pp"] < 0.0
    assert p60["paired_cell_bootstrap"]["upper_delta_mae_pp"] > 0.0


def test_public_landmark_evidence_contains_only_training_selected_rules() -> None:
    candidate_summary = pd.read_csv(
        ROOT / "showcase/evidence_v5/dynamic_landmark_training_candidate_summary.csv"
    )
    evaluation = pd.read_csv(
        ROOT
        / "showcase/evidence_v5/dynamic_landmark_evaluation_selected_update_scores.csv"
    )

    assert len(candidate_summary) == 20 * 3
    assert candidate_summary["physical_cell_count"].eq(41).all()
    eligible = candidate_summary.loc[candidate_summary["eligible"]]
    assert eligible["candidate_id"].tolist() == ["offset_median_all_a0p25"]
    assert set(
        evaluation.loc[evaluation["current_prefix_cycle"] == 40, "candidate_id"]
    ) == {"offset_median_all_a0p25"}
    assert set(
        evaluation.loc[
            evaluation["current_prefix_cycle"].isin([60, 100]), "candidate_id"
        ]
    ) == {"no_update"}
    assert not evaluation["candidate_id"].str.startswith("gp_").any()
    assert not {
        "discharge_capacity_ah",
        "charge_capacity_ah",
        "voltage_v",
        "current_a",
    } & set(evaluation.columns)
