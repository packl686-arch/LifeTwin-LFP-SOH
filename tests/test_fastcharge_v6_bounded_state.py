from __future__ import annotations

import inspect
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import fastcharge_v6_bounded_state as bounded
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/experiments/v6_bounded_state_update_development.json"
)
GATED_PROTOCOL_PATH = (
    ROOT / "configs/experiments/v6_1_gated_state_update_development.json"
)
BLIND_CANDIDATE_PATH = (
    ROOT / "configs/experiments/v6_1_p100_gated_state_blind_candidate.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _gated_protocol() -> dict[str, object]:
    return json.loads(GATED_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _synthetic_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_index, cell_id in enumerate(("cell_a", "cell_b", "cell_c")):
        for prefix in (20, 40, 60, 100):
            for cycle in range(prefix + 1, 121):
                truth = 100.0 - 0.01 * cycle - 0.02 * cell_index
                residual = 0.001 * (cycle - prefix)
                rows.append(
                    {
                        "cell_id": cell_id,
                        "prefix_cycle": prefix,
                        "forecast_cycle": cycle,
                        "observed_retention_pct": truth,
                        "candidate_prediction_pct": truth - residual,
                    }
                )
    return pd.DataFrame(rows)


def test_protocol_keeps_v5_frozen_and_excludes_exposed_evaluation() -> None:
    protocol = _protocol()
    firewall = protocol["data_firewall"]

    assert protocol["champion"]["modification_permitted"] is False
    assert firewall["candidate_development_uses_training_cells_only"] is True
    assert firewall["future_target_outcomes_available_to_update"] is False
    assert firewall["exposed_public_evaluation_outcomes_permitted"] is False
    assert (
        protocol["inputs"]["exposed_public_evaluation_predictions"][
            "permitted_use"
        ]
        == "none"
    )
    assert (
        protocol["promotion_gate"][
            "minimum_fraction_cell_transitions_improved"
        ]
        == 0.7
    )


def test_candidate_library_is_small_unique_and_deterministic() -> None:
    rules = bounded.candidate_rules(_protocol())
    identifiers = [rule.candidate_id for rule in rules]

    assert len(rules) == 11
    assert len(set(identifiers)) == len(identifiers)
    assert identifiers[0] == "no_update"
    assert sum(rule.family == "robust_local_trend" for rule in rules) == 6
    assert sum(rule.family == "bounded_alpha_beta" for rule in rules) == 4


def test_bounded_correction_enforces_landmark_firewall_and_cap() -> None:
    protocol = _protocol()
    rule = next(
        rule
        for rule in bounded.candidate_rules(protocol)
        if rule.family == "robust_local_trend"
    )
    correction = bounded.bounded_state_correction(
        np.arange(21, 41),
        np.arange(20, dtype=float) * 100.0,
        np.arange(41, 301),
        previous_prefix=20,
        current_prefix=40,
        rule=rule,
        config=protocol,
    )
    assert np.max(np.abs(correction)) <= 1.0

    with pytest.raises(FastChargeV5PairwiseError, match="landmark firewall"):
        bounded.bounded_state_correction(
            np.arange(21, 42),
            np.zeros(21),
            np.arange(41, 61),
            previous_prefix=20,
            current_prefix=40,
            rule=rule,
            config=protocol,
        )


def test_correction_api_has_no_future_outcome_argument() -> None:
    parameters = inspect.signature(bounded.bounded_state_correction).parameters

    assert "future_outcomes" not in parameters
    assert "future_truth" not in parameters
    assert set(parameters) == {
        "history_cycles",
        "history_residuals",
        "future_cycles",
        "previous_prefix",
        "current_prefix",
        "rule",
        "config",
    }


def test_robust_trend_challenger_improves_monotone_residual_drift() -> None:
    protocol = _protocol()
    scores = bounded.score_bounded_state_candidates(
        _synthetic_predictions(), protocol
    )
    summary = bounded.summarize_candidate_scores(scores, protocol)
    selected = bounded.select_rules(summary)

    assert set(selected) == {40, 60, 100}
    assert all(candidate_id != "no_update" for candidate_id in selected.values())
    for current, candidate_id in selected.items():
        row = summary.loc[
            (summary["current_prefix_cycle"] == current)
            & (summary["candidate_id"] == candidate_id)
        ]
        assert len(row) == 1
        assert bool(row.iloc[0]["eligible"])
        assert float(row.iloc[0]["mean_delta_mae_pp"]) < 0.0


def test_nested_audit_and_promotion_keep_physical_cell_as_unit() -> None:
    protocol = _protocol()
    scores = bounded.score_bounded_state_candidates(
        _synthetic_predictions(), protocol
    )
    nested = bounded.nested_selector_audit(scores, protocol)
    promotion = bounded.promotion_summary(nested, protocol)

    assert len(nested) == 3 * 3
    assert nested["selection_fit_cell_count"].eq(2).all()
    assert promotion["overall"]["physical_cell_transition_count"] == 9
    assert promotion["passed"] is True


def test_v6_1_protocol_is_training_only_and_does_not_activate_challenger() -> None:
    protocol = _gated_protocol()

    assert protocol["champion"]["modification_permitted"] is False
    assert (
        protocol["champion"][
            "activation_before_new_blind_confirmation_permitted"
        ]
        is False
    )
    assert (
        protocol["data_firewall"][
            "outer_held_out_future_outcomes_available_to_gate_selection"
        ]
        is False
    )
    assert (
        protocol["inputs"]["exposed_public_evaluation_predictions"][
            "permitted_use"
        ]
        == "none"
    )


def test_activation_gate_library_is_small_and_deterministic() -> None:
    gates = bounded.activation_gates(_gated_protocol())

    assert len(gates) == 11
    assert gates[0].gate_id == "no_update"
    assert len({gate.gate_id for gate in gates}) == len(gates)
    assert all(
        gate.require_slope_sign_agreement for gate in gates[1:]
    )


def test_nested_activation_gate_can_abstain_without_changing_center() -> None:
    protocol = _gated_protocol()
    protocol["future_blind_nomination_gate"]["minimum_activation_count"] = 1
    frame = _synthetic_predictions()
    correction_ids = {
        gate.correction_candidate_id
        for gate in bounded.activation_gates(protocol)
        if gate.correction_candidate_id is not None
    }
    rules = [
        rule
        for rule in bounded.candidate_rules(protocol)
        if rule.candidate_id in correction_ids
    ]
    scores = bounded.score_bounded_state_candidates(
        frame, protocol, rules=rules
    )
    table = bounded.build_activation_table(frame, scores, protocol)
    summary = bounded.summarize_activation_gates(table, protocol)
    nested = bounded.nested_activation_gate_audit(table, protocol)
    nomination = bounded.activation_nomination_summary(nested, protocol)

    assert len(table) == 3 * 3 * 2
    assert len(nested) == 3 * 3
    assert nested["selection_fit_cell_count"].eq(2).all()
    assert (nested.loc[~nested["activated"], "gated_delta_mae_pp"] == 0.0).all()
    assert summary["gate_id"].nunique() == 11
    assert nomination["any_transition_nominated"] is True
    json.dumps(nomination, allow_nan=False)


def test_gate_activation_predicate_does_not_read_outcome_delta() -> None:
    protocol = _gated_protocol()
    frame = _synthetic_predictions()
    correction_ids = {
        gate.correction_candidate_id
        for gate in bounded.activation_gates(protocol)
        if gate.correction_candidate_id is not None
    }
    rules = [
        rule
        for rule in bounded.candidate_rules(protocol)
        if rule.candidate_id in correction_ids
    ]
    scores = bounded.score_bounded_state_candidates(
        frame, protocol, rules=rules
    )
    table = bounded.build_activation_table(frame, scores, protocol)
    gate = next(
        gate
        for gate in bounded.activation_gates(protocol)
        if gate.correction_candidate_id == "theil_all_a0p1"
    )
    selected = table.loc[table["current_prefix_cycle"] == 100].copy()
    _, original_mask = bounded._gate_rows(selected, gate)
    selected["raw_delta_mae_pp"] = 999.0
    _, attacked_mask = bounded._gate_rows(selected, gate)

    assert np.array_equal(original_mask, attacked_mask)


def test_published_v6_decisions_match_protocols_and_claim_boundaries() -> None:
    ungated = json.loads(
        (ROOT / "showcase/evidence_v6/ungated_state_decision.json").read_text(
            encoding="utf-8"
        )
    )
    gated = json.loads(
        (ROOT / "showcase/evidence_v6/gated_state_decision.json").read_text(
            encoding="utf-8"
        )
    )

    assert ungated["decision"] == "retain_frozen_v5_champion"
    assert ungated["promotion_gate"]["passed"] is False
    assert ungated["exposed_81_cell_evaluation_used"] is False
    assert gated["v5_champion_remains_active"] is True
    assert gated["v6_1_gate_activated"] is False
    assert gated["exposed_81_cell_evaluation_used"] is False
    assert gated["nominated_current_prefix_cycles"] == [100]
    assert gated["runtime_versions"]["python"] == "3.12.13"

    for decision in (ungated, gated):
        implementation = decision["implementation"]
        module_path = ROOT / implementation["module_path"]
        runner_path = ROOT / implementation["runner_path"]
        assert hashlib.sha256(module_path.read_bytes()).hexdigest() == (
            implementation["module_sha256"]
        )
        assert hashlib.sha256(runner_path.read_bytes()).hexdigest() == (
            implementation["runner_sha256"]
        )

    p100 = next(
        row
        for row in gated["nested_future_blind_nomination"]["transitions"]
        if row["current_prefix_cycle"] == 100
    )
    assert p100["activation_count"] == 10
    assert p100["activation_coverage"] == pytest.approx(10 / 41)
    assert p100["activation_precision"] == pytest.approx(0.8)
    assert p100["mean_gated_delta_mae_pp"] == pytest.approx(
        -0.019628500544838452
    )
    assert p100["active_p90_delta_mae_pp"] == pytest.approx(
        0.03202154032657387
    )


def test_p100_blind_candidate_is_frozen_without_new_outcomes() -> None:
    protocol = json.loads(BLIND_CANDIDATE_PATH.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(GATED_PROTOCOL_PATH.read_bytes()).hexdigest()
    update = protocol["frozen_update_rule"]
    requirements = protocol["blind_data_requirements"]
    ungated_decision = ROOT / "showcase/evidence_v6/ungated_state_decision.json"
    gated_development = _gated_protocol()

    assert "frozen_after_training_development" in protocol["status"]
    assert protocol["source_development_protocol"]["sha256"] == source_hash
    assert gated_development["development_lineage"][
        "parent_local_decision_sha256"
    ] == hashlib.sha256(ungated_decision.read_bytes()).hexdigest()
    assert protocol["eligible_transition"]["current_prefix_cycle"] == 100
    assert update["require_slope_sign_agreement"] is True
    assert update["minimum_absolute_projected_history_change_pp"] == 0.04
    assert update["absolute_correction_cap_pp"] == 1.0
    assert requirements["same_batch_retuning_permitted"] is False
    assert requirements["exposed_81_cell_public_evaluation_permitted"] is False
    assert (
        requirements[
            "outcomes_hidden_until_predictions_and_activation_flags_are_hash_committed"
        ]
        is True
    )
