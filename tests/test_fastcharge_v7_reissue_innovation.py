from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import fastcharge_v7_reissue_innovation as innovation
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/experiments/v7_reissue_innovation_development.json"
)
BLIND_CANDIDATE_PATH = (
    ROOT / "configs/experiments/v7_p100_reissue_innovation_blind_candidate.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _synthetic_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for batch in (1, 2):
        for cell_index in range(4):
            cell_id = f"MATR_B{batch}C{cell_index}"
            for prefix in (20, 40, 60, 100):
                for cycle in range(prefix + 1, 141):
                    truth = 100.0 - 0.01 * cycle - 0.01 * cell_index
                    residual = 0.004 * (cycle - prefix)
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


def test_protocol_keeps_v5_frozen_and_excludes_exposed_outcomes() -> None:
    protocol = _protocol()
    firewall = protocol["data_firewall"]

    assert protocol["champion"]["modification_permitted"] is False
    assert (
        protocol["champion"][
            "activation_before_new_blind_confirmation_permitted"
        ]
        is False
    )
    assert firewall["outer_held_out_future_outcomes_available_to_gate_selection"] is False
    assert (
        firewall["future_target_outcomes_available_to_correction_or_activation"]
        is False
    )
    assert firewall["exposed_public_evaluation_outcomes_permitted"] is False
    assert (
        protocol["inputs"]["exposed_public_evaluation_predictions"][
            "permitted_use"
        ]
        == "none"
    )


def test_innovation_gate_library_is_bounded_unique_and_deterministic() -> None:
    gates = innovation.innovation_gates(_protocol())

    assert len(gates) == 19
    assert gates[0].gate_id == "no_update"
    assert len({gate.gate_id for gate in gates}) == len(gates)
    assert all(gate.require_history_slope_sign_agreement for gate in gates[1:])
    assert all(
        gate.require_innovation_history_sign_agreement for gate in gates[1:]
    )


def test_reissue_innovation_subtracts_absorbed_slope_and_enforces_cap() -> None:
    protocol = _protocol()
    history_cycles = np.arange(21, 41, dtype=float)
    history_residuals = 0.01 * history_cycles
    future_cycles = np.arange(41, 301, dtype=float)
    previous_center = np.zeros(len(future_cycles))
    current_center = 0.006 * future_cycles

    correction, diagnostics = innovation.reissue_innovation_correction(
        history_cycles,
        history_residuals,
        future_cycles,
        previous_center,
        current_center,
        previous_prefix=20,
        current_prefix=40,
        projection_scale=0.5,
        config=protocol,
    )

    assert diagnostics["history_theil_slope_pp_per_cycle"] == pytest.approx(0.01)
    assert diagnostics["reissue_shift_theil_slope_pp_per_cycle"] == pytest.approx(
        0.006
    )
    assert diagnostics[
        "bounded_unassimilated_slope_pp_per_cycle"
    ] == pytest.approx(0.004)
    assert correction[0] == pytest.approx(0.002)
    assert np.max(np.abs(correction)) <= 1.0

    with pytest.raises(FastChargeV5PairwiseError, match="landmark firewall"):
        innovation.reissue_innovation_correction(
            np.arange(21, 42, dtype=float),
            np.zeros(21),
            future_cycles,
            previous_center,
            current_center,
            previous_prefix=20,
            current_prefix=40,
            projection_scale=0.5,
            config=protocol,
        )


def test_correction_api_has_no_future_outcome_argument() -> None:
    parameters = inspect.signature(
        innovation.reissue_innovation_correction
    ).parameters

    assert "future_outcomes" not in parameters
    assert "future_truth" not in parameters
    assert set(parameters) == {
        "history_cycles",
        "history_residuals",
        "future_cycles",
        "previous_future_center",
        "current_future_center",
        "previous_prefix",
        "current_prefix",
        "projection_scale",
        "config",
    }


def test_gate_predicate_does_not_read_held_out_outcome_delta() -> None:
    protocol = _protocol()
    table = innovation.score_innovation_candidates(
        _synthetic_predictions(), protocol
    )
    gate = next(
        gate
        for gate in innovation.innovation_gates(protocol)
        if gate.gate_id == "a0p5_d0p01_r0p0"
    )
    selected = table.loc[table["current_prefix_cycle"] == 100].copy()
    _, original = innovation._gate_rows(selected, gate)
    selected["raw_delta_mae_pp"] = 999.0
    selected["updated_trajectory_mae_pp"] = -999.0
    _, attacked = innovation._gate_rows(selected, gate)

    assert np.array_equal(original, attacked)


def test_nested_cell_and_batch_audits_keep_holdouts_out_of_selection() -> None:
    protocol = _protocol()
    table = innovation.score_innovation_candidates(
        _synthetic_predictions(), protocol
    )
    nested = innovation.nested_innovation_gate_audit(table, protocol)
    batch = innovation.batch_holdout_innovation_gate_audit(table, protocol)
    nomination = innovation.innovation_nomination_summary(
        nested, batch, protocol
    )

    assert len(nested) == 8 * 3
    assert nested["selection_fit_cell_count"].eq(7).all()
    assert len(batch) == 8 * 3
    assert set(batch["held_out_batch"]) == {"1", "2"}
    assert set(batch["selection_fit_cell_count"]) == {4}
    assert (nested.loc[~nested["activated"], "gated_delta_mae_pp"] == 0.0).all()
    assert nomination["any_transition_nominated"] is True
    json.dumps(nomination, allow_nan=False)


def test_published_v7_decision_matches_protocol_and_claim_boundaries() -> None:
    decision_path = ROOT / "showcase/evidence_v7/reissue_innovation_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["v5_champion_remains_active"] is True
    assert decision["v7_innovation_gate_activated"] is False
    assert decision["exposed_81_cell_evaluation_used"] is False
    assert decision["nominated_current_prefix_cycles"] == [100]
    assert decision["runtime_versions"]["python"] == "3.12.13"
    assert decision["protocol_sha256"] == hashlib.sha256(
        PROTOCOL_PATH.read_bytes()
    ).hexdigest()

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
        for row in decision["nested_and_batch_future_blind_nomination"][
            "transitions"
        ]
        if row["current_prefix_cycle"] == 100
    )
    cell = p100["cell_holdout"]
    assert cell["activation_count"] == 9
    assert cell["activation_precision"] == pytest.approx(1.0)
    assert cell["mean_gated_delta_mae_pp"] == pytest.approx(
        -0.03731617477042056
    )
    assert cell["active_maximum_delta_mae_pp"] == pytest.approx(
        -0.02409879411348157
    )
    assert p100["batch_holdout"]["passed"] is True


def test_v7_p100_blind_candidate_is_frozen_without_new_outcomes() -> None:
    protocol = json.loads(BLIND_CANDIDATE_PATH.read_text(encoding="utf-8"))
    update = protocol["frozen_update_rule"]
    requirements = protocol["blind_data_requirements"]
    result_path = ROOT / protocol["source_development_result"]["public_path"]

    assert "frozen_after_training_development" in protocol["status"]
    assert protocol["source_development_protocol"]["sha256"] == hashlib.sha256(
        PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    assert protocol["source_development_result"]["sha256"] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    assert protocol["eligible_transition"]["current_prefix_cycle"] == 100
    assert update["projection_scale"] == 0.5
    assert update["minimum_projected_unassimilated_change_pp"] == 0.01
    assert update["require_history_slope_sign_agreement"] is True
    assert update["require_unassimilated_history_slope_sign_agreement"] is True
    assert update["absolute_correction_cap_pp"] == 1.0
    assert requirements["same_batch_retuning_permitted"] is False
    assert requirements["exposed_81_cell_public_evaluation_permitted"] is False
    assert (
        requirements[
            "outcomes_hidden_until_predictions_and_activation_flags_are_hash_committed"
        ]
        is True
    )
