from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import fastcharge_v7_prefix_robustness as robustness


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiments/v7_frozen_gate_prefix_robustness_audit.json"
CANDIDATE_PATH = (
    ROOT / "configs/experiments/v7_p100_reissue_innovation_blind_candidate.json"
)
PUBLISHED_DECISION_PATH = ROOT / "showcase/evidence_v7_robustness/decision.json"
V8_TEMPLATE_PATH = (
    ROOT / "configs/experiments/v8_measurement_stability_blind_protocol.template.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))


def _synthetic_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_index in range(8):
        cell_id = f"CELL_{cell_index:02d}"
        history_slope = 0.005 if cell_index < 4 else 0.0001
        current_slope = 0.003 if cell_index < 4 else 0.0001
        for cycle in range(61, 141):
            truth = 100.0 - 0.01 * cycle - 0.01 * cell_index
            rows.append(
                {
                    "cell_id": cell_id,
                    "prefix_cycle": 60,
                    "forecast_cycle": cycle,
                    "observed_retention_pct": truth,
                    "candidate_prediction_pct": (truth - history_slope * (cycle - 60)),
                }
            )
        for cycle in range(101, 141):
            truth = 100.0 - 0.01 * cycle - 0.01 * cell_index
            rows.append(
                {
                    "cell_id": cell_id,
                    "prefix_cycle": 100,
                    "forecast_cycle": cycle,
                    "observed_retention_pct": truth,
                    "candidate_prediction_pct": (truth - current_slope * (cycle - 100)),
                }
            )
    return pd.DataFrame(rows)


def _small_protocol() -> dict[str, object]:
    protocol = copy.deepcopy(_protocol())
    protocol["input"]["physical_cell_count"] = 8
    protocol["scenarios"] = [
        {
            "scenario_id": "reference_unperturbed",
            "kind": "none",
            "replicates": 1,
            "role": "integrity",
            "thresholds": {
                "minimum_decision_agreement": 1.0,
                "maximum_p95_effective_correction_deviation_pp": 1e-12,
            },
        },
        {
            "scenario_id": "constant_offset_p0p20_pp",
            "kind": "constant_offset",
            "offset_pp": 0.2,
            "replicates": 1,
            "role": "integrity",
            "thresholds": {
                "minimum_decision_agreement": 1.0,
                "maximum_p95_effective_correction_deviation_pp": 1e-10,
            },
        },
    ]
    return protocol


def test_protocol_binds_frozen_candidate_and_forbids_retuning() -> None:
    protocol = _protocol()

    assert protocol["source_candidate"]["modification_permitted"] is False
    assert protocol["firewall"]["threshold_retuning_after_results_permitted"] is False
    assert (
        protocol["firewall"]["candidate_rule_retuning_after_results_permitted"] is False
    )
    assert protocol["input"]["exposed_public_81_cell_evaluation_permitted"] is False
    assert (
        protocol["source_candidate"]["sha256"]
        == hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()
    )


def test_frozen_gate_api_has_no_future_outcome_argument() -> None:
    parameters = inspect.signature(robustness.frozen_gate_update).parameters

    assert "future_truth" not in parameters
    assert "future_outcomes" not in parameters
    assert set(parameters) == {
        "history_cycles",
        "history_residuals",
        "future_cycles",
        "previous_future_center",
        "current_future_center",
        "candidate",
    }


def test_constant_residual_offset_is_an_exact_slope_gate_invariance() -> None:
    candidate = _candidate()
    history_cycles = np.arange(61, 101, dtype=float)
    history_residuals = 0.004 * (history_cycles - 60.0)
    future_cycles = np.arange(101, 141, dtype=float)
    previous_center = 100.0 - 0.01 * future_cycles
    current_center = previous_center + 0.001 * future_cycles

    base, base_active, _ = robustness.frozen_gate_update(
        history_cycles,
        history_residuals,
        future_cycles,
        previous_center,
        current_center,
        candidate,
    )
    shifted, shifted_active, _ = robustness.frozen_gate_update(
        history_cycles,
        history_residuals + 0.2,
        future_cycles,
        previous_center,
        current_center,
        candidate,
    )

    assert shifted_active == base_active
    assert np.allclose(shifted, base, atol=1e-12, rtol=0.0)


def test_future_truth_attack_cannot_change_perturbation_decisions() -> None:
    protocol = _small_protocol()
    frame = _synthetic_predictions()
    attacked = frame.copy()
    attacked.loc[attacked["forecast_cycle"] > 100, "observed_retention_pct"] += 50.0

    _, original, _, _ = robustness.run_frozen_prefix_robustness(
        frame, protocol, _candidate()
    )
    _, changed, _, _ = robustness.run_frozen_prefix_robustness(
        attacked, protocol, _candidate()
    )
    decision_columns = [
        "scenario_id",
        "replicate_index",
        "cell_id",
        "activated",
        "history_theil_slope_pp_per_cycle",
        "reissue_shift_theil_slope_pp_per_cycle",
        "bounded_unassimilated_slope_pp_per_cycle",
    ]

    pd.testing.assert_frame_equal(original[decision_columns], changed[decision_columns])
    assert not np.allclose(
        original["base_trajectory_mae_pp"],
        changed["base_trajectory_mae_pp"],
    )


def test_robustness_audit_is_row_order_invariant_and_keeps_v5_active() -> None:
    frame = _synthetic_predictions()
    protocol = _small_protocol()
    baseline, decisions, summaries, decision = robustness.run_frozen_prefix_robustness(
        frame, protocol, _candidate()
    )
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    baseline_2, decisions_2, summaries_2, decision_2 = (
        robustness.run_frozen_prefix_robustness(shuffled, protocol, _candidate())
    )

    pd.testing.assert_frame_equal(baseline, baseline_2)
    pd.testing.assert_frame_equal(decisions, decisions_2)
    pd.testing.assert_frame_equal(summaries, summaries_2)
    assert decision == decision_2
    assert decision["all_required_scenarios_passed"] is True
    assert decision["v5_champion_remains_active"] is True
    assert decision["v7_candidate_activated"] is False
    assert baseline["activated"].sum() == 4
    assert summaries["passed"].eq(True).all()


@pytest.mark.parametrize(
    ("scenario", "expected_count"),
    [
        ({"kind": "random_missing", "missing_fraction": 0.1}, 36),
        ({"kind": "recent_block_missing", "missing_count": 5}, 35),
    ],
)
def test_registered_missingness_never_uses_future_rows(
    scenario: dict[str, object], expected_count: int
) -> None:
    cycles = np.arange(61, 101, dtype=float)
    residuals = np.linspace(0.0, 0.2, len(cycles))
    perturbed_cycles, perturbed = robustness.perturb_history(
        cycles,
        residuals,
        scenario,
        np.random.default_rng(9),
    )

    assert len(perturbed_cycles) == expected_count
    assert len(perturbed) == expected_count
    assert perturbed_cycles.min() >= 61
    assert perturbed_cycles.max() <= 100


def test_published_robustness_failure_is_bound_and_does_not_activate_v7() -> None:
    result = json.loads(PUBLISHED_DECISION_PATH.read_text(encoding="utf-8"))

    assert (
        result["protocol_sha256"]
        == hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    )
    assert (
        result["frozen_candidate_sha256"]
        == hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()
    )
    assert result["baseline"]["activation_count"] == 9
    assert result["baseline"]["activation_precision"] == pytest.approx(1.0)
    assert result["all_required_scenarios_passed"] is False
    assert set(result["failed_required_scenarios"]) == {
        "iid_noise_sigma_0p02_pp",
        "iid_noise_sigma_0p05_pp",
        "noise_0p05_pp_plus_missing_10pct",
    }
    assert result["v5_champion_remains_active"] is True
    assert result["v7_candidate_activated"] is False
    assert result["exposed_81_cell_evaluation_used"] is False

    implementation = result["implementation"]
    assert (
        implementation["module_sha256"]
        == hashlib.sha256(
            (ROOT / implementation["module_path"]).read_bytes()
        ).hexdigest()
    )
    assert (
        implementation["runner_sha256"]
        == hashlib.sha256(
            (ROOT / implementation["runner_path"]).read_bytes()
        ).hexdigest()
    )


def test_v8_protocol_is_blocked_until_new_measurement_and_cell_data_exist() -> None:
    protocol = json.loads(V8_TEMPLATE_PATH.read_text(encoding="utf-8"))

    assert protocol["status"].startswith("blocked_template")
    assert (
        protocol["lineage"]["same_41_cell_outcomes_permitted_for_v8_selection"] is False
    )
    assert protocol["lineage"]["exposed_81_cell_public_evaluation_permitted"] is False
    assert (
        protocol["stage_b_outcome_free_stability_issuance"]["future_outcomes_available"]
        is False
    )
    assert (
        protocol["stage_b_outcome_free_stability_issuance"][
            "minimum_new_physical_cell_count"
        ]
        >= 60
    )
    assert protocol["stopping_rules"]["production_activation_permitted"] is False
