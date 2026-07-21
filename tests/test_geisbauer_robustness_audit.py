from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from lifetwin.data.geisbauer_calendar import load_geisbauer_calendar_observations
from lifetwin.experiments.geisbauer_robustness_audit import (
    DESIGN_STATUS,
    INFERENCE_STATUS,
    default_geisbauer_robustness_audit_protocol,
    exact_two_sided_mean_sign_flip_test,
    exact_two_sided_sign_test,
    run_geisbauer_robustness_audit,
    validate_geisbauer_robustness_audit_protocol,
)
from scripts import run_geisbauer_robustness_audit as audit_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = PROJECT_ROOT / "data/external/geisbauer_2022/LFP_Data.csv"
EXTERNAL_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs/experiments/geisbauer_lfp_calendar_external_stress.json"
)
AUDIT_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs/experiments/geisbauer_lfp_calendar_robustness_audit.json"
)
SOURCE_PATH = PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv"


@pytest.fixture
def writable_root() -> Path:
    root = PROJECT_ROOT / "artifacts/test-scratch" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


@pytest.fixture(scope="module")
def robustness_inputs() -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    target, _ = load_geisbauer_calendar_observations(TARGET_PATH)
    external_protocol = json.loads(EXTERNAL_PROTOCOL_PATH.read_text(encoding="utf-8"))
    audit_protocol = json.loads(AUDIT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    return target, external_protocol, audit_protocol


@pytest.fixture(scope="module")
def robustness_run(
    observations: pd.DataFrame,
    robustness_inputs: tuple[pd.DataFrame, dict[str, object], dict[str, object]],
) -> tuple:
    target, external_protocol, audit_protocol = robustness_inputs
    return run_geisbauer_robustness_audit(
        observations,
        target,
        external_protocol=external_protocol,
        audit_protocol=audit_protocol,
    )


def test_audit_protocol_is_exactly_locked(
    robustness_inputs: tuple[pd.DataFrame, dict[str, object], dict[str, object]],
) -> None:
    audit_protocol = robustness_inputs[2]
    assert validate_geisbauer_robustness_audit_protocol(audit_protocol) == (
        default_geisbauer_robustness_audit_protocol()
    )
    assert audit_protocol["numerical_zero_tolerance_pp"] == 1e-12
    assert audit_protocol["sensitivity"]["post_hoc_equivalence_margin_sensitivity"][
        "margins_pp"
    ] == [0.0, 0.01, 0.05, 0.1]
    assert (
        audit_protocol["sensitivity"]["leave_one_physical_cell_out"][
            "scenarios_are_independent_replications"
        ]
        is False
    )
    changed = {
        **audit_protocol,
        "claim_boundaries": {
            **audit_protocol["claim_boundaries"],
            "long_term_validation_eligible": True,
        },
    }
    with pytest.raises(ValueError, match="protocol changed"):
        validate_geisbauer_robustness_audit_protocol(changed)


def test_exact_cell_level_diagnostics_use_numerical_zeros_only_for_filtering() -> None:
    sign = exact_two_sided_sign_test([1.0, 2.0, 3.0])
    assert sign == {
        "negative_count": 0,
        "positive_count": 3,
        "numerical_zero_count": 0,
        "nonzero_count": 3,
        "two_sided_p": 0.25,
    }
    flip = exact_two_sided_mean_sign_flip_test([1.0, 1.0, 1.0])
    assert flip["nonzero_count"] == 3
    assert flip["permutation_count"] == 8
    assert flip["observed_mean_delta_pp"] == 1.0
    assert flip["extreme_threshold_pp"] == abs(flip["observed_mean_delta_pp"])
    assert flip["inclusive_comparison_machine_epsilon"] == np.finfo(float).eps
    assert flip["two_sided_p"] == 0.25
    numerical_zeros = exact_two_sided_sign_test([0.0, 1e-13, -1e-13])
    assert numerical_zeros["numerical_zero_count"] == 3
    assert numerical_zeros["nonzero_count"] == 0
    assert numerical_zeros["two_sided_p"] == 1.0
    near_zero_but_included = exact_two_sided_mean_sign_flip_test(
        [1.1e-12, 1.1e-12, 1.1e-12]
    )
    assert near_zero_but_included["extreme_threshold_pp"] == 1.1e-12
    assert near_zero_but_included["two_sided_p"] == 0.25
    with pytest.raises(ValueError, match="exceeds"):
        exact_two_sided_mean_sign_flip_test(np.ones(21), maximum_exact_units=20)


def test_real_audit_preserves_cell_unit_and_exposes_fragility(
    robustness_run: tuple,
) -> None:
    result, cell_deltas, cell_day_deltas, strata, leave_one_out = robustness_run
    assert result["design_status"] == DESIGN_STATUS
    assert result["inference_status"] == INFERENCE_STATUS
    assert result["numerical_zero_tolerance_pp"] == 1e-12
    assert len(cell_deltas) == cell_deltas["cell_id"].nunique() == 15
    assert len(cell_day_deltas) == 30
    assert not cell_day_deltas.duplicated(["cell_id", "target_elapsed_days"]).any()
    assert len(strata) == 12
    assert len(leave_one_out) == 15
    assert set(leave_one_out["remaining_physical_cell_count"]) == {14}
    assert leave_one_out["direction_flipped_from_full_sample"].sum() == 2
    assert "candidate_numerical_zero_count" in strata.columns
    assert "exact_mean_sign_flip_extreme_threshold_pp" in strata.columns

    overall = result["overall_paired_diagnostic"]
    assert overall["candidate_error_mean_pp"] == pytest.approx(3.973450583)
    assert overall["comparator_error_mean_pp"] == pytest.approx(3.885214664)
    assert overall["mean_paired_delta_pp"] == pytest.approx(0.088235919)
    assert overall["median_paired_delta_pp"] < 0.0
    assert overall["candidate_better_cell_count"] == 8
    assert overall["candidate_worse_cell_count"] == 7
    assert overall["exact_sign_test_two_sided_p"] == 1.0
    assert overall["exact_mean_sign_flip_two_sided_p"] == pytest.approx(
        0.57501220703125
    )
    assert overall["exact_mean_sign_flip_extreme_threshold_pp"] == abs(
        overall["exact_mean_sign_flip_observed_mean_delta_pp"]
    )
    assert overall["exact_mean_sign_flip_inclusive_comparison_epsilon"] == (
        np.finfo(float).eps
    )
    assert not overall["nominal_diagnostics_are_confirmatory"]

    sensitivity = result["leave_one_cell_out"]
    assert sensitivity["minimum_mean_paired_delta_pp"] < 0.0
    assert sensitivity["maximum_mean_paired_delta_pp"] > 0.0
    assert sensitivity["candidate_better_direction_count"] == 2
    assert sensitivity["candidate_worse_direction_count"] == 13
    assert sensitivity["scenario_count"] == 15
    assert sensitivity["scenarios_are_highly_overlapping"] is True
    assert sensitivity["scenarios_are_independent_replications"] is False
    assert sensitivity["direction_flip_scenario_count"] == 2

    equivalence = result["post_hoc_equivalence_margin_sensitivity"]
    assert equivalence["engineering_acceptance_gate"] is False
    assert "not engineering acceptance thresholds" in equivalence["interpretation"]
    margins = equivalence["margins"]
    assert [row["equivalence_margin_pp"] for row in margins] == [
        0.0,
        0.01,
        0.05,
        0.1,
    ]
    assert [
        (
            row["candidate_better_cell_count"],
            row["candidate_worse_cell_count"],
            row["equivalent_cell_count"],
        )
        for row in margins
    ] == [(8, 7, 0), (5, 7, 3), (5, 4, 6), (5, 4, 6)]
    assert all(
        row["candidate_better_cell_count"]
        + row["candidate_worse_cell_count"]
        + row["equivalent_cell_count"]
        == 15
        for row in margins
    )


def test_negative_transfer_is_localised_to_fallback_and_high_soc_mean(
    robustness_run: tuple,
) -> None:
    result, _, _, strata, _ = robustness_run
    route = result["route_reality"]
    assert route["candidate_exactly_equals_hierarchical_power_fallback"]
    assert route["activation_gate_ready_physical_cell_count"] == 0
    assert not route["activation_specialist_tested"]
    diagnosis = result["negative_transfer_diagnosis"]
    assert diagnosis["aggregate_mean_negative_transfer_observed"]
    assert diagnosis["physical_cells_with_negative_transfer"] == 7
    assert diagnosis["soc_strata_with_candidate_better_mean"] == [
        "soc_0.2",
        "soc_0.5",
    ]
    assert diagnosis["soc_strata_with_negative_transfer_mean"] == ["soc_1.0"]
    assert diagnosis["mean_median_direction_conflict"]
    assert diagnosis["soc_strata_with_mean_median_direction_conflict"] == ["soc_0.2"]
    assert (
        diagnosis["soc_strata_cell_outcomes"]["soc_0.2"]["candidate_worse_cell_count"]
        == 3
    )
    assert (
        diagnosis["soc_strata_cell_outcomes"]["soc_1.0"]["candidate_worse_cell_count"]
        == 4
    )
    soc = strata.loc[strata["scope_type"] == "storage_soc_fraction"].set_index("scope")
    assert soc.loc["soc_0.2", "mean_paired_delta_pp"] < 0.0
    assert soc.loc["soc_0.5", "mean_paired_delta_pp"] < 0.0
    assert soc.loc["soc_1.0", "mean_paired_delta_pp"] > 0.0
    assert not result["claim_boundary"]["confirmatory_inference_allowed"]
    assert not result["claim_boundary"][
        "independent_long_term_validation_claim_allowed"
    ]


def test_audit_rejects_changed_target_outcome_snapshot(
    observations: pd.DataFrame,
    robustness_inputs: tuple[pd.DataFrame, dict[str, object], dict[str, object]],
) -> None:
    target, external_protocol, audit_protocol = robustness_inputs
    changed = target.copy()
    row = changed.index[-1]
    changed.loc[row, "capacity_ah"] *= 0.99
    cell_id = changed.loc[row, "cell_id"]
    initial = float(
        changed.loc[
            (changed["cell_id"] == cell_id) & (changed["checkup_index"] == 0),
            "capacity_ah",
        ].iloc[0]
    )
    changed.loc[row, "capacity_retention_pct"] = (
        100.0 * float(changed.loc[row, "capacity_ah"]) / initial
    )
    changed.loc[row, "capacity_loss_pct"] = (
        100.0 - changed.loc[row, "capacity_retention_pct"]
    )
    with pytest.raises(ValueError, match="target outcome snapshot mismatch"):
        run_geisbauer_robustness_audit(
            observations,
            changed,
            external_protocol=external_protocol,
            audit_protocol=audit_protocol,
        )


def test_runner_writes_hashed_audit_artifacts_without_overwrite(
    writable_root: Path,
) -> None:
    output_dir = writable_root / "robustness"
    result = audit_runner.run(
        SOURCE_PATH,
        TARGET_PATH,
        EXTERNAL_PROTOCOL_PATH,
        AUDIT_PROTOCOL_PATH,
        output_dir,
    )
    expected_rows = {
        "cell_paired_deltas": 15,
        "cell_day_paired_deltas": 30,
        "stratum_diagnostics": 12,
        "leave_one_cell_out": 15,
    }
    assert set(result["artifacts"]) == set(expected_rows)
    for name, row_count in expected_rows.items():
        metadata = result["artifacts"][name]
        path = Path(metadata["path"])
        assert path.is_file()
        assert metadata["row_count"] == row_count
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    parsed = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert parsed["design_status"] == DESIGN_STATUS
    assert parsed["scope"]["maximum_observed_days"] == 120.0
    assert (
        parsed["claim_boundary"]["independent_long_term_validation_claim_allowed"]
        is False
    )
    with pytest.raises(FileExistsError, match="never overwrites"):
        audit_runner.run(
            SOURCE_PATH,
            TARGET_PATH,
            EXTERNAL_PROTOCOL_PATH,
            AUDIT_PROTOCOL_PATH,
            output_dir,
        )
