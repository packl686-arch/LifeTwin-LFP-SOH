from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from lifetwin.data.snl import RPT_REPEAT_COLUMNS
from lifetwin.experiments.fastcharge_v10_snl_rpt_noise import (
    characterize_snl_rpt_repeatability,
    validate_repeat_measurements,
)
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


def _config() -> dict[str, object]:
    return {
        "experiment_id": "unit_v10",
        "repeat_contract": {
            "minimum_physical_cell_count": 3,
            "minimum_visits_per_cell": 2,
            "minimum_repeats_per_visit": 3,
        },
        "noise_model_selection": {
            "candidate_families": [
                {"model_id": "gaussian", "distribution": "gaussian"},
                {
                    "model_id": "student_t_df5",
                    "distribution": "student_t",
                    "degrees_of_freedom": 5,
                },
            ],
            "minimum_scale_pp": 1e-6,
            "tie_tolerance_log_score": 1e-9,
            "tie_break_order": ["gaussian", "student_t_df5"],
        },
        "engineering_gates": {
            "maximum_repeatability_scale_pp": 0.2,
            "maximum_median_absolute_repeat_order_slope_pp_per_repeat": 0.2,
        },
        "identifiability": {
            "available_components": ["within_visit_repeatability"],
            "unavailable_components": ["daily_reference_common_bias"],
        },
        "claim_boundaries": ["component_only"],
    }


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260811)
    rows: list[dict[str, object]] = []
    for cell_index in range(3):
        for visit_index in range(2):
            center = 100.0 - 0.4 * visit_index - 0.05 * cell_index
            values = center + rng.normal(0.0, 0.03, size=3)
            visit_center = float(np.median(values))
            for repeat_index, value in enumerate(values):
                rows.append(
                    {
                        "dataset_id": "snl",
                        "cell_id": f"CELL_{cell_index}",
                        "condition_id": f"COND_{cell_index % 2}",
                        "visit_index": visit_index,
                        "repeat_index": repeat_index,
                        "source_cycle_index": 10 * visit_index + repeat_index + 1,
                        "measurement_time": (
                            f"2026-01-{visit_index + 1:02d}T0{repeat_index}:00:00"
                        ),
                        "elapsed_days": float(visit_index),
                        "equivalent_full_cycles": float(visit_index * 10),
                        "capacity_ah": 1.1 * float(value) / 100.0,
                        "retention_pct": float(value),
                        "visit_center_capacity_ah": 1.1 * visit_center / 100.0,
                        "visit_center_retention_pct": visit_center,
                        "rpt_cycle_count": 3,
                    }
                )
    return pd.DataFrame(rows, columns=RPT_REPEAT_COLUMNS)


def test_characterization_is_deterministic_and_remains_component_only() -> None:
    first = characterize_snl_rpt_repeatability(_frame(), _config())
    second = characterize_snl_rpt_repeatability(_frame(), _config())
    assert first[-1] == second[-1]
    decision = first[-1]
    assert decision["repeatability_component_gates"]["passed"] is True
    assert decision["full_measurement_model_identified"] is False
    assert decision["eligible_for_full_v9_qualification"] is False
    assert decision["future_outcomes_used_for_noise_estimation"] is False
    assert decision["public_aggregate_release_permitted"] is False


def test_repeat_schema_rejects_future_or_unregistered_columns() -> None:
    bad = _frame().assign(future_capacity_pct=80.0)
    with pytest.raises(FastChargeV5PairwiseError, match="columns changed"):
        validate_repeat_measurements(bad, _config())


def test_repeat_contract_fails_closed_when_one_visit_has_too_few_repeats() -> None:
    bad = _frame().drop(index=[0]).reset_index(drop=True)
    with pytest.raises(FastChargeV5PairwiseError, match="too few repeats"):
        validate_repeat_measurements(bad, _config())


def test_invalid_student_t_candidate_is_rejected() -> None:
    config = deepcopy(_config())
    config["noise_model_selection"]["candidate_families"][1]["degrees_of_freedom"] = 2
    with pytest.raises(FastChargeV5PairwiseError, match="df > 2"):
        characterize_snl_rpt_repeatability(_frame(), config)
