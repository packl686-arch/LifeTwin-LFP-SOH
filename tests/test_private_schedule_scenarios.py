from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.private_schedule_scenarios import (
    PrivateScheduleScenarioError,
    build_private_schedule_scenarios,
    validate_private_schedule_scenario_config,
)
from lifetwin.experiments.private_schedule_v4 import FORECAST_SCHEDULE_COLUMNS
from lifetwin.validation.private_cycle_adapter import PARTITIONED_PREFIX_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/private_schedule_scenarios_v1.json"


def _prefixes() -> pd.DataFrame:
    core = []
    for visit in range(4):
        core.append(
            {
                "dataset_id": "synthetic",
                "cell_id": "cell_a",
                "condition_id": "condition_a",
                "temperature_c": 25.0,
                "min_soc_pct": 10.0,
                "max_soc_pct": 90.0,
                "dod_fraction": 0.8,
                "charge_c_rate": 0.5,
                "discharge_c_rate": 1.0,
                "visit_index": visit,
                "elapsed_days": float(visit * 50),
                "equivalent_full_cycles": float(visit * 250),
                "capacity_ah": 280.0 * (100.0 - visit) / 100.0,
                "capacity_retention_pct": 100.0 - visit,
                "rpt_cycle_count": 1,
            }
        )
    trajectory = pd.DataFrame(core, columns=RPT_TRAJECTORY_COLUMNS)
    rows = []
    for landmark in (3, 4):
        selected = trajectory.iloc[:landmark].copy()
        selected.insert(0, "partition", "calibration")
        selected.insert(3, "landmark_visit_count", landmark)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).loc[:, PARTITIONED_PREFIX_COLUMNS]


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_outcome_free_scenarios_cover_each_landmark_and_frozen_grid() -> None:
    model = {
        "landmark_visit_counts": [3, 4],
        "score_end_equivalent_full_cycles": 1250.0,
        "forecast_grid_step_equivalent_full_cycles": 250.0,
    }
    schedules, manifest = build_private_schedule_scenarios(
        _prefixes(), model, _config()
    )
    assert set(schedules) == {
        "reference_plan",
        "cool_low_utilization",
        "hot_high_utilization",
    }
    assert manifest["capacity_outcome_fields_accepted"] is False
    assert manifest["truth_vault_opened"] is False
    for schedule in schedules.values():
        assert tuple(schedule.columns) == FORECAST_SCHEDULE_COLUMNS
        assert not any("capacity" in column for column in schedule.columns)
    reference = schedules["reference_plan"]
    cool = schedules["cool_low_utilization"]
    hot = schedules["hot_high_utilization"]
    assert (cool["forecast_elapsed_days"] > reference["forecast_elapsed_days"]).all()
    assert (hot["forecast_elapsed_days"] < reference["forecast_elapsed_days"]).all()
    assert (hot["planned_temperature_c"] > reference["planned_temperature_c"]).all()


def test_scenario_config_rejects_duplicate_identity_and_outcome_field() -> None:
    config = _config()
    config["scenarios"][1]["scenario_id"] = "reference_plan"
    with pytest.raises(PrivateScheduleScenarioError, match="identity"):
        validate_private_schedule_scenario_config(config)
    leaked = _config()
    leaked["scenarios"][0]["capacity_retention_pct"] = 90.0
    with pytest.raises(PrivateScheduleScenarioError, match="fields changed"):
        validate_private_schedule_scenario_config(leaked)
