"""Outcome-free operating-scenario schedules for private forecasts."""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.private_schedule_v4 import (
    FORECAST_SCHEDULE_COLUMNS,
    validate_private_forecast_schedule,
)
from lifetwin.validation.private_cycle_adapter import PARTITIONED_PREFIX_COLUMNS


SCENARIO_SCHEMA = "lifetwin.private_schedule_scenarios.v1"
SCENARIO_FIELDS = {
    "scenario_id",
    "scenario_role",
    "temperature_delta_c",
    "planned_min_soc_pct",
    "planned_max_soc_pct",
    "charge_c_rate_multiplier",
    "discharge_c_rate_multiplier",
    "duty_rate_multiplier",
}


class PrivateScheduleScenarioError(ValueError):
    """Raised when an outcome-free scenario definition is invalid."""


def validate_private_schedule_scenario_config(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate scenario fields without accepting capacity outcomes."""
    config = deepcopy(dict(value))
    if set(config) != {
        "schema_version",
        "protocol_id",
        "status",
        "scenarios",
        "claim_boundary",
    }:
        raise PrivateScheduleScenarioError("Scenario config fields changed")
    if config["schema_version"] != SCENARIO_SCHEMA:
        raise PrivateScheduleScenarioError("Scenario config schema changed")
    if config["status"] != "frozen_before_hithium_data_access":
        raise PrivateScheduleScenarioError("Scenario config status changed")
    scenarios = config["scenarios"]
    if not isinstance(scenarios, list) or not 2 <= len(scenarios) <= 5:
        raise PrivateScheduleScenarioError("Scenario count must be between 2 and 5")
    identities: set[str] = set()
    reference_count = 0
    for scenario in scenarios:
        if not isinstance(scenario, Mapping) or set(scenario) != SCENARIO_FIELDS:
            raise PrivateScheduleScenarioError("Scenario fields changed")
        scenario_id = str(scenario["scenario_id"])
        if (
            re.fullmatch(r"[a-z0-9_]+", scenario_id) is None
            or scenario_id in identities
        ):
            raise PrivateScheduleScenarioError("Scenario identity is invalid")
        identities.add(scenario_id)
        role = str(scenario["scenario_role"])
        if role not in {"reference", "stress"}:
            raise PrivateScheduleScenarioError("Scenario role is unsupported")
        reference_count += int(role == "reference")
        numeric_fields = SCENARIO_FIELDS - {"scenario_id", "scenario_role"}
        numbers = {field: float(scenario[field]) for field in numeric_fields}
        if not all(math.isfinite(number) for number in numbers.values()):
            raise PrivateScheduleScenarioError("Scenario contains non-finite values")
        if not -30.0 <= numbers["temperature_delta_c"] <= 30.0:
            raise PrivateScheduleScenarioError("Scenario temperature delta is invalid")
        if not (
            0.0
            <= numbers["planned_min_soc_pct"]
            < numbers["planned_max_soc_pct"]
            <= 100.0
        ):
            raise PrivateScheduleScenarioError("Scenario SOC window is invalid")
        for field in (
            "charge_c_rate_multiplier",
            "discharge_c_rate_multiplier",
            "duty_rate_multiplier",
        ):
            if not 0.0 < numbers[field] <= 5.0:
                raise PrivateScheduleScenarioError(
                    "Scenario rate multiplier is invalid"
                )
    if reference_count != 1:
        raise PrivateScheduleScenarioError("Exactly one reference scenario is required")
    return config


def _forecast_grid(
    prefix: pd.DataFrame, model_config: Mapping[str, object]
) -> np.ndarray:
    last_efc = float(prefix.iloc[-1]["equivalent_full_cycles"])
    end = float(model_config["score_end_equivalent_full_cycles"])
    step = float(model_config["forecast_grid_step_equivalent_full_cycles"])
    if not all(map(math.isfinite, (last_efc, end, step))) or step <= 0.0:
        raise PrivateScheduleScenarioError("Forecast grid configuration is invalid")
    first = math.ceil((last_efc + 1e-12) / step) * step
    grid = np.arange(first, end + step * 0.5, step, dtype=float)
    return grid[grid > last_efc]


def build_private_schedule_scenarios(
    target_prefixes: pd.DataFrame,
    model_config: Mapping[str, object],
    scenario_config: Mapping[str, object],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build one sealed, outcome-free schedule per operating scenario."""
    config = validate_private_schedule_scenario_config(scenario_config)
    if tuple(target_prefixes.columns) != PARTITIONED_PREFIX_COLUMNS:
        raise PrivateScheduleScenarioError("Target prefix columns changed")
    if target_prefixes.empty or target_prefixes.isna().any().any():
        raise PrivateScheduleScenarioError("Target prefixes are empty or incomplete")
    schedules: dict[str, pd.DataFrame] = {}
    for scenario in config["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        rows: list[dict[str, object]] = []
        for (cell_id, landmark), raw_prefix in target_prefixes.groupby(
            ["cell_id", "landmark_visit_count"], sort=True
        ):
            prefix = raw_prefix.sort_values("visit_index", kind="stable")
            last = prefix.iloc[-1]
            last_days = float(last["elapsed_days"])
            last_efc = float(last["equivalent_full_cycles"])
            if last_days <= 0.0 or last_efc <= 0.0:
                raise PrivateScheduleScenarioError(
                    "Scenario construction needs positive prefix exposure"
                )
            future_duty = last_efc / last_days * float(scenario["duty_rate_multiplier"])
            for exposure in _forecast_grid(prefix, model_config):
                rows.append(
                    {
                        "partition": str(last["partition"]),
                        "cell_id": str(cell_id),
                        "condition_id": str(last["condition_id"]),
                        "landmark_visit_count": int(landmark),
                        "scenario_id": scenario_id,
                        "schedule_role": "deployment_candidate",
                        "schedule_source": "operator_scenario",
                        "declared_at_elapsed_days": last_days,
                        "forecast_elapsed_days": float(
                            last_days + (float(exposure) - last_efc) / future_duty
                        ),
                        "forecast_equivalent_full_cycles": float(exposure),
                        "planned_temperature_c": float(last["temperature_c"])
                        + float(scenario["temperature_delta_c"]),
                        "planned_min_soc_pct": float(scenario["planned_min_soc_pct"]),
                        "planned_max_soc_pct": float(scenario["planned_max_soc_pct"]),
                        "planned_charge_c_rate": float(last["charge_c_rate"])
                        * float(scenario["charge_c_rate_multiplier"]),
                        "planned_discharge_c_rate": float(last["discharge_c_rate"])
                        * float(scenario["discharge_c_rate_multiplier"]),
                    }
                )
        schedule = pd.DataFrame(rows, columns=FORECAST_SCHEDULE_COLUMNS)
        schedules[scenario_id] = validate_private_forecast_schedule(
            schedule, target_prefixes, model_config
        )
    manifest: dict[str, object] = {
        "schema_version": "lifetwin.private_schedule_scenario_bundle.v1",
        "protocol_id": config["protocol_id"],
        "scenario_config_sha256": canonical_json_sha256(config),
        "target_prefix_rows_sha256": canonical_frame_sha256(
            target_prefixes, PARTITIONED_PREFIX_COLUMNS
        ),
        "scenario_schedule_rows_sha256": {
            scenario_id: canonical_frame_sha256(schedule, FORECAST_SCHEDULE_COLUMNS)
            for scenario_id, schedule in schedules.items()
        },
        "scenario_count": len(schedules),
        "capacity_outcome_fields_accepted": False,
        "truth_vault_opened": False,
        "primary_evidence_result": False,
        "public_release_permitted": False,
        "claim_boundary": config["claim_boundary"],
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return schedules, manifest


__all__ = [
    "SCENARIO_SCHEMA",
    "PrivateScheduleScenarioError",
    "build_private_schedule_scenarios",
    "validate_private_schedule_scenario_config",
]
