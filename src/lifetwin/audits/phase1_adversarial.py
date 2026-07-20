from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lifetwin.data.naumann import (
    EXPECTED_CALENDAR_ELAPSED_TIME_S,
    EXPECTED_CALENDAR_PROFILE,
    NAUMANN_CALENDAR_DATASET_ID,
    NAUMANN_CALENDAR_DOI,
    NAUMANN_CALENDAR_LICENSE,
    NAUMANN_CALENDAR_LICENSE_URL,
    NAUMANN_CALENDAR_OBSERVATIONS_SHA256,
    NAUMANN_CALENDAR_SOURCE_URL,
    NAUMANN_REPLICATE_SEMANTICS,
    NAUMANN_STATISTICAL_UNIT,
    validate_naumann_calendar_observations,
)
from lifetwin.experiments.calendar_v3_activation_development import (
    EXPECTED_PREFIXES,
    GATED_TARGET_ACTIVATION_METHOD,
    METHOD_NAMES,
    PREDICTION_KEY_COLUMNS,
    PRIMARY_PREFIX,
    calendar_v3_prediction_sha256,
    calendar_v3_sensitivity_sha256,
    run_calendar_v3_activation_development,
    score_calendar_v3_predictions,
)
from lifetwin.models.calendar_v2 import (
    HIERARCHICAL_POWER_METHOD,
    TARGET_SQRT_METHOD,
)
from lifetwin.models.calendar_v3_activation import (
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
    HIERARCHICAL_ACTIVATION_METHOD,
    TARGET_ACTIVATION_METHOD,
    activation_mechanism_gate,
)


AUDIT_ID = "lifetwin_phase1_adversarial_audit_v1"
AUDIT_DATE = "2026-07-20"
FUTURE_RETENTION_SHOCK_PP = -4.0
NUMERIC_TOLERANCE = 1e-10
FAILURE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Phase1AuditBundle:
    summary: dict[str, object]
    data_conditions: pd.DataFrame
    future_label_attacks: pd.DataFrame
    metric_recalculation: pd.DataFrame
    ablations: pd.DataFrame
    gate_boundaries: pd.DataFrame
    failure_conditions: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, *, sort_by: list[str]) -> str:
    ordered = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _max_abs(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.max(np.abs(array))) if len(array) else 0.0


def _data_identity_and_unit_audit(
    observations: pd.DataFrame,
    *,
    data_path: Path | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    validate_naumann_calendar_observations(observations)
    observed_data_sha256 = _sha256(data_path) if data_path is not None else None
    ordered = observations.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)
    if data_path is not None:
        file_observations = pd.read_csv(data_path).sort_values(
            ["condition_id", "checkup_index"], kind="stable"
        ).reset_index(drop=True)
        observations_match_source_file = ordered.equals(
            file_observations[ordered.columns]
        )
    else:
        observations_match_source_file = False
    grouped = ordered.groupby("condition_id", sort=True)
    initial_capacity = grouped["capacity_ah"].transform("first")
    initial_resistance = grouped["resistance_dc_ohm"].transform("first")
    expected_retention = 100.0 * ordered["capacity_ah"] / initial_capacity
    expected_growth = 100.0 * (
        ordered["resistance_dc_ohm"] / initial_resistance - 1.0
    )
    observed_profiles = {
        str(condition_id): (
            float(condition["temperature_c"].iloc[0]),
            float(condition["storage_soc_fraction"].iloc[0]),
        )
        for condition_id, condition in grouped
    }
    published_time_axis_exact = all(
        np.allclose(
            condition.sort_values("checkup_index", kind="stable")[
                "elapsed_time_s"
            ],
            EXPECTED_CALENDAR_ELAPSED_TIME_S,
            rtol=0.0,
            atol=1e-6,
        )
        for _, condition in grouped
    )

    checks = {
        "dataset_identity_exact": set(ordered["dataset_id"].astype(str))
        == {NAUMANN_CALENDAR_DATASET_ID},
        "source_doi_exact": set(ordered["source_doi"].astype(str))
        == {NAUMANN_CALENDAR_DOI},
        "source_url_exact": set(ordered["source_url"].astype(str))
        == {NAUMANN_CALENDAR_SOURCE_URL},
        "source_license_exact": set(ordered["source_license"].astype(str))
        == {NAUMANN_CALENDAR_LICENSE},
        "source_license_url_exact": set(
            ordered["source_license_url"].astype(str)
        )
        == {NAUMANN_CALENDAR_LICENSE_URL},
        "no_missing_values": not ordered.isna().any().any(),
        "no_exact_duplicate_rows": not ordered.duplicated().any(),
        "no_duplicate_condition_checkups": not ordered.duplicated(
            ["condition_id", "checkup_index"]
        ).any(),
        "condition_count_is_17": ordered["condition_id"].nunique() == 17,
        "rows_per_condition_are_35": bool((grouped.size() == 35).all()),
        "logical_ids_are_condition_means": bool(
            (ordered["condition_id"].astype(str) == ordered["cell_id"].astype(str)).all()
        ),
        "replicate_semantics_exact": set(ordered["replicate_semantics"].astype(str))
        == {NAUMANN_REPLICATE_SEMANTICS},
        "statistical_unit_exact": set(ordered["statistical_unit"].astype(str))
        == {NAUMANN_STATISTICAL_UNIT},
        "three_replicates_per_condition": set(
            pd.to_numeric(ordered["physical_replicates_aggregated"])
        )
        == {3},
        "published_condition_profile_exact": (
            observed_profiles == EXPECTED_CALENDAR_PROFILE
        ),
        "published_time_axis_exact": published_time_axis_exact,
        "nominal_capacity_is_3_ah": bool(
            np.allclose(ordered["nominal_capacity_ah"], 3.0, rtol=0.0, atol=1e-12)
        ),
        "resistance_pulse_is_10_seconds": bool(
            np.allclose(
                ordered["resistance_dc_pulse_duration_s"],
                10.0,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "resistance_soc_uses_percent_units": bool(
            np.allclose(
                ordered["resistance_dc_soc_pct"],
                100.0 * ordered["storage_soc_fraction"],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "resistance_values_are_ohm_scale": bool(
            ordered["resistance_dc_ohm"].between(0.001, 1.0).all()
        ),
        "elapsed_hours_consistent": bool(
            np.allclose(
                ordered["elapsed_hours"],
                ordered["elapsed_time_s"] / 3600.0,
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "elapsed_days_consistent": bool(
            np.allclose(
                ordered["elapsed_days"],
                ordered["elapsed_time_s"] / 86400.0,
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "capacity_loss_identity_consistent": bool(
            np.allclose(
                ordered["capacity_loss_pct"],
                100.0 - ordered["capacity_retention_pct"],
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "capacity_retention_consistent_with_ah": bool(
            np.allclose(
                ordered["capacity_retention_pct"],
                expected_retention,
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "resistance_growth_consistent_with_ohm": bool(
            np.allclose(
                ordered["resistance_growth_pct"],
                expected_growth,
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "canonical_snapshot_sha256_verified": (
            observed_data_sha256 == NAUMANN_CALENDAR_OBSERVATIONS_SHA256
        ),
        "in_memory_observations_match_source_file": observations_match_source_file,
    }
    condition_rows: list[dict[str, object]] = []
    for condition_id, condition in grouped:
        condition = condition.sort_values("checkup_index", kind="stable")
        condition_rows.append(
            {
                "condition_id": str(condition_id),
                "test_id": str(condition["test_id"].iloc[0]),
                "source_cell_id": str(condition["source_cell_id"].iloc[0]),
                "temperature_c": float(condition["temperature_c"].iloc[0]),
                "storage_soc_fraction": float(
                    condition["storage_soc_fraction"].iloc[0]
                ),
                "observation_count": len(condition),
                "first_checkup_index": int(condition["checkup_index"].min()),
                "last_checkup_index": int(condition["checkup_index"].max()),
                "maximum_elapsed_days": float(condition["elapsed_days"].max()),
                "minimum_capacity_retention_pct": float(
                    condition["capacity_retention_pct"].min()
                ),
                "maximum_capacity_retention_pct": float(
                    condition["capacity_retention_pct"].max()
                ),
                "missing_value_count": int(condition.isna().sum().sum()),
                "duplicate_checkup_count": int(
                    condition.duplicated(["checkup_index"]).sum()
                ),
                "physical_replicates_aggregated": int(
                    condition["physical_replicates_aggregated"].iloc[0]
                ),
                "statistical_unit": str(condition["statistical_unit"].iloc[0]),
            }
        )
    condition_table = pd.DataFrame(condition_rows).sort_values(
        "condition_id", kind="stable"
    ).reset_index(drop=True)
    passed = all(bool(value) for value in checks.values())
    result: dict[str, object] = {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "observation_count": len(ordered),
        "logical_condition_count": int(ordered["condition_id"].nunique()),
        "published_physical_cell_count": int(
            ordered["condition_id"].nunique()
            * ordered["physical_replicates_aggregated"].iloc[0]
        ),
        "effective_independent_n": int(ordered["condition_id"].nunique()),
        "exact_duplicate_row_count": int(ordered.duplicated().sum()),
        "duplicate_condition_checkup_count": int(
            ordered.duplicated(["condition_id", "checkup_index"]).sum()
        ),
        "missing_value_count": int(ordered.isna().sum().sum()),
        "maximum_elapsed_hour_conversion_error": _max_abs(
            ordered["elapsed_hours"] - ordered["elapsed_time_s"] / 3600.0
        ),
        "maximum_elapsed_day_conversion_error": _max_abs(
            ordered["elapsed_days"] - ordered["elapsed_time_s"] / 86400.0
        ),
        "maximum_capacity_retention_consistency_error_pp": _max_abs(
            ordered["capacity_retention_pct"] - expected_retention
        ),
        "maximum_resistance_growth_consistency_error_pp": _max_abs(
            ordered["resistance_growth_pct"] - expected_growth
        ),
        "data_sha256": observed_data_sha256,
        "expected_data_sha256": NAUMANN_CALENDAR_OBSERVATIONS_SHA256,
        "guardrail": (
            "The 595 rows are repeated checkups of 17 condition-mean trajectories; "
            "the effective independent sample size is 17, not 595 or 51."
        ),
    }
    return result, condition_table


def _mutate_future_outcomes(
    observations: pd.DataFrame,
    *,
    prefix_checkups: int,
) -> tuple[pd.DataFrame, int]:
    mutated = observations.copy(deep=True)
    mask = pd.to_numeric(mutated["checkup_index"]) >= prefix_checkups
    mutated.loc[mask, "capacity_retention_pct"] = (
        pd.to_numeric(mutated.loc[mask, "capacity_retention_pct"])
        + FUTURE_RETENTION_SHOCK_PP
    )
    mutated.loc[mask, "capacity_loss_pct"] = (
        100.0 - mutated.loc[mask, "capacity_retention_pct"]
    )
    initial_capacity = (
        mutated.loc[pd.to_numeric(mutated["checkup_index"]) == 0]
        .set_index("condition_id")["capacity_ah"]
        .astype(float)
    )
    mutated.loc[mask, "capacity_ah"] = (
        mutated.loc[mask, "condition_id"].map(initial_capacity).astype(float)
        * mutated.loc[mask, "capacity_retention_pct"].astype(float)
        / 100.0
    )
    validate_naumann_calendar_observations(mutated)
    return mutated, int(mask.sum())


def _mutate_one_target_prefix(
    observations: pd.DataFrame,
    *,
    condition_id: str,
    prefix_checkups: int,
) -> tuple[pd.DataFrame, int]:
    mutated = observations.copy(deep=True)
    checkups = pd.to_numeric(mutated["checkup_index"])
    mask = (
        (mutated["condition_id"].astype(str) == condition_id)
        & (checkups > 0)
        & (checkups < prefix_checkups)
    )
    mutated.loc[mask, "capacity_retention_pct"] = (
        pd.to_numeric(mutated.loc[mask, "capacity_retention_pct"]) - 0.5
    )
    mutated.loc[mask, "capacity_loss_pct"] = (
        100.0 - mutated.loc[mask, "capacity_retention_pct"]
    )
    initial_capacity = float(
        mutated.loc[
            (mutated["condition_id"].astype(str) == condition_id)
            & (checkups == 0),
            "capacity_ah",
        ].iloc[0]
    )
    mutated.loc[mask, "capacity_ah"] = (
        initial_capacity
        * pd.to_numeric(mutated.loc[mask, "capacity_retention_pct"])
        / 100.0
    )
    validate_naumann_calendar_observations(mutated)
    return mutated, int(mask.sum())


def _target_prefix_split_isolation_attack(
    observations: pd.DataFrame,
    *,
    config: Mapping[str, object],
    baseline_run: tuple,
) -> dict[str, object]:
    target_condition_id = "NAUMANN_CAL_T40_SOC12.5"
    prefix = PRIMARY_PREFIX
    baseline_predictions = baseline_run[1]
    baseline_diagnostics = baseline_run[5]
    baseline_parameters = baseline_run[6]
    splits = baseline_run[7]
    target_assignments = splits.loc[
        (splits["condition_id"].astype(str) == target_condition_id)
        & (splits["role"].astype(str) == "target")
    ]
    candidates: list[tuple[str, str]] = []
    for assignment in target_assignments.itertuples(index=False):
        same_fold = splits.loc[
            (splits["scenario"] == assignment.scenario)
            & (splits["fold_id"] == assignment.fold_id)
            & (splits["role"].astype(str) == "target")
        ]
        if len(same_fold) > 1:
            candidates.append((str(assignment.scenario), str(assignment.fold_id)))
    if not candidates:
        raise ValueError("No multi-target fold exists for split-isolation attack")
    scenario, fold_id = sorted(candidates)[0]
    mutated, mutated_rows = _mutate_one_target_prefix(
        observations,
        condition_id=target_condition_id,
        prefix_checkups=prefix,
    )
    attacked_run = run_calendar_v3_activation_development(mutated, config=config)
    attacked_predictions = attacked_run[1]
    attacked_diagnostics = attacked_run[5]
    attacked_parameters = attacked_run[6]
    attacked_splits = attacked_run[7]

    fold_mask_baseline = (
        (baseline_predictions["scenario"] == scenario)
        & (baseline_predictions["fold_id"].astype(str) == fold_id)
        & (baseline_predictions["prefix_checkups"] == prefix)
    )
    fold_mask_attacked = (
        (attacked_predictions["scenario"] == scenario)
        & (attacked_predictions["fold_id"].astype(str) == fold_id)
        & (attacked_predictions["prefix_checkups"] == prefix)
    )
    baseline_fold = baseline_predictions.loc[fold_mask_baseline].reset_index(
        drop=True
    )
    attacked_fold = attacked_predictions.loc[fold_mask_attacked].reset_index(
        drop=True
    )
    other_target_mask_baseline = (
        baseline_fold["target_condition_id"].astype(str) != target_condition_id
    )
    other_target_mask_attacked = (
        attacked_fold["target_condition_id"].astype(str) != target_condition_id
    )
    baseline_other = baseline_fold.loc[other_target_mask_baseline].reset_index(
        drop=True
    )
    attacked_other = attacked_fold.loc[other_target_mask_attacked].reset_index(
        drop=True
    )
    baseline_changed_target = baseline_fold.loc[
        ~other_target_mask_baseline
    ].reset_index(drop=True)
    attacked_changed_target = attacked_fold.loc[
        ~other_target_mask_attacked
    ].reset_index(drop=True)
    parameter_mask_baseline = (
        (baseline_parameters["scenario"] == scenario)
        & (baseline_parameters["fold_id"].astype(str) == fold_id)
        & (baseline_parameters["prefix_checkups"] == prefix)
    )
    parameter_mask_attacked = (
        (attacked_parameters["scenario"] == scenario)
        & (attacked_parameters["fold_id"].astype(str) == fold_id)
        & (attacked_parameters["prefix_checkups"] == prefix)
    )
    diagnostic_mask_baseline = (
        (baseline_diagnostics["scenario"] == scenario)
        & (baseline_diagnostics["fold_id"].astype(str) == fold_id)
        & (baseline_diagnostics["prefix_checkups"] == prefix)
        & (
            baseline_diagnostics["target_condition_id"].astype(str)
            != target_condition_id
        )
    )
    diagnostic_mask_attacked = (
        (attacked_diagnostics["scenario"] == scenario)
        & (attacked_diagnostics["fold_id"].astype(str) == fold_id)
        & (attacked_diagnostics["prefix_checkups"] == prefix)
        & (
            attacked_diagnostics["target_condition_id"].astype(str)
            != target_condition_id
        )
    )
    checks = {
        "mutated_target_is_not_a_training_condition": not bool(
            (
                (splits["scenario"] == scenario)
                & (splits["fold_id"].astype(str) == fold_id)
                & (splits["condition_id"].astype(str) == target_condition_id)
                & (splits["role"].astype(str) == "training")
            ).any()
        ),
        "fold_parameters_unchanged": baseline_parameters.loc[
            parameter_mask_baseline
        ].reset_index(drop=True).equals(
            attacked_parameters.loc[parameter_mask_attacked].reset_index(drop=True)
        ),
        "other_target_predictions_unchanged": baseline_other.equals(attacked_other),
        "other_target_diagnostics_unchanged": baseline_diagnostics.loc[
            diagnostic_mask_baseline
        ].reset_index(drop=True).equals(
            attacked_diagnostics.loc[diagnostic_mask_attacked].reset_index(drop=True)
        ),
        "changed_target_predictions_respond": not baseline_changed_target.equals(
            attacked_changed_target
        ),
        "splits_unchanged": splits.equals(attacked_splits),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scenario": scenario,
        "fold_id": fold_id,
        "prefix_checkups": prefix,
        "mutated_target_condition_id": target_condition_id,
        "mutated_prefix_row_count": mutated_rows,
        "other_target_prediction_row_count": len(baseline_other),
        "interpretation": (
            "Changing one held-out target's observed prefix changed its own update "
            "but not the fold prior or predictions for other targets in that fold."
        ),
    }


def _future_label_firewall_audit(
    observations: pd.DataFrame,
    *,
    config: Mapping[str, object],
    baseline_run: tuple,
) -> tuple[dict[str, object], pd.DataFrame]:
    baseline_predictions = baseline_run[1]
    baseline_metrics = baseline_run[2]
    baseline_diagnostics = baseline_run[5]
    baseline_parameters = baseline_run[6]
    baseline_splits = baseline_run[7]
    baseline_sensitivity = baseline_run[8]
    metric_keys = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "method",
    ]
    attack_rows: list[dict[str, object]] = []
    for prefix in EXPECTED_PREFIXES:
        mutated, mutated_rows = _mutate_future_outcomes(
            observations,
            prefix_checkups=prefix,
        )
        attacked_run = run_calendar_v3_activation_development(mutated, config=config)
        attacked_predictions = attacked_run[1]
        attacked_metrics = attacked_run[2]
        attacked_diagnostics = attacked_run[5]
        attacked_parameters = attacked_run[6]
        attacked_splits = attacked_run[7]
        attacked_sensitivity = attacked_run[8]

        baseline_prediction_prefix = baseline_predictions.loc[
            baseline_predictions["prefix_checkups"] == prefix
        ].reset_index(drop=True)
        attacked_prediction_prefix = attacked_predictions.loc[
            attacked_predictions["prefix_checkups"] == prefix
        ].reset_index(drop=True)
        baseline_diagnostic_prefix = baseline_diagnostics.loc[
            baseline_diagnostics["prefix_checkups"] == prefix
        ].reset_index(drop=True)
        attacked_diagnostic_prefix = attacked_diagnostics.loc[
            attacked_diagnostics["prefix_checkups"] == prefix
        ].reset_index(drop=True)
        baseline_parameter_prefix = baseline_parameters.loc[
            baseline_parameters["prefix_checkups"] == prefix
        ].reset_index(drop=True)
        attacked_parameter_prefix = attacked_parameters.loc[
            attacked_parameters["prefix_checkups"] == prefix
        ].reset_index(drop=True)

        baseline_metric_prefix = baseline_metrics.loc[
            baseline_metrics["prefix_checkups"] == prefix
        ]
        attacked_metric_prefix = attacked_metrics.loc[
            attacked_metrics["prefix_checkups"] == prefix
        ]
        metric_comparison = baseline_metric_prefix.merge(
            attacked_metric_prefix,
            on=metric_keys,
            suffixes=("_baseline", "_attacked"),
            validate="one_to_one",
        )
        metric_delta = (
            metric_comparison["trajectory_iae_pp_attacked"]
            - metric_comparison["trajectory_iae_pp_baseline"]
        )
        sensitivity_identical: bool | None = None
        sensitivity_hash_baseline: str | None = None
        sensitivity_hash_attacked: str | None = None
        if prefix == PRIMARY_PREFIX:
            sensitivity_identical = baseline_sensitivity.equals(
                attacked_sensitivity
            )
            sensitivity_hash_baseline = calendar_v3_sensitivity_sha256(
                baseline_sensitivity
            )
            sensitivity_hash_attacked = calendar_v3_sensitivity_sha256(
                attacked_sensitivity
            )

        prediction_hash_baseline = _frame_sha256(
            baseline_prediction_prefix,
            sort_by=PREDICTION_KEY_COLUMNS,
        )
        prediction_hash_attacked = _frame_sha256(
            attacked_prediction_prefix,
            sort_by=PREDICTION_KEY_COLUMNS,
        )
        attack_rows.append(
            {
                "prefix_checkups": prefix,
                "minimum_mutated_checkup_index": prefix,
                "mutated_row_count": mutated_rows,
                "retention_shock_pp": FUTURE_RETENTION_SHOCK_PP,
                "prediction_frame_identical": baseline_prediction_prefix.equals(
                    attacked_prediction_prefix
                ),
                "prediction_sha256_baseline": prediction_hash_baseline,
                "prediction_sha256_attacked": prediction_hash_attacked,
                "diagnostics_identical": baseline_diagnostic_prefix.equals(
                    attacked_diagnostic_prefix
                ),
                "parameters_identical": baseline_parameter_prefix.equals(
                    attacked_parameter_prefix
                ),
                "splits_identical": baseline_splits.equals(attacked_splits),
                "sensitivity_applicable": prefix == PRIMARY_PREFIX,
                "sensitivity_predictions_identical": sensitivity_identical,
                "sensitivity_sha256_baseline": sensitivity_hash_baseline,
                "sensitivity_sha256_attacked": sensitivity_hash_attacked,
                "score_changed": bool(
                    np.any(
                        np.abs(metric_delta.to_numpy(dtype=float))
                        > NUMERIC_TOLERANCE
                    )
                ),
                "maximum_absolute_prediction_delta_pp": _max_abs(
                    attacked_prediction_prefix["predicted_capacity_retention_pct"]
                    - baseline_prediction_prefix[
                        "predicted_capacity_retention_pct"
                    ]
                ),
                "maximum_absolute_score_change_pp": _max_abs(metric_delta),
            }
        )
    attack_table = pd.DataFrame(attack_rows).sort_values(
        "prefix_checkups", kind="stable"
    ).reset_index(drop=True)
    split_isolation = _target_prefix_split_isolation_attack(
        observations,
        config=config,
        baseline_run=baseline_run,
    )
    checks = {
        "every_prefix_prediction_frame_identical": bool(
            attack_table["prediction_frame_identical"].all()
        ),
        "every_prefix_prediction_hash_identical": bool(
            (
                attack_table["prediction_sha256_baseline"]
                == attack_table["prediction_sha256_attacked"]
            ).all()
        ),
        "every_prefix_diagnostics_identical": bool(
            attack_table["diagnostics_identical"].all()
        ),
        "every_prefix_parameters_identical": bool(
            attack_table["parameters_identical"].all()
        ),
        "every_attack_preserves_splits": bool(
            attack_table["splits_identical"].all()
        ),
        "primary_sensitivity_predictions_identical": bool(
            attack_table.loc[
                attack_table["sensitivity_applicable"],
                "sensitivity_predictions_identical",
            ].all()
        ),
        "every_attack_changes_corresponding_scores": bool(
            attack_table["score_changed"].all()
        ),
        "prediction_pack_contains_no_outcome_columns": not bool(
            {
                "capacity_ah",
                "capacity_loss_pct",
                "capacity_retention_pct",
                "true_capacity_retention_pct",
                "prediction_error_pp",
            }
            & set(baseline_predictions.columns)
        ),
        "held_out_target_prefix_isolated_from_fold_prior": (
            split_isolation["status"] == "passed"
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "attack": {
            "mutated_columns": [
                "capacity_ah",
                "capacity_retention_pct",
                "capacity_loss_pct",
            ],
            "retention_shock_pp": FUTURE_RETENTION_SHOCK_PP,
            "tested_prefix_checkups": list(EXPECTED_PREFIXES),
            "attack_count": len(attack_table),
        },
        "maximum_absolute_prediction_delta_pp": float(
            attack_table["maximum_absolute_prediction_delta_pp"].max()
        ),
        "maximum_absolute_score_change_pp": float(
            attack_table["maximum_absolute_score_change_pp"].max()
        ),
        "held_out_target_prefix_split_isolation": split_isolation,
        "interpretation": (
            "At every registered landmark, labels at and after that landmark changed "
            "the scores but not that landmark's predictions, fitted parameters, "
            "diagnostics, splits, or label-free prediction hash."
        ),
    }, attack_table


def _independently_recompute_metrics(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    outcomes = observations[
        [
            "condition_id",
            "checkup_index",
            "elapsed_days",
            "temperature_c",
            "storage_soc_fraction",
            "capacity_retention_pct",
        ]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "elapsed_days": "truth_elapsed_days",
            "temperature_c": "truth_temperature_c",
            "storage_soc_fraction": "truth_storage_soc_fraction",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    joined = predictions.merge(
        outcomes,
        on=["target_condition_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
    )
    if joined["true_capacity_retention_pct"].isna().any():
        raise ValueError("Independent metric audit could not join every outcome")
    coordinate_pairs = (
        ("elapsed_days", "truth_elapsed_days"),
        ("temperature_c", "truth_temperature_c"),
        ("storage_soc_fraction", "truth_storage_soc_fraction"),
    )
    for predicted, truth in coordinate_pairs:
        if not np.allclose(
            pd.to_numeric(joined[predicted]).to_numpy(dtype=float),
            pd.to_numeric(joined[truth]).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                f"Prediction coordinate {predicted} disagrees with observations"
            )
    grouping = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "method",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in joined.groupby(grouping, sort=True):
        ordered = group.sort_values("target_checkup_index", kind="stable")
        elapsed = ordered["truth_elapsed_days"].to_numpy(dtype=float)
        error = (
            ordered["predicted_capacity_retention_pct"].to_numpy(dtype=float)
            - ordered["true_capacity_retention_pct"].to_numpy(dtype=float)
        )
        absolute = np.abs(error)
        prefix = int(ordered["prefix_checkups"].iloc[0])
        target_indices = pd.to_numeric(
            ordered["target_checkup_index"]
        ).to_numpy(dtype=int)
        expected_indices = np.arange(
            prefix,
            int(pd.to_numeric(observations["checkup_index"]).max()) + 1,
            dtype=int,
        )
        if not np.array_equal(target_indices, expected_indices):
            raise ValueError("Independent metric audit found incomplete future support")
        derived_final = target_indices == target_indices.max()
        if not np.array_equal(
            ordered["is_final_checkup"].to_numpy(dtype=bool),
            derived_final,
        ):
            raise ValueError("Prediction final-checkup marker disagrees with outcomes")
        if len(elapsed) < 2 or elapsed[-1] <= elapsed[0]:
            raise ValueError("Independent metric audit found incomplete support")
        final_position = int(np.flatnonzero(derived_final)[0])
        rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "future_checkup_count_recomputed": len(ordered),
                "trajectory_iae_pp_recomputed": float(
                    np.trapezoid(absolute, elapsed) / (elapsed[-1] - elapsed[0])
                ),
                "future_point_mae_pp_recomputed": float(absolute.mean()),
                "final_true_retention_pct_recomputed": float(
                    ordered["true_capacity_retention_pct"].iloc[final_position]
                ),
                "final_predicted_retention_pct_recomputed": float(
                    ordered["predicted_capacity_retention_pct"].iloc[final_position]
                ),
                "final_error_pp_recomputed": float(error[final_position]),
            }
        )
    return pd.DataFrame(rows).sort_values(grouping, kind="stable").reset_index(
        drop=True
    )


def _metric_recalculation_audit(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    condition_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    recomputed = _independently_recompute_metrics(predictions, observations)
    keys = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "method",
    ]
    official_columns = [
        *keys,
        "future_checkup_count",
        "trajectory_iae_pp",
        "future_point_mae_pp",
        "final_true_retention_pct",
        "final_predicted_retention_pct",
        "final_error_pp",
    ]
    checked = condition_metrics[official_columns].merge(
        recomputed,
        on=keys,
        validate="one_to_one",
    )
    metric_pairs = {
        "future_checkup_count": "future_checkup_count_recomputed",
        "trajectory_iae_pp": "trajectory_iae_pp_recomputed",
        "future_point_mae_pp": "future_point_mae_pp_recomputed",
        "final_true_retention_pct": "final_true_retention_pct_recomputed",
        "final_predicted_retention_pct": (
            "final_predicted_retention_pct_recomputed"
        ),
        "final_error_pp": "final_error_pp_recomputed",
    }
    maximum_differences: dict[str, float] = {}
    for official, independent in metric_pairs.items():
        difference_column = f"{official}_audit_difference"
        checked[difference_column] = checked[official] - checked[independent]
        maximum_differences[official] = _max_abs(checked[difference_column])

    summary_differences: list[float] = []
    for row in comparisons.itertuples(index=False):
        candidate = checked.loc[
            (checked["scenario"] == row.scenario)
            & (checked["prefix_checkups"] == row.prefix_checkups)
            & (checked["method"] == row.candidate_method)
        ]
        comparator = checked.loc[
            (checked["scenario"] == row.scenario)
            & (checked["prefix_checkups"] == row.prefix_checkups)
            & (checked["method"] == HIERARCHICAL_POWER_METHOD)
        ]
        candidate_mean = float(candidate["trajectory_iae_pp_recomputed"].mean())
        comparator_mean = float(comparator["trajectory_iae_pp_recomputed"].mean())
        summary_differences.extend(
            [
                candidate_mean - row.candidate_trajectory_iae_pp_mean,
                comparator_mean - row.comparator_trajectory_iae_pp_mean,
                (candidate_mean - comparator_mean) - row.mean_paired_delta_iae_pp,
            ]
        )
    max_summary_difference = _max_abs(np.asarray(summary_differences))
    tampered_time = predictions.copy(deep=True)
    tampered_time.loc[tampered_time.index[0], "elapsed_days"] += 1.0
    tampered_time_rejected = False
    try:
        score_calendar_v3_predictions(
            tampered_time,
            observations,
            frozen_prediction_sha256=calendar_v3_prediction_sha256(tampered_time),
        )
    except ValueError:
        tampered_time_rejected = True

    tampered_final = predictions.copy(deep=True)
    first_key = tuple(tampered_final.loc[0, keys])
    first_group_mask = np.logical_and.reduce(
        [tampered_final[column] == value for column, value in zip(keys, first_key)]
    )
    first_group_indices = tampered_final.index[first_group_mask]
    tampered_final.loc[first_group_indices, "is_final_checkup"] = False
    tampered_final.loc[first_group_indices[0], "is_final_checkup"] = True
    tampered_final_rejected = False
    try:
        score_calendar_v3_predictions(
            tampered_final,
            observations,
            frozen_prediction_sha256=calendar_v3_prediction_sha256(tampered_final),
        )
    except ValueError:
        tampered_final_rejected = True
    checks = {
        "all_official_groups_recomputed": len(checked) == len(condition_metrics),
        "all_condition_metrics_match": all(
            value <= NUMERIC_TOLERANCE for value in maximum_differences.values()
        ),
        "all_summary_means_match": max_summary_difference <= NUMERIC_TOLERANCE,
        "statistical_unit_is_condition_trajectory": set(
            observations["statistical_unit"].astype(str)
        )
        == {NAUMANN_STATISTICAL_UNIT},
        "rehashed_tampered_time_axis_rejected": tampered_time_rejected,
        "rehashed_tampered_final_marker_rejected": tampered_final_rejected,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "condition_method_group_count": len(checked),
        "maximum_absolute_metric_differences": maximum_differences,
        "maximum_absolute_summary_difference": max_summary_difference,
        "prediction_coordinate_tamper_attacks": {
            "elapsed_days_rehashed_and_rejected": tampered_time_rejected,
            "final_marker_rehashed_and_rejected": tampered_final_rejected,
        },
        "formula": (
            "trapezoid(abs(prediction-truth), elapsed_days) divided by the "
            "elapsed span from the first to final future checkup"
        ),
    }
    return result, checked.sort_values(keys, kind="stable").reset_index(drop=True)


def _ablation_table(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        (
            "sqrt_to_hierarchical_v2",
            HIERARCHICAL_POWER_METHOD,
            TARGET_SQRT_METHOD,
            "hierarchy_plus_free_exponent",
            False,
            "cross-condition hierarchy versus target-only square-root baseline",
        ),
        (
            "v2_to_ungated_hierarchical_activation",
            HIERARCHICAL_ACTIVATION_METHOD,
            HIERARCHICAL_POWER_METHOD,
            "activation_term",
            True,
            "activation term within the same hierarchical expert family",
        ),
        (
            "ungated_to_gated_target_activation",
            GATED_TARGET_ACTIVATION_METHOD,
            TARGET_ACTIVATION_METHOD,
            "conservative_gate",
            True,
            "safety cost of exact V2 fallback around the target specialist",
        ),
        (
            "ungated_to_gated_hierarchical_activation",
            GATED_HIERARCHICAL_ACTIVATION_METHOD,
            HIERARCHICAL_ACTIVATION_METHOD,
            "conservative_gate",
            True,
            "safety cost of exact V2 fallback around the hierarchical specialist",
        ),
        (
            "gated_hierarchical_to_gated_target_specialist",
            GATED_TARGET_ACTIVATION_METHOD,
            GATED_HIERARCHICAL_ACTIVATION_METHOD,
            "specialist_branch",
            True,
            "target-specific versus hierarchical specialist under the same gate",
        ),
        (
            "v2_to_gated_target_activation",
            GATED_TARGET_ACTIVATION_METHOD,
            HIERARCHICAL_POWER_METHOD,
            "combined_candidate",
            False,
            "primary gated target specialist versus frozen V2",
        ),
    )
    primary = condition_metrics.loc[
        condition_metrics["prefix_checkups"] == PRIMARY_PREFIX
    ].copy()
    keys = ["scenario", "fold_id", "target_condition_id", "prefix_checkups"]
    rows: list[dict[str, object]] = []
    for (
        label,
        candidate_method,
        comparator_method,
        causal_factor,
        single_factor_isolation,
        interpretation,
    ) in comparisons:
        for scenario in sorted(primary["scenario"].unique()):
            candidate = primary.loc[
                (primary["scenario"] == scenario)
                & (primary["method"] == candidate_method),
                [*keys, "trajectory_iae_pp", "activation_gate_ready"],
            ].rename(columns={"trajectory_iae_pp": "candidate_iae_pp"})
            comparator = primary.loc[
                (primary["scenario"] == scenario)
                & (primary["method"] == comparator_method),
                [*keys, "trajectory_iae_pp"],
            ].rename(columns={"trajectory_iae_pp": "comparator_iae_pp"})
            paired = candidate.merge(
                comparator,
                on=keys,
                validate="one_to_one",
            )
            delta = paired["candidate_iae_pp"] - paired["comparator_iae_pp"]
            candidate_mean = float(paired["candidate_iae_pp"].mean())
            comparator_mean = float(paired["comparator_iae_pp"].mean())
            rows.append(
                {
                    "ablation": label,
                    "scenario": scenario,
                    "prefix_checkups": PRIMARY_PREFIX,
                    "candidate_method": candidate_method,
                    "comparator_method": comparator_method,
                    "condition_count": len(paired),
                    "candidate_iae_pp_mean": candidate_mean,
                    "comparator_iae_pp_mean": comparator_mean,
                    "mean_delta_iae_pp": float(delta.mean()),
                    "relative_improvement_fraction": float(
                        (comparator_mean - candidate_mean) / comparator_mean
                    ),
                    "candidate_better_condition_count": int(
                        np.sum(delta < -FAILURE_TOLERANCE)
                    ),
                    "candidate_worse_condition_count": int(
                        np.sum(delta > FAILURE_TOLERANCE)
                    ),
                    "candidate_equal_condition_count": int(
                        np.sum(np.abs(delta) <= FAILURE_TOLERANCE)
                    ),
                    "gate_ready_condition_count": int(
                        paired["activation_gate_ready"].sum()
                    ),
                    "causal_factor": causal_factor,
                    "single_factor_isolation": single_factor_isolation,
                    "same_condition_support": True,
                    "confirmatory_claim_allowed": False,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["ablation", "scenario"], kind="stable"
    ).reset_index(drop=True)


def _baseline_fairness_audit(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    config: Mapping[str, object],
    condition_metrics: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    support_keys = [column for column in PREDICTION_KEY_COLUMNS if column != "method"]
    expected_methods = tuple(sorted(METHOD_NAMES))
    method_support = predictions.groupby(support_keys, sort=True)["method"].agg(
        lambda values: tuple(sorted(values.astype(str)))
    )
    invariant_columns = [
        "training_history_policy",
        "prefix_end_checkup_index",
        "prefix_end_days",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_days",
        "is_final_checkup",
        "activation_gate_ready",
        "negative_loss_evidence",
        "positive_time_observation_count",
        "minimum_prefix_capacity_loss_pct",
        "training_support_days",
        "validation_horizon_days",
        "time_extrapolation_ratio",
        "training_state_sha256",
        "prediction_state_sha256",
    ]
    invariant_cardinality = predictions.groupby(support_keys, sort=True)[
        invariant_columns
    ].nunique(dropna=False)

    prediction_pivot = predictions.pivot(
        index=support_keys,
        columns="method",
        values="predicted_capacity_retention_pct",
    )
    gate_ready = predictions.groupby(support_keys, sort=True)[
        "activation_gate_ready"
    ].first()
    selected_pivot = predictions.pivot(
        index=support_keys,
        columns="method",
        values="activation_component_selected",
    ).astype(bool)
    fallback_rows = ~gate_ready
    specialist_rows = gate_ready

    fallback_target_delta = _max_abs(
        prediction_pivot.loc[fallback_rows, GATED_TARGET_ACTIVATION_METHOD]
        - prediction_pivot.loc[fallback_rows, HIERARCHICAL_POWER_METHOD]
    )
    fallback_hierarchical_delta = _max_abs(
        prediction_pivot.loc[
            fallback_rows, GATED_HIERARCHICAL_ACTIVATION_METHOD
        ]
        - prediction_pivot.loc[fallback_rows, HIERARCHICAL_POWER_METHOD]
    )
    specialist_target_delta = _max_abs(
        prediction_pivot.loc[specialist_rows, GATED_TARGET_ACTIVATION_METHOD]
        - prediction_pivot.loc[specialist_rows, TARGET_ACTIVATION_METHOD]
    )
    specialist_hierarchical_delta = _max_abs(
        prediction_pivot.loc[
            specialist_rows, GATED_HIERARCHICAL_ACTIVATION_METHOD
        ]
        - prediction_pivot.loc[specialist_rows, HIERARCHICAL_ACTIVATION_METHOD]
    )

    split_group = splits.groupby(["scenario", "fold_id"], sort=True)
    split_condition_counts = split_group["condition_id"].nunique()
    split_role_counts = split_group["role"].nunique()
    forbidden_outcome_columns = {
        "capacity_ah",
        "capacity_loss_pct",
        "capacity_retention_pct",
        "true_capacity_retention_pct",
        "prediction_error_pp",
    }
    tau_config = config["timescale_sensitivity"]
    checks = {
        "every_future_coordinate_has_all_methods": bool(
            (method_support == expected_methods).all()
        ),
        "method_invariant_support_is_identical": bool(
            (invariant_cardinality == 1).all().all()
        ),
        "all_predictions_are_after_their_prefix": bool(
            (
                pd.to_numeric(predictions["target_checkup_index"])
                >= pd.to_numeric(predictions["prefix_checkups"])
            ).all()
        ),
        "all_methods_use_global_landmark": set(
            predictions["training_history_policy"].astype(str)
        )
        == {"global_landmark_prefix"},
        "prediction_pack_has_no_future_outcomes": not bool(
            forbidden_outcome_columns & set(predictions.columns)
        ),
        "each_fold_assigns_all_17_conditions_once": bool(
            (split_condition_counts == 17).all()
            and not splits.duplicated(["scenario", "fold_id", "condition_id"]).any()
        ),
        "each_fold_contains_training_and_target_roles": bool(
            (split_role_counts == 2).all()
        ),
        "fallback_target_path_equals_v2": fallback_target_delta
        <= NUMERIC_TOLERANCE,
        "fallback_hierarchical_path_equals_v2": fallback_hierarchical_delta
        <= NUMERIC_TOLERANCE,
        "ready_target_path_equals_specialist": specialist_target_delta
        <= NUMERIC_TOLERANCE,
        "ready_hierarchical_path_equals_specialist": specialist_hierarchical_delta
        <= NUMERIC_TOLERANCE,
        "gated_selection_matches_gate": bool(
            (
                selected_pivot[GATED_TARGET_ACTIVATION_METHOD]
                == gate_ready.to_numpy(dtype=bool)
            ).all()
            and (
                selected_pivot[GATED_HIERARCHICAL_ACTIVATION_METHOD]
                == gate_ready.to_numpy(dtype=bool)
            ).all()
        ),
        "post_hoc_tau_is_explicit": (
            tau_config["selection_status"]
            == "post_hoc_fixed_after_phase7_failure_audit"
            and tau_config["future_outcomes_used_for_timescale_selection"] is True
            and tau_config["formal_hyperparameter_selection_claim_allowed"] is False
        ),
    }
    ablations = _ablation_table(condition_metrics)
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "method_count": len(METHOD_NAMES),
        "future_coordinate_count": len(method_support),
        "ready_future_coordinate_count": int(specialist_rows.sum()),
        "fallback_future_coordinate_count": int(fallback_rows.sum()),
        "maximum_routing_differences_pp": {
            "fallback_target_vs_v2": fallback_target_delta,
            "fallback_hierarchical_vs_v2": fallback_hierarchical_delta,
            "ready_target_vs_specialist": specialist_target_delta,
            "ready_hierarchical_vs_specialist": specialist_hierarchical_delta,
        },
        "fairness_boundary": (
            "V2 and V3 share target conditions, prefixes, future coordinates, and "
            "training folds. The square-root baseline is target-only and therefore a "
            "lower-information conventional reference. Tau=7 remains post-hoc, so "
            "the comparison is retrospective rather than confirmatory."
        ),
    }
    return result, ablations


def _gate_frame(positive_count: int, minimum_loss: float) -> pd.DataFrame:
    elapsed = np.arange(positive_count + 1, dtype=float)
    losses = np.full(positive_count + 1, 0.2, dtype=float)
    losses[0] = 0.0
    if positive_count:
        losses[1] = minimum_loss
    return pd.DataFrame(
        {
            "condition_id": ["SYNTHETIC_GATE"] * len(elapsed),
            "temperature_c": [25.0] * len(elapsed),
            "storage_soc_fraction": [0.125] * len(elapsed),
            "elapsed_days": elapsed,
            "capacity_loss_pct": losses,
        }
    )


def _gate_boundary_audit() -> tuple[dict[str, object], pd.DataFrame]:
    specifications = (
        ("six_points_with_negative_loss", 6, -0.2, 0.0, False, True),
        ("seven_points_with_negative_loss", 7, -0.2, 0.0, True, True),
        ("seven_points_without_negative_loss", 7, 0.0, 0.0, False, False),
        ("negative_loss_exactly_at_margin", 7, -0.1, 0.1, False, False),
        ("negative_loss_beyond_margin", 7, -0.100001, 0.1, True, True),
        ("ten_points_without_negative_loss", 10, 0.05, 0.0, False, False),
    )
    rows: list[dict[str, object]] = []
    for (
        case,
        positive_count,
        minimum_loss,
        threshold,
        expected_ready,
        expected_negative,
    ) in specifications:
        gate = activation_mechanism_gate(
            _gate_frame(positive_count, minimum_loss),
            minimum_positive_time_observations=7,
            negative_loss_threshold_pp=threshold,
        )
        rows.append(
            {
                "case": case,
                "positive_time_observation_count": positive_count,
                "minimum_capacity_loss_pct": minimum_loss,
                "negative_loss_threshold_pp": threshold,
                "expected_negative_loss_evidence": expected_negative,
                "observed_negative_loss_evidence": gate.negative_loss_evidence,
                "expected_gate_ready": expected_ready,
                "observed_gate_ready": gate.ready,
                "case_passed": bool(
                    gate.ready == expected_ready
                    and gate.negative_loss_evidence == expected_negative
                    and gate.positive_time_observation_count == positive_count
                ),
            }
        )
    table = pd.DataFrame(rows)

    invalid_minimum_rejected = False
    negative_threshold_rejected = False
    multi_condition_rejected = False
    zero_positive_time_rejected = False
    try:
        activation_mechanism_gate(_gate_frame(0, 0.0))
    except ValueError:
        zero_positive_time_rejected = True
    try:
        activation_mechanism_gate(
            _gate_frame(7, -0.2), minimum_positive_time_observations=2
        )
    except ValueError:
        invalid_minimum_rejected = True
    try:
        activation_mechanism_gate(
            _gate_frame(7, -0.2), negative_loss_threshold_pp=-0.1
        )
    except ValueError:
        negative_threshold_rejected = True
    multi = _gate_frame(7, -0.2)
    multi.loc[multi.index[-1], "condition_id"] = "SECOND_CONDITION"
    try:
        activation_mechanism_gate(multi)
    except ValueError:
        multi_condition_rejected = True

    checks = {
        "all_boundary_cases_passed": bool(table["case_passed"].all()),
        "invalid_minimum_observation_count_rejected": invalid_minimum_rejected,
        "negative_threshold_rejected": negative_threshold_rejected,
        "multi_condition_input_rejected": multi_condition_rejected,
        "zero_positive_time_input_rejected": zero_positive_time_rejected,
    }
    return (
        {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "case_count": len(table),
            "strict_boundary_rule": (
                "negative evidence requires minimum_loss < -threshold; equality "
                "does not activate the specialist"
            ),
        },
        table,
    )


def _failure_condition_table(
    condition_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    keys = ["scenario", "fold_id", "target_condition_id", "prefix_checkups"]
    metric_columns = [
        *keys,
        "trajectory_iae_pp",
        "future_point_mae_pp",
        "final_true_retention_pct",
        "final_predicted_retention_pct",
        "final_error_pp",
        "activation_gate_ready",
    ]
    candidate = condition_metrics.loc[
        condition_metrics["method"] == GATED_TARGET_ACTIVATION_METHOD,
        metric_columns,
    ].rename(
        columns={
            "trajectory_iae_pp": "candidate_trajectory_iae_pp",
            "future_point_mae_pp": "candidate_future_point_mae_pp",
            "final_true_retention_pct": "final_true_retention_pct",
            "final_predicted_retention_pct": (
                "candidate_final_predicted_retention_pct"
            ),
            "final_error_pp": "candidate_final_error_pp",
        }
    )
    comparator = condition_metrics.loc[
        condition_metrics["method"] == HIERARCHICAL_POWER_METHOD,
        [
            *keys,
            "trajectory_iae_pp",
            "future_point_mae_pp",
            "final_predicted_retention_pct",
            "final_error_pp",
        ],
    ].rename(
        columns={
            "trajectory_iae_pp": "comparator_trajectory_iae_pp",
            "future_point_mae_pp": "comparator_future_point_mae_pp",
            "final_predicted_retention_pct": (
                "comparator_final_predicted_retention_pct"
            ),
            "final_error_pp": "comparator_final_error_pp",
        }
    )
    ungated_target = condition_metrics.loc[
        condition_metrics["method"] == TARGET_ACTIVATION_METHOD,
        [*keys, "trajectory_iae_pp"],
    ].rename(
        columns={"trajectory_iae_pp": "ungated_target_trajectory_iae_pp"}
    )
    gated_hierarchical = condition_metrics.loc[
        condition_metrics["method"] == GATED_HIERARCHICAL_ACTIVATION_METHOD,
        [*keys, "trajectory_iae_pp"],
    ].rename(
        columns={"trajectory_iae_pp": "gated_hierarchical_trajectory_iae_pp"}
    )
    support = (
        predictions.loc[
            predictions["method"] == GATED_TARGET_ACTIVATION_METHOD
        ]
        .groupby(keys, sort=True)
        .agg(
            temperature_c=("temperature_c", "first"),
            storage_soc_fraction=("storage_soc_fraction", "first"),
            activation_component_selected=(
                "activation_component_selected",
                "first",
            ),
            negative_loss_evidence=("negative_loss_evidence", "first"),
            positive_time_observation_count=(
                "positive_time_observation_count",
                "first",
            ),
            minimum_prefix_capacity_loss_pct=(
                "minimum_prefix_capacity_loss_pct",
                "first",
            ),
            prefix_end_days=("prefix_end_days", "first"),
            validation_horizon_days=("validation_horizon_days", "first"),
            training_support_days=("training_support_days", "first"),
            time_extrapolation_ratio=("time_extrapolation_ratio", "first"),
            future_checkup_count=("target_checkup_index", "size"),
        )
        .reset_index()
    )
    runtime = diagnostics[
        [
            *keys,
            "fallback_reason",
            "target_activation_fit_status",
            "target_activation_fit_error",
        ]
    ].copy()
    table = (
        candidate.merge(comparator, on=keys, validate="one_to_one")
        .merge(ungated_target, on=keys, validate="one_to_one")
        .merge(gated_hierarchical, on=keys, validate="one_to_one")
        .merge(support, on=keys, validate="one_to_one")
        .merge(runtime, on=keys, validate="one_to_one")
    )
    table["primary_vs_v2_delta_iae_pp"] = (
        table["candidate_trajectory_iae_pp"]
        - table["comparator_trajectory_iae_pp"]
    )
    table["primary_vs_v2_relative_improvement_fraction"] = (
        table["comparator_trajectory_iae_pp"]
        - table["candidate_trajectory_iae_pp"]
    ) / table["comparator_trajectory_iae_pp"]
    table["ungated_target_vs_v2_delta_iae_pp"] = (
        table["ungated_target_trajectory_iae_pp"]
        - table["comparator_trajectory_iae_pp"]
    )
    table["gated_hierarchical_vs_v2_delta_iae_pp"] = (
        table["gated_hierarchical_trajectory_iae_pp"]
        - table["comparator_trajectory_iae_pp"]
    )
    table["candidate_final_absolute_error_pp"] = table[
        "candidate_final_error_pp"
    ].abs()
    table["comparator_final_absolute_error_pp"] = table[
        "comparator_final_error_pp"
    ].abs()
    table["horizon_to_prefix_time_ratio"] = (
        table["validation_horizon_days"] / table["prefix_end_days"]
    )
    table["is_primary_prefix"] = table["prefix_checkups"] == PRIMARY_PREFIX
    table["statistical_unit"] = NAUMANN_STATISTICAL_UNIT
    table["independent_evidence_key"] = table["target_condition_id"]
    table["scenario_occurrence_count"] = table.groupby(
        ["prefix_checkups", "target_condition_id"], sort=False
    )["scenario"].transform("nunique")
    table["duplicated_across_scenarios"] = table["scenario_occurrence_count"] > 1
    table["candidate_error_rank_desc"] = table.groupby(
        ["scenario", "prefix_checkups"], sort=False
    )["candidate_trajectory_iae_pp"].rank(method="min", ascending=False)
    group_size = table.groupby(
        ["scenario", "prefix_checkups"], sort=False
    )["target_condition_id"].transform("size")
    top_quartile_rank = np.ceil(group_size.astype(float) * 0.25)
    table["candidate_error_top_quartile"] = (
        table["candidate_error_rank_desc"] <= top_quartile_rank
    )
    minimum_temperature = float(table["temperature_c"].min())
    maximum_temperature = float(table["temperature_c"].max())
    table["temperature_outside_training_hull"] = (
        table["scenario"].astype(str).str.contains("temperature")
        & table["temperature_c"].isin(
            [minimum_temperature, maximum_temperature]
        )
    )

    table["selected_branch"] = np.where(
        table["activation_component_selected"],
        "target_activation_specialist",
        "hierarchical_v2_fallback",
    )
    table["gate_evidence_gap"] = np.where(
        table["activation_gate_ready"],
        "none",
        np.where(
            table["positive_time_observation_count"] < 7,
            "insufficient_positive_time_observations",
            "negative_loss_evidence_absent",
        ),
    )

    risk_flags: list[str] = []
    trust_status: list[str] = []
    outcome_classes: list[str] = []
    recommended_actions: list[str] = []
    for row in table.itertuples(index=False):
        flags = ["retrospective_post_hoc", "condition_mean_not_cell_level"]
        delta = float(row.primary_vs_v2_delta_iae_pp)
        if not bool(row.activation_gate_ready):
            flags.append("specialist_not_activated")
        if delta > FAILURE_TOLERANCE:
            flags.append("primary_regression_vs_v2")
            outcome_class = "relative_regression"
        elif abs(delta) <= FAILURE_TOLERANCE:
            flags.append("fallback_same_as_v2")
            outcome_class = "exact_v2_fallback"
        else:
            flags.append("retrospective_improvement_signal")
            outcome_class = "retrospective_improvement"
        if bool(row.activation_gate_ready) and delta >= -FAILURE_TOLERANCE:
            flags.append("gate_triggered_without_trajectory_gain")
        if (
            float(row.candidate_final_absolute_error_pp)
            > float(row.comparator_final_absolute_error_pp) + FAILURE_TOLERANCE
        ):
            flags.append("candidate_worse_at_final_checkup")
        if float(row.horizon_to_prefix_time_ratio) > 5.0:
            flags.append("long_horizon_from_short_prefix")
        if bool(row.candidate_error_top_quartile):
            flags.append("top_quartile_absolute_error_within_scenario_landmark")
        if bool(row.temperature_outside_training_hull):
            flags.append("temperature_outside_training_convex_hull")
        if bool(row.duplicated_across_scenarios):
            flags.append("scenario_duplicate_not_independent_evidence")
        if float(row.ungated_target_vs_v2_delta_iae_pp) > FAILURE_TOLERANCE:
            flags.append("ungated_specialist_would_regress")
        if float(row.gated_hierarchical_vs_v2_delta_iae_pp) > FAILURE_TOLERANCE:
            flags.append("alternative_hierarchical_gate_regresses")

        if "primary_regression_vs_v2" in flags or (
            "gate_triggered_without_trajectory_gain" in flags
        ):
            status = "observed_relative_failure"
        elif not bool(row.activation_gate_ready):
            status = "fallback_only_no_v3_evidence"
        else:
            status = "development_signal_requires_external_validation"
        if status == "observed_relative_failure":
            action = "Do not use the candidate; diagnose the specialist and retain V2."
        elif not bool(row.activation_gate_ready):
            action = (
                "Retain the V2 fallback and collect denser early condition-level data."
            )
        else:
            action = (
                "Freeze the rule and test an independent cell-level cohort before use."
            )
        risk_flags.append(";".join(flags))
        trust_status.append(status)
        outcome_classes.append(outcome_class)
        recommended_actions.append(action)
    table["risk_flags"] = risk_flags
    table["trust_status"] = trust_status
    table["outcome_class"] = outcome_classes
    table["recommended_action"] = recommended_actions
    table["deployment_trusted"] = False
    table["claim_allowed"] = False
    table["claim_boundary"] = (
        "public retrospective condition-mean development only"
    )
    table = table.sort_values(
        [
            "is_primary_prefix",
            "candidate_trajectory_iae_pp",
            "prefix_checkups",
            "scenario",
            "target_condition_id",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    status_counts = {
        str(key): int(value)
        for key, value in table["trust_status"].value_counts().sort_index().items()
    }
    primary_table = table.loc[table["is_primary_prefix"]].copy()
    observed_failures = primary_table.loc[
        primary_table["trust_status"] == "observed_relative_failure",
        ["scenario", "target_condition_id", "risk_flags"],
    ]
    gate_ready_condition_ids = sorted(
        primary_table.loc[
            primary_table["activation_gate_ready"], "target_condition_id"
        ]
        .astype(str)
        .unique()
    )
    primary_gain = (
        primary_table["comparator_trajectory_iae_pp"]
        - primary_table["candidate_trajectory_iae_pp"]
    ).clip(lower=0.0)
    focus_condition = "NAUMANN_CAL_T40_SOC12.5"
    focus_mask = primary_table["target_condition_id"] == focus_condition
    unseen_temperature_mask = primary_table["scenario"].astype(str).str.contains(
        "temperature"
    )
    total_gain = float(primary_gain.sum())
    unseen_temperature_gain = float(primary_gain.loc[unseen_temperature_mask].sum())
    focus_total_gain = float(primary_gain.loc[focus_mask].sum())
    focus_unseen_gain = float(
        primary_gain.loc[focus_mask & unseen_temperature_mask].sum()
    )
    improved_primary = primary_table.loc[
        primary_table["primary_vs_v2_delta_iae_pp"] < -FAILURE_TOLERANCE
    ]
    result = {
        "status": "generated",
        "row_count": len(table),
        "landmark_prefixes": list(EXPECTED_PREFIXES),
        "primary_prefix_row_count": len(primary_table),
        "status_counts": status_counts,
        "primary_observed_relative_regression_count": len(observed_failures),
        "primary_observed_relative_regression_keys": [
            f"{row.scenario}:{row.target_condition_id}"
            for row in observed_failures.itertuples(index=False)
        ],
        "primary_exact_v2_fallback_row_count": int(
            (primary_table["outcome_class"] == "exact_v2_fallback").sum()
        ),
        "primary_improvement_signal_row_count": int(
            (primary_table["outcome_class"] == "retrospective_improvement").sum()
        ),
        "primary_unique_improved_condition_count": int(
            improved_primary["target_condition_id"].nunique()
        ),
        "primary_unique_improved_condition_ids": sorted(
            improved_primary["target_condition_id"].astype(str).unique()
        ),
        "primary_specialist_gate_ready_row_count": int(
            primary_table["activation_gate_ready"].sum()
        ),
        "unique_specialist_gate_ready_condition_count": len(
            gate_ready_condition_ids
        ),
        "unique_specialist_gate_ready_condition_ids": gate_ready_condition_ids,
        "long_horizon_row_count": int(
            (table["horizon_to_prefix_time_ratio"] > 5.0).sum()
        ),
        "top_quartile_absolute_error_row_count": int(
            table["candidate_error_top_quartile"].sum()
        ),
        "improvement_concentration": {
            "focus_condition_id": focus_condition,
            "focus_scenario_occurrence_count": int(focus_mask.sum()),
            "focus_share_of_primary_total_gain_fraction": (
                focus_total_gain / total_gain if total_gain > 0.0 else None
            ),
            "focus_share_of_unseen_temperature_gain_fraction": (
                focus_unseen_gain / unseen_temperature_gain
                if unseen_temperature_gain > 0.0
                else None
            ),
            "independence_warning": (
                "The same condition appears in two scenarios; those occurrences are "
                "not independent evidence."
            ),
        },
        "interpretation": (
            "The primary p=10 gate shows no observed regression versus V2, but that "
            "is structurally aided by exact fallback and is not evidence of general "
            "accuracy. Absolute high-error, extrapolation, fallback, and duplicated-"
            "scenario rows remain explicitly untrusted."
        ),
    }
    return result, table


def run_phase1_adversarial_audit(
    observations: pd.DataFrame,
    *,
    config: Mapping[str, object],
    data_path: Path | None = None,
    baseline_run: tuple | None = None,
) -> Phase1AuditBundle:
    data_audit, data_conditions = _data_identity_and_unit_audit(
        observations, data_path=data_path
    )
    if data_audit["status"] != "passed":
        raise ValueError(
            "Phase 1 canonical data identity audit failed before model execution"
        )
    baseline = (
        baseline_run
        if baseline_run is not None
        else run_calendar_v3_activation_development(observations, config=config)
    )
    result = baseline[0]
    predictions = baseline[1]
    condition_metrics = baseline[2]
    comparisons = baseline[4]
    splits = baseline[7]

    firewall_audit, future_label_attacks = _future_label_firewall_audit(
        observations,
        config=config,
        baseline_run=baseline,
    )
    metric_audit, metric_recalculation = _metric_recalculation_audit(
        predictions,
        observations,
        condition_metrics,
        comparisons,
    )
    fairness_audit, ablations = _baseline_fairness_audit(
        predictions,
        splits,
        config,
        condition_metrics,
    )
    gate_audit, gate_boundaries = _gate_boundary_audit()
    failure_summary, failure_conditions = _failure_condition_table(
        condition_metrics,
        predictions,
        baseline[5],
    )
    technical_sections = (
        data_audit,
        firewall_audit,
        metric_audit,
        fairness_audit,
        gate_audit,
    )
    technical_passed = all(section["status"] == "passed" for section in technical_sections)
    summary: dict[str, object] = {
        "audit_id": AUDIT_ID,
        "audit_date": AUDIT_DATE,
        "audit_execution_status": "passed" if technical_passed else "failed",
        "model_validation_status": "not_confirmed",
        "data_identity_units_duplicates_missing": data_audit,
        "future_label_firewall_attack": firewall_audit,
        "independent_metric_recalculation": metric_audit,
        "baseline_fairness_and_ablation": fairness_audit,
        "gate_boundary_and_routing": gate_audit,
        "failure_condition_inventory": failure_summary,
        "development_gate": result["development_gate"],
        "prohibited_claims": result["prohibited_claims"],
        "completion_scope": (
            "Passing this audit means the public retrospective implementation and "
            "its guardrails survived the specified attacks. It does not validate "
            "15-25 year accuracy, Hithium products, activation physics, or deployment."
        ),
    }
    return Phase1AuditBundle(
        summary=summary,
        data_conditions=data_conditions,
        future_label_attacks=future_label_attacks,
        metric_recalculation=metric_recalculation,
        ablations=ablations,
        gate_boundaries=gate_boundaries,
        failure_conditions=failure_conditions,
    )


def write_phase1_audit_bundle(
    bundle: Phase1AuditBundle,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "phase1_adversarial_audit.json",
        "data_conditions": output_dir / "data_condition_audit.csv",
        "future_label_attacks": output_dir / "future_label_attack_cases.csv",
        "metric_recalculation": output_dir / "independent_metric_audit.csv",
        "ablations": output_dir / "ablation_audit.csv",
        "gate_boundaries": output_dir / "gate_boundary_cases.csv",
        "failure_conditions": output_dir / "failure_condition_table.csv",
    }
    with paths["summary"].open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(bundle.summary, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
    for key, frame in (
        ("data_conditions", bundle.data_conditions),
        ("future_label_attacks", bundle.future_label_attacks),
        ("metric_recalculation", bundle.metric_recalculation),
        ("ablations", bundle.ablations),
        ("gate_boundaries", bundle.gate_boundaries),
        ("failure_conditions", bundle.failure_conditions),
    ):
        frame.to_csv(
            paths[key],
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
    return paths
