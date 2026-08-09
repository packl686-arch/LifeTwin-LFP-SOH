"""Truth-isolated enterprise prediction and scoring for the private V3 model."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.private_dual_clock_prior_v3 import (
    PRIMARY_MODEL_ID,
    PrivateDualClockPriorV3Error,
    default_private_dual_clock_prior_v3_config,
    predict_private_dual_clock_prior_capsule,
    train_private_dual_clock_prior_capsule,
    validate_private_dual_clock_prior_v3_config,
)
from lifetwin.experiments.snl_rpt_loco import _trajectory_iae
from lifetwin.experiments.private_schedule_v4 import (
    BOUNDED_SCHEDULE_MODE_ID,
    ELAPSED_SCHEDULE_MODE_ID,
    FORECAST_SCHEDULE_COLUMNS,
    SCHEDULE_MODE_ID,
    PrivateScheduleV4Error,
    canonicalize_private_forecast_schedule,
    predict_private_dual_clock_bounded_schedule_capsule,
    predict_private_dual_clock_elapsed_schedule_capsule,
    predict_private_dual_clock_schedule_capsule,
    validate_private_forecast_schedule,
)
from lifetwin.validation.private_cycle_adapter import (
    PARTITIONED_PREFIX_COLUMNS,
    PARTITIONED_TRAJECTORY_COLUMNS,
    validate_private_cycle_adapter_config,
    validate_private_cycle_bundle_manifest,
    verify_private_cycle_bundle_frame,
)


EXPERIMENT_ID = "private_enterprise_cycle_v1"
PREDICTION_COLUMNS = (
    "experiment_id",
    "adapter_id",
    "dataset_id",
    "partition",
    "cell_id",
    "condition_id",
    "landmark_visit_count",
    "model_id",
    "forecast_elapsed_days",
    "forecast_equivalent_full_cycles",
    "predicted_capacity_retention_pct",
    "diagnostic_lower_capacity_retention_pct",
    "diagnostic_upper_capacity_retention_pct",
)
DECISION_COLUMNS = (
    "experiment_id",
    "adapter_id",
    "dataset_id",
    "partition",
    "cell_id",
    "condition_id",
    "landmark_visit_count",
    "issued",
    "evidence_status",
    "abstention_reason",
    "selected_landmark_visit_count",
    "prefix_duty_rate_efc_per_day",
    "nearest_condition_distance",
    "condition_ood_threshold",
    "future_schedule_source",
)
SCORE_COLUMNS = (
    "experiment_id",
    "adapter_id",
    "dataset_id",
    "partition",
    "cell_id",
    "condition_id",
    "landmark_visit_count",
    "future_observation_count",
    "trajectory_iae_pp",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
    "pointwise_interval_coverage",
    "simultaneous_trajectory_covered",
    "mean_full_interval_width_pp",
)


class PrivateEnterpriseCycleError(ValueError):
    """Raised when enterprise prediction/scoring isolation is violated."""


def default_private_enterprise_v3_config(
    adapter_config: Mapping[str, object],
) -> dict[str, object]:
    """Bind the generic V3 family to one frozen enterprise adapter contract."""
    adapter = validate_private_cycle_adapter_config(adapter_config)
    config = default_private_dual_clock_prior_v3_config()
    config["dataset_id"] = str(adapter["dataset_id"])
    config["landmark_visit_counts"] = list(
        adapter["trajectory_policy"]["landmark_visit_counts"]
    )
    score_end = float(adapter["trajectory_policy"]["score_end_equivalent_full_cycles"])
    config["score_end_equivalent_full_cycles"] = score_end
    config["uncertainty"]["horizon_bins_efc"] = [
        0.0,
        0.2 * score_end,
        0.4 * score_end,
        0.6 * score_end,
        score_end,
    ]
    return validate_private_dual_clock_prior_v3_config(config)


def _validate_model_binding(
    model_config: Mapping[str, object], adapter_config: Mapping[str, object]
) -> dict[str, object]:
    adapter = validate_private_cycle_adapter_config(adapter_config)
    model = validate_private_dual_clock_prior_v3_config(model_config)
    if model["dataset_id"] != adapter["dataset_id"]:
        raise PrivateEnterpriseCycleError("Enterprise model dataset binding changed")
    if (
        model["landmark_visit_counts"]
        != adapter["trajectory_policy"]["landmark_visit_counts"]
    ):
        raise PrivateEnterpriseCycleError("Enterprise model landmarks changed")
    if not math.isclose(
        float(model["score_end_equivalent_full_cycles"]),
        float(adapter["trajectory_policy"]["score_end_equivalent_full_cycles"]),
        abs_tol=1e-12,
    ):
        raise PrivateEnterpriseCycleError("Enterprise model score window changed")
    if adapter["model_contract"]["primary_model_id"] != PRIMARY_MODEL_ID:
        raise PrivateEnterpriseCycleError("Enterprise primary model binding changed")
    return model


def _forecast_grid(
    prefix: pd.DataFrame, model_config: Mapping[str, object]
) -> np.ndarray:
    x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
    end = float(model_config["score_end_equivalent_full_cycles"])
    step = float(model_config["forecast_grid_step_equivalent_full_cycles"])
    first = math.ceil((x0 + 1e-12) / step) * step
    later = np.arange(first, end + step * 0.5, step, dtype=float)
    return later[later > x0]


def _prefix_core(prefix: pd.DataFrame, landmark: int) -> pd.DataFrame:
    ordered = prefix.loc[:, RPT_TRAJECTORY_COLUMNS].sort_values(
        "visit_index", kind="stable"
    )
    if len(ordered) != landmark:
        raise PrivateEnterpriseCycleError("Enterprise prefix landmark is not exact")
    if not np.array_equal(
        ordered["visit_index"].to_numpy(dtype=int), np.arange(landmark)
    ):
        raise PrivateEnterpriseCycleError("Enterprise prefix visits changed")
    return ordered


def predict_private_enterprise_cycle(
    development_trajectories: pd.DataFrame,
    target_prefixes: pd.DataFrame,
    bundle_manifest: Mapping[str, object],
    adapter_config: Mapping[str, object],
    model_config: Mapping[str, object],
    *,
    forecast_schedule: pd.DataFrame | None = None,
    schedule_mode_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Train on development data and predict prefixes without a truth argument."""
    adapter = validate_private_cycle_adapter_config(adapter_config)
    bundle = validate_private_cycle_bundle_manifest(bundle_manifest, adapter)
    model = _validate_model_binding(model_config, adapter)
    if tuple(development_trajectories.columns) != RPT_TRAJECTORY_COLUMNS:
        raise PrivateEnterpriseCycleError("Development trajectory columns changed")
    if tuple(target_prefixes.columns) != PARTITIONED_PREFIX_COLUMNS:
        raise PrivateEnterpriseCycleError("Enterprise prefix columns changed")
    partitions = sorted(set(target_prefixes["partition"].astype(str)))
    if len(partitions) != 1 or partitions[0] not in {"calibration", "locked_test"}:
        raise PrivateEnterpriseCycleError("Predict exactly one target partition")
    partition = partitions[0]
    verify_private_cycle_bundle_frame(
        "development_trajectories", development_trajectories, bundle
    )
    verify_private_cycle_bundle_frame(f"{partition}_prefixes", target_prefixes, bundle)
    if development_trajectories["condition_id"].nunique() < 2:
        raise PrivateEnterpriseCycleError(
            "V3 nested selection requires at least two development conditions"
        )
    frozen_landmarks = {int(value) for value in model["landmark_visit_counts"]}
    prefix_keys = {
        (str(row.cell_id), int(row.landmark_visit_count))
        for row in target_prefixes.loc[:, ["cell_id", "landmark_visit_count"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    expected_prefix_keys = {
        (str(cell_id), landmark)
        for cell_id in sorted(target_prefixes["cell_id"].astype(str).unique())
        for landmark in frozen_landmarks
    }
    if prefix_keys != expected_prefix_keys:
        raise PrivateEnterpriseCycleError(
            "Enterprise prefixes do not cover every frozen cell/landmark key"
        )
    normalized_schedule = (
        None
        if forecast_schedule is None
        else validate_private_forecast_schedule(
            forecast_schedule, target_prefixes, model
        )
    )
    if normalized_schedule is None:
        if schedule_mode_id is not None:
            raise PrivateEnterpriseCycleError(
                "A schedule mode cannot be selected without a forecast schedule"
            )
        selected_schedule_mode = None
    else:
        selected_schedule_mode = schedule_mode_id or ELAPSED_SCHEDULE_MODE_ID
        if selected_schedule_mode not in {
            BOUNDED_SCHEDULE_MODE_ID,
            ELAPSED_SCHEDULE_MODE_ID,
            SCHEDULE_MODE_ID,
        }:
            raise PrivateEnterpriseCycleError("Enterprise schedule mode is unsupported")
    training_identity = {
        "adapter_id": str(adapter["adapter_id"]),
        "dataset_id": str(adapter["dataset_id"]),
        "bundle_manifest_content_sha256": bundle["manifest_content_sha256"],
        "development_rows_sha256": canonical_frame_sha256(
            development_trajectories, RPT_TRAJECTORY_COLUMNS
        ),
        "development_row_count": len(development_trajectories),
        "raw_measurements_embedded": False,
    }
    capsule = train_private_dual_clock_prior_capsule(
        development_trajectories,
        model,
        training_identity=training_identity,
    )
    prediction_rows = []
    decision_rows = []
    for (cell_id, landmark), raw_prefix in target_prefixes.groupby(
        ["cell_id", "landmark_visit_count"], sort=True
    ):
        landmark_int = int(landmark)
        if landmark_int not in frozen_landmarks:
            raise PrivateEnterpriseCycleError("Enterprise landmark is not frozen")
        prefix = _prefix_core(raw_prefix, landmark_int)
        condition_values = sorted(set(prefix["condition_id"].astype(str)))
        if len(condition_values) != 1:
            raise PrivateEnterpriseCycleError("Enterprise cell condition changed")
        grid = _forecast_grid(prefix, model)
        if len(grid) == 0:
            raise PrivateEnterpriseCycleError("Enterprise forecast grid is empty")
        try:
            if normalized_schedule is None:
                predicted, metadata = predict_private_dual_clock_prior_capsule(
                    prefix, grid, capsule, strict_ood=False
                )
            else:
                cell_schedule = normalized_schedule.loc[
                    (normalized_schedule["cell_id"].astype(str) == str(cell_id))
                    & (normalized_schedule["landmark_visit_count"] == landmark_int)
                ]
                schedule_predictor = {
                    ELAPSED_SCHEDULE_MODE_ID: (
                        predict_private_dual_clock_elapsed_schedule_capsule
                    ),
                    BOUNDED_SCHEDULE_MODE_ID: (
                        predict_private_dual_clock_bounded_schedule_capsule
                    ),
                    SCHEDULE_MODE_ID: predict_private_dual_clock_schedule_capsule,
                }[selected_schedule_mode]
                predicted, metadata = schedule_predictor(
                    prefix, cell_schedule, capsule, strict_ood=False
                )
        except (PrivateDualClockPriorV3Error, PrivateScheduleV4Error) as exc:
            raise PrivateEnterpriseCycleError(
                f"Enterprise capsule prediction failed for {cell_id}: {exc}"
            ) from exc
        issued = str(metadata["evidence_status"]) == "supported"
        reason = "none" if issued else "condition_or_duty_outside_training_support"
        if issued:
            for row in predicted.itertuples(index=False):
                prediction_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "adapter_id": str(adapter["adapter_id"]),
                        "dataset_id": str(adapter["dataset_id"]),
                        "partition": partition,
                        "cell_id": str(cell_id),
                        "condition_id": condition_values[0],
                        "landmark_visit_count": landmark_int,
                        "model_id": PRIMARY_MODEL_ID,
                        "forecast_elapsed_days": float(row.forecast_elapsed_days),
                        "forecast_equivalent_full_cycles": float(
                            row.forecast_equivalent_full_cycles
                        ),
                        "predicted_capacity_retention_pct": float(
                            row.predicted_capacity_retention_pct
                        ),
                        "diagnostic_lower_capacity_retention_pct": float(
                            row.diagnostic_lower_capacity_retention_pct
                        ),
                        "diagnostic_upper_capacity_retention_pct": float(
                            row.diagnostic_upper_capacity_retention_pct
                        ),
                    }
                )
        decision_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "adapter_id": str(adapter["adapter_id"]),
                "dataset_id": str(adapter["dataset_id"]),
                "partition": partition,
                "cell_id": str(cell_id),
                "condition_id": condition_values[0],
                "landmark_visit_count": landmark_int,
                "issued": issued,
                "evidence_status": str(metadata["evidence_status"]),
                "abstention_reason": reason,
                "selected_landmark_visit_count": int(
                    metadata["selected_landmark_visit_count"]
                ),
                "prefix_duty_rate_efc_per_day": float(
                    metadata["prefix_duty_rate_efc_per_day"]
                ),
                "nearest_condition_distance": float(
                    metadata["nearest_condition_distance"]
                ),
                "condition_ood_threshold": float(metadata["condition_ood_threshold"]),
                "future_schedule_source": str(metadata["future_schedule_source"]),
            }
        )
    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).sort_values(
        ["cell_id", "landmark_visit_count", "forecast_equivalent_full_cycles"],
        kind="stable",
        ignore_index=True,
    )
    decisions = pd.DataFrame(decision_rows, columns=DECISION_COLUMNS).sort_values(
        ["cell_id", "landmark_visit_count"], kind="stable", ignore_index=True
    )
    decision_keys = set(
        zip(
            decisions["cell_id"].astype(str),
            decisions["landmark_visit_count"].astype(int),
            strict=True,
        )
    )
    if decision_keys != expected_prefix_keys or len(decisions) != len(decision_keys):
        raise PrivateEnterpriseCycleError("Enterprise decision coverage changed")
    prediction_keys = set(
        zip(
            predictions["cell_id"].astype(str),
            predictions["landmark_visit_count"].astype(int),
            strict=True,
        )
    )
    issued_keys = {
        (str(row.cell_id), int(row.landmark_visit_count))
        for row in decisions.loc[decisions["issued"]].itertuples(index=False)
    }
    if prediction_keys != issued_keys:
        raise PrivateEnterpriseCycleError(
            "Enterprise numeric curves do not match issued decisions"
        )
    prediction_manifest: dict[str, object] = {
        "schema_version": "lifetwin.private_enterprise_cycle.prediction_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "adapter_id": str(adapter["adapter_id"]),
        "dataset_id": str(adapter["dataset_id"]),
        "partition": partition,
        "private_only": True,
        "adapter_config_sha256": canonical_json_sha256(adapter),
        "model_config_sha256": canonical_json_sha256(model),
        "bundle_manifest_content_sha256": bundle["manifest_content_sha256"],
        "development_rows_sha256": canonical_frame_sha256(
            development_trajectories, RPT_TRAJECTORY_COLUMNS
        ),
        "target_prefix_rows_sha256": canonical_frame_sha256(
            target_prefixes, PARTITIONED_PREFIX_COLUMNS
        ),
        "capsule_content_sha256": capsule["capsule_content_sha256"],
        "prediction_rows_sha256": canonical_frame_sha256(
            predictions, PREDICTION_COLUMNS
        ),
        "decision_rows_sha256": canonical_frame_sha256(decisions, DECISION_COLUMNS),
        "prediction_row_count": len(predictions),
        "decision_row_count": len(decisions),
        "target_truth_argument_accepted": False,
        "target_suffix_rows_present": False,
        "truth_vault_opened": False,
        "future_schedule_assumption": (
            "constant_prefix_efc_per_day"
            if normalized_schedule is None
            else "declared_piecewise_operating_schedule"
        ),
        "public_release_permitted": False,
    }
    if normalized_schedule is not None:
        schedule_role = str(normalized_schedule.iloc[0]["schedule_role"])
        schedule_covariates = (
            [
                "forecast_elapsed_days",
                "forecast_equivalent_full_cycles",
            ]
            if selected_schedule_mode == ELAPSED_SCHEDULE_MODE_ID
            else [
                "forecast_elapsed_days",
                "forecast_equivalent_full_cycles",
                "planned_temperature_c",
                "planned_min_soc_pct",
                "planned_max_soc_pct",
                "planned_discharge_c_rate",
            ]
        )
        prediction_manifest.update(
            {
                "prediction_mode_id": selected_schedule_mode,
                "forecast_schedule_rows_sha256": canonical_frame_sha256(
                    normalized_schedule, FORECAST_SCHEDULE_COLUMNS
                ),
                "forecast_schedule_row_count": len(normalized_schedule),
                "schedule_role": schedule_role,
                "schedule_source": str(normalized_schedule.iloc[0]["schedule_source"]),
                "schedule_declared_without_capacity_outcomes": True,
                "schedule_covariates_used_by_model": schedule_covariates,
                "schedule_covariates_used_for_support_diagnostics": [
                    "planned_temperature_c",
                    "planned_min_soc_pct",
                    "planned_max_soc_pct",
                    "planned_discharge_c_rate",
                    "segment_efc_per_day",
                ],
                "planned_charge_c_rate_used_by_model": False,
                "primary_evidence_eligible": schedule_role == "deployment_candidate",
            }
        )
    prediction_manifest["manifest_content_sha256"] = canonical_json_sha256(
        prediction_manifest
    )
    return predictions, decisions, capsule, prediction_manifest


def _validate_prediction_replay(
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    bundle_manifest: Mapping[str, object],
    adapter_config: Mapping[str, object],
    model_config: Mapping[str, object],
    forecast_schedule: pd.DataFrame | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    adapter = validate_private_cycle_adapter_config(adapter_config)
    bundle = validate_private_cycle_bundle_manifest(bundle_manifest, adapter)
    model = _validate_model_binding(model_config, adapter)
    manifest = deepcopy(dict(prediction_manifest))
    expected_hash = manifest.pop("manifest_content_sha256", None)
    if expected_hash != canonical_json_sha256(manifest):
        raise PrivateEnterpriseCycleError("Enterprise prediction manifest changed")
    manifest["manifest_content_sha256"] = expected_hash
    if manifest.get("schema_version") != (
        "lifetwin.private_enterprise_cycle.prediction_manifest.v1"
    ):
        raise PrivateEnterpriseCycleError("Enterprise prediction schema changed")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise PrivateEnterpriseCycleError("Enterprise prediction columns changed")
    if tuple(decisions.columns) != DECISION_COLUMNS:
        raise PrivateEnterpriseCycleError("Enterprise decision columns changed")
    if manifest.get("adapter_config_sha256") != canonical_json_sha256(adapter):
        raise PrivateEnterpriseCycleError("Enterprise adapter binding changed")
    if manifest.get("model_config_sha256") != canonical_json_sha256(model):
        raise PrivateEnterpriseCycleError("Enterprise model binding changed")
    if (
        manifest.get("bundle_manifest_content_sha256")
        != bundle["manifest_content_sha256"]
    ):
        raise PrivateEnterpriseCycleError("Enterprise bundle binding changed")
    if manifest.get("prediction_rows_sha256") != canonical_frame_sha256(
        predictions, PREDICTION_COLUMNS
    ):
        raise PrivateEnterpriseCycleError("Enterprise predictions changed after freeze")
    if manifest.get("decision_rows_sha256") != canonical_frame_sha256(
        decisions, DECISION_COLUMNS
    ):
        raise PrivateEnterpriseCycleError("Enterprise decisions changed after freeze")
    if int(manifest.get("prediction_row_count", -1)) != len(predictions):
        raise PrivateEnterpriseCycleError("Enterprise prediction row count changed")
    if int(manifest.get("decision_row_count", -1)) != len(decisions):
        raise PrivateEnterpriseCycleError("Enterprise decision row count changed")
    partition = str(manifest.get("partition", ""))
    for frame, name in ((predictions, "predictions"), (decisions, "decisions")):
        if len(frame) and set(frame["partition"].astype(str)) != {partition}:
            raise PrivateEnterpriseCycleError(
                f"Enterprise {name} partition identity changed"
            )
    if (
        manifest.get("target_truth_argument_accepted") is not False
        or manifest.get("truth_vault_opened") is not False
    ):
        raise PrivateEnterpriseCycleError("Enterprise prediction firewall changed")
    schedule_assumption = manifest.get("future_schedule_assumption")
    if schedule_assumption == "constant_prefix_efc_per_day":
        if forecast_schedule is not None:
            raise PrivateEnterpriseCycleError(
                "Constant-duty prediction unexpectedly received a schedule"
            )
    elif schedule_assumption == "declared_piecewise_operating_schedule":
        if forecast_schedule is None:
            raise PrivateEnterpriseCycleError(
                "Declared-schedule prediction is missing its sealed schedule"
            )
        normalized = canonicalize_private_forecast_schedule(forecast_schedule)
        if int(manifest.get("forecast_schedule_row_count", -1)) != len(normalized):
            raise PrivateEnterpriseCycleError("Enterprise schedule row count changed")
        if manifest.get("forecast_schedule_rows_sha256") != canonical_frame_sha256(
            normalized, FORECAST_SCHEDULE_COLUMNS
        ):
            raise PrivateEnterpriseCycleError(
                "Enterprise schedule changed after freeze"
            )
        mode_id = manifest.get("prediction_mode_id")
        if mode_id not in {
            BOUNDED_SCHEDULE_MODE_ID,
            ELAPSED_SCHEDULE_MODE_ID,
            SCHEDULE_MODE_ID,
        }:
            raise PrivateEnterpriseCycleError("Enterprise schedule mode changed")
        expected_covariates = (
            ["forecast_elapsed_days", "forecast_equivalent_full_cycles"]
            if mode_id == ELAPSED_SCHEDULE_MODE_ID
            else [
                "forecast_elapsed_days",
                "forecast_equivalent_full_cycles",
                "planned_temperature_c",
                "planned_min_soc_pct",
                "planned_max_soc_pct",
                "planned_discharge_c_rate",
            ]
        )
        if manifest.get("schedule_covariates_used_by_model") != expected_covariates:
            raise PrivateEnterpriseCycleError(
                "Enterprise schedule model covariates changed"
            )
        if manifest.get("schedule_role") != str(normalized.iloc[0]["schedule_role"]):
            raise PrivateEnterpriseCycleError("Enterprise schedule role changed")
    else:
        raise PrivateEnterpriseCycleError("Enterprise future schedule contract changed")
    return manifest, bundle, model


def score_private_enterprise_cycle(
    truth_vault: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    bundle_manifest: Mapping[str, object],
    adapter_config: Mapping[str, object],
    model_config: Mapping[str, object],
    *,
    forecast_schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Open a sealed truth vault only after validating frozen predictions."""
    manifest, bundle, model = _validate_prediction_replay(
        predictions,
        decisions,
        prediction_manifest,
        bundle_manifest,
        adapter_config,
        model_config,
        forecast_schedule,
    )
    adapter = validate_private_cycle_adapter_config(adapter_config)
    if tuple(truth_vault.columns) != PARTITIONED_TRAJECTORY_COLUMNS:
        raise PrivateEnterpriseCycleError("Enterprise truth-vault columns changed")
    partition = str(manifest["partition"])
    if set(truth_vault["partition"].astype(str)) != {partition}:
        raise PrivateEnterpriseCycleError("Enterprise truth partition changed")
    verify_private_cycle_bundle_frame(f"{partition}_truth_vault", truth_vault, bundle)
    truth_identities = truth_vault.loc[:, ["cell_id", "condition_id"]].drop_duplicates()
    if truth_identities["cell_id"].duplicated().any():
        raise PrivateEnterpriseCycleError("Enterprise truth cell condition changed")
    expected_decision_keys = {
        (str(cell_id), int(landmark))
        for cell_id in sorted(truth_identities["cell_id"].astype(str))
        for landmark in model["landmark_visit_counts"]
    }
    decision_keys = [
        (str(row.cell_id), int(row.landmark_visit_count))
        for row in decisions.itertuples(index=False)
    ]
    if (
        len(decision_keys) != len(set(decision_keys))
        or set(decision_keys) != expected_decision_keys
    ):
        raise PrivateEnterpriseCycleError(
            "Enterprise decisions do not cover the sealed truth population"
        )
    issued_keys = {
        (str(row.cell_id), int(row.landmark_visit_count))
        for row in decisions.loc[decisions["issued"]].itertuples(index=False)
    }
    prediction_keys = set(
        zip(
            predictions["cell_id"].astype(str),
            predictions["landmark_visit_count"].astype(int),
            strict=True,
        )
    )
    if prediction_keys != issued_keys:
        raise PrivateEnterpriseCycleError(
            "Enterprise issued decisions and numeric curves disagree"
        )
    score_end = float(model["score_end_equivalent_full_cycles"])
    rows = []
    for decision in decisions.itertuples(index=False):
        if not bool(decision.issued):
            continue
        cell = truth_vault.loc[
            truth_vault["cell_id"] == decision.cell_id,
            RPT_TRAJECTORY_COLUMNS,
        ].sort_values("visit_index", kind="stable")
        landmark = int(decision.landmark_visit_count)
        if len(cell) <= landmark:
            raise PrivateEnterpriseCycleError("Enterprise truth lacks a future suffix")
        prefix = cell.iloc[:landmark]
        future = cell.loc[
            (cell["visit_index"] >= landmark)
            & (cell["equivalent_full_cycles"] <= score_end)
        ]
        minimum_future = int(adapter["trajectory_policy"]["minimum_future_visits"])
        if len(future) < minimum_future:
            raise PrivateEnterpriseCycleError("Enterprise scored future is incomplete")
        curve = predictions.loc[
            (predictions["cell_id"] == decision.cell_id)
            & (predictions["landmark_visit_count"] == landmark)
        ].sort_values("forecast_equivalent_full_cycles", kind="stable")
        if curve.empty:
            raise PrivateEnterpriseCycleError("Issued enterprise curve is missing")
        forecast = future["equivalent_full_cycles"].to_numpy(dtype=float)
        actual = future["capacity_retention_pct"].to_numpy(dtype=float)
        grid = curve["forecast_equivalent_full_cycles"].to_numpy(dtype=float)
        predicted = np.interp(
            forecast,
            grid,
            curve["predicted_capacity_retention_pct"].to_numpy(dtype=float),
        )
        lower = np.interp(
            forecast,
            grid,
            curve["diagnostic_lower_capacity_retention_pct"].to_numpy(dtype=float),
        )
        upper = np.interp(
            forecast,
            grid,
            curve["diagnostic_upper_capacity_retention_pct"].to_numpy(dtype=float),
        )
        error = predicted - actual
        covered = (actual >= lower) & (actual <= upper)
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "adapter_id": str(adapter["adapter_id"]),
                "dataset_id": str(adapter["dataset_id"]),
                "partition": partition,
                "cell_id": str(decision.cell_id),
                "condition_id": str(decision.condition_id),
                "landmark_visit_count": landmark,
                "future_observation_count": len(future),
                "trajectory_iae_pp": _trajectory_iae(
                    float(prefix.iloc[-1]["equivalent_full_cycles"]),
                    forecast,
                    actual,
                    predicted,
                ),
                "trajectory_mae_pp": float(np.mean(np.abs(error))),
                "trajectory_rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                "endpoint_absolute_error_pp": float(abs(error[-1])),
                "pointwise_interval_coverage": float(np.mean(covered)),
                "simultaneous_trajectory_covered": bool(np.all(covered)),
                "mean_full_interval_width_pp": float(np.mean(upper - lower)),
            }
        )
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values(
        ["cell_id", "landmark_visit_count"], kind="stable", ignore_index=True
    )
    summaries = {}
    for landmark in (int(value) for value in model["landmark_visit_counts"]):
        selected_decisions = decisions.loc[
            decisions["landmark_visit_count"] == landmark
        ]
        selected_scores = scores.loc[scores["landmark_visit_count"] == landmark]
        condition_means = (
            selected_scores.groupby("condition_id", sort=True)[
                [
                    "trajectory_iae_pp",
                    "trajectory_mae_pp",
                    "trajectory_rmse_pp",
                    "endpoint_absolute_error_pp",
                    "pointwise_interval_coverage",
                    "simultaneous_trajectory_covered",
                    "mean_full_interval_width_pp",
                ]
            ].mean()
            if len(selected_scores)
            else pd.DataFrame()
        )
        summaries[str(landmark)] = {
            "decision_count": len(selected_decisions),
            "issued_count": int(selected_decisions["issued"].sum()),
            "issued_fraction": float(selected_decisions["issued"].mean()),
            "issued_condition_count": (
                int(selected_scores["condition_id"].nunique())
                if len(selected_scores)
                else 0
            ),
            "condition_equal_trajectory_iae_pp": (
                float(condition_means["trajectory_iae_pp"].mean())
                if len(condition_means)
                else None
            ),
            "condition_equal_trajectory_mae_pp": (
                float(condition_means["trajectory_mae_pp"].mean())
                if len(condition_means)
                else None
            ),
            "condition_equal_trajectory_rmse_pp": (
                float(condition_means["trajectory_rmse_pp"].mean())
                if len(condition_means)
                else None
            ),
            "condition_equal_endpoint_absolute_error_pp": (
                float(condition_means["endpoint_absolute_error_pp"].mean())
                if len(condition_means)
                else None
            ),
            "condition_equal_pointwise_interval_coverage": (
                float(condition_means["pointwise_interval_coverage"].mean())
                if len(condition_means)
                else None
            ),
            "condition_equal_simultaneous_trajectory_coverage": (
                float(condition_means["simultaneous_trajectory_covered"].mean())
                if len(condition_means)
                else None
            ),
            "condition_equal_mean_full_interval_width_pp": (
                float(condition_means["mean_full_interval_width_pp"].mean())
                if len(condition_means)
                else None
            ),
        }
    summary: dict[str, object] = {
        "schema_version": "lifetwin.private_enterprise_cycle.score_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "adapter_id": str(adapter["adapter_id"]),
        "dataset_id": str(adapter["dataset_id"]),
        "partition": partition,
        "private_only": True,
        "evidence_role": (
            "private_batch_disjoint_locked_evaluation"
            if partition == "locked_test"
            else "private_batch_disjoint_calibration"
        ),
        "prediction_manifest_content_sha256": manifest["manifest_content_sha256"],
        "summary_by_landmark": summaries,
        "score_rows_sha256": canonical_frame_sha256(scores, SCORE_COLUMNS),
        "claim_boundary": (
            "Enterprise-private score only. Claim strength depends on enforced "
            "truth-vault access control, batch independence, sample size, and an "
            "unchanged model after calibration."
        ),
        "public_release_permitted": False,
    }
    if manifest["future_schedule_assumption"] == (
        "declared_piecewise_operating_schedule"
    ):
        summary.update(
            {
                "prediction_mode_id": manifest["prediction_mode_id"],
                "schedule_role": manifest["schedule_role"],
                "schedule_source": manifest["schedule_source"],
                "primary_evidence_eligible": manifest["primary_evidence_eligible"],
            }
        )
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    return scores, summary


__all__ = [
    "DECISION_COLUMNS",
    "EXPERIMENT_ID",
    "PREDICTION_COLUMNS",
    "SCORE_COLUMNS",
    "PrivateEnterpriseCycleError",
    "default_private_enterprise_v3_config",
    "predict_private_enterprise_cycle",
    "score_private_enterprise_cycle",
]
