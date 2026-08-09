from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import numpy as np
import pytest

from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.private_enterprise_cycle import (
    DECISION_COLUMNS,
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    PrivateEnterpriseCycleError,
    default_private_enterprise_v3_config,
    predict_private_enterprise_cycle,
    score_private_enterprise_cycle,
)
from lifetwin.experiments.private_schedule_v4 import (
    BOUNDED_SCHEDULE_MODE_ID,
    ELAPSED_SCHEDULE_MODE_ID,
    FORECAST_SCHEDULE_COLUMNS,
    SCHEDULE_MODE_ID,
    PrivateScheduleV4Error,
    validate_private_forecast_schedule,
)
from lifetwin.validation.private_cycle_adapter import (
    PARTITION_METADATA_COLUMNS,
    PRIVATE_MEASUREMENT_COLUMNS,
    PrivateCycleAdapterError,
    build_private_cycle_blind_bundle,
    freeze_private_cycle_partitions,
    normalize_private_cycle_measurements,
    validate_private_cycle_adapter_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/validation/hithium_private_cycle_adapter_v1.json"


def _adapter_config() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["trajectory_policy"]["score_end_equivalent_full_cycles"] = 1250.0
    return validate_private_cycle_adapter_config(config)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": f"enterprise_cell_{batch}",
                "batch_id": f"enterprise_batch_{batch}",
                "condition_id": f"enterprise_condition_{batch}",
            }
            for batch in range(8)
        ],
        columns=PARTITION_METADATA_COLUMNS,
    )


def _measurements() -> pd.DataFrame:
    rows = []
    for batch, identity in enumerate(_metadata().itertuples(index=False)):
        temperature = 25.0 if batch % 2 == 0 else 35.0
        rate = 1.0 if batch % 2 == 0 else 2.0
        fade_scale = 1.0 + 0.2 * (batch % 2)
        for visit in range(6):
            fade = fade_scale * visit * 0.45
            rows.append(
                {
                    "record_id": f"{identity.cell_id}_rpt_{visit}",
                    "cell_id": identity.cell_id,
                    "batch_id": identity.batch_id,
                    "condition_id": identity.condition_id,
                    "cathode_chemistry": "LFP",
                    "temperature_c": temperature,
                    "min_soc_pct": 10.0,
                    "max_soc_pct": 90.0,
                    "charge_c_rate": 0.5,
                    "discharge_c_rate": rate,
                    "visit_index": visit,
                    "elapsed_days": float(visit * 45),
                    "equivalent_full_cycles": float(visit * 250),
                    "capacity_ah": 280.0 * (100.0 - fade) / 100.0,
                    "reference_capacity_ah": 280.0,
                    "quality_status": "accepted",
                }
            )
    return pd.DataFrame(rows, columns=PRIVATE_MEASUREMENT_COLUMNS)


def _bundle() -> tuple[dict[str, object], dict[str, pd.DataFrame], dict[str, object]]:
    adapter = _adapter_config()
    partition_manifest = freeze_private_cycle_partitions(_metadata(), adapter)
    normalized = normalize_private_cycle_measurements(
        _measurements(), partition_manifest, adapter
    )
    frames, bundle_manifest = build_private_cycle_blind_bundle(
        normalized, partition_manifest, adapter
    )
    return adapter, frames, bundle_manifest


def _model_config(adapter: dict[str, object]) -> dict[str, object]:
    model = default_private_enterprise_v3_config(adapter)
    model["forecast_grid_step_equivalent_full_cycles"] = 250.0
    model["dual_clock_family"].update(
        {
            "time_exponents": [0.3],
            "cycle_exponents": [1.0],
            "kernel_gammas": [0.3],
            "coefficient_shrinkages": [1.0],
            "anchor_weights": [0.5],
        }
    )
    return model


def _forecast_schedule(
    prefixes: pd.DataFrame,
    model: dict[str, object],
    *,
    temperature_delta_c: float = 0.0,
    elapsed_scale: float = 1.0,
    schedule_role: str = "deployment_candidate",
) -> pd.DataFrame:
    rows = []
    for (cell_id, landmark), prefix in prefixes.groupby(
        ["cell_id", "landmark_visit_count"], sort=True
    ):
        ordered = prefix.sort_values("visit_index", kind="stable")
        last = ordered.iloc[-1]
        x0 = float(last["equivalent_full_cycles"])
        end = float(model["score_end_equivalent_full_cycles"])
        step = float(model["forecast_grid_step_equivalent_full_cycles"])
        first = math.ceil((x0 + 1e-12) / step) * step
        grid = np.arange(first, end + step * 0.5, step, dtype=float)
        grid = grid[grid > x0]
        duty = x0 / float(last["elapsed_days"])
        for exposure in grid:
            rows.append(
                {
                    "partition": str(last["partition"]),
                    "cell_id": str(cell_id),
                    "condition_id": str(last["condition_id"]),
                    "landmark_visit_count": int(landmark),
                    "scenario_id": "declared_plan_v1",
                    "schedule_role": schedule_role,
                    "schedule_source": (
                        "realized_future_schedule"
                        if schedule_role == "oracle_upper_bound"
                        else "declared_operating_plan"
                    ),
                    "declared_at_elapsed_days": float(last["elapsed_days"]),
                    "forecast_elapsed_days": float(
                        float(last["elapsed_days"])
                        + (exposure / duty - float(last["elapsed_days"]))
                        * elapsed_scale
                    ),
                    "forecast_equivalent_full_cycles": float(exposure),
                    "planned_temperature_c": float(last["temperature_c"])
                    + temperature_delta_c,
                    "planned_min_soc_pct": 10.0,
                    "planned_max_soc_pct": 90.0,
                    "planned_charge_c_rate": 0.5,
                    "planned_discharge_c_rate": float(last["discharge_c_rate"]),
                }
            )
    return pd.DataFrame(rows, columns=FORECAST_SCHEDULE_COLUMNS)


def test_declared_constant_schedule_reproduces_v3_and_is_sealed() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    prefixes = frames["calibration_prefixes"]
    schedule = _forecast_schedule(prefixes, model)
    baseline = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
    )
    scheduled = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=schedule,
    )
    pd.testing.assert_frame_equal(
        baseline[0], scheduled[0], check_exact=False, atol=1e-12, rtol=0.0
    )
    assert set(scheduled[1]["future_schedule_source"]) == {"declared_operating_plan"}
    assert scheduled[3]["prediction_mode_id"] == ELAPSED_SCHEDULE_MODE_ID
    assert scheduled[3]["schedule_covariates_used_by_model"] == [
        "forecast_elapsed_days",
        "forecast_equivalent_full_cycles",
    ]
    assert scheduled[3]["primary_evidence_eligible"] is True
    scores, summary = score_private_enterprise_cycle(
        frames["calibration_truth_vault"],
        scheduled[0],
        scheduled[1],
        scheduled[3],
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=schedule,
    )
    assert len(scores) == len(scheduled[1])
    assert summary["prediction_mode_id"] == ELAPSED_SCHEDULE_MODE_ID
    assert summary["primary_evidence_eligible"] is True


def test_elapsed_schedule_ignores_condition_fields_but_uses_future_time() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    prefixes = frames["calibration_prefixes"]
    constant = _forecast_schedule(prefixes, model)
    condition_changed = _forecast_schedule(prefixes, model, temperature_delta_c=0.1)
    time_changed = _forecast_schedule(prefixes, model, elapsed_scale=1.1)
    first = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=constant,
    )
    second = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=condition_changed,
    )
    third = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=time_changed,
    )
    assert first[3]["target_truth_argument_accepted"] is False
    assert second[3]["target_truth_argument_accepted"] is False
    pd.testing.assert_frame_equal(first[0], second[0])
    assert (
        first[3]["forecast_schedule_rows_sha256"]
        != second[3]["forecast_schedule_rows_sha256"]
    )
    assert not np.allclose(
        first[0]["predicted_capacity_retention_pct"],
        third[0]["predicted_capacity_retention_pct"],
        atol=1e-10,
        rtol=0.0,
    )


def test_full_v4_negative_control_still_changes_with_planned_temperature() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    prefixes = frames["calibration_prefixes"]
    constant = _forecast_schedule(prefixes, model)
    changed = _forecast_schedule(prefixes, model, temperature_delta_c=8.0)
    first = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=constant,
        schedule_mode_id=SCHEDULE_MODE_ID,
    )
    second = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=changed,
        schedule_mode_id=SCHEDULE_MODE_ID,
    )
    assert first[3]["prediction_mode_id"] == SCHEDULE_MODE_ID
    assert not np.allclose(
        first[0]["predicted_capacity_retention_pct"],
        second[0]["predicted_capacity_retention_pct"],
    )


def test_v4_2_condition_delta_is_support_gated_and_smaller_than_full_v4() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    prefixes = frames["calibration_prefixes"]
    schedule = _forecast_schedule(prefixes, model, temperature_delta_c=0.1)
    common = (
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
    )
    elapsed = predict_private_enterprise_cycle(
        *common,
        forecast_schedule=schedule,
    )
    bounded = predict_private_enterprise_cycle(
        *common,
        forecast_schedule=schedule,
        schedule_mode_id=BOUNDED_SCHEDULE_MODE_ID,
    )
    full = predict_private_enterprise_cycle(
        *common,
        forecast_schedule=schedule,
        schedule_mode_id=SCHEDULE_MODE_ID,
    )
    elapsed_values = elapsed[0]["predicted_capacity_retention_pct"].to_numpy()
    bounded_delta = (
        bounded[0]["predicted_capacity_retention_pct"].to_numpy() - elapsed_values
    )
    full_delta = full[0]["predicted_capacity_retention_pct"].to_numpy() - elapsed_values
    assert bounded[3]["prediction_mode_id"] == BOUNDED_SCHEDULE_MODE_ID
    assert np.max(np.abs(bounded_delta)) > 0.0
    assert np.all(np.abs(bounded_delta) <= np.abs(full_delta) + 1e-12)


def test_schedule_validation_blocks_late_or_incomplete_declarations() -> None:
    adapter, frames, _ = _bundle()
    model = _model_config(adapter)
    prefixes = frames["calibration_prefixes"]
    schedule = _forecast_schedule(prefixes, model)
    late = schedule.copy()
    late["declared_at_elapsed_days"] = late.groupby(
        ["cell_id", "landmark_visit_count"], sort=False
    )["forecast_elapsed_days"].transform("min")
    with pytest.raises(PrivateScheduleV4Error, match="declared after"):
        validate_private_forecast_schedule(late, prefixes, model)
    incomplete = schedule.iloc[1:].reset_index(drop=True)
    with pytest.raises(PrivateScheduleV4Error, match="forecast grid"):
        validate_private_forecast_schedule(incomplete, prefixes, model)
    leaked = schedule.assign(capacity_retention_pct=90.0)
    with pytest.raises(PrivateScheduleV4Error, match="columns changed"):
        validate_private_forecast_schedule(leaked, prefixes, model)


def test_elapsed_schedule_fails_closed_when_planned_conditions_are_ood() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    prefixes = frames["calibration_prefixes"]
    schedule = _forecast_schedule(prefixes, model)
    schedule["planned_temperature_c"] = 70.0
    predictions, decisions, _, manifest = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=schedule,
    )
    assert predictions.empty
    assert not decisions["issued"].any()
    assert set(decisions["evidence_status"]) == {
        "schedule_or_prefix_outside_training_support"
    }
    assert manifest["prediction_mode_id"] == ELAPSED_SCHEDULE_MODE_ID


def test_schedule_mode_requires_a_schedule_and_known_identity() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    arguments = (
        frames["development_trajectories"],
        frames["calibration_prefixes"],
        bundle_manifest,
        adapter,
        model,
    )
    with pytest.raises(PrivateEnterpriseCycleError, match="without a forecast"):
        predict_private_enterprise_cycle(
            *arguments,
            schedule_mode_id=ELAPSED_SCHEDULE_MODE_ID,
        )
    with pytest.raises(PrivateEnterpriseCycleError, match="unsupported"):
        predict_private_enterprise_cycle(
            *arguments,
            forecast_schedule=_forecast_schedule(arguments[1], model),
            schedule_mode_id="unknown_schedule_mode",
        )


def test_schedule_tampering_and_oracle_claim_are_fail_closed() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    prefixes = frames["calibration_prefixes"]
    oracle = _forecast_schedule(prefixes, model, schedule_role="oracle_upper_bound")
    predictions, decisions, _, manifest = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        prefixes,
        bundle_manifest,
        adapter,
        model,
        forecast_schedule=oracle,
    )
    assert manifest["primary_evidence_eligible"] is False
    changed = oracle.copy()
    changed.loc[0, "planned_temperature_c"] += 0.1
    with pytest.raises(PrivateEnterpriseCycleError, match="schedule changed"):
        score_private_enterprise_cycle(
            frames["calibration_truth_vault"],
            predictions,
            decisions,
            manifest,
            bundle_manifest,
            adapter,
            model,
            forecast_schedule=changed,
        )


def test_enterprise_prediction_never_accepts_truth_and_scores_after_freeze() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    predictions, decisions, capsule, prediction_manifest = (
        predict_private_enterprise_cycle(
            frames["development_trajectories"],
            frames["calibration_prefixes"],
            bundle_manifest,
            adapter,
            model,
        )
    )
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert tuple(decisions.columns) == DECISION_COLUMNS
    assert decisions["issued"].all()
    assert capsule["raw_training_rows_in_capsule"] is False
    assert prediction_manifest["target_truth_argument_accepted"] is False
    assert prediction_manifest["truth_vault_opened"] is False
    scores, summary = score_private_enterprise_cycle(
        frames["calibration_truth_vault"],
        predictions,
        decisions,
        prediction_manifest,
        bundle_manifest,
        adapter,
        model,
    )
    assert tuple(scores.columns) == SCORE_COLUMNS
    assert len(scores) == len(decisions)
    assert summary["evidence_role"] == "private_batch_disjoint_calibration"
    assert summary["public_release_permitted"] is False


def test_enterprise_prediction_replay_is_independent_of_truth_values() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    first = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        frames["locked_test_prefixes"],
        bundle_manifest,
        adapter,
        model,
    )
    changed_truth = frames["locked_test_truth_vault"].copy()
    changed_truth.loc[changed_truth["visit_index"] >= 3, "capacity_retention_pct"] -= (
        10.0
    )
    second = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        frames["locked_test_prefixes"],
        bundle_manifest,
        adapter,
        model,
    )
    pd.testing.assert_frame_equal(first[0], second[0], check_exact=True)
    pd.testing.assert_frame_equal(first[1], second[1], check_exact=True)
    assert first[2] == second[2]
    assert first[3] == second[3]
    with pytest.raises(PrivateCycleAdapterError, match="frame changed"):
        score_private_enterprise_cycle(
            changed_truth,
            first[0],
            first[1],
            first[3],
            bundle_manifest,
            adapter,
            model,
        )


def test_enterprise_score_rejects_prediction_and_manifest_tampering() -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    predictions, decisions, _, prediction_manifest = predict_private_enterprise_cycle(
        frames["development_trajectories"],
        frames["calibration_prefixes"],
        bundle_manifest,
        adapter,
        model,
    )
    attacked = predictions.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] += 0.1
    with pytest.raises(PrivateEnterpriseCycleError, match="changed after freeze"):
        score_private_enterprise_cycle(
            frames["calibration_truth_vault"],
            attacked,
            decisions,
            prediction_manifest,
            bundle_manifest,
            adapter,
            model,
        )
    attacked_manifest = dict(prediction_manifest)
    attacked_manifest["truth_vault_opened"] = True
    with pytest.raises(PrivateEnterpriseCycleError, match="manifest changed"):
        score_private_enterprise_cycle(
            frames["calibration_truth_vault"],
            predictions,
            decisions,
            attacked_manifest,
            bundle_manifest,
            adapter,
            model,
        )

    omitted_decisions = decisions.iloc[1:].reset_index(drop=True)
    removed = decisions.iloc[0]
    omitted_predictions = predictions.loc[
        ~(
            (predictions["cell_id"] == removed["cell_id"])
            & (predictions["landmark_visit_count"] == removed["landmark_visit_count"])
        )
    ].reset_index(drop=True)
    internally_rehashed = dict(prediction_manifest)
    internally_rehashed.pop("manifest_content_sha256")
    internally_rehashed["prediction_rows_sha256"] = canonical_frame_sha256(
        omitted_predictions, PREDICTION_COLUMNS
    )
    internally_rehashed["decision_rows_sha256"] = canonical_frame_sha256(
        omitted_decisions, DECISION_COLUMNS
    )
    internally_rehashed["prediction_row_count"] = len(omitted_predictions)
    internally_rehashed["decision_row_count"] = len(omitted_decisions)
    internally_rehashed["manifest_content_sha256"] = canonical_json_sha256(
        internally_rehashed
    )
    with pytest.raises(PrivateEnterpriseCycleError, match="sealed truth population"):
        score_private_enterprise_cycle(
            frames["calibration_truth_vault"],
            omitted_predictions,
            omitted_decisions,
            internally_rehashed,
            bundle_manifest,
            adapter,
            model,
        )


def test_private_enterprise_cli_prediction_then_scoring(tmp_path: Path) -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    input_directory = tmp_path / "input"
    prediction_directory = tmp_path / "prediction"
    score_directory = tmp_path / "score"
    input_directory.mkdir()
    adapter_path = input_directory / "adapter.json"
    bundle_path = input_directory / "bundle.json"
    model_path = input_directory / "model.json"
    development_path = input_directory / "development.parquet"
    prefix_path = input_directory / "prefixes.parquet"
    truth_path = input_directory / "truth.parquet"
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle_manifest), encoding="utf-8")
    model_path.write_text(json.dumps(model), encoding="utf-8")
    frames["development_trajectories"].to_parquet(development_path, index=False)
    frames["calibration_prefixes"].to_parquet(prefix_path, index=False)
    frames["calibration_truth_vault"].to_parquet(truth_path, index=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    runner = ROOT / "scripts/run_private_enterprise_cycle.py"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "predict",
            str(development_path),
            str(prefix_path),
            "--adapter-config",
            str(adapter_path),
            "--bundle-manifest",
            str(bundle_path),
            "--model-config",
            str(model_path),
            "--output-directory",
            str(prediction_directory),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    prediction_complete = json.loads(
        (prediction_directory / "prediction_complete.private.json").read_text(
            encoding="utf-8"
        )
    )
    assert prediction_complete["status"] == "complete"
    assert prediction_complete["metadata"]["truth_vault_opened"] is False
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "score",
            str(truth_path),
            "--prediction-directory",
            str(prediction_directory),
            "--adapter-config",
            str(adapter_path),
            "--bundle-manifest",
            str(bundle_path),
            "--output-directory",
            str(score_directory),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    score_complete = json.loads(
        (score_directory / "score_complete.private.json").read_text(encoding="utf-8")
    )
    assert score_complete["status"] == "complete"
    assert score_complete["metadata"]["operation"] == "post_freeze_truth_linkage"


def test_private_enterprise_cli_seals_declared_schedule(tmp_path: Path) -> None:
    adapter, frames, bundle_manifest = _bundle()
    model = _model_config(adapter)
    schedule = _forecast_schedule(frames["calibration_prefixes"], model)
    input_directory = tmp_path / "schedule_input"
    prediction_directory = tmp_path / "schedule_prediction"
    score_directory = tmp_path / "schedule_score"
    input_directory.mkdir()
    inputs = {
        "adapter": input_directory / "adapter.json",
        "bundle": input_directory / "bundle.json",
        "model": input_directory / "model.json",
        "development": input_directory / "development.parquet",
        "prefixes": input_directory / "prefixes.parquet",
        "truth": input_directory / "truth.parquet",
        "schedule": input_directory / "schedule.parquet",
    }
    inputs["adapter"].write_text(json.dumps(adapter), encoding="utf-8")
    inputs["bundle"].write_text(json.dumps(bundle_manifest), encoding="utf-8")
    inputs["model"].write_text(json.dumps(model), encoding="utf-8")
    frames["development_trajectories"].to_parquet(inputs["development"], index=False)
    frames["calibration_prefixes"].to_parquet(inputs["prefixes"], index=False)
    frames["calibration_truth_vault"].to_parquet(inputs["truth"], index=False)
    schedule.to_parquet(inputs["schedule"], index=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    runner = ROOT / "scripts/run_private_enterprise_cycle.py"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "predict",
            str(inputs["development"]),
            str(inputs["prefixes"]),
            "--forecast-schedule",
            str(inputs["schedule"]),
            "--adapter-config",
            str(inputs["adapter"]),
            "--bundle-manifest",
            str(inputs["bundle"]),
            "--model-config",
            str(inputs["model"]),
            "--output-directory",
            str(prediction_directory),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (prediction_directory / "forecast_schedule.private.parquet").is_file()
    prediction_manifest = json.loads(
        (prediction_directory / "prediction_manifest.private.json").read_text(
            encoding="utf-8"
        )
    )
    completion_manifest = json.loads(
        (prediction_directory / "prediction_complete.private.json").read_text(
            encoding="utf-8"
        )
    )
    assert prediction_manifest["prediction_mode_id"] == ELAPSED_SCHEDULE_MODE_ID
    assert completion_manifest["metadata"]["prediction_mode_id"] == (
        ELAPSED_SCHEDULE_MODE_ID
    )
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "score",
            str(inputs["truth"]),
            "--prediction-directory",
            str(prediction_directory),
            "--adapter-config",
            str(inputs["adapter"]),
            "--bundle-manifest",
            str(inputs["bundle"]),
            "--output-directory",
            str(score_directory),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(
        (score_directory / "score_summary.private.json").read_text(encoding="utf-8")
    )
    assert summary["prediction_mode_id"] == ELAPSED_SCHEDULE_MODE_ID
    assert summary["primary_evidence_eligible"] is True
