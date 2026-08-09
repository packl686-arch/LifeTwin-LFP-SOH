from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from lifetwin.experiments.private_enterprise_cycle import (
    default_private_enterprise_v3_config,
    predict_private_enterprise_cycle,
    score_private_enterprise_cycle,
)
from lifetwin.experiments.private_schedule_v4 import (
    BOUNDED_SCHEDULE_MODE_ID,
    ELAPSED_SCHEDULE_MODE_ID,
    FORECAST_SCHEDULE_COLUMNS,
)
from lifetwin.experiments.private_schedule_v4_gates import (
    evaluate_private_schedule_v4_gates,
)
from lifetwin.private_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    build_completion_manifest,
    exclusive_run_lock,
)
from lifetwin.validation.private_cycle_adapter import (
    PARTITION_METADATA_COLUMNS,
    PRIVATE_MEASUREMENT_COLUMNS,
    build_private_cycle_blind_bundle,
    freeze_private_cycle_partitions,
    normalize_private_cycle_measurements,
    validate_private_cycle_adapter_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": f"synthetic_cell_{index:02d}",
                "batch_id": f"synthetic_batch_{index:02d}",
                "condition_id": f"synthetic_condition_{index:02d}",
            }
            for index in range(24)
        ],
        columns=PARTITION_METADATA_COLUMNS,
    )


def _synthetic_measurements(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, identity in enumerate(metadata.itertuples(index=False)):
        temperature = 20.0 + 5.0 * (index % 4)
        discharge_rate = 0.5 + 0.5 * (index % 3)
        fade_scale = 0.75 + 0.08 * (index % 5)
        for visit in range(7):
            retention = 100.0 - fade_scale * visit * 0.55
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
                    "discharge_c_rate": discharge_rate,
                    "visit_index": visit,
                    "elapsed_days": float(visit * 45),
                    "equivalent_full_cycles": float(visit * 250),
                    "capacity_ah": 280.0 * retention / 100.0,
                    "reference_capacity_ah": 280.0,
                    "quality_status": "accepted",
                }
            )
    return pd.DataFrame(rows, columns=PRIVATE_MEASUREMENT_COLUMNS)


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


def _schedule(
    prefixes: pd.DataFrame,
    model: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cell_id, landmark), prefix in prefixes.groupby(
        ["cell_id", "landmark_visit_count"], sort=True
    ):
        last = prefix.sort_values("visit_index", kind="stable").iloc[-1]
        last_efc = float(last["equivalent_full_cycles"])
        last_days = float(last["elapsed_days"])
        step = float(model["forecast_grid_step_equivalent_full_cycles"])
        end = float(model["score_end_equivalent_full_cycles"])
        first = math.ceil((last_efc + 1e-12) / step) * step
        grid = np.arange(first, end + step * 0.5, step, dtype=float)
        duty = last_efc / last_days
        for exposure in grid[grid > last_efc]:
            rows.append(
                {
                    "partition": str(last["partition"]),
                    "cell_id": str(cell_id),
                    "condition_id": str(last["condition_id"]),
                    "landmark_visit_count": int(landmark),
                    "scenario_id": "synthetic_constant_plan_v1",
                    "schedule_role": "deployment_candidate",
                    "schedule_source": "declared_operating_plan",
                    "declared_at_elapsed_days": last_days,
                    "forecast_elapsed_days": float(exposure / duty),
                    "forecast_equivalent_full_cycles": float(exposure),
                    "planned_temperature_c": float(last["temperature_c"]),
                    "planned_min_soc_pct": 10.0,
                    "planned_max_soc_pct": 90.0,
                    "planned_charge_c_rate": 0.5,
                    "planned_discharge_c_rate": float(last["discharge_c_rate"]),
                }
            )
    return pd.DataFrame(rows, columns=FORECAST_SCHEDULE_COLUMNS)


def _predict_and_score(
    mode_id: str | None,
    development: pd.DataFrame,
    prefixes: pd.DataFrame,
    truth: pd.DataFrame,
    schedule: pd.DataFrame | None,
    bundle: dict[str, object],
    adapter: dict[str, object],
    model: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    predictions, decisions, _, manifest = predict_private_enterprise_cycle(
        development,
        prefixes,
        bundle,
        adapter,
        model,
        forecast_schedule=schedule,
        schedule_mode_id=mode_id,
    )
    scores, summary = score_private_enterprise_cycle(
        truth,
        predictions,
        decisions,
        manifest,
        bundle,
        adapter,
        model,
        forecast_schedule=schedule,
    )
    return scores, summary, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an outcome-labeled synthetic dry run of the private workflow."
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Synthetic dry-run output directory is not empty")
    with exclusive_run_lock(output):
        adapter = json.loads(
            (
                ROOT / "configs/validation/hithium_private_cycle_adapter_v1.json"
            ).read_text(encoding="utf-8")
        )
        adapter["dataset_id"] = "SYNTHETIC_PRIVATE_DRY_RUN_V1"
        adapter["partition_policy"]["hash_seed"] = (
            "synthetic-dry-run-seed-not-for-production"
        )
        adapter["trajectory_policy"]["score_end_equivalent_full_cycles"] = 1500.0
        adapter = validate_private_cycle_adapter_config(adapter)
        metadata = _synthetic_metadata()
        partition_manifest = freeze_private_cycle_partitions(metadata, adapter)
        normalized = normalize_private_cycle_measurements(
            _synthetic_measurements(metadata), partition_manifest, adapter
        )
        frames, bundle = build_private_cycle_blind_bundle(
            normalized, partition_manifest, adapter
        )
        model = _model_config(adapter)
        development = frames["development_trajectories"]
        calibration_prefixes = frames["calibration_prefixes"]
        calibration_truth = frames["calibration_truth_vault"]
        schedule = _schedule(calibration_prefixes, model)
        baseline_scores, baseline_summary, _ = _predict_and_score(
            None,
            development,
            calibration_prefixes,
            calibration_truth,
            None,
            bundle,
            adapter,
            model,
        )
        candidates: dict[str, tuple[pd.DataFrame, dict[str, object]]] = {}
        protocols = {
            ELAPSED_SCHEDULE_MODE_ID: (
                ROOT
                / "configs/experiments/private_enterprise_schedule_v4_1_amendment.json"
            ),
            BOUNDED_SCHEDULE_MODE_ID: (
                ROOT
                / "configs/experiments/private_enterprise_schedule_v4_2_preregistered.json"
            ),
        }
        gate_results: dict[str, object] = {}
        for mode_id, protocol_path in protocols.items():
            scores, summary, _ = _predict_and_score(
                mode_id,
                development,
                calibration_prefixes,
                calibration_truth,
                schedule,
                bundle,
                adapter,
                model,
            )
            candidates[mode_id] = (scores, summary)
            gate_results[mode_id] = evaluate_private_schedule_v4_gates(
                baseline_scores,
                scores,
                baseline_summary,
                summary,
                json.loads(protocol_path.read_text(encoding="utf-8")),
            )
        promoted = [
            mode_id
            for mode_id, result in gate_results.items()
            if result["promote_candidate"]
        ]
        if len(promoted) == 1:
            selected_mode = promoted[0]
        elif len(promoted) == 2:
            risks = {
                mode_id: float(
                    np.mean(
                        [
                            row["condition_equal_trajectory_iae_pp"]
                            for row in candidates[mode_id][1][
                                "summary_by_landmark"
                            ].values()
                        ]
                    )
                )
                for mode_id in promoted
            }
            difference = abs(
                risks[ELAPSED_SCHEDULE_MODE_ID] - risks[BOUNDED_SCHEDULE_MODE_ID]
            )
            selected_mode = (
                ELAPSED_SCHEDULE_MODE_ID
                if difference <= 0.02
                else min(risks, key=risks.get)
            )
        else:
            selected_mode = None
        locked_schedule = (
            _schedule(frames["locked_test_prefixes"], model)
            if selected_mode is not None
            else None
        )
        locked_predictions, locked_decisions, _, locked_manifest = (
            predict_private_enterprise_cycle(
                development,
                frames["locked_test_prefixes"],
                bundle,
                adapter,
                model,
                forecast_schedule=locked_schedule,
                schedule_mode_id=selected_mode,
            )
        )
        atomic_write_csv(baseline_scores, output / "calibration_v3_scores.csv")
        atomic_write_json(baseline_summary, output / "calibration_v3_summary.json")
        for mode_id, (scores, summary) in candidates.items():
            atomic_write_csv(scores, output / f"calibration_{mode_id}_scores.csv")
            atomic_write_json(summary, output / f"calibration_{mode_id}_summary.json")
            atomic_write_json(gate_results[mode_id], output / f"gate_{mode_id}.json")
        atomic_write_parquet(
            locked_predictions, output / "locked_predictions.synthetic.parquet"
        )
        atomic_write_parquet(
            locked_decisions, output / "locked_decisions.synthetic.parquet"
        )
        atomic_write_json(
            locked_manifest, output / "locked_prediction_manifest.synthetic.json"
        )
        result = {
            "schema_version": "lifetwin.private_enterprise.synthetic_dry_run.v1",
            "synthetic_only": True,
            "hithium_data_accessed": False,
            "calibration_truth_opened": True,
            "locked_test_truth_opened": False,
            "candidate_gate_pass": {
                mode_id: bool(gate["promote_candidate"])
                for mode_id, gate in gate_results.items()
            },
            "selected_schedule_mode": selected_mode,
            "fallback_to_v3": selected_mode is None,
            "locked_prediction_manifest_sha256": locked_manifest[
                "manifest_content_sha256"
            ],
            "claim_boundary": (
                "Synthetic software rehearsal only; no battery or enterprise "
                "accuracy evidence."
            ),
        }
        atomic_write_json(result, output / "dry_run_summary.json")
        artifacts = {
            path.name: path
            for path in output.iterdir()
            if path.is_file() and path.name != "dry_run_complete.json"
        }
        completion = build_completion_manifest(
            output,
            artifacts,
            metadata={
                "operation": "synthetic_private_enterprise_rehearsal",
                "hithium_data_accessed": False,
                "locked_test_truth_opened": False,
                "public_release_permitted": False,
            },
        )
        atomic_write_json(completion, output / "dry_run_complete.json")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
