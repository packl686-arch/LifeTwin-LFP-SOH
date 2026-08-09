from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.validation.private_cycle_adapter import (
    PARTITIONED_PREFIX_COLUMNS,
    PARTITIONED_TRAJECTORY_COLUMNS,
    PARTITION_METADATA_COLUMNS,
    PRIVATE_MEASUREMENT_COLUMNS,
    PrivateCycleAdapterError,
    build_private_cycle_blind_bundle,
    freeze_private_cycle_partitions,
    normalize_private_cycle_measurements,
    validate_private_cycle_adapter_config,
    validate_private_cycle_partition_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/validation/hithium_private_cycle_adapter_v1.json"


def _config() -> dict[str, object]:
    return validate_private_cycle_adapter_config(
        json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": f"cell_{batch}_{replicate}",
                "batch_id": f"batch_{batch}",
                "condition_id": f"condition_{batch % 3}",
            }
            for batch in range(5)
            for replicate in range(2)
        ],
        columns=PARTITION_METADATA_COLUMNS,
    )


def _measurements() -> pd.DataFrame:
    rows = []
    for identity in _metadata().itertuples(index=False):
        for visit in range(6):
            rows.append(
                {
                    "record_id": f"{identity.cell_id}_rpt_{visit}",
                    "cell_id": identity.cell_id,
                    "batch_id": identity.batch_id,
                    "condition_id": identity.condition_id,
                    "cathode_chemistry": "LFP",
                    "temperature_c": 25.0,
                    "min_soc_pct": 10.0,
                    "max_soc_pct": 90.0,
                    "charge_c_rate": 0.5,
                    "discharge_c_rate": 1.0,
                    "visit_index": visit,
                    "elapsed_days": float(visit * 30),
                    "equivalent_full_cycles": float(visit * 250),
                    "capacity_ah": 280.0 - 1.4 * visit,
                    "reference_capacity_ah": 280.0,
                    "quality_status": "accepted",
                }
            )
    return pd.DataFrame(rows, columns=PRIVATE_MEASUREMENT_COLUMNS)


def test_partition_freeze_is_metadata_only_deterministic_and_batch_disjoint() -> None:
    config = _config()
    first = freeze_private_cycle_partitions(_metadata(), config)
    second = freeze_private_cycle_partitions(_metadata(), config)
    assert first == second
    assert first["measurement_value_fields_accepted"] is False
    assert first["partition_selection_uses_target_outcomes"] is False
    assignments = pd.DataFrame(first["assignments"])
    assert assignments.groupby("batch_id")["partition"].nunique().max() == 1
    assert set(assignments["partition"]) == {
        "development",
        "calibration",
        "locked_test",
    }
    exposed = _metadata().assign(capacity_ah=280.0)
    with pytest.raises(PrivateCycleAdapterError, match="accepts only"):
        freeze_private_cycle_partitions(exposed, config)


def test_partition_manifest_tampering_is_rejected() -> None:
    config = _config()
    manifest = freeze_private_cycle_partitions(_metadata(), config)
    original = str(manifest["assignments"][0]["partition"])
    manifest["assignments"][0]["partition"] = (
        "development" if original != "development" else "locked_test"
    )
    with pytest.raises(PrivateCycleAdapterError, match="hash changed"):
        validate_private_cycle_partition_manifest(manifest, config)


def test_private_bundle_separates_prefixes_from_truth_vaults() -> None:
    config = _config()
    manifest = freeze_private_cycle_partitions(_metadata(), config)
    normalized = normalize_private_cycle_measurements(
        _measurements(), manifest, config
    )
    assert tuple(normalized.columns) == PARTITIONED_TRAJECTORY_COLUMNS
    assert normalized["capacity_retention_pct"].between(0.0, 110.0).all()
    frames, bundle = build_private_cycle_blind_bundle(
        normalized, manifest, config
    )
    assert bundle["prediction_inputs_contain_target_suffix_outcomes"] is False
    assert bundle["truth_vault_must_be_inaccessible_to_prediction_process"] is True
    for partition in ("calibration", "locked_test"):
        prefixes = frames[f"{partition}_prefixes"]
        truth = frames[f"{partition}_truth_vault"]
        assert tuple(prefixes.columns) == PARTITIONED_PREFIX_COLUMNS
        assert tuple(truth.columns) == PARTITIONED_TRAJECTORY_COLUMNS
        for (_, landmark), group in prefixes.groupby(
            ["cell_id", "landmark_visit_count"]
        ):
            assert len(group) == int(landmark)
        assert set(prefixes["cell_id"]) == set(truth["cell_id"])
        assert len(prefixes) < 2 * len(truth)
    development_cells = set(frames["development_trajectories"]["cell_id"])
    target_cells = set(frames["calibration_truth_vault"]["cell_id"]) | set(
        frames["locked_test_truth_vault"]["cell_id"]
    )
    assert development_cells.isdisjoint(target_cells)


def test_measurement_identity_and_quality_fail_closed() -> None:
    config = _config()
    manifest = freeze_private_cycle_partitions(_metadata(), config)
    changed_identity = _measurements()
    changed_identity.loc[0, "batch_id"] = "another_batch"
    with pytest.raises(PrivateCycleAdapterError, match="identity changed"):
        normalize_private_cycle_measurements(changed_identity, manifest, config)
    bad_quality = _measurements()
    bad_quality.loc[0, "quality_status"] = "manual_override"
    with pytest.raises(PrivateCycleAdapterError, match="not allowlisted"):
        normalize_private_cycle_measurements(bad_quality, manifest, config)


def test_bundle_rejects_insufficient_frozen_future_support() -> None:
    config = _config()
    manifest = freeze_private_cycle_partitions(_metadata(), config)
    measurements = _measurements().loc[lambda frame: frame["visit_index"] < 5]
    normalized = normalize_private_cycle_measurements(measurements, manifest, config)
    with pytest.raises(PrivateCycleAdapterError, match="score-window future support"):
        build_private_cycle_blind_bundle(normalized, manifest, config)


def test_bundle_requires_future_visits_inside_score_window() -> None:
    config = _config()
    config["trajectory_policy"]["score_end_equivalent_full_cycles"] = 800.0
    manifest = freeze_private_cycle_partitions(_metadata(), config)
    normalized = normalize_private_cycle_measurements(
        _measurements(), manifest, config
    )
    with pytest.raises(PrivateCycleAdapterError, match="score-window future support"):
        build_private_cycle_blind_bundle(normalized, manifest, config)
