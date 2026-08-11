from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd

from lifetwin.validation.private_cycle_adapter import (
    PARTITION_METADATA_COLUMNS,
    PRIVATE_MEASUREMENT_COLUMNS,
    build_private_cycle_blind_bundle,
    freeze_private_cycle_partitions,
    normalize_private_cycle_measurements,
    validate_private_cycle_adapter_config,
)
from lifetwin.validation.private_prefix_readiness import (
    audit_private_prefix_readiness,
    validate_private_prefix_readiness_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _adapter() -> dict[str, object]:
    return validate_private_cycle_adapter_config(
        json.loads(
            (
                ROOT / "configs/validation/hithium_private_cycle_adapter_v1.json"
            ).read_text(encoding="utf-8")
        )
    )


def _readiness() -> dict[str, object]:
    return validate_private_prefix_readiness_config(
        json.loads(
            (
                ROOT / "configs/validation/hithium_private_prefix_readiness_v1.json"
            ).read_text(encoding="utf-8")
        )
    )


def _bundle() -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    metadata = pd.DataFrame(
        [
            {
                "cell_id": f"cell_{batch}_{replicate}",
                "batch_id": f"batch_{batch}",
                "condition_id": "condition_shared",
            }
            for batch in range(5)
            for replicate in range(2)
        ],
        columns=PARTITION_METADATA_COLUMNS,
    )
    rows = []
    for identity in metadata.itertuples(index=False):
        for visit in range(6):
            rows.append(
                {
                    "record_id": f"{identity.cell_id}_{visit}",
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
    adapter = _adapter()
    partition = freeze_private_cycle_partitions(metadata, adapter)
    normalized = normalize_private_cycle_measurements(
        pd.DataFrame(rows, columns=PRIVATE_MEASUREMENT_COLUMNS),
        partition,
        adapter,
    )
    frames, _ = build_private_cycle_blind_bundle(normalized, partition, adapter)
    return frames, partition


def test_readiness_passes_supported_prefixes_without_truth_inputs() -> None:
    frames, partition = _bundle()
    drift, decision = audit_private_prefix_readiness(
        frames["development_trajectories"],
        frames["calibration_prefixes"],
        frames["locked_test_prefixes"],
        partition,
        _adapter(),
        _readiness(),
    )
    assert not drift.empty
    assert drift["within_development_support"].all()
    assert decision["ready_to_issue_predictions"] is True
    assert decision["truth_vault_inputs_read"] is False
    assert decision["locked_test_truth_may_be_opened"] is False
    assert decision["model_accuracy_evidence_created"] is False


def test_readiness_fails_closed_on_prefix_domain_shift() -> None:
    frames, partition = _bundle()
    calibration = frames["calibration_prefixes"].copy()
    calibration["temperature_c"] = 80.0
    _, decision = audit_private_prefix_readiness(
        frames["development_trajectories"],
        calibration,
        frames["locked_test_prefixes"],
        partition,
        _adapter(),
        _readiness(),
    )
    assert decision["ready_to_issue_predictions"] is False
    assert (
        decision["decision_checks"]["calibration_within_development_support"] is False
    )


def test_readiness_api_has_no_truth_vault_parameter() -> None:
    parameters = inspect.signature(audit_private_prefix_readiness).parameters
    assert not any("truth" in name for name in parameters)
