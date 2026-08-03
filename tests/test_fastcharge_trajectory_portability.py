from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.data.fastcharge_portability import (
    CANONICAL_CYCLE_COLUMNS,
    DATASET_ID,
    TARGET_PREFIX_COLUMNS,
    build_fastcharge_prediction_inputs,
    prepare_fastcharge_portability_cycles,
)
from lifetwin.experiments.fastcharge_trajectory_portability import (
    CALIBRATION_COLUMNS,
    MODEL_IDS,
    PREDICTION_COLUMNS,
    FastChargeTrajectoryPortabilityError,
    _streaming_frame_sha256,
    load_fastcharge_trajectory_config,
    predict_fastcharge_trajectory_portability,
    validate_fastcharge_trajectory_config,
)
from lifetwin.experiments.nasa_prefix_loco import canonical_frame_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/fastcharge_lfp_trajectory_portability_v1.json"
)


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return load_fastcharge_trajectory_config(CONFIG_PATH)


def _canonical_cycles() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_sizes = (("train", 41), ("primary_test", 41), ("secondary_test", 40))
    global_index = 0
    for split, count in split_sizes:
        for local_index in range(count):
            cell_id = f"SYN_{split.upper()}_{local_index:03d}"
            rate = 0.000025 + global_index * 0.0000007
            curvature = 0.00035 + (global_index % 9) * 0.000015
            phase = (global_index % 13) / 3.0
            initial = 1.05 + 0.002 * (global_index % 11)
            for cycle_index in range(1, 201):
                progress = (cycle_index - 1) / 199.0
                retention = (
                    1.0
                    - rate * (cycle_index - 1)
                    - curvature * np.sqrt(cycle_index - 1)
                    + 0.0005 * np.sin(cycle_index / 11.0 + phase)
                )
                rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "cell_id": cell_id,
                        "paper_split": split,
                        "cycle_index": cycle_index,
                        "discharge_capacity_ah": float(initial * retention),
                        "internal_resistance_ohm": float(
                            0.014
                            + 0.00002 * (global_index % 7)
                            + 0.000005 * cycle_index
                        ),
                        "temperature_max_c": float(
                            34.0 + (global_index % 5) + 0.4 * progress
                        ),
                        "charge_time_s": float(
                            590.0 + 4.0 * (global_index % 17) + 8.0 * progress
                        ),
                        "energy_efficiency": float(
                            0.88 - 0.00001 * cycle_index + 0.0001 * (global_index % 3)
                        ),
                    }
                )
            global_index += 1
    return pd.DataFrame(rows, columns=CANONICAL_CYCLE_COLUMNS)


@pytest.fixture(scope="module")
def prediction_bundle(
    config: dict[str, object],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    cycles = _canonical_cycles()
    training, prefixes, _ = build_fastcharge_prediction_inputs(cycles, config)
    predictions, manifest, calibration = predict_fastcharge_trajectory_portability(
        training, prefixes, config
    )
    return cycles, training, prefixes, predictions, manifest, calibration


def test_config_freeze_and_claim_boundaries(config: dict[str, object]) -> None:
    assert (
        config["split_and_firewall"]["evaluation_target_suffix_available_to_prediction"]
        is False
    )
    assert config["uncertainty"]["formal_exchangeable_coverage_claim"] is False
    assert config["evidence_role"].startswith("outcome_exposed_cohort")
    assert "independent_outcome_blind_external_confirmation" in set(
        config["claim_boundaries"]["prohibited_claims"]
    )
    attacked = copy.deepcopy(config)
    attacked["mixture"]["risk_inverse_power"] = 3.0
    with pytest.raises(FastChargeTrajectoryPortabilityError, match="config changed"):
        validate_fastcharge_trajectory_config(attacked)


def test_streaming_prediction_hash_matches_canonical_hash() -> None:
    frame = pd.DataFrame(
        [
            {"a": "x", "b": 1, "c": 2.5},
            {"a": "y", "b": 2, "c": 3.5},
        ],
        columns=("a", "b", "c"),
    )
    assert _streaming_frame_sha256(frame, frame.columns) == canonical_frame_sha256(
        frame,
        frame.columns,
    )


def test_prediction_surface_is_prefix_only_and_deterministic(
    config: dict[str, object],
    prediction_bundle: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, object],
        pd.DataFrame,
    ],
) -> None:
    cycles, training, prefixes, predictions, manifest, calibration = prediction_bundle
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert tuple(calibration.columns) == CALIBRATION_COLUMNS
    assert set(predictions["model_id"]) == set(MODEL_IDS)
    assert manifest["evaluation_target_future_outcomes_used"] is False
    assert manifest["interval_calibration_target_suffix_used"] is False
    assert manifest["prediction_row_count"] == len(predictions)
    assert len(predictions) == 375_840
    for (_, prefix_cycle), group in prefixes.groupby(
        ["cell_id", "prefix_cycle"], sort=True
    ):
        assert group["cycle_index"].max() == prefix_cycle
        assert len(group) == prefix_cycle

    mutated = cycles.copy()
    suffix = (mutated["paper_split"] != "train") & (mutated["cycle_index"] > 100)
    mutated.loc[suffix, "discharge_capacity_ah"] *= 0.8
    mutated_training, mutated_prefixes, _ = build_fastcharge_prediction_inputs(
        mutated,
        config,
    )
    pd.testing.assert_frame_equal(training, mutated_training)
    pd.testing.assert_frame_equal(prefixes, mutated_prefixes)


def test_prediction_rejects_future_row_injection(
    config: dict[str, object],
    prediction_bundle: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, object],
        pd.DataFrame,
    ],
) -> None:
    _, training, prefixes, _, _, _ = prediction_bundle
    attacked = prefixes.copy()
    source = attacked.loc[
        (attacked["cell_id"] == attacked["cell_id"].iloc[0])
        & (attacked["prefix_cycle"] == 40)
        & (attacked["cycle_index"] == 40)
    ].copy()
    source["cycle_index"] = 41
    attacked = pd.concat([attacked, source], ignore_index=True)
    with pytest.raises(
        FastChargeTrajectoryPortabilityError,
        match="not exactly truncated",
    ):
        predict_fastcharge_trajectory_portability(training, attacked, config)


def test_authoritative_adapter_removes_outcome_columns(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
) -> None:
    identities: list[dict[str, object]] = []
    split_ids = [
        ("train", [f"TRAIN_{index:03d}" for index in range(41)]),
        (
            "primary_test",
            ["MATR_B1C18", "MATR_B2C1"]
            + [f"PRIMARY_{index:03d}" for index in range(41)],
        ),
        (
            "secondary_test",
            [f"SECONDARY_{index:03d}" for index in range(40)],
        ),
    ]
    for split, cell_ids in split_ids:
        for cell_id in cell_ids:
            identities.append(
                {
                    "cell_id": cell_id,
                    "barcode": f"BARCODE_{cell_id}",
                    "paper_split": split,
                    "cycle_life": 999,
                }
            )
    crosswalk = pd.DataFrame(identities)
    crosswalk.attrs["sha256"] = config["dataset"]["authoritative_crosswalk_sha256"]
    monkeypatch.setattr(
        "lifetwin.data.fastcharge_portability.load_severson_crosswalk",
        lambda _: crosswalk,
    )
    raw_rows: list[dict[str, object]] = []
    for row in identities:
        if row["cell_id"] == "MATR_B1C18":
            continue
        support = 172 if row["cell_id"] == "MATR_B2C1" else 200
        for cycle_index in range(1, support + 1):
            raw_rows.append(
                {
                    "source_barcode": row["barcode"],
                    "cell_id": f"RAW_{row['cell_id']}",
                    "cycle_index": cycle_index,
                    "discharge_capacity_ah": 1.1 - 0.0001 * cycle_index,
                    "internal_resistance_ohm": 0.015,
                    "temperature_max_c": 35.0,
                    "charge_time_s": 600.0,
                    "energy_efficiency": 0.88,
                    "cycle_life": 999,
                }
            )
    canonical, audit = prepare_fastcharge_portability_cycles(
        pd.DataFrame(raw_rows),
        "mock-crosswalk.csv",
        config,
    )
    assert tuple(canonical.columns) == CANONICAL_CYCLE_COLUMNS
    assert "cycle_life" not in canonical
    assert canonical["cell_id"].nunique() == 122
    assert audit["missing_official_cells"] == ["MATR_B1C18"]
    assert audit["fixed_horizon_support_exclusions"] == ["MATR_B2C1"]
    assert audit["trajectory_suffix_scoring_performed"] is False
    assert tuple(build_fastcharge_prediction_inputs(canonical, config)[1].columns) == (
        TARGET_PREFIX_COLUMNS
    )
