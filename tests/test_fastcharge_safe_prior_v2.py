from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.data.fastcharge_portability import (
    CANONICAL_CYCLE_COLUMNS,
    FastChargePortabilityDataError,
    prepare_fastcharge_portability_cycles,
)
from lifetwin.experiments.fastcharge_safe_prior_v2 import (
    FastChargeSafePriorV2Error,
    _safe_model_predictions,
    _safe_prior,
    _validate_replay,
    load_fastcharge_safe_prior_v2_config,
    validate_fastcharge_safe_prior_v2_config,
)
from lifetwin.experiments.fastcharge_trajectory_portability import (
    BASE_MODEL_IDS,
    PREDICTION_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/fastcharge_lfp_safe_prior_v2.json"


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return load_fastcharge_safe_prior_v2_config(CONFIG_PATH)


def test_config_freeze_horizon_and_claim_boundaries(
    config: dict[str, object],
) -> None:
    assert config["split_and_firewall"]["score_end_cycle"] == 300
    assert (
        config["pre_prediction_data_audit"][
            "target_cycle_201_to_300_capacity_scores_inspected"
        ]
        is False
    )
    assert (
        config["split_and_firewall"]["evaluation_target_suffix_available_to_prediction"]
        is False
    )
    assert config["uncertainty"]["formal_exchangeable_coverage_claim"] is False
    assert "independent_outcome_blind_external_confirmation" in set(
        config["claim_boundaries"]["prohibited_claims"]
    )
    attacked = copy.deepcopy(config)
    attacked["safe_prior"]["risk_inverse_power"] = 3.0
    with pytest.raises(FastChargeSafePriorV2Error, match="config changed"):
        validate_fastcharge_safe_prior_v2_config(attacked)


def test_safe_prior_assigns_exactly_zero_weight_to_unsafe_experts(
    config: dict[str, object],
) -> None:
    errors = {
        f"TRAIN_{index:02d}": {
            "target_prefix_persistence": 1.0,
            "target_prefix_full_linear": 100.0,
            "target_prefix_robust_recent_linear": 0.9,
            "target_prefix_constrained_sqrt_linear": 1.2,
            "nearest_neighbor_delta_transfer": 0.5,
        }
        for index in range(10)
    }
    prior = _safe_prior(errors, config)
    assert prior["eligible"] == [
        "target_prefix_persistence",
        "target_prefix_robust_recent_linear",
        "nearest_neighbor_delta_transfer",
    ]
    assert prior["weights"]["target_prefix_full_linear"] == 0.0
    assert prior["weights"]["target_prefix_constrained_sqrt_linear"] == 0.0

    experts = {
        model_id: np.asarray([float(index), float(index + 1)])
        for index, model_id in enumerate(BASE_MODEL_IDS)
    }
    diagnostics = {
        "risks": {
            "target_prefix_persistence": 1.0,
            "target_prefix_full_linear": 0.001,
            "target_prefix_robust_recent_linear": 0.8,
            "target_prefix_constrained_sqrt_linear": 0.002,
            "nearest_neighbor_delta_transfer": 0.4,
        },
        "selection_strength": 1.0,
    }
    _, metadata = _safe_model_predictions(experts, diagnostics, prior, config)
    assert metadata["final_weights"]["target_prefix_full_linear"] == 0.0
    assert metadata["final_weights"]["target_prefix_constrained_sqrt_linear"] == 0.0
    assert metadata["hard_model"] == "nearest_neighbor_delta_transfer"
    assert sum(metadata["final_weights"].values()) == pytest.approx(1.0)


def test_score_replay_rejects_self_consistent_prediction_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame(
        [{column: "frozen" for column in PREDICTION_COLUMNS}],
        columns=PREDICTION_COLUMNS,
    )
    manifest = {"prediction_sha256": "committed", "prediction_row_count": 1}
    monkeypatch.setattr(
        "lifetwin.experiments.fastcharge_safe_prior_v2."
        "predict_fastcharge_safe_prior_v2",
        lambda *_: (expected, manifest, pd.DataFrame(), pd.DataFrame()),
    )
    _validate_replay(
        expected.copy(),
        dict(manifest),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )

    attacked = expected.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] = "tampered"
    with pytest.raises(FastChargeSafePriorV2Error, match="deterministic frozen replay"):
        _validate_replay(
            attacked,
            dict(manifest),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )

    attacked_manifest = dict(manifest)
    attacked_manifest["prediction_sha256"] = "self-consistent-forgery"
    with pytest.raises(FastChargeSafePriorV2Error, match="manifest mismatch"):
        _validate_replay(
            expected.copy(),
            attacked_manifest,
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )


def _adapter_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    identities: list[dict[str, object]] = []
    split_ids = [
        (
            "train",
            ["MATR_B2C0"] + [f"TRAIN_{index:03d}" for index in range(40)],
        ),
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
                }
            )
    rows: list[dict[str, object]] = []
    for identity in identities:
        cell_id = str(identity["cell_id"])
        if cell_id == "MATR_B1C18":
            continue
        support = 299 if cell_id == "MATR_B2C1" else 300
        for cycle_index in range(1, support + 1):
            charge_time = 600.0 + cycle_index
            if cell_id == "MATR_B2C0" and cycle_index == 251:
                charge_time = np.nan
            rows.append(
                {
                    "source_barcode": identity["barcode"],
                    "cycle_index": cycle_index,
                    "discharge_capacity_ah": 1.1 - 0.0001 * cycle_index,
                    "internal_resistance_ohm": 0.015,
                    "temperature_max_c": 35.0,
                    "charge_time_s": charge_time,
                    "energy_efficiency": 0.88,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(identities)


def test_adapter_uses_past_only_fill_and_never_imputes_capacity(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
) -> None:
    raw, crosswalk = _adapter_fixture()
    crosswalk.attrs["sha256"] = config["dataset"]["authoritative_crosswalk_sha256"]
    monkeypatch.setattr(
        "lifetwin.data.fastcharge_portability.load_severson_crosswalk",
        lambda _: crosswalk,
    )
    canonical, audit = prepare_fastcharge_portability_cycles(
        raw,
        "mock-crosswalk.csv",
        config,
    )
    assert tuple(canonical.columns) == CANONICAL_CYCLE_COLUMNS
    filled = canonical.loc[
        (canonical["cell_id"] == "MATR_B2C0") & (canonical["cycle_index"] == 251),
        "charge_time_s",
    ].item()
    assert filled == 850.0
    assert audit["past_only_forward_fill_count"] == 1
    assert audit["past_only_forward_fill_records"] == [
        {
            "paper_split": "train",
            "cell_id": "MATR_B2C0",
            "cycle_index": 251,
            "column": "charge_time_s",
        }
    ]

    attacked = raw.copy()
    attacked.loc[
        (attacked["source_barcode"] == "BARCODE_MATR_B2C0")
        & (attacked["cycle_index"] == 251),
        "discharge_capacity_ah",
    ] = np.nan
    with pytest.raises(
        FastChargePortabilityDataError,
        match="non-imputable",
    ):
        prepare_fastcharge_portability_cycles(
            attacked,
            "mock-crosswalk.csv",
            config,
        )
