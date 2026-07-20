from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.experiments.calendar_v3_activation_development import (
    GATED_TARGET_ACTIVATION_METHOD,
    PRIMARY_PREFIX,
    SOC_SCENARIO,
    TEMPERATURE_SCENARIO,
)
from lifetwin.models.calendar_v3_activation import activation_mechanism_gate
from scripts.verify_public_release import verify


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv"
EXPECTED_DATA_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_data_identity_license_and_statistical_unit(
    observations: pd.DataFrame,
) -> None:
    assert _sha256(DATA_PATH) == EXPECTED_DATA_SHA256
    assert len(observations) == 595
    assert observations["condition_id"].nunique() == 17
    assert set(observations["source_license"]) == {"CC-BY-4.0"}
    assert set(observations["physical_replicates_aggregated"]) == {3}
    assert set(observations["statistical_unit"]) == {
        "temperature_soc_condition_mean_trajectory"
    }


def test_low_soc_gate_requires_shape_evidence_and_sufficient_prefix(
    observations: pd.DataFrame,
) -> None:
    target = observations.loc[
        observations["condition_id"] == "NAUMANN_CAL_T40_SOC12.5"
    ]
    gate5 = activation_mechanism_gate(target.loc[target["checkup_index"] < 5])
    gate8 = activation_mechanism_gate(target.loc[target["checkup_index"] < 8])
    assert gate5.negative_loss_evidence
    assert not gate5.ready
    assert gate8.negative_loss_evidence
    assert gate8.ready


def test_phase8_public_reproduction(completed_run: tuple) -> None:
    result, predictions, _, _, comparisons, diagnostics = completed_run[:6]
    assert len(predictions) == 12978
    assert result["development_gate"]["confirmation_status"] == (
        "blocked_pending_independent_dataset"
    )
    assert result["mechanism_support"][
        "unique_gate_ready_condition_ids_at_primary_prefix"
    ] == [
        "NAUMANN_CAL_T25_SOC0",
        "NAUMANN_CAL_T40_SOC0",
        "NAUMANN_CAL_T40_SOC12.5",
    ]
    primary = comparisons.loc[
        (comparisons["prefix_checkups"] == PRIMARY_PREFIX)
        & (comparisons["candidate_method"] == GATED_TARGET_ACTIVATION_METHOD)
    ].set_index("scenario")
    assert primary.loc[
        TEMPERATURE_SCENARIO, "candidate_trajectory_iae_pp_mean"
    ] == pytest.approx(0.36622558367548513, abs=1e-8)
    assert primary.loc[
        SOC_SCENARIO, "candidate_trajectory_iae_pp_mean"
    ] == pytest.approx(0.20968832280689345, abs=1e-8)
    assert not primary[
        "descriptive_strict_superiority_criterion_met"
    ].any()
    assert diagnostics.loc[
        (diagnostics["prefix_checkups"] == PRIMARY_PREFIX)
        & diagnostics["activation_gate_ready"],
        "target_condition_id",
    ].nunique() == 3


def test_release_manifest_and_exclusion_rules() -> None:
    result = verify(PROJECT_ROOT)
    assert result["status"] == "passed", result
    assert result["version_consistency"]["status"] == "passed"
