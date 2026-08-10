from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import fastcharge_v8_measurement_stability as v8
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError
from scripts import compile_fastcharge_v8_stage_b_commitment as compile_v8
from scripts import issue_fastcharge_v8_measurement_stability as issue_v8
from scripts import prepare_fastcharge_v8_measurement_quality as prepare_v8


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/experiments/v8_measurement_stability_synthetic_dry_run.json"
)
EXECUTION_CONFIG_PATH = (
    ROOT / "configs/experiments/v8_measurement_stability_execution.template.json"
)
PARENT_PATH = (
    ROOT / "configs/experiments/v8_measurement_stability_blind_protocol.template.json"
)
CANDIDATE_PATH = (
    ROOT / "configs/experiments/v7_p100_reissue_innovation_blind_candidate.json"
)
PUBLISHED_DECISION_PATH = ROOT / "showcase/evidence_v8_dry_run/decision.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))


def _row(
    *,
    role: str,
    retention: float,
    tester: str,
    chamber: str,
    measurement_date: date,
    repeat_index: int,
    cell_id: str = "",
    landmark: int = 0,
    reference_id: str = "",
    bridge_id: str = "",
) -> dict[str, object]:
    return {
        "record_role": role,
        "physical_cell_id": cell_id,
        "landmark_cycle": landmark,
        "repeat_index": repeat_index,
        "retention_pct": retention,
        "tester_id": tester,
        "temperature_chamber_id": chamber,
        "measurement_date": measurement_date.isoformat(),
        "reference_channel_id": reference_id,
        "bridge_id": bridge_id,
    }


def _measurement_frame(seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    start = date(2026, 2, 2)
    for cell_index in range(24):
        tester = "TESTER_A" if cell_index % 2 == 0 else "TESTER_B"
        chamber = "CHAMBER_1" if tester == "TESTER_A" else "CHAMBER_2"
        for landmark in (60, 100):
            center = 100.0 - 0.01 * landmark - 0.01 * cell_index
            for repeat_index in range(3):
                rows.append(
                    _row(
                        role="cell_repeat",
                        retention=float(center + rng.normal(0.0, 0.003)),
                        tester=tester,
                        chamber=chamber,
                        measurement_date=start + timedelta(days=cell_index % 4),
                        repeat_index=repeat_index,
                        cell_id=f"CELL_{cell_index:02d}",
                        landmark=landmark,
                    )
                )
    for tester, chamber in (
        ("TESTER_A", "CHAMBER_1"),
        ("TESTER_B", "CHAMBER_2"),
    ):
        for day_index in range(4):
            for repeat_index in range(3):
                rows.append(
                    _row(
                        role="daily_reference",
                        retention=float(
                            99.5 + 0.002 * day_index + rng.normal(0.0, 0.001)
                        ),
                        tester=tester,
                        chamber=chamber,
                        measurement_date=start + timedelta(days=day_index),
                        repeat_index=repeat_index,
                        reference_id=f"REF_{tester}",
                    )
                )
    for bridge_index in range(4):
        for tester, chamber, offset in (
            ("TESTER_A", "CHAMBER_1", 0.0),
            ("TESTER_B", "CHAMBER_2", 0.005),
        ):
            for repeat_index in range(3):
                rows.append(
                    _row(
                        role="tester_bridge",
                        retention=float(
                            98.0 - 0.01 * bridge_index + offset + rng.normal(0.0, 0.001)
                        ),
                        tester=tester,
                        chamber=chamber,
                        measurement_date=start + timedelta(days=bridge_index),
                        repeat_index=repeat_index,
                        cell_id=f"BRIDGE_CELL_{bridge_index}",
                        landmark=100,
                        bridge_id=f"BRIDGE_{bridge_index}",
                    )
                )
    return pd.DataFrame(rows)


def _stability_coordinates(
    history_slope: float = 0.006,
    reissue_slope: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history_cycles = np.arange(61, 101, dtype=float)
    history_residuals = history_slope * (history_cycles - 60.0)
    future_cycles = np.arange(101, 301, dtype=float)
    previous = 100.0 - 0.01 * future_cycles
    current = previous + reissue_slope * (future_cycles - 100.0)
    return history_cycles, history_residuals, future_cycles, previous, current


def _issuance_request(
    *, tester: str = "TESTER_A", chamber: str = "CHAMBER_1"
) -> dict[str, object]:
    history_cycles, residuals, future_cycles, previous, current = (
        _stability_coordinates()
    )
    history_center = 100.0 - 0.01 * history_cycles
    return {
        "schema_version": ("lifetwin.fastcharge_v8.measurement_stability.request.v1"),
        "issuance_id": "SYNTHETIC_ISSUANCE_001",
        "cell_id": "SYNTHETIC_CELL_001",
        "manufacturing_batch_id": "SYNTHETIC_BATCH_001",
        "tester_id": tester,
        "temperature_chamber_id": chamber,
        "history_cycles": history_cycles.tolist(),
        "history_observed_retention_pct": (history_center + residuals).tolist(),
        "history_previous_v5_center_pct": history_center.tolist(),
        "future_cycles": future_cycles.tolist(),
        "previous_v5_center_pct": previous.tolist(),
        "current_v5_center_pct": current.tolist(),
    }


def test_dry_run_protocol_is_synthetic_and_hash_binds_its_parents() -> None:
    config = _config()

    assert config["status"] == "software_only_synthetic_dry_run_not_a_model_result"
    assert config["data_firewall"]["future_capacity_or_soh_columns_permitted"] is False
    assert config["data_firewall"]["real_cell_data_used"] is False
    assert config["data_firewall"]["same_41_cell_outcomes_used"] is False
    assert (
        config["parent_protocol"]["sha256"]
        == hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest()
    )
    assert (
        config["frozen_v7_rule"]["sha256"]
        == hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()
    )


def test_real_execution_template_preserves_protocol_and_uses_full_draw_count() -> None:
    config = json.loads(EXECUTION_CONFIG_PATH.read_text(encoding="utf-8"))
    dry_run = _config()

    assert config["status"].startswith("preregistered_template_pending")
    assert (
        config["parent_protocol"]["sha256"]
        == hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest()
    )
    assert (
        config["frozen_v7_rule"]["sha256"]
        == hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()
    )
    assert config["measurement_contract"] == dry_run["measurement_contract"]
    assert config["noise_model_selection"] == dry_run["noise_model_selection"]
    assert config["drift_quality_gates"] == dry_run["drift_quality_gates"]
    assert config["stability_gate"]["draw_count"] == 1024
    assert config["stability_gate"]["failed_action"] == (
        "exact_zero_update_to_v5_center"
    )


def test_measurement_contract_rejects_future_outcome_columns() -> None:
    frame = _measurement_frame()
    frame["future_retention_pct"] = 90.0

    with pytest.raises(FastChargeV5PairwiseError, match="future outcomes"):
        v8.validate_measurement_frame(frame, _config())


def test_noise_ledger_is_physical_cell_crossfit_and_row_order_invariant() -> None:
    frame = _measurement_frame()
    scores, ledger, quality = v8.characterize_measurement_noise(frame, _config())
    shuffled = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    scores_2, ledger_2, quality_2 = v8.characterize_measurement_noise(
        shuffled, _config()
    )

    pd.testing.assert_frame_equal(scores, scores_2)
    pd.testing.assert_frame_equal(ledger, ledger_2)
    assert quality == quality_2
    assert quality["measurement_quality_passed"] is True
    assert quality["selected_noise_model_id"] == "zero_mean_gaussian"
    assert quality["physical_cell_count"] == 24
    assert len(ledger) == 2
    assert scores["passed_all_folds"].all()
    assert scores["valid_fold_count"].eq(24).all()


def test_repeat_residual_scale_is_corrected_to_single_measurement_error() -> None:
    repeats = pd.DataFrame(
        {
            "physical_cell_id": ["A", "A", "A"],
            "landmark_cycle": [60, 60, 60],
            "tester_id": ["T", "T", "T"],
            "temperature_chamber_id": ["C", "C", "C"],
            "retention_pct": [99.0, 100.0, 101.0],
        }
    )

    residuals = v8._repeat_residuals(repeats)

    observed_rms = float(
        np.sqrt(np.mean(np.square(residuals["measurement_error_proxy_pp"])))
    )
    assert observed_rms == pytest.approx(1.0)


def test_noise_ledger_rejects_duplicate_tester_chamber_mapping() -> None:
    _, ledger, _ = v8.characterize_measurement_noise(_measurement_frame(), _config())
    duplicate = pd.concat([ledger, ledger.iloc[[0]]], ignore_index=True)

    with pytest.raises(FastChargeV5PairwiseError, match="duplicate"):
        v8.validate_noise_ledger(duplicate)


def test_daily_reference_drift_fails_measurement_quality_without_outcomes() -> None:
    frame = _measurement_frame()
    mask = (
        (frame["record_role"] == "daily_reference")
        & (frame["tester_id"] == "TESTER_A")
        & (frame["measurement_date"] == "2026-02-02")
    )
    frame.loc[mask, "retention_pct"] += 0.2

    _, _, quality = v8.characterize_measurement_noise(frame, _config())

    assert quality["daily_reference"]["passed"] is False
    assert quality["measurement_quality_passed"] is False


def test_each_noise_group_requires_its_own_daily_reference_coverage() -> None:
    frame = _measurement_frame()
    frame = frame.loc[
        ~(
            (frame["record_role"] == "daily_reference")
            & (frame["tester_id"] == "TESTER_B")
        )
    ].reset_index(drop=True)

    _, _, quality = v8.characterize_measurement_noise(frame, _config())

    assert quality["daily_reference"]["all_noise_groups_present"] is False
    assert quality["daily_reference"]["passed"] is False
    assert quality["measurement_quality_passed"] is False


def test_stability_api_cannot_receive_future_outcomes() -> None:
    parameters = inspect.signature(v8.measurement_stability_update).parameters

    assert "future_truth" not in parameters
    assert "future_outcomes" not in parameters
    assert "observed_future_retention_pct" not in parameters


def test_stability_request_accepts_only_registered_outcome_free_support() -> None:
    request = _issuance_request()

    validated = v8.validate_stability_request(request, _candidate())

    assert validated.cell_id == "SYNTHETIC_CELL_001"
    assert np.array_equal(validated.history_cycles, np.arange(61, 101))
    assert np.array_equal(validated.future_cycles, np.arange(101, 301))

    leaked = dict(request)
    leaked["future_truth"] = [90.0] * 200
    with pytest.raises(FastChargeV5PairwiseError, match="schema mismatch"):
        v8.validate_stability_request(leaked, _candidate())

    wrong_support = dict(request)
    wrong_support["history_cycles"] = list(range(60, 100))
    with pytest.raises(FastChargeV5PairwiseError, match="history support"):
        v8.validate_stability_request(wrong_support, _candidate())


def test_low_noise_stability_gate_is_deterministic_and_nonzero() -> None:
    coordinates = _stability_coordinates()
    model = v8.MeasurementNoiseModel(
        model_id="low_noise",
        distribution="gaussian",
        scale_pp=0.002,
        tester_id="TESTER_A",
        temperature_chamber_id="CHAMBER_1",
    )
    first, first_status = v8.measurement_stability_update(
        *coordinates,
        _candidate(),
        _config()["stability_gate"],
        model,
        protocol_sha256="a" * 64,
        cell_id="CELL_A",
        measurement_quality_passed=True,
    )
    second, second_status = v8.measurement_stability_update(
        *coordinates,
        _candidate(),
        _config()["stability_gate"],
        model,
        protocol_sha256="a" * 64,
        cell_id="CELL_A",
        measurement_quality_passed=True,
    )

    assert np.array_equal(first, second)
    assert first_status == second_status
    assert first_status["quality_activated"] is True
    assert first_status["measurement_resampled_activation_probability"] >= 0.95
    assert np.any(first != 0.0)


def test_unstable_noise_and_missing_mapping_fall_back_exactly_to_zero() -> None:
    coordinates = _stability_coordinates(history_slope=0.0004, reissue_slope=0.0)
    noisy = v8.MeasurementNoiseModel(
        model_id="high_noise",
        distribution="gaussian",
        scale_pp=0.05,
        tester_id="TESTER_A",
        temperature_chamber_id="CHAMBER_1",
    )
    correction, status = v8.measurement_stability_update(
        *coordinates,
        _candidate(),
        _config()["stability_gate"],
        noisy,
        protocol_sha256="b" * 64,
        cell_id="CELL_B",
        measurement_quality_passed=True,
    )
    missing, missing_status = v8.measurement_stability_update(
        *coordinates,
        _candidate(),
        _config()["stability_gate"],
        None,
        protocol_sha256="b" * 64,
        cell_id="CELL_C",
        measurement_quality_passed=True,
    )

    assert status["unperturbed_v7_activated"] is True
    assert status["quality_activated"] is False
    assert np.all(correction == 0.0)
    assert missing_status["quality_activated"] is False
    assert "noise_mapping_missing" in missing_status["reasons"]
    assert np.all(missing == 0.0)


def test_cohort_readiness_requires_full_size_batch_and_activation_support() -> None:
    protocol = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
    issuances: list[dict[str, object]] = []
    for index in range(60):
        active = index < 6
        batch = f"BATCH_{index % 3}"
        issuances.append(
            {
                "schema_version": (
                    "lifetwin.fastcharge_v8.measurement_stability.issuance_result.v1"
                ),
                "issuance_id": f"ISSUE_{index:03d}",
                "cell_id": f"CELL_{index:03d}",
                "manufacturing_batch_id": batch,
                "config_sha256": "a" * 64,
                "candidate_sha256": "b" * 64,
                "measurement_quality_decision_sha256": "c" * 64,
                "noise_ledger_sha256": "d" * 64,
                "forecast_correction_sha256": "e" * 64,
                "measurement_quality_passed": True,
                "stability": {"quality_activated": active},
                "decision": (
                    "v8_stable_correction_issued"
                    if active
                    else "exact_v5_fallback_issued"
                ),
                "exact_v5_fallback": not active,
                "future_outcomes_read": False,
                "model_accuracy_evidence_created": False,
                "v5_champion_changed": False,
            }
        )

    readiness = v8.cohort_readiness_decision(issuances, protocol)

    assert readiness["stage_c_outcome_opening_authorized"] is True
    assert readiness["physical_cell_count"] == 60
    assert readiness["manufacturing_batch_count"] == 3
    assert readiness["stable_activation_count"] == 6
    assert readiness["stable_activation_coverage"] == pytest.approx(0.1)
    assert readiness["activated_batch_count"] == 3


def test_stage_a_to_stage_b_issuance_is_hash_bound_and_outcome_free(
    tmp_path: Path,
) -> None:
    measurement_path = tmp_path / "repeatability.csv"
    _measurement_frame().to_csv(measurement_path, index=False)
    quality_output = tmp_path / "quality"
    prepare_v8.run(
        argparse.Namespace(
            config=str(EXECUTION_CONFIG_PATH),
            measurements=str(measurement_path),
            output_directory=str(quality_output),
        )
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(_issuance_request(), allow_nan=False), encoding="utf-8"
    )
    issuance_root = tmp_path / "issuances"
    issuance_output = issuance_root / "cell_001"

    issue_v8.run(
        argparse.Namespace(
            config=str(EXECUTION_CONFIG_PATH),
            candidate=str(CANDIDATE_PATH),
            request=str(request_path),
            measurement_quality_decision=str(quality_output / "decision.json"),
            noise_ledger=str(quality_output / "noise_ledger.csv"),
            output_directory=str(issuance_output),
        )
    )

    decision = json.loads(
        (issuance_output / "decision.json").read_text(encoding="utf-8")
    )
    commitment = json.loads(
        (issuance_output / "prediction_commitment.json").read_text(encoding="utf-8")
    )
    forecast = pd.read_csv(issuance_output / "forecast_correction.csv")
    assert decision["decision"] == "v8_stable_correction_issued"
    assert decision["future_outcomes_read"] is False
    assert decision["model_accuracy_evidence_created"] is False
    assert decision["v5_champion_changed"] is False
    assert commitment["stage_c_outcome_opening_authorized"] is False
    assert len(forecast) == 200
    assert forecast["v8_effective_correction_pp"].abs().max() > 0.0

    cohort_output = tmp_path / "cohort"
    compile_v8.run(
        argparse.Namespace(
            config=str(EXECUTION_CONFIG_PATH),
            protocol=str(PARENT_PATH),
            candidate=str(CANDIDATE_PATH),
            issuance_root=str(issuance_root),
            output_directory=str(cohort_output),
        )
    )
    cohort = json.loads(
        (cohort_output / "cohort_decision.json").read_text(encoding="utf-8")
    )
    assert cohort["decision"] == "stage_c_opening_blocked_retain_v5"
    assert cohort["future_outcomes_read"] is False
    assert cohort["readiness"]["physical_cell_count"] == 1
    assert "physical_cell_count_below_threshold" in cohort["readiness"]["reasons"]


def test_published_v8_dry_run_is_software_evidence_only() -> None:
    result = json.loads(PUBLISHED_DECISION_PATH.read_text(encoding="utf-8"))

    assert (
        result["config_sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    )
    assert result["decision"] == ("software_dry_run_passed_without_real_model_evidence")
    assert result["measurement_quality"]["measurement_quality_passed"] is True
    assert result["stable_path"]["quality_activated"] is True
    assert result["missing_mapping_fallback"]["exact_zero_correction"] is True
    assert result["v5_champion_changed"] is False
    assert result["real_model_evidence_created"] is False
    assert result["measurement_fixture"]["future_outcome_columns_present"] is False

    implementation = result["implementation"]
    assert (
        implementation["module_sha256"]
        == hashlib.sha256(
            (ROOT / implementation["module_path"]).read_bytes()
        ).hexdigest()
    )
    assert (
        implementation["runner_sha256"]
        == hashlib.sha256(
            (ROOT / implementation["runner_path"]).read_bytes()
        ).hexdigest()
    )
