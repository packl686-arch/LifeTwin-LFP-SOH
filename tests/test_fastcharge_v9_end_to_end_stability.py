from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import fastcharge_v9_end_to_end_stability as v9
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError
from scripts import run_fastcharge_v9_end_to_end_synthetic_dry_run as run_v9
from scripts import evaluate_fastcharge_v9_replicate_ledger as evaluate_v9


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/experiments/v9_end_to_end_correlated_stability_synthetic_dry_run.json"
)
EXECUTION_PATH = (
    ROOT
    / "configs/experiments/v9_end_to_end_correlated_stability_execution.template.json"
)
PROTOCOL_PATH = (
    ROOT
    / "configs/experiments/v9_end_to_end_correlated_stability_blind_protocol.template.json"
)
CANDIDATE_PATH = (
    ROOT / "configs/experiments/v7_p100_reissue_innovation_blind_candidate.json"
)
PUBLISHED_DECISION_PATH = ROOT / "showcase/evidence_v9_dry_run/decision.json"


def _config(draw_count: int = 4) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["stability_gate"]["draw_count"] = draw_count
    return config


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))


def _ledger(draw_count: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    p60_cycles = np.arange(61, 301, dtype=int)
    history_cycles = np.arange(61, 101, dtype=int)
    p100_cycles = np.arange(101, 301, dtype=int)
    references = [f"REFERENCE_{index:02d}" for index in range(12)]
    for draw_index in range(draw_count + 1):
        perturbation = 0.0 if draw_index == 0 else 0.0002 * draw_index
        previous = 100.0 - 0.01 * p60_cycles + perturbation
        observed = (
            100.0
            - 0.01 * history_cycles
            + 0.006 * (history_cycles - 60.0)
            + perturbation
        )
        current = (
            100.0 - 0.01 * p100_cycles + 0.002 * (p100_cycles - 100.0) + perturbation
        )
        for role, cycles, values, role_references in (
            ("p100_observed_prefix", history_cycles, observed, []),
            ("p60_v5_center", p60_cycles, previous, references),
            ("p100_v5_center", p100_cycles, current, references),
        ):
            encoded = json.dumps(role_references, separators=(",", ":"))
            source_hash = hashlib.sha256(
                f"{draw_index}|{role}".encode("utf-8")
            ).hexdigest()
            for cycle, value in zip(cycles, values, strict=True):
                rows.append(
                    {
                        "schema_version": v9.LEDGER_SCHEMA_VERSION,
                        "issuance_id": "ISSUE_001",
                        "cell_id": "CELL_001",
                        "manufacturing_batch_id": "BATCH_001",
                        "draw_index": draw_index,
                        "trajectory_role": role,
                        "cycle_index": int(cycle),
                        "retention_pct": float(value),
                        "reference_cell_ids_json": encoded,
                        "source_sha256": source_hash,
                    }
                )
    return pd.DataFrame(rows, columns=v9.LEDGER_COLUMNS)


def test_protocol_binds_v8_v7_v5_and_execution_preserves_gates() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))

    assert (
        config["parent_protocol"]["sha256"]
        == hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    )
    assert execution["parent_protocol"] == config["parent_protocol"]
    assert execution["model_contract"] == config["model_contract"]
    synthetic_gate = dict(config["stability_gate"])
    real_gate = dict(execution["stability_gate"])
    assert synthetic_gate.pop("draw_count") == 24
    assert real_gate.pop("draw_count") == 1024
    assert synthetic_gate == real_gate
    assert (
        protocol["data_firewall"][
            "target_cycles_101_through_300_permitted_before_commitment"
        ]
        is False
    )


def test_synthetic_target_is_created_only_through_cycle_100() -> None:
    training, target = run_v9.build_synthetic_inputs(_config())

    assert training["cycle_index"].max() == 300
    assert target["cycle_index"].tolist() == list(range(1, 101))
    assert target["cell_id"].nunique() == 1


def test_distributed_real_execution_template_fails_closed() -> None:
    with pytest.raises(FastChargeV5PairwiseError, match="template cannot be executed"):
        evaluate_v9.run(
            argparse.Namespace(
                config=str(EXECUTION_PATH),
                replicate_ledger=str(ROOT / "artifacts/not_read.csv"),
                output_directory=str(ROOT / "artifacts/not_written"),
            )
        )


def test_qualification_api_cannot_receive_future_truth() -> None:
    parameters = inspect.signature(v9.evaluate_end_to_end_stability).parameters

    assert "future_truth" not in parameters
    assert "future_outcomes" not in parameters
    assert "observed_future_retention_pct" not in parameters


def test_ledger_rejects_extra_future_outcome_column_and_missing_draw() -> None:
    leaked = _ledger()
    leaked["future_truth"] = 90.0
    with pytest.raises(FastChargeV5PairwiseError, match="schema mismatch"):
        v9.validate_replicate_ledger(leaked, _config())

    missing = _ledger().loc[lambda frame: frame["draw_index"] != 4]
    with pytest.raises(FastChargeV5PairwiseError, match="every registered draw"):
        v9.validate_replicate_ledger(missing, _config())


def test_stable_end_to_end_ledger_qualifies_deterministically() -> None:
    ledger = _ledger()
    correction, status, metrics = v9.evaluate_end_to_end_stability(
        ledger,
        _candidate(),
        _config(),
        protocol_sha256="a" * 64,
    )
    shuffled = ledger.sample(frac=1.0, random_state=19).reset_index(drop=True)
    correction_2, status_2, metrics_2 = v9.evaluate_end_to_end_stability(
        shuffled,
        _candidate(),
        _config(),
        protocol_sha256="a" * 64,
    )

    assert status["quality_activated"] is True
    assert status["refit_activation_probability"] == 1.0
    assert status["refit_correction_sign_probability"] == 1.0
    assert np.any(correction != 0.0)
    assert np.array_equal(correction, correction_2)
    assert status == status_2
    pd.testing.assert_frame_equal(metrics, metrics_2)


def test_unstable_refit_and_neighbour_selection_falls_back_exactly() -> None:
    ledger = _ledger()
    mask = (ledger["draw_index"] > 0) & ledger["trajectory_role"].isin(
        ("p60_v5_center", "p100_v5_center")
    )
    ledger.loc[mask, "reference_cell_ids_json"] = json.dumps(
        [f"UNSTABLE_REFERENCE_{index:02d}" for index in range(12)],
        separators=(",", ":"),
    )
    ledger.loc[mask, "retention_pct"] += 0.2

    correction, status, _ = v9.evaluate_end_to_end_stability(
        ledger,
        _candidate(),
        _config(),
        protocol_sha256="b" * 64,
    )

    assert status["quality_activated"] is False
    assert status["failed_action"] == (
        "exact_zero_update_to_unperturbed_p100_v5_center"
    )
    assert "p05_p60_reference_jaccard_below_threshold" in status["reasons"]
    assert np.all(correction == 0.0)


def test_published_v9_result_is_only_synthetic_software_evidence() -> None:
    result = json.loads(PUBLISHED_DECISION_PATH.read_text(encoding="utf-8"))

    assert result["decision"] == (
        "synthetic_end_to_end_software_dry_run_passed_without_model_evidence"
    )
    assert result["future_outcomes_read"] is False
    assert result["model_accuracy_evidence_created"] is False
    assert result["real_model_evidence_created"] is False
    assert result["v5_champion_changed"] is False
    assert result["synthetic_fixture"]["target_future_rows_generated"] == 0
    assert result["synthetic_fixture"]["historical_v5_refit_each_draw"] is True
    assert result["stress_negative_control"]["exact_zero_correction"] is True
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
