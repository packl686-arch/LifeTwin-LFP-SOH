from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v015_runner as runner,
)
from lifetwin.experiments.calendar_long_horizon_v015_firewall import (
    AttemptProgress,
    FormalAttemptIdentity,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FROZEN_PROTOCOL_ID,
    canonical_json_bytes,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    LabelFreePipelineResult,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256,
)


def _identity() -> FormalAttemptIdentity:
    return FormalAttemptIdentity(
        attempt_id="fixture-attempt",
        git_commit="a" * 40,
        config_byte_sha256=FROZEN_CONFIG_BYTE_SHA256,
    )


def _paths(tmp_path: Path) -> runner.FormalRunPaths:
    return runner.FormalRunPaths.resolve(
        repo_root=tmp_path / "repo",
        label_free_root=tmp_path / "label",
        sealed_truth_root=tmp_path / "sealed",
        score_root=tmp_path / "scores",
    )


def test_prediction_stage_has_no_sealed_truth_capability() -> None:
    parameters = tuple(inspect.signature(runner.run_formal_prediction_stage).parameters)
    assert parameters == ("label_free_root", "attempt_id", "repo_root")
    assert not any("truth" in name for name in parameters)


def test_roots_must_be_disjoint_and_fresh(tmp_path: Path) -> None:
    with pytest.raises(runner.V015RunnerError, match="disjoint"):
        runner.FormalRunPaths.resolve(
            repo_root=tmp_path,
            label_free_root=tmp_path / "run",
            sealed_truth_root=tmp_path / "run" / "sealed",
            score_root=tmp_path / "scores",
        )

    label = tmp_path / "label"
    label.mkdir()
    (label / "unexpected.txt").write_text("hidden", encoding="utf-8")
    with pytest.raises(runner.V015RunnerError, match="completely empty"):
        runner._require_fresh_root(label, context="label-free")


def test_atomic_exclusive_create_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    runner._exclusive_create_bytes(target, b"first")
    with pytest.raises(runner.V015RunnerError, match="already exists"):
        runner._exclusive_create_bytes(target, b"second")
    assert target.read_bytes() == b"first"
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_freeze_record_distinguishes_protocol_and_implementation_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "freeze.json"
    record.write_text(
        json.dumps(
            {
                "freeze_commit": ("b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91"),
                "implementation_source_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_FREEZE_RECORD", record)
    assert runner._freeze_record_commits() == (
        "b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91",
        "a" * 40,
    )


def _pipeline_for_ids(
    identifiers: tuple[str, ...],
    content_order: tuple[int, ...],
) -> LabelFreePipelineResult:
    content = pd.DataFrame(
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "risk_development",
                "cluster_id": identifier,
                "random_policy_content_sha256": f"{index + 10:064x}",
                "arm_a_content_sha256": f"{index + 10:064x}",
                "arm_b_content_sha256": f"{index + 20:064x}",
                "placebo_content_sha256": f"{index + 30:064x}",
            }
            for identifier, index in zip(identifiers, content_order, strict=True)
        ]
    )
    empty = pd.DataFrame()
    return LabelFreePipelineResult(
        prediction_bundle=empty,
        feature_bundle=empty,
        primary_risk_bundle=empty,
        decision_bundle=empty,
        predictor_content_bundle=content,
    )


def _truth_for_ids(
    identifiers: tuple[str, ...],
    content_order: tuple[int, ...],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for identifier, index in zip(identifiers, content_order, strict=True):
        for day_index, day in enumerate(runner.FORECAST_DAYS):
            records.append(
                {
                    "protocol_id": FROZEN_PROTOCOL_ID,
                    "partition": "risk_development",
                    "cluster_id": identifier,
                    "truth_family": "fixture",
                    "truth_parameters_json": "{}",
                    "gamma": 0.1 + index * 0.001,
                    "forecast_day": day,
                    "latent_retention_pct": 95.0 - index - day_index,
                    "noisy_retention_pct": 95.1 - index - day_index,
                }
            )
    return pd.DataFrame(records)


def test_training_order_is_invariant_to_opaque_id_relabeling() -> None:
    original_ids = ("opaque-z", "opaque-a", "opaque-m")
    relabeled_ids = ("renamed-1", "renamed-9", "renamed-0")
    content_order = (2, 0, 1)
    original_pipeline = _pipeline_for_ids(original_ids, content_order)
    relabeled_pipeline = _pipeline_for_ids(relabeled_ids, content_order)
    original_truth = _truth_for_ids(original_ids, content_order)
    relabeled_truth = _truth_for_ids(relabeled_ids, content_order)

    original = runner._ordered_training_cluster_ids(
        original_pipeline,
        original_truth,
        partition="risk_development",
        include_operating_content=True,
    )
    relabeled = runner._ordered_training_cluster_ids(
        relabeled_pipeline,
        relabeled_truth,
        partition="risk_development",
        include_operating_content=True,
    )
    original_by_id = original_pipeline.predictor_content_bundle.set_index("cluster_id")
    relabeled_by_id = relabeled_pipeline.predictor_content_bundle.set_index(
        "cluster_id"
    )
    assert [original_by_id.loc[item, "arm_a_content_sha256"] for item in original] == [
        relabeled_by_id.loc[item, "arm_a_content_sha256"] for item in relabeled
    ]


def test_fit_commitment_rejects_post_reveal_byte_mutation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.label_free_root.mkdir()
    entries = []
    for index, filename in enumerate(runner._FIT_COMMITMENT_FILENAMES):
        raw = f"fixture-{index}\n".encode()
        path = paths.label_free_root / filename
        path.write_bytes(raw)
        entries.append(
            {
                "path": filename,
                "row_count": 1,
                "byte_count": len(raw),
                "byte_sha256": runner._sha256_bytes(raw),
            }
        )
    payload = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "git_commit": _identity().git_commit,
        "worker_count": 6,
        "files": entries,
        "created_utc": "2026-07-23T00:00:00Z",
    }
    commitment = paths.label_free_root / "fit_commitment.json"
    commitment.write_bytes(canonical_json_bytes(payload))
    observed = runner._verify_fit_commitment(
        label_free_root=paths.label_free_root,
        contract=load_artifact_contract(),
        identity=_identity(),
    )
    assert observed == runner._sha256_path(commitment)

    (paths.label_free_root / "operating_pack.csv").write_bytes(b"changed\n")
    with pytest.raises(runner.V015RunnerError, match="changed"):
        runner._verify_fit_commitment(
            label_free_root=paths.label_free_root,
            contract=load_artifact_contract(),
            identity=_identity(),
        )


def test_fit_commitment_uses_frozen_large_table_cardinalities(
    tmp_path: Path,
) -> None:
    contract = load_artifact_contract()
    diagnostics = tmp_path / "member_fit_diagnostics.csv"
    forecasts = tmp_path / "member_forecast_bundle.csv"
    diagnostics.write_bytes(b"fixture\n")
    forecasts.write_bytes(b"fixture\n")

    diagnostic_entry = runner._fit_commitment_entry(diagnostics, contract=contract)
    forecast_entry = runner._fit_commitment_entry(forecasts, contract=contract)

    assert diagnostic_entry["row_count"] == 511_700
    assert forecast_entry["row_count"] == 4_093_600


def test_model_state_commitment_chains_every_state_file(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.label_free_root.mkdir()
    for index, filename in enumerate(runner._MODEL_STATE_COMMITMENT_FILENAMES):
        (paths.label_free_root / filename).write_bytes(f"state-{index}\n".encode())
    digest = runner._create_model_state_commitment(paths=paths, identity=_identity())
    assert (
        runner._verify_model_state_commitment(
            label_free_root=paths.label_free_root,
            identity=_identity(),
            expected_byte_sha256=digest,
        )
        == digest
    )

    (paths.label_free_root / "model_state.json").write_bytes(b"tampered\n")
    with pytest.raises(runner.V015RunnerError, match="changed"):
        runner._verify_model_state_commitment(
            label_free_root=paths.label_free_root,
            identity=_identity(),
            expected_byte_sha256=digest,
        )


@pytest.mark.parametrize(
    ("phase", "field"),
    (
        ("label_free_fit_committed", "fit_commitment_byte_sha256"),
        (
            "center_state_committed",
            "center_state_checkpoint_byte_sha256",
        ),
        ("risk_state_committed", "risk_state_checkpoint_byte_sha256"),
        (
            "model_state_committed",
            "model_state_commitment_byte_sha256",
        ),
    ),
)
def test_runner_consumes_all_machine_bound_phase_commitments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    field: str,
) -> None:
    contract = load_artifact_contract()
    artifact = tmp_path / f"{phase}.json"
    artifact.write_bytes(b"committed\n")
    digest = runner._sha256_path(artifact)
    values = {
        "fit_commitment_byte_sha256": None,
        "center_state_checkpoint_byte_sha256": None,
        "risk_state_checkpoint_byte_sha256": None,
        "model_state_commitment_byte_sha256": None,
    }
    values[field] = digest
    progress = AttemptProgress(
        identity=_identity(),
        completed_phase=phase,
        pending_phase=None,
        truth_commitments_byte_sha256="c" * 64,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(),
        terminal_failed=False,
        **values,
    )
    monkeypatch.setattr(
        runner,
        "validate_formal_exposure_log",
        lambda *_: {_identity().attempt_id: progress},
    )
    assert (
        runner._verify_ledger_artifact(
            ledger_path=tmp_path / "ledger.jsonl",
            contract=contract,
            identity=_identity(),
            phase=phase,
            artifact_path=artifact,
        )
        == digest
    )
    artifact.write_bytes(b"tampered\n")
    with pytest.raises(Exception, match="differs"):
        runner._verify_ledger_artifact(
            ledger_path=tmp_path / "ledger.jsonl",
            contract=contract,
            identity=_identity(),
            phase=phase,
            artifact_path=artifact,
        )
