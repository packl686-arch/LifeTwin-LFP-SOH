from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_analysis as v015_analysis
from lifetwin.experiments import calendar_long_horizon_v016_analysis as v021_analysis
from lifetwin.experiments import calendar_long_horizon_v016_io as v021_io
from lifetwin.experiments import calendar_long_horizon_v016_scoring as scoring
from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_PROTOCOL_ID,
)


def _stochastic_fixture() -> pd.DataFrame:
    family = v021_analysis.TEST_FAMILIES[0]
    rows: list[dict[str, object]] = []
    for index in range(12):
        rows.append(
            {
                "truth_family": family,
                "canonical_prefix_content_sha256": f"{index + 1:064x}",
                "catastrophic": index % 3 == 0,
                "hard_eligible_visible_stress": True,
                "risk_prefix_only": float(index % 4),
                "risk_visible_stress": float((index * 3) % 5),
                "risk_placebo_8": float((index * 5) % 7),
                "risk_hash_prefix_only": f"{1000 + index:064x}",
                "risk_hash_visible_stress": f"{2000 + index:064x}",
                "risk_hash_placebo_8": f"{3000 + index:064x}",
            }
        )
    return pd.DataFrame(rows)


def _terminal_capabilities() -> tuple[object, object]:
    model = SimpleNamespace(
        attempt_id="v021-score-fixture",
        validated_model_state=SimpleNamespace(model_state_byte_sha256="1" * 64),
        model_state_commitment_artifact_byte_sha256="2" * 64,
    )
    prediction = SimpleNamespace(prediction_commitment_byte_sha256="3" * 64)
    return model, prediction


def _sealed_prediction_evidence() -> v021_io.V021PredictionCommitmentEvidence:
    entries = [
        {
            "path": filename,
            "row_count": index + 1,
            "byte_count": index + 10,
            "byte_sha256": hashlib.sha256(filename.encode("ascii")).hexdigest(),
        }
        for index, filename in enumerate(v021_io._COMMITMENT_FILE_REGISTRY)
    ]
    return v021_io._issue_prediction_commitment_evidence(
        attempt_id="v021-score-fixture",
        byte_sha256="1" * 64,
        artifact_set_sha256="2" * 64,
        actual_analysis_hash_ledger_commitment_byte_sha256=next(
            str(entry["byte_sha256"])
            for entry in entries
            if entry["path"] == "actual_analysis_hash_ledger_commitment.json"
        ),
        file_entries=entries,
        ledger_committed=True,
    )


def test_formal_surface_and_registry_are_exactly_locked() -> None:
    assert scoring.REQUIRED_SCORE_ARTIFACTS == (
        "point_scores.csv",
        "trajectory_scores.csv",
        "family_metrics.csv",
        "matched_pair_scores.csv",
        "bootstrap_replicates.csv",
        "random_ranking_metrics.csv",
        "stress_permutation_metrics.csv",
        "negative_control_metrics.json",
        "score_report.json",
        "run_manifest.json",
    )
    assert scoring._FORMAL_ANALYSIS_COUNTS == scoring._AnalysisCounts(
        random_rankings=10_000,
        bootstrap_resamples=5_000,
        stress_permutations=10_000,
    )
    parameters = inspect.signature(scoring.score_committed_artifacts).parameters
    assert tuple(parameters) == (
        "prediction_frames",
        "truth_frames",
        "model_state_envelope",
        "prediction_commitment_envelope",
    )
    assert {
        "counts",
        "formal",
        "decoded_model_state",
        "model_state_bytes",
    }.isdisjoint(parameters)


def test_formal_surface_rejects_bare_or_forged_model_state_capabilities() -> None:
    with pytest.raises(scoring.V021ScoringError, match="IO-issued"):
        scoring.score_committed_artifacts(
            prediction_frames={},
            truth_frames={},
            model_state_envelope=object(),  # type: ignore[arg-type]
            prediction_commitment_envelope=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="issued only"):
        scoring.V021PredictionCommitmentEnvelope(
            _issuer_key=object(),
            protocol_id=V021_PROTOCOL_ID,
            config_sha256="0" * 64,
            attempt_id="v021-forged",
            evidence=object(),  # type: ignore[arg-type]
            prediction_commitment_byte_sha256="1" * 64,
            artifact_set_sha256="2" * 64,
            artifact_metadata=(),
            provenance_sha256="3" * 64,
        )
    issuer_source = inspect.getsource(
        scoring._issue_prediction_commitment_envelope_v021
    )
    assert "V021PredictionCommitmentEvidence" in issuer_source
    assert "ledger_committed" in issuer_source
    assert "_prediction_metadata_from_evidence" in issuer_source
    assert "_canonical_prediction_metadata" in issuer_source
    assert (
        "prediction_commitment_byte_sha256"
        not in inspect.signature(
            scoring._issue_prediction_commitment_envelope_v021
        ).parameters
    )


def test_small_stochastic_hook_uses_v21_roots_without_v15_global_mutation() -> None:
    trajectories = _stochastic_fixture()
    before = (
        v015_analysis.RANDOM_ROOT,
        v015_analysis.BOOTSTRAP_ROOT,
        v015_analysis.STRESS_PERMUTATION_ROOT,
    )
    counts = scoring._AnalysisCounts(
        random_rankings=7,
        bootstrap_resamples=5,
        stress_permutations=3,
    )
    random, bootstrap = scoring._run_stochastic_fixture_analyses_v021(
        trajectories,
        issue_count=4,
        counts=counts,
    )
    predecessor_random = v015_analysis.deterministic_random_rankings(
        trajectories,
        issue_count=4,
        rankings=7,
    )
    predecessor_bootstrap = v015_analysis.bootstrap_risk_reductions(
        trajectories,
        protocol_id=V021_PROTOCOL_ID,
        issue_count=4,
        resamples=5,
        families=(v021_analysis.TEST_FAMILIES[0],),
    )

    assert random["ranking_index"].tolist() == list(range(7))
    assert bootstrap["replicate_index"].tolist() == list(range(5))
    assert not random.equals(predecessor_random)
    assert not bootstrap.equals(predecessor_bootstrap)
    assert before == (
        v015_analysis.RANDOM_ROOT,
        v015_analysis.BOOTSTRAP_ROOT,
        v015_analysis.STRESS_PERMUTATION_ROOT,
    )


def test_prediction_evidence_exposes_exact_committed_prediction_metadata() -> None:
    evidence = _sealed_prediction_evidence()
    metadata = scoring._prediction_metadata_from_evidence(evidence)

    assert tuple(item[0] for item in metadata) == scoring._PREDICTION_FRAME_FILENAMES
    evidence_by_path = {entry["path"]: entry for entry in evidence.file_entries}
    assert metadata == tuple(
        (
            filename,
            evidence_by_path[filename]["row_count"],
            evidence_by_path[filename]["byte_count"],
            evidence_by_path[filename]["byte_sha256"],
        )
        for filename in scoring._PREDICTION_FRAME_FILENAMES
    )
    model = SimpleNamespace(
        model_state_commitment_artifact_byte_sha256=evidence_by_path[
            "model_state_commitment.json"
        ]["byte_sha256"],
        validated_model_state=SimpleNamespace(
            model_state_byte_sha256=evidence_by_path["model_state.json"]["byte_sha256"],
            training_provenance=SimpleNamespace(
                commitment_byte_sha256={
                    "actual_analysis_hash_ledger": (
                        evidence.actual_analysis_hash_ledger_commitment_byte_sha256
                    )
                }
            ),
        ),
    )
    scoring._require_evidence_model_binding(evidence, model)
    model.model_state_commitment_artifact_byte_sha256 = "0" * 64
    with pytest.raises(scoring.V021ScoringError, match="committed model chain"):
        scoring._require_evidence_model_binding(evidence, model)


def test_v21_isotonic_diagnostics_use_only_committed_eligible_population() -> None:
    trajectories = pd.DataFrame(
        {
            "protocol_id": [V021_PROTOCOL_ID] * 3,
            "partition": ["calibration"] * 3,
            "cluster_id": ["c_0", "c_1", "c_2"],
            "catastrophic": [False, True, True],
        }
    )
    risks = pd.DataFrame(
        [
            {
                "protocol_id": V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                "score_id": score_id,
                "calibrated_catastrophic_probability": probability,
            }
            for score_id in ("prefix_only", "visible_stress")
            for cluster_id, probability in (
                ("c_0", 0.2),
                ("c_1", 0.8),
                ("c_2", np.nan),
            )
        ]
    )
    isotonic = SimpleNamespace(x_thresholds=(0.0, 1.0))
    decoded = SimpleNamespace(
        training_state=SimpleNamespace(
            calibration=SimpleNamespace(
                prefix_only_isotonic=isotonic,
                visible_stress_isotonic=isotonic,
            )
        )
    )
    records = scoring._isotonic_calibration_diagnostics_v021(
        trajectories,
        risks,
        decoded,
        source_calibration_count=3,
        risk_isotonic_eligible_count=2,
    )

    assert len(records) == 2
    assert all(record["available"] is True for record in records)
    assert all(record["n"] == 2 for record in records)
    assert all(record["positive_count"] == 1 for record in records)


def test_terminal_score_registry_contains_all_ten_v21_bound_artifacts() -> None:
    model, prediction = _terminal_capabilities()
    result = scoring._terminal_result(
        "fixture structural failure",
        status_kind="void",
        model=model,  # type: ignore[arg-type]
        prediction=prediction,  # type: ignore[arg-type]
        view=load_v021_contract_view(),
    )
    payloads = scoring.required_score_artifact_payloads_v021(result)

    assert tuple(payloads) == scoring.REQUIRED_SCORE_ARTIFACTS
    for filename in scoring.REQUIRED_SCORE_CSV_ARTIFACTS:
        assert payloads[filename].count(b"\n") == 1
    report = json.loads(payloads["score_report.json"])
    controls = json.loads(payloads["negative_control_metrics.json"])
    manifest = json.loads(payloads["run_manifest.json"])
    assert report["protocol_id"] == V021_PROTOCOL_ID
    assert controls["protocol_id"] == V021_PROTOCOL_ID
    assert manifest["protocol_id"] == V021_PROTOCOL_ID
    assert manifest["analysis_counts"] == {
        "random_rankings": 10_000,
        "bootstrap_resamples": 5_000,
        "stress_permutations": 10_000,
    }
    assert [entry["path"] for entry in manifest["scored_artifacts"]] == list(
        scoring.REQUIRED_SCORE_ARTIFACTS[:-1]
    )

    tampered = replace(
        result,
        score_report={**result.score_report, "protocol_id": "wrong"},
    )
    with pytest.raises(scoring.V021ScoringError, match="protocol_id"):
        scoring.required_score_artifact_payloads_v021(tampered)

    broken_manifest = {
        **result.run_manifest,
        "analysis_counts": {
            "random_rankings": 7,
            "bootstrap_resamples": 5,
            "stress_permutations": 3,
        },
    }
    with pytest.raises(scoring.V021ScoringError, match="identity or registry"):
        scoring.required_score_artifact_payloads_v021(
            replace(result, run_manifest=broken_manifest)
        )


def test_recomputation_and_scoring_source_use_v21_adapters_not_v15_mutation() -> None:
    recompute_source = inspect.getsource(
        scoring.validate_and_recompute_committed_predictions_v021
    )
    score_source = inspect.getsource(scoring._score_committed_artifacts_v021)
    assert "recompute_label_free_pipeline_v021" in recompute_source
    assert "deterministic_random_rankings" in score_source
    assert "bootstrap_risk_reductions" in score_source
    assert "stress_permutation_metrics" in score_source

    tree = ast.parse(inspect.getsource(scoring))
    assignments_to_v15 = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "_v015"
    ]
    assert assignments_to_v15 == []
