from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import uuid

import pandas as pd
import pytest

from scripts import verify_v014_synthetic_evidence as verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = (
    PROJECT_ROOT
    / "showcase"
    / "evidence_v014"
    / "synthetic_long_horizon_identifiability_v1"
)


@pytest.fixture()
def evidence_copy() -> Path:
    destination = PROJECT_ROOT / f".pytest-v014-published-{uuid.uuid4().hex}"
    shutil.copytree(EVIDENCE_DIR, destination)
    try:
        yield destination
    finally:
        shutil.rmtree(destination)


def _published_config() -> dict[str, object]:
    environment = verifier._load_json_strict(EVIDENCE_DIR / "environment.json")
    evidence = verifier._load_json_strict(EVIDENCE_DIR / "evidence_manifest.json")
    return verifier._verify_execution_sources(PROJECT_ROOT, environment, evidence)[
        "config"
    ]


def test_published_negative_result_passes_integrity_verification() -> None:
    result = verifier.verify(PROJECT_ROOT, EVIDENCE_DIR)

    assert result["verification_status"] == "passed"
    assert result["result_status"] == "failure"
    assert result["negative_result_accepted"] is True
    assert result["primary_gates"] == {
        "catastrophic_risk_reduction_at_50_percent_issuance": False,
        "matched_prefix_both_members_rejected": False,
        "issued_trajectory_iae_noninferiority": True,
    }
    assert result["safety_gates"] == {
        "minimum_counts_and_finite_forecasts": True,
        "audit_directional_consistency": True,
        "random_rankings_fully_defined": True,
        "bootstrap_fully_defined": True,
    }
    assert result["headline"] == pytest.approx(
        {
            "test_random_ranking_risk_reduction_fraction": 0.21650713075238526,
            "test_analytic_risk_reduction_fraction": 0.21658204334365327,
            "random_mean_catastrophic_rate": 0.4135328,
            "bootstrap_one_sided_95pct_lower_bound": 0.1631074238519576,
            "matched_both_rejected_fraction": 0.27,
            "matched_model_failure_member_count": 14,
            "issued_iae_noninferiority_delta_pp": 0.014561718429392911,
            "audit_analytic_risk_reduction_fraction": 0.24781818181818183,
        }
    )
    assert [
        (item["partition"], item["truth_family"]) for item in result["family_reversals"]
    ] == [("test", "late_knee"), ("audit", "late_knee")]
    assert result["verification_scope"] == {
        "candidate_issued_iae": "independently_recomputed_from_trajectory_scores",
        "baseline_issued_iae": (
            "commitment_and_hash_bound_only_not_independently_recomputed_"
            "from_compact_evidence"
        ),
        "baseline_model_metrics_sha256": (
            "f0ae98ba57152633fd868c06ab4e15dc650975c917483f267d09390d61c06743"
        ),
    }
    json.dumps(result, allow_nan=False)


def test_cli_returns_zero_and_distinguishes_failure_from_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_v014_synthetic_evidence.py",
            "--project-root",
            str(PROJECT_ROOT),
            "--evidence-dir",
            str(EVIDENCE_DIR),
        ],
    )

    assert verifier.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification_status"] == "passed"
    assert payload["result_status"] == "failure"


def test_artifact_byte_tampering_is_rejected(evidence_copy: Path) -> None:
    bootstrap = evidence_copy / "bootstrap.csv"
    bootstrap.write_bytes(bootstrap.read_bytes() + b"\n")

    with pytest.raises(
        verifier.EvidenceVerificationError,
        match="bootstrap.csv byte count differs from its manifest",
    ):
        verifier.verify(PROJECT_ROOT, evidence_copy)


def test_default_evidence_path_is_relative_to_supplied_project_root() -> None:
    fake_root = PROJECT_ROOT / f".pytest-v014-published-root-{uuid.uuid4().hex}"
    fake_root.mkdir()
    expected = (fake_root / verifier.DEFAULT_EVIDENCE_RELATIVE).resolve()
    try:
        with pytest.raises(
            verifier.EvidenceVerificationError,
            match=re.escape(f"Published evidence directory is missing: {expected}"),
        ):
            verifier.verify(fake_root)
    finally:
        fake_root.rmdir()


def test_full_manifest_canonical_digest_tampering_is_rejected(
    evidence_copy: Path,
) -> None:
    path = evidence_copy / "full_bundle_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["canonical_entries_sha256"] = "0" * 64
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        verifier.EvidenceVerificationError,
        match="Full-bundle manifest differs from the verifier-pinned digest",
    ):
        verifier.verify(PROJECT_ROOT, evidence_copy)


@pytest.mark.parametrize(
    "raw,pattern",
    [
        (b'{"value":NaN}', "nonfinite value"),
        (b'{"value":1e999}', "nonfinite number"),
        (b'{"value":1,"value":2}', "duplicate key"),
    ],
)
def test_strict_json_rejects_nonfinite_numbers_and_duplicate_keys(
    raw: bytes, pattern: str
) -> None:
    with pytest.raises(verifier.EvidenceVerificationError, match=pattern):
        verifier._loads_json_strict(raw, context="adversarial JSON")


def test_exposure_reordering_is_rejected() -> None:
    exposure = verifier._load_json_strict(EVIDENCE_DIR / "exposure_log.json")
    exposure["events"][2], exposure["events"][3] = (
        exposure["events"][3],
        exposure["events"][2],
    )

    with pytest.raises(
        verifier.EvidenceVerificationError,
        match="exposure event 3 keys changed|Exposure sequence or event name changed",
    ):
        verifier._verify_exposure(
            exposure,
            verifier._load_json_strict(EVIDENCE_DIR / "environment.json"),
            verifier._load_json_strict(EVIDENCE_DIR / "truth_commitment.json"),
            verifier._load_json_strict(EVIDENCE_DIR / "prediction_commitment.json"),
            verifier._load_json_strict(EVIDENCE_DIR / "score_report.json"),
        )


def test_execution_source_hash_must_match_git_blob() -> None:
    environment = verifier._load_json_strict(EVIDENCE_DIR / "environment.json")
    evidence = verifier._load_json_strict(EVIDENCE_DIR / "evidence_manifest.json")
    tampered = deepcopy(environment)
    tampered["source_sha256"][verifier.CONFIG_PATH] = "0" * 64

    with pytest.raises(
        verifier.EvidenceVerificationError,
        match="Git blob differs from formal execution source hash",
    ):
        verifier._verify_execution_sources(PROJECT_ROOT, tampered, evidence)


def test_primary_risk_gate_uses_published_random_ranking_mean() -> None:
    config = deepcopy(_published_config())
    primary = config["endpoints"]["primary"]
    risk_endpoint = next(
        item
        for item in primary
        if item["endpoint_id"] == "catastrophic_risk_reduction_at_50_percent_issuance"
    )
    # The midpoint separates the empirical 10,000-ranking estimate from the
    # analytic matched-count diagnostic in this frozen result.
    risk_endpoint["threshold"] = 0.21655
    report = verifier._load_json_strict(EVIDENCE_DIR / "score_report.json")

    result = verifier._verify_metrics(EVIDENCE_DIR, report, config)

    assert result["random_ranking_risk_reduction_fraction"] < 0.21655
    assert result["test"]["analytic_risk_reduction_fraction"] > 0.21655
    assert (
        result["primary_gates"]["catastrophic_risk_reduction_at_50_percent_issuance"]
        is False
    )


def test_matched_members_must_be_unique_and_disjoint(evidence_copy: Path) -> None:
    path = evidence_copy / "matched_prefix_pairs.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[1, "left_cluster_id"] = frame.loc[0, "left_cluster_id"]
    frame.to_csv(path, index=False)
    report = verifier._load_json_strict(evidence_copy / "score_report.json")

    with pytest.raises(
        verifier.EvidenceVerificationError,
        match="Every matched-prefix member must occur in exactly one pair",
    ):
        verifier._verify_metrics(evidence_copy, report, _published_config())


def test_candidate_iae_is_recomputed_from_compact_trajectory_evidence() -> None:
    report = verifier._load_json_strict(EVIDENCE_DIR / "score_report.json")
    tampered = deepcopy(report)
    comparison = tampered["mean_forecast_comparison"]
    comparison["candidate_issued_mean_trajectory_iae_pp"] += 1.0
    comparison["candidate_minus_baseline_iae_pp"] += 1.0

    with pytest.raises(
        verifier.EvidenceVerificationError,
        match="candidate_issued_mean_trajectory_iae_pp differs",
    ):
        verifier._verify_metrics(EVIDENCE_DIR, tampered, _published_config())


def test_nonfinite_bootstrap_value_is_rejected_by_semantic_recomputation(
    evidence_copy: Path,
) -> None:
    path = evidence_copy / "bootstrap.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "risk_reduction_fraction"] = float("inf")
    frame.to_csv(path, index=False)
    report = verifier._load_json_strict(evidence_copy / "score_report.json")

    with pytest.raises(
        verifier.EvidenceVerificationError,
        match="bootstrap risk reduction contains a prohibited nonfinite value",
    ):
        verifier._verify_metrics(evidence_copy, report, _published_config())
