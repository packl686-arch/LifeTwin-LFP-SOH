from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_analysis as analysis
from lifetwin.experiments.calendar_long_horizon_synthetic import (
    FrozenScoreResult,
    MatchedPairAuditResult,
    SyntheticProtocolError,
    TRUTH_FAMILY_IDS,
    load_frozen_protocol_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v1.json"
)
TRUTH_FAMILIES = TRUTH_FAMILY_IDS


def _protocol():
    return load_frozen_protocol_config(CONFIG_PATH)


def _score_fixture(
    *, eligible_test_count: int = 1_000, nonfinite_calibration_baseline: bool = False
) -> FrozenScoreResult:
    protocol = _protocol()
    records: list[dict[str, object]] = []
    for partition, count in (("calibration", 500), ("test", 1_000), ("audit", 500)):
        for index in range(count):
            cluster_id = f"{partition}-{index:04d}"
            prefix_hash = hashlib.sha256(cluster_id.encode("ascii")).hexdigest()
            family = TRUTH_FAMILIES[index % len(TRUTH_FAMILIES)]
            catastrophic = (partition == "test" and index >= 900) or (
                partition == "audit" and index >= 450
            )
            eligible = partition != "test" or index < eligible_test_count
            issued = bool(
                (partition == "test" and eligible and index < 500)
                or (partition == "audit" and index < 250)
            )
            disagreement = (10.0 if catastrophic else 0.1) + index / count
            if not eligible:
                disagreement = math.inf
            for day_index, day in enumerate(protocol.forecast_days):
                truth = 95.0 - 0.1 * day_index
                persistence = truth + 0.5
                if (
                    nonfinite_calibration_baseline
                    and partition == "calibration"
                    and index == 0
                ):
                    persistence = math.nan
                records.append(
                    {
                        "protocol_id": protocol.protocol_id,
                        "partition": partition,
                        "cluster_id": cluster_id,
                        "forecast_day": day,
                        "candidate_point_forecast_pct": truth
                        + (6.0 if catastrophic else 0.0),
                        "persistence_forecast_pct": persistence,
                        "sqrt_time_forecast_pct": truth + 0.4,
                        "bounded_power_forecast_pct": truth + 0.3,
                        "structure_envelope_lower_pct": truth - 0.2,
                        "structure_envelope_upper_pct": truth + 0.2,
                        "canonical_prefix_content_sha256": prefix_hash,
                        "truth_family": family,
                        "latent_retention_pct": truth,
                        "noisy_retention_pct": truth,
                        "credible_structure_family_count": 3 if eligible else 1,
                        "fit_failure_count": 1 if index % 17 == 0 else 0,
                        "best_prefix_rmse_pp": (
                            disagreement / 10.0 if eligible else math.inf
                        ),
                        "disagreement_score_pp": disagreement,
                        "hard_eligible": eligible,
                        "primary_issuance_rank": index + 1 if eligible else None,
                        "primary_issued": issued,
                        "abstention_reasons": "" if issued else "not_issued",
                    }
                )
    point_scores = pd.DataFrame(records)
    return FrozenScoreResult(
        point_scores=point_scores,
        trajectory_scores=pd.DataFrame(),
        prediction_sha256="1" * 64,
        decision_sha256="2" * 64,
        prefix_sha256="6" * 64,
        forecast_coordinates_sha256="3" * 64,
        member_diagnostics_sha256="4" * 64,
        truth_sha256="5" * 64,
        verified_decision_bytes=b"test-only",
        _verification_token=object(),
    )


def _matched(
    *,
    qualified: int = 200,
    fraction: float = 0.9,
    pair_scores: pd.DataFrame | None = None,
):
    return MatchedPairAuditResult(
        pair_scores=pd.DataFrame() if pair_scores is None else pair_scores,
        calibration_disagreement_threshold_pp=1.0,
        endpoint_available=True,
        unavailable_reason=None,
        qualified_pair_count=qualified,
        both_rejected_pair_count=int(qualified * fraction),
        both_rejected_fraction=fraction,
    )


def _replace_expensive_resampling(
    monkeypatch: pytest.MonkeyPatch, *, defined: bool = True
) -> None:
    def fake_random(
        eligible: pd.DataFrame,
        *,
        issue_count: int,
        ranking_count: int | None = None,
    ) -> pd.DataFrame:
        count = (
            analysis.RANDOM_RANKING_COUNT if ranking_count is None else ranking_count
        )
        is_defined = defined and len(eligible) >= issue_count
        return pd.DataFrame(
            {
                "ranking_index": np.arange(count),
                "status": (
                    ["defined"] * count
                    if is_defined
                    else ["undefined_insufficient_eligible"] * count
                ),
                "issued_count": [issue_count if is_defined else 0] * count,
                "catastrophic_count": [50 if is_defined else None] * count,
                "catastrophic_rate": [0.1 if is_defined else None] * count,
            }
        )

    def fake_bootstrap(
        test: pd.DataFrame,
        *,
        issue_count: int = analysis.TEST_ISSUE_COUNT,
        resamples: int | None = None,
    ) -> pd.DataFrame:
        count = analysis.BOOTSTRAP_RESAMPLES if resamples is None else resamples
        is_defined = defined and int(test["hard_eligible"].sum()) >= issue_count
        return pd.DataFrame(
            {
                "replicate": np.arange(count),
                "status": (
                    ["defined"] * count
                    if is_defined
                    else ["undefined_insufficient_eligible"] * count
                ),
                "hard_eligible_count": [int(test["hard_eligible"].sum())] * count,
                "random_expected_catastrophic_rate": [0.1 if is_defined else None]
                * count,
                "issued_catastrophic_rate": [0.05 if is_defined else None] * count,
                "risk_reduction_fraction": [0.5 if is_defined else None] * count,
            }
        )

    monkeypatch.setattr(analysis, "_random_rejection_distribution", fake_random)
    monkeypatch.setattr(analysis, "_bootstrap_risk_reduction", fake_bootstrap)


def test_random_rankings_use_full_sha256_order() -> None:
    eligible = pd.DataFrame(
        {
            "canonical_prefix_content_sha256": [
                hashlib.sha256(f"prefix-{index}".encode()).hexdigest()
                for index in range(4)
            ],
            "catastrophic_error": [False, True, False, True],
        }
    )
    result = analysis._random_rejection_distribution(
        eligible, issue_count=2, ranking_count=1
    )
    hashes = eligible["canonical_prefix_content_sha256"].tolist()
    selected = sorted(
        range(4),
        key=lambda index: hashlib.sha256(
            f"{analysis.RANDOM_REJECTION_SEED}|0|{hashes[index]}".encode("ascii")
        ).digest(),
    )[:2]
    expected = int(eligible.iloc[selected]["catastrophic_error"].sum())
    assert result.loc[0, "catastrophic_count"] == expected
    assert result.loc[0, "status"] == "defined"


def test_random_ranking_full_digest_collision_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eligible = pd.DataFrame(
        {
            "canonical_prefix_content_sha256": ["a" * 64, "b" * 64],
            "catastrophic_error": [False, True],
        }
    )

    class FakeHash:
        def digest(self) -> bytes:
            return b"\x00" * 32

    monkeypatch.setattr(analysis.hashlib, "sha256", lambda _: FakeHash())
    with pytest.raises(SyntheticProtocolError, match="collision"):
        analysis._random_rejection_distribution(
            eligible, issue_count=1, ranking_count=1
        )


def test_bootstrap_reports_every_undefined_replicate() -> None:
    test = pd.DataFrame(
        {
            "canonical_prefix_content_sha256": [
                hashlib.sha256(family.encode("ascii")).hexdigest()
                for family in TRUTH_FAMILIES
            ],
            "truth_family": TRUTH_FAMILIES,
            "hard_eligible": [False] * len(TRUTH_FAMILIES),
            "disagreement_score_pp": [math.inf] * len(TRUTH_FAMILIES),
            "catastrophic_error": [True, False, True, False, True],
        }
    )
    result = analysis._bootstrap_risk_reduction(test, issue_count=1, resamples=7)
    assert len(result) == 7
    assert set(result["status"]) == {"undefined_insufficient_eligible"}
    assert result["risk_reduction_fraction"].isna().all()


def test_bootstrap_rng_strata_follow_frozen_truth_family_declaration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = pd.DataFrame(
        {
            "canonical_prefix_content_sha256": [
                hashlib.sha256(f"order-{family}".encode("ascii")).hexdigest()
                for family in reversed(TRUTH_FAMILIES)
            ],
            "truth_family": tuple(reversed(TRUTH_FAMILIES)),
            "hard_eligible": [False] * len(TRUTH_FAMILIES),
            "disagreement_score_pp": [math.inf] * len(TRUTH_FAMILIES),
            "catastrophic_error": [False] * len(TRUTH_FAMILIES),
        }
    )
    sampled_family_order: list[str] = []

    class RecordingGenerator:
        def choice(
            self,
            indices: np.ndarray,
            *,
            size: int,
            replace: bool,
        ) -> np.ndarray:
            assert size == len(indices)
            assert replace is True
            sampled_family_order.append(str(test.iloc[int(indices[0])]["truth_family"]))
            return indices

    monkeypatch.setattr(analysis.np.random, "Generator", lambda _: RecordingGenerator())
    analysis._bootstrap_risk_reduction(test, issue_count=1, resamples=1)
    assert tuple(sampled_family_order) == TRUTH_FAMILY_IDS


def test_bootstrap_rejects_missing_or_unknown_truth_family_strata() -> None:
    test = pd.DataFrame(
        {
            "canonical_prefix_content_sha256": ["a" * 64, "b" * 64],
            "truth_family": [TRUTH_FAMILIES[0], "unknown_family"],
            "hard_eligible": [False, False],
            "disagreement_score_pp": [math.inf, math.inf],
            "catastrophic_error": [False, False],
        }
    )
    with pytest.raises(SyntheticProtocolError, match="frozen v1"):
        analysis._bootstrap_risk_reduction(test, issue_count=1, resamples=1)


def test_family_metrics_explicitly_report_risk_reversal() -> None:
    test = pd.DataFrame(
        {
            "truth_family": ["reversal", "reversal", "reversal", "reversal"],
            "finite_forecast": [True] * 4,
            "hard_eligible": [True] * 4,
            "primary_issued": [True, True, False, False],
            "catastrophic_error": [True, True, False, False],
            "endpoint_absolute_error_pp": [6.0, 6.0, 0.0, 0.0],
            "trajectory_iae_pp": [3.0, 3.0, 0.0, 0.0],
            "disagreement_score_pp": [0.1, 0.2, 1.0, 1.1],
        }
    )
    metrics = analysis._family_metrics(test)
    family = metrics.loc[metrics["truth_family"].eq("reversal")].iloc[0]
    assert family["issued_catastrophic_count"] == 2
    assert family["issued_catastrophic_rate"] == pytest.approx(1.0)
    assert family["analytic_random_expected_catastrophic_rate"] == pytest.approx(0.5)
    assert family["issued_vs_analytic_random_risk_reduction_fraction"] == pytest.approx(
        -1.0
    )
    assert bool(family["family_specific_reversal"]) is True
    reversals = analysis._family_specific_reversal_records(metrics, partition="test")
    assert reversals == [
        {
            "partition": "test",
            "truth_family": "reversal",
            "hard_eligible_count": 4,
            "issued_count": 2,
            "issued_catastrophic_count": 2,
            "issued_catastrophic_rate": 1.0,
            "analytic_random_expected_catastrophic_rate": 0.5,
            "issued_vs_analytic_random_risk_reduction_fraction": -1.0,
            "family_specific_reversal": True,
        }
    ]


def test_analysis_reports_all_frozen_secondary_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_expensive_resampling(monkeypatch)
    result = analysis.analyze_synthetic_identifiability(
        _score_fixture(), _matched(), _protocol()
    )
    assert result.report["status"] == "success"
    assert len(result.random_rejection) == 10_000
    assert len(result.bootstrap) == 5_000
    assert set(result.rejection_policy_metrics["policy_id"]) == {
        "structure_disagreement",
        "prefix_fit_error",
    }
    assert set(
        result.rejection_policy_metrics["issuance_fraction_of_all_test_clusters"]
    ) == {0.25, 0.5, 0.75}
    assert set(result.forecast_day_metrics["forecast_day"]) == set(
        _protocol().forecast_days
    )
    assert "__all__" in set(result.family_metrics["truth_family"])
    assert set(result.member_fit_metrics["fit_failure_status"]) == {
        "no_failures",
        "one_or_more_failures",
    }
    assert analysis._declared_candidate_variant_count(_protocol()) == 85
    assert set(result.member_fit_metrics["declared_variant_count_per_cluster"]) == {85}
    assert np.allclose(
        result.member_fit_metrics["failed_variant_fit_rate"],
        result.member_fit_metrics["total_failed_variant_count"]
        / result.member_fit_metrics["total_declared_variant_fit_count"],
    )
    fit_summary = result.report["secondary"]["model_fit_failure_summary"]
    test_fit_summary = next(item for item in fit_summary if item["partition"] == "test")
    assert test_fit_summary["declared_variant_count_per_cluster"] == 85
    assert test_fit_summary["total_declared_variant_fit_count"] == 85_000
    assert test_fit_summary["failed_variant_fit_count"] == 59
    assert test_fit_summary["cluster_with_any_fit_failure_count"] == 59
    assert set(result.noise_sensitivity_metrics["truth_family"]) == {
        "__all__",
        *TRUTH_FAMILIES,
    }
    assert result.report["secondary"]["noise_free_versus_noisy_future_sensitivity"]
    assert result.report["secondary"]["family_specific_reversal_count"] == 0
    assert result.report["secondary"]["family_specific_reversals"] == []
    assert len(result.report["secondary"]["audit_family_metrics"]) == 6
    assert result.report["secondary"][
        "disagreement_auroc_for_catastrophic_error"
    ] == pytest.approx(1.0)
    json.dumps(result.report, allow_nan=False)


def test_matched_prefix_report_counts_nonfinite_disagreements_as_model_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_expensive_resampling(monkeypatch)
    pair_scores = pd.DataFrame(
        {
            "left_disagreement_score_pp": [0.5, math.inf, math.nan],
            "right_disagreement_score_pp": [1.5, 2.5, math.inf],
        }
    )
    result = analysis.analyze_synthetic_identifiability(
        _score_fixture(), _matched(pair_scores=pair_scores), _protocol()
    )
    matched_report = result.report["matched_prefix_audit"]
    assert matched_report["evaluated_pair_row_count"] == 3
    assert matched_report["evaluated_member_count"] == 6
    assert matched_report["nonfinite_disagreement_member_count"] == 3
    assert matched_report["model_failure_member_count"] == 3
    assert matched_report["pair_with_any_nonfinite_disagreement_count"] == 2
    assert matched_report["pair_with_any_model_failure_count"] == 2
    assert matched_report["model_failure_definition"] == (
        "nonfinite_disagreement_score"
    )
    json.dumps(result.report, allow_nan=False)


def test_insufficient_eligibility_is_inconclusive_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_expensive_resampling(monkeypatch)
    result = analysis.analyze_synthetic_identifiability(
        _score_fixture(eligible_test_count=499), _matched(), _protocol()
    )
    assert result.report["status"] == "inconclusive_not_success"
    assert (
        "fewer_than_500_hard_eligible_test_clusters"
        in result.report["inconclusive_reasons"]
    )
    assert result.report["test_policy"]["issued_count"] == 499
    assert result.report["mean_forecast_comparison"]["evaluable"] is False
    json.dumps(result.report, allow_nan=False)


def test_nonfinite_calibration_baseline_makes_iae_endpoint_unevaluable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_expensive_resampling(monkeypatch)
    result = analysis.analyze_synthetic_identifiability(
        _score_fixture(nonfinite_calibration_baseline=True),
        _matched(),
        _protocol(),
    )
    comparison = result.report["mean_forecast_comparison"]
    assert result.report["status"] == "failure"
    assert comparison["evaluable"] is False
    assert comparison["unavailable_reason"] == (
        "calibration_baseline_selection_unavailable"
    )
    persistence = next(
        item
        for item in result.report["calibration_baseline_selection"]
        if item["model_id"] == "target_prefix_persistence"
    )
    assert persistence["unavailable_trajectory_iae_count"] == 1
    assert persistence["mean_trajectory_iae_pp"] is None
    json.dumps(result.report, allow_nan=False)


def test_auroc_excludes_ineligible_nonfinite_disagreement() -> None:
    test = pd.DataFrame(
        {
            "hard_eligible": [True, True, False],
            "finite_forecast": [True, True, True],
            "catastrophic_error": [False, True, True],
            "disagreement_score_pp": [0.1, 1.0, math.inf],
        }
    )
    auc, count = analysis._disagreement_auroc(test)
    assert auc == pytest.approx(1.0)
    assert count == 2
