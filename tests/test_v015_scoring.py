from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_analysis as analysis
from lifetwin.experiments import calendar_long_horizon_v015_scoring as scoring
from lifetwin.experiments.calendar_long_horizon_v015_environment import (
    FormalEnvironmentIdentity,
)
from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    fit_structure_library,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    ConformalExpansionState,
    IsotonicState,
    LogisticRiskState,
    StandardizerState,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
    PLACEBO_FEATURE_NAMES,
    VISIBLE_STRESS_FEATURE_NAMES,
    recompute_label_free_pipeline,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
    load_frozen_protocol_config,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import (
    CalibrationDevelopmentState,
    CenterDevelopmentState,
    FrozenTrainingState,
    RiskDevelopmentState,
    default_software_versions,
    deserialize_model_state_json,
    serialize_model_state_json,
)


_CONTRACT = load_artifact_contract()


def _frame(filename: str, records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=_CONTRACT.csv_schema(filename).columns)


def _risk_state(names: tuple[str, ...]) -> LogisticRiskState:
    return LogisticRiskState(
        feature_names=names,
        standardizer=StandardizerState(
            mean=(0.0,) * len(names),
            scale=(1.0,) * len(names),
            zero_variance=(False,) * len(names),
        ),
        intercept=-2.0,
        coefficients=(0.0,) * len(names),
    )


@lru_cache(maxsize=1)
def _model_state_bytes() -> bytes:
    risk = RiskDevelopmentState(
        prefix_only_risk=_risk_state(PREFIX_FEATURE_NAMES),
        visible_stress_risk=_risk_state(VISIBLE_STRESS_FEATURE_NAMES),
        placebo_risk=_risk_state(PLACEBO_FEATURE_NAMES),
        arm_a_plus_s_plan_risk=_risk_state(ARM_A_PLUS_S_PLAN_FEATURE_NAMES),
        strongest_single_feature_name=PREFIX_FEATURE_NAMES[0],
        strongest_single_feature_orientation=1,
        strongest_single_feature_auroc=0.75,
        development_cluster_count=600,
        eligible_cluster_count=600,
        positive_label_count=300,
        negative_label_count=300,
    )
    calibration = CalibrationDevelopmentState(
        prefix_only_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        visible_stress_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        conformal=ConformalExpansionState(
            coverage=0.90,
            calibration_count=900,
            order_statistic_index=811,
            expansion_pp=1.0,
        ),
        selected_mean_baseline="target_prefix_persistence",
        mean_baseline_iae_pp=(
            ("target_prefix_persistence", 1.0),
            ("target_prefix_sqrt_time", 2.0),
            ("target_prefix_bounded_power_law", 3.0),
        ),
        calibration_cluster_count=900,
        positive_label_count=450,
        negative_label_count=450,
    )
    return serialize_model_state_json(
        FrozenTrainingState(
            center=CenterDevelopmentState(beta=0.5),
            risk=risk,
            calibration=calibration,
        ),
        center_development_input_hashes={"fixture": "a" * 64},
        risk_development_input_hashes={"fixture": "b" * 64},
        calibration_input_hashes={"fixture": "c" * 64},
        software_versions=default_software_versions(),
        created_utc="2026-07-23T08:00:00Z",
    )


@lru_cache(maxsize=1)
def _prediction_frames() -> dict[str, pd.DataFrame]:
    prefix = _frame(
        "prefix_pack.csv",
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": "score-fixture",
                "prefix_day": day,
                "observed_retention_pct": (100.0 - 0.8 * math.sqrt(day / 365.25)),
            }
            for day in PREFIX_DAYS
        ],
    )
    coordinates = _frame(
        "forecast_coordinates.csv",
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": "score-fixture",
                "forecast_day": day,
            }
            for day in FORECAST_DAYS
        ],
    )
    operating = _frame(
        "operating_pack.csv",
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": "score-fixture",
                **dict(
                    zip(
                        REAL_OPERATING_FIELDS,
                        (25.0, 0.5, 0.55, 250.0, 31.0, 0.6, 0.65, 300.0),
                        strict=True,
                    )
                ),
                **{
                    field: -0.8 + 0.2 * index
                    for index, field in enumerate(PLACEBO_FIELDS)
                },
            }
        ],
    )
    fitted = fit_structure_library(prefix, coordinates)
    state = deserialize_model_state_json(_model_state_bytes()).frozen_label_free_state
    downstream = recompute_label_free_pipeline(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        operating_pack=operating,
        member_fit_diagnostics=fitted.member_fit_diagnostics,
        member_forecast_bundle=fitted.member_forecast_bundle,
        state=state,
    )
    return {
        "prefix_pack.csv": prefix,
        "forecast_coordinates.csv": coordinates,
        "operating_pack.csv": operating,
        "member_fit_diagnostics.csv": fitted.member_fit_diagnostics,
        "member_forecast_bundle.csv": fitted.member_forecast_bundle,
        "prediction_bundle.csv": downstream.prediction_bundle,
        "risk_bundle.csv": downstream.primary_risk_bundle,
        "decision_bundle.csv": downstream.decision_bundle,
    }


def _score_bundle_fixture(
    *,
    intervals: str = "finite",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction = pd.DataFrame(
        {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "partition": "test",
            "cluster_id": "interval-fixture",
            "forecast_day": FORECAST_DAYS,
            "center_forecast_pct": np.linspace(98.0, 90.0, 8),
            "sqrt_time_forecast_pct": np.linspace(98.0, 91.0, 8),
            "bounded_power_forecast_pct": np.linspace(98.0, 89.0, 8),
            "base_interval_lower_pct": np.linspace(92.0, 82.0, 8),
            "base_interval_upper_pct": np.linspace(101.0, 99.0, 8),
            "calibrated_interval_lower_pct": np.linspace(90.0, 80.0, 8),
            "calibrated_interval_upper_pct": np.linspace(102.0, 100.0, 8),
            "canonical_prefix_content_sha256": "0" * 64,
        }
    )
    truth = pd.DataFrame(
        {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "partition": "test",
            "cluster_id": "interval-fixture",
            "truth_family": "single_power",
            "forecast_day": FORECAST_DAYS,
            "latent_retention_pct": np.linspace(98.0, 88.0, 8),
            "noisy_retention_pct": np.linspace(98.0, 88.0, 8),
        }
    )
    hashes = {
        score_id: hashlib.sha256(score_id.encode("ascii")).hexdigest()
        for score_id in analysis.RISK_SCORE_IDS
    }
    hashes["prefix_only"] = "0" * 64
    risk = pd.DataFrame(
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "test",
                "cluster_id": "interval-fixture",
                "score_id": score_id,
                "raw_risk_score": 0.1 + 0.01 * index,
                "calibrated_catastrophic_probability": (
                    0.25 if score_id in {"prefix_only", "visible_stress"} else np.nan
                ),
                "canonical_predictor_content_sha256": hashes[score_id],
                "successful_structure_family_count": 3,
            }
            for index, score_id in enumerate(analysis.RISK_SCORE_IDS)
        ]
    )
    decision = pd.DataFrame(
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "test",
                "cluster_id": "interval-fixture",
                "arm": arm,
                "hard_eligible": True,
                "issued": True,
                "issuance_rank": 1,
                "raw_risk_score": float(
                    risk.loc[risk["score_id"].eq(arm), "raw_risk_score"].iloc[0]
                ),
                "canonical_predictor_content_sha256": hashes[arm],
            }
            for arm in ("prefix_only", "visible_stress")
        ]
    )
    interval_columns = (
        "base_interval_lower_pct",
        "base_interval_upper_pct",
        "calibrated_interval_lower_pct",
        "calibrated_interval_upper_pct",
    )
    if intervals == "missing":
        prediction.loc[:, list(interval_columns)] = np.nan
        risk["successful_structure_family_count"] = 0
        risk["raw_risk_score"] = np.nan
        risk["calibrated_catastrophic_probability"] = np.nan
        decision["hard_eligible"] = False
        decision["issued"] = False
        decision["issuance_rank"] = None
        decision["raw_risk_score"] = np.nan
    elif intervals == "partial":
        prediction.loc[0, "base_interval_lower_pct"] = np.nan
    elif intervals == "infinite":
        prediction.loc[0, "calibrated_interval_upper_pct"] = np.inf
    return prediction, truth, risk, decision


def _policy_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for index in range(12):
        record: dict[str, object] = {
            "catastrophic": bool(index % 2),
            "hard_eligible_visible_stress": True,
        }
        for score_index, score_id in enumerate(analysis.RISK_SCORE_IDS):
            record[f"risk_{score_id}"] = float(index + score_index / 100.0)
            record[f"risk_hash_{score_id}"] = hashlib.sha256(
                f"{score_id}|{index}".encode("ascii")
            ).hexdigest()
        records.append(record)
    random = pd.DataFrame(
        {
            "ranking_index": range(5),
            "issued_count": [6] * 5,
            "issued_catastrophic_rate": [0.5] * 5,
            "analytic_random_expected_rate": [0.5] * 5,
            "relative_risk_reduction": [0.0] * 5,
        }
    )
    return pd.DataFrame(records), random


def _stochastic_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for family in analysis.TEST_FAMILIES:
        for within_family in range(2):
            rows.append(
                {
                    "truth_family": family,
                    "canonical_prefix_content_sha256": hashlib.sha256(
                        f"prefix|{index}".encode("ascii")
                    ).hexdigest(),
                    "catastrophic": bool(within_family),
                    "hard_eligible_visible_stress": True,
                    "risk_prefix_only": float(index),
                    "risk_visible_stress": float(20 - index),
                    "risk_placebo_8": float(index % 3),
                    "risk_hash_prefix_only": hashlib.sha256(
                        f"a|{index}".encode("ascii")
                    ).hexdigest(),
                    "risk_hash_visible_stress": hashlib.sha256(
                        f"b|{index}".encode("ascii")
                    ).hexdigest(),
                    "risk_hash_placebo_8": hashlib.sha256(
                        f"p|{index}".encode("ascii")
                    ).hexdigest(),
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _formal_environment() -> FormalEnvironmentIdentity:
    return FormalEnvironmentIdentity(
        git_commit="b" * 40,
        git_dirty=False,
        config_byte_sha256=_CONTRACT.config_byte_sha256,
        config_canonical_sha256=(
            load_frozen_protocol_config(_CONTRACT.config_path).config_sha256
        ),
        preregistration_byte_sha256="b" * 64,
        environment_lock_byte_sha256="c" * 64,
        python_version="3.12.13",
        python_implementation="CPython",
        platform="fixture-platform",
        machine="fixture-machine",
        processor="fixture-cpu",
        package_versions={"numpy": "2.5.1"},
        deterministic_environment={"OMP_NUM_THREADS": "1"},
        source_byte_hashes={"src/fixture.py": "d" * 64},
        active_threadpools=(),
    )


def _formal_metadata() -> dict[str, dict[str, object]]:
    return {
        filename: {
            "path": filename,
            "row_count": 1,
            "byte_count": len(filename.encode("utf-8")),
            "byte_sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
        }
        for filename in scoring.REQUIRED_FORMAL_NON_SCORE_ARTIFACTS
    }


def _finalize(result: scoring.V015ScoringResult) -> scoring.V015ScoringResult:
    protocol = load_frozen_protocol_config(_CONTRACT.config_path)
    metadata = _formal_metadata()
    metadata["model_state.json"]["byte_sha256"] = result.score_report[
        "model_state_byte_sha256"
    ]
    return scoring.finalize_run_manifest(
        result,
        environment_identity=_formal_environment(),
        implementation_freeze_record_sha256="e" * 64,
        protocol_freeze_git_commit=("b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91"),
        implementation_source_git_commit="a" * 40,
        seed_roots=protocol.seed_root_map(),
        seed_derivation=protocol.config()["design_partitions"]["seed_derivation"],
        prediction_worker_count=6,
        wall_time_seconds=1.25,
        formal_artifact_metadata=metadata,
    )


def test_exact_required_artifact_registry_and_canonical_schemas() -> None:
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
    assert load_frozen_protocol_config(_CONTRACT.config_path).config()[
        "firewall_and_artifacts"
    ]["required_score_artifacts"] == list(scoring.REQUIRED_SCORE_ARTIFACTS)
    assert tuple(scoring.SCORE_FRAME_SCHEMAS) == (scoring.REQUIRED_SCORE_CSV_ARTIFACTS)
    for schema in scoring.SCORE_FRAME_SCHEMAS.values():
        assert len(schema.columns) == len(set(schema.columns))
        assert set(schema.key).issubset(schema.columns)


def test_real_label_free_recomputation_accepts_exact_and_rejects_tamper() -> None:
    frames = {
        name: frame.copy(deep=True) for name, frame in _prediction_frames().items()
    }
    _, recomputed = scoring.validate_and_recompute_committed_predictions(
        prediction_frames=frames,
        model_state_bytes=_model_state_bytes(),
        formal=False,
    )
    assert recomputed.prediction_bundle.equals(frames["prediction_bundle.csv"])

    frames["prediction_bundle.csv"].loc[0, "center_forecast_pct"] += 0.001
    with pytest.raises(
        scoring.V015ScoringError,
        match="differs byte-for-byte",
    ):
        scoring.validate_and_recompute_committed_predictions(
            prediction_frames=frames,
            model_state_bytes=_model_state_bytes(),
            formal=False,
        )


def test_all_nan_unsupported_intervals_are_inconclusive_not_void() -> None:
    points, truth, risk, decisions = _score_bundle_fixture(intervals="missing")
    _, trajectories = analysis.score_trajectory_table(points, truth, risk, decisions)
    assert pd.isna(trajectories.loc[0, "simultaneous_interval_covered"])
    assert np.isnan(trajectories.loc[0, "max_interval_width_pp"])

    gates = [
        analysis.GateEvaluation(
            gate_id,
            "pass",
            True,
            "fixture",
        )
        for gate_id in analysis.REQUIRED_GATE_IDS
    ]
    target = "core_test_simultaneous_trajectory_coverage"
    gates[analysis.REQUIRED_GATE_IDS.index(target)] = analysis.GateEvaluation(
        target,
        "inconclusive",
        None,
        "fixture",
        ("legitimate zero structural support",),
    )
    status = analysis.resolve_result_status(gates)
    assert status["status"] == "inconclusive_not_success"
    assert status["void_reasons"] == []


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("partial", "partially missing"),
        ("infinite", "cannot contain infinity"),
    ],
)
def test_partial_or_infinite_intervals_make_public_score_void(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    match: str,
) -> None:
    fixture = _score_bundle_fixture(intervals=mode)

    def invalid_score(**_: object) -> scoring.V015ScoringResult:
        analysis.score_trajectory_table(*fixture)
        raise AssertionError("unreachable")

    monkeypatch.setattr(scoring, "_score_committed_artifacts", invalid_score)
    result = scoring.score_committed_artifacts(
        prediction_frames={},
        truth_frames={},
        model_state_bytes=b"fixture",
    )
    assert result.score_report["status"]["status"] == "void"
    assert match in result.score_report["status"]["void_reasons"][0]


def test_all_nine_frozen_policies_are_reported() -> None:
    trajectories, random = _policy_fixture()
    table, reductions = scoring._policy_comparison(
        trajectories,
        random,
        issue_count=6,
        partition="test",
    )
    assert tuple(table["score_id"]) == analysis.RISK_SCORE_IDS
    assert set(reductions) == set(analysis.RISK_SCORE_IDS)
    assert table["issued_count"].eq(6).all()
    curves = scoring._risk_coverage_curves(trajectories)
    assert len(curves) == len(analysis.RISK_SCORE_IDS) * 3
    assert {row["score_id"] for row in curves} == set(analysis.RISK_SCORE_IDS)
    assert {row["issuance_fraction"] for row in curves} == {
        0.25,
        0.50,
        0.75,
    }


def test_status_resolution_keeps_frozen_precedence() -> None:
    required = ("a", "b")
    passed = analysis.GateEvaluation("a", "pass", True, "fixture")
    failed = analysis.GateEvaluation("b", "fail", False, "fixture")
    unresolved = analysis.GateEvaluation(
        "b", "inconclusive", None, "fixture", ("missing",)
    )
    assert (
        analysis.resolve_result_status(
            (passed, analysis.GateEvaluation("b", "pass", True, "fixture")),
            required_gate_ids=required,
        )["status"]
        == "success"
    )
    assert (
        analysis.resolve_result_status((passed, failed), required_gate_ids=required)[
            "status"
        ]
        == "failure"
    )
    assert (
        analysis.resolve_result_status(
            (passed, unresolved), required_gate_ids=required
        )["status"]
        == "inconclusive_not_success"
    )
    assert (
        analysis.resolve_result_status(
            (passed, failed),
            void_reasons=("commitment mismatch",),
            required_gate_ids=required,
        )["status"]
        == "void"
    )
    mixed = analysis.resolve_result_status(
        (
            passed,
            failed,
            analysis.GateEvaluation(
                "c",
                "inconclusive",
                None,
                "fixture",
                ("undefined",),
            ),
        ),
        required_gate_ids=("a", "b", "c"),
    )
    assert mixed["status"] == "failure"
    assert mixed["status_resolution_convention"] == (
        "void > failure > inconclusive_not_success > success"
    )


def test_undefined_bootstrap_preserves_available_coverage_gates() -> None:
    reduction = analysis.RiskReduction(
        issued_count=950,
        issued_catastrophic_rate=0.1,
        random_expected_catastrophic_rate=0.2,
        relative_risk_reduction=0.5,
    )
    core = analysis.CoverageSummary(
        n=1500,
        covered=1400,
        coverage=1400 / 1500,
        one_sided_95_lower=0.90,
        median_max_width_pp=20.0,
        percentile_95_max_width_pp=30.0,
    )
    intrinsic = analysis.CoverageSummary(
        n=250,
        covered=225,
        coverage=0.90,
        one_sided_95_lower=0.85,
        median_max_width_pp=20.0,
        percentile_95_max_width_pp=30.0,
    )
    gates = scoring._primary_gates(
        visible=reduction,
        prefix=reduction,
        bootstrap=None,
        core=core,
        intrinsic=intrinsic,
        issued_iae_delta=0.0,
    )
    by_id = {gate.gate_id: gate for gate in gates}
    assert by_id["visible_stress_catastrophic_risk_reduction"].state == "inconclusive"
    assert by_id["core_test_simultaneous_trajectory_coverage"].state == "pass"
    assert by_id["intrinsic_pair_simultaneous_both_future_coverage"].state == "pass"


def test_small_stochastic_hook_is_deterministic_without_formal_counts() -> None:
    trajectories = _stochastic_fixture()
    counts = scoring._AnalysisCounts(
        random_rankings=7,
        bootstrap_resamples=5,
        stress_permutations=3,
    )
    first = scoring._run_stochastic_fixture_analyses(
        trajectories, issue_count=4, counts=counts
    )
    second = scoring._run_stochastic_fixture_analyses(
        trajectories.sample(frac=1.0, random_state=91),
        issue_count=4,
        counts=counts,
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert len(first[0]) == 7
    assert len(first[1]) == 5


def test_bootstrap_precomputes_tie_hashes_once_per_source_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectories = _stochastic_fixture()
    original = analysis._arm_tie_digest
    calls = 0

    def counted(protocol_id: str, arm: str, content_hash: object) -> str:
        nonlocal calls
        calls += 1
        return original(protocol_id, arm, content_hash)

    monkeypatch.setattr(analysis, "_arm_tie_digest", counted)
    analysis.bootstrap_risk_reductions(
        trajectories,
        protocol_id=FROZEN_PROTOCOL_ID,
        issue_count=4,
        resamples=11,
    )
    assert calls == len(trajectories) * 3


def test_mandatory_descriptive_reporting_helpers_are_finite() -> None:
    decoded = deserialize_model_state_json(_model_state_bytes())
    calibration_trajectories = pd.DataFrame(
        {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "partition": "calibration",
            "cluster_id": [f"c-{index:04d}" for index in range(900)],
            "catastrophic": [bool(index % 2) for index in range(900)],
        }
    )
    calibration_risks = pd.DataFrame(
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": f"c-{index:04d}",
                "score_id": score_id,
                "calibrated_catastrophic_probability": (0.75 if index % 2 else 0.25),
            }
            for index in range(900)
            for score_id in ("prefix_only", "visible_stress")
        ]
    )
    isotonic = scoring._isotonic_calibration_diagnostics(
        calibration_trajectories,
        calibration_risks,
        decoded,
    )
    assert len(isotonic) == 2
    assert all(record["available"] for record in isotonic)
    assert all(record["n"] == 900 for record in isotonic)

    features = pd.DataFrame(
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": partition,
                "cluster_id": f"{partition}-{index}",
                "successful_structure_family_count": 3.0 + index,
                "effective_unique_shape_count": 4.0 + index,
                "fit_failure_count": 4.0 - index,
                "parameter_boundary_hit_fraction": 0.1 * index,
            }
            for partition in ("test", "audit")
            for index in range(2)
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "protocol_id": row["protocol_id"],
                "partition": row["partition"],
                "cluster_id": row["cluster_id"],
                "credible_variant": credible,
                "parameter_boundary_hit_fraction": 0.25 if credible else np.nan,
            }
            for row in features.to_dict(orient="records")
            for credible in (True, False)
        ]
    )
    structure = scoring._structure_diagnostics(features, diagnostics)
    assert {record["partition"] for record in structure} == {
        "test",
        "audit",
    }
    assert all(record["available"] for record in structure)

    operating = pd.DataFrame(
        [
            {
                "partition": partition,
                **{
                    field: float(index + field_index + (partition == "audit"))
                    for field_index, field in enumerate(REAL_OPERATING_FIELDS)
                },
            }
            for partition in ("test", "audit")
            for index in range(3)
        ]
    )
    shift_trajectories = pd.DataFrame(
        [
            {
                "partition": partition,
                "truth_family": analysis.TEST_FAMILIES[index % 2],
            }
            for partition in ("test", "audit")
            for index in range(3)
        ]
    )
    shift = scoring._test_audit_distribution_shift(operating, shift_trajectories)
    assert shift["available"] is True
    assert len(shift["operating_fields"]) == len(REAL_OPERATING_FIELDS)
    assert len(shift["family_proportions"]) == len(analysis.TEST_FAMILIES)


def test_formal_api_counts_are_exact_and_not_caller_overridable() -> None:
    assert scoring._FORMAL_ANALYSIS_COUNTS == scoring._AnalysisCounts(
        random_rankings=10_000,
        bootstrap_resamples=5_000,
        stress_permutations=10_000,
    )
    assert (
        "counts" not in inspect.signature(scoring.score_committed_artifacts).parameters
    )


def test_void_artifacts_have_header_only_csvs_and_final_manifest_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_score(**_: object) -> scoring.V015ScoringResult:
        raise analysis.V015AnalysisError("fixture interval corruption")

    monkeypatch.setattr(scoring, "_score_committed_artifacts", invalid_score)
    provisional = scoring.score_committed_artifacts(
        prediction_frames={},
        truth_frames={},
        model_state_bytes=b"fixture",
    )
    with pytest.raises(scoring.V015ScoringError, match="must be finalized"):
        scoring.required_score_artifact_payloads(provisional)
    report = provisional.score_report
    assert len(report["risk_coverage_curves_secondary"]) == 27
    assert len(report["isotonic_calibration_diagnostics"]) == 2
    assert report["structure_diagnostics"]
    assert report["test_audit_distribution_shift"]["available"] is False
    scoring.canonical_result_summary_bytes(report)

    finalized = _finalize(provisional)
    payloads = scoring.required_score_artifact_payloads(finalized)
    assert tuple(payloads) == scoring.REQUIRED_SCORE_ARTIFACTS
    for filename in scoring.REQUIRED_SCORE_CSV_ARTIFACTS:
        assert payloads[filename].count(b"\n") == 1
    assert [item["path"] for item in finalized.run_manifest["artifacts"]] == [
        *scoring.REQUIRED_FORMAL_NON_SCORE_ARTIFACTS,
        *scoring.REQUIRED_SCORE_ARTIFACTS[:-1],
    ]
    provenance = finalized.run_manifest["provenance"]
    assert provenance["protocol_freeze_git_commit"] == (
        "b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91"
    )
    assert provenance["implementation_source_git_commit"] == "a" * 40
    assert provenance["execution_git_commit"] == "b" * 40

    tampered = finalized.point_scores.copy()
    tampered.loc[0] = [0] * len(tampered.columns)
    broken = replace(finalized, point_scores=tampered)
    with pytest.raises(
        scoring.V015ScoringError,
        match="does not bind",
    ):
        scoring.required_score_artifact_payloads(broken)


def test_declared_count_failure_is_inconclusive_with_publishable_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_score(**_: object) -> scoring.V015ScoringResult:
        raise analysis.V015InconclusiveError(
            "test_common_eligible_count=1804 minimum=1805"
        )

    monkeypatch.setattr(scoring, "_score_committed_artifacts", unavailable_score)
    provisional = scoring.score_committed_artifacts(
        prediction_frames={},
        truth_frames={},
        model_state_bytes=b"fixture",
    )
    status = provisional.score_report["status"]
    assert status["status"] == "inconclusive_not_success"
    assert status["void_reasons"] == []
    assert any(
        "test_common_eligible_count=1804" in reason
        for reason in status["inconclusive_reasons"]
    )

    finalized = _finalize(provisional)
    payloads = scoring.required_score_artifact_payloads(finalized)
    for filename in scoring.REQUIRED_SCORE_CSV_ARTIFACTS:
        assert payloads[filename].count(b"\n") == 1
