from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import inspect

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_analysis as v015
from lifetwin.experiments import calendar_long_horizon_v016_analysis as analysis
from lifetwin.experiments import calendar_long_horizon_v016_collision as collision
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_EXPECTED_SEED_ROOTS,
    V021_PROTOCOL_ID,
)


def _risk_frame(members_per_family: int = 8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(analysis.TEST_FAMILIES):
        for member_index in range(members_per_family):
            ordinal = family_index * members_per_family + member_index
            rows.append(
                {
                    "truth_family": family,
                    "canonical_prefix_content_sha256": f"{ordinal + 1:064x}",
                    "catastrophic": (member_index + family_index) % 4 == 0,
                    "hard_eligible_visible_stress": True,
                    "risk_prefix_only": float((member_index * 3 + family_index) % 11),
                    "risk_visible_stress": float(
                        (member_index * 5 + family_index * 2) % 13
                    ),
                    "risk_placebo_8": float((member_index * 7 + family_index * 3) % 17),
                    "risk_hash_prefix_only": f"{10_000 + ordinal:064x}",
                    "risk_hash_visible_stress": f"{20_000 + ordinal:064x}",
                    "risk_hash_placebo_8": f"{30_000 + ordinal:064x}",
                }
            )
    return pd.DataFrame(rows)


def _policy_frame() -> pd.DataFrame:
    frame = _risk_frame()
    for score_index, score_id in enumerate(analysis.RISK_SCORE_IDS):
        frame[f"risk_{score_id}"] = [
            float((row_index * (score_index + 3)) % 19)
            for row_index in range(len(frame))
        ]
        frame[f"risk_hash_{score_id}"] = [
            f"{100_000 + score_index * 1_000 + row_index:064x}"
            for row_index in range(len(frame))
        ]
    return frame


class _VisibleStressState:
    feature_names = ("prefix_feature", *analysis.REAL_OPERATING_FIELDS)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        operating = matrix[:, 1:]
        offsets = np.arange(len(analysis.REAL_OPERATING_FIELDS), dtype=float) * 100.0
        np.testing.assert_array_equal(
            operating - operating[:, [0]],
            np.broadcast_to(offsets, operating.shape),
        )
        return operating[:, 0]


def _stress_frames(
    members_per_family: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    for family_index, family in enumerate(analysis.TEST_FAMILIES):
        for member_index in range(members_per_family):
            ordinal = family_index * members_per_family + member_index
            cluster_id = f"cluster-{ordinal:03d}"
            rows.append(
                {
                    "partition": "test",
                    "cluster_id": cluster_id,
                    "truth_family": family,
                    "canonical_prefix_content_sha256": f"{ordinal + 1:064x}",
                    "hard_eligible_visible_stress": True,
                    "catastrophic": member_index in {0, 3},
                    "prefix_feature": float(ordinal),
                }
            )
            base = float(member_index * len(analysis.TEST_FAMILIES) + family_index)
            operating_rows.append(
                {
                    "partition": "test",
                    "cluster_id": cluster_id,
                    **{
                        field: base + field_index * 100.0
                        for field_index, field in enumerate(
                            analysis.REAL_OPERATING_FIELDS
                        )
                    },
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(operating_rows)


def _run_stress(
    frame: pd.DataFrame,
    operating: pd.DataFrame,
    *,
    permutations: int = 13,
) -> pd.DataFrame:
    empty_coordinates = pd.DataFrame(columns=["partition", "cluster_id"])
    return analysis.stress_permutation_metrics(
        frame,
        operating,
        empty_coordinates,
        empty_coordinates,
        visible_stress_state=_VisibleStressState(),
        random_expected_catastrophic_rate=0.4,
        observed_prefix_only_risk_reduction=0.1,
        issue_count=16,
        permutations=permutations,
    )


def test_frozen_v21_cardinalities_roots_and_identity() -> None:
    assert analysis.RANDOM_RANKING_COUNT == 10_000
    assert analysis.BOOTSTRAP_RESAMPLES == 5_000
    assert analysis.STRESS_PERMUTATIONS == 10_000
    assert analysis.RANDOM_ROOT == 202607260210
    assert analysis.BOOTSTRAP_ROOT == 202607260211
    assert analysis.STRESS_PERMUTATION_ROOT == 202607260212
    assert analysis.RANDOM_ROOT == V021_EXPECTED_SEED_ROOTS["random_rankings"]
    assert analysis.BOOTSTRAP_ROOT == V021_EXPECTED_SEED_ROOTS["bootstrap"]
    assert (
        analysis.STRESS_PERMUTATION_ROOT
        == V021_EXPECTED_SEED_ROOTS["stress_permutations"]
    )
    assert {
        analysis.RANDOM_ROOT,
        analysis.BOOTSTRAP_ROOT,
        analysis.STRESS_PERMUTATION_ROOT,
    }.isdisjoint(
        {
            v015.RANDOM_ROOT,
            v015.BOOTSTRAP_ROOT,
            v015.STRESS_PERMUTATION_ROOT,
        }
    )
    assert (
        inspect.signature(analysis.deterministic_random_rankings)
        .parameters["rankings"]
        .default
        == 10_000
    )
    assert (
        inspect.signature(analysis.bootstrap_risk_reductions)
        .parameters["resamples"]
        .default
        == 5_000
    )
    assert (
        inspect.signature(analysis.stress_permutation_metrics)
        .parameters["permutations"]
        .default
        == 10_000
    )


def test_analysis_and_collision_share_every_stochastic_derivation() -> None:
    assert (
        analysis.derive_random_ranking_digest is collision.derive_random_ranking_digest
    )
    assert analysis.derive_bootstrap_seed is collision.derive_bootstrap_seed
    assert analysis.derive_analysis_tie_digest is collision.derive_analysis_tie_digest
    assert (
        analysis.derive_stress_permutation_digest
        is collision.derive_stress_permutation_digest
    )

    content_hash = "a" * 64
    assert analysis._bootstrap_seed(
        replicate=3,
        family=analysis.TEST_FAMILIES[0],
        protocol_id=analysis.V021_PROTOCOL_ID,
        bootstrap_root=analysis.BOOTSTRAP_ROOT,
    ) == collision.derive_bootstrap_seed(
        analysis.V021_PROTOCOL_ID,
        seed_root=analysis.BOOTSTRAP_ROOT,
        replicate_index=3,
        family_id=analysis.TEST_FAMILIES[0],
    )
    assert analysis._permutation_digest(
        stress_permutation_root=analysis.STRESS_PERMUTATION_ROOT,
        permutation_index=4,
        family=analysis.TEST_FAMILIES[0],
        random_policy_hash=content_hash,
    ) == collision.derive_stress_permutation_digest(
        seed_root=analysis.STRESS_PERMUTATION_ROOT,
        permutation_index=4,
        family_id=analysis.TEST_FAMILIES[0],
        random_policy_content_sha256=content_hash,
    )
    for arm in analysis.RISK_SCORE_IDS:
        assert analysis.derive_analysis_tie_digest(
            analysis.V021_PROTOCOL_ID,
            arm=arm,
            content_hash=content_hash,
        ) == collision.derive_analysis_tie_digest(
            analysis.V021_PROTOCOL_ID,
            arm=arm,
            content_hash=content_hash,
        )


def test_random_rankings_are_deterministic_order_invariant_and_new() -> None:
    frame = _risk_frame()
    first = analysis.deterministic_random_rankings(
        frame,
        issue_count=24,
        rankings=19,
    )
    reordered = analysis.deterministic_random_rankings(
        frame.sample(frac=1.0, random_state=2026),
        issue_count=24,
        rankings=19,
    )
    predecessor = v015.deterministic_random_rankings(
        frame,
        issue_count=24,
        rankings=19,
    )

    pd.testing.assert_frame_equal(first, reordered)
    assert not first.equals(predecessor)
    assert first["ranking_index"].tolist() == list(range(19))
    assert set(first["issued_count"]) == {24}


def test_policy_surface_uses_small_declared_random_table() -> None:
    frame = _policy_frame()
    rankings = analysis.deterministic_random_rankings(
        frame,
        issue_count=20,
        rankings=17,
    )
    result = analysis.evaluate_policy_rankings(
        frame,
        rankings,
        issue_count=20,
        rankings=17,
    )

    assert tuple(result["score_id"]) == analysis.RISK_SCORE_IDS
    assert set(result["issued_count"]) == {20}
    assert np.isfinite(result["relative_risk_reduction"]).all()


def test_random_ranking_summary_accepts_only_declared_cardinality() -> None:
    frame = _risk_frame()
    frame["issued"] = False
    frame.loc[frame.index[:16], "issued"] = True
    rankings = analysis.deterministic_random_rankings(
        frame,
        issue_count=16,
        rankings=7,
    )
    result = analysis.risk_reduction_against_random_rankings(
        frame,
        rankings,
        issued_column="issued",
        expected_issue_count=16,
        rankings=7,
    )

    assert result.issued_count == 16
    with pytest.raises(analysis.V021AnalysisError, match="8 random rankings"):
        analysis.risk_reduction_against_random_rankings(
            frame,
            rankings,
            issued_column="issued",
            rankings=8,
        )


def test_bootstrap_is_deterministic_order_invariant_and_new() -> None:
    frame = _risk_frame()
    first = analysis.bootstrap_risk_reductions(
        frame,
        issue_count=24,
        resamples=13,
    )
    reordered = analysis.bootstrap_risk_reductions(
        frame.sample(frac=1.0, random_state=2607),
        issue_count=24,
        resamples=13,
    )
    predecessor = v015.bootstrap_risk_reductions(
        frame,
        protocol_id=V021_PROTOCOL_ID,
        issue_count=24,
        resamples=13,
    )

    pd.testing.assert_frame_equal(first, reordered)
    assert first["defined"].all()
    assert not first.equals(predecessor)


def test_bootstrap_summary_preserves_frozen_quantiles() -> None:
    replicates = pd.DataFrame(
        {
            "replicate_index": np.arange(8, dtype=int),
            "defined": [True] * 8,
            "eligible_count": [16] * 8,
            "random_expected_catastrophic_rate": [0.25] * 8,
            "visible_stress_risk_reduction": np.linspace(0.1, 0.8, 8),
            "visible_minus_prefix_increment": np.linspace(-0.2, 0.5, 8),
            "placebo_minus_prefix_increment": np.linspace(-0.1, 0.1, 8),
        }
    )
    result = analysis.bootstrap_gate_summary(
        replicates,
        issue_count=16,
        resamples=8,
    )

    assert result["visible_one_sided_95_lower"] == pytest.approx(
        np.quantile(
            replicates["visible_stress_risk_reduction"],
            0.05,
            method="linear",
        )
    )
    assert result["increment_one_sided_95_lower"] == pytest.approx(
        np.quantile(
            replicates["visible_minus_prefix_increment"],
            0.05,
            method="linear",
        )
    )
    assert result["placebo_two_sided_95_lower"] == pytest.approx(
        np.quantile(
            replicates["placebo_minus_prefix_increment"],
            0.025,
            method="linear",
        )
    )
    assert result["placebo_two_sided_95_upper"] == pytest.approx(
        np.quantile(
            replicates["placebo_minus_prefix_increment"],
            0.975,
            method="linear",
        )
    )


def test_stress_permutations_are_deterministic_order_invariant_and_new() -> None:
    frame, operating = _stress_frames()
    first = _run_stress(frame, operating)
    reordered_frame = frame.sample(frac=1.0, random_state=21)
    reordered_operating = operating.sample(frac=1.0, random_state=22)
    reordered = _run_stress(reordered_frame, reordered_operating)
    empty_coordinates = pd.DataFrame(columns=["partition", "cluster_id"])
    predecessor = v015.stress_permutation_metrics(
        frame,
        operating,
        empty_coordinates,
        empty_coordinates,
        visible_stress_state=_VisibleStressState(),
        protocol_id=V021_PROTOCOL_ID,
        random_expected_catastrophic_rate=0.4,
        observed_prefix_only_risk_reduction=0.1,
        issue_count=16,
        permutations=13,
    )

    pd.testing.assert_frame_equal(first, reordered)
    assert not first.equals(predecessor)
    assert set(first["issued_count"]) == {16}


def test_stress_summary_preserves_strict_99_percent_gate() -> None:
    passing = pd.DataFrame(
        {
            "permutation_index": np.arange(10, dtype=int),
            "issued_count": [16] * 10,
            "issued_catastrophic_rate": [0.25] * 10,
            "visible_minus_prefix_increment": np.zeros(10, dtype=float),
        }
    )
    passed = analysis.summarize_stress_permutations(
        passing,
        observed_visible_minus_prefix_increment=1.0,
        issue_count=16,
        permutations=10,
    )
    failing = passing.copy()
    failing.loc[0, "visible_minus_prefix_increment"] = 1.0
    failed = analysis.summarize_stress_permutations(
        failing,
        observed_visible_minus_prefix_increment=1.0,
        issue_count=16,
        permutations=10,
    )

    assert passed.strictly_lower_count == 10
    assert passed.gate_passed is True
    assert failed.strictly_lower_count == 9
    assert failed.gate_passed is False


def test_summaries_reject_index_count_and_rate_drift() -> None:
    frame = _risk_frame()
    frame["issued"] = False
    frame.loc[frame.index[:16], "issued"] = True
    rankings = analysis.deterministic_random_rankings(
        frame,
        issue_count=16,
        rankings=4,
    )

    duplicate_index = rankings.copy()
    duplicate_index.loc[3, "ranking_index"] = 2
    with pytest.raises(analysis.V021AnalysisError, match="continuous indices"):
        analysis.risk_reduction_against_random_rankings(
            frame,
            duplicate_index,
            issued_column="issued",
            expected_issue_count=16,
            rankings=4,
        )

    invalid_rate = rankings.copy()
    invalid_rate.loc[0, "issued_catastrophic_rate"] = 1.1
    with pytest.raises(analysis.V021AnalysisError, match="counts or rates"):
        analysis.evaluate_policy_rankings(
            _policy_frame(),
            invalid_rate,
            issue_count=16,
            rankings=4,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: analysis.deterministic_random_rankings(
            pd.DataFrame(),
            issue_count=1,
            protocol_id="wrong",
        ),
        lambda: analysis.deterministic_random_rankings(
            pd.DataFrame(),
            issue_count=1,
            random_root=analysis.RANDOM_ROOT + 1,
        ),
        lambda: analysis.bootstrap_risk_reductions(
            pd.DataFrame(),
            protocol_id="wrong",
        ),
        lambda: analysis.bootstrap_risk_reductions(
            pd.DataFrame(),
            bootstrap_root=analysis.BOOTSTRAP_ROOT + 1,
        ),
        lambda: analysis.stress_permutation_metrics(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            visible_stress_state=None,
            random_expected_catastrophic_rate=0.5,
            observed_prefix_only_risk_reduction=0.0,
            protocol_id="wrong",
        ),
        lambda: analysis.stress_permutation_metrics(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            visible_stress_state=None,
            random_expected_catastrophic_rate=0.5,
            observed_prefix_only_risk_reduction=0.0,
            stress_permutation_root=analysis.STRESS_PERMUTATION_ROOT + 1,
        ),
        lambda: analysis.rank_policy(
            pd.DataFrame(),
            arm="visible_stress",
            score_column="risk_visible_stress",
            predictor_hash_column="risk_hash_visible_stress",
            issue_count=1,
            protocol_id="wrong",
        ),
        lambda: analysis.rank_policy(
            pd.DataFrame(),
            arm="visible_stress",
            score_column="risk_visible_stress",
            predictor_hash_column="risk_hash_visible_stress",
            issue_count=1,
            random_root=analysis.RANDOM_ROOT + 1,
        ),
    ],
)
def test_wrong_v21_identity_or_root_fails_before_data_access(call: object) -> None:
    with pytest.raises(analysis.V021AnalysisError):
        call()


def test_concurrent_calls_and_exceptions_do_not_mutate_v015_state() -> None:
    frame = _risk_frame()
    before = (
        v015.RANDOM_ROOT,
        v015.BOOTSTRAP_ROOT,
        v015.STRESS_PERMUTATION_ROOT,
    )

    def run(seed: int) -> pd.DataFrame:
        reordered = frame.sample(frac=1.0, random_state=seed)
        return analysis.deterministic_random_rankings(
            reordered,
            issue_count=24,
            rankings=11,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(4)))
    for output in outputs[1:]:
        pd.testing.assert_frame_equal(outputs[0], output)
    with pytest.raises(analysis.V021AnalysisError):
        analysis.deterministic_random_rankings(
            frame,
            issue_count=24,
            random_root=v015.RANDOM_ROOT,
            rankings=2,
        )

    after = (
        v015.RANDOM_ROOT,
        v015.BOOTSTRAP_ROOT,
        v015.STRESS_PERMUTATION_ROOT,
    )
    assert after == before


def test_adapter_has_no_file_truth_or_synthetic_generator_capability() -> None:
    source = inspect.getsource(analysis)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assigned_attributes = [
        target
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
    ]

    assert imported_modules.isdisjoint({"os", "pathlib"})
    assert all("v016_generator" not in module for module in imported_from)
    assert all("truth" not in module for module in imported_from)
    assert called_names.isdisjoint(
        {
            "open",
            "read_csv",
            "read_json",
            "read_parquet",
            "open_truth_for_phase",
            "generate_protocol",
        }
    )
    assert assigned_attributes == []
    assert not hasattr(analysis, "Path")
    assert not hasattr(analysis, "open_truth_for_phase")
