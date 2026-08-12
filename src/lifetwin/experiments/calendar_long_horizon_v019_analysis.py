"""V2.4 stochastic analysis adapter.

The scientific estimands and gates remain those of V0.15/V2.  This module
reconstructs only the stochastic surfaces whose deterministic identity must
move to the V2.4 protocol and seed-root namespace.  It deliberately does not
mutate the predecessor analysis module.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_analysis import (
    RISK_SCORE_IDS,
    TEST_FAMILIES,
    TEST_ISSUE_COUNT,
    RiskReduction,
    StressPermutationSummary,
    V015AnalysisError,
    V015InconclusiveError,
    _aligned_operating_matrix,
    _finite_vector,
    _require_columns,
    _sha256_text,
    _strict_bool_series,
    risk_reduction,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    REAL_OPERATING_FIELDS,
)
from lifetwin.experiments.calendar_long_horizon_v019_collision import (
    ANALYSIS_TIE_ARMS,
    derive_analysis_tie_digest,
    derive_bootstrap_seed,
    derive_random_ranking_digest,
    derive_stress_permutation_digest,
)
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractView,
    resolve_contract_view,
)


RANDOM_RANKING_COUNT = 10_000
BOOTSTRAP_RESAMPLES = 5_000
STRESS_PERMUTATIONS = 10_000
_DEFAULT_PROTOCOL = resolve_contract_view(None).protocol
_DEFAULT_ROOTS = dict(_DEFAULT_PROTOCOL.seed_roots)
RANDOM_ROOT = _DEFAULT_ROOTS["random_rankings"]
BOOTSTRAP_ROOT = _DEFAULT_ROOTS["bootstrap"]
STRESS_PERMUTATION_ROOT = _DEFAULT_ROOTS["stress_permutations"]

_DECLARED_STOCHASTIC_ROOTS = {
    "random_rankings": RANDOM_ROOT,
    "bootstrap": BOOTSTRAP_ROOT,
    "stress_permutations": STRESS_PERMUTATION_ROOT,
}
if tuple(RISK_SCORE_IDS) != ANALYSIS_TIE_ARMS:
    raise RuntimeError("V2.4 analysis tie-arm registry drifted from V2")


class V024AnalysisError(V015AnalysisError):
    """Raised when a V2.4 analysis call drifts from its frozen identity."""


def _validate_identity(
    *,
    contract_view: V024ContractView | None,
    protocol_id: str | None,
    observed_root: int | None,
    root_name: str,
    operation: str,
) -> tuple[str, int]:
    contract = resolve_contract_view(contract_view)
    expected_protocol_id = contract.protocol.protocol_id
    expected_root = dict(contract.protocol.seed_roots)[root_name]
    actual_protocol_id = expected_protocol_id if protocol_id is None else protocol_id
    actual_root = expected_root if observed_root is None else observed_root
    if actual_protocol_id != expected_protocol_id:
        raise V024AnalysisError(f"{operation} protocol identity changed")
    if (
        isinstance(actual_root, bool)
        or not isinstance(actual_root, int)
        or actual_root != expected_root
    ):
        raise V024AnalysisError(f"{operation} seed root changed")
    return actual_protocol_id, actual_root


def _positive_count(value: int, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise V024AnalysisError(f"{context} must be a positive integer")
    return value


def _validate_zero_based_indices(
    frame: pd.DataFrame,
    *,
    column: str,
    expected_count: int,
    context: str,
) -> None:
    _require_columns(frame, {column}, context=context)
    series = frame[column]
    if series.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise V024AnalysisError(f"{context} indices are not exact integers")
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    expected = np.arange(expected_count, dtype=float)
    if (
        len(values) != expected_count
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or not np.array_equal(np.sort(values), expected)
    ):
        raise V024AnalysisError(
            f"{context} requires unique continuous indices 0..{expected_count - 1}"
        )


def _validate_issued_counts_and_rates(
    frame: pd.DataFrame,
    *,
    expected_issue_count: int | None,
    context: str,
) -> tuple[int, np.ndarray]:
    _require_columns(
        frame,
        {"issued_count", "issued_catastrophic_rate"},
        context=context,
    )
    counts = pd.to_numeric(frame["issued_count"], errors="coerce").to_numpy(float)
    rates = _finite_vector(
        frame["issued_catastrophic_rate"],
        context=f"{context} catastrophic rates",
    )
    if (
        not len(counts)
        or not np.isfinite(counts).all()
        or not np.equal(counts, np.floor(counts)).all()
        or np.any(counts < 1)
        or not np.all(counts == counts[0])
        or np.any((rates < 0.0) | (rates > 1.0))
    ):
        raise V024AnalysisError(f"{context} issued counts or rates are invalid")
    issued_count = int(counts[0])
    if expected_issue_count is not None and issued_count != expected_issue_count:
        raise V024AnalysisError(f"{context} uses the wrong issuance count")
    return issued_count, rates


def deterministic_random_rankings(
    trajectories: pd.DataFrame,
    *,
    issue_count: int,
    protocol_id: str | None = None,
    random_root: int | None = None,
    rankings: int = RANDOM_RANKING_COUNT,
    contract_view: V024ContractView | None = None,
) -> pd.DataFrame:
    """Construct same-count random rankings in the V2.4 random namespace."""
    protocol_id, random_root = _validate_identity(
        contract_view=contract_view,
        protocol_id=protocol_id,
        observed_root=random_root,
        root_name="random_rankings",
        operation="Random rankings",
    )
    _positive_count(issue_count, context="Random-ranking issue_count")
    _positive_count(rankings, context="Random-ranking count")
    _require_columns(
        trajectories,
        {
            "canonical_prefix_content_sha256",
            "hard_eligible_visible_stress",
            "catastrophic",
        },
        context="Random-ranking table",
    )
    eligible = trajectories.loc[
        _strict_bool_series(
            trajectories["hard_eligible_visible_stress"],
            context="hard_eligible_visible_stress",
        )
    ].copy()
    if len(eligible) < issue_count:
        raise V015InconclusiveError("Random ranking has too few eligible rows")
    if eligible["canonical_prefix_content_sha256"].duplicated().any():
        raise V024AnalysisError("Ordinary random-policy content must be unique")
    hashes = np.asarray(
        [
            _sha256_text(value, context="random-policy content hash")
            for value in eligible["canonical_prefix_content_sha256"]
        ],
        dtype="U64",
    )
    catastrophe = _strict_bool_series(
        eligible["catastrophic"], context="catastrophic"
    ).to_numpy()
    expected = float(catastrophe.mean())
    if expected <= 0.0:
        raise V015InconclusiveError(
            "Random-ranking eligible-pool catastrophic prevalence is zero"
        )

    records: list[dict[str, object]] = []
    for ranking_index in range(rankings):
        order = np.argsort(
            np.asarray(
                [
                    derive_random_ranking_digest(
                        seed_root=random_root,
                        ranking_index=ranking_index,
                        content_hash=value,
                    )
                    for value in hashes
                ],
                dtype="U64",
            ),
            kind="stable",
        )
        rate = float(catastrophe[order[:issue_count]].mean())
        records.append(
            {
                "ranking_index": ranking_index,
                "issued_count": issue_count,
                "issued_catastrophic_rate": rate,
                "analytic_random_expected_rate": expected,
                "relative_risk_reduction": 1.0 - rate / expected,
            }
        )
    return pd.DataFrame(records)


def rank_policy(
    frame: pd.DataFrame,
    *,
    arm: str,
    score_column: str,
    predictor_hash_column: str,
    issue_count: int,
    protocol_id: str | None = None,
    random_root: int | None = None,
    eligibility_column: str = "hard_eligible_visible_stress",
    contract_view: V024ContractView | None = None,
) -> pd.Series:
    """Return the frozen lowest-danger mask with the V2.4 protocol tie key."""
    protocol_id, _ = _validate_identity(
        contract_view=contract_view,
        protocol_id=protocol_id,
        observed_root=random_root,
        root_name="random_rankings",
        operation="Policy ranking",
    )
    _positive_count(issue_count, context="Policy issue_count")
    _require_columns(
        frame,
        {score_column, predictor_hash_column, eligibility_column},
        context=f"Ranking/{arm}",
    )
    if not frame.index.is_unique:
        raise V024AnalysisError("Policy ranking requires a unique row index")
    eligible = _strict_bool_series(
        frame[eligibility_column], context=eligibility_column
    )
    if int(eligible.sum()) < issue_count:
        raise V015InconclusiveError("Ranking has too few eligible rows")
    working = frame.loc[eligible, [score_column, predictor_hash_column]].copy()
    working["_validated_score"] = _finite_vector(
        working[score_column], context=f"Risk score/{arm}"
    )
    verified_hashes = [
        _sha256_text(value, context=f"Predictor content/{arm}")
        for value in working[predictor_hash_column]
    ]
    if len(set(verified_hashes)) != len(verified_hashes):
        raise V024AnalysisError(f"Ordinary predictor content is duplicated for {arm}")
    working["_tie_hash"] = [
        derive_analysis_tie_digest(
            protocol_id,
            arm=arm,
            content_hash=value,
        )
        for value in verified_hashes
    ]
    selected_index = working.sort_values(
        ["_validated_score", "_tie_hash"], kind="stable"
    ).index[:issue_count]
    result = pd.Series(False, index=frame.index, dtype=bool)
    result.loc[selected_index] = True
    return result


def risk_reduction_against_random_rankings(
    trajectories: pd.DataFrame,
    random_rankings: pd.DataFrame,
    *,
    issued_column: str,
    eligibility_column: str = "hard_eligible_visible_stress",
    expected_issue_count: int | None = None,
    rankings: int = RANDOM_RANKING_COUNT,
) -> RiskReduction:
    """Use the mean V2.4 same-count random-ranking risk as the point baseline."""
    _positive_count(rankings, context="Random-ranking count")
    _require_columns(
        random_rankings,
        {"ranking_index", "issued_count", "issued_catastrophic_rate"},
        context="Random-ranking metrics",
    )
    if len(random_rankings) != rankings:
        raise V024AnalysisError(f"All {rankings} random rankings are required")
    _validate_zero_based_indices(
        random_rankings,
        column="ranking_index",
        expected_count=rankings,
        context="Random-ranking metrics",
    )
    frozen_count, rates = _validate_issued_counts_and_rates(
        random_rankings,
        expected_issue_count=expected_issue_count,
        context="Random-ranking metrics",
    )
    point = risk_reduction(
        trajectories,
        issued_column=issued_column,
        eligibility_column=eligibility_column,
        expected_issue_count=frozen_count,
    )
    random_mean = float(np.mean(rates))
    if random_mean <= 0.0:
        raise V015InconclusiveError("Mean random-ranking risk is zero")
    return RiskReduction(
        issued_count=point.issued_count,
        issued_catastrophic_rate=point.issued_catastrophic_rate,
        random_expected_catastrophic_rate=random_mean,
        relative_risk_reduction=1.0 - point.issued_catastrophic_rate / random_mean,
    )


def evaluate_policy_rankings(
    trajectories: pd.DataFrame,
    random_rankings: pd.DataFrame,
    *,
    issue_count: int,
    protocol_id: str | None = None,
    random_root: int | None = None,
    rankings: int = RANDOM_RANKING_COUNT,
    contract_view: V024ContractView | None = None,
) -> pd.DataFrame:
    """Report all frozen heads against one V2.4 random-ranking baseline."""
    protocol_id, random_root = _validate_identity(
        contract_view=contract_view,
        protocol_id=protocol_id,
        observed_root=random_root,
        root_name="random_rankings",
        operation="Policy comparison",
    )
    _positive_count(issue_count, context="Policy issue_count")
    _positive_count(rankings, context="Random-ranking count")
    _require_columns(
        trajectories,
        {
            "catastrophic",
            "hard_eligible_visible_stress",
            *(f"risk_{score_id}" for score_id in RISK_SCORE_IDS),
            *(f"risk_hash_{score_id}" for score_id in RISK_SCORE_IDS),
        },
        context="Policy comparison",
    )
    _require_columns(
        random_rankings,
        {"ranking_index", "issued_count", "issued_catastrophic_rate"},
        context="Policy random baseline",
    )
    if len(random_rankings) != rankings:
        raise V024AnalysisError(
            f"Policy comparison requires {rankings} random rankings"
        )
    _validate_zero_based_indices(
        random_rankings,
        column="ranking_index",
        expected_count=rankings,
        context="Policy random baseline",
    )
    _, random_rates = _validate_issued_counts_and_rates(
        random_rankings,
        expected_issue_count=issue_count,
        context="Policy random baseline",
    )
    random_mean = float(np.mean(random_rates))
    if random_mean <= 0.0:
        raise V024AnalysisError("Policy random baseline is zero")
    eligible = _strict_bool_series(
        trajectories["hard_eligible_visible_stress"],
        context="policy eligibility",
    )
    catastrophic = _strict_bool_series(
        trajectories["catastrophic"], context="policy catastrophic"
    )
    rows: list[dict[str, object]] = []
    for score_id in RISK_SCORE_IDS:
        issued = rank_policy(
            trajectories,
            protocol_id=protocol_id,
            random_root=random_root,
            arm=score_id,
            score_column=f"risk_{score_id}",
            predictor_hash_column=f"risk_hash_{score_id}",
            issue_count=issue_count,
            contract_view=contract_view,
        )
        issued_rate = float(catastrophic[issued].mean())
        rows.append(
            {
                "score_id": score_id,
                "source_count": len(trajectories),
                "eligible_count": int(eligible.sum()),
                "issued_count": int(issued.sum()),
                "source_coverage": int(issued.sum()) / len(trajectories),
                "eligible_coverage": int(issued.sum()) / int(eligible.sum()),
                "issued_catastrophic_rate": issued_rate,
                "mean_random_issued_catastrophic_rate": random_mean,
                "relative_risk_reduction": 1.0 - issued_rate / random_mean,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_seed(
    *,
    replicate: int,
    family: str,
    protocol_id: str,
    bootstrap_root: int,
) -> int:
    return derive_bootstrap_seed(
        protocol_id,
        seed_root=bootstrap_root,
        replicate_index=replicate,
        family_id=family,
    )


def bootstrap_risk_reductions(
    trajectories: pd.DataFrame,
    *,
    protocol_id: str | None = None,
    bootstrap_root: int | None = None,
    issue_count: int = TEST_ISSUE_COUNT,
    resamples: int = BOOTSTRAP_RESAMPLES,
    families: Sequence[str] = TEST_FAMILIES,
    contract_view: V024ContractView | None = None,
) -> pd.DataFrame:
    """Run the frozen family-stratified bootstrap in the V2.4 namespace."""
    protocol_id, bootstrap_root = _validate_identity(
        contract_view=contract_view,
        protocol_id=protocol_id,
        observed_root=bootstrap_root,
        root_name="bootstrap",
        operation="Bootstrap",
    )
    _positive_count(issue_count, context="Bootstrap issue_count")
    _positive_count(resamples, context="Bootstrap resample count")
    if not families or len(set(families)) != len(families):
        raise V024AnalysisError("Bootstrap families must be unique and nonempty")
    required = {
        "truth_family",
        "canonical_prefix_content_sha256",
        "catastrophic",
        "hard_eligible_visible_stress",
        "risk_prefix_only",
        "risk_visible_stress",
        "risk_placebo_8",
        "risk_hash_prefix_only",
        "risk_hash_visible_stress",
        "risk_hash_placebo_8",
    }
    _require_columns(trajectories, required, context="Bootstrap table")
    family_frames: dict[str, pd.DataFrame] = {}
    for family in families:
        subset = trajectories.loc[trajectories["truth_family"].eq(family)].copy()
        if subset.empty:
            raise V024AnalysisError(f"Bootstrap family absent: {family}")
        subset = subset.sort_values(
            "canonical_prefix_content_sha256", kind="stable"
        ).reset_index(drop=True)
        if subset["canonical_prefix_content_sha256"].duplicated().any():
            raise V024AnalysisError(
                f"Bootstrap ordinary content is duplicated in {family}"
            )
        for score_id in ("prefix_only", "visible_stress", "placebo_8"):
            subset[f"_tie_{score_id}"] = [
                derive_analysis_tie_digest(
                    protocol_id,
                    arm=score_id,
                    content_hash=value,
                )
                for value in subset[f"risk_hash_{score_id}"]
            ]
        family_frames[family] = subset

    rows: list[dict[str, object]] = []
    for replicate in range(resamples):
        sampled: list[pd.DataFrame] = []
        for family_index, family in enumerate(families):
            source = family_frames[family]
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    _bootstrap_seed(
                        replicate=replicate,
                        family=family,
                        protocol_id=protocol_id,
                        bootstrap_root=bootstrap_root,
                    )
                )
            )
            indices = rng.integers(
                0,
                len(source),
                size=len(source),
                endpoint=False,
                dtype=np.int64,
            )
            draw = source.iloc[indices].copy()
            draw["_family_index"] = family_index
            draw["_occurrence_ordinal"] = np.arange(len(source), dtype=np.int64)
            sampled.append(draw)
        boot = pd.concat(sampled, ignore_index=True)
        eligible = _strict_bool_series(
            boot["hard_eligible_visible_stress"],
            context="bootstrap eligibility",
        )
        defined = bool(eligible.sum() >= issue_count)
        random_rate = float(
            _strict_bool_series(
                boot.loc[eligible, "catastrophic"],
                context="bootstrap catastrophic",
            ).mean()
        )
        defined = defined and math.isfinite(random_rate) and random_rate > 0.0
        record: dict[str, object] = {
            "replicate_index": replicate,
            "defined": defined,
            "eligible_count": int(eligible.sum()),
            "random_expected_catastrophic_rate": random_rate,
        }
        if not defined:
            record.update(
                {
                    "prefix_only_risk_reduction": math.nan,
                    "visible_stress_risk_reduction": math.nan,
                    "visible_minus_prefix_increment": math.nan,
                    "placebo_minus_prefix_increment": math.nan,
                }
            )
            rows.append(record)
            continue
        reductions: dict[str, float] = {}
        for score_id in ("prefix_only", "visible_stress", "placebo_8"):
            ranked = boot.loc[eligible].sort_values(
                [
                    f"risk_{score_id}",
                    f"_tie_{score_id}",
                    "_family_index",
                    "_occurrence_ordinal",
                ],
                kind="stable",
            )
            rate = float(
                _strict_bool_series(
                    ranked.iloc[:issue_count]["catastrophic"],
                    context="bootstrap issued catastrophic",
                ).mean()
            )
            reductions[score_id] = 1.0 - rate / random_rate
        record.update(
            {
                "prefix_only_risk_reduction": reductions["prefix_only"],
                "visible_stress_risk_reduction": reductions["visible_stress"],
                "visible_minus_prefix_increment": (
                    reductions["visible_stress"] - reductions["prefix_only"]
                ),
                "placebo_minus_prefix_increment": (
                    reductions["placebo_8"] - reductions["prefix_only"]
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_gate_summary(
    replicates: pd.DataFrame,
    *,
    issue_count: int = TEST_ISSUE_COUNT,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Mapping[str, float]:
    """Apply the unchanged bootstrap quantiles to the declared cardinality."""
    _positive_count(issue_count, context="Bootstrap issue_count")
    _positive_count(resamples, context="Bootstrap resample count")
    required = {
        "replicate_index",
        "defined",
        "eligible_count",
        "random_expected_catastrophic_rate",
        "visible_stress_risk_reduction",
        "visible_minus_prefix_increment",
        "placebo_minus_prefix_increment",
    }
    _require_columns(replicates, required, context="Bootstrap replicates")
    if len(replicates) != resamples:
        raise V024AnalysisError(f"Every one of {resamples} replicates is required")
    _validate_zero_based_indices(
        replicates,
        column="replicate_index",
        expected_count=resamples,
        context="Bootstrap replicates",
    )
    if not _strict_bool_series(
        replicates["defined"], context="bootstrap defined"
    ).all():
        raise V024AnalysisError("An undefined bootstrap makes gates inconclusive")
    eligible_counts = pd.to_numeric(
        replicates["eligible_count"],
        errors="coerce",
    ).to_numpy(float)
    random_rates = _finite_vector(
        replicates["random_expected_catastrophic_rate"],
        context="Bootstrap random expected rates",
    )
    if (
        not np.isfinite(eligible_counts).all()
        or not np.equal(eligible_counts, np.floor(eligible_counts)).all()
        or np.any(eligible_counts < issue_count)
        or np.any((random_rates <= 0.0) | (random_rates > 1.0))
    ):
        raise V024AnalysisError(
            "Bootstrap eligible counts or random expected rates are invalid"
        )
    visible = _finite_vector(
        replicates["visible_stress_risk_reduction"], context="Visible bootstrap"
    )
    increment = _finite_vector(
        replicates["visible_minus_prefix_increment"], context="Increment bootstrap"
    )
    placebo = _finite_vector(
        replicates["placebo_minus_prefix_increment"], context="Placebo bootstrap"
    )
    return {
        "visible_one_sided_95_lower": float(
            np.quantile(visible, 0.05, method="linear")
        ),
        "increment_one_sided_95_lower": float(
            np.quantile(increment, 0.05, method="linear")
        ),
        "placebo_two_sided_95_lower": float(
            np.quantile(placebo, 0.025, method="linear")
        ),
        "placebo_two_sided_95_upper": float(
            np.quantile(placebo, 0.975, method="linear")
        ),
    }


def _permutation_digest(
    *,
    stress_permutation_root: int,
    permutation_index: int,
    family: str,
    random_policy_hash: object,
) -> str:
    verified = _sha256_text(
        random_policy_hash, context="permutation random-policy hash"
    )
    return derive_stress_permutation_digest(
        seed_root=stress_permutation_root,
        permutation_index=permutation_index,
        family_id=family,
        random_policy_content_sha256=verified,
    )


def _permuted_boundary_tie_hashes(
    *,
    frame: pd.DataFrame,
    assigned_operating: np.ndarray,
    tie_positions: np.ndarray,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    protocol_id: str,
) -> list[str]:
    # This pure content-hash helper is the only V0.15 IO-layer import.  It
    # receives in-memory tables and performs no file or truth access.
    from lifetwin.experiments.calendar_long_horizon_v015_io import (
        predictor_content_hashes,
    )

    key = ["partition", "cluster_id"]
    operating_index = operating_pack.set_index(key)
    if not operating_index.index.is_unique:
        raise V024AnalysisError("Operating pack contains duplicate cluster keys")
    hashes: list[str] = []
    for position in tie_positions:
        row = frame.iloc[int(position)]
        partition = str(row["partition"])
        cluster_id = str(row["cluster_id"])
        prefix = prefix_pack.loc[
            prefix_pack["partition"].eq(partition)
            & prefix_pack["cluster_id"].eq(cluster_id)
        ]
        coordinates = forecast_coordinates.loc[
            forecast_coordinates["partition"].eq(partition)
            & forecast_coordinates["cluster_id"].eq(cluster_id)
        ]
        original = operating_index.loc[(partition, cluster_id)].to_dict()
        for field, value in zip(
            REAL_OPERATING_FIELDS,
            assigned_operating[int(position)],
            strict=True,
        ):
            original[field] = float(value)
        content = predictor_content_hashes(prefix, coordinates, original)
        hashes.append(
            derive_analysis_tie_digest(
                protocol_id,
                arm="visible_stress",
                content_hash=content.arm_b,
            )
        )
    return hashes


def stress_permutation_metrics(
    trajectories_and_features: pd.DataFrame,
    operating_pack: pd.DataFrame,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    *,
    visible_stress_state: Any,
    random_expected_catastrophic_rate: float,
    observed_prefix_only_risk_reduction: float,
    protocol_id: str | None = None,
    stress_permutation_root: int | None = None,
    issue_count: int = TEST_ISSUE_COUNT,
    permutations: int = STRESS_PERMUTATIONS,
    contract_view: V024ContractView | None = None,
) -> pd.DataFrame:
    """Jointly permute each eight-field stress block under V2.4 identity."""
    protocol_id, stress_permutation_root = _validate_identity(
        contract_view=contract_view,
        protocol_id=protocol_id,
        observed_root=stress_permutation_root,
        root_name="stress_permutations",
        operation="Stress permutation",
    )
    _positive_count(issue_count, context="Stress-permutation issue_count")
    _positive_count(permutations, context="Stress-permutation count")
    prefix_feature_names = tuple(visible_stress_state.feature_names)[
        : -len(REAL_OPERATING_FIELDS)
    ]
    expected_feature_names = prefix_feature_names + REAL_OPERATING_FIELDS
    if tuple(visible_stress_state.feature_names) != expected_feature_names:
        raise V024AnalysisError(
            "Visible-stress state does not end in the frozen operating fields"
        )
    required = {
        "partition",
        "cluster_id",
        "truth_family",
        "canonical_prefix_content_sha256",
        "hard_eligible_visible_stress",
        "catastrophic",
        *prefix_feature_names,
    }
    _require_columns(trajectories_and_features, required, context="Stress permutation")
    frame = trajectories_and_features.loc[
        trajectories_and_features["partition"].eq("test")
    ].copy()
    if frame.empty or frame["cluster_id"].duplicated().any():
        raise V024AnalysisError("Stress permutation test pool is invalid")
    if (
        not math.isfinite(random_expected_catastrophic_rate)
        or random_expected_catastrophic_rate <= 0.0
        or not math.isfinite(observed_prefix_only_risk_reduction)
    ):
        raise V024AnalysisError("Stress permutation reference metrics are invalid")

    random_hashes = [
        _sha256_text(value, context="permutation random-policy hash")
        for value in frame["canonical_prefix_content_sha256"]
    ]
    if len(set(random_hashes)) != len(random_hashes):
        raise V024AnalysisError("Permutation ordinary content is duplicated")
    eligible = _strict_bool_series(
        frame["hard_eligible_visible_stress"],
        context="permutation eligibility",
    ).to_numpy()
    if int(eligible.sum()) < issue_count:
        raise V015InconclusiveError("Permutation pool has too few eligible rows")
    catastrophic = _strict_bool_series(
        frame["catastrophic"], context="permutation catastrophic"
    ).to_numpy()
    prefix_features = (
        frame.loc[:, prefix_feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    if not np.isfinite(prefix_features).all():
        raise V024AnalysisError("Permutation prefix features are nonfinite")
    operating_values, _ = _aligned_operating_matrix(frame, operating_pack)
    family_positions: dict[str, np.ndarray] = {}
    for family in TEST_FAMILIES:
        positions = np.flatnonzero(frame["truth_family"].eq(family).to_numpy())
        if not len(positions):
            raise V024AnalysisError(f"Permutation family is absent: {family}")
        family_positions[family] = np.asarray(
            sorted(positions, key=lambda position: random_hashes[int(position)]),
            dtype=np.int64,
        )

    records: list[dict[str, object]] = []
    for permutation_index in range(permutations):
        assigned = np.empty_like(operating_values)
        for family in TEST_FAMILIES:
            recipients = family_positions[family]
            donors = sorted(
                recipients,
                key=lambda position: _permutation_digest(
                    stress_permutation_root=stress_permutation_root,
                    permutation_index=permutation_index,
                    family=family,
                    random_policy_hash=random_hashes[int(position)],
                ),
            )
            assigned[recipients] = operating_values[np.asarray(donors, dtype=np.int64)]
        feature_matrix = np.concatenate((prefix_features, assigned), axis=1)
        scores = np.asarray(
            visible_stress_state.decision_function(feature_matrix), dtype=float
        )
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise V024AnalysisError("Permuted visible-stress score is invalid")
        eligible_positions = np.flatnonzero(eligible)
        eligible_scores = scores[eligible_positions]
        cutoff = float(np.partition(eligible_scores, issue_count - 1)[issue_count - 1])
        below = eligible_positions[eligible_scores < cutoff]
        boundary = eligible_positions[eligible_scores == cutoff]
        needed = issue_count - len(below)
        if needed < 1 or needed > len(boundary):
            raise V024AnalysisError("Permutation cutoff reconstruction failed")
        if len(boundary) > needed:
            tie_hashes = _permuted_boundary_tie_hashes(
                frame=frame,
                assigned_operating=assigned,
                tie_positions=boundary,
                prefix_pack=prefix_pack,
                forecast_coordinates=forecast_coordinates,
                operating_pack=operating_pack,
                protocol_id=protocol_id,
            )
            chosen_order = np.argsort(
                np.asarray(tie_hashes, dtype="U64"), kind="stable"
            )
            boundary = boundary[chosen_order[:needed]]
        selected = np.concatenate((below, boundary[:needed]))
        issued_rate = float(catastrophic[selected].mean())
        visible_reduction = 1.0 - issued_rate / random_expected_catastrophic_rate
        records.append(
            {
                "permutation_index": permutation_index,
                "issued_count": len(selected),
                "issued_catastrophic_rate": issued_rate,
                "visible_stress_risk_reduction": visible_reduction,
                "visible_minus_prefix_increment": (
                    visible_reduction - observed_prefix_only_risk_reduction
                ),
            }
        )
    return pd.DataFrame(records)


def summarize_stress_permutations(
    metrics: pd.DataFrame,
    *,
    observed_visible_minus_prefix_increment: float,
    issue_count: int = TEST_ISSUE_COUNT,
    permutations: int = STRESS_PERMUTATIONS,
) -> StressPermutationSummary:
    """Apply the unchanged strict 99% stress-permutation gate."""
    _positive_count(issue_count, context="Stress-permutation issue_count")
    _positive_count(permutations, context="Stress-permutation count")
    _require_columns(
        metrics,
        {
            "permutation_index",
            "issued_count",
            "issued_catastrophic_rate",
            "visible_minus_prefix_increment",
        },
        context="Stress permutation metrics",
    )
    if len(metrics) != permutations:
        raise V024AnalysisError(f"All {permutations} stress permutations are required")
    _validate_zero_based_indices(
        metrics,
        column="permutation_index",
        expected_count=permutations,
        context="Stress permutation metrics",
    )
    _validate_issued_counts_and_rates(
        metrics,
        expected_issue_count=issue_count,
        context="Stress permutation metrics",
    )
    values = _finite_vector(
        metrics["visible_minus_prefix_increment"],
        context="Stress permutation increments",
    )
    if not math.isfinite(observed_visible_minus_prefix_increment):
        raise V024AnalysisError("Observed stress increment is nonfinite")
    lower_count = int(np.sum(values < observed_visible_minus_prefix_increment))
    return StressPermutationSummary(
        permutation_count=len(values),
        observed_visible_minus_prefix_increment=(
            observed_visible_minus_prefix_increment
        ),
        strictly_lower_count=lower_count,
        strictly_lower_fraction=lower_count / len(values),
        gate_passed=lower_count >= math.ceil(0.99 * permutations),
    )
