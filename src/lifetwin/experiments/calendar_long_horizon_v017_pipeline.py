"""V2.2 identity adapter for the frozen V2 label-free numerical pipeline.

The inherited V2 implementation hard-codes its protocol identity.  This module
validates V2.2 inputs, translates identity only in memory, reuses the frozen
numerical core, and then rebuilds every identity-sensitive V2.2 output.  It has
no truth, generator, fitting, filesystem, or scoring capability.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    canonicalize_frame,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    rank_for_issuance,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    PRIMARY_ISSUE_COUNTS,
    FrozenLabelFreeState,
    LabelFreePipelineResult,
    PrimaryArmRanking,
    V015PipelineError,
    recompute_label_free_pipeline,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID as V2_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v017_protocol import (
    V022_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v017_partition import (
    PARTITION_COUNTS,
    ValidatedPartitionView,
    canonicalize_partition_output,
    consume_partition_frames,
)
from lifetwin.experiments.calendar_long_horizon_v017_provenance import (
    V022CommittedModelStateEnvelope,
    V022ProvenanceError,
    _extract_label_free_state_for_formal_v022,
)


_INPUT_FILENAMES = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
)
_PRIMARY_SCORE_IDS = ("prefix_only", "visible_stress")


class V022PipelineError(ValueError):
    """Raised when the V2.2 label-free adapter cannot prove its boundaries."""


def _canonical_input(
    frame: pd.DataFrame,
    *,
    filename: str,
    contract: FrozenArtifactContract,
    formal: bool,
) -> pd.DataFrame:
    try:
        return canonicalize_frame(
            frame,
            contract.csv_schema(filename),
            contract,
            formal=formal,
        )
    except V015ArtifactError as exc:
        raise V022PipelineError(str(exc)) from exc


def _translate_protocol_column(
    frame: pd.DataFrame,
    *,
    source: str,
    destination: str,
) -> pd.DataFrame:
    if "protocol_id" not in frame.columns:
        raise V022PipelineError("An identity-bearing table lacks protocol_id")
    if set(frame["protocol_id"].astype(str)) != {source}:
        raise V022PipelineError("A table contains an unexpected protocol identity")
    translated = frame.copy(deep=False)
    translated["protocol_id"] = destination
    return translated


def _tie_hash_v022(arm: str, content_hash: str) -> str:
    material = f"{V022_PROTOCOL_ID}|{arm}|{content_hash}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def rank_primary_arms_v022(
    *,
    prefix_only_scores: Sequence[float],
    visible_stress_scores: Sequence[float],
    prefix_only_hashes: Sequence[str],
    visible_stress_hashes: Sequence[str],
    hard_eligible: Sequence[bool],
    issue_count: int,
) -> PrimaryArmRanking:
    """Apply the inherited ranking rule with the V2.2 tie-hash domain."""

    prefix_scores = np.asarray(prefix_only_scores, dtype=np.float64)
    visible_scores = np.asarray(visible_stress_scores, dtype=np.float64)
    eligible_raw = np.asarray(hard_eligible, dtype=object)
    hashes_a = tuple(str(value) for value in prefix_only_hashes)
    hashes_b = tuple(str(value) for value in visible_stress_hashes)
    size = prefix_scores.size
    if (
        prefix_scores.ndim != 1
        or visible_scores.shape != (size,)
        or eligible_raw.shape != (size,)
        or len(hashes_a) != size
        or len(hashes_b) != size
    ):
        raise V022PipelineError("Primary ranking inputs have inconsistent lengths")
    if any(not isinstance(value, (bool, np.bool_)) for value in eligible_raw):
        raise V022PipelineError("hard_eligible must contain strict booleans")
    if isinstance(issue_count, bool) or not isinstance(issue_count, int):
        raise V022PipelineError("issue_count must be a strict integer")

    eligible = eligible_raw.astype(bool)
    indices = np.flatnonzero(eligible)
    if issue_count < 0 or issue_count > len(indices):
        raise V022PipelineError(
            "The common hard-eligibility pool is smaller than the issue count"
        )
    if (
        not np.isfinite(prefix_scores[indices]).all()
        or not np.isfinite(visible_scores[indices]).all()
    ):
        raise V022PipelineError("An eligible primary risk score is nonfinite")

    ranking_a = rank_for_issuance(
        prefix_scores[indices],
        tuple(_tie_hash_v022("prefix_only", hashes_a[index]) for index in indices),
        issue_count,
    )
    ranking_b = rank_for_issuance(
        visible_scores[indices],
        tuple(_tie_hash_v022("visible_stress", hashes_b[index]) for index in indices),
        issue_count,
    )
    ranks_a: list[int | None] = [None] * size
    ranks_b: list[int | None] = [None] * size
    issued_a = [False] * size
    issued_b = [False] * size
    for local, global_index in enumerate(indices):
        index = int(global_index)
        ranks_a[index] = ranking_a.ranks[local]
        ranks_b[index] = ranking_b.ranks[local]
        issued_a[index] = ranking_a.issued[local]
        issued_b[index] = ranking_b.issued[local]
    return PrimaryArmRanking(
        prefix_only_ranks=tuple(ranks_a),
        visible_stress_ranks=tuple(ranks_b),
        prefix_only_issued=tuple(issued_a),
        visible_stress_issued=tuple(issued_b),
    )


def _suppress_ineligible_probabilities(
    risk_bundle: pd.DataFrame,
    feature_bundle: pd.DataFrame,
) -> pd.DataFrame:
    eligibility = feature_bundle.set_index(["partition", "cluster_id"])["hard_eligible"]
    result = risk_bundle.copy(deep=True)
    primary = result["score_id"].isin(_PRIMARY_SCORE_IDS)
    keys = pd.MultiIndex.from_frame(result.loc[:, ["partition", "cluster_id"]])
    try:
        eligible = eligibility.reindex(keys).to_numpy(dtype=bool)
    except (KeyError, TypeError, ValueError) as exc:
        raise V022PipelineError("Risk rows do not align with eligibility rows") from exc
    if eligibility.index.has_duplicates or eligibility.reindex(keys).isna().any():
        raise V022PipelineError("Risk rows do not align with eligibility rows")
    result.loc[primary & ~eligible, "calibrated_catastrophic_probability"] = np.nan
    eligible_primary = primary & eligible
    probabilities = pd.to_numeric(
        result.loc[eligible_primary, "calibrated_catastrophic_probability"],
        errors="coerce",
    ).to_numpy(float)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise V022PipelineError("An eligible primary calibrated probability is invalid")
    return result


def _rebuild_decision_bundle(
    *,
    feature_bundle: pd.DataFrame,
    risk_bundle: pd.DataFrame,
    content_bundle: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for partition, raw_features in feature_bundle.groupby("partition", sort=True):
        features = raw_features.sort_values(
            "cluster_id",
            kind="stable",
        ).reset_index(drop=True)
        risks = risk_bundle.loc[risk_bundle["partition"].eq(partition)].pivot(
            index="cluster_id",
            columns="score_id",
            values="raw_risk_score",
        )
        contents = content_bundle.loc[
            content_bundle["partition"].eq(partition)
        ].set_index("cluster_id")
        cluster_ids = features["cluster_id"].astype(str).tolist()
        if set(risks.index.astype(str)) != set(cluster_ids) or set(
            contents.index.astype(str)
        ) != set(cluster_ids):
            raise V022PipelineError("Decision inputs contain different cluster sets")
        risks = risks.loc[cluster_ids]
        contents = contents.loc[cluster_ids]
        if partition in PRIMARY_ISSUE_COUNTS:
            ranking = rank_primary_arms_v022(
                prefix_only_scores=risks["prefix_only"].to_numpy(float),
                visible_stress_scores=risks["visible_stress"].to_numpy(float),
                prefix_only_hashes=contents["arm_a_content_sha256"].astype(str),
                visible_stress_hashes=contents["arm_b_content_sha256"].astype(str),
                hard_eligible=features["hard_eligible"].tolist(),
                issue_count=PRIMARY_ISSUE_COUNTS[str(partition)],
            )
        else:
            empty_ranks: tuple[int | None, ...] = (None,) * len(features)
            empty_issued = (False,) * len(features)
            ranking = PrimaryArmRanking(
                prefix_only_ranks=empty_ranks,
                visible_stress_ranks=empty_ranks,
                prefix_only_issued=empty_issued,
                visible_stress_issued=empty_issued,
            )
        for index, row in features.iterrows():
            cluster_id = str(row["cluster_id"])
            for arm, ranks, issued, hash_column in (
                (
                    "prefix_only",
                    ranking.prefix_only_ranks,
                    ranking.prefix_only_issued,
                    "arm_a_content_sha256",
                ),
                (
                    "visible_stress",
                    ranking.visible_stress_ranks,
                    ranking.visible_stress_issued,
                    "arm_b_content_sha256",
                ),
            ):
                records.append(
                    {
                        "protocol_id": V022_PROTOCOL_ID,
                        "partition": str(partition),
                        "cluster_id": cluster_id,
                        "arm": arm,
                        "raw_risk_score": float(risks.loc[cluster_id, arm]),
                        "hard_eligible": bool(row["hard_eligible"]),
                        "issuance_rank": ranks[index],
                        "issued": issued[index],
                        "abstention_reasons": str(row["abstention_reasons"]),
                        "canonical_predictor_content_sha256": str(
                            contents.loc[cluster_id, hash_column]
                        ),
                    }
                )
    return pd.DataFrame(records)


def _recompute_label_free_pipeline_with_state_v022(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    member_fit_diagnostics: pd.DataFrame,
    member_forecast_bundle: pd.DataFrame,
    state: FrozenLabelFreeState,
    contract: FrozenArtifactContract,
    formal: bool,
) -> LabelFreePipelineResult:
    """Internal numerical adapter after provenance has released its state."""

    if contract.protocol_id != V022_PROTOCOL_ID:
        raise V022PipelineError("The artifact contract is not V2.2")
    supplied = (
        prefix_pack,
        forecast_coordinates,
        operating_pack,
        member_fit_diagnostics,
        member_forecast_bundle,
    )
    canonical = tuple(
        _canonical_input(
            frame,
            filename=filename,
            contract=contract,
            formal=formal,
        )
        for frame, filename in zip(supplied, _INPUT_FILENAMES, strict=True)
    )
    translated = tuple(
        _translate_protocol_column(
            frame,
            source=V022_PROTOCOL_ID,
            destination=V2_PROTOCOL_ID,
        )
        for frame in canonical
    )
    try:
        inherited = recompute_label_free_pipeline(
            prefix_pack=translated[0],
            forecast_coordinates=translated[1],
            operating_pack=translated[2],
            member_fit_diagnostics=translated[3],
            member_forecast_bundle=translated[4],
            state=state,
        )
    except V015PipelineError as exc:
        raise V022PipelineError(
            "The inherited label-free numerical pipeline rejected V2.2 inputs"
        ) from exc

    prediction = _translate_protocol_column(
        inherited.prediction_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    features = _translate_protocol_column(
        inherited.feature_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    risk = _translate_protocol_column(
        inherited.primary_risk_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    contents = _translate_protocol_column(
        inherited.predictor_content_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    risk = _suppress_ineligible_probabilities(risk, features)
    decisions = _rebuild_decision_bundle(
        feature_bundle=features,
        risk_bundle=risk,
        content_bundle=contents,
    )

    prediction = _canonical_input(
        prediction,
        filename="prediction_bundle.csv",
        contract=contract,
        formal=formal,
    )
    risk = _canonical_input(
        risk,
        filename="risk_bundle.csv",
        contract=contract,
        formal=formal,
    )
    decisions = _canonical_input(
        decisions,
        filename="decision_bundle.csv",
        contract=contract,
        formal=formal,
    )
    return LabelFreePipelineResult(
        prediction_bundle=prediction,
        feature_bundle=features.sort_values(
            ["partition", "cluster_id"],
            kind="stable",
        ).reset_index(drop=True),
        primary_risk_bundle=risk,
        decision_bundle=decisions,
        predictor_content_bundle=contents.sort_values(
            ["partition", "cluster_id"],
            kind="stable",
        ).reset_index(drop=True),
    )


def recompute_label_free_pipeline_v022(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    member_fit_diagnostics: pd.DataFrame,
    member_forecast_bundle: pd.DataFrame,
    model_state_envelope: V022CommittedModelStateEnvelope,
    contract: FrozenArtifactContract,
) -> LabelFreePipelineResult:
    """Run V2.2 prediction only from a codec-validated model capability."""

    if contract.protocol_id != V022_PROTOCOL_ID:
        raise V022PipelineError("The artifact contract is not V2.2")
    try:
        state = _extract_label_free_state_for_formal_v022(
            model_state_envelope,
            config_sha256=contract.config_byte_sha256,
        )
    except V022ProvenanceError as exc:
        raise V022PipelineError(str(exc)) from exc
    return _recompute_label_free_pipeline_with_state_v022(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        operating_pack=operating_pack,
        member_fit_diagnostics=member_fit_diagnostics,
        member_forecast_bundle=member_forecast_bundle,
        state=state,
        contract=contract,
        formal=True,
    )


def recompute_validated_partition_with_state_v022(
    partition_view: ValidatedPartitionView,
    *,
    state: FrozenLabelFreeState,
    contract: FrozenArtifactContract,
) -> LabelFreePipelineResult:
    """Run the inherited numerical core from an exact partition capability.

    This formal path has no boolean validation switch. Inputs are consumed only
    after their mutation guards and derived partition contracts pass, and the
    three persisted output schemas are derived from the same frozen contract.
    """

    if type(state) is not FrozenLabelFreeState:
        raise V022PipelineError("Partition state has the wrong exact type")
    if contract.protocol_id != V022_PROTOCOL_ID:
        raise V022PipelineError("The artifact contract is not V2.2")
    frames = consume_partition_frames(partition_view, contract=contract)
    partition = partition_view.partition
    counts = PARTITION_COUNTS[partition]
    canonical = tuple(frames[name] for name in _INPUT_FILENAMES)
    translated = tuple(
        _translate_protocol_column(
            frame,
            source=V022_PROTOCOL_ID,
            destination=V2_PROTOCOL_ID,
        )
        for frame in canonical
    )
    try:
        inherited = recompute_label_free_pipeline(
            prefix_pack=translated[0],
            forecast_coordinates=translated[1],
            operating_pack=translated[2],
            member_fit_diagnostics=translated[3],
            member_forecast_bundle=translated[4],
            state=state,
        )
    except V015PipelineError as exc:
        raise V022PipelineError(
            "The inherited numerical core rejected a validated V2.2 partition"
        ) from exc

    prediction = _translate_protocol_column(
        inherited.prediction_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    features = _translate_protocol_column(
        inherited.feature_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    risk = _translate_protocol_column(
        inherited.primary_risk_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    contents = _translate_protocol_column(
        inherited.predictor_content_bundle,
        source=V2_PROTOCOL_ID,
        destination=V022_PROTOCOL_ID,
    )
    risk = _suppress_ineligible_probabilities(risk, features)
    decisions = _rebuild_decision_bundle(
        feature_bundle=features,
        risk_bundle=risk,
        content_bundle=contents,
    )
    cluster_count = counts["clusters"]
    prediction = canonicalize_partition_output(
        prediction,
        filename="prediction_bundle.csv",
        partition=partition,
        required_rows=cluster_count * 8,
        contract=contract,
    )
    risk = canonicalize_partition_output(
        risk,
        filename="risk_bundle.csv",
        partition=partition,
        required_rows=cluster_count * 9,
        contract=contract,
    )
    decisions = canonicalize_partition_output(
        decisions,
        filename="decision_bundle.csv",
        partition=partition,
        required_rows=cluster_count * 2,
        contract=contract,
    )
    return LabelFreePipelineResult(
        prediction_bundle=prediction,
        feature_bundle=features.sort_values(
            ["partition", "cluster_id"], kind="stable"
        ).reset_index(drop=True),
        primary_risk_bundle=risk,
        decision_bundle=decisions,
        predictor_content_bundle=contents.sort_values(
            ["partition", "cluster_id"], kind="stable"
        ).reset_index(drop=True),
    )


__all__ = [
    "V022PipelineError",
    "rank_primary_arms_v022",
    "recompute_label_free_pipeline_v022",
    "recompute_validated_partition_with_state_v022",
]
