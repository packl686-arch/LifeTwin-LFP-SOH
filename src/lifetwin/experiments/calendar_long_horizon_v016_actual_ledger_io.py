"""Outcome-free construction of the V2.1 actual-analysis hash ledger.

This module owns the only dependency from artifact content back to generation
coordinates.  Prediction modules must not import it.  The formal creator and
fresh-fit IO issuer import it lazily, recompute every test/audit record, and
then persist the resulting byte hash through the fit/model commitment chain.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    V015ArtifactError,
    predictor_content_hashes,
)
from lifetwin.experiments.calendar_long_horizon_v016_collision import (
    ANALYSIS_TIE_ARMS,
    ActualAnalysisContentRecord,
    V021CollisionError,
    _ordinary_identity,
    audit_formal_v021_generation_plan,
    bind_formal_v021_actual_analysis_hash_ledger,
    build_formal_plan_specs,
    derive_stream_seed,
    verify_actual_analysis_hash_ledger_commitment,
)
from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    V021ContractView,
)


class V021ActualLedgerIOError(ValueError):
    """Raised when generated label-free content cannot bind the formal ledger."""


_LABEL_INPUTS = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
)


def recompute_generation_plan_commitment_bytes_v021(
    view: V021ContractView,
) -> bytes:
    """Return the exact fully enumerated frozen generation-plan bytes."""

    try:
        return audit_formal_v021_generation_plan(view).canonical_bytes
    except V021CollisionError as exc:
        raise V021ActualLedgerIOError(
            "Formal generation-plan recomputation failed"
        ) from exc


def ordinary_family_lookup_v021(
    view: V021ContractView,
) -> dict[tuple[str, str], str]:
    """Map each formal test/audit opaque member ID to its frozen family."""

    try:
        current, _ = build_formal_plan_specs(view)
        roots = current.seed_root_map()
        lookup: dict[tuple[str, str], str] = {}
        for group in current.ordinary_groups:
            if group.partition not in {"test", "audit"}:
                continue
            for index in range(group.count):
                opaque_seed = derive_stream_seed(
                    current.protocol_id,
                    seed_root=roots[group.seed_root_name],
                    partition=group.partition,
                    family_id=group.family_id,
                    zero_based_index=index,
                    stream_name="opaque_id",
                )
                member_id, _ = _ordinary_identity(
                    current.protocol_id,
                    opaque_seed,
                )
                key = (group.partition, member_id)
                if key in lookup:
                    raise V021ActualLedgerIOError(
                        "Formal ordinary member identifiers are not unique"
                    )
                lookup[key] = group.family_id
    except V021CollisionError as exc:
        raise V021ActualLedgerIOError(
            "Formal ordinary identity derivation failed"
        ) from exc
    return lookup


def actual_analysis_content_records_v021(
    frames: Mapping[str, pd.DataFrame],
    *,
    view: V021ContractView,
) -> tuple[ActualAnalysisContentRecord, ...]:
    """Hash the exact predictor content for every formal test/audit member."""

    if set(frames) != set(_LABEL_INPUTS):
        raise V021ActualLedgerIOError("Actual-analysis source table registry changed")
    family_by_member = ordinary_family_lookup_v021(view)
    expected = set(family_by_member)
    grouped: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
    for filename in _LABEL_INPUTS:
        frame = frames[filename]
        selected = frame.loc[frame["partition"].isin(("test", "audit"))]
        groups = {
            (str(partition), str(member_id)): group.copy(deep=False)
            for (partition, member_id), group in selected.groupby(
                ["partition", "cluster_id"],
                sort=False,
                observed=True,
            )
        }
        if set(groups) != expected:
            raise V021ActualLedgerIOError(
                f"{filename} test/audit members differ from the formal plan"
            )
        grouped[filename] = groups

    arm_sources = {
        "prefix_only": "arm_a",
        "visible_stress": "arm_b",
        "placebo_8": "placebo",
        "arm_a_plus_s_plan": "arm_b",
        "strongest_single_feature": "arm_a",
        "planned_stress_only": "arm_b",
        "prefix_rmse_only": "arm_a",
        "v1_max_envelope_only": "arm_a",
        "center_sqrt_abs_difference_only": "arm_a",
    }
    if tuple(arm_sources) != ANALYSIS_TIE_ARMS:
        raise V021ActualLedgerIOError("Actual-analysis tie-arm mapping changed")

    records: list[ActualAnalysisContentRecord] = []
    for partition, member_id in sorted(expected):
        operating_rows = grouped["operating_pack.csv"][(partition, member_id)]
        if len(operating_rows) != 1:
            raise V021ActualLedgerIOError("Actual-analysis operating row is not unique")
        try:
            hashes = predictor_content_hashes(
                grouped["prefix_pack.csv"][(partition, member_id)],
                grouped["forecast_coordinates.csv"][(partition, member_id)],
                operating_rows.iloc[0],
                enforce_frozen_counts=True,
            )
            by_source = {
                "arm_a": hashes.arm_a,
                "arm_b": hashes.arm_b,
                "placebo": hashes.placebo,
            }
            records.append(
                ActualAnalysisContentRecord(
                    partition=partition,
                    family_id=family_by_member[(partition, member_id)],
                    member_id=member_id,
                    random_policy_content_sha256=hashes.random_policy,
                    predictor_content_hashes=tuple(
                        (arm, by_source[arm_sources[arm]]) for arm in ANALYSIS_TIE_ARMS
                    ),
                )
            )
        except (V015ArtifactError, V021CollisionError) as exc:
            raise V021ActualLedgerIOError(
                f"Actual-analysis content hashing failed for {member_id}"
            ) from exc
    return tuple(records)


def recompute_actual_analysis_hash_ledger_bytes_v021(
    frames: Mapping[str, pd.DataFrame],
    *,
    view: V021ContractView,
) -> bytes:
    """Return exact canonical bytes from all generated analysis inputs."""

    records = actual_analysis_content_records_v021(frames, view=view)
    try:
        return bind_formal_v021_actual_analysis_hash_ledger(
            records,
            view,
        ).canonical_bytes
    except V021CollisionError as exc:
        raise V021ActualLedgerIOError(
            "Formal actual-analysis hash-ledger recomputation failed"
        ) from exc


def verify_actual_analysis_hash_ledger_payload_v021(
    payload: Mapping[str, object],
    *,
    expected_byte_sha256: str,
    view: V021ContractView,
) -> None:
    """Structurally verify a detached formal actual-analysis commitment."""

    try:
        current, _ = build_formal_plan_specs(view)
        roots = current.seed_root_map()
        verify_actual_analysis_hash_ledger_commitment(
            payload,
            expected_byte_sha256=expected_byte_sha256,
            expected_protocol_id=current.protocol_id,
            expected_random_ranking_root=roots[current.random_ranking_root_name],
            expected_stress_permutation_root=roots[
                current.stress_permutation_root_name
            ],
        )
    except V021CollisionError as exc:
        raise V021ActualLedgerIOError(
            "Actual-analysis hash-ledger structure is invalid"
        ) from exc


__all__ = [
    "V021ActualLedgerIOError",
    "actual_analysis_content_records_v021",
    "ordinary_family_lookup_v021",
    "recompute_actual_analysis_hash_ledger_bytes_v021",
    "recompute_generation_plan_commitment_bytes_v021",
    "verify_actual_analysis_hash_ledger_payload_v021",
]
