"""One-shot, post-freeze data generation for the frozen V0.15 protocol.

The public entry point in this module intentionally accepts only two physical
output roots.  Seeds, configuration paths, partitions, families, and counts
are loaded from the byte-committed V2 protocol and cannot be overridden.

Importing this module never consumes a formal seed.  Unit tests exercise only
the hand-fixture aggregation, collision, and write-preflight helpers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_collision import (
    CollisionLedger,
    GenerationCollisionSummary,
    V015CollisionError,
    audit_generation_coordinate_plan,
)
from lifetwin.experiments.calendar_long_horizon_v015_environment import (
    FormalEnvironmentIdentity,
    verify_formal_environment,
)
from lifetwin.experiments.calendar_long_horizon_v015_firewall import (
    FormalAttemptIdentity,
    V015FirewallError,
    append_phase_error_without_masking,
    append_formal_exposure_event,
    validate_formal_exposure_log,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    ArtifactMetadata,
    DEFAULT_V2_CONFIG_PATH,
    FrozenArtifactContract,
    V015ArtifactError,
    assert_separate_truth_roots,
    canonical_csv_bytes,
    canonical_json_bytes,
    canonicalize_frame,
    load_artifact_contract,
    predictor_content_payloads,
    validate_sealed_truth_bundle,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FORECAST_COORDINATE_COLUMNS,
    INTRINSIC_MATCHED_PARTITION,
    MATCHED_PAIR_COLUMNS,
    OPERATING_COLUMNS,
    ORDINARY_PARTITIONS,
    PREFIX_COLUMNS,
    STRESS_PLAN_MATCHED_PARTITION,
    TRUTH_COLUMNS,
    GeneratedMemberPacks,
    MatchedPairPacks,
    ValidatedV015Protocol,
    generate_cluster_packs,
    generate_intrinsic_matched_pair,
    generate_operating_covariates,
    generate_stress_plan_matched_pair,
    load_frozen_protocol_config,
    sample_truth_spec,
    validate_unique_stream_seeds,
)


LABEL_FREE_CSV_FILENAMES = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
)
TRUTH_COMMITMENT_FILENAME = "truth_commitments.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class V015GenerationError(ValueError):
    """Raised when a generated V0.15 bundle is not exactly the frozen design."""


@dataclass(frozen=True)
class OrdinaryPackRecord:
    """One ordinary generated member and its frozen generation coordinate."""

    partition: str
    family_id: str
    zero_based_index: int
    packs: GeneratedMemberPacks


@dataclass(frozen=True)
class GeneratedArtifactFrames:
    """All generated data before canonical serialization."""

    label_free: Mapping[str, pd.DataFrame]
    sealed: Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class PreparedGenerationArtifacts:
    """Fully validated canonical bytes, ready for exclusive creation."""

    label_free_bytes: Mapping[str, bytes]
    sealed_bytes: Mapping[str, bytes]
    row_counts: Mapping[str, int]
    truth_commitment_payload: Mapping[str, object]


@dataclass(frozen=True)
class WrittenGenerationArtifacts:
    """Metadata returned after every prepared byte string is written."""

    generation_collision_summary: GenerationCollisionSummary
    label_free_metadata: tuple[ArtifactMetadata, ...]
    sealed_metadata: tuple[ArtifactMetadata, ...]
    truth_commitment_byte_sha256: str


def _cluster_coordinates(
    frame: pd.DataFrame,
    *,
    context: str,
) -> set[tuple[str, str]]:
    required = {"partition", "cluster_id"}
    if frame.empty or not required.issubset(frame.columns):
        raise V015GenerationError(
            f"{context} must contain nonempty partition/cluster coordinates"
        )
    if frame.loc[:, ["partition", "cluster_id"]].isna().any().any():
        raise V015GenerationError(f"{context} contains a missing cluster coordinate")
    return {
        (str(partition), str(cluster_id))
        for partition, cluster_id in frame.loc[
            :, ["partition", "cluster_id"]
        ].itertuples(index=False, name=None)
    }


def _validate_pack_alignment(
    packs: GeneratedMemberPacks,
    *,
    context: str,
) -> set[tuple[str, str]]:
    expected_columns = (
        (packs.prefix_pack, PREFIX_COLUMNS, "prefix"),
        (
            packs.forecast_coordinates,
            FORECAST_COORDINATE_COLUMNS,
            "forecast coordinates",
        ),
        (packs.operating_pack, OPERATING_COLUMNS, "operating"),
        (packs.truth_pack, TRUTH_COLUMNS, "truth"),
    )
    coordinate_sets: list[set[tuple[str, str]]] = []
    for frame, columns, label in expected_columns:
        if tuple(frame.columns) != tuple(columns):
            raise V015GenerationError(
                f"{context} {label} columns differ from the frozen schema"
            )
        coordinate_sets.append(
            _cluster_coordinates(frame, context=f"{context} {label}")
        )
    if any(item != coordinate_sets[0] for item in coordinate_sets[1:]):
        raise V015GenerationError(
            f"{context} prefix, forecast, operating, and truth members differ"
        )
    return coordinate_sets[0]


def concatenate_member_packs(
    packs: Sequence[GeneratedMemberPacks],
) -> GeneratedMemberPacks:
    """Aggregate hand-built or generated packs after local alignment checks."""

    items = tuple(packs)
    if not items:
        raise V015GenerationError("At least one member pack is required")
    observed: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        coordinates = _validate_pack_alignment(item, context=f"member pack {index}")
        overlap = observed.intersection(coordinates)
        if overlap:
            raise V015GenerationError(
                "A cluster coordinate occurs in more than one member pack"
            )
        observed.update(coordinates)
    return GeneratedMemberPacks(
        prefix_pack=pd.concat(
            [item.prefix_pack for item in items],
            ignore_index=True,
        ).loc[:, PREFIX_COLUMNS],
        forecast_coordinates=pd.concat(
            [item.forecast_coordinates for item in items],
            ignore_index=True,
        ).loc[:, FORECAST_COORDINATE_COLUMNS],
        operating_pack=pd.concat(
            [item.operating_pack for item in items],
            ignore_index=True,
        ).loc[:, OPERATING_COLUMNS],
        truth_pack=pd.concat(
            [item.truth_pack for item in items],
            ignore_index=True,
        ).loc[:, TRUTH_COLUMNS],
    )


def validate_ordinary_family_counts(
    records: Sequence[OrdinaryPackRecord],
    expected_counts: Mapping[str, Mapping[str, int]],
) -> None:
    """Validate ordinary family counts without invoking any random generator."""

    observed: dict[str, dict[str, int]] = {}
    coordinates: set[tuple[str, str, int]] = set()
    for record in records:
        if (
            not isinstance(record.zero_based_index, int)
            or isinstance(record.zero_based_index, bool)
            or record.zero_based_index < 0
        ):
            raise V015GenerationError("Ordinary index must be a nonnegative integer")
        coordinate = (
            record.partition,
            record.family_id,
            record.zero_based_index,
        )
        if coordinate in coordinates:
            raise V015GenerationError("An ordinary generation coordinate is repeated")
        coordinates.add(coordinate)
        partition_counts = observed.setdefault(record.partition, {})
        partition_counts[record.family_id] = (
            partition_counts.get(record.family_id, 0) + 1
        )
    normalized_expected = {
        str(partition): {str(family): int(count) for family, count in families.items()}
        for partition, families in expected_counts.items()
    }
    if observed != normalized_expected:
        raise V015GenerationError(
            "Ordinary family counts differ from the frozen generation plan: "
            f"observed={observed}, expected={normalized_expected}"
        )


def _validate_ordinary_records(
    records: Sequence[OrdinaryPackRecord],
    protocol: ValidatedV015Protocol,
    *,
    formal: bool,
) -> None:
    items = tuple(records)
    generation_coordinates: set[tuple[str, str, int]] = set()
    for position, record in enumerate(items):
        if (
            not isinstance(record.zero_based_index, int)
            or isinstance(record.zero_based_index, bool)
            or record.zero_based_index < 0
        ):
            raise V015GenerationError("Ordinary index must be a nonnegative integer")
        generation_coordinate = (
            record.partition,
            record.family_id,
            record.zero_based_index,
        )
        if generation_coordinate in generation_coordinates:
            raise V015GenerationError("An ordinary generation coordinate is repeated")
        generation_coordinates.add(generation_coordinate)
        coordinates = _validate_pack_alignment(
            record.packs, context=f"ordinary pack {position}"
        )
        if len(coordinates) != 1:
            raise V015GenerationError(
                "Every ordinary generated pack must contain exactly one member"
            )
        ((partition, _),) = coordinates
        if partition != record.partition:
            raise V015GenerationError(
                "Ordinary record partition differs from its generated pack"
            )
        truth_families = set(record.packs.truth_pack["truth_family"].astype(str))
        if truth_families != {record.family_id}:
            raise V015GenerationError(
                "Ordinary record family differs from its generated truth pack"
            )
    if formal:
        expected_counts = protocol.partition_count_map()
        validate_ordinary_family_counts(items, expected_counts)
        expected_order = tuple(
            (partition, family_id, index)
            for partition, families in protocol.partition_family_counts
            for family_id, count in families
            for index in range(count)
        )
        observed_order = tuple(
            (item.partition, item.family_id, item.zero_based_index) for item in items
        )
        if observed_order != expected_order:
            raise V015GenerationError(
                "Ordinary members were not generated in frozen "
                "partition/family/index order"
            )


def _concatenate_pair_packs(
    packs: Sequence[MatchedPairPacks],
    *,
    context: str,
) -> tuple[GeneratedMemberPacks, pd.DataFrame]:
    items = tuple(packs)
    if not items:
        raise V015GenerationError(f"{context} cannot be empty")
    member_packs: list[GeneratedMemberPacks] = []
    mappings: list[pd.DataFrame] = []
    for index, item in enumerate(items):
        member = GeneratedMemberPacks(
            prefix_pack=item.prefix_pack,
            forecast_coordinates=item.forecast_coordinates,
            operating_pack=item.operating_pack,
            truth_pack=item.truth_pack,
        )
        coordinates = _validate_pack_alignment(
            member, context=f"{context} pair {index}"
        )
        if len(coordinates) != 2:
            raise V015GenerationError(
                f"{context} pair {index} must contain exactly two members"
            )
        if tuple(item.matched_pairs.columns) != MATCHED_PAIR_COLUMNS:
            raise V015GenerationError(f"{context} pair {index} mapping columns changed")
        if len(item.matched_pairs) != 1:
            raise V015GenerationError(
                f"{context} pair {index} must contain exactly one mapping row"
            )
        mapping = item.matched_pairs.iloc[0]
        mapped = {
            (str(mapping["pair_partition"]), str(mapping["left_cluster_id"])),
            (str(mapping["pair_partition"]), str(mapping["right_cluster_id"])),
        }
        if mapped != coordinates:
            raise V015GenerationError(
                f"{context} pair {index} mapping does not cover its members"
            )
        member_packs.append(member)
        mappings.append(item.matched_pairs)
    return (
        concatenate_member_packs(member_packs),
        pd.concat(mappings, ignore_index=True).loc[:, MATCHED_PAIR_COLUMNS],
    )


def assemble_generated_artifact_frames(
    *,
    ordinary_records: Sequence[OrdinaryPackRecord],
    intrinsic_pairs: Sequence[MatchedPairPacks],
    stress_plan_pairs: Sequence[MatchedPairPacks],
    protocol: ValidatedV015Protocol,
    formal: bool = True,
) -> GeneratedArtifactFrames:
    """Aggregate protocol primitives into the exact twelve generated CSVs."""

    ordinary = tuple(ordinary_records)
    _validate_ordinary_records(ordinary, protocol, formal=formal)
    if formal and (len(intrinsic_pairs) != 250 or len(stress_plan_pairs) != 250):
        raise V015GenerationError("Each frozen matched partition requires 250 pairs")

    ordinary_by_partition: dict[str, list[GeneratedMemberPacks]] = {
        partition: [] for partition in ORDINARY_PARTITIONS
    }
    for record in ordinary:
        if record.partition not in ordinary_by_partition:
            raise V015GenerationError(
                f"Unknown ordinary partition: {record.partition!r}"
            )
        ordinary_by_partition[record.partition].append(record.packs)
    if any(not items for items in ordinary_by_partition.values()):
        raise V015GenerationError(
            "Every ordinary partition must contribute generated members"
        )

    intrinsic_members, intrinsic_mapping = _concatenate_pair_packs(
        intrinsic_pairs, context=INTRINSIC_MATCHED_PARTITION
    )
    stress_members, stress_mapping = _concatenate_pair_packs(
        stress_plan_pairs, context=STRESS_PLAN_MATCHED_PARTITION
    )
    ordinary_members = {
        partition: concatenate_member_packs(items)
        for partition, items in ordinary_by_partition.items()
    }
    all_members = concatenate_member_packs(
        [
            *(ordinary_members[partition] for partition in ORDINARY_PARTITIONS),
            intrinsic_members,
            stress_members,
        ]
    )

    sealed = {
        **{
            f"{partition}_truth.csv": ordinary_members[partition].truth_pack
            for partition in ORDINARY_PARTITIONS
        },
        "intrinsic_matched_truth.csv": intrinsic_members.truth_pack,
        "stress_plan_matched_truth.csv": stress_members.truth_pack,
        "intrinsic_matched_pairs.csv": intrinsic_mapping,
        "stress_plan_matched_pairs.csv": stress_mapping,
    }
    label_free = {
        "prefix_pack.csv": all_members.prefix_pack,
        "forecast_coordinates.csv": all_members.forecast_coordinates,
        "operating_pack.csv": all_members.operating_pack,
    }
    return GeneratedArtifactFrames(
        label_free=MappingProxyType(label_free),
        sealed=MappingProxyType(sealed),
    )


def _post_aggregation_family_counts(
    sealed: Mapping[str, pd.DataFrame],
    protocol: ValidatedV015Protocol,
) -> None:
    expected = protocol.partition_count_map()
    for partition in ORDINARY_PARTITIONS:
        frame = sealed[f"{partition}_truth.csv"]
        members = frame.loc[
            :, ["partition", "cluster_id", "truth_family"]
        ].drop_duplicates()
        if members.duplicated(["partition", "cluster_id"]).any():
            raise V015GenerationError(
                f"{partition} changes truth family within a member"
            )
        observed = {
            str(family): int(count)
            for family, count in members["truth_family"]
            .value_counts(sort=False)
            .items()
        }
        if observed != expected[partition]:
            raise V015GenerationError(
                f"{partition} post-aggregation family counts changed: "
                f"observed={observed}, expected={expected[partition]}"
            )


def _validate_global_member_alignment(
    frames: GeneratedArtifactFrames,
    *,
    formal: bool,
) -> set[tuple[str, str]]:
    label_coordinates = {
        filename: _cluster_coordinates(frame, context=filename)
        for filename, frame in frames.label_free.items()
    }
    reference = label_coordinates["prefix_pack.csv"]
    for filename, coordinates in label_coordinates.items():
        if coordinates != reference:
            raise V015GenerationError(
                f"{filename} cluster set differs from prefix_pack.csv"
            )

    truth_coordinates: set[tuple[str, str]] = set()
    for filename, frame in frames.sealed.items():
        if filename.endswith("_truth.csv"):
            coordinates = _cluster_coordinates(frame, context=filename)
            overlap = truth_coordinates.intersection(coordinates)
            if overlap:
                raise V015GenerationError(
                    "A generated truth member occurs in multiple truth files"
                )
            truth_coordinates.update(coordinates)
    if truth_coordinates != reference:
        raise V015GenerationError(
            "Label-free and sealed truth member sets are not identical"
        )
    cluster_ids = [cluster_id for _, cluster_id in reference]
    if len(set(cluster_ids)) != len(cluster_ids):
        raise V015GenerationError(
            "Opaque cluster IDs are not globally unique across all partitions"
        )
    if formal and len(reference) != 5950:
        raise V015GenerationError(
            f"Generated member count is {len(reference)}, expected 5950"
        )
    return reference


def _content_payloads_by_coordinate(
    frames: GeneratedArtifactFrames,
    *,
    enforce_frozen_counts: bool,
) -> dict[tuple[str, str], Mapping[str, bytes]]:
    prefix = frames.label_free["prefix_pack.csv"]
    forecast = frames.label_free["forecast_coordinates.csv"]
    operating = frames.label_free["operating_pack.csv"]
    prefix_groups = {
        (str(partition), str(cluster_id)): group
        for (partition, cluster_id), group in prefix.groupby(
            ["partition", "cluster_id"], sort=False
        )
    }
    forecast_groups = {
        (str(partition), str(cluster_id)): group
        for (partition, cluster_id), group in forecast.groupby(
            ["partition", "cluster_id"], sort=False
        )
    }
    operating_groups = {
        (str(partition), str(cluster_id)): group
        for (partition, cluster_id), group in operating.groupby(
            ["partition", "cluster_id"], sort=False
        )
    }
    if (
        set(prefix_groups) != set(forecast_groups)
        or set(prefix_groups) != set(operating_groups)
        or any(len(group) != 1 for group in operating_groups.values())
    ):
        raise V015GenerationError(
            "Predictor content cannot be aligned to exactly one operating row"
        )

    result: dict[tuple[str, str], Mapping[str, bytes]] = {}
    for coordinate in sorted(prefix_groups):
        payloads = predictor_content_payloads(
            prefix_groups[coordinate],
            forecast_groups[coordinate],
            operating_groups[coordinate].iloc[0],
            enforce_frozen_counts=enforce_frozen_counts,
        )
        if payloads.random_policy != payloads.arm_a:
            raise V015GenerationError(
                "Frozen random-policy and Arm-A predictor content diverged"
            )
        result[coordinate] = MappingProxyType(
            {
                "random_policy": payloads.random_policy,
                "arm_a": payloads.arm_a,
                "arm_b": payloads.arm_b,
                "placebo": payloads.placebo,
            }
        )
    return result


def _mapping_pairs(
    frame: pd.DataFrame,
    *,
    partition: str,
) -> set[frozenset[str]]:
    if set(frame["pair_partition"].astype(str)) != {partition}:
        raise V015GenerationError(
            f"{partition} mapping contains the wrong pair partition"
        )
    pairs = {
        frozenset((str(row.left_cluster_id), str(row.right_cluster_id)))
        for row in frame.itertuples(index=False)
    }
    if any(len(pair) != 2 for pair in pairs) or len(pairs) != len(frame):
        raise V015GenerationError(f"{partition} mapping is not a one-to-one pair set")
    return pairs


def _validate_prescribed_matched_repeats(
    payloads: Mapping[tuple[str, str], Mapping[str, bytes]],
    *,
    partition: str,
    pairs: set[frozenset[str]],
    duplicated_fields: Sequence[str],
    unique_fields: Sequence[str],
) -> None:
    partition_payloads = {
        cluster_id: content
        for (item_partition, cluster_id), content in payloads.items()
        if item_partition == partition
    }
    expected_members = set().union(*pairs) if pairs else set()
    if set(partition_payloads) != expected_members:
        raise V015GenerationError(
            f"{partition} predictor members differ from its mapping"
        )
    for field in duplicated_fields:
        groups: dict[str, set[str]] = {}
        for cluster_id, content in partition_payloads.items():
            digest = hashlib.sha256(content[field]).hexdigest()
            groups.setdefault(digest, set()).add(cluster_id)
        if {frozenset(cluster_ids) for cluster_ids in groups.values()} != pairs:
            raise V015GenerationError(
                f"{partition} {field} repeats are not exactly the declared pairs"
            )
    for field in unique_fields:
        digests = [
            hashlib.sha256(content[field]).hexdigest()
            for content in partition_payloads.values()
        ]
        if len(set(digests)) != len(digests):
            raise V015GenerationError(
                f"{partition} contains an undeclared {field} content repeat"
            )


def validate_predictor_content_collision_policy(
    frames: GeneratedArtifactFrames,
    *,
    enforce_frozen_counts: bool = True,
) -> int:
    """Audit predictor SHA/content consistency and frozen duplicate policy."""

    payloads = _content_payloads_by_coordinate(
        frames, enforce_frozen_counts=enforce_frozen_counts
    )
    ledger = CollisionLedger()
    for (partition, cluster_id), content_by_name in payloads.items():
        for name, content in content_by_name.items():
            digest = hashlib.sha256(content).hexdigest()
            try:
                ledger.register_content_hash(
                    namespace="global_predictor_content",
                    digest=digest,
                    canonical_content=content,
                    unique_content_required=False,
                )
                if partition in {"test", "audit"}:
                    ledger.register_content_hash(
                        # The freeze says ordinary test *and* audit content is
                        # unique.  Use one cross-partition namespace so an
                        # accidental test/audit duplicate cannot hide behind
                        # two otherwise valid partition labels.
                        namespace=f"ordinary_test_audit/{name}",
                        digest=digest,
                        canonical_content=content,
                        unique_content_required=True,
                    )
            except V015CollisionError as exc:
                raise V015GenerationError(str(exc)) from exc

    intrinsic_pairs = _mapping_pairs(
        frames.sealed["intrinsic_matched_pairs.csv"],
        partition=INTRINSIC_MATCHED_PARTITION,
    )
    _validate_prescribed_matched_repeats(
        payloads,
        partition=INTRINSIC_MATCHED_PARTITION,
        pairs=intrinsic_pairs,
        duplicated_fields=("random_policy", "arm_a", "arm_b", "placebo"),
        unique_fields=(),
    )
    stress_pairs = _mapping_pairs(
        frames.sealed["stress_plan_matched_pairs.csv"],
        partition=STRESS_PLAN_MATCHED_PARTITION,
    )
    _validate_prescribed_matched_repeats(
        payloads,
        partition=STRESS_PLAN_MATCHED_PARTITION,
        pairs=stress_pairs,
        duplicated_fields=("random_policy", "arm_a", "placebo"),
        unique_fields=("arm_b",),
    )
    return ledger.content_hash_count


def validate_generated_artifact_frames(
    frames: GeneratedArtifactFrames,
    *,
    protocol: ValidatedV015Protocol,
    contract: FrozenArtifactContract,
    formal: bool = True,
) -> None:
    """Validate all intra- and cross-artifact contracts before serialization."""

    if set(frames.label_free) != set(LABEL_FREE_CSV_FILENAMES):
        raise V015GenerationError("Generated label-free file membership changed")
    if set(frames.sealed) != set(contract.sealed_filenames):
        raise V015GenerationError("Generated sealed file membership changed")
    try:
        for filename, frame in frames.label_free.items():
            canonicalize_frame(
                frame,
                contract.csv_schema(filename),
                contract,
                formal=formal,
            )
        validate_sealed_truth_bundle(
            frames.sealed,
            contract,
            formal=formal,
        )
    except V015ArtifactError as exc:
        raise V015GenerationError(str(exc)) from exc
    _validate_global_member_alignment(frames, formal=formal)
    if formal:
        _post_aggregation_family_counts(frames.sealed, protocol)
    try:
        validate_predictor_content_collision_policy(
            frames,
            enforce_frozen_counts=formal,
        )
    except (V015ArtifactError, V015CollisionError) as exc:
        raise V015GenerationError(str(exc)) from exc


def _truth_commitment_payload(
    *,
    sealed_bytes: Mapping[str, bytes],
    row_counts: Mapping[str, int],
    contract: FrozenArtifactContract,
    created_utc: str,
) -> dict[str, object]:
    try:
        parsed = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise V015GenerationError(
            "Truth commitment timestamp must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise V015GenerationError("Truth commitment timestamp must include a timezone")
    if tuple(sealed_bytes) != contract.sealed_filenames:
        raise V015GenerationError(
            "Truth commitment must cover all nine sealed files in frozen order"
        )
    files = [
        {
            "path": filename,
            "row_count": int(row_counts[filename]),
            "byte_count": len(sealed_bytes[filename]),
            "byte_sha256": hashlib.sha256(sealed_bytes[filename]).hexdigest(),
        }
        for filename in contract.sealed_filenames
    ]
    payload: dict[str, object] = {
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_byte_sha256,
        "files": files,
        "created_utc": created_utc,
        "truth_values_withheld_by_physical_path": True,
    }
    if set(payload) != set(contract.json_keys(TRUTH_COMMITMENT_FILENAME)):
        raise V015GenerationError("Truth commitment JSON allowlist changed")
    return payload


def prepare_generated_artifacts(
    frames: GeneratedArtifactFrames,
    *,
    protocol: ValidatedV015Protocol,
    contract: FrozenArtifactContract,
    created_utc: str,
    formal: bool = True,
) -> PreparedGenerationArtifacts:
    """Canonicalize every output byte before allowing the first filesystem write."""

    validate_generated_artifact_frames(
        frames,
        protocol=protocol,
        contract=contract,
        formal=formal,
    )
    try:
        label_bytes = {
            filename: canonical_csv_bytes(
                frames.label_free[filename],
                contract.csv_schema(filename),
                contract,
                formal=formal,
            )
            for filename in LABEL_FREE_CSV_FILENAMES
        }
        sealed_bytes = {
            filename: canonical_csv_bytes(
                frames.sealed[filename],
                contract.csv_schema(filename),
                contract,
                formal=formal,
            )
            for filename in contract.sealed_filenames
        }
    except V015ArtifactError as exc:
        raise V015GenerationError(str(exc)) from exc
    row_counts = {
        **{
            filename: len(frames.label_free[filename])
            for filename in LABEL_FREE_CSV_FILENAMES
        },
        **{
            filename: len(frames.sealed[filename])
            for filename in contract.sealed_filenames
        },
    }
    commitment = _truth_commitment_payload(
        sealed_bytes=sealed_bytes,
        row_counts=row_counts,
        contract=contract,
        created_utc=created_utc,
    )
    commitment_bytes = canonical_json_bytes(commitment)
    if not commitment_bytes.endswith(b"\n"):
        raise V015GenerationError("Truth commitment JSON is not canonical")
    label_bytes[TRUTH_COMMITMENT_FILENAME] = commitment_bytes
    return PreparedGenerationArtifacts(
        label_free_bytes=MappingProxyType(label_bytes),
        sealed_bytes=MappingProxyType(sealed_bytes),
        row_counts=MappingProxyType(row_counts),
        truth_commitment_payload=MappingProxyType(commitment),
    )


def assert_generation_destinations_available(
    *,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
    contract: FrozenArtifactContract,
) -> tuple[Path, Path]:
    """Fail before generation when any destination is nested or partially present."""

    try:
        label, sealed = assert_separate_truth_roots(label_free_root, sealed_truth_root)
    except V015ArtifactError as exc:
        raise V015GenerationError(str(exc)) from exc
    for root, context in ((label, "label-free"), (sealed, "sealed")):
        if root.exists() and not root.is_dir():
            raise V015GenerationError(f"{context} root exists but is not a directory")
    targets = [
        *(label / filename for filename in LABEL_FREE_CSV_FILENAMES),
        label / TRUTH_COMMITMENT_FILENAME,
        *(sealed / filename for filename in contract.sealed_filenames),
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise V015GenerationError(
            "Generation refuses overwrite or partial prior output: "
            + ", ".join(str(path) for path in existing)
        )
    return label, sealed


def _exclusive_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise V015GenerationError(
            f"Generation destination appeared after preflight: {path}"
        ) from exc


def write_prepared_generation_artifacts(
    prepared: PreparedGenerationArtifacts,
    *,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
    contract: FrozenArtifactContract,
) -> tuple[tuple[ArtifactMetadata, ...], tuple[ArtifactMetadata, ...], str]:
    """Exclusively create one already validated generation bundle."""

    label, sealed = assert_generation_destinations_available(
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
        contract=contract,
    )
    if (
        tuple(prepared.label_free_bytes)
        != (*LABEL_FREE_CSV_FILENAMES, TRUTH_COMMITMENT_FILENAME)
        or tuple(prepared.sealed_bytes) != contract.sealed_filenames
    ):
        raise V015GenerationError("Prepared artifact membership or order changed")

    for filename in contract.sealed_filenames:
        _exclusive_write(sealed / filename, prepared.sealed_bytes[filename])
    for filename in (*LABEL_FREE_CSV_FILENAMES, TRUTH_COMMITMENT_FILENAME):
        _exclusive_write(label / filename, prepared.label_free_bytes[filename])

    sealed_metadata = tuple(
        ArtifactMetadata(
            path=filename,
            row_count=prepared.row_counts[filename],
            byte_count=len(prepared.sealed_bytes[filename]),
            byte_sha256=hashlib.sha256(prepared.sealed_bytes[filename]).hexdigest(),
        )
        for filename in contract.sealed_filenames
    )
    label_metadata = tuple(
        ArtifactMetadata(
            path=filename,
            row_count=(
                prepared.row_counts[filename]
                if filename in LABEL_FREE_CSV_FILENAMES
                else 1
            ),
            byte_count=len(prepared.label_free_bytes[filename]),
            byte_sha256=hashlib.sha256(prepared.label_free_bytes[filename]).hexdigest(),
        )
        for filename in (*LABEL_FREE_CSV_FILENAMES, TRUTH_COMMITMENT_FILENAME)
    )
    for root, content in (
        (sealed, prepared.sealed_bytes),
        (label, prepared.label_free_bytes),
    ):
        for filename, expected in content.items():
            if (root / filename).read_bytes() != expected:
                raise V015GenerationError(
                    f"Written bytes changed for {root / filename}"
                )
    commitment_hash = hashlib.sha256(
        prepared.label_free_bytes[TRUTH_COMMITMENT_FILENAME]
    ).hexdigest()
    return label_metadata, sealed_metadata, commitment_hash


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _require_pre_generation_ledger(
    *,
    label_free_root: str | Path,
    contract: FrozenArtifactContract,
    environment: FormalEnvironmentIdentity,
) -> str:
    ledger_path = Path(label_free_root).resolve() / "exposure_log.jsonl"
    if not ledger_path.is_file():
        raise V015GenerationError(
            "Formal exposure_log.jsonl must exist before generation"
        )
    try:
        states = validate_formal_exposure_log(ledger_path, contract)
    except (OSError, V015ArtifactError, V015FirewallError) as exc:
        raise V015GenerationError(
            "Formal exposure ledger failed pre-generation validation"
        ) from exc
    if any(
        state.identity.git_commit != environment.git_commit
        or state.identity.config_byte_sha256 != environment.config_byte_sha256
        for state in states.values()
    ):
        raise V015GenerationError(
            "A prior formal attempt used a different implementation identity"
        )
    if any(
        state.truth_commitments_byte_sha256 is not None
        or state.completed_phase == "scoring_completed"
        for state in states.values()
    ):
        raise V015GenerationError(
            "Committed truth may be reused for recovery but never regenerated"
        )
    candidates = [
        state
        for state in states.values()
        if (
            state.completed_phase == "before_generation"
            and state.pending_phase is None
            and not state.terminal_failed
        )
    ]
    if len(candidates) != 1:
        raise V015GenerationError(
            "Exactly one attempt must be checkpointed before generation"
        )
    current = candidates[0]
    unfinished_other = [
        state
        for state in states.values()
        if (
            state.identity.attempt_id != current.identity.attempt_id
            and not state.terminal_failed
            and state.completed_phase != "scoring_completed"
        )
    ]
    if unfinished_other:
        raise V015GenerationError(
            "Another formal attempt remains unfinished in the exposure ledger"
        )
    if (
        current.identity.git_commit != environment.git_commit
        or current.identity.config_byte_sha256 != environment.config_byte_sha256
        or current.truth_commitments_byte_sha256 is not None
        or current.prediction_commitment_byte_sha256 is not None
        or current.opened_truth_files
    ):
        raise V015GenerationError(
            "Pre-generation attempt identity or firewall state is invalid"
        )
    return current.identity.attempt_id


def generate_frozen_v015_artifacts(
    *,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
) -> WrittenGenerationArtifacts:
    """Execute the one-shot frozen generator.

    The caller must invoke this only after committing a clean implementation
    freeze and checkpointing ``before_generation`` in the formal exposure
    ledger.  There is deliberately no argument for a seed, config, family,
    partition, count, or generation mode.
    """

    environment = verify_formal_environment(_PROJECT_ROOT)
    protocol = load_frozen_protocol_config(DEFAULT_V2_CONFIG_PATH)
    contract = load_artifact_contract()
    if protocol.protocol_id != contract.protocol_id:
        raise V015GenerationError("Protocol and artifact contracts disagree")
    attempt_id = _require_pre_generation_ledger(
        label_free_root=label_free_root,
        contract=contract,
        environment=environment,
    )
    assert_generation_destinations_available(
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
        contract=contract,
    )

    identity = FormalAttemptIdentity(
        attempt_id=attempt_id,
        git_commit=environment.git_commit,
        config_byte_sha256=environment.config_byte_sha256,
    )
    ledger_path = Path(label_free_root).resolve() / "exposure_log.jsonl"
    try:
        # This exhaustive, RNG-free audit must precede every formal seed use.
        collision_summary = audit_generation_coordinate_plan(protocol)

        ordinary: list[OrdinaryPackRecord] = []
        truth_specs = []
        for partition, family_counts in protocol.partition_family_counts:
            for family_id, count in family_counts:
                for index in range(count):
                    operating = generate_operating_covariates(
                        protocol,
                        partition=partition,
                        family_id=family_id,
                        zero_based_index=index,
                    )
                    truth_spec, fixed_operating = sample_truth_spec(
                        protocol,
                        partition=partition,
                        family_id=family_id,
                        zero_based_index=index,
                        operating=operating,
                    )
                    generated = generate_cluster_packs(
                        protocol, truth_spec, fixed_operating
                    )
                    ordinary.append(
                        OrdinaryPackRecord(
                            partition=partition,
                            family_id=family_id,
                            zero_based_index=index,
                            packs=generated.packs,
                        )
                    )
                    truth_specs.append(truth_spec)
        validate_unique_stream_seeds(truth_specs)

        intrinsic = tuple(
            generate_intrinsic_matched_pair(protocol, zero_based_pair_index=pair_index)
            for pair_index in range(250)
        )
        stress = tuple(
            generate_stress_plan_matched_pair(
                protocol, zero_based_pair_index=pair_index
            )
            for pair_index in range(250)
        )
        frames = assemble_generated_artifact_frames(
            ordinary_records=ordinary,
            intrinsic_pairs=intrinsic,
            stress_plan_pairs=stress,
            protocol=protocol,
            formal=True,
        )
        prepared = prepare_generated_artifacts(
            frames,
            protocol=protocol,
            contract=contract,
            created_utc=_utc_now(),
            formal=True,
        )
        label_metadata, sealed_metadata, commitment_hash = (
            write_prepared_generation_artifacts(
                prepared,
                label_free_root=label_free_root,
                sealed_truth_root=sealed_truth_root,
                contract=contract,
            )
        )
    except KeyboardInterrupt as exc:
        append_phase_error_without_masking(
            error=exc,
            ledger_path=ledger_path,
            identity=identity,
            contract=contract,
            created_utc=_utc_now(),
            phase="truth_committed",
            exit_status="interrupted",
            truth_commitments_byte_sha256=None,
            prediction_commitment_byte_sha256=None,
            message="Formal generation was interrupted before truth commitment.",
        )
        raise
    except BaseException as exc:
        append_phase_error_without_masking(
            error=exc,
            ledger_path=ledger_path,
            identity=identity,
            contract=contract,
            created_utc=_utc_now(),
            phase="truth_committed",
            exit_status="failed",
            truth_commitments_byte_sha256=None,
            prediction_commitment_byte_sha256=None,
            message="Formal generation failed before truth commitment.",
        )
        raise

    append_formal_exposure_event(
        path=ledger_path,
        identity=identity,
        contract=contract,
        created_utc=_utc_now(),
        phase="truth_committed",
        exit_status="completed",
        truth_commitments_byte_sha256=commitment_hash,
        prediction_commitment_byte_sha256=None,
        message="All nine sealed files were committed before any truth access.",
    )
    return WrittenGenerationArtifacts(
        generation_collision_summary=collision_summary,
        label_free_metadata=label_metadata,
        sealed_metadata=sealed_metadata,
        truth_commitment_byte_sha256=commitment_hash,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper with no protocol override surface."""

    parser = argparse.ArgumentParser(
        description="Generate the byte-frozen V0.15 synthetic data artifacts once."
    )
    parser.add_argument("--label-free-root", type=Path, required=True)
    parser.add_argument("--sealed-truth-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_frozen_v015_artifacts(
        label_free_root=args.label_free_root,
        sealed_truth_root=args.sealed_truth_root,
    )
    print(
        json.dumps(
            {
                "truth_commitment_byte_sha256": (result.truth_commitment_byte_sha256),
                "generated_cluster_count": (
                    result.generation_collision_summary.cluster_id_count
                ),
                "generated_pair_count": (
                    result.generation_collision_summary.pair_id_count
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised only after freeze.
    raise SystemExit(main())
