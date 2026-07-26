"""RNG-free generation-coordinate namespace commitments for V2.1.

The formal entry point enumerates the complete V2.1 and V2 coordinate plans.
It derives only seeds, opaque identifiers, identity digests, and explicit
formula witnesses.  Content-dependent hashes are bound separately after
generation and before prediction; this module imports no experiment generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    V021ContractView,
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_AMENDMENT_SEMANTIC_SHA256,
    V021_EXPECTED_SEED_ROOTS,
    V021_PROTOCOL_ID,
    V2_SEED_ROOTS,
)


class V021CollisionError(ValueError):
    """Raised when a coordinate plan or collision commitment is invalid."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_IDENTIFIER = re.compile(r"^[cp]_[0-9a-f]{32}$")
_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SEED_MODULUS = 2**63 - 1
_ORDINARY_STREAMS = (
    "opaque_id",
    "operating_covariates",
    "truth_parameters",
    "measurement_noise",
)
_MATCHED_STREAMS = (
    "shared_truth",
    "shared_operating",
    "shared_measurement",
    "placebo",
    "opaque_ids",
    "opaque_swap",
)
_MATCHED_PARTITIONS = (
    "intrinsic_matched_pairs",
    "stress_plan_matched_pairs",
)
_NOVEL_FAMILIES = ("smooth_broken_power", "saturating_logistic_knee")
_V2_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2"
_FORMAL_MATCHED_PAIR_COUNT = 250
_FORMAL_BOOTSTRAP_RESAMPLES = 5_000
_FORMAL_RANDOM_RANKINGS = 10_000
_FORMAL_STRESS_PERMUTATIONS = 10_000
_FORMAL_RANKING_PARTITIONS = ("test", "audit")
_FORMAL_TEST_FAMILIES = (
    "single_power",
    "dual_power",
    "saturating_plus_slow",
    "early_activation_plus_power",
    "late_knee",
    "linear_drift_plus_power",
    "smooth_broken_power",
    "saturating_logistic_knee",
)
_COMMITMENT_SCHEMA = "lifetwin.generation_coordinate_namespace_commitment.v1"
_ACTUAL_HASH_COMMITMENT_SCHEMA = "lifetwin.actual_analysis_hash_ledger.v1"
ANALYSIS_TIE_ARMS = (
    "prefix_only",
    "visible_stress",
    "placebo_8",
    "arm_a_plus_s_plan",
    "strongest_single_feature",
    "planned_stress_only",
    "prefix_rmse_only",
    "v1_max_envelope_only",
    "center_sqrt_abs_difference_only",
)
_DERIVATION_WITNESS_CONTENT_SHA256 = "0" * 64
_DOMAIN_ORDER = (
    "ordinary_stream_seed",
    "ordinary_cluster_identifier",
    "ordinary_identity_digest",
    "matched_stream_seed",
    "matched_identifier",
    "matched_identity_digest",
    "bootstrap_seed",
    "random_ranking_formula_witness",
    "stress_permutation_formula_witness",
    "analysis_tie_formula_witness",
)


def _component(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or _COMPONENT.fullmatch(value) is None
        or "|" in value
    ):
        raise V021CollisionError(f"{context} is not a canonical ASCII component")
    return value


def _positive_integer(value: object, *, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value < _SEED_MODULUS
    ):
        raise V021CollisionError(f"{context} must be a positive bounded integer")
    return value


def _count(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise V021CollisionError(f"{context} must be a positive integer")
    return value


def _sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V021CollisionError(f"{context} must be a lowercase SHA256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise V021CollisionError("Commitment payload is not canonical JSON") from exc


def _hash_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class OrdinaryPlanGroup:
    """One ordinary partition/family coordinate range."""

    partition: str
    family_id: str
    count: int
    seed_root_name: str

    def __post_init__(self) -> None:
        _component(self.partition, context="ordinary partition")
        _component(self.family_id, context="ordinary family")
        _count(self.count, context="ordinary count")
        _component(self.seed_root_name, context="ordinary seed-root name")


@dataclass(frozen=True, slots=True)
class MatchedPlanGroup:
    """One matched-pair coordinate range."""

    partition: str
    pair_count: int
    seed_root_name: str

    def __post_init__(self) -> None:
        _component(self.partition, context="matched partition")
        _count(self.pair_count, context="matched pair count")
        _component(self.seed_root_name, context="matched seed-root name")


@dataclass(frozen=True, slots=True)
class GenerationPlanSpec:
    """Complete declarative input to one protocol's coordinate audit."""

    protocol_id: str
    protocol_byte_sha256: str
    protocol_semantic_sha256: str
    seed_roots: tuple[tuple[str, int], ...]
    ordinary_groups: tuple[OrdinaryPlanGroup, ...]
    matched_groups: tuple[MatchedPlanGroup, ...]
    bootstrap_partition: str
    bootstrap_families: tuple[str, ...]
    bootstrap_resamples: int
    ranking_partitions: tuple[str, ...]
    random_ranking_count: int
    stress_partition: str
    stress_families: tuple[str, ...]
    stress_permutation_count: int
    placebo_seed_root_name: str = "placebo_covariate"
    bootstrap_seed_root_name: str = "bootstrap"
    random_ranking_root_name: str = "random_rankings"
    stress_permutation_root_name: str = "stress_permutations"
    analysis_tie_arms: tuple[str, ...] = ANALYSIS_TIE_ARMS

    def __post_init__(self) -> None:
        _validate_plan_spec(self)

    def seed_root_map(self) -> dict[str, int]:
        """Return a fresh root map after validation."""

        return dict(self.seed_roots)


@dataclass(frozen=True, slots=True)
class PlanCounts:
    """Counts committed for one fully enumerated plan."""

    ordinary_clusters: int
    matched_pairs: int
    generated_members: int
    bootstrap_coordinates: int
    random_ranking_coordinates: int
    stress_permutation_coordinates: int
    analysis_tie_coordinates: int
    seed_count: int
    identifier_count: int
    digest_count: int
    ordered_ledger_records: int

    def to_payload(self) -> dict[str, int]:
        return {
            "analysis_tie_formula_witnesses": self.analysis_tie_coordinates,
            "bootstrap_coordinates": self.bootstrap_coordinates,
            "generated_members": self.generated_members,
            "identity_and_formula_witness_digest_count": self.digest_count,
            "identifier_count": self.identifier_count,
            "matched_pairs": self.matched_pairs,
            "namespace_ledger_records": self.ordered_ledger_records,
            "ordinary_clusters": self.ordinary_clusters,
            "random_ranking_coordinates": self.random_ranking_coordinates,
            "seed_count": self.seed_count,
            "stress_permutation_coordinates": self.stress_permutation_coordinates,
        }


@dataclass(frozen=True, slots=True)
class DomainCommitment:
    """Digest and count for one complete ordered coordinate domain."""

    name: str
    record_count: int
    ordered_ledger_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ordered_ledger_sha256": self.ordered_ledger_sha256,
            "record_count": self.record_count,
        }


@dataclass(frozen=True, slots=True)
class ProtocolPlanCommitment:
    """Compact commitment to one protocol's complete expanded ledger."""

    protocol_id: str
    protocol_byte_sha256: str
    protocol_semantic_sha256: str
    seed_roots: tuple[tuple[str, int], ...]
    coordinate_spec_sha256: str
    ordered_ledger_sha256: str
    counts: PlanCounts
    domains: tuple[DomainCommitment, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "coordinate_domains": [item.to_payload() for item in self.domains],
            "coordinate_spec_sha256": self.coordinate_spec_sha256,
            "counts": self.counts.to_payload(),
            "ordered_namespace_ledger_sha256": self.ordered_ledger_sha256,
            "protocol_byte_sha256": self.protocol_byte_sha256,
            "protocol_id": self.protocol_id,
            "protocol_semantic_sha256": self.protocol_semantic_sha256,
            "seed_roots": [
                {"name": name, "value": value} for name, value in self.seed_roots
            ],
        }


@dataclass(frozen=True, slots=True)
class GenerationCoordinateNamespaceCommitment:
    """Canonical commitment to pre-generation coordinates and formula witnesses."""

    current: ProtocolPlanCommitment
    predecessor: ProtocolPlanCommitment
    complete_ordered_ledger_sha256: str
    complete_ordered_ledger_records: int

    @property
    def payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible payload."""

        return {
            "cross_protocol_coordinate_namespace": {
                "complete_ordered_namespace_ledger_records": (
                    self.complete_ordered_ledger_records
                ),
                "complete_ordered_namespace_ledger_sha256": (
                    self.complete_ordered_ledger_sha256
                ),
                "content_dependent_hash_comparison": ("deferred_until_post_generation"),
                "formula_witness_content_sha256": (_DERIVATION_WITNESS_CONTENT_SHA256),
                "ledger_order": "current_then_predecessor",
                "scope": (
                    "actual_seeds_identifiers_identity_digests_and_"
                    "content_independent_formula_witnesses"
                ),
            },
            "current": self.current.to_payload(),
            "predecessor": self.predecessor.to_payload(),
            "schema_version": _COMMITMENT_SCHEMA,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.payload) + b"\n"

    @property
    def byte_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes)


# Compatibility for the generation lifecycle while its persisted filename remains
# the protocol's frozen ``generation_plan_commitment.json``.
GenerationPlanCommitment = GenerationCoordinateNamespaceCommitment


@dataclass(frozen=True, slots=True)
class ActualAnalysisContentRecord:
    """Generated content hashes needed by every stochastic analysis derivation."""

    partition: str
    family_id: str
    member_id: str
    random_policy_content_sha256: str
    predictor_content_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _component(self.partition, context="actual-hash partition")
        _component(self.family_id, context="actual-hash family")
        _component(self.member_id, context="actual-hash member ID")
        _sha256(
            self.random_policy_content_sha256,
            context="actual random-policy content hash",
        )
        if not isinstance(self.predictor_content_hashes, tuple):
            raise V021CollisionError(
                "Predictor content hashes must be an ordered tuple"
            )
        arms: list[str] = []
        for item in self.predictor_content_hashes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise V021CollisionError(
                    "Each predictor content hash must be an arm/hash pair"
                )
            arm, digest = item
            arms.append(_component(arm, context="predictor arm"))
            _sha256(digest, context=f"predictor content hash/{arm}")
        if not arms or len(arms) != len(set(arms)):
            raise V021CollisionError(
                "Predictor content hash arms must be unique and nonempty"
            )


@dataclass(frozen=True, slots=True)
class ActualHashDomainCommitment:
    """Count and digest for one ordered domain of observed or derived hashes."""

    name: str
    record_count: int
    ordered_ledger_sha256: str

    def __post_init__(self) -> None:
        _component(self.name, context="actual-hash domain")
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count < 0
        ):
            raise V021CollisionError("Actual-hash domain count is invalid")
        _sha256(
            self.ordered_ledger_sha256,
            context=f"actual-hash domain digest/{self.name}",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ordered_ledger_sha256": self.ordered_ledger_sha256,
            "record_count": self.record_count,
        }


@dataclass(frozen=True, slots=True)
class ActualAnalysisHashLedgerCommitment:
    """Post-generation, pre-prediction binding of actual analysis hashes."""

    protocol_id: str
    random_ranking_root: int
    stress_permutation_root: int
    ranking_partitions: tuple[str, ...]
    random_ranking_count: int
    stress_partition: str
    stress_families: tuple[str, ...]
    stress_permutation_count: int
    tie_arms: tuple[str, ...]
    source_record_count: int
    ranking_partition_record_counts: tuple[tuple[str, int], ...]
    stress_family_record_counts: tuple[tuple[str, int], ...]
    ordered_ledger_sha256: str
    domains: tuple[ActualHashDomainCommitment, ...]

    @property
    def payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible commitment payload."""

        counts = {item.name: item.record_count for item in self.domains}
        return {
            "coordinate_semantics": {
                "random_ranking": (
                    "partition_selects_pool_only_root_index_and_content_hash_"
                    "define_the_key"
                ),
                "stress_permutation": (
                    "root_permutation_index_family_and_random_policy_content_"
                    "hash_define_the_key"
                ),
                "tie_breaking": "protocol_id_arm_and_predictor_content_hash",
            },
            "domains": [item.to_payload() for item in self.domains],
            "ledger_counts": {
                **counts,
                "complete_ordered_ledger_records": sum(counts.values()),
                "source_content_records": self.source_record_count,
            },
            "ordered_ledger_sha256": self.ordered_ledger_sha256,
            "protocol_id": self.protocol_id,
            "random_ranking_count": self.random_ranking_count,
            "random_ranking_root": self.random_ranking_root,
            "ranking_partitions": list(self.ranking_partitions),
            "schema_version": _ACTUAL_HASH_COMMITMENT_SCHEMA,
            "source_coordinate_counts": {
                "ranking_partitions": [
                    {"partition": name, "record_count": count}
                    for name, count in self.ranking_partition_record_counts
                ],
                "stress_families": [
                    {"family_id": name, "record_count": count}
                    for name, count in self.stress_family_record_counts
                ],
            },
            "stage": "post_generation_pre_prediction",
            "stress_families": list(self.stress_families),
            "stress_partition": self.stress_partition,
            "stress_permutation_count": self.stress_permutation_count,
            "stress_permutation_root": self.stress_permutation_root,
            "tie_arms": list(self.tie_arms),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.payload) + b"\n"

    @property
    def byte_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes)


class CoordinateCollisionLedger:
    """Fail-closed registries for actual namespace values and formula witnesses."""

    def __init__(self) -> None:
        self._coordinates: dict[str, str] = {}
        self._seeds: dict[int, str] = {}
        self._identifiers: dict[str, str] = {}
        self._digests: dict[str, str] = {}

    def _check_coordinate(self, *, label: str, kind: str) -> None:
        canonical = _component_path(label, context="ledger label")
        prior = self._coordinates.get(canonical)
        if prior is not None:
            raise V021CollisionError(
                f"Duplicate coordinate {canonical!r} ({prior} and {kind})"
            )

    def _commit_coordinate(self, *, label: str, kind: str) -> None:
        self._coordinates[label] = kind

    def register_seed(self, *, label: str, seed: int) -> None:
        self._check_coordinate(label=label, kind="seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed < _SEED_MODULUS
        ):
            raise V021CollisionError(f"{label} is not a valid derived seed")
        prior = self._seeds.get(seed)
        if prior is not None:
            raise V021CollisionError(f"Seed collision between {prior!r} and {label!r}")
        self._seeds[seed] = label
        self._commit_coordinate(label=label, kind="seed")

    def register_identifier(self, *, label: str, identifier: str) -> None:
        self._check_coordinate(label=label, kind="identifier")
        if (
            not isinstance(identifier, str)
            or _OPAQUE_IDENTIFIER.fullmatch(identifier) is None
        ):
            raise V021CollisionError(f"{label} is not a canonical opaque identifier")
        prior = self._identifiers.get(identifier)
        if prior is not None:
            raise V021CollisionError(
                f"Identifier collision between {prior!r} and {label!r}"
            )
        self._identifiers[identifier] = label
        self._commit_coordinate(label=label, kind="identifier")

    def register_digest(self, *, label: str, digest: str) -> None:
        self._check_coordinate(label=label, kind="digest")
        _sha256(digest, context=label)
        prior = self._digests.get(digest)
        if prior is not None:
            raise V021CollisionError(
                f"Digest collision between {prior!r} and {label!r}"
            )
        self._digests[digest] = label
        self._commit_coordinate(label=label, kind="digest")

    @property
    def coordinate_count(self) -> int:
        return len(self._coordinates)

    @property
    def seed_count(self) -> int:
        return len(self._seeds)

    @property
    def identifier_count(self) -> int:
        return len(self._identifiers)

    @property
    def digest_count(self) -> int:
        return len(self._digests)


def _component_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise V021CollisionError(f"{context} must be nonempty ASCII")
    parts = value.split("/")
    if any(_COMPONENT.fullmatch(part) is None for part in parts):
        raise V021CollisionError(f"{context} is not a canonical component path")
    return value


def _validate_string_tuple(
    value: object,
    *,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise V021CollisionError(f"{context} must be a nonempty tuple")
    result = tuple(_component(item, context=f"{context} member") for item in value)
    if len(result) != len(set(result)):
        raise V021CollisionError(f"{context} contains duplicates")
    return result


def _validate_plan_spec(spec: GenerationPlanSpec) -> None:
    _component(spec.protocol_id, context="protocol ID")
    _sha256(spec.protocol_byte_sha256, context="protocol byte commitment")
    _sha256(spec.protocol_semantic_sha256, context="protocol semantic commitment")
    if not isinstance(spec.seed_roots, tuple) or not spec.seed_roots:
        raise V021CollisionError("Seed roots must be a nonempty ordered tuple")
    root_names: list[str] = []
    root_values: list[int] = []
    for item in spec.seed_roots:
        if not isinstance(item, tuple) or len(item) != 2:
            raise V021CollisionError("Each seed root must be a name/value pair")
        name, value = item
        root_names.append(_component(name, context="seed-root name"))
        root_values.append(_positive_integer(value, context=f"seed root {name}"))
    if len(root_names) != len(set(root_names)):
        raise V021CollisionError("Seed-root names contain duplicates")
    if len(root_values) != len(set(root_values)):
        raise V021CollisionError("Seed-root values collide")

    if not isinstance(spec.ordinary_groups, tuple) or not spec.ordinary_groups:
        raise V021CollisionError("Ordinary groups must be a nonempty tuple")
    if any(type(item) is not OrdinaryPlanGroup for item in spec.ordinary_groups):
        raise V021CollisionError("Ordinary groups contain an invalid value")
    ordinary_keys = [(item.partition, item.family_id) for item in spec.ordinary_groups]
    if len(ordinary_keys) != len(set(ordinary_keys)):
        raise V021CollisionError("Ordinary coordinates contain duplicate groups")

    if not isinstance(spec.matched_groups, tuple) or not spec.matched_groups:
        raise V021CollisionError("Matched groups must be a nonempty tuple")
    if any(type(item) is not MatchedPlanGroup for item in spec.matched_groups):
        raise V021CollisionError("Matched groups contain an invalid value")
    matched_names = [item.partition for item in spec.matched_groups]
    if len(matched_names) != len(set(matched_names)):
        raise V021CollisionError("Matched coordinates contain duplicate groups")
    if set(matched_names).intersection(partition for partition, _ in ordinary_keys):
        raise V021CollisionError("Ordinary and matched partitions overlap")

    _component(spec.bootstrap_partition, context="bootstrap partition")
    bootstrap_families = _validate_string_tuple(
        spec.bootstrap_families,
        context="bootstrap families",
    )
    _count(spec.bootstrap_resamples, context="bootstrap resamples")
    ranking_partitions = _validate_string_tuple(
        spec.ranking_partitions,
        context="ranking partitions",
    )
    _count(spec.random_ranking_count, context="random-ranking count")
    _component(spec.stress_partition, context="stress partition")
    stress_families = _validate_string_tuple(
        spec.stress_families,
        context="stress-permutation families",
    )
    _count(spec.stress_permutation_count, context="stress-permutation count")
    _validate_string_tuple(
        spec.analysis_tie_arms,
        context="analysis tie arms",
    )

    ordinary_by_partition: dict[str, list[str]] = {}
    for group in spec.ordinary_groups:
        ordinary_by_partition.setdefault(group.partition, []).append(group.family_id)
    if spec.bootstrap_partition not in ordinary_by_partition:
        raise V021CollisionError("Bootstrap partition is not ordinary")
    if tuple(ordinary_by_partition[spec.bootstrap_partition]) != bootstrap_families:
        raise V021CollisionError(
            "Bootstrap families do not exactly match partition insertion order"
        )
    if spec.stress_partition != spec.bootstrap_partition:
        raise V021CollisionError(
            "Stress permutations must use the bootstrap source partition"
        )
    if stress_families != bootstrap_families:
        raise V021CollisionError(
            "Stress-permutation families do not match bootstrap families"
        )
    if any(partition not in ordinary_by_partition for partition in ranking_partitions):
        raise V021CollisionError("A ranking partition is not ordinary")
    expected_ranking_order = tuple(
        partition
        for partition in ordinary_by_partition
        if partition in set(ranking_partitions)
    )
    if ranking_partitions != expected_ranking_order:
        raise V021CollisionError(
            "Ranking partitions must retain ordinary-partition insertion order"
        )

    named_roots = (
        spec.placebo_seed_root_name,
        spec.bootstrap_seed_root_name,
        spec.random_ranking_root_name,
        spec.stress_permutation_root_name,
    )
    for name in named_roots:
        _component(name, context="analysis seed-root name")
    required_roots = {
        *(item.seed_root_name for item in spec.ordinary_groups),
        *(item.seed_root_name for item in spec.matched_groups),
        *named_roots,
    }
    if set(root_names) != required_roots:
        raise V021CollisionError(
            "Seed-root registry does not exactly match coordinate domains"
        )


def derive_stream_seed(
    protocol_id: str,
    *,
    seed_root: int,
    partition: str,
    family_id: str,
    zero_based_index: int,
    stream_name: str,
) -> int:
    """Apply the inherited first-16-hex stream formula without an RNG."""

    _component(protocol_id, context="protocol ID")
    _positive_integer(seed_root, context="seed root")
    _component(partition, context="partition")
    _component(family_id, context="family")
    if (
        isinstance(zero_based_index, bool)
        or not isinstance(zero_based_index, int)
        or zero_based_index < 0
    ):
        raise V021CollisionError("Stream index must be a nonnegative integer")
    _component(stream_name, context="stream name")
    material = (
        f"{protocol_id}|{seed_root}|{partition}|{family_id}|"
        f"{zero_based_index}|{stream_name}"
    ).encode("ascii")
    return int(hashlib.sha256(material).hexdigest()[:16], 16) % _SEED_MODULUS


def derive_bootstrap_seed(
    protocol_id: str,
    *,
    seed_root: int,
    replicate_index: int,
    family_id: str,
) -> int:
    """Apply the inherited bootstrap formula without an RNG."""

    _component(protocol_id, context="protocol ID")
    _positive_integer(seed_root, context="bootstrap root")
    if (
        isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or replicate_index < 0
    ):
        raise V021CollisionError("Bootstrap index must be a nonnegative integer")
    _component(family_id, context="bootstrap family")
    material = (
        f"{protocol_id}|{seed_root}|bootstrap|{replicate_index}|{family_id}"
    ).encode("ascii")
    return int(hashlib.sha256(material).hexdigest()[:16], 16) % _SEED_MODULUS


def derive_random_ranking_digest(
    *,
    seed_root: int,
    ranking_index: int,
    content_hash: str,
) -> str:
    """Apply the unchanged V2 random-ranking key formula."""

    _positive_integer(seed_root, context="random-ranking root")
    if (
        isinstance(ranking_index, bool)
        or not isinstance(ranking_index, int)
        or ranking_index < 0
    ):
        raise V021CollisionError("Ranking index must be a nonnegative integer")
    verified = _sha256(content_hash, context="random-policy content hash")
    material = f"{seed_root}|{ranking_index}|{verified}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def derive_analysis_tie_digest(
    protocol_id: str,
    *,
    arm: str,
    content_hash: str,
) -> str:
    """Apply the unchanged V2 policy/bootstrap/boundary tie formula."""

    _component(protocol_id, context="protocol ID")
    _component(arm, context="analysis arm")
    verified = _sha256(content_hash, context=f"{arm} content hash")
    return hashlib.sha256(f"{protocol_id}|{arm}|{verified}".encode("ascii")).hexdigest()


def derive_stress_permutation_digest(
    *,
    seed_root: int,
    permutation_index: int,
    family_id: str,
    random_policy_content_sha256: str,
) -> str:
    """Apply the unchanged V2 stress-permutation donor-key formula."""

    _positive_integer(seed_root, context="stress-permutation root")
    if (
        isinstance(permutation_index, bool)
        or not isinstance(permutation_index, int)
        or permutation_index < 0
    ):
        raise V021CollisionError("Permutation index must be a nonnegative integer")
    _component(family_id, context="stress-permutation family")
    verified = _sha256(
        random_policy_content_sha256,
        context="permutation random-policy content hash",
    )
    material = (f"{seed_root}|{permutation_index}|{family_id}|{verified}").encode(
        "ascii"
    )
    return hashlib.sha256(material).hexdigest()


_ACTUAL_HASH_DOMAIN_ORDER = (
    "predictor_content_hash",
    "analysis_tie_hash",
    "random_ranking_hash",
    "stress_permutation_hash",
)


class _ActualHashAccumulator:
    def __init__(self, *, protocol_id: str) -> None:
        self.protocol_id = protocol_id
        self.combined_hasher = hashlib.sha256()
        self.domain_hashers = {
            name: hashlib.sha256() for name in _ACTUAL_HASH_DOMAIN_ORDER
        }
        self.domain_counts = {name: 0 for name in _ACTUAL_HASH_DOMAIN_ORDER}

    def register(
        self,
        *,
        domain: str,
        coordinate: tuple[object, ...],
        material: tuple[object, ...],
        value: str,
    ) -> None:
        if domain not in self.domain_hashers:
            raise V021CollisionError("Internal actual-hash domain is invalid")
        _sha256(value, context=f"actual analysis hash/{domain}")
        record = {
            "coordinate": list(coordinate),
            "domain": domain,
            "material": list(material),
            "protocol_id": self.protocol_id,
            "value": value,
        }
        raw = _canonical_json_bytes(record) + b"\n"
        self.combined_hasher.update(raw)
        self.domain_hashers[domain].update(raw)
        self.domain_counts[domain] += 1


def _validate_expected_actual_groups(
    records: tuple[ActualAnalysisContentRecord, ...],
    expected_group_counts: tuple[tuple[str, str, int], ...] | None,
) -> None:
    if expected_group_counts is None:
        return
    if not isinstance(expected_group_counts, tuple) or not expected_group_counts:
        raise V021CollisionError(
            "Expected actual-hash groups must be a nonempty ordered tuple"
        )
    expected: dict[tuple[str, str], int] = {}
    for item in expected_group_counts:
        if not isinstance(item, tuple) or len(item) != 3:
            raise V021CollisionError(
                "Each expected actual-hash group must be partition/family/count"
            )
        partition, family_id, count = item
        key = (
            _component(partition, context="expected actual-hash partition"),
            _component(family_id, context="expected actual-hash family"),
        )
        if key in expected:
            raise V021CollisionError("Expected actual-hash groups contain duplicates")
        expected[key] = _count(count, context=f"expected actual-hash count/{key}")
    observed: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.partition, record.family_id)
        observed[key] = observed.get(key, 0) + 1
    if observed != expected:
        raise V021CollisionError(
            "Actual analysis content groups do not match the declared generation plan"
        )


def bind_actual_analysis_hash_ledger(
    *,
    protocol_id: str,
    random_ranking_root: int,
    stress_permutation_root: int,
    records: tuple[ActualAnalysisContentRecord, ...],
    ranking_partitions: tuple[str, ...],
    random_ranking_count: int,
    stress_partition: str,
    stress_families: tuple[str, ...],
    stress_permutation_count: int,
    tie_arms: tuple[str, ...] = ANALYSIS_TIE_ARMS,
    expected_group_counts: tuple[tuple[str, str, int], ...] | None = None,
) -> ActualAnalysisHashLedgerCommitment:
    """Bind every actual content-dependent analysis hash before prediction.

    Ranking partitions select independent row pools but are not part of the
    inherited ranking-key formula.  The ledger includes partition as an audit
    coordinate while deriving each value from root, index, and content hash.
    """

    canonical_protocol_id = _component(protocol_id, context="actual-hash protocol ID")
    ranking_root = _positive_integer(
        random_ranking_root,
        context="actual random-ranking root",
    )
    stress_root = _positive_integer(
        stress_permutation_root,
        context="actual stress-permutation root",
    )
    partitions = _validate_string_tuple(
        ranking_partitions,
        context="actual ranking partitions",
    )
    ranking_count = _count(
        random_ranking_count,
        context="actual random-ranking count",
    )
    canonical_stress_partition = _component(
        stress_partition,
        context="actual stress partition",
    )
    families = _validate_string_tuple(
        stress_families,
        context="actual stress families",
    )
    permutation_count = _count(
        stress_permutation_count,
        context="actual stress-permutation count",
    )
    arms = _validate_string_tuple(tie_arms, context="actual tie arms")
    if canonical_stress_partition not in partitions:
        raise V021CollisionError("Actual stress partition is not a ranking partition")
    if not isinstance(records, tuple) or not records:
        raise V021CollisionError(
            "Actual analysis content records must be a nonempty tuple"
        )
    if any(type(record) is not ActualAnalysisContentRecord for record in records):
        raise V021CollisionError("Actual analysis content record type is invalid")

    partition_order = {name: index for index, name in enumerate(partitions)}
    family_order = {name: index for index, name in enumerate(families)}
    keys: set[tuple[str, str, str]] = set()
    random_hashes: dict[tuple[str, str], str] = {}
    predictor_hashes: dict[tuple[str, str, str], str] = {}
    for record in records:
        if record.partition not in partition_order:
            raise V021CollisionError(
                "Actual analysis content uses an undeclared ranking partition"
            )
        if record.family_id not in family_order:
            raise V021CollisionError(
                "Actual analysis content uses an undeclared analysis family"
            )
        key = (record.partition, record.family_id, record.member_id)
        if key in keys:
            raise V021CollisionError("Actual analysis content coordinate is duplicated")
        keys.add(key)
        if tuple(arm for arm, _ in record.predictor_content_hashes) != arms:
            raise V021CollisionError(
                "Actual predictor arms do not exactly match the declared tie arms"
            )
        random_key = (record.partition, record.random_policy_content_sha256)
        if random_key in random_hashes:
            raise V021CollisionError(
                "Random-policy content is duplicated within a ranking partition"
            )
        random_hashes[random_key] = record.member_id
        for arm, digest in record.predictor_content_hashes:
            predictor_key = (record.partition, arm, digest)
            if predictor_key in predictor_hashes:
                raise V021CollisionError(
                    f"Predictor content is duplicated within {record.partition}/{arm}"
                )
            predictor_hashes[predictor_key] = record.member_id

    _validate_expected_actual_groups(records, expected_group_counts)
    if any(
        not any(record.partition == partition for record in records)
        for partition in partitions
    ):
        raise V021CollisionError("An actual random-ranking partition is absent")
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                partition_order[record.partition],
                family_order[record.family_id],
                record.member_id,
            ),
        )
    )
    accumulator = _ActualHashAccumulator(protocol_id=canonical_protocol_id)
    tie_hashes_by_scope: dict[tuple[str, str], set[str]] = {}

    for record in ordered:
        for arm, content_hash in record.predictor_content_hashes:
            coordinate = (
                record.partition,
                record.family_id,
                record.member_id,
                arm,
            )
            accumulator.register(
                domain="predictor_content_hash",
                coordinate=coordinate,
                material=("observed_predictor_content", content_hash),
                value=content_hash,
            )
            tie_hash = derive_analysis_tie_digest(
                canonical_protocol_id,
                arm=arm,
                content_hash=content_hash,
            )
            tie_scope = (record.partition, arm)
            tie_hashes = tie_hashes_by_scope.setdefault(tie_scope, set())
            if tie_hash in tie_hashes:
                raise V021CollisionError(
                    f"Analysis tie-hash collision within {record.partition}/{arm}"
                )
            tie_hashes.add(tie_hash)
            accumulator.register(
                domain="analysis_tie_hash",
                coordinate=coordinate,
                material=(canonical_protocol_id, arm, content_hash),
                value=tie_hash,
            )

    for partition in partitions:
        partition_records = tuple(
            record for record in ordered if record.partition == partition
        )
        for ranking_index in range(ranking_count):
            ranking_hashes: set[str] = set()
            for record in partition_records:
                content_hash = record.random_policy_content_sha256
                ranking_hash = derive_random_ranking_digest(
                    seed_root=ranking_root,
                    ranking_index=ranking_index,
                    content_hash=content_hash,
                )
                if ranking_hash in ranking_hashes:
                    raise V021CollisionError(
                        "Random-ranking hash collision within "
                        f"{partition}/ranking-{ranking_index}"
                    )
                ranking_hashes.add(ranking_hash)
                accumulator.register(
                    domain="random_ranking_hash",
                    coordinate=(
                        partition,
                        ranking_index,
                        record.family_id,
                        record.member_id,
                    ),
                    material=(ranking_root, ranking_index, content_hash),
                    value=ranking_hash,
                )

    stress_records_by_family = {
        family: tuple(
            record
            for record in ordered
            if record.partition == canonical_stress_partition
            and record.family_id == family
        )
        for family in families
    }
    if any(not family_records for family_records in stress_records_by_family.values()):
        raise V021CollisionError("An actual stress-permutation family is absent")
    for permutation_index in range(permutation_count):
        for family in families:
            permutation_hashes: set[str] = set()
            for record in stress_records_by_family[family]:
                content_hash = record.random_policy_content_sha256
                permutation_hash = derive_stress_permutation_digest(
                    seed_root=stress_root,
                    permutation_index=permutation_index,
                    family_id=family,
                    random_policy_content_sha256=content_hash,
                )
                if permutation_hash in permutation_hashes:
                    raise V021CollisionError(
                        "Stress-permutation hash collision within "
                        f"permutation-{permutation_index}/{family}"
                    )
                permutation_hashes.add(permutation_hash)
                accumulator.register(
                    domain="stress_permutation_hash",
                    coordinate=(
                        canonical_stress_partition,
                        permutation_index,
                        family,
                        record.member_id,
                    ),
                    material=(
                        stress_root,
                        permutation_index,
                        family,
                        content_hash,
                    ),
                    value=permutation_hash,
                )

    domains = tuple(
        ActualHashDomainCommitment(
            name=name,
            record_count=accumulator.domain_counts[name],
            ordered_ledger_sha256=accumulator.domain_hashers[name].hexdigest(),
        )
        for name in _ACTUAL_HASH_DOMAIN_ORDER
    )
    return ActualAnalysisHashLedgerCommitment(
        protocol_id=canonical_protocol_id,
        random_ranking_root=ranking_root,
        stress_permutation_root=stress_root,
        ranking_partitions=partitions,
        random_ranking_count=ranking_count,
        stress_partition=canonical_stress_partition,
        stress_families=families,
        stress_permutation_count=permutation_count,
        tie_arms=arms,
        source_record_count=len(ordered),
        ranking_partition_record_counts=tuple(
            (
                partition,
                sum(record.partition == partition for record in ordered),
            )
            for partition in partitions
        ),
        stress_family_record_counts=tuple(
            (family, len(stress_records_by_family[family])) for family in families
        ),
        ordered_ledger_sha256=accumulator.combined_hasher.hexdigest(),
        domains=domains,
    )


def _ordinary_identity(protocol_id: str, opaque_seed: int) -> tuple[str, str]:
    material = f"{protocol_id}|{opaque_seed}|opaque_cluster_id".encode("ascii")
    digest = hashlib.sha256(material).hexdigest()
    return "c_" + digest[:32], digest


def _pair_identity_values(
    protocol_id: str,
    *,
    seed_root: int,
    pair_index: int,
) -> tuple[tuple[str, str, str], tuple[tuple[str, str], ...]]:
    digests = tuple(
        (
            suffix,
            hashlib.sha256(
                f"{protocol_id}|{seed_root}|{pair_index}|{suffix}".encode("ascii")
            ).hexdigest(),
        )
        for suffix in ("opaque_pool|0", "opaque_pool|1", "opaque_swap", "pair_id")
    )
    by_suffix = dict(digests)
    pool = [
        "c_" + by_suffix["opaque_pool|0"][:32],
        "c_" + by_suffix["opaque_pool|1"][:32],
    ]
    if bytes.fromhex(by_suffix["opaque_swap"])[-1] & 1:
        pool.reverse()
    pair_id = "p_" + by_suffix["pair_id"][:32]
    return (pool[0], pool[1], pair_id), digests


def _spec_payload(spec: GenerationPlanSpec) -> dict[str, object]:
    return {
        "analysis": {
            "bootstrap": {
                "families": list(spec.bootstrap_families),
                "partition": spec.bootstrap_partition,
                "resamples": spec.bootstrap_resamples,
                "root_name": spec.bootstrap_seed_root_name,
            },
            "random_rankings": {
                "count": spec.random_ranking_count,
                "coordinate_semantics": (
                    "root_and_index_shared_across_partitions_content_hash_varies"
                ),
                "formula": "SHA256(root|ranking_index|content_hash)",
                "partitions": list(spec.ranking_partitions),
                "root_name": spec.random_ranking_root_name,
            },
            "stress_permutations": {
                "count": spec.stress_permutation_count,
                "families": list(spec.stress_families),
                "formula": (
                    "SHA256(root|permutation_index|family|random_policy_content_hash)"
                ),
                "partition": spec.stress_partition,
                "root_name": spec.stress_permutation_root_name,
            },
            "tie_breaking": {
                "arms": list(spec.analysis_tie_arms),
                "formula": "SHA256(protocol_id|arm|content_hash)",
            },
        },
        "pre_generation_scope": {
            "actual_content_hashes_available": False,
            "formula_witness_content_sha256": _DERIVATION_WITNESS_CONTENT_SHA256,
            "post_generation_binding_required": True,
        },
        "matched_groups": [
            {
                "pair_count": item.pair_count,
                "partition": item.partition,
                "seed_root_name": item.seed_root_name,
            }
            for item in spec.matched_groups
        ],
        "ordinary_groups": [
            {
                "count": item.count,
                "family_id": item.family_id,
                "partition": item.partition,
                "seed_root_name": item.seed_root_name,
            }
            for item in spec.ordinary_groups
        ],
        "placebo_seed_root_name": spec.placebo_seed_root_name,
        "protocol_byte_sha256": spec.protocol_byte_sha256,
        "protocol_id": spec.protocol_id,
        "protocol_semantic_sha256": spec.protocol_semantic_sha256,
        "seed_roots": [
            {"name": name, "value": value} for name, value in spec.seed_roots
        ],
    }


class _PlanAccumulator:
    def __init__(
        self,
        *,
        role: str,
        spec: GenerationPlanSpec,
        ledger: CoordinateCollisionLedger,
        combined_hasher: object,
    ) -> None:
        self.role = role
        self.spec = spec
        self.ledger = ledger
        self.combined_hasher = combined_hasher
        self.plan_hasher = hashlib.sha256()
        self.domain_hashers = {name: hashlib.sha256() for name in _DOMAIN_ORDER}
        self.domain_counts = {name: 0 for name in _DOMAIN_ORDER}
        self.kind_counts = {"seed": 0, "identifier": 0, "digest": 0}

    def register(
        self,
        *,
        kind: str,
        domain: str,
        coordinate: tuple[object, ...],
        value: int | str,
    ) -> None:
        if domain not in self.domain_hashers or kind not in self.kind_counts:
            raise V021CollisionError("Internal ledger domain or kind is invalid")
        coordinate_text = "/".join(str(item) for item in coordinate)
        label = f"{self.role}/{kind}/{coordinate_text}"
        if kind == "seed":
            if not isinstance(value, int):
                raise V021CollisionError("Internal seed ledger value is invalid")
            self.ledger.register_seed(label=label, seed=value)
        elif kind == "identifier":
            if not isinstance(value, str):
                raise V021CollisionError("Internal identifier ledger value is invalid")
            self.ledger.register_identifier(label=label, identifier=value)
        else:
            if not isinstance(value, str):
                raise V021CollisionError("Internal digest ledger value is invalid")
            self.ledger.register_digest(label=label, digest=value)
        record = {
            "coordinate": list(coordinate),
            "domain": domain,
            "kind": kind,
            "protocol_byte_sha256": self.spec.protocol_byte_sha256,
            "protocol_id": self.spec.protocol_id,
            "role": self.role,
            "value": value,
        }
        raw = _canonical_json_bytes(record) + b"\n"
        self.plan_hasher.update(raw)
        self.domain_hashers[domain].update(raw)
        self.combined_hasher.update(raw)
        self.domain_counts[domain] += 1
        self.kind_counts[kind] += 1


def _enumerate_plan(
    *,
    role: str,
    spec: GenerationPlanSpec,
    ledger: CoordinateCollisionLedger,
    combined_hasher: object,
) -> ProtocolPlanCommitment:
    roots = spec.seed_root_map()
    accumulator = _PlanAccumulator(
        role=role,
        spec=spec,
        ledger=ledger,
        combined_hasher=combined_hasher,
    )

    ordinary_clusters = 0
    for group in spec.ordinary_groups:
        for index in range(group.count):
            ordinary_clusters += 1
            opaque_seed = -1
            for stream in _ORDINARY_STREAMS:
                seed = derive_stream_seed(
                    spec.protocol_id,
                    seed_root=roots[group.seed_root_name],
                    partition=group.partition,
                    family_id=group.family_id,
                    zero_based_index=index,
                    stream_name=stream,
                )
                accumulator.register(
                    kind="seed",
                    domain="ordinary_stream_seed",
                    coordinate=(
                        "ordinary",
                        group.partition,
                        group.family_id,
                        index,
                        stream,
                    ),
                    value=seed,
                )
                if stream == "opaque_id":
                    opaque_seed = seed
            placebo_seed = derive_stream_seed(
                spec.protocol_id,
                seed_root=roots[spec.placebo_seed_root_name],
                partition=group.partition,
                family_id=group.family_id,
                zero_based_index=index,
                stream_name="placebo_covariates",
            )
            accumulator.register(
                kind="seed",
                domain="ordinary_stream_seed",
                coordinate=(
                    "ordinary",
                    group.partition,
                    group.family_id,
                    index,
                    "placebo_covariates",
                ),
                value=placebo_seed,
            )
            identifier, digest = _ordinary_identity(spec.protocol_id, opaque_seed)
            accumulator.register(
                kind="identifier",
                domain="ordinary_cluster_identifier",
                coordinate=(
                    "ordinary",
                    group.partition,
                    group.family_id,
                    index,
                    "cluster_id",
                ),
                value=identifier,
            )
            accumulator.register(
                kind="digest",
                domain="ordinary_identity_digest",
                coordinate=(
                    "ordinary",
                    group.partition,
                    group.family_id,
                    index,
                    "opaque_cluster_id",
                ),
                value=digest,
            )

    matched_pairs = 0
    for group in spec.matched_groups:
        root = roots[group.seed_root_name]
        for pair_index in range(group.pair_count):
            matched_pairs += 1
            for stream in _MATCHED_STREAMS:
                accumulator.register(
                    kind="seed",
                    domain="matched_stream_seed",
                    coordinate=(
                        "matched",
                        group.partition,
                        pair_index,
                        stream,
                    ),
                    value=derive_stream_seed(
                        spec.protocol_id,
                        seed_root=root,
                        partition=group.partition,
                        family_id=group.partition,
                        zero_based_index=pair_index,
                        stream_name=stream,
                    ),
                )
            identifiers, digests = _pair_identity_values(
                spec.protocol_id,
                seed_root=root,
                pair_index=pair_index,
            )
            for name, identifier in zip(
                ("left_cluster_id", "right_cluster_id", "pair_id"),
                identifiers,
                strict=True,
            ):
                accumulator.register(
                    kind="identifier",
                    domain="matched_identifier",
                    coordinate=("matched", group.partition, pair_index, name),
                    value=identifier,
                )
            for suffix, digest in digests:
                accumulator.register(
                    kind="digest",
                    domain="matched_identity_digest",
                    coordinate=(
                        "matched",
                        group.partition,
                        pair_index,
                        suffix.replace("|", "_"),
                    ),
                    value=digest,
                )

    bootstrap_coordinates = 0
    bootstrap_root = roots[spec.bootstrap_seed_root_name]
    for replicate_index in range(spec.bootstrap_resamples):
        for family_id in spec.bootstrap_families:
            bootstrap_coordinates += 1
            accumulator.register(
                kind="seed",
                domain="bootstrap_seed",
                coordinate=(
                    "analysis",
                    "bootstrap",
                    replicate_index,
                    family_id,
                ),
                value=derive_bootstrap_seed(
                    spec.protocol_id,
                    seed_root=bootstrap_root,
                    replicate_index=replicate_index,
                    family_id=family_id,
                ),
            )

    ranking_root = roots[spec.random_ranking_root_name]
    for ranking_index in range(spec.random_ranking_count):
        accumulator.register(
            kind="digest",
            domain="random_ranking_formula_witness",
            coordinate=(
                "analysis",
                "random_rankings",
                ranking_index,
                "formula_witness",
            ),
            value=derive_random_ranking_digest(
                seed_root=ranking_root,
                ranking_index=ranking_index,
                content_hash=_DERIVATION_WITNESS_CONTENT_SHA256,
            ),
        )

    stress_coordinates = 0
    stress_root = roots[spec.stress_permutation_root_name]
    for permutation_index in range(spec.stress_permutation_count):
        for family_id in spec.stress_families:
            stress_coordinates += 1
            accumulator.register(
                kind="digest",
                domain="stress_permutation_formula_witness",
                coordinate=(
                    "analysis",
                    "stress_permutations",
                    permutation_index,
                    family_id,
                    "formula_witness",
                ),
                value=derive_stress_permutation_digest(
                    seed_root=stress_root,
                    permutation_index=permutation_index,
                    family_id=family_id,
                    random_policy_content_sha256=(_DERIVATION_WITNESS_CONTENT_SHA256),
                ),
            )

    for arm in spec.analysis_tie_arms:
        accumulator.register(
            kind="digest",
            domain="analysis_tie_formula_witness",
            coordinate=("analysis", "tie", arm, "formula_witness"),
            value=derive_analysis_tie_digest(
                spec.protocol_id,
                arm=arm,
                content_hash=_DERIVATION_WITNESS_CONTENT_SHA256,
            ),
        )

    counts = PlanCounts(
        ordinary_clusters=ordinary_clusters,
        matched_pairs=matched_pairs,
        generated_members=ordinary_clusters + 2 * matched_pairs,
        bootstrap_coordinates=bootstrap_coordinates,
        random_ranking_coordinates=spec.random_ranking_count,
        stress_permutation_coordinates=stress_coordinates,
        analysis_tie_coordinates=len(spec.analysis_tie_arms),
        seed_count=accumulator.kind_counts["seed"],
        identifier_count=accumulator.kind_counts["identifier"],
        digest_count=accumulator.kind_counts["digest"],
        ordered_ledger_records=sum(accumulator.kind_counts.values()),
    )
    domains = tuple(
        DomainCommitment(
            name=name,
            record_count=accumulator.domain_counts[name],
            ordered_ledger_sha256=accumulator.domain_hashers[name].hexdigest(),
        )
        for name in _DOMAIN_ORDER
    )
    if sum(item.record_count for item in domains) != counts.ordered_ledger_records:
        raise V021CollisionError("Expanded ledger domain counts are inconsistent")
    return ProtocolPlanCommitment(
        protocol_id=spec.protocol_id,
        protocol_byte_sha256=spec.protocol_byte_sha256,
        protocol_semantic_sha256=spec.protocol_semantic_sha256,
        seed_roots=spec.seed_roots,
        coordinate_spec_sha256=_hash_payload(_spec_payload(spec)),
        ordered_ledger_sha256=accumulator.plan_hasher.hexdigest(),
        counts=counts,
        domains=domains,
    )


def commit_generation_coordinate_namespaces(
    *,
    current: GenerationPlanSpec,
    predecessor: GenerationPlanSpec,
) -> GenerationCoordinateNamespaceCommitment:
    """Commit two complete plans without claiming generated-content comparison."""

    if type(current) is not GenerationPlanSpec:
        raise V021CollisionError("Current plan has an invalid type")
    if type(predecessor) is not GenerationPlanSpec:
        raise V021CollisionError("Predecessor plan has an invalid type")
    _validate_plan_spec(current)
    _validate_plan_spec(predecessor)
    if current.protocol_id == predecessor.protocol_id:
        raise V021CollisionError("Current and predecessor protocol IDs must differ")
    if current.protocol_byte_sha256 == predecessor.protocol_byte_sha256:
        raise V021CollisionError("Current and predecessor byte identities collide")
    if current.protocol_semantic_sha256 == predecessor.protocol_semantic_sha256:
        raise V021CollisionError("Current and predecessor semantic identities collide")
    if set(current.seed_root_map().values()).intersection(
        predecessor.seed_root_map().values()
    ):
        raise V021CollisionError("Current and predecessor seed roots overlap")

    ledger = CoordinateCollisionLedger()
    combined_hasher = hashlib.sha256()
    current_commitment = _enumerate_plan(
        role="current",
        spec=current,
        ledger=ledger,
        combined_hasher=combined_hasher,
    )
    predecessor_commitment = _enumerate_plan(
        role="predecessor",
        spec=predecessor,
        ledger=ledger,
        combined_hasher=combined_hasher,
    )
    record_count = (
        current_commitment.counts.ordered_ledger_records
        + predecessor_commitment.counts.ordered_ledger_records
    )
    if (
        ledger.coordinate_count != record_count
        or ledger.seed_count
        != current_commitment.counts.seed_count
        + predecessor_commitment.counts.seed_count
        or ledger.identifier_count
        != current_commitment.counts.identifier_count
        + predecessor_commitment.counts.identifier_count
        or ledger.digest_count
        != current_commitment.counts.digest_count
        + predecessor_commitment.counts.digest_count
    ):
        raise V021CollisionError(
            "Cross-protocol coordinate namespace registries are inconsistent"
        )
    return GenerationCoordinateNamespaceCommitment(
        current=current_commitment,
        predecessor=predecessor_commitment,
        complete_ordered_ledger_sha256=combined_hasher.hexdigest(),
        complete_ordered_ledger_records=record_count,
    )


def audit_generation_coordinate_plans(
    *,
    current: GenerationPlanSpec,
    predecessor: GenerationPlanSpec,
) -> GenerationCoordinateNamespaceCommitment:
    """Compatibility wrapper for the honestly scoped namespace commitment."""

    return commit_generation_coordinate_namespaces(
        current=current,
        predecessor=predecessor,
    )


def _validate_formal_analysis_plan_lock(
    current: GenerationPlanSpec,
    predecessor: GenerationPlanSpec,
) -> None:
    for role, spec in (("current", current), ("predecessor", predecessor)):
        if spec.bootstrap_resamples != _FORMAL_BOOTSTRAP_RESAMPLES:
            raise V021CollisionError(f"Formal {role} bootstrap count changed")
        if spec.random_ranking_count != _FORMAL_RANDOM_RANKINGS:
            raise V021CollisionError(f"Formal {role} random-ranking count changed")
        if spec.stress_permutation_count != _FORMAL_STRESS_PERMUTATIONS:
            raise V021CollisionError(f"Formal {role} stress-permutation count changed")
        if spec.ranking_partitions != _FORMAL_RANKING_PARTITIONS:
            raise V021CollisionError(f"Formal {role} ranking partitions changed")
        if spec.bootstrap_partition != "test" or spec.stress_partition != "test":
            raise V021CollisionError(f"Formal {role} analysis partition changed")
        if (
            spec.bootstrap_families != _FORMAL_TEST_FAMILIES
            or spec.stress_families != _FORMAL_TEST_FAMILIES
        ):
            raise V021CollisionError(f"Formal {role} analysis families changed")
        if spec.analysis_tie_arms != ANALYSIS_TIE_ARMS:
            raise V021CollisionError(f"Formal {role} tie-arm families changed")


def build_formal_plan_specs(
    view: V021ContractView | None = None,
) -> tuple[GenerationPlanSpec, GenerationPlanSpec]:
    """Build, but do not enumerate, the frozen-size V2.1 and V2 plans."""

    contract = load_v021_contract_view() if view is None else view
    if type(contract) is not V021ContractView:
        raise V021CollisionError("Formal contract view has an invalid type")
    protocol = contract.protocol
    if protocol.protocol_id != V021_PROTOCOL_ID:
        raise V021CollisionError("Formal current protocol identity is not V2.1")
    if protocol.config_sha256 != contract.artifacts.config_byte_sha256:
        raise V021CollisionError("Formal V2.1 amendment commitments disagree")
    current_roots = tuple(protocol.seed_roots)
    if current_roots != tuple(V021_EXPECTED_SEED_ROOTS.items()):
        raise V021CollisionError("Formal V2.1 seed roots changed")
    predecessor_roots = tuple(
        zip(
            (name for name, _ in current_roots),
            V2_SEED_ROOTS,
            strict=True,
        )
    )
    if contract.base_config_byte_sha256 == protocol.config_sha256:
        raise V021CollisionError("Formal current and predecessor hashes collide")

    ordinary_groups = tuple(
        OrdinaryPlanGroup(
            partition=partition,
            family_id=family_id,
            count=count,
            seed_root_name=(
                f"novel_mechanism_{partition}"
                if family_id in _NOVEL_FAMILIES
                else partition
            ),
        )
        for partition, family_counts in protocol.partition_family_counts
        for family_id, count in family_counts
    )
    matched_groups = tuple(
        MatchedPlanGroup(
            partition=partition,
            pair_count=_FORMAL_MATCHED_PAIR_COUNT,
            seed_root_name=partition,
        )
        for partition in _MATCHED_PARTITIONS
    )
    test_families = tuple(
        item.family_id for item in ordinary_groups if item.partition == "test"
    )
    if test_families != _FORMAL_TEST_FAMILIES:
        raise V021CollisionError("Formal V2.1 test-family registry changed")

    common = {
        "ordinary_groups": ordinary_groups,
        "matched_groups": matched_groups,
        "bootstrap_partition": "test",
        "bootstrap_families": test_families,
        "bootstrap_resamples": _FORMAL_BOOTSTRAP_RESAMPLES,
        "ranking_partitions": _FORMAL_RANKING_PARTITIONS,
        "random_ranking_count": _FORMAL_RANDOM_RANKINGS,
        "stress_partition": "test",
        "stress_families": test_families,
        "stress_permutation_count": _FORMAL_STRESS_PERMUTATIONS,
        "analysis_tie_arms": ANALYSIS_TIE_ARMS,
    }
    current = GenerationPlanSpec(
        protocol_id=protocol.protocol_id,
        protocol_byte_sha256=protocol.config_sha256,
        protocol_semantic_sha256=V021_AMENDMENT_SEMANTIC_SHA256,
        seed_roots=current_roots,
        **common,
    )
    predecessor = GenerationPlanSpec(
        protocol_id=_V2_PROTOCOL_ID,
        protocol_byte_sha256=contract.base_config_byte_sha256,
        protocol_semantic_sha256=contract.base_config_canonical_sha256,
        seed_roots=predecessor_roots,
        **common,
    )
    _validate_formal_analysis_plan_lock(current, predecessor)
    return current, predecessor


def audit_formal_v021_generation_plan(
    view: V021ContractView | None = None,
) -> GenerationCoordinateNamespaceCommitment:
    """Run the full formal namespace commitment only after freeze."""

    current, predecessor = build_formal_plan_specs(view)
    _validate_formal_analysis_plan_lock(current, predecessor)
    return commit_generation_coordinate_namespaces(
        current=current,
        predecessor=predecessor,
    )


def bind_formal_v021_actual_analysis_hash_ledger(
    records: tuple[ActualAnalysisContentRecord, ...],
    view: V021ContractView | None = None,
) -> ActualAnalysisHashLedgerCommitment:
    """Bind the full frozen actual-hash ledger after generation, before prediction."""

    current, predecessor = build_formal_plan_specs(view)
    _validate_formal_analysis_plan_lock(current, predecessor)
    roots = current.seed_root_map()
    expected_groups = tuple(
        (group.partition, group.family_id, group.count)
        for group in current.ordinary_groups
        if group.partition in _FORMAL_RANKING_PARTITIONS
    )
    return bind_actual_analysis_hash_ledger(
        protocol_id=current.protocol_id,
        random_ranking_root=roots[current.random_ranking_root_name],
        stress_permutation_root=roots[current.stress_permutation_root_name],
        records=records,
        ranking_partitions=_FORMAL_RANKING_PARTITIONS,
        random_ranking_count=_FORMAL_RANDOM_RANKINGS,
        stress_partition="test",
        stress_families=_FORMAL_TEST_FAMILIES,
        stress_permutation_count=_FORMAL_STRESS_PERMUTATIONS,
        tie_arms=ANALYSIS_TIE_ARMS,
        expected_group_counts=expected_groups,
    )


def verify_generation_coordinate_namespace_commitment(
    payload: Mapping[str, object],
    *,
    expected_byte_sha256: str,
    expected_current_protocol_id: str | None = None,
    expected_current_protocol_byte_sha256: str | None = None,
    expected_predecessor_protocol_id: str | None = None,
    expected_predecessor_protocol_byte_sha256: str | None = None,
) -> str:
    """Verify a detached pre-generation namespace commitment."""

    expected_hash = _sha256(
        expected_byte_sha256,
        context="expected commitment byte hash",
    )
    if not isinstance(payload, Mapping):
        raise V021CollisionError(
            "Generation coordinate namespace commitment must be an object"
        )
    if set(payload) != {
        "cross_protocol_coordinate_namespace",
        "current",
        "predecessor",
        "schema_version",
    }:
        raise V021CollisionError(
            "Generation coordinate namespace commitment keys changed"
        )
    if payload.get("schema_version") != _COMMITMENT_SCHEMA:
        raise V021CollisionError(
            "Generation coordinate namespace commitment schema changed"
        )
    current = payload.get("current")
    predecessor = payload.get("predecessor")
    cross = payload.get("cross_protocol_coordinate_namespace")
    if not all(isinstance(item, Mapping) for item in (current, predecessor, cross)):
        raise V021CollisionError("Generation-plan commitment sections are invalid")
    assert isinstance(current, Mapping)
    assert isinstance(predecessor, Mapping)
    assert isinstance(cross, Mapping)

    expected_plan_keys = {
        "coordinate_domains",
        "coordinate_spec_sha256",
        "counts",
        "ordered_namespace_ledger_sha256",
        "protocol_byte_sha256",
        "protocol_id",
        "protocol_semantic_sha256",
        "seed_roots",
    }
    for name, section in (("current", current), ("predecessor", predecessor)):
        if set(section) != expected_plan_keys:
            raise V021CollisionError(f"{name} plan commitment keys changed")
        _component(section.get("protocol_id"), context=f"{name} protocol ID")
        _sha256(
            section.get("protocol_byte_sha256"),
            context=f"{name} protocol byte hash",
        )
        _sha256(
            section.get("protocol_semantic_sha256"),
            context=f"{name} protocol semantic hash",
        )
        _sha256(
            section.get("coordinate_spec_sha256"),
            context=f"{name} coordinate-spec hash",
        )
        _sha256(
            section.get("ordered_namespace_ledger_sha256"),
            context=f"{name} namespace-ledger hash",
        )
        if not isinstance(section.get("counts"), Mapping):
            raise V021CollisionError(f"{name} counts are invalid")
        if not isinstance(section.get("coordinate_domains"), list):
            raise V021CollisionError(f"{name} coordinate domains are invalid")
        if not isinstance(section.get("seed_roots"), list):
            raise V021CollisionError(f"{name} seed roots are invalid")

    expected_cross_keys = {
        "complete_ordered_namespace_ledger_records",
        "complete_ordered_namespace_ledger_sha256",
        "content_dependent_hash_comparison",
        "formula_witness_content_sha256",
        "ledger_order",
        "scope",
    }
    if set(cross) != expected_cross_keys:
        raise V021CollisionError("Cross-protocol namespace commitment keys changed")
    if (
        cross.get("content_dependent_hash_comparison")
        != "deferred_until_post_generation"
    ):
        raise V021CollisionError(
            "Content-dependent comparison scope is not honestly deferred"
        )
    if (
        cross.get("formula_witness_content_sha256")
        != _DERIVATION_WITNESS_CONTENT_SHA256
    ):
        raise V021CollisionError("Formula-witness content hash changed")
    if cross.get("ledger_order") != "current_then_predecessor":
        raise V021CollisionError("Complete namespace ledger order changed")
    if cross.get("scope") != (
        "actual_seeds_identifiers_identity_digests_and_"
        "content_independent_formula_witnesses"
    ):
        raise V021CollisionError("Pre-generation namespace scope changed")
    _sha256(
        cross.get("complete_ordered_namespace_ledger_sha256"),
        context="complete ordered namespace-ledger hash",
    )
    records = cross.get("complete_ordered_namespace_ledger_records")
    if isinstance(records, bool) or not isinstance(records, int) or records < 1:
        raise V021CollisionError("Complete ledger record count is invalid")

    expected_identities = (
        (
            "current",
            current,
            expected_current_protocol_id,
            expected_current_protocol_byte_sha256,
        ),
        (
            "predecessor",
            predecessor,
            expected_predecessor_protocol_id,
            expected_predecessor_protocol_byte_sha256,
        ),
    )
    for name, section, protocol_id, protocol_hash in expected_identities:
        if protocol_id is not None and section.get("protocol_id") != protocol_id:
            raise V021CollisionError(f"{name} protocol identity changed")
        if protocol_hash is not None:
            _sha256(protocol_hash, context=f"expected {name} protocol byte hash")
            if section.get("protocol_byte_sha256") != protocol_hash:
                raise V021CollisionError(f"{name} protocol byte identity changed")

    observed = hashlib.sha256(_canonical_json_bytes(payload) + b"\n").hexdigest()
    if observed != expected_hash:
        raise V021CollisionError(
            "Generation coordinate namespace commitment bytes changed"
        )
    return observed


def verify_generation_plan_commitment(
    payload: Mapping[str, object],
    *,
    expected_byte_sha256: str,
    expected_current_protocol_id: str | None = None,
    expected_current_protocol_byte_sha256: str | None = None,
    expected_predecessor_protocol_id: str | None = None,
    expected_predecessor_protocol_byte_sha256: str | None = None,
) -> str:
    """Compatibility wrapper for the renamed namespace verifier."""

    return verify_generation_coordinate_namespace_commitment(
        payload,
        expected_byte_sha256=expected_byte_sha256,
        expected_current_protocol_id=expected_current_protocol_id,
        expected_current_protocol_byte_sha256=(expected_current_protocol_byte_sha256),
        expected_predecessor_protocol_id=expected_predecessor_protocol_id,
        expected_predecessor_protocol_byte_sha256=(
            expected_predecessor_protocol_byte_sha256
        ),
    )


def verify_actual_analysis_hash_ledger_commitment(
    payload: Mapping[str, object],
    *,
    expected_byte_sha256: str,
    expected_protocol_id: str | None = None,
    expected_random_ranking_root: int | None = None,
    expected_stress_permutation_root: int | None = None,
) -> str:
    """Verify a detached post-generation actual-hash ledger commitment."""

    expected_hash = _sha256(
        expected_byte_sha256,
        context="expected actual-hash commitment byte hash",
    )
    if not isinstance(payload, Mapping):
        raise V021CollisionError("Actual analysis hash commitment must be an object")
    expected_keys = {
        "coordinate_semantics",
        "domains",
        "ledger_counts",
        "ordered_ledger_sha256",
        "protocol_id",
        "random_ranking_count",
        "random_ranking_root",
        "ranking_partitions",
        "schema_version",
        "source_coordinate_counts",
        "stage",
        "stress_families",
        "stress_partition",
        "stress_permutation_count",
        "stress_permutation_root",
        "tie_arms",
    }
    if set(payload) != expected_keys:
        raise V021CollisionError("Actual analysis hash commitment keys changed")
    if payload.get("schema_version") != _ACTUAL_HASH_COMMITMENT_SCHEMA:
        raise V021CollisionError("Actual analysis hash commitment schema changed")
    if payload.get("stage") != "post_generation_pre_prediction":
        raise V021CollisionError("Actual analysis hash commitment stage changed")

    protocol_id = _component(
        payload.get("protocol_id"),
        context="actual-hash commitment protocol ID",
    )
    random_root = _positive_integer(
        payload.get("random_ranking_root"),
        context="actual-hash commitment random root",
    )
    stress_root = _positive_integer(
        payload.get("stress_permutation_root"),
        context="actual-hash commitment stress root",
    )
    ranking_count = _count(
        payload.get("random_ranking_count"),
        context="actual-hash commitment ranking count",
    )
    permutation_count = _count(
        payload.get("stress_permutation_count"),
        context="actual-hash commitment permutation count",
    )
    ranking_partitions = payload.get("ranking_partitions")
    stress_families = payload.get("stress_families")
    tie_arms = payload.get("tie_arms")
    if not all(
        isinstance(item, list)
        for item in (ranking_partitions, stress_families, tie_arms)
    ):
        raise V021CollisionError("Actual analysis hash registries are invalid")
    assert isinstance(ranking_partitions, list)
    assert isinstance(stress_families, list)
    assert isinstance(tie_arms, list)
    partitions = _validate_string_tuple(
        tuple(ranking_partitions),
        context="committed ranking partitions",
    )
    families = _validate_string_tuple(
        tuple(stress_families),
        context="committed stress families",
    )
    arms = _validate_string_tuple(
        tuple(tie_arms),
        context="committed tie arms",
    )
    stress_partition = _component(
        payload.get("stress_partition"),
        context="committed stress partition",
    )
    if stress_partition not in partitions:
        raise V021CollisionError("Committed stress partition is not ranked")

    semantics = payload.get("coordinate_semantics")
    if semantics != {
        "random_ranking": (
            "partition_selects_pool_only_root_index_and_content_hash_define_the_key"
        ),
        "stress_permutation": (
            "root_permutation_index_family_and_random_policy_content_"
            "hash_define_the_key"
        ),
        "tie_breaking": "protocol_id_arm_and_predictor_content_hash",
    }:
        raise V021CollisionError("Actual analysis hash coordinate semantics changed")

    domains = payload.get("domains")
    if not isinstance(domains, list) or len(domains) != len(_ACTUAL_HASH_DOMAIN_ORDER):
        raise V021CollisionError("Actual analysis hash domains are invalid")
    domain_counts: dict[str, int] = {}
    for expected_name, section in zip(
        _ACTUAL_HASH_DOMAIN_ORDER,
        domains,
        strict=True,
    ):
        if not isinstance(section, Mapping) or set(section) != {
            "name",
            "ordered_ledger_sha256",
            "record_count",
        }:
            raise V021CollisionError("Actual analysis hash domain keys changed")
        if section.get("name") != expected_name:
            raise V021CollisionError("Actual analysis hash domain order changed")
        _sha256(
            section.get("ordered_ledger_sha256"),
            context=f"actual analysis domain hash/{expected_name}",
        )
        count = section.get("record_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise V021CollisionError("Actual analysis hash domain count is invalid")
        domain_counts[expected_name] = count

    ledger_counts = payload.get("ledger_counts")
    expected_count_keys = {
        *_ACTUAL_HASH_DOMAIN_ORDER,
        "complete_ordered_ledger_records",
        "source_content_records",
    }
    if not isinstance(ledger_counts, Mapping) or set(ledger_counts) != (
        expected_count_keys
    ):
        raise V021CollisionError("Actual analysis ledger counts changed")
    for name, count in domain_counts.items():
        if ledger_counts.get(name) != count:
            raise V021CollisionError("Actual analysis domain counts disagree")
    source_count = ledger_counts.get("source_content_records")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count < 1
    ):
        raise V021CollisionError("Actual analysis source count is invalid")
    if ledger_counts.get("complete_ordered_ledger_records") != sum(
        domain_counts.values()
    ):
        raise V021CollisionError("Actual analysis total ledger count disagrees")
    if domain_counts["predictor_content_hash"] != source_count * len(arms):
        raise V021CollisionError("Actual predictor content count disagrees")
    if domain_counts["analysis_tie_hash"] != source_count * len(arms):
        raise V021CollisionError("Actual tie-hash count disagrees")
    if domain_counts["random_ranking_hash"] != source_count * ranking_count:
        raise V021CollisionError("Actual random-ranking hash count disagrees")

    source_counts = payload.get("source_coordinate_counts")
    if not isinstance(source_counts, Mapping) or set(source_counts) != {
        "ranking_partitions",
        "stress_families",
    }:
        raise V021CollisionError("Actual analysis source-coordinate counts changed")
    partition_sections = source_counts.get("ranking_partitions")
    family_sections = source_counts.get("stress_families")
    if not isinstance(partition_sections, list) or not isinstance(
        family_sections,
        list,
    ):
        raise V021CollisionError("Actual analysis source counts are invalid")
    partition_total = 0
    observed_partitions: list[str] = []
    for section in partition_sections:
        if not isinstance(section, Mapping) or set(section) != {
            "partition",
            "record_count",
        }:
            raise V021CollisionError("Actual ranking-partition counts are invalid")
        observed_partitions.append(
            _component(
                section.get("partition"),
                context="committed ranking-count partition",
            )
        )
        partition_total += _count(
            section.get("record_count"),
            context="committed ranking-partition record count",
        )
    if tuple(observed_partitions) != partitions or partition_total != source_count:
        raise V021CollisionError("Actual ranking-partition counts disagree")

    stress_total = 0
    observed_families: list[str] = []
    for section in family_sections:
        if not isinstance(section, Mapping) or set(section) != {
            "family_id",
            "record_count",
        }:
            raise V021CollisionError("Actual stress-family counts are invalid")
        observed_families.append(
            _component(
                section.get("family_id"),
                context="committed stress-count family",
            )
        )
        stress_total += _count(
            section.get("record_count"),
            context="committed stress-family record count",
        )
    if tuple(observed_families) != families:
        raise V021CollisionError("Actual stress-family order changed")
    if domain_counts["stress_permutation_hash"] != stress_total * permutation_count:
        raise V021CollisionError("Actual stress-permutation hash count disagrees")

    _sha256(
        payload.get("ordered_ledger_sha256"),
        context="actual analysis ordered-ledger hash",
    )
    if expected_protocol_id is not None and protocol_id != expected_protocol_id:
        raise V021CollisionError("Actual analysis protocol identity changed")
    if (
        expected_random_ranking_root is not None
        and random_root != expected_random_ranking_root
    ):
        raise V021CollisionError("Actual analysis random-ranking root changed")
    if (
        expected_stress_permutation_root is not None
        and stress_root != expected_stress_permutation_root
    ):
        raise V021CollisionError("Actual analysis stress-permutation root changed")

    observed = hashlib.sha256(_canonical_json_bytes(payload) + b"\n").hexdigest()
    if observed != expected_hash:
        raise V021CollisionError("Actual analysis hash commitment bytes changed")
    return observed


__all__ = [
    "ANALYSIS_TIE_ARMS",
    "ActualAnalysisContentRecord",
    "ActualAnalysisHashLedgerCommitment",
    "ActualHashDomainCommitment",
    "CoordinateCollisionLedger",
    "DomainCommitment",
    "GenerationCoordinateNamespaceCommitment",
    "GenerationPlanCommitment",
    "GenerationPlanSpec",
    "MatchedPlanGroup",
    "OrdinaryPlanGroup",
    "PlanCounts",
    "ProtocolPlanCommitment",
    "V021CollisionError",
    "audit_formal_v021_generation_plan",
    "audit_generation_coordinate_plans",
    "bind_actual_analysis_hash_ledger",
    "bind_formal_v021_actual_analysis_hash_ledger",
    "build_formal_plan_specs",
    "commit_generation_coordinate_namespaces",
    "derive_analysis_tie_digest",
    "derive_bootstrap_seed",
    "derive_random_ranking_digest",
    "derive_stream_seed",
    "derive_stress_permutation_digest",
    "verify_actual_analysis_hash_ledger_commitment",
    "verify_generation_coordinate_namespace_commitment",
    "verify_generation_plan_commitment",
]
