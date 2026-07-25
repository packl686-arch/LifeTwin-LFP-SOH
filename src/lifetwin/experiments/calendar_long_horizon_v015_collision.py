"""Global seed, identity, and predictor-content collision checks for V0.15."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Mapping

from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    MATCHED_PARTITIONS,
    ValidatedV015Protocol,
    _pair_ids,
    _pair_stream_seed,
    derive_ordinary_cluster_id,
    derive_ordinary_stream_seeds,
)


class V015CollisionError(ValueError):
    """Raised when independently named formal values collide."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MATCHED_STREAMS = (
    "shared_truth",
    "shared_operating",
    "shared_measurement",
    "placebo",
    "opaque_ids",
    "opaque_swap",
)
_BOOTSTRAP_RESAMPLES = 5_000


@dataclass
class CollisionLedger:
    """Track globally unique streams/IDs and hash-to-content consistency."""

    _seeds: dict[int, str] = field(default_factory=dict)
    _identifiers: dict[str, str] = field(default_factory=dict)
    _content_by_hash: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def register_seed(self, *, label: str, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise V015CollisionError(f"{label} is not a valid integer seed")
        prior = self._seeds.get(seed)
        if prior is not None:
            raise V015CollisionError(f"Seed collision between {prior!r} and {label!r}")
        self._seeds[seed] = label

    def register_identifier(self, *, label: str, identifier: str) -> None:
        if not isinstance(identifier, str) or not identifier:
            raise V015CollisionError(f"{label} is not a nonempty identifier")
        prior = self._identifiers.get(identifier)
        if prior is not None:
            raise V015CollisionError(
                f"Identifier collision between {prior!r} and {label!r}"
            )
        self._identifiers[identifier] = label

    def register_content_hash(
        self,
        *,
        namespace: str,
        digest: str,
        canonical_content: bytes,
        unique_content_required: bool,
    ) -> None:
        if not namespace or _SHA256.fullmatch(digest) is None:
            raise V015CollisionError("Content-hash namespace or digest is invalid")
        if not isinstance(canonical_content, bytes):
            raise V015CollisionError("Canonical predictor content must be bytes")
        observed = hashlib.sha256(canonical_content).hexdigest()
        if observed != digest:
            raise V015CollisionError("Declared content hash does not match its bytes")
        key = (namespace, digest)
        prior = self._content_by_hash.get(key)
        if prior is not None:
            if prior != canonical_content:
                raise V015CollisionError(
                    "Distinct predictor content produced the same SHA256"
                )
            if unique_content_required:
                raise V015CollisionError(
                    "Duplicate predictor content is forbidden in this pool"
                )
            return
        self._content_by_hash[key] = canonical_content

    @property
    def seed_count(self) -> int:
        return len(self._seeds)

    @property
    def identifier_count(self) -> int:
        return len(self._identifiers)

    @property
    def content_hash_count(self) -> int:
        return len(self._content_by_hash)


@dataclass(frozen=True)
class GenerationCollisionSummary:
    seed_count: int
    cluster_id_count: int
    pair_id_count: int
    identity_digest_count: int


def derive_bootstrap_analysis_seed(
    protocol_id: str,
    *,
    replicate_index: int,
    family_id: str,
    seed_root: int,
) -> int:
    """Apply the bootstrap-specific seed formula frozen in endpoint V2."""

    if (
        not isinstance(protocol_id, str)
        or not protocol_id
        or isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or not 0 <= replicate_index < _BOOTSTRAP_RESAMPLES
        or not isinstance(family_id, str)
        or not family_id
        or isinstance(seed_root, bool)
        or not isinstance(seed_root, int)
        or seed_root < 1
    ):
        raise V015CollisionError("Bootstrap seed coordinate is invalid")
    material = (
        f"{protocol_id}|{seed_root}|bootstrap|{replicate_index}|{family_id}"
    ).encode("ascii")
    return int(hashlib.sha256(material).hexdigest()[:16], 16) % (2**63 - 1)


def _identity_digest(
    protocol: ValidatedV015Protocol,
    *,
    partition: str,
    pair_index: int,
    suffix: str,
) -> str:
    root = protocol.seed_root_map()[partition]
    material = f"{protocol.protocol_id}|{root}|{pair_index}|{suffix}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def audit_generation_coordinate_plan(
    protocol: ValidatedV015Protocol,
) -> GenerationCollisionSummary:
    """Audit every frozen coordinate without consuming an RNG.

    This function is intended to run once after implementation freeze and
    immediately before the isolated formal generator.  Unit tests exercise the
    ledger with hand-written values and do not call this formal-plan function.
    """

    ledger = CollisionLedger()
    cluster_ids = 0
    pair_ids = 0
    identity_digests: dict[str, str] = {}

    for partition, family_counts in protocol.partition_count_map().items():
        for family_id, count in family_counts.items():
            for index in range(count):
                coordinate = f"ordinary/{partition}/{family_id}/{index}"
                for stream, seed in derive_ordinary_stream_seeds(
                    protocol,
                    partition=partition,
                    family_id=family_id,
                    zero_based_index=index,
                ).items():
                    ledger.register_seed(label=f"{coordinate}/{stream}", seed=seed)
                ledger.register_identifier(
                    label=f"{coordinate}/cluster_id",
                    identifier=derive_ordinary_cluster_id(
                        protocol,
                        partition=partition,
                        family_id=family_id,
                        zero_based_index=index,
                    ),
                )
                cluster_ids += 1

    for partition in MATCHED_PARTITIONS:
        for pair_index in range(250):
            coordinate = f"matched/{partition}/{pair_index}"
            for stream in _MATCHED_STREAMS:
                ledger.register_seed(
                    label=f"{coordinate}/{stream}",
                    seed=_pair_stream_seed(protocol, partition, pair_index, stream),
                )
            left_id, right_id, pair_id = _pair_ids(protocol, partition, pair_index)
            ledger.register_identifier(
                label=f"{coordinate}/left_cluster_id", identifier=left_id
            )
            ledger.register_identifier(
                label=f"{coordinate}/right_cluster_id", identifier=right_id
            )
            ledger.register_identifier(
                label=f"{coordinate}/pair_id", identifier=pair_id
            )
            cluster_ids += 2
            pair_ids += 1
            for suffix in (
                "opaque_pool|0",
                "opaque_pool|1",
                "opaque_swap",
                "pair_id",
            ):
                digest = _identity_digest(
                    protocol,
                    partition=partition,
                    pair_index=pair_index,
                    suffix=suffix,
                )
                prior = identity_digests.get(digest)
                if prior is not None:
                    raise V015CollisionError(
                        f"Identity digest collision between {prior!r} and "
                        f"{coordinate}/{suffix!r}"
                    )
                identity_digests[digest] = f"{coordinate}/{suffix}"

    # Bootstrap seeds are derived only after truth reveal, but the global
    # collision policy covers every V2 stream.  Audit their complete frozen
    # coordinate set now, before the first generation RNG is consumed.
    bootstrap_root = protocol.seed_root_map()["bootstrap"]
    test_families = tuple(protocol.partition_count_map()["test"])
    for replicate_index in range(_BOOTSTRAP_RESAMPLES):
        for family_id in test_families:
            ledger.register_seed(
                label=f"analysis/bootstrap/{replicate_index}/{family_id}",
                seed=derive_bootstrap_analysis_seed(
                    protocol.protocol_id,
                    replicate_index=replicate_index,
                    family_id=family_id,
                    seed_root=bootstrap_root,
                ),
            )

    expected_clusters = sum(
        sum(families.values()) for families in protocol.partition_count_map().values()
    ) + 2 * 250 * len(MATCHED_PARTITIONS)
    if cluster_ids != expected_clusters or cluster_ids != 5950:
        raise V015CollisionError("Frozen generated-member count changed")
    if pair_ids != 500:
        raise V015CollisionError("Frozen matched-pair count changed")
    return GenerationCollisionSummary(
        seed_count=ledger.seed_count,
        cluster_id_count=cluster_ids,
        pair_id_count=pair_ids,
        identity_digest_count=len(identity_digests),
    )


def validate_predictor_hash_ledger(
    entries: Mapping[str, tuple[str, bytes, bool]],
) -> int:
    """Validate named ``(digest, bytes, unique-required)`` predictor entries."""

    ledger = CollisionLedger()
    for label, (digest, content, unique) in entries.items():
        namespace = label.split("/", maxsplit=1)[0]
        ledger.register_content_hash(
            namespace=namespace,
            digest=digest,
            canonical_content=content,
            unique_content_required=unique,
        )
    return ledger.content_hash_count


__all__ = [
    "CollisionLedger",
    "GenerationCollisionSummary",
    "V015CollisionError",
    "audit_generation_coordinate_plan",
    "derive_bootstrap_analysis_seed",
    "validate_predictor_hash_ledger",
]
