"""Non-generative V2.4 adapter for the frozen V2 protocol contracts.

The V2.4 amendment changes identity, generation roots, and calibration handling.
This module adapts only the identity/configuration surface needed by later V2.4
code.  It deliberately provides no seed derivation, RNG, generation, fitting,
prediction, truth access, or scoring entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    DEFAULT_V2_CONFIG_PATH,
    FROZEN_PROTOCOL_ID as V2_ARTIFACT_PROTOCOL_ID,
    FROZEN_SCHEMA_VERSION,
    FrozenArtifactContract,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256,
    FROZEN_CONFIG_CANONICAL_SHA256,
    FROZEN_PROTOCOL_ID as V2_PROTOCOL_ID,
    ValidatedV015Protocol,
    load_frozen_protocol_config,
)
from lifetwin.experiments.calendar_long_horizon_v019_protocol import (
    DEFAULT_V024_AMENDMENT_PATH,
    V024_ALLOWED_DESIGN_STATUSES,
    V024_EXPECTED_SEED_ROOTS,
    V024_PROTOCOL_ID,
    ValidatedV024Design,
    load_v024_design,
)


_PROTOCOL_ADAPTED_FIELDS = frozenset(
    {"protocol_id", "config_sha256", "seed_roots", "config_json"}
)
_ARTIFACT_ADAPTED_FIELDS = frozenset(
    {"protocol_id", "config_path", "config_byte_sha256"}
)


class V024ContractError(ValueError):
    """Raised when a V2.4 contract view would weaken frozen inheritance."""


@dataclass(frozen=True, slots=True)
class V024ContractView:
    """Immutable, non-generative V2.4 view over the two frozen V2 contracts.

    ``protocol`` and ``artifacts`` retain their legacy dataclass types so that
    narrowly scoped, outcome-free V2 helpers can consume them.  The V2 formal
    runner remains fail-closed because its frozen environment and config checks
    reject the V2.4 identity and amendment hash.
    """

    protocol: ValidatedV015Protocol
    artifacts: FrozenArtifactContract
    design_status: str
    base_config_canonical_sha256: str
    base_config_byte_sha256: str

    def __post_init__(self) -> None:
        _validate_adapted_view(self)


def _file_sha256(path: Path, *, context: str) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise V024ContractError(f"Cannot read {context}") from exc
    return hashlib.sha256(raw).hexdigest()


def _require_exact_type(value: object, expected: type[object], *, context: str) -> None:
    if type(value) is not expected:
        raise V024ContractError(f"{context} must have exact type {expected.__name__}")


def _base_seed_roots(protocol: ValidatedV015Protocol) -> tuple[tuple[str, int], ...]:
    roots = protocol.seed_roots
    if (
        len(roots) != 13
        or len({name for name, _ in roots}) != 13
        or len({root for _, root in roots}) != 13
    ):
        raise V024ContractError("Frozen V2 seed-root registry is not exact")
    return roots


def _validate_base_contracts(
    protocol: ValidatedV015Protocol,
    artifacts: FrozenArtifactContract,
) -> None:
    _require_exact_type(protocol, ValidatedV015Protocol, context="base_protocol")
    _require_exact_type(artifacts, FrozenArtifactContract, context="base_artifacts")
    if V2_PROTOCOL_ID != V2_ARTIFACT_PROTOCOL_ID:
        raise V024ContractError("Imported V2 protocol identities disagree")
    if protocol.protocol_id != V2_PROTOCOL_ID:
        raise V024ContractError("Base protocol is not the frozen V2 identity")
    if artifacts.protocol_id != V2_PROTOCOL_ID:
        raise V024ContractError("Base artifact contract is not the frozen V2 identity")
    if protocol.config_sha256 != FROZEN_CONFIG_CANONICAL_SHA256:
        raise V024ContractError("Base protocol canonical commitment drifted")
    if artifacts.config_byte_sha256 != FROZEN_CONFIG_BYTE_SHA256:
        raise V024ContractError("Base artifact byte commitment drifted")
    if artifacts.schema_version != FROZEN_SCHEMA_VERSION:
        raise V024ContractError("Base artifact schema version drifted")
    if (
        _file_sha256(artifacts.config_path, context="frozen V2 config")
        != FROZEN_CONFIG_BYTE_SHA256
    ):
        raise V024ContractError("Base artifact config bytes drifted")
    if protocol.prefix_days != artifacts.prefix_days:
        raise V024ContractError("Base prefix grids disagree")
    if protocol.forecast_days != artifacts.forecast_days:
        raise V024ContractError("Base forecast grids disagree")

    partition_counts = protocol.partition_count_map()
    for partition, family_counts in partition_counts.items():
        expected = sum(family_counts.values())
        if artifacts.partition_member_counts.get(partition) != expected:
            raise V024ContractError(
                f"Base member count disagrees for partition {partition}"
            )

    config = protocol.config()
    if config.get("protocol_id") != V2_PROTOCOL_ID:
        raise V024ContractError("Base protocol JSON identity drifted")
    design_partitions = config.get("design_partitions")
    if not isinstance(design_partitions, Mapping):
        raise V024ContractError("Base design_partitions is not an object")
    raw_roots = design_partitions.get("seed_roots")
    if not isinstance(raw_roots, Mapping):
        raise V024ContractError("Base seed_roots is not an object")
    if tuple(raw_roots.items()) != _base_seed_roots(protocol):
        raise V024ContractError("Base protocol JSON and parsed seed roots disagree")


def _validate_design_identity(design: ValidatedV024Design) -> None:
    _require_exact_type(design, ValidatedV024Design, context="design")
    if design.protocol_id != V024_PROTOCOL_ID:
        raise V024ContractError("V2.4 design identity drifted")
    if design.status not in V024_ALLOWED_DESIGN_STATUSES:
        raise V024ContractError("V2.4 design status is not adapter-safe")
    if design.config_path != design.config_path.resolve():
        raise V024ContractError("V2.4 amendment path must be absolute")
    if (
        _file_sha256(design.config_path, context="V2.4 amendment")
        != design.config_byte_sha256
    ):
        raise V024ContractError("V2.4 amendment byte commitment drifted")
    expected_roots = tuple(V024_EXPECTED_SEED_ROOTS.items())
    observed_roots = tuple(design.seed_roots.items())
    if observed_roots != expected_roots:
        raise V024ContractError("V2.4 fresh seed roots drifted")
    if len(observed_roots) != 13 or len({root for _, root in observed_roots}) != 13:
        raise V024ContractError("V2.4 must have thirteen distinct fresh roots")


def _adapt_config_json(
    base_protocol: ValidatedV015Protocol,
    design: ValidatedV024Design,
) -> str:
    """Patch only identity and declared root values in a fresh parsed V2 object."""

    adapted: dict[str, Any] = base_protocol.config()
    adapted["protocol_id"] = design.protocol_id
    design_partitions = adapted.get("design_partitions")
    if not isinstance(design_partitions, dict):
        raise V024ContractError("Parsed base design_partitions is not mutable JSON")
    design_partitions["seed_roots"] = dict(design.seed_roots)

    base_for_comparison = base_protocol.config()
    expected = base_protocol.config()
    expected["protocol_id"] = design.protocol_id
    expected_partitions = expected.get("design_partitions")
    if not isinstance(expected_partitions, dict):
        raise V024ContractError("Parsed base design_partitions is not mutable JSON")
    expected_partitions["seed_roots"] = dict(design.seed_roots)
    if adapted != expected:
        raise V024ContractError("V2.4 config adaptation changed an undeclared field")

    restored = json.loads(json.dumps(adapted, ensure_ascii=True, allow_nan=False))
    restored["protocol_id"] = V2_PROTOCOL_ID
    restored_partitions = restored.get("design_partitions")
    if not isinstance(restored_partitions, dict):
        raise V024ContractError("Adapted design_partitions is not an object")
    restored_partitions["seed_roots"] = dict(_base_seed_roots(base_protocol))
    if restored != base_for_comparison:
        raise V024ContractError("V2.4 config is not an exact two-field V2 adaptation")

    return json.dumps(
        adapted,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _assert_only_allowed_dataclass_changes(
    base: object,
    adapted: object,
    *,
    allowed: frozenset[str],
    context: str,
) -> None:
    if type(base) is not type(adapted):
        raise V024ContractError(f"{context} type changed")
    for field in fields(base):
        if field.name not in allowed and getattr(base, field.name) != getattr(
            adapted, field.name
        ):
            raise V024ContractError(
                f"{context}.{field.name} changed outside the amendment"
            )


def _validate_adapted_view(view: V024ContractView) -> None:
    _require_exact_type(
        view.protocol, ValidatedV015Protocol, context="adapted protocol"
    )
    _require_exact_type(
        view.artifacts, FrozenArtifactContract, context="adapted artifacts"
    )
    if view.protocol.protocol_id != V024_PROTOCOL_ID:
        raise V024ContractError("Adapted protocol identity is not V2.4")
    if view.artifacts.protocol_id != V024_PROTOCOL_ID:
        raise V024ContractError("Adapted artifact identity is not V2.4")
    if view.protocol.config_sha256 != view.artifacts.config_byte_sha256:
        raise V024ContractError("Adapted config commitments disagree")
    if view.protocol.config_sha256 != _file_sha256(
        view.artifacts.config_path, context="adapted amendment"
    ):
        raise V024ContractError("Adapted config commitment is not the amendment")
    if view.protocol.seed_roots != tuple(V024_EXPECTED_SEED_ROOTS.items()):
        raise V024ContractError("Adapted protocol roots are not the fresh V2.4 roots")
    if view.protocol.prefix_days != view.artifacts.prefix_days:
        raise V024ContractError("Adapted prefix grids disagree")
    if view.protocol.forecast_days != view.artifacts.forecast_days:
        raise V024ContractError("Adapted forecast grids disagree")
    if view.design_status not in V024_ALLOWED_DESIGN_STATUSES:
        raise V024ContractError("Adapted design status drifted")
    if view.base_config_canonical_sha256 != FROZEN_CONFIG_CANONICAL_SHA256:
        raise V024ContractError("Base canonical provenance drifted")
    if view.base_config_byte_sha256 != FROZEN_CONFIG_BYTE_SHA256:
        raise V024ContractError("Base byte provenance drifted")

    adapted_config = view.protocol.config()
    if adapted_config.get("protocol_id") != V024_PROTOCOL_ID:
        raise V024ContractError("Adapted protocol JSON identity drifted")
    design_partitions = adapted_config.get("design_partitions")
    if not isinstance(design_partitions, Mapping):
        raise V024ContractError("Adapted design_partitions is not an object")
    roots = design_partitions.get("seed_roots")
    if not isinstance(roots, Mapping):
        raise V024ContractError("Adapted seed_roots is not an object")
    if tuple(roots.items()) != view.protocol.seed_roots:
        raise V024ContractError("Adapted JSON and parsed roots disagree")


def adapt_v024_contract_view(
    *,
    design: ValidatedV024Design,
    base_protocol: ValidatedV015Protocol,
    base_artifacts: FrozenArtifactContract,
) -> V024ContractView:
    """Construct an immutable V2.4 view without deriving or consuming any seed."""

    _validate_design_identity(design)
    _validate_base_contracts(base_protocol, base_artifacts)

    base_names = tuple(name for name, _ in _base_seed_roots(base_protocol))
    fresh_roots = tuple(design.seed_roots.items())
    if tuple(name for name, _ in fresh_roots) != base_names:
        raise V024ContractError("V2.4 seed-root registry names or order changed")
    if {root for _, root in fresh_roots}.intersection(
        root for _, root in base_protocol.seed_roots
    ):
        raise V024ContractError("V2.4 roots collide with frozen V2 roots")

    protocol = replace(
        base_protocol,
        protocol_id=design.protocol_id,
        config_sha256=design.config_byte_sha256,
        seed_roots=fresh_roots,
        config_json=_adapt_config_json(base_protocol, design),
    )
    artifacts = replace(
        base_artifacts,
        protocol_id=design.protocol_id,
        config_path=design.config_path,
        config_byte_sha256=design.config_byte_sha256,
    )
    _assert_only_allowed_dataclass_changes(
        base_protocol,
        protocol,
        allowed=_PROTOCOL_ADAPTED_FIELDS,
        context="protocol",
    )
    _assert_only_allowed_dataclass_changes(
        base_artifacts,
        artifacts,
        allowed=_ARTIFACT_ADAPTED_FIELDS,
        context="artifacts",
    )
    return V024ContractView(
        protocol=protocol,
        artifacts=artifacts,
        design_status=design.status,
        base_config_canonical_sha256=base_protocol.config_sha256,
        base_config_byte_sha256=base_artifacts.config_byte_sha256,
    )


def load_v024_contract_view(
    amendment_path: str | Path = DEFAULT_V024_AMENDMENT_PATH,
    base_config_path: str | Path = DEFAULT_V2_CONFIG_PATH,
) -> V024ContractView:
    """Validate both committed configs and return the non-generative V2.4 view."""

    design = load_v024_design(amendment_path)
    base_protocol = load_frozen_protocol_config(base_config_path)
    base_artifacts = load_artifact_contract(base_config_path)
    return adapt_v024_contract_view(
        design=design,
        base_protocol=base_protocol,
        base_artifacts=base_artifacts,
    )


__all__ = [
    "V024ContractError",
    "V024ContractView",
    "adapt_v024_contract_view",
    "load_v024_contract_view",
]
