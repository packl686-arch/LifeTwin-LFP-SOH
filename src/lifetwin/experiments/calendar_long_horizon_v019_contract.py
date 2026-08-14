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
import re
from types import MappingProxyType
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
    V024_AMENDMENT_SEMANTIC_SHA256,
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
_PROTOCOL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_WHOLE_ROW_INDEX = {
    "prefix_pack.csv": 1,
    "forecast_coordinates.csv": 2,
    "operating_pack.csv": 3,
    "member_fit_diagnostics.csv": 4,
    "member_forecast_bundle.csv": 5,
}


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
    config_canonical_sha256: str
    whole_rows: Mapping[str, int]
    partition_rows: Mapping[str, tuple[int, ...]]
    base_config_canonical_sha256: str
    base_config_byte_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.whole_rows) is not _MAPPING_PROXY_TYPE
            or type(self.partition_rows) is not _MAPPING_PROXY_TYPE
        ):
            raise V024ContractError("Row registries must be immutable mappings")
        object.__setattr__(self, "whole_rows", MappingProxyType(dict(self.whole_rows)))
        object.__setattr__(
            self,
            "partition_rows",
            MappingProxyType(
                {name: tuple(counts) for name, counts in self.partition_rows.items()}
            ),
        )
        if type(self.artifacts) is FrozenArtifactContract:
            object.__setattr__(
                self,
                "artifacts",
                replace(
                    self.artifacts,
                    csv_schemas=MappingProxyType(dict(self.artifacts.csv_schemas)),
                    json_key_allowlists=MappingProxyType(
                        dict(self.artifacts.json_key_allowlists)
                    ),
                    partition_member_counts=MappingProxyType(
                        dict(self.artifacts.partition_member_counts)
                    ),
                ),
            )
        require_contract_view(self)


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
        if field.name in allowed:
            continue
        base_value = getattr(base, field.name)
        adapted_value = getattr(adapted, field.name)
        if type(base_value) is not type(adapted_value) or base_value != adapted_value:
            raise V024ContractError(
                f"{context}.{field.name} changed outside the amendment"
            )


def _load_bound_amendment(view: V024ContractView) -> Mapping[str, Any]:
    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise V024ContractError("Adapted amendment has duplicate JSON keys")
            result[key] = item
        return result

    def reject_nonfinite(token: str) -> None:
        raise V024ContractError(f"Adapted amendment contains {token}")

    try:
        raw = view.artifacts.config_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V024ContractError("Cannot parse adapted amendment") from exc
    if not isinstance(payload, Mapping):
        raise V024ContractError("Adapted amendment must be a JSON object")
    if hashlib.sha256(raw).hexdigest() != view.artifacts.config_byte_sha256:
        raise V024ContractError("Adapted config commitment is not the amendment")
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != view.config_canonical_sha256:
        raise V024ContractError("Adapted canonical config commitment drifted")

    if payload.get("protocol_id") != V024_PROTOCOL_ID:
        from lifetwin.experiments.calendar_long_horizon_v020_protocol import (  # noqa: PLC0415
            V025_PROTOCOL_ID,
            load_v025_design,
        )

        if payload.get("protocol_id") == V025_PROTOCOL_ID:
            design = load_v025_design(view.artifacts.config_path)
            if (
                design.config_byte_sha256 != view.artifacts.config_byte_sha256
                or design.config_semantic_sha256 != view.config_canonical_sha256
            ):
                raise V024ContractError("V2.5 design and contract commitments disagree")
            return payload

        from lifetwin.experiments.calendar_long_horizon_v021_protocol import (  # noqa: PLC0415
            V026_PROTOCOL_ID,
            load_v026_design,
        )

        if payload.get("protocol_id") == V026_PROTOCOL_ID:
            design = load_v026_design(view.artifacts.config_path)
            if (
                design.config_byte_sha256 != view.artifacts.config_byte_sha256
                or design.config_semantic_sha256 != view.config_canonical_sha256
            ):
                raise V024ContractError("V2.6 design and contract commitments disagree")
            return payload

        from lifetwin.experiments.calendar_long_horizon_v022_protocol import (  # noqa: PLC0415
            V027_PROTOCOL_ID,
            load_v027_design,
        )

        if payload.get("protocol_id") == V027_PROTOCOL_ID:
            design = load_v027_design(view.artifacts.config_path)
            if (
                design.config_byte_sha256 != view.artifacts.config_byte_sha256
                or design.config_semantic_sha256 != view.config_canonical_sha256
            ):
                raise V024ContractError("V2.7 design and contract commitments disagree")
            return payload

        from lifetwin.experiments.calendar_long_horizon_v023_protocol import (  # noqa: PLC0415
            V028_PROTOCOL_ID,
            load_v028_design,
        )

        if payload.get("protocol_id") == V028_PROTOCOL_ID:
            design = load_v028_design(view.artifacts.config_path)
            if (
                design.config_byte_sha256 != view.artifacts.config_byte_sha256
                or design.config_semantic_sha256 != view.config_canonical_sha256
            ):
                raise V024ContractError("V2.8 design and contract commitments disagree")
            return payload

        from lifetwin.experiments.calendar_long_horizon_v024_protocol import (  # noqa: PLC0415
            V029_PROTOCOL_ID,
            load_v029_design,
        )

        if payload.get("protocol_id") == V029_PROTOCOL_ID:
            design = load_v029_design(view.artifacts.config_path)
            if (
                design.config_byte_sha256 != view.artifacts.config_byte_sha256
                or design.config_semantic_sha256 != view.config_canonical_sha256
            ):
                raise V024ContractError("V2.9 design and contract commitments disagree")
            return payload

        from lifetwin.experiments.calendar_long_horizon_v025_protocol import (  # noqa: PLC0415
            V030_PROTOCOL_ID,
            load_v030_design,
        )

        if payload.get("protocol_id") == V030_PROTOCOL_ID:
            design = load_v030_design(view.artifacts.config_path)
            if (
                design.config_byte_sha256 != view.artifacts.config_byte_sha256
                or design.config_semantic_sha256 != view.config_canonical_sha256
            ):
                raise V024ContractError(
                    "V2.10 design and contract commitments disagree"
                )
            return payload

    load_v024_design(DEFAULT_V024_AMENDMENT_PATH)
    frozen = json.loads(DEFAULT_V024_AMENDMENT_PATH.read_text(encoding="utf-8"))
    restored = json.loads(json.dumps(payload, ensure_ascii=True, allow_nan=False))
    restored["protocol_id"] = frozen["protocol_id"]
    restored["status"] = frozen["status"]
    restored_generation = restored.get("fresh_generation")
    if not isinstance(restored_generation, dict):
        raise V024ContractError("Adapted amendment generation contract is absent")
    restored_generation["seed_roots"] = frozen["fresh_generation"]["seed_roots"]
    if restored != frozen:
        raise V024ContractError("Adapted amendment changed a scientific field")
    return payload


def require_contract_view(value: object) -> V024ContractView:
    """Validate one immutable identity/config/seed source of truth."""

    if type(value) is not V024ContractView:
        raise V024ContractError("contract view must have exact V024ContractView type")
    view = value
    _require_exact_type(
        view.protocol, ValidatedV015Protocol, context="adapted protocol"
    )
    _require_exact_type(
        view.artifacts, FrozenArtifactContract, context="adapted artifacts"
    )
    base_protocol = load_frozen_protocol_config(DEFAULT_V2_CONFIG_PATH)
    base_artifacts = load_artifact_contract(DEFAULT_V2_CONFIG_PATH)
    _validate_base_contracts(base_protocol, base_artifacts)
    _assert_only_allowed_dataclass_changes(
        base_protocol,
        view.protocol,
        allowed=_PROTOCOL_ADAPTED_FIELDS,
        context="protocol",
    )
    _assert_only_allowed_dataclass_changes(
        base_artifacts,
        view.artifacts,
        allowed=_ARTIFACT_ADAPTED_FIELDS,
        context="artifacts",
    )
    protocol_id = view.protocol.protocol_id
    if (
        not isinstance(protocol_id, str)
        or _PROTOCOL_ID.fullmatch(protocol_id) is None
        or view.artifacts.protocol_id != protocol_id
    ):
        raise V024ContractError("Adapted protocol identities disagree")
    if view.protocol.config_sha256 != view.artifacts.config_byte_sha256:
        raise V024ContractError("Adapted config commitments disagree")
    config_sha256 = view.protocol.config_sha256
    if not isinstance(config_sha256, str) or _SHA256.fullmatch(config_sha256) is None:
        raise V024ContractError("Adapted config commitment is invalid")
    if view.artifacts.config_path != view.artifacts.config_path.resolve():
        raise V024ContractError("Adapted amendment path must be absolute")
    if (
        not isinstance(view.config_canonical_sha256, str)
        or _SHA256.fullmatch(view.config_canonical_sha256) is None
    ):
        raise V024ContractError("Adapted canonical config commitment is invalid")
    roots = view.protocol.seed_roots
    if (
        len(roots) != 13
        or len({name for name, _ in roots}) != 13
        or len({root for _, root in roots}) != 13
        or any(
            not isinstance(name, str) or type(root) is not int for name, root in roots
        )
    ):
        raise V024ContractError("Adapted seed-root registry is not exact")
    if (
        type(view.whole_rows) is not _MAPPING_PROXY_TYPE
        or set(view.whole_rows) != set(_WHOLE_ROW_INDEX)
        or any(
            type(count) is not int or count < 1 for count in view.whole_rows.values()
        )
    ):
        raise V024ContractError("Adapted whole-row registry is invalid")
    if (
        type(view.partition_rows) is not _MAPPING_PROXY_TYPE
        or tuple(view.partition_rows) != view.artifacts.partitions
        or any(
            not isinstance(counts, tuple)
            or len(counts) != 6
            or any(type(count) is not int or count < 1 for count in counts)
            for counts in view.partition_rows.values()
        )
    ):
        raise V024ContractError("Adapted partition-row registry is invalid")
    if view.protocol.prefix_days != view.artifacts.prefix_days:
        raise V024ContractError("Adapted prefix grids disagree")
    if view.protocol.forecast_days != view.artifacts.forecast_days:
        raise V024ContractError("Adapted forecast grids disagree")
    if not isinstance(view.design_status, str) or not view.design_status:
        raise V024ContractError("Adapted design status is invalid")
    if view.base_config_canonical_sha256 != FROZEN_CONFIG_CANONICAL_SHA256:
        raise V024ContractError("Base canonical provenance drifted")
    if view.base_config_byte_sha256 != FROZEN_CONFIG_BYTE_SHA256:
        raise V024ContractError("Base byte provenance drifted")

    adapted_config = view.protocol.config()
    if adapted_config.get("protocol_id") != protocol_id:
        raise V024ContractError("Adapted protocol JSON identity drifted")
    design_partitions = adapted_config.get("design_partitions")
    if not isinstance(design_partitions, Mapping):
        raise V024ContractError("Adapted design_partitions is not an object")
    config_roots = design_partitions.get("seed_roots")
    if not isinstance(config_roots, Mapping):
        raise V024ContractError("Adapted seed_roots is not an object")
    if tuple(config_roots.items()) != roots:
        raise V024ContractError("Adapted JSON and parsed roots disagree")
    expected_config = base_protocol.config()
    expected_config["protocol_id"] = protocol_id
    expected_partitions = expected_config.get("design_partitions")
    if not isinstance(expected_partitions, dict):
        raise V024ContractError("Base design_partitions is not mutable JSON")
    expected_partitions["seed_roots"] = dict(roots)
    if adapted_config != expected_config:
        raise V024ContractError("Adapted protocol JSON changed a scientific field")

    amendment = _load_bound_amendment(view)
    if (
        amendment.get("protocol_id") != protocol_id
        or amendment.get("status") != view.design_status
    ):
        raise V024ContractError("Adapted amendment identity or status drifted")
    fresh_generation = amendment.get("fresh_generation")
    if not isinstance(fresh_generation, Mapping):
        raise V024ContractError("Adapted amendment generation contract is absent")
    amendment_roots = fresh_generation.get("seed_roots")
    if (
        not isinstance(amendment_roots, Mapping)
        or tuple(amendment_roots.items()) != roots
    ):
        raise V024ContractError("Adapted amendment seed roots drifted")
    row_contract_source = amendment
    if "whole_bundle_contract" not in row_contract_source:
        load_v024_design(DEFAULT_V024_AMENDMENT_PATH)
        row_contract_source = json.loads(
            DEFAULT_V024_AMENDMENT_PATH.read_text(encoding="utf-8")
        )
    whole_contract = row_contract_source.get("whole_bundle_contract")
    partition_contract = row_contract_source.get("partition_contract")
    if not isinstance(whole_contract, Mapping) or not isinstance(
        partition_contract, Mapping
    ):
        raise V024ContractError("Adapted amendment row contracts are absent")
    if whole_contract.get("required_tables") != dict(view.whole_rows):
        raise V024ContractError("Adapted whole-row registry is not amendment-bound")
    raw_partition_rows = partition_contract.get("cardinality")
    if not isinstance(raw_partition_rows, Mapping) or {
        str(name): tuple(counts) if isinstance(counts, list) else counts
        for name, counts in raw_partition_rows.items()
    } != dict(view.partition_rows):
        raise V024ContractError("Adapted partition-row registry is not amendment-bound")
    for filename, index in _WHOLE_ROW_INDEX.items():
        if (
            sum(counts[index] for counts in view.partition_rows.values())
            != view.whole_rows[filename]
        ):
            raise V024ContractError("Adapted whole and partition row counts disagree")
    if any(
        counts[0] != view.artifacts.partition_member_counts[partition]
        for partition, counts in view.partition_rows.items()
    ):
        raise V024ContractError("Adapted partition member counts disagree")
    return view


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
    whole_contract = design.raw.get("whole_bundle_contract")
    partition_contract = design.raw.get("partition_contract")
    if not isinstance(whole_contract, Mapping) or not isinstance(
        partition_contract, Mapping
    ):
        raise V024ContractError("Design row contracts are absent")
    whole_rows = whole_contract.get("required_tables")
    partition_rows = partition_contract.get("cardinality")
    if not isinstance(whole_rows, Mapping) or not isinstance(partition_rows, Mapping):
        raise V024ContractError("Design row registries are absent")
    return V024ContractView(
        protocol=protocol,
        artifacts=artifacts,
        design_status=design.status,
        config_canonical_sha256=V024_AMENDMENT_SEMANTIC_SHA256,
        whole_rows=MappingProxyType(
            {str(name): int(count) for name, count in whole_rows.items()}
        ),
        partition_rows=MappingProxyType(
            {
                str(name): tuple(int(count) for count in counts)
                for name, counts in partition_rows.items()
            }
        ),
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


def resolve_contract_view(
    value: FrozenArtifactContract | V024ContractView | None,
) -> V024ContractView:
    """Resolve a generic view or the exact legacy V2.4 artifact boundary."""

    if value is None:
        return load_v024_contract_view()
    if type(value) is V024ContractView:
        return require_contract_view(value)
    if type(value) is FrozenArtifactContract:
        legacy = load_v024_contract_view()
        if value == legacy.artifacts:
            return legacy
    raise V024ContractError("Artifact contract has no authenticated contract view")


__all__ = [
    "V024ContractError",
    "V024ContractView",
    "adapt_v024_contract_view",
    "load_v024_contract_view",
    "require_contract_view",
    "resolve_contract_view",
]
