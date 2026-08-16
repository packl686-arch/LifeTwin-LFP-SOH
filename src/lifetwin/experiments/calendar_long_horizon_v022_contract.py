"""Authenticated V2.7 boundary over the generic V0.20 lifecycle core."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractError,
    V024ContractView,
    require_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v021_contract import (
    load_v026_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v022_protocol import (
    DEFAULT_V027_AMENDMENT_PATH,
    V027_FIXED_CORE_COMMIT,
    ValidatedV027Design,
    load_v027_design,
)


class V027ContractError(V024ContractError):
    """Raised when the V2.7 identity weakens the authenticated base."""


def _adapt_config_json(
    base: V024ContractView,
    design: ValidatedV027Design,
) -> str:
    config = base.protocol.config()
    config["protocol_id"] = design.protocol_id
    partitions = config.get("design_partitions")
    if not isinstance(partitions, dict):
        raise V027ContractError("Inherited design_partitions is invalid")
    partitions["seed_roots"] = dict(design.seed_roots)

    restored = json.loads(json.dumps(config, ensure_ascii=True, allow_nan=False))
    restored["protocol_id"] = base.protocol.protocol_id
    restored_partitions = restored.get("design_partitions")
    if not isinstance(restored_partitions, dict):
        raise V027ContractError("Adapted design_partitions is invalid")
    restored_partitions["seed_roots"] = dict(base.protocol.seed_roots)
    if restored != base.protocol.config():
        raise V027ContractError("V2.7 changed an inherited scientific field")
    return json.dumps(
        config,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def load_v027_contract_view(
    amendment_path: str | Path = DEFAULT_V027_AMENDMENT_PATH,
) -> V024ContractView:
    """Load the exact V2.7 identity without deriving or consuming a seed."""

    design = load_v027_design(amendment_path)
    base = load_v026_contract_view()
    if design.raw["base_contract"]["fixed_core_commit"] != V027_FIXED_CORE_COMMIT:
        raise V027ContractError("V2.7 fixed-core ancestry changed")
    if set(design.seed_roots.values()).intersection(
        root for _, root in base.protocol.seed_roots
    ):
        raise V027ContractError("V2.7 roots collide with V2.6")

    view = V024ContractView(
        protocol=replace(
            base.protocol,
            protocol_id=design.protocol_id,
            config_sha256=design.config_byte_sha256,
            seed_roots=tuple(design.seed_roots.items()),
            config_json=_adapt_config_json(base, design),
        ),
        artifacts=replace(
            base.artifacts,
            protocol_id=design.protocol_id,
            config_path=design.config_path,
            config_byte_sha256=design.config_byte_sha256,
        ),
        design_status=design.status,
        config_canonical_sha256=design.config_semantic_sha256,
        whole_rows=MappingProxyType(dict(base.whole_rows)),
        partition_rows=MappingProxyType(dict(base.partition_rows)),
        base_config_canonical_sha256=base.base_config_canonical_sha256,
        base_config_byte_sha256=base.base_config_byte_sha256,
    )
    return require_contract_view(view)


__all__ = ["V027ContractError", "load_v027_contract_view"]
