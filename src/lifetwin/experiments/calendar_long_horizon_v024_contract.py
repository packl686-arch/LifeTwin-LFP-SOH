"""Authenticated V2.9 boundary over the generic V0.20 lifecycle core."""

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
from lifetwin.experiments.calendar_long_horizon_v023_contract import (
    load_v028_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v024_protocol import (
    DEFAULT_V029_AMENDMENT_PATH,
    V029_FIXED_CORE_COMMIT,
    ValidatedV029Design,
    load_v029_design,
)


class V029ContractError(V024ContractError):
    """Raised when the V2.9 identity weakens the authenticated base."""


def _adapt_config_json(
    base: V024ContractView,
    design: ValidatedV029Design,
) -> str:
    config = base.protocol.config()
    config["protocol_id"] = design.protocol_id
    partitions = config.get("design_partitions")
    if not isinstance(partitions, dict):
        raise V029ContractError("Inherited design_partitions is invalid")
    partitions["seed_roots"] = dict(design.seed_roots)

    restored = json.loads(json.dumps(config, ensure_ascii=True, allow_nan=False))
    restored["protocol_id"] = base.protocol.protocol_id
    restored_partitions = restored.get("design_partitions")
    if not isinstance(restored_partitions, dict):
        raise V029ContractError("Adapted design_partitions is invalid")
    restored_partitions["seed_roots"] = dict(base.protocol.seed_roots)
    if restored != base.protocol.config():
        raise V029ContractError("V2.9 changed an inherited scientific field")
    return json.dumps(
        config,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def load_v029_contract_view(
    amendment_path: str | Path = DEFAULT_V029_AMENDMENT_PATH,
) -> V024ContractView:
    """Load the exact V2.9 identity without deriving or consuming a seed."""

    design = load_v029_design(amendment_path)
    base = load_v028_contract_view()
    if design.raw["base_contract"]["fixed_core_commit"] != V029_FIXED_CORE_COMMIT:
        raise V029ContractError("V2.9 fixed-core ancestry changed")
    if set(design.seed_roots.values()).intersection(
        root for _, root in base.protocol.seed_roots
    ):
        raise V029ContractError("V2.9 roots collide with V2.8")

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


__all__ = ["V029ContractError", "load_v029_contract_view"]
