"""Capability-bound whole-bundle and partition validation for V2.3.

The formal path has no ``formal`` switch. Whole files are validated with the
authoritative frozen contract first. Partition schemas are derived from that
contract only after an exact whole-bundle capability exists, and are again
validated with ``formal=True`` inside this module.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    canonical_csv_bytes,
    canonicalize_frame,
    read_canonical_csv,
)
from lifetwin.experiments.calendar_long_horizon_v018_protocol import (
    V023_PROTOCOL_ID,
    load_v023_design,
)


INPUT_FILENAMES = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
)
_COUNT_ORDER = (
    "clusters",
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
)
_ISSUER = object()


class V023WholeBundleContractError(ValueError):
    """Raised before slicing when a complete formal bundle is invalid."""


class V023PartitionContractError(ValueError):
    """Raised when a capability-derived partition violates its exact contract."""


class V023PartitionCapabilityError(ValueError):
    """Raised when a partition capability is forged, mutated or misused."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _partition_counts() -> Mapping[str, Mapping[str, int]]:
    design = load_v023_design()
    raw = design.raw["partition_contract"]["cardinality"]
    result: dict[str, Mapping[str, int]] = {}
    for partition, values in raw.items():
        if not isinstance(partition, str) or len(values) != len(_COUNT_ORDER):
            raise V023PartitionContractError("Partition cardinality registry changed")
        parsed = {
            name: int(value) for name, value in zip(_COUNT_ORDER, values, strict=True)
        }
        if any(value <= 0 for value in parsed.values()):
            raise V023PartitionContractError("Partition cardinality is not positive")
        result[partition] = MappingProxyType(parsed)
    return MappingProxyType(result)


PARTITION_COUNTS = _partition_counts()


def _whole_counts() -> Mapping[str, int]:
    design = load_v023_design()
    raw = design.raw["whole_bundle_contract"]["required_tables"]
    if set(raw) != set(INPUT_FILENAMES):
        raise V023WholeBundleContractError("Whole-bundle table registry changed")
    parsed = {name: int(raw[name]) for name in INPUT_FILENAMES}
    if any(value <= 0 for value in parsed.values()):
        raise V023WholeBundleContractError("Whole-bundle cardinality is not positive")
    return MappingProxyType(parsed)


WHOLE_COUNTS = _whole_counts()


class WholeBundleValidated:
    """Opaque exact-type capability issued only after all five full files pass."""

    __slots__ = ("_contract_hash", "_frames", "_source_hashes", "_source_sizes")

    def __init__(
        self,
        *,
        issuer: object,
        contract_hash: str,
        frames: Mapping[str, pd.DataFrame],
        source_hashes: Mapping[str, str],
        source_sizes: Mapping[str, int],
    ) -> None:
        if issuer is not _ISSUER:
            raise V023PartitionCapabilityError(
                "WholeBundleValidated cannot be constructed by a caller"
            )
        self._contract_hash = contract_hash
        self._frames = MappingProxyType(
            {name: frame.copy(deep=False) for name, frame in frames.items()}
        )
        self._source_hashes = MappingProxyType(dict(source_hashes))
        self._source_sizes = MappingProxyType(dict(source_sizes))

    @property
    def source_hashes(self) -> Mapping[str, str]:
        return self._source_hashes

    @property
    def source_sizes(self) -> Mapping[str, int]:
        return self._source_sizes


class ValidatedPartitionView:
    """Opaque immutable view over one exact V2.3 partition."""

    __slots__ = (
        "_contract_hash",
        "_frame_hashes",
        "_frames",
        "_partition",
        "_source_hashes",
    )

    def __init__(
        self,
        *,
        issuer: object,
        partition: str,
        contract_hash: str,
        frames: Mapping[str, pd.DataFrame],
        frame_hashes: Mapping[str, str],
        source_hashes: Mapping[str, str],
    ) -> None:
        if issuer is not _ISSUER:
            raise V023PartitionCapabilityError(
                "ValidatedPartitionView cannot be constructed by a caller"
            )
        self._partition = partition
        self._contract_hash = contract_hash
        self._frames = MappingProxyType(
            {name: frame.copy(deep=False) for name, frame in frames.items()}
        )
        self._frame_hashes = MappingProxyType(dict(frame_hashes))
        self._source_hashes = MappingProxyType(dict(source_hashes))

    @property
    def partition(self) -> str:
        return self._partition

    @property
    def source_hashes(self) -> Mapping[str, str]:
        return self._source_hashes


def _require_contract(contract: FrozenArtifactContract) -> None:
    if type(contract) is not FrozenArtifactContract:
        raise V023PartitionCapabilityError("Artifact contract has the wrong exact type")
    if contract.protocol_id != V023_PROTOCOL_ID:
        raise V023PartitionCapabilityError("Artifact contract is not V2.3")


def _direct_file(root: Path, filename: str) -> Path:
    path = root / filename
    if path.parent != root or not path.is_file() or path.is_symlink():
        raise V023WholeBundleContractError(f"Invalid direct artifact: {filename}")
    return path


def _require_finite_numeric(frame: pd.DataFrame, *, filename: str) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise V023PartitionContractError(
            f"{filename} contains a nonfinite numeric value"
        )


def validate_whole_bundle_from_root(
    root: str | Path,
    contract: FrozenArtifactContract,
) -> WholeBundleValidated:
    """Validate and bind the five authoritative complete label-free files."""

    _require_contract(contract)
    source_root = Path(root).resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise V023WholeBundleContractError("Label-free root is not a physical directory")
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for filename in INPUT_FILENAMES:
        path = _direct_file(source_root, filename)
        try:
            frames[filename] = read_canonical_csv(path, contract, formal=True)
        except V015ArtifactError as exc:
            raise V023WholeBundleContractError(str(exc)) from exc
        try:
            _require_finite_numeric(frames[filename], filename=filename)
        except V023PartitionContractError as exc:
            raise V023WholeBundleContractError(str(exc)) from exc
        if len(frames[filename]) != WHOLE_COUNTS[filename]:
            raise V023WholeBundleContractError(
                f"{filename} must contain exactly {WHOLE_COUNTS[filename]} rows"
            )
        raw = path.read_bytes()
        hashes[filename] = _sha256(raw)
        sizes[filename] = len(raw)
    if tuple(frames) != INPUT_FILENAMES:
        raise V023WholeBundleContractError("Whole-bundle file order changed")
    return WholeBundleValidated(
        issuer=_ISSUER,
        contract_hash=contract.config_byte_sha256,
        frames=frames,
        source_hashes=hashes,
        source_sizes=sizes,
    )


def _partition_schema(
    contract: FrozenArtifactContract,
    *,
    filename: str,
    partition: str,
    required_rows: int,
):
    schema = contract.csv_schema(filename)
    return replace(
        schema,
        required_rows=required_rows,
        expected_partition=partition,
    )


def _canonical_partition(
    frame: pd.DataFrame,
    *,
    filename: str,
    partition: str,
    required_rows: int,
    contract: FrozenArtifactContract,
    require_all_numeric_finite: bool = True,
) -> tuple[pd.DataFrame, str]:
    schema = _partition_schema(
        contract,
        filename=filename,
        partition=partition,
        required_rows=required_rows,
    )
    try:
        canonical = canonicalize_frame(frame, schema, contract, formal=True)
        raw = canonical_csv_bytes(canonical, schema, contract, formal=True)
    except V015ArtifactError as exc:
        raise V023PartitionContractError(str(exc)) from exc
    if require_all_numeric_finite:
        _require_finite_numeric(canonical, filename=filename)
    observed_keys = frame.loc[:, list(schema.key)].reset_index(drop=True)
    canonical_keys = canonical.loc[:, list(schema.key)].reset_index(drop=True)
    if not observed_keys.equals(canonical_keys):
        raise V023PartitionContractError(
            f"{filename} partition rows are not in frozen canonical order"
        )
    return canonical, _sha256(raw)


def derive_partition_view(
    whole: WholeBundleValidated,
    *,
    partition: str,
    contract: FrozenArtifactContract,
) -> ValidatedPartitionView:
    """Derive one exact formal partition only from a valid whole capability."""

    _require_contract(contract)
    if type(whole) is not WholeBundleValidated:
        raise V023PartitionCapabilityError("Whole bundle capability has wrong type")
    if whole._contract_hash != contract.config_byte_sha256:
        raise V023PartitionCapabilityError("Whole capability contract binding changed")
    if partition not in PARTITION_COUNTS:
        raise V023PartitionContractError(f"Unknown frozen partition: {partition}")
    counts = PARTITION_COUNTS[partition]
    selected: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for filename in INPUT_FILENAMES:
        source = whole._frames[filename]
        subset = source.loc[source["partition"].eq(partition)].reset_index(drop=True)
        canonical, digest = _canonical_partition(
            subset,
            filename=filename,
            partition=partition,
            required_rows=counts[filename],
            contract=contract,
        )
        selected[filename] = canonical
        hashes[filename] = digest
    return ValidatedPartitionView(
        issuer=_ISSUER,
        partition=partition,
        contract_hash=contract.config_byte_sha256,
        frames=selected,
        frame_hashes=hashes,
        source_hashes=whole.source_hashes,
    )


def consume_partition_frames(
    view: ValidatedPartitionView,
    *,
    contract: FrozenArtifactContract,
) -> Mapping[str, pd.DataFrame]:
    """Revalidate mutation guards and return isolated consumer copies."""

    _require_contract(contract)
    if type(view) is not ValidatedPartitionView:
        raise V023PartitionCapabilityError("Partition capability has wrong type")
    if view._contract_hash != contract.config_byte_sha256:
        raise V023PartitionCapabilityError("Partition capability contract changed")
    counts = PARTITION_COUNTS.get(view.partition)
    if counts is None:
        raise V023PartitionCapabilityError("Partition capability identity changed")
    result: dict[str, pd.DataFrame] = {}
    for filename in INPUT_FILENAMES:
        canonical, digest = _canonical_partition(
            view._frames[filename],
            filename=filename,
            partition=view.partition,
            required_rows=counts[filename],
            contract=contract,
        )
        if digest != view._frame_hashes[filename]:
            raise V023PartitionCapabilityError(
                f"Partition capability mutated after issuance: {filename}"
            )
        result[filename] = canonical.copy(deep=True)
    return MappingProxyType(result)


def canonicalize_partition_output(
    frame: pd.DataFrame,
    *,
    filename: str,
    partition: str,
    required_rows: int,
    contract: FrozenArtifactContract,
) -> pd.DataFrame:
    """Apply a derived exact formal schema to one numerical output partition."""

    canonical, _ = _canonical_partition(
        frame,
        filename=filename,
        partition=partition,
        required_rows=required_rows,
        contract=contract,
        require_all_numeric_finite=False,
    )
    return canonical


__all__ = [
    "INPUT_FILENAMES",
    "PARTITION_COUNTS",
    "ValidatedPartitionView",
    "V023PartitionCapabilityError",
    "V023PartitionContractError",
    "V023WholeBundleContractError",
    "WholeBundleValidated",
    "WHOLE_COUNTS",
    "canonicalize_partition_output",
    "consume_partition_frames",
    "derive_partition_view",
    "validate_whole_bundle_from_root",
]
