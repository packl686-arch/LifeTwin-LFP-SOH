"""Capability-bound whole-bundle and partition validation for V2.4.

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
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    V024ContractView,
    resolve_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v019_numeric_contract import (
    V024MemberFitNumericContractError,
    validate_member_fit_numeric_contract,
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
_MEMBER_TABLES = frozenset({"member_fit_diagnostics.csv", "member_forecast_bundle.csv"})


class V024WholeBundleContractError(ValueError):
    """Raised before slicing when a complete formal bundle is invalid."""


class V024PartitionContractError(ValueError):
    """Raised when a capability-derived partition violates its exact contract."""


class V024PartitionCapabilityError(ValueError):
    """Raised when a partition capability is forged, mutated or misused."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _partition_counts(
    view: V024ContractView,
) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for partition, values in view.partition_rows.items():
        parsed = {name: value for name, value in zip(_COUNT_ORDER, values, strict=True)}
        if any(value <= 0 for value in parsed.values()):
            raise V024PartitionContractError("Partition cardinality is not positive")
        result[partition] = MappingProxyType(parsed)
    return MappingProxyType(result)


_DEFAULT_VIEW = resolve_contract_view(None)
PARTITION_COUNTS = _partition_counts(_DEFAULT_VIEW)


def _whole_counts(view: V024ContractView) -> Mapping[str, int]:
    parsed = {name: int(view.whole_rows[name]) for name in INPUT_FILENAMES}
    if any(value <= 0 for value in parsed.values()):
        raise V024WholeBundleContractError("Whole-bundle cardinality is not positive")
    return MappingProxyType(parsed)


WHOLE_COUNTS = _whole_counts(_DEFAULT_VIEW)


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
            raise V024PartitionCapabilityError(
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
    """Opaque immutable view over one exact V2.4 partition."""

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
            raise V024PartitionCapabilityError(
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


def _resolve_contract(
    value: FrozenArtifactContract | V024ContractView,
) -> V024ContractView:
    try:
        return resolve_contract_view(value)
    except (TypeError, ValueError) as exc:
        raise V024PartitionCapabilityError("Contract view is invalid") from exc


def _direct_file(root: Path, filename: str) -> Path:
    path = root / filename
    if path.parent != root or not path.is_file() or path.is_symlink():
        raise V024WholeBundleContractError(f"Invalid direct artifact: {filename}")
    return path


def _require_finite_numeric(frame: pd.DataFrame, *, filename: str) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise V024PartitionContractError(
            f"{filename} contains a nonfinite numeric value"
        )


def _validate_member_tables(
    frames: Mapping[str, pd.DataFrame],
    *,
    error_type: type[ValueError] | None,
) -> None:
    try:
        validate_member_fit_numeric_contract(
            frames["member_fit_diagnostics.csv"],
            frames["member_forecast_bundle.csv"],
        )
    except V024MemberFitNumericContractError as exc:
        if error_type is None:
            raise
        raise error_type(str(exc)) from exc


def validate_whole_bundle_from_root(
    root: str | Path,
    contract: FrozenArtifactContract | V024ContractView,
) -> WholeBundleValidated:
    """Validate and bind the five authoritative complete label-free files."""

    view = _resolve_contract(contract)
    artifacts = view.artifacts
    source_root = Path(root).resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise V024WholeBundleContractError(
            "Label-free root is not a physical directory"
        )
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    whole_counts = _whole_counts(view)
    for filename in INPUT_FILENAMES:
        path = _direct_file(source_root, filename)
        try:
            frames[filename] = read_canonical_csv(path, artifacts, formal=True)
        except V015ArtifactError as exc:
            raise V024WholeBundleContractError(str(exc)) from exc
        if filename not in _MEMBER_TABLES:
            try:
                _require_finite_numeric(frames[filename], filename=filename)
            except V024PartitionContractError as exc:
                raise V024WholeBundleContractError(str(exc)) from exc
        if len(frames[filename]) != whole_counts[filename]:
            raise V024WholeBundleContractError(
                f"{filename} must contain exactly {whole_counts[filename]} rows"
            )
        raw = path.read_bytes()
        hashes[filename] = _sha256(raw)
        sizes[filename] = len(raw)
    if tuple(frames) != INPUT_FILENAMES:
        raise V024WholeBundleContractError("Whole-bundle file order changed")
    _validate_member_tables(frames, error_type=None)
    return WholeBundleValidated(
        issuer=_ISSUER,
        contract_hash=artifacts.config_byte_sha256,
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
        raise V024PartitionContractError(str(exc)) from exc
    if require_all_numeric_finite:
        _require_finite_numeric(canonical, filename=filename)
    observed_keys = frame.loc[:, list(schema.key)].reset_index(drop=True)
    canonical_keys = canonical.loc[:, list(schema.key)].reset_index(drop=True)
    if not observed_keys.equals(canonical_keys):
        raise V024PartitionContractError(
            f"{filename} partition rows are not in frozen canonical order"
        )
    return canonical, _sha256(raw)


def derive_partition_view(
    whole: WholeBundleValidated,
    *,
    partition: str,
    contract: FrozenArtifactContract | V024ContractView,
) -> ValidatedPartitionView:
    """Derive one exact formal partition only from a valid whole capability."""

    view = _resolve_contract(contract)
    artifacts = view.artifacts
    if type(whole) is not WholeBundleValidated:
        raise V024PartitionCapabilityError("Whole bundle capability has wrong type")
    if whole._contract_hash != artifacts.config_byte_sha256:
        raise V024PartitionCapabilityError("Whole capability contract binding changed")
    partition_counts = _partition_counts(view)
    if partition not in partition_counts:
        raise V024PartitionContractError(f"Unknown frozen partition: {partition}")
    counts = partition_counts[partition]
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
            contract=artifacts,
            require_all_numeric_finite=filename not in _MEMBER_TABLES,
        )
        selected[filename] = canonical
        hashes[filename] = digest
    _validate_member_tables(selected, error_type=V024PartitionContractError)
    return ValidatedPartitionView(
        issuer=_ISSUER,
        partition=partition,
        contract_hash=artifacts.config_byte_sha256,
        frames=selected,
        frame_hashes=hashes,
        source_hashes=whole.source_hashes,
    )


def consume_partition_frames(
    view: ValidatedPartitionView,
    *,
    contract: FrozenArtifactContract | V024ContractView,
) -> Mapping[str, pd.DataFrame]:
    """Revalidate mutation guards and return isolated consumer copies."""

    contract_view = _resolve_contract(contract)
    artifacts = contract_view.artifacts
    if type(view) is not ValidatedPartitionView:
        raise V024PartitionCapabilityError("Partition capability has wrong type")
    if view._contract_hash != artifacts.config_byte_sha256:
        raise V024PartitionCapabilityError("Partition capability contract changed")
    counts = _partition_counts(contract_view).get(view.partition)
    if counts is None:
        raise V024PartitionCapabilityError("Partition capability identity changed")
    result: dict[str, pd.DataFrame] = {}
    for filename in INPUT_FILENAMES:
        canonical, digest = _canonical_partition(
            view._frames[filename],
            filename=filename,
            partition=view.partition,
            required_rows=counts[filename],
            contract=artifacts,
            require_all_numeric_finite=filename not in _MEMBER_TABLES,
        )
        if digest != view._frame_hashes[filename]:
            raise V024PartitionCapabilityError(
                f"Partition capability mutated after issuance: {filename}"
            )
        result[filename] = canonical.copy(deep=True)
    _validate_member_tables(result, error_type=V024PartitionCapabilityError)
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
    "V024PartitionCapabilityError",
    "V024PartitionContractError",
    "V024WholeBundleContractError",
    "WholeBundleValidated",
    "WHOLE_COUNTS",
    "canonicalize_partition_output",
    "consume_partition_frames",
    "derive_partition_view",
    "validate_whole_bundle_from_root",
]
