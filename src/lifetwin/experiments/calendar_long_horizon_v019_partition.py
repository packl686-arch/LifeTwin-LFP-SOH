"""V0.19 candidate whole/partition wiring for member-fit structural masks.

This development-only adapter preserves the V0.18 capability types and frozen
artifact contract.  It replaces only the blanket finite check for the paired
member-fit tables with the exact V0.19 state-aware validator.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    read_canonical_csv,
)
from lifetwin.experiments import calendar_long_horizon_v018_partition as _v018
from lifetwin.experiments.calendar_long_horizon_v019_numeric_contract import (
    V024MemberFitNumericContractError,
    validate_member_fit_numeric_contract,
)


INPUT_FILENAMES = _v018.INPUT_FILENAMES
PARTITION_COUNTS = _v018.PARTITION_COUNTS
WHOLE_COUNTS = _v018.WHOLE_COUNTS
ValidatedPartitionView = _v018.ValidatedPartitionView
V023PartitionCapabilityError = _v018.V023PartitionCapabilityError
V023PartitionContractError = _v018.V023PartitionContractError
V023WholeBundleContractError = _v018.V023WholeBundleContractError
WholeBundleValidated = _v018.WholeBundleValidated
_MEMBER_TABLES = frozenset({"member_fit_diagnostics.csv", "member_forecast_bundle.csv"})


def _validate_member_tables(
    frames: Mapping[str, pd.DataFrame],
    *,
    error_type: type[ValueError],
) -> None:
    try:
        validate_member_fit_numeric_contract(
            frames["member_fit_diagnostics.csv"],
            frames["member_forecast_bundle.csv"],
        )
    except V024MemberFitNumericContractError as exc:
        raise error_type(str(exc)) from exc


def validate_whole_bundle_from_root(
    root: str | Path,
    contract: FrozenArtifactContract,
) -> WholeBundleValidated:
    """Validate five exact whole files, including member-fit status masks."""

    _v018._require_contract(contract)
    source_root = Path(root).resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise V023WholeBundleContractError(
            "Label-free root is not a physical directory"
        )
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for filename in INPUT_FILENAMES:
        path = _v018._direct_file(source_root, filename)
        try:
            frames[filename] = read_canonical_csv(path, contract, formal=True)
        except V015ArtifactError as exc:
            raise V023WholeBundleContractError(str(exc)) from exc
        if filename not in _MEMBER_TABLES:
            try:
                _v018._require_finite_numeric(frames[filename], filename=filename)
            except V023PartitionContractError as exc:
                raise V023WholeBundleContractError(str(exc)) from exc
        if len(frames[filename]) != WHOLE_COUNTS[filename]:
            raise V023WholeBundleContractError(
                f"{filename} must contain exactly {WHOLE_COUNTS[filename]} rows"
            )
        raw = path.read_bytes()
        hashes[filename] = _v018._sha256(raw)
        sizes[filename] = len(raw)
    _validate_member_tables(frames, error_type=V023WholeBundleContractError)
    return WholeBundleValidated(
        issuer=_v018._ISSUER,
        contract_hash=contract.config_byte_sha256,
        frames=frames,
        source_hashes=hashes,
        source_sizes=sizes,
    )


def _canonical_partition_table(
    frame: pd.DataFrame,
    *,
    filename: str,
    partition: str,
    required_rows: int,
    contract: FrozenArtifactContract,
) -> tuple[pd.DataFrame, str]:
    return _v018._canonical_partition(
        frame,
        filename=filename,
        partition=partition,
        required_rows=required_rows,
        contract=contract,
        require_all_numeric_finite=filename not in _MEMBER_TABLES,
    )


def derive_partition_view(
    whole: WholeBundleValidated,
    *,
    partition: str,
    contract: FrozenArtifactContract,
) -> ValidatedPartitionView:
    """Derive and revalidate one exact partition from a whole capability."""

    _v018._require_contract(contract)
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
        subset = (
            whole._frames[filename]
            .loc[whole._frames[filename]["partition"].eq(partition)]
            .reset_index(drop=True)
        )
        canonical, digest = _canonical_partition_table(
            subset,
            filename=filename,
            partition=partition,
            required_rows=counts[filename],
            contract=contract,
        )
        selected[filename] = canonical
        hashes[filename] = digest
    _validate_member_tables(selected, error_type=V023PartitionContractError)
    return ValidatedPartitionView(
        issuer=_v018._ISSUER,
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
    """Recheck V0.19 masks and V0.18 mutation hashes before consumption."""

    _v018._require_contract(contract)
    if type(view) is not ValidatedPartitionView:
        raise V023PartitionCapabilityError("Partition capability has wrong type")
    if view._contract_hash != contract.config_byte_sha256:
        raise V023PartitionCapabilityError("Partition capability contract changed")
    counts = PARTITION_COUNTS.get(view.partition)
    if counts is None:
        raise V023PartitionCapabilityError("Partition capability identity changed")
    result: dict[str, pd.DataFrame] = {}
    for filename in INPUT_FILENAMES:
        canonical, digest = _canonical_partition_table(
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
    _validate_member_tables(result, error_type=V023PartitionCapabilityError)
    return MappingProxyType(result)


canonicalize_partition_output = _v018.canonicalize_partition_output


__all__ = [
    "INPUT_FILENAMES",
    "PARTITION_COUNTS",
    "ValidatedPartitionView",
    "V023PartitionCapabilityError",
    "V023PartitionContractError",
    "V023WholeBundleContractError",
    "WHOLE_COUNTS",
    "WholeBundleValidated",
    "canonicalize_partition_output",
    "consume_partition_frames",
    "derive_partition_view",
    "validate_whole_bundle_from_root",
]
