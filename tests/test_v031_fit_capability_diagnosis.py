from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lifetwin.experiments import calendar_long_horizon_v019_io as io
from lifetwin.experiments import calendar_long_horizon_v019_fit as fit
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    canonical_csv_bytes,
    read_canonical_csv,
)
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    load_v024_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v019_partition import (
    PARTITION_COUNTS,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID as V2_PROTOCOL_ID,
)


def _exact_label_input_frames() -> dict[str, pd.DataFrame]:
    view = load_v024_contract_view()
    contract = view.artifacts
    partitions: list[str] = []
    cluster_ids: list[str] = []
    for partition, counts in PARTITION_COUNTS.items():
        for index in range(counts["clusters"]):
            partitions.append(partition)
            cluster_ids.append(f"{partition}-{index:04d}")
    cluster_count = len(cluster_ids)
    partition_values = np.asarray(partitions, dtype=object)
    cluster_values = np.asarray(cluster_ids, dtype=object)

    prefix_index = np.repeat(np.arange(cluster_count), len(contract.prefix_days))
    prefix = pd.DataFrame(
        {
            "protocol_id": np.full(len(prefix_index), contract.protocol_id),
            "partition": partition_values[prefix_index],
            "cluster_id": cluster_values[prefix_index],
            "prefix_day": np.tile(contract.prefix_days, cluster_count),
            "observed_retention_pct": (
                100.0 - 0.01 * np.tile(np.arange(len(contract.prefix_days)), cluster_count)
            ),
        },
        columns=contract.csv_schema("prefix_pack.csv").columns,
    )

    coordinate_index = np.repeat(
        np.arange(cluster_count), len(contract.forecast_days)
    )
    coordinates = pd.DataFrame(
        {
            "protocol_id": np.full(len(coordinate_index), contract.protocol_id),
            "partition": partition_values[coordinate_index],
            "cluster_id": cluster_values[coordinate_index],
            "forecast_day": np.tile(contract.forecast_days, cluster_count),
        },
        columns=contract.csv_schema("forecast_coordinates.csv").columns,
    )

    operating_schema = contract.csv_schema("operating_pack.csv")
    operating_values: dict[str, object] = {
        "protocol_id": np.full(cluster_count, contract.protocol_id),
        "partition": partition_values,
        "cluster_id": cluster_values,
    }
    for column in operating_schema.columns[3:]:
        if column.endswith("temperature_c"):
            value = 25.0
        elif column.endswith("soc_fraction") or column.endswith("dod_fraction"):
            value = 0.5
        elif column.endswith("efc_per_year"):
            value = 250.0
        else:
            value = 0.0
        operating_values[column] = np.full(cluster_count, value)
    operating = pd.DataFrame(operating_values, columns=operating_schema.columns)
    return {
        "prefix_pack.csv": prefix,
        "forecast_coordinates.csv": coordinates,
        "operating_pack.csv": operating,
    }


def test_exact_cardinality_fresh_bundle_revalidation_and_extraction(tmp_path) -> None:
    view = load_v024_contract_view()
    contract = view.artifacts
    source_frames = _exact_label_input_frames()
    for filename, frame in source_frames.items():
        (tmp_path / filename).write_bytes(
            canonical_csv_bytes(
                frame,
                contract.csv_schema(filename),
                contract,
                formal=True,
            )
        )
    for filename in io._GENERATION_FILES.difference(source_frames):
        (tmp_path / filename).write_bytes(
            f"v031-diagnostic-placeholder:{filename}\n".encode("ascii")
        )

    loaded_frames = {
        filename: read_canonical_csv(tmp_path / filename, contract, formal=True)
        for filename in io._LABEL_INPUTS
    }
    file_hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in io._GENERATION_FILES
    }
    ledger_prefix = (tmp_path / "exposure_log.jsonl").read_bytes()
    bundle = io.V024FreshGenerationBundle(
        _seal=io._SEAL,
        root=tmp_path,
        contract_view=view,
        identity=SimpleNamespace(attempt_id="v031-diagnostic-capability"),
        frames=loaded_frames,
        file_hashes=file_hashes,
        ledger_prefix=ledger_prefix,
    )

    prefix, coordinates, observed_contract = (
        io._extract_fresh_generation_frames_for_formal_fit_v024(bundle)
    )

    assert observed_contract is contract
    assert len(prefix) == 5_950 * len(contract.prefix_days)
    assert len(coordinates) == 5_950 * len(contract.forecast_days)
    assert prefix is not bundle._frames[0][1]
    assert coordinates is not bundle._frames[1][1]

    inherited_prefix, inherited_coordinates = fit._prepare_inputs(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        contract=contract,
    )
    assert set(inherited_prefix["protocol_id"]) == {V2_PROTOCOL_ID}
    assert set(inherited_coordinates["protocol_id"]) == {V2_PROTOCOL_ID}
