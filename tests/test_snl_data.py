from __future__ import annotations

import csv
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from lifetwin.data.snl import (
    CYCLE_DATA_HEADERS,
    DATASET_ID,
    METADATA_COLUMNS,
    SNLCycleDataError,
    audit_snl_archive_structure,
    extract_snl_rpt_trajectories,
    load_snl_metadata,
    prepare_snl_cycle_inputs,
)


def _write_metadata_xlsx(path: Path) -> None:
    headers = [
        "Battery Archive Cell ID",
        "cathode",
        "anode",
        "temperature",
        "min SOC",
        "max SOC",
        "charge C rate",
        "discharge C rate",
        "Ah",
        "Factor",
    ]
    values: list[object] = [
        "SNL_18650_LFP_25C_20-80_0.5/1C_a",
        "LFP",
        "graphite",
        25,
        20,
        80,
        0.5,
        1,
        1.1,
        18650,
    ]

    def row_xml(row_index: int, row: list[object]) -> str:
        cells = []
        for column_index, value in enumerate(row):
            column = chr(ord("A") + column_index)
            reference = f"{column}{row_index}"
            if isinstance(value, str):
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>'
                )
            else:
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
        return f'<row r="{row_index}">{"".join(cells)}</row>'

    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="SNL" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{row_xml(1, headers)}{row_xml(2, values)}</sheetData>'
        "</worksheet>"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _cycle_row(
    cycle: int,
    start: datetime,
    *,
    full_check: bool,
    capacity: float,
) -> dict[str, object]:
    duration = timedelta(hours=4 if full_check else 2)
    return {
        "Cycle_Index": float(cycle),
        "Start_Time": start.isoformat(sep=" "),
        "End_Time": (start + duration).isoformat(sep=" "),
        "Test_Time (s)": cycle * 1000.0,
        "Min_Current (A)": -0.55,
        "Max_Current (A)": 0.55,
        "Min_Voltage (V)": 2.0 if full_check else 3.1,
        "Max_Voltage (V)": 3.6 if full_check else 3.4,
        "Charge_Capacity (Ah)": capacity,
        "Discharge_Capacity (Ah)": capacity,
        "Charge_Energy (Wh)": capacity * 3.3,
        "Discharge_Energy (Wh)": capacity * 3.2,
    }


def _write_snl_zip(path: Path, metadata: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 1)
    for cycle in range(1, 29):
        full = cycle in {1, 2, 3, 11, 12, 13, 15, 16, 17, 26, 27, 28}
        capacity = 1.05 - 0.002 * cycle if full else 0.55
        if cycle == 14:
            capacity = 0.0
        rows.append(_cycle_row(cycle, start, full_check=full, capacity=capacity))
        start += timedelta(hours=5 if full else 3)
        if cycle == 14:
            start += timedelta(hours=2)
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=CYCLE_DATA_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with ZipFile(path, "w") as archive:
        archive.writestr(str(metadata.iloc[0]["cycle_member"]), stream.getvalue())
        archive.writestr(str(metadata.iloc[0]["timeseries_member"]), "placeholder\n")


def test_snl_metadata_identity_mapping_and_rpt_extraction(tmp_path: Path) -> None:
    workbook = tmp_path / "metadata.xlsx"
    _write_metadata_xlsx(workbook)
    metadata, first_audit = load_snl_metadata(
        workbook, expected_cell_count=1, expected_condition_count=1
    )
    assert tuple(metadata.columns) == METADATA_COLUMNS
    assert metadata.iloc[0]["cell_id"] == "SNL_18650_LFP_25C_20-80_0.5-1C_a"
    assert metadata.iloc[0]["condition_id"] == "T25_SOC20-80_D1"
    _, replay_audit = load_snl_metadata(
        workbook,
        expected_lfp_rows_sha256=first_audit["canonical_raw_lfp_rows_sha256"],
        expected_cell_count=1,
        expected_condition_count=1,
    )
    assert replay_audit == first_audit

    bundle = tmp_path / "snl.zip"
    _write_snl_zip(bundle, metadata)
    structure = audit_snl_archive_structure(bundle, metadata)
    assert structure["member_count"] == 2
    assert structure["capacity_values_read"] is False

    trajectories, audit = extract_snl_rpt_trajectories(
        bundle, metadata, duplicate_visit_efc=4.0
    )
    assert set(trajectories["dataset_id"]) == {DATASET_ID}
    assert len(trajectories) == 3
    assert trajectories["visit_index"].tolist() == [0, 1, 2]
    assert trajectories.iloc[0]["capacity_retention_pct"] == pytest.approx(100.0)
    assert audit["minimum_rpt_visit_count"] == 3


def test_per_cycle_adapter_fails_closed_on_zero_capacity(tmp_path: Path) -> None:
    workbook = tmp_path / "metadata.xlsx"
    _write_metadata_xlsx(workbook)
    metadata, _ = load_snl_metadata(
        workbook, expected_cell_count=1, expected_condition_count=1
    )
    bundle = tmp_path / "snl.zip"
    _write_snl_zip(bundle, metadata)
    with pytest.raises(SNLCycleDataError, match="Non-positive"):
        prepare_snl_cycle_inputs(
            bundle,
            metadata,
            prefix_cycles=(5, 10),
            score_end_cycle=20,
        )
