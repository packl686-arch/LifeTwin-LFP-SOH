"""Fail-closed adapter for the privately shared Battery Archive SNL LFP bundle."""

from __future__ import annotations

import csv
from datetime import datetime
from io import TextIOWrapper
import math
from pathlib import Path
import posixpath
import re
from typing import Mapping
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pandas as pd
import numpy as np

from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)


DATASET_ID = "SNL_BATTERY_ARCHIVE_LFP_CYCLE_V1"
METADATA_HEADERS = (
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
)
CYCLE_DATA_HEADERS = (
    "Cycle_Index",
    "Start_Time",
    "End_Time",
    "Test_Time (s)",
    "Min_Current (A)",
    "Max_Current (A)",
    "Min_Voltage (V)",
    "Max_Voltage (V)",
    "Charge_Capacity (Ah)",
    "Discharge_Capacity (Ah)",
    "Charge_Energy (Wh)",
    "Discharge_Energy (Wh)",
)
METADATA_COLUMNS = (
    "dataset_id",
    "cell_id",
    "source_cell_id",
    "condition_id",
    "temperature_c",
    "min_soc_pct",
    "max_soc_pct",
    "dod_fraction",
    "charge_c_rate",
    "discharge_c_rate",
    "nominal_capacity_ah",
    "form_factor",
    "cycle_member",
    "timeseries_member",
)
CANONICAL_CYCLE_COLUMNS = (
    "dataset_id",
    "cell_id",
    "condition_id",
    "temperature_c",
    "min_soc_pct",
    "max_soc_pct",
    "dod_fraction",
    "charge_c_rate",
    "discharge_c_rate",
    "cycle_index",
    "equivalent_full_cycle",
    "discharge_capacity_ah",
)
TARGET_PREFIX_COLUMNS = (*CANONICAL_CYCLE_COLUMNS, "prefix_cycle")
TRUTH_COLUMNS = (
    "dataset_id",
    "cell_id",
    "condition_id",
    "cycle_index",
    "discharge_capacity_ah",
)
RPT_TRAJECTORY_COLUMNS = (
    "dataset_id",
    "cell_id",
    "condition_id",
    "temperature_c",
    "min_soc_pct",
    "max_soc_pct",
    "dod_fraction",
    "charge_c_rate",
    "discharge_c_rate",
    "visit_index",
    "elapsed_days",
    "equivalent_full_cycles",
    "capacity_ah",
    "capacity_retention_pct",
    "rpt_cycle_count",
)
RPT_REPEAT_COLUMNS = (
    "dataset_id",
    "cell_id",
    "condition_id",
    "visit_index",
    "repeat_index",
    "source_cycle_index",
    "measurement_time",
    "elapsed_days",
    "equivalent_full_cycles",
    "capacity_ah",
    "retention_pct",
    "visit_center_capacity_ah",
    "visit_center_retention_pct",
    "rpt_cycle_count",
)
SOURCE_TRAINING_COLUMNS = (
    "dataset_id",
    "cell_id",
    "paper_split",
    "cycle_index",
    "discharge_capacity_ah",
)

_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


class SNLCycleDataError(ValueError):
    """Raised when the SNL artifacts violate the frozen adapter contract."""


def _cell_column_index(reference: str) -> int:
    match = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", reference)
    if match is None:
        raise SNLCycleDataError(f"Invalid XLSX cell reference: {reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _shared_strings(archive: ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(name))
    return [
        "".join(node.text or "" for node in item.findall(f".//{{{_SHEET_NS}}}t"))
        for item in root.findall(f"{{{_SHEET_NS}}}si")
    ]


def _worksheet_path(archive: ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relation_id: str | None = None
    for sheet in workbook.findall(f".//{{{_SHEET_NS}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relation_id = sheet.attrib.get(f"{{{_DOC_REL_NS}}}id")
            break
    if relation_id is None:
        raise SNLCycleDataError(f"XLSX sheet is missing: {sheet_name}")
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relation in relationships.findall(f"{{{_REL_NS}}}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib.get("Target", "")
            path = posixpath.normpath(posixpath.join("xl", target.lstrip("/")))
            if path not in archive.namelist():
                raise SNLCycleDataError(f"XLSX worksheet target is missing: {path}")
            return path
    raise SNLCycleDataError(f"XLSX relationship is missing: {relation_id}")


def _xlsx_scalar(cell: ElementTree.Element, shared: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{{{_SHEET_NS}}}t"))
    value_node = cell.find(f"{{{_SHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    text = value_node.text
    if cell_type == "s":
        try:
            return shared[int(text)]
        except (IndexError, ValueError) as exc:
            raise SNLCycleDataError("Invalid XLSX shared-string index") from exc
    if cell_type in {"str", "e"}:
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _xlsx_rows(path: Path, *, sheet_name: str) -> list[list[object]]:
    try:
        with ZipFile(path) as archive:
            shared = _shared_strings(archive)
            root = ElementTree.fromstring(
                archive.read(_worksheet_path(archive, sheet_name))
            )
    except (BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise SNLCycleDataError(f"Cannot parse metadata workbook: {path}") from exc
    rows: list[list[object]] = []
    for row in root.findall(f".//{{{_SHEET_NS}}}sheetData/{{{_SHEET_NS}}}row"):
        values: dict[int, object] = {}
        for cell in row.findall(f"{{{_SHEET_NS}}}c"):
            reference = cell.attrib.get("r", "")
            values[_cell_column_index(reference)] = _xlsx_scalar(cell, shared)
        if values:
            width = max(values) + 1
            rows.append([values.get(index) for index in range(width)])
    return rows


def _require_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SNLCycleDataError(f"Metadata field must be numeric: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise SNLCycleDataError(f"Metadata field must be finite: {field}")
    return result


def _rate_text(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _condition_id(record: Mapping[str, object]) -> str:
    return (
        f"T{_rate_text(float(record['temperature_c']))}_"
        f"SOC{_rate_text(float(record['min_soc_pct']))}-"
        f"{_rate_text(float(record['max_soc_pct']))}_"
        f"D{_rate_text(float(record['discharge_c_rate']))}"
    )


def load_snl_metadata(
    path: str | Path,
    *,
    expected_lfp_rows_sha256: str | None = None,
    expected_cell_count: int = 30,
    expected_condition_count: int = 12,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read only the SNL metadata sheet and freeze the LFP identity mapping."""
    workbook_path = Path(path)
    rows = _xlsx_rows(workbook_path, sheet_name="SNL")
    if not rows or tuple(rows[0]) != METADATA_HEADERS:
        raise SNLCycleDataError("SNL metadata headers changed")
    raw_records = [
        dict(zip(METADATA_HEADERS, row, strict=False))
        for row in rows[1:]
        if row and str(row[1]).strip().casefold() == "lfp"
    ]
    raw_hash = canonical_json_sha256(raw_records)
    if expected_lfp_rows_sha256 and raw_hash != expected_lfp_rows_sha256:
        raise SNLCycleDataError("Canonical LFP metadata rows changed")

    records: list[dict[str, object]] = []
    for raw in raw_records:
        source_cell_id = str(raw["Battery Archive Cell ID"]).strip()
        if source_cell_id.count("/") != 1:
            raise SNLCycleDataError(
                f"Metadata cell ID has an unexpected rate separator: {source_cell_id}"
            )
        cell_id = source_cell_id.replace("/", "-", 1)
        temperature = _require_number(raw["temperature"], field="temperature")
        min_soc = _require_number(raw["min SOC"], field="min SOC")
        max_soc = _require_number(raw["max SOC"], field="max SOC")
        charge_rate = _require_number(raw["charge C rate"], field="charge C rate")
        discharge_rate = _require_number(
            raw["discharge C rate"], field="discharge C rate"
        )
        nominal_capacity = _require_number(raw["Ah"], field="Ah")
        form_factor = _require_number(raw["Factor"], field="Factor")
        if not 0.0 <= min_soc < max_soc <= 100.0:
            raise SNLCycleDataError(f"Invalid SOC interval for {source_cell_id}")
        record: dict[str, object] = {
            "dataset_id": DATASET_ID,
            "cell_id": cell_id,
            "source_cell_id": source_cell_id,
            "condition_id": "",
            "temperature_c": temperature,
            "min_soc_pct": min_soc,
            "max_soc_pct": max_soc,
            "dod_fraction": (max_soc - min_soc) / 100.0,
            "charge_c_rate": charge_rate,
            "discharge_c_rate": discharge_rate,
            "nominal_capacity_ah": nominal_capacity,
            "form_factor": int(form_factor),
            "cycle_member": f"SNL LFP/{cell_id}_cycle_data.csv",
            "timeseries_member": f"SNL LFP/{cell_id}_timeseries.csv",
        }
        record["condition_id"] = _condition_id(record)
        records.append(record)
    metadata = pd.DataFrame(records, columns=METADATA_COLUMNS).sort_values(
        "cell_id", kind="stable", ignore_index=True
    )
    if len(metadata) != expected_cell_count or metadata["cell_id"].nunique() != len(
        metadata
    ):
        raise SNLCycleDataError("SNL LFP physical-cell count changed")
    if metadata["condition_id"].nunique() != expected_condition_count:
        raise SNLCycleDataError("SNL LFP condition-cluster count changed")
    audit: dict[str, object] = {
        "schema_version": "lifetwin.snl_metadata_audit.v1",
        "dataset_id": DATASET_ID,
        "lfp_row_count": len(metadata),
        "physical_cell_count": int(metadata["cell_id"].nunique()),
        "condition_cluster_count": int(metadata["condition_id"].nunique()),
        "canonical_raw_lfp_rows_sha256": raw_hash,
        "canonical_identity_table_sha256": canonical_frame_sha256(
            metadata, METADATA_COLUMNS
        ),
        "capacity_values_read": False,
    }
    return metadata, audit


def audit_snl_archive_structure(
    zip_path: str | Path,
    metadata: pd.DataFrame,
) -> dict[str, object]:
    """Verify archive identities and headers without parsing capacity values."""
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise SNLCycleDataError("Metadata frame columns changed")
    try:
        with ZipFile(zip_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members if not member.is_dir()]
            if len(names) != len(set(names)):
                raise SNLCycleDataError("SNL archive has duplicate member names")
            expected_cycle = set(metadata["cycle_member"].astype(str))
            expected_timeseries = set(metadata["timeseries_member"].astype(str))
            if set(names) != expected_cycle | expected_timeseries:
                raise SNLCycleDataError("SNL archive members do not match metadata")
            for member_name in sorted(expected_cycle):
                with TextIOWrapper(
                    archive.open(member_name), encoding="utf-8-sig", newline=""
                ) as stream:
                    headers = next(csv.reader(stream), None)
                if tuple(headers or ()) != CYCLE_DATA_HEADERS:
                    raise SNLCycleDataError(
                        f"Cycle-summary headers changed: {member_name}"
                    )
            uncompressed_bytes = sum(member.file_size for member in members)
    except (BadZipFile, KeyError, OSError) as exc:
        raise SNLCycleDataError(f"Cannot audit SNL archive: {zip_path}") from exc
    return {
        "schema_version": "lifetwin.snl_archive_structure_audit.v1",
        "dataset_id": DATASET_ID,
        "member_count": len(names),
        "cycle_summary_member_count": len(expected_cycle),
        "timeseries_member_count": len(expected_timeseries),
        "uncompressed_byte_size": uncompressed_bytes,
        "cycle_summary_headers_sha256": canonical_json_sha256(list(CYCLE_DATA_HEADERS)),
        "member_identity_sha256": canonical_json_sha256(sorted(names)),
        "capacity_values_read": False,
    }


def _finite_positive(text: object, *, field: str, cell_id: str, cycle: int) -> float:
    try:
        value = float(str(text))
    except (TypeError, ValueError) as exc:
        raise SNLCycleDataError(f"Invalid {field} for {cell_id} cycle {cycle}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise SNLCycleDataError(
            f"Non-positive or non-finite {field} for {cell_id} cycle {cycle}"
        )
    return value


def _integer_cycle(text: object, *, cell_id: str) -> int:
    try:
        value = float(str(text))
    except (TypeError, ValueError) as exc:
        raise SNLCycleDataError(f"Invalid cycle index for {cell_id}") from exc
    if not math.isfinite(value) or value != math.floor(value):
        raise SNLCycleDataError(f"Non-integer cycle index for {cell_id}: {text}")
    return int(value)


def prepare_snl_cycle_inputs(
    zip_path: str | Path,
    metadata: pd.DataFrame,
    *,
    prefix_cycles: tuple[int, ...],
    score_end_cycle: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create prefix-only prediction inputs and a physically separate truth table."""
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise SNLCycleDataError("Metadata frame columns changed")
    prefixes = tuple(sorted(set(int(value) for value in prefix_cycles)))
    if not prefixes or prefixes[0] < 5 or prefixes[-1] >= score_end_cycle:
        raise SNLCycleDataError("Invalid frozen prefix/score horizon")
    cycle_records: list[dict[str, object]] = []
    try:
        with ZipFile(zip_path) as archive:
            for meta in metadata.to_dict(orient="records"):
                cell_id = str(meta["cell_id"])
                member_name = str(meta["cycle_member"])
                selected: dict[int, float] = {}
                with TextIOWrapper(
                    archive.open(member_name), encoding="utf-8-sig", newline=""
                ) as stream:
                    reader = csv.DictReader(stream)
                    if tuple(reader.fieldnames or ()) != CYCLE_DATA_HEADERS:
                        raise SNLCycleDataError(
                            f"Cycle-summary headers changed: {member_name}"
                        )
                    for row in reader:
                        cycle = _integer_cycle(row["Cycle_Index"], cell_id=cell_id)
                        if not 1 <= cycle <= score_end_cycle:
                            continue
                        if cycle in selected:
                            raise SNLCycleDataError(
                                f"Duplicate cycle {cycle} for {cell_id}"
                            )
                        selected[cycle] = _finite_positive(
                            row["Discharge_Capacity (Ah)"],
                            field="discharge capacity",
                            cell_id=cell_id,
                            cycle=cycle,
                        )
                required = set(range(1, score_end_cycle + 1))
                if set(selected) != required:
                    missing = sorted(required - set(selected))
                    raise SNLCycleDataError(
                        f"Required cycle support changed for {cell_id}; "
                        f"first_missing={missing[:5]}"
                    )
                for cycle in range(1, score_end_cycle + 1):
                    cycle_records.append(
                        {
                            "dataset_id": DATASET_ID,
                            "cell_id": cell_id,
                            "condition_id": str(meta["condition_id"]),
                            "temperature_c": float(meta["temperature_c"]),
                            "min_soc_pct": float(meta["min_soc_pct"]),
                            "max_soc_pct": float(meta["max_soc_pct"]),
                            "dod_fraction": float(meta["dod_fraction"]),
                            "charge_c_rate": float(meta["charge_c_rate"]),
                            "discharge_c_rate": float(meta["discharge_c_rate"]),
                            "cycle_index": cycle,
                            "equivalent_full_cycle": (
                                cycle * float(meta["dod_fraction"])
                            ),
                            "discharge_capacity_ah": selected[cycle],
                        }
                    )
    except (BadZipFile, KeyError, OSError) as exc:
        raise SNLCycleDataError(
            f"Cannot parse SNL cycle summaries: {zip_path}"
        ) from exc
    cycles = pd.DataFrame(cycle_records, columns=CANONICAL_CYCLE_COLUMNS).sort_values(
        ["cell_id", "cycle_index"], kind="stable", ignore_index=True
    )
    prefix_frames: list[pd.DataFrame] = []
    for prefix_cycle in prefixes:
        frame = cycles.loc[cycles["cycle_index"] <= prefix_cycle].copy()
        frame["prefix_cycle"] = prefix_cycle
        prefix_frames.append(frame.loc[:, TARGET_PREFIX_COLUMNS])
    target_prefixes = pd.concat(prefix_frames, ignore_index=True).sort_values(
        ["cell_id", "prefix_cycle", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )
    truth = cycles.loc[:, TRUTH_COLUMNS].copy()
    audit = {
        "schema_version": "lifetwin.snl_cycle_prepare_audit.v1",
        "dataset_id": DATASET_ID,
        "physical_cell_count": int(cycles["cell_id"].nunique()),
        "condition_cluster_count": int(cycles["condition_id"].nunique()),
        "score_end_cycle": score_end_cycle,
        "prefix_cycles": list(prefixes),
        "canonical_cycle_row_count": len(cycles),
        "target_prefix_row_count": len(target_prefixes),
        "truth_row_count": len(truth),
        "target_prefix_sha256": canonical_frame_sha256(
            target_prefixes, TARGET_PREFIX_COLUMNS
        ),
        "truth_sha256": canonical_frame_sha256(truth, TRUTH_COLUMNS),
        "prediction_generated": False,
        "truth_linked_to_prediction": False,
    }
    return target_prefixes, truth, audit


def prepare_source_training_capacity(
    source: pd.DataFrame,
    *,
    expected_cell_count: int,
    score_end_cycle: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reduce the frozen MATR training artifact to capacity-only source histories."""
    missing = sorted(set(SOURCE_TRAINING_COLUMNS) - set(source.columns))
    if missing:
        raise SNLCycleDataError(f"Source training columns are missing: {missing}")
    result = source.loc[:, SOURCE_TRAINING_COLUMNS].copy()
    if set(result["paper_split"].astype(str)) != {"train"}:
        raise SNLCycleDataError("Source artifact is not the frozen training split")
    result["cycle_index"] = pd.to_numeric(result["cycle_index"], errors="raise")
    result["discharge_capacity_ah"] = pd.to_numeric(
        result["discharge_capacity_ah"], errors="raise"
    )
    cycle_values = result["cycle_index"].to_numpy(dtype=float)
    if not np.equal(cycle_values, np.floor(cycle_values)).all():
        raise SNLCycleDataError("Source cycle indices must be integers")
    result["cycle_index"] = result["cycle_index"].astype(int)
    result = result.loc[result["cycle_index"] <= score_end_cycle].copy()
    if result["cell_id"].nunique() != expected_cell_count:
        raise SNLCycleDataError("Source training cell count changed")
    for cell_id, cell in result.groupby("cell_id", sort=True):
        if sorted(cell["cycle_index"].tolist()) != list(range(1, score_end_cycle + 1)):
            raise SNLCycleDataError(
                f"Source training support is incomplete for {cell_id}"
            )
        values = cell["discharge_capacity_ah"].to_numpy(dtype=float)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise SNLCycleDataError(f"Source capacity is invalid for {cell_id}")
    result = result.sort_values(
        ["cell_id", "cycle_index"], kind="stable", ignore_index=True
    )
    audit = {
        "schema_version": "lifetwin.snl_source_training_capacity_audit.v1",
        "source_dataset_id": str(result["dataset_id"].iloc[0]),
        "training_cell_count": int(result["cell_id"].nunique()),
        "score_end_cycle": score_end_cycle,
        "row_count": len(result),
        "canonical_capacity_training_sha256": canonical_frame_sha256(
            result, SOURCE_TRAINING_COLUMNS
        ),
        "target_dataset_rows_present": False,
    }
    return result, audit


def _parse_timestamp(value: object, *, field: str, cell_id: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise SNLCycleDataError(
            f"Invalid {field} timestamp for {cell_id}: {value}"
        ) from exc


def _finite_number(value: object, *, field: str, cell_id: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise SNLCycleDataError(f"Invalid {field} for {cell_id}") from exc
    if not math.isfinite(result):
        raise SNLCycleDataError(f"Non-finite {field} for {cell_id}")
    return result


def _is_standalone_lfp_capacity_check(
    row: Mapping[str, object],
    *,
    nominal_capacity_ah: float,
    cell_id: str,
) -> bool:
    min_current = _finite_number(
        row["Min_Current (A)"], field="minimum current", cell_id=cell_id
    )
    max_current = _finite_number(
        row["Max_Current (A)"], field="maximum current", cell_id=cell_id
    )
    min_voltage = _finite_number(
        row["Min_Voltage (V)"], field="minimum voltage", cell_id=cell_id
    )
    max_voltage = _finite_number(
        row["Max_Voltage (V)"], field="maximum voltage", cell_id=cell_id
    )
    discharge_capacity = _finite_number(
        row["Discharge_Capacity (Ah)"],
        field="discharge capacity",
        cell_id=cell_id,
    )
    current_lower = 0.4 * nominal_capacity_ah
    current_upper = 0.65 * nominal_capacity_ah
    return (
        -current_upper <= min_current <= -current_lower
        and current_lower <= max_current <= current_upper
        and min_voltage <= 2.05
        and max_voltage >= 3.55
        and 0.6 * nominal_capacity_ah <= discharge_capacity <= 1.3 * nominal_capacity_ah
    )


def _candidate_runs(candidates: list[bool]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index, candidate in enumerate(candidates):
        if candidate:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _merge_position_groups(groups: list[set[int]]) -> list[set[int]]:
    merged: list[set[int]] = []
    for group in sorted(groups, key=lambda value: min(value)):
        if not group:
            continue
        overlaps = [index for index, existing in enumerate(merged) if existing & group]
        if not overlaps:
            merged.append(set(group))
            continue
        combined = set(group)
        for index in reversed(overlaps):
            combined.update(merged.pop(index))
        merged.append(combined)
    return sorted(merged, key=lambda value: min(value))


def _detected_rpt_groups_for_cell(
    rows: list[dict[str, object]],
    *,
    meta: Mapping[str, object],
    rest_gap_hours: float,
    duplicate_visit_efc: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    cell_id = str(meta["cell_id"])
    nominal_capacity = float(meta["nominal_capacity_ah"])
    parsed: list[dict[str, object]] = []
    seen_coordinates: set[tuple[str, str, str]] = set()
    duplicate_rows = 0
    cumulative_ah = 0.0
    for row in rows:
        coordinate = (
            str(row["Cycle_Index"]),
            str(row["Start_Time"]),
            str(row["End_Time"]),
        )
        if coordinate in seen_coordinates:
            duplicate_rows += 1
            continue
        seen_coordinates.add(coordinate)
        start = _parse_timestamp(row["Start_Time"], field="start", cell_id=cell_id)
        end = _parse_timestamp(row["End_Time"], field="end", cell_id=cell_id)
        if end < start:
            raise SNLCycleDataError(f"Negative row duration for {cell_id}")
        discharge_capacity = _finite_number(
            row["Discharge_Capacity (Ah)"],
            field="discharge capacity",
            cell_id=cell_id,
        )
        if discharge_capacity < 0.0:
            raise SNLCycleDataError(f"Negative discharge capacity for {cell_id}")
        cumulative_ah += discharge_capacity
        parsed.append(
            {
                "row": row,
                "start": start,
                "end": end,
                "capacity_ah": discharge_capacity,
                "cumulative_efc": cumulative_ah / nominal_capacity,
            }
        )
    if len(parsed) < 3:
        raise SNLCycleDataError(f"Too few SNL rows for {cell_id}")
    candidates = [
        _is_standalone_lfp_capacity_check(
            item["row"], nominal_capacity_ah=nominal_capacity, cell_id=cell_id
        )
        for item in parsed
    ]
    groups: list[set[int]] = []
    initial = [index for index in range(min(8, len(parsed))) if candidates[index]][:3]
    if len(initial) != 3:
        raise SNLCycleDataError(
            f"Initial three-cycle capacity check missing for {cell_id}"
        )
    groups.append(set(initial))

    runs = _candidate_runs(candidates)
    for run in runs:
        if 3 <= len(run) <= 8:
            groups.append(set(run))

    for right in range(1, len(parsed)):
        gap_hours = (
            parsed[right]["start"] - parsed[right - 1]["end"]
        ).total_seconds() / 3600.0
        if gap_hours < rest_gap_hours:
            continue
        before = [
            index for index in range(max(0, right - 8), right) if candidates[index]
        ][-3:]
        after = [
            index
            for index in range(right, min(len(parsed), right + 8))
            if candidates[index]
        ][:3]
        if len(before) == 3:
            groups.append(set(before))
        if len(after) == 3:
            groups.append(set(after))

    final = [
        index
        for index in range(max(0, len(parsed) - 8), len(parsed))
        if candidates[index]
    ][-3:]
    if len(final) == 3:
        groups.append(set(final))
    groups = _merge_position_groups(groups)
    raw_visits: list[dict[str, object]] = []
    for group in groups:
        positions = sorted(group)
        if len(positions) < 3:
            continue
        capacities = [float(parsed[index]["capacity_ah"]) for index in positions]
        efcs = [float(parsed[index]["cumulative_efc"]) for index in positions]
        timestamps = [parsed[index]["end"] for index in positions]
        raw_visits.append(
            {
                "positions": set(positions),
                "capacity_ah": float(pd.Series(capacities).median()),
                "equivalent_full_cycles_raw": float(pd.Series(efcs).median()),
                "timestamp": sorted(timestamps)[len(timestamps) // 2],
                "rpt_cycle_count": len(positions),
            }
        )
    raw_visits.sort(key=lambda value: float(value["equivalent_full_cycles_raw"]))
    collapsed: list[dict[str, object]] = []
    for visit in raw_visits:
        if (
            collapsed
            and float(visit["equivalent_full_cycles_raw"])
            - float(collapsed[-1]["equivalent_full_cycles_raw"])
            <= duplicate_visit_efc
        ):
            combined_positions = set(collapsed[-1]["positions"]) | set(
                visit["positions"]
            )
            positions = sorted(combined_positions)
            capacities = [float(parsed[index]["capacity_ah"]) for index in positions]
            efcs = [float(parsed[index]["cumulative_efc"]) for index in positions]
            timestamps = [parsed[index]["end"] for index in positions]
            collapsed[-1] = {
                "positions": combined_positions,
                "capacity_ah": float(pd.Series(capacities).median()),
                "equivalent_full_cycles_raw": float(pd.Series(efcs).median()),
                "timestamp": sorted(timestamps)[len(timestamps) // 2],
                "rpt_cycle_count": len(positions),
            }
        else:
            collapsed.append(visit)
    if len(collapsed) < 2:
        raise SNLCycleDataError(f"Fewer than two RPT visits detected for {cell_id}")
    return (
        parsed,
        raw_visits,
        collapsed,
        {
            "cell_id": cell_id,
            "raw_row_count": len(rows),
            "deduplicated_row_count": len(parsed),
            "exact_duplicate_row_count": duplicate_rows,
            "standalone_capacity_check_row_count": int(sum(candidates)),
        },
    )


def _rpt_visits_for_cell(
    rows: list[dict[str, object]],
    *,
    meta: Mapping[str, object],
    rest_gap_hours: float,
    duplicate_visit_efc: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    parsed, _, collapsed, detection_audit = _detected_rpt_groups_for_cell(
        rows,
        meta=meta,
        rest_gap_hours=rest_gap_hours,
        duplicate_visit_efc=duplicate_visit_efc,
    )
    cell_id = str(meta["cell_id"])
    initial_capacity = float(collapsed[0]["capacity_ah"])
    initial_efc = float(collapsed[0]["equivalent_full_cycles_raw"])
    initial_time = collapsed[0]["timestamp"]
    if initial_capacity <= 0.0:
        raise SNLCycleDataError(f"Invalid initial RPT capacity for {cell_id}")
    visits: list[dict[str, object]] = []
    for visit_index, visit in enumerate(collapsed):
        capacity = float(visit["capacity_ah"])
        elapsed_days = (visit["timestamp"] - initial_time).total_seconds() / 86400.0
        efc = float(visit["equivalent_full_cycles_raw"]) - initial_efc
        if elapsed_days < 0.0 or efc < -1e-9:
            raise SNLCycleDataError(f"RPT visit order changed for {cell_id}")
        visits.append(
            {
                "dataset_id": DATASET_ID,
                "cell_id": cell_id,
                "condition_id": str(meta["condition_id"]),
                "temperature_c": float(meta["temperature_c"]),
                "min_soc_pct": float(meta["min_soc_pct"]),
                "max_soc_pct": float(meta["max_soc_pct"]),
                "dod_fraction": float(meta["dod_fraction"]),
                "charge_c_rate": float(meta["charge_c_rate"]),
                "discharge_c_rate": float(meta["discharge_c_rate"]),
                "visit_index": visit_index,
                "elapsed_days": elapsed_days,
                "equivalent_full_cycles": max(0.0, efc),
                "capacity_ah": capacity,
                "capacity_retention_pct": 100.0 * capacity / initial_capacity,
                "rpt_cycle_count": int(visit["rpt_cycle_count"]),
            }
        )
    return visits, {
        **detection_audit,
        "rpt_visit_count": len(visits),
        "maximum_elapsed_days": float(visits[-1]["elapsed_days"]),
        "maximum_equivalent_full_cycles": float(visits[-1]["equivalent_full_cycles"]),
    }


def _rpt_repeats_for_cell(
    rows: list[dict[str, object]],
    *,
    meta: Mapping[str, object],
    rest_gap_hours: float,
    duplicate_visit_efc: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    parsed, raw_visits, _, detection_audit = _detected_rpt_groups_for_cell(
        rows,
        meta=meta,
        rest_gap_hours=rest_gap_hours,
        duplicate_visit_efc=duplicate_visit_efc,
    )
    cell_id = str(meta["cell_id"])
    initial_capacity = float(raw_visits[0]["capacity_ah"])
    initial_efc = float(raw_visits[0]["equivalent_full_cycles_raw"])
    initial_time = raw_visits[0]["timestamp"]
    records: list[dict[str, object]] = []
    for visit_index, visit in enumerate(raw_visits):
        positions = sorted(visit["positions"])
        center_capacity = float(visit["capacity_ah"])
        for repeat_index, position in enumerate(positions):
            item = parsed[position]
            capacity = float(item["capacity_ah"])
            elapsed_days = (item["end"] - initial_time).total_seconds() / 86400.0
            equivalent_full_cycles = float(item["cumulative_efc"]) - initial_efc
            records.append(
                {
                    "dataset_id": DATASET_ID,
                    "cell_id": cell_id,
                    "condition_id": str(meta["condition_id"]),
                    "visit_index": visit_index,
                    "repeat_index": repeat_index,
                    "source_cycle_index": _integer_cycle(
                        item["row"]["Cycle_Index"], cell_id=cell_id
                    ),
                    "measurement_time": item["end"].isoformat(),
                    "elapsed_days": max(0.0, elapsed_days),
                    "equivalent_full_cycles": max(0.0, equivalent_full_cycles),
                    "capacity_ah": capacity,
                    "retention_pct": 100.0 * capacity / initial_capacity,
                    "visit_center_capacity_ah": center_capacity,
                    "visit_center_retention_pct": (
                        100.0 * center_capacity / initial_capacity
                    ),
                    "rpt_cycle_count": len(positions),
                }
            )
    counts = pd.Series([len(visit["positions"]) for visit in raw_visits], dtype=int)
    return records, {
        **detection_audit,
        "rpt_visit_count": len(raw_visits),
        "repeat_measurement_count": len(records),
        "minimum_repeats_per_visit": int(counts.min()),
        "median_repeats_per_visit": float(counts.median()),
        "maximum_repeats_per_visit": int(counts.max()),
    }


def extract_snl_rpt_trajectories(
    zip_path: str | Path,
    metadata: pd.DataFrame,
    *,
    rest_gap_hours: float = 1.0,
    duplicate_visit_efc: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract periodic 0.5C full-depth capacity-check trajectories."""
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise SNLCycleDataError("Metadata frame columns changed")
    if rest_gap_hours <= 0.0 or duplicate_visit_efc < 0.0:
        raise SNLCycleDataError("RPT extraction thresholds must be non-negative")
    records: list[dict[str, object]] = []
    cell_audits: list[dict[str, object]] = []
    try:
        with ZipFile(zip_path) as archive:
            for meta in metadata.to_dict(orient="records"):
                member_name = str(meta["cycle_member"])
                with TextIOWrapper(
                    archive.open(member_name), encoding="utf-8-sig", newline=""
                ) as stream:
                    reader = csv.DictReader(stream)
                    if tuple(reader.fieldnames or ()) != CYCLE_DATA_HEADERS:
                        raise SNLCycleDataError(
                            f"Cycle-summary headers changed: {member_name}"
                        )
                    cell_rows = list(reader)
                visits, cell_audit = _rpt_visits_for_cell(
                    cell_rows,
                    meta=meta,
                    rest_gap_hours=rest_gap_hours,
                    duplicate_visit_efc=duplicate_visit_efc,
                )
                records.extend(visits)
                cell_audits.append(cell_audit)
    except (BadZipFile, KeyError, OSError) as exc:
        raise SNLCycleDataError(
            f"Cannot extract SNL RPT trajectories: {zip_path}"
        ) from exc
    trajectories = pd.DataFrame(records, columns=RPT_TRAJECTORY_COLUMNS).sort_values(
        ["condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )
    audit = {
        "schema_version": "lifetwin.snl_rpt_extraction_audit.v1",
        "dataset_id": DATASET_ID,
        "extraction_rule": {
            "capacity_check_protocol": "0.5C_full_depth_2.0V_to_3.6V",
            "rest_gap_hours": rest_gap_hours,
            "duplicate_visit_efc": duplicate_visit_efc,
            "standalone_capacity_bounds_fraction_of_nominal": [0.6, 1.3],
            "exact_duplicate_rows_removed": True,
        },
        "physical_cell_count": int(trajectories["cell_id"].nunique()),
        "condition_cluster_count": int(trajectories["condition_id"].nunique()),
        "trajectory_row_count": len(trajectories),
        "minimum_rpt_visit_count": int(trajectories.groupby("cell_id").size().min()),
        "median_rpt_visit_count": float(
            trajectories.groupby("cell_id").size().median()
        ),
        "maximum_rpt_visit_count": int(trajectories.groupby("cell_id").size().max()),
        "minimum_maximum_elapsed_days": float(
            trajectories.groupby("cell_id")["elapsed_days"].max().min()
        ),
        "minimum_maximum_equivalent_full_cycles": float(
            trajectories.groupby("cell_id")["equivalent_full_cycles"].max().min()
        ),
        "canonical_rpt_trajectory_sha256": canonical_frame_sha256(
            trajectories, RPT_TRAJECTORY_COLUMNS
        ),
        "cell_audits": sorted(cell_audits, key=lambda value: str(value["cell_id"])),
    }
    return trajectories, audit


def extract_snl_rpt_repeat_measurements(
    zip_path: str | Path,
    metadata: pd.DataFrame,
    *,
    rest_gap_hours: float = 1.0,
    duplicate_visit_efc: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract within-visit RPT repeats without publishing raw cycle summaries."""
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise SNLCycleDataError("Metadata frame columns changed")
    if rest_gap_hours <= 0.0 or duplicate_visit_efc < 0.0:
        raise SNLCycleDataError("RPT extraction thresholds must be non-negative")
    records: list[dict[str, object]] = []
    cell_audits: list[dict[str, object]] = []
    try:
        with ZipFile(zip_path) as archive:
            for meta in metadata.to_dict(orient="records"):
                member_name = str(meta["cycle_member"])
                with TextIOWrapper(
                    archive.open(member_name), encoding="utf-8-sig", newline=""
                ) as stream:
                    reader = csv.DictReader(stream)
                    if tuple(reader.fieldnames or ()) != CYCLE_DATA_HEADERS:
                        raise SNLCycleDataError(
                            f"Cycle-summary headers changed: {member_name}"
                        )
                    cell_rows = list(reader)
                repeats, cell_audit = _rpt_repeats_for_cell(
                    cell_rows,
                    meta=meta,
                    rest_gap_hours=rest_gap_hours,
                    duplicate_visit_efc=duplicate_visit_efc,
                )
                records.extend(repeats)
                cell_audits.append(cell_audit)
    except (BadZipFile, KeyError, OSError) as exc:
        raise SNLCycleDataError(
            f"Cannot extract SNL RPT repeat measurements: {zip_path}"
        ) from exc
    repeats = pd.DataFrame(records, columns=RPT_REPEAT_COLUMNS).sort_values(
        ["condition_id", "cell_id", "visit_index", "repeat_index"],
        kind="stable",
        ignore_index=True,
    )
    audit = {
        "schema_version": "lifetwin.snl_rpt_repeat_extraction_audit.v1",
        "dataset_id": DATASET_ID,
        "evidence_role": "private_post_outcome_repeatability_development",
        "physical_cell_count": int(repeats["cell_id"].nunique()),
        "condition_cluster_count": int(repeats["condition_id"].nunique()),
        "visit_count": int(
            repeats[["cell_id", "visit_index"]].drop_duplicates().shape[0]
        ),
        "repeat_measurement_count": len(repeats),
        "minimum_repeats_per_visit": int(
            repeats.groupby(["cell_id", "visit_index"]).size().min()
        ),
        "canonical_repeat_table_sha256": canonical_frame_sha256(
            repeats, RPT_REPEAT_COLUMNS
        ),
        "raw_or_row_level_release_permitted": False,
        "cell_audits": sorted(cell_audits, key=lambda value: str(value["cell_id"])),
    }
    return repeats, audit


__all__ = [
    "CANONICAL_CYCLE_COLUMNS",
    "CYCLE_DATA_HEADERS",
    "DATASET_ID",
    "METADATA_COLUMNS",
    "RPT_REPEAT_COLUMNS",
    "RPT_TRAJECTORY_COLUMNS",
    "SNLCycleDataError",
    "SOURCE_TRAINING_COLUMNS",
    "TARGET_PREFIX_COLUMNS",
    "TRUTH_COLUMNS",
    "audit_snl_archive_structure",
    "extract_snl_rpt_trajectories",
    "extract_snl_rpt_repeat_measurements",
    "load_snl_metadata",
    "prepare_snl_cycle_inputs",
    "prepare_source_training_capacity",
]
