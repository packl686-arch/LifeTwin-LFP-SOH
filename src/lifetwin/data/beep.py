from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any

import pandas as pd

from lifetwin.data.schema import validate_cycle_summary


_CYCLES_MARKER = b'"cycles_interpolated"'
_VERSION_PATTERN = re.compile(
    rb'"@version"\s*:\s*("(?:\\.|[^"\\])*")'
)
_PROTOCOL_PATTERN = re.compile(
    r"\d{8}-(?P<first>\d+(?:_\d+)?)C[-_]"
    r"(?P<percent>\d+)per[-_]"
    r"(?P<second>\d+(?:[._]\d+)?)C",
    re.IGNORECASE,
)
_SOURCE_BATCHES = {
    "2017-05-12_tests": "MATR_RAW_2017_05_12",
    "2017-06-30_tests": "MATR_RAW_2017_06_30",
    "2018-04-12_batch8": "MATR_RAW_2018_04_12",
}
_SUMMARY_COLUMNS = {
    "discharge_capacity": "discharge_capacity_ah",
    "charge_capacity": "charge_capacity_ah",
    "discharge_energy": "discharge_energy_wh",
    "charge_energy": "charge_energy_wh",
    "dc_internal_resistance": "internal_resistance_ohm",
    "temperature_maximum": "temperature_max_c",
    "temperature_average": "temperature_avg_c",
    "temperature_minimum": "temperature_min_c",
    "date_time_iso": "date_time_iso",
    "energy_efficiency": "energy_efficiency",
    "charge_throughput": "charge_throughput_ah",
    "energy_throughput": "energy_throughput_wh",
    "charge_duration": "charge_time_s",
    "time_temperature_integrated": "time_temperature_integrated",
    "paused": "paused",
}


@dataclass(frozen=True)
class BeepSummaryRecord:
    source_path: Path
    source_size_bytes: int
    source_sha256: str | None
    barcode: str
    protocol_raw: str
    source_domain: str
    batch_id: str
    protocol_id: str
    channel_id: int
    beep_version: str
    summary: dict[str, list[Any]]

    @property
    def source_file(self) -> str:
        return self.source_path.name

    @property
    def summary_rows(self) -> int:
        return len(self.summary["cycle_index"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_rate(value: str) -> str:
    number = float(value.replace("_", "."))
    return f"{number:g}"


def normalize_fastcharge_protocol(protocol: str) -> tuple[str, str, str]:
    """Return source domain, raw batch id, and readable charge policy."""
    parts = re.split(r"[\\/]", protocol)
    if len(parts) < 2:
        raise ValueError(f"Unexpected FastCharge protocol path: {protocol}")
    source_domain = parts[0]
    batch_id = _SOURCE_BATCHES.get(source_domain.casefold())
    if batch_id is None:
        raise ValueError(f"Unknown FastCharge source domain: {source_domain}")
    match = _PROTOCOL_PATTERN.search(parts[-1])
    if match is None:
        raise ValueError(f"Cannot parse FastCharge charge policy: {protocol}")
    policy = (
        f"{_format_rate(match.group('first'))}C"
        f"({int(match.group('percent'))}%)-"
        f"{_format_rate(match.group('second'))}C"
    )
    if "newstructure" in parts[-1].casefold():
        policy += "-newstructure"
    return source_domain, batch_id, policy


def _read_summary_prefix(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
    max_prefix_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Parse only the BEEP header and summary before the large curve payload."""
    buffer = bytearray()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(
                    f"Missing cycles_interpolated boundary in BEEP JSON: {path}"
                )
            buffer.extend(chunk)
            marker_index = buffer.find(_CYCLES_MARKER)
            if marker_index >= 0:
                prefix = bytes(buffer[:marker_index]).rstrip()
                if not prefix.endswith(b","):
                    raise ValueError(
                        f"Unexpected BEEP top-level field boundary in {path}"
                    )
                return json.loads(prefix[:-1] + b"}")
            if len(buffer) > max_prefix_bytes:
                raise ValueError(
                    f"BEEP summary prefix exceeds {max_prefix_bytes} bytes: {path}"
                )


def _read_tail_version(path: Path, *, max_tail_bytes: int = 64 * 1024) -> str:
    """Read the trailing BEEP version without loading the curve payload."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - max_tail_bytes))
        tail = handle.read()
    matches = list(_VERSION_PATTERN.finditer(tail))
    if not matches:
        return "unknown"
    return str(json.loads(matches[-1].group(1)))


def read_beep_summary(
    path: str | Path,
    *,
    hash_source: bool = True,
) -> BeepSummaryRecord:
    source_path = Path(path)
    value = _read_summary_prefix(source_path)
    if value.get("@module") != "beep.structure":
        raise ValueError(f"Unexpected BEEP module in {source_path}")
    if value.get("@class") != "ProcessedCyclerRun":
        raise ValueError(f"Unexpected BEEP class in {source_path}")

    barcode = str(value.get("barcode") or "").strip().upper()
    protocol_raw = str(value.get("protocol") or "").strip()
    if not barcode or not protocol_raw:
        raise ValueError(f"BEEP barcode and protocol are required: {source_path}")
    source_domain, batch_id, protocol_id = normalize_fastcharge_protocol(protocol_raw)

    raw_summary = value.get("summary")
    if not isinstance(raw_summary, dict):
        raise ValueError(f"BEEP summary must be an object: {source_path}")
    required = {"cycle_index", *_SUMMARY_COLUMNS}
    missing = sorted(required - set(raw_summary))
    if missing:
        raise ValueError(f"Missing BEEP summary fields {missing}: {source_path}")
    summary: dict[str, list[Any]] = {}
    for key, item in raw_summary.items():
        if not isinstance(item, list):
            raise ValueError(f"BEEP summary field {key} must be a list: {source_path}")
        summary[str(key)] = item
    lengths = {len(item) for item in summary.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"BEEP summary arrays must have one nonzero length: {source_path}")

    try:
        channel_id = int(value["channel_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid BEEP channel_id: {source_path}") from exc
    return BeepSummaryRecord(
        source_path=source_path,
        source_size_bytes=source_path.stat().st_size,
        source_sha256=_sha256_file(source_path) if hash_source else None,
        barcode=barcode,
        protocol_raw=protocol_raw,
        source_domain=source_domain,
        batch_id=batch_id,
        protocol_id=protocol_id,
        channel_id=channel_id,
        beep_version=_read_tail_version(source_path),
        summary=summary,
    )


def _segment_sort_key(record: BeepSummaryRecord) -> tuple[str, str]:
    timestamps = record.summary["date_time_iso"]
    first_timestamp = str(timestamps[0] or "") if timestamps else ""
    return first_timestamp, record.source_file


def _records_to_cycle_frame(records: list[BeepSummaryRecord]) -> pd.DataFrame:
    if not records:
        raise ValueError("At least one BEEP segment is required")
    records = sorted(records, key=_segment_sort_key)
    identity = {(record.batch_id, record.protocol_id) for record in records}
    if len(identity) != 1:
        raise ValueError(
            f"BEEP segments for barcode {records[0].barcode} disagree on identity"
        )

    frames: list[pd.DataFrame] = []
    cycle_offset = 0
    for segment_index, record in enumerate(records, start=1):
        raw_index = pd.to_numeric(record.summary["cycle_index"], errors="raise")
        integer_index = raw_index.astype("int64")
        if not (raw_index == integer_index).all() or integer_index.tolist() != list(
            range(len(integer_index))
        ):
            raise ValueError(f"Invalid raw cycle index in {record.source_path}")
        frame = pd.DataFrame(
            {
                "dataset_id": "MATR_FASTCHARGE_RAW_JSON",
                "cell_id": f"MATR_BARCODE_{record.barcode}",
                "source_barcode": record.barcode,
                "batch_id": record.batch_id,
                "protocol_id": record.protocol_id,
                "cycle_index": integer_index + cycle_offset + 1,
                "segment_index": segment_index,
                "segment_count": len(records),
                "segment_cycle_index": integer_index + 1,
                "raw_cycle_index": integer_index,
                "source_file": record.source_file,
                "channel_id": record.channel_id,
            }
        )
        for source_column, target_column in _SUMMARY_COLUMNS.items():
            frame[target_column] = record.summary[source_column]
        frames.append(frame)
        cycle_offset += len(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_fastcharge_frames(
    source_directory: str | Path,
    *,
    hash_sources: bool = True,
    observation_cycle: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build canonical cycle rows, a source inventory, and a provenance audit."""
    if observation_cycle < 2:
        raise ValueError("observation_cycle must be at least 2")
    source_root = Path(source_directory)
    paths = sorted(source_root.glob("*_structure.json"))
    if not paths:
        raise FileNotFoundError(f"No *_structure.json files under {source_root}")

    records: list[BeepSummaryRecord] = []
    records_by_barcode: dict[str, list[BeepSummaryRecord]] = defaultdict(list)
    files_by_barcode: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        record = read_beep_summary(path, hash_source=hash_sources)
        records.append(record)
        records_by_barcode[record.barcode].append(record)
        files_by_barcode[record.barcode].append(record.source_file)

    segment_order = {
        record.source_file: index
        for barcode_records in records_by_barcode.values()
        for index, record in enumerate(
            sorted(barcode_records, key=_segment_sort_key), start=1
        )
    }
    cell_summary_rows = {
        barcode: sum(record.summary_rows for record in barcode_records)
        for barcode, barcode_records in records_by_barcode.items()
    }
    inventory_rows: list[dict[str, Any]] = []
    for record in records:
        raw_index = record.summary["cycle_index"]
        inventory_rows.append(
            {
                "source_file": record.source_file,
                "source_size_bytes": record.source_size_bytes,
                "source_sha256": record.source_sha256,
                "barcode": record.barcode,
                "cell_id": f"MATR_BARCODE_{record.barcode}",
                "protocol_raw": record.protocol_raw,
                "source_domain": record.source_domain,
                "batch_id": record.batch_id,
                "protocol_id": record.protocol_id,
                "channel_id": record.channel_id,
                "beep_version": record.beep_version,
                "summary_rows": record.summary_rows,
                "first_raw_cycle_index": int(raw_index[0]),
                "last_raw_cycle_index": int(raw_index[-1]),
                "segment_count": len(files_by_barcode[record.barcode]),
                "segment_index": segment_order[record.source_file],
                "cell_summary_rows": cell_summary_rows[record.barcode],
                "included_for_barcode": True,
                "usable_at_observation_cycle": (
                    cell_summary_rows[record.barcode] >= observation_cycle
                ),
            }
        )
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["source_domain", "source_file"], kind="stable"
    )

    cell_groups = sorted(
        records_by_barcode.values(),
        key=lambda group: (
            group[0].batch_id,
            group[0].protocol_id,
            group[0].barcode,
        ),
    )
    cycles = pd.concat(
        [_records_to_cycle_frame(group) for group in cell_groups],
        ignore_index=True,
    )
    validation = validate_cycle_summary(cycles)

    multi_segment_barcodes = {
        barcode: {
            "files_in_chronological_order": [
                record.source_file
                for record in sorted(records_by_barcode[barcode], key=_segment_sort_key)
            ],
            "segment_rows": [
                record.summary_rows
                for record in sorted(records_by_barcode[barcode], key=_segment_sort_key)
            ],
            "merged_rows": cell_summary_rows[barcode],
        }
        for barcode, files in sorted(files_by_barcode.items())
        if len(files) > 1
    }
    manifest_items = [
        {
            "file": record.source_file,
            "bytes": record.source_size_bytes,
            "sha256": record.source_sha256,
        }
        for record in sorted(records, key=lambda item: item.source_file)
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest_items,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cell_lengths = list(cell_summary_rows.values())
    short_records = [
        {
            "barcode": barcode,
            "source_files": sorted(files_by_barcode[barcode]),
            "summary_rows": rows,
        }
        for barcode, rows in sorted(cell_summary_rows.items())
        if rows < observation_cycle
    ]
    audit: dict[str, Any] = {
        "warning": (
            "Public MATR fast-charge cell data only. This ingest does not establish "
            "stationary-storage, calendar-aging, Hithium-product, or 15-25 year accuracy."
        ),
        "source_directory": str(source_root.resolve()),
        "source_file_pattern": "*_structure.json",
        "source_file_count": len(records),
        "source_total_bytes": sum(record.source_size_bytes for record in records),
        "source_hashes_complete": hash_sources,
        "source_manifest_sha256": manifest_sha256,
        "beep_class": "beep.structure.ProcessedCyclerRun",
        "beep_version_counts": dict(
            sorted(Counter(record.beep_version for record in records).items())
        ),
        "unique_barcode_count": len(cell_groups),
        "multi_segment_barcode_count": len(multi_segment_barcodes),
        "additional_segment_file_count": len(records) - len(cell_groups),
        "segment_merge_policy": (
            "Preserve every segment and concatenate by first ISO timestamp; "
            "source cycle indices restart within each segment."
        ),
        "multi_segment_barcodes": multi_segment_barcodes,
        "source_domain_file_counts": dict(
            sorted(Counter(record.source_domain for record in records).items())
        ),
        "physical_batch_cell_counts": dict(
            sorted(Counter(group[0].batch_id for group in cell_groups).items())
        ),
        "physical_protocol_count": len(
            {group[0].protocol_id for group in cell_groups}
        ),
        "summary_rows": {
            "minimum": min(cell_lengths),
            "median": float(median(cell_lengths)),
            "maximum": max(cell_lengths),
        },
        "observation_cycle": observation_cycle,
        "usable_cell_count": len(cell_groups) - len(short_records),
        "short_cell_records": short_records,
        "canonical_cycle_validation": {
            "row_count": validation.row_count,
            "cell_count": validation.cell_count,
            "warnings": list(validation.warnings),
        },
        "paper_cell_mapping_gate": {
            "status": "blocked",
            "reason": (
                "The BEEP JSON exposes barcodes, while the public paper cohort uses "
                "b1c/b2c/b3c identifiers without an authoritative barcode crosswalk."
            ),
            "prohibited_action": (
                "Do not infer confirmatory labels by outcome-ranked matching."
            ),
        },
        "industrial_evidence": False,
    }
    return cycles, inventory.reset_index(drop=True), audit
