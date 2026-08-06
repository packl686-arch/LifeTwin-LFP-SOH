from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

from lifetwin.data.beep_identity import BeepIdentityRecord, read_beep_identity


EXPECTED_TOTAL_FILES = 261
EXPECTED_LOCK_FILES = 3
EXPECTED_VALID_FILES = 258
EXPECTED_TOTAL_BYTES = 33_038_472_637
SNL_POLICY = {
    "main_model_access": "metadata_inventory_only",
    "outcome_access": "forbidden",
    "training_allowed": False,
    "model_selection_allowed": False,
    "reserved_external_holdout": True,
    "physical_cell_count": 30,
    "local_csv_count": 60,
    "metadata_catalog_record_count": 86,
    "locally_available_lfp_record_count": 30,
}
NASA_CHEMISTRY = "unspecified_li_ion_not_lfp_evidence"
NASA_ROLE = "cross_domain_cycle_trajectory_and_rul_stress"
NASA_DUPLICATE_IDS = ("B0025", "B0026", "B0027", "B0028")
_BATTERY_ID = re.compile(r"(?i)(?:^|[^A-Z0-9])(B\d{4})(?=[^A-Z0-9]|$)")
_RANDOMIZED_ID = re.compile(r"(?i)(?:^|[^A-Z0-9])(RW\d+)(?=[^A-Z0-9]|$)")


class DataAssetIntakeError(ValueError):
    pass


@dataclass(frozen=True)
class FileMetadata:
    relative_path: str
    name: str
    extension: str
    size_bytes: int
    modified_utc: str
    is_lock_file: bool


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_file_metadata(root: str | Path) -> list[FileMetadata]:
    """Collect directory metadata only; this function never opens file content."""
    source_root = Path(root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Data asset root is missing: {source_root}")
    rows: list[FileMetadata] = []
    for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(
            FileMetadata(
                relative_path=_relative(path, source_root),
                name=path.name,
                extension=path.suffix.casefold(),
                size_bytes=int(stat.st_size),
                modified_utc=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                )
                .isoformat()
                .replace("+00:00", "Z"),
                is_lock_file=path.name.startswith("~$"),
            )
        )
    return rows


def snapshot_summary(files: Iterable[FileMetadata]) -> dict[str, object]:
    rows = list(files)
    valid = [row for row in rows if not row.is_lock_file]
    modified = sorted(row.modified_utc for row in rows)
    return {
        "total_file_count": len(rows),
        "lock_file_count": len(rows) - len(valid),
        "valid_file_count": len(valid),
        "total_bytes": sum(row.size_bytes for row in rows),
        "earliest_modified_utc": modified[0] if modified else None,
        "latest_modified_utc": modified[-1] if modified else None,
    }


def validate_expected_inventory(files: Iterable[FileMetadata]) -> dict[str, object]:
    rows = list(files)
    summary = snapshot_summary(rows)
    valid = [row for row in rows if not row.is_lock_file]
    expected = {
        "total_file_count": EXPECTED_TOTAL_FILES,
        "lock_file_count": EXPECTED_LOCK_FILES,
        "valid_file_count": EXPECTED_VALID_FILES,
        "total_bytes": EXPECTED_TOTAL_BYTES,
        "fastcharge_json_count": 140,
        "root_matr_mat_count": 5,
        "snl_csv_count": 60,
        "nasa_battery_zip_count": 6,
        "nasa_randomized_zip_count": 7,
        "calce_xlsx_count": 34,
        "calce_xls_count": 2,
        "calce_lock_count": 3,
        "oxford_file_count": 3,
    }
    observed = {
        **summary,
        "fastcharge_json_count": sum(
            row.relative_path.startswith("FastCharge/") and row.extension == ".json"
            for row in valid
        ),
        "root_matr_mat_count": sum(
            "/" not in row.relative_path and row.extension == ".mat" for row in valid
        ),
        "snl_csv_count": sum(
            row.relative_path.startswith("SNL LFP/SNL LFP/")
            and row.extension == ".csv"
            for row in valid
        ),
        "nasa_battery_zip_count": sum(
            row.relative_path.startswith("5. Battery Data Set/")
            and row.extension == ".zip"
            for row in valid
        ),
        "nasa_randomized_zip_count": sum(
            row.relative_path.startswith("11. Randomized Battery Usage Data Set/")
            and row.extension == ".zip"
            for row in valid
        ),
        "calce_xlsx_count": sum(
            row.relative_path.startswith("CALCE_A123/")
            and row.extension == ".xlsx"
            for row in valid
        ),
        "calce_xls_count": sum(
            row.relative_path.startswith("CALCE_A123/") and row.extension == ".xls"
            for row in valid
        ),
        "calce_lock_count": sum(
            row.relative_path.startswith("CALCE_A123/") and row.is_lock_file
            for row in rows
        ),
        "oxford_file_count": sum(
            row.relative_path.startswith("Oxford/") for row in valid
        ),
    }
    differences = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if differences:
        raise DataAssetIntakeError(f"Inventory differs from freeze: {differences}")
    return {"status": "passed", "expected": expected, "observed": observed}


def audit_snl_metadata(files: Iterable[FileMetadata]) -> dict[str, object]:
    """Pair SNL filenames using supplied metadata; no source path is opened."""
    rows = [
        row
        for row in files
        if row.relative_path.startswith("SNL LFP/SNL LFP/")
        and row.extension == ".csv"
        and not row.is_lock_file
    ]
    pairs: dict[str, set[str]] = defaultdict(set)
    unknown: list[str] = []
    for row in rows:
        match = re.fullmatch(r"(.+)_(cycle_data|timeseries)\.csv", row.name)
        if match is None:
            unknown.append(row.name)
            continue
        pairs[match.group(1)].add(match.group(2))
    incomplete = {
        key: sorted(kinds)
        for key, kinds in pairs.items()
        if kinds != {"cycle_data", "timeseries"}
    }
    if len(rows) != 60 or len(pairs) != 30 or incomplete or unknown:
        raise DataAssetIntakeError("SNL metadata does not form 30 complete pairs")
    return {
        "status": "passed_metadata_only",
        "content_open_count": 0,
        "content_hash_count": 0,
        "cycle_data_file_count": 30,
        "timeseries_file_count": 30,
        "pair_count": 30,
        "incomplete_pair_count": 0,
        "policy": dict(SNL_POLICY),
    }


def audit_matr_json(source_directory: str | Path) -> dict[str, object]:
    paths = sorted(Path(source_directory).glob("*_structure.json"))
    records = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            records.append(read_beep_identity(path))
        except (OSError, TypeError, ValueError) as exc:
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    grouped: dict[str, list[BeepIdentityRecord]] = defaultdict(list)
    for record in records:
        grouped[record.barcode].append(record)
    conflicts = {
        barcode: sorted(
            {f"{record.batch_id}|{record.protocol_id}" for record in group}
        )
        for barcode, group in grouped.items()
        if len({(record.batch_id, record.protocol_id) for record in group}) != 1
    }
    detail = {
        barcode: {
            "segment_count": len(group),
            "batch_id": group[0].batch_id,
            "protocol_id": group[0].protocol_id,
            "channel_ids": sorted({record.channel_id for record in group}),
            "source_files": sorted(record.source_filename for record in group),
        }
        for barcode, group in sorted(grouped.items())
    }
    return {
        "status": (
            "blocked" if errors else "failed" if conflicts else "passed"
        ),
        "json_file_count": len(paths),
        "parsed_json_file_count": len(records),
        "unique_barcode_count": len(grouped),
        "multi_segment_barcode_count": sum(len(group) > 1 for group in grouped.values()),
        "additional_segment_file_count": len(records) - len(grouped),
        "batch_distribution_physical_cells": dict(
            sorted(Counter(group[0].batch_id for group in grouped.values()).items())
        ),
        "batch_distribution_files": dict(
            sorted(Counter(record.batch_id for record in records).items())
        ),
        "protocol_distribution_physical_cells": dict(
            sorted(Counter(group[0].protocol_id for group in grouped.values()).items())
        ),
        "protocol_count": len({group[0].protocol_id for group in grouped.values()}),
        "summary_row_count": "not_read",
        "parse_error_count": len(errors),
        "identity_conflict_count": len(conflicts),
        "parse_errors": errors,
        "identity_conflicts": conflicts,
        "hash_source": False,
        "summary_materialized": False,
        "cycles_interpolated_materialized": False,
        "outcome_value_materialized_count": 0,
        "read_beep_summary_call_count": 0,
        "identity_field_whitelist": [
            "barcode",
            "channel_id",
            "source_domain",
            "batch_id",
            "protocol_raw",
            "protocol_id",
            "beep_version",
            "source_filename",
            "source_size_bytes",
        ],
        "physical_cell_key": "barcode",
        "barcode_detail": detail,
    }


def audit_matr_mat_representations(root: str | Path) -> dict[str, object]:
    source_root = Path(root)
    rows = []
    for path in sorted(source_root.glob("*.mat")):
        with path.open("rb") as stream:
            header = stream.read(128)
        text = header.decode("latin-1", errors="replace").rstrip("\x00")
        rows.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "format": (
                    "matlab_7_3_hdf5" if "MATLAB 7.3 MAT-file" in text else "matlab_5"
                ),
                "header_descriptor": text[:116].strip(),
                "mapping_status": "dataset_level_only_not_counted_as_new_cells",
            }
        )
    if len(rows) != 5:
        raise DataAssetIntakeError("Expected five root MATR MAT representations")
    return {
        "status": "passed_no_duplicate_counting",
        "mat_file_count": len(rows),
        "physical_cells_added_by_mat": 0,
        "per_cell_mapping_status": "ambiguous_without_value_based_linkage_excluded",
        "future_outcomes_used_for_mapping": False,
        "representations": rows,
    }


def _zip_identity_rows(
    directory: Path,
    pattern: re.Pattern[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for zip_path in sorted(directory.glob("*.zip")):
        with ZipFile(zip_path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                matches = sorted({value.upper() for value in pattern.findall(entry.filename)})
                for physical_id in matches:
                    rows.append(
                        {
                            "source_zip": zip_path.name,
                            "archive_entry": entry.filename,
                            "physical_battery_id": physical_id,
                            "uncompressed_size": entry.file_size,
                            "crc32": f"{entry.CRC:08x}",
                        }
                    )
    return rows


def audit_nasa_battery_zip_metadata(directory: str | Path) -> dict[str, object]:
    rows = _zip_identity_rows(Path(directory), _BATTERY_ID)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["physical_battery_id"])].append(row)
    duplicates: dict[str, dict[str, object]] = {}
    for physical_id, group in sorted(grouped.items()):
        if len(group) <= 1:
            continue
        identities = {(row["uncompressed_size"], row["crc32"]) for row in group}
        duplicates[physical_id] = {
            "source_count": len(group),
            "central_directory_identity_count": len(identities),
            "resolution": (
                "same_uncompressed_size_and_crc32_canonicalize_one_copy"
                if len(identities) == 1
                else "conflict_stop_automatic_intake"
            ),
        }
    observed_duplicates = tuple(sorted(duplicates))
    if len(grouped) != 34 or observed_duplicates != NASA_DUPLICATE_IDS:
        raise DataAssetIntakeError("NASA battery identities differ from frozen inventory")
    if any(item["central_directory_identity_count"] != 1 for item in duplicates.values()):
        raise DataAssetIntakeError("NASA duplicate battery entries conflict")
    return {
        "status": "passed_central_directory_only",
        "zip_file_count": len({row["source_zip"] for row in rows}),
        "archive_battery_entry_count": len(rows),
        "unique_physical_battery_count": len(grouped),
        "duplicate_physical_battery_ids": duplicates,
        "chemistry": NASA_CHEMISTRY,
        "task_role": NASA_ROLE,
        "entry_content_read": False,
        "identity_rows": rows,
    }


def audit_nasa_randomized_zip_metadata(directory: str | Path) -> dict[str, object]:
    rows = _zip_identity_rows(Path(directory), _RANDOMIZED_ID)
    return {
        "status": "inventory_only_physical_identity_not_verified",
        "zip_file_count": len({row["source_zip"] for row in rows}),
        "internal_candidate_id_count": len(
            {row["physical_battery_id"] for row in rows}
        ),
        "training_allowed": False,
        "scoring_allowed": False,
        "entry_content_read": False,
        "identity_rows": rows,
    }


def audit_calce_metadata(files: Iterable[FileMetadata]) -> dict[str, object]:
    rows = [
        row
        for row in files
        if row.relative_path.startswith("CALCE_A123/") and not row.is_lock_file
    ]
    ocv_groups = {
        row.relative_path.split("/")[2]
        for row in rows
        if row.relative_path.startswith("CALCE_A123/OCV/")
    }
    dynamic_groups = {
        row.relative_path.split("/")[2]
        for row in rows
        if row.relative_path.startswith("CALCE_A123/Dynamic_Profile/")
    }
    candidate_cells = sorted(
        {
            match.group(1)
            for row in rows
            if (match := re.match(r"(A1-\d{3})-", row.name)) is not None
        }
    )
    if len(rows) != 36 or len(ocv_groups) != 8 or len(dynamic_groups) != 8:
        raise DataAssetIntakeError("CALCE workbook grouping differs from expectation")
    return {
        "status": "passed_metadata_only",
        "valid_workbook_count": len(rows),
        "xlsx_count": sum(row.extension == ".xlsx" for row in rows),
        "xls_count": sum(row.extension == ".xls" for row in rows),
        "ocv_temperature_group_count": len(ocv_groups),
        "dynamic_profile_temperature_group_count": len(dynamic_groups),
        "filename_level_physical_cell_ids": candidate_cells,
        "workbooks_are_physical_cells": False,
        "task_role": "feature_and_input_validation",
        "lifetime_training_allowed": False,
    }


def audit_oxford_metadata(files: Iterable[FileMetadata]) -> dict[str, object]:
    rows = [
        row
        for row in files
        if row.relative_path.startswith("Oxford/") and not row.is_lock_file
    ]
    if len(rows) != 3:
        raise DataAssetIntakeError("Oxford file count differs from expectation")
    return {
        "status": "passed_source_record_and_file_metadata",
        "observed_file_count": len(rows),
        "verified_physical_cell_count": 8,
        "example_cycle_is_additional_cell": False,
        "chemistry": "unspecified_li_ion_not_confirmed_lfp",
        "task_role": "cross_chemistry_degradation_and_rejection_stress",
        "formal_scoring_allowed_this_intake": False,
    }


def file_metadata_records(files: Iterable[FileMetadata]) -> list[dict[str, object]]:
    return [asdict(row) for row in files]
