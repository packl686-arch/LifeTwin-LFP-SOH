from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from scipy.io import whosmat


EXPECTED_MAT_FILE_COUNT = 38
EXPECTED_TEXT_FILE_COUNT = 10
EXPECTED_PHYSICAL_BATTERY_COUNT = 34
EXPECTED_DUPLICATE_IDS = ("B0025", "B0026", "B0027", "B0028")
EXPOSURE_CLASSIFICATION = (
    "development_only_outcomes_and_protocol_structure_exposed"
)
EVIDENCE_ROLE = "development_only"
_BATTERY_FILENAME = re.compile(r"(?i)^(B\d{4})\.mat$")


class NasaExtractedMetadataError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            _relative(path, root),
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    )


def _matlab_format(path: Path) -> str:
    with path.open("rb") as stream:
        header = stream.read(128)
    if b"MATLAB 7.3 MAT-file" in header:
        return "matlab_7_3_hdf5"
    if b"MATLAB 5.0 MAT-file" in header:
        return "matlab_5"
    if header[:4] == b"\x00\x00\x00\x00":
        return "matlab_4_or_unknown"
    return "unknown_matlab_format"


def _schema_inventory(path: Path) -> dict[str, object]:
    matlab_format = _matlab_format(path)
    try:
        variables = whosmat(path)
    except (OSError, TypeError, ValueError, NotImplementedError) as exc:
        return {
            "status": "blocked_schema_unreadable",
            "matlab_format": matlab_format,
            "top_level_variable_names": [],
            "top_level_variable_types": [],
            "top_level_variable_shapes": [],
            "schema_error_type": type(exc).__name__,
        }
    return {
        "status": "passed_schema_only",
        "matlab_format": matlab_format,
        "top_level_variable_names": [str(name) for name, _, _ in variables],
        "top_level_variable_types": [str(kind) for _, _, kind in variables],
        "top_level_variable_shapes": [list(shape) for _, shape, _ in variables],
        "schema_error_type": None,
    }


def _line_numbers(text: str, pattern: re.Pattern[str]) -> list[int]:
    return [
        index
        for index, line in enumerate(text.splitlines(), start=1)
        if pattern.search(line)
    ]


def _readme_exposure(path: Path, root: Path, sha256: str) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    patterns = {
        "eol_threshold": re.compile(
            r"(?i)\b(?:eol|end[- ]of[- ]life)\b|(?:20|30)\s*%"
        ),
        "capacity_stop_condition": re.compile(
            r"(?i)\bcapacity\b.*\b(?:stop|terminate|threshold|fade)\b|"
            r"\b(?:1\.4|1\.6)\s*a\s*h\b"
        ),
        "anomaly_or_software_crash": re.compile(
            r"(?i)\b(?:crash|software|abnormal|anomal|low capacity)\w*\b"
        ),
        "license_text": re.compile(
            r"(?i)\b(?:licen[cs]e|copyright|terms of use|permission)\b"
        ),
        "citation_or_acknowledgement": re.compile(
            r"(?i)\b(?:cit(?:e|ation)|acknowledg\w*|reference|publication)\b"
        ),
    }
    locations = {name: _line_numbers(text, pattern) for name, pattern in patterns.items()}
    return {
        "relative_path": _relative(path, root),
        "sha256": sha256,
        "eol_threshold_appears": bool(locations["eol_threshold"]),
        "capacity_stop_condition_appears": bool(
            locations["capacity_stop_condition"]
        ),
        "experiment_anomaly_or_software_crash_appears": bool(
            locations["anomaly_or_software_crash"]
        ),
        "license_text_appears": bool(locations["license_text"]),
        "citation_or_acknowledgement_appears": bool(
            locations["citation_or_acknowledgement"]
        ),
        "matching_line_numbers": locations,
        "exposure_classification": EXPOSURE_CLASSIFICATION,
        "model_or_threshold_selection_allowed": False,
    }


def _rights_review() -> dict[str, object]:
    return {
        "official_access_level": "public",
        "official_intended_use_includes_prognostic_algorithm_development": True,
        "repository_requests_publication_acknowledgement": True,
        "dataset_specific_license_identifier": None,
        "nasa_open_data_license_status": "not_specified",
        "raw_redistribution_allowed": False,
        "formal_model_execution_allowed": False,
        "public_aggregate_result_release_confirmed": False,
        "blocking_reason": "dataset_specific_license_not_explicitly_resolved",
    }


def audit_nasa_extracted_metadata(source_directory: str | Path) -> dict[str, object]:
    """Audit extracted NASA files without loading any MATLAB value."""
    root = Path(source_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"NASA extracted directory is missing: {root}")
    before_snapshot = _file_snapshot(root)
    top_level_directories = sorted(path.name for path in root.iterdir() if path.is_dir())
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if any(path.suffix.casefold() == ".zip" for path in files):
        raise NasaExtractedMetadataError("Extracted NASA directory must contain no ZIP")
    mat_paths = [path for path in files if path.suffix.casefold() == ".mat"]
    text_paths = [
        path
        for path in files
        if path.suffix.casefold() == ".txt" or path.name.casefold().startswith("readme")
    ]
    if len(top_level_directories) != 6:
        raise NasaExtractedMetadataError("Expected six top-level directories")
    if len(mat_paths) != EXPECTED_MAT_FILE_COUNT:
        raise NasaExtractedMetadataError("Expected 38 extracted MAT files")
    if len(text_paths) != EXPECTED_TEXT_FILE_COUNT:
        raise NasaExtractedMetadataError("Expected 10 README/TXT files")
    if len(files) != len(mat_paths) + len(text_paths):
        raise NasaExtractedMetadataError("Unexpected extracted NASA file type")

    file_rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for path in files:
        relative_path = _relative(path, root)
        stat = path.stat()
        sha256 = _sha256_file(path)
        hashes[relative_path] = sha256
        file_rows.append(
            {
                "relative_path": relative_path,
                "byte_count": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z"),
                "extension": path.suffix.casefold(),
                "sha256": sha256,
            }
        )

    grouped: dict[str, list[Path]] = {}
    for path in mat_paths:
        match = _BATTERY_FILENAME.fullmatch(path.name)
        if match is None:
            raise NasaExtractedMetadataError(
                f"MAT filename has no physical Bxxxx identity: {path.name}"
            )
        grouped.setdefault(match.group(1).upper(), []).append(path)
    if len(grouped) != EXPECTED_PHYSICAL_BATTERY_COUNT:
        raise NasaExtractedMetadataError("Expected 34 unique physical batteries")
    duplicate_ids = tuple(sorted(key for key, group in grouped.items() if len(group) > 1))
    if duplicate_ids != EXPECTED_DUPLICATE_IDS:
        raise NasaExtractedMetadataError("Duplicate physical battery IDs changed")
    if any(len(group) not in {1, 2} for group in grouped.values()):
        raise NasaExtractedMetadataError("Physical battery representation count changed")

    identity_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    canonical_by_path: dict[str, tuple[str, str]] = {}
    for physical_id, group in sorted(grouped.items()):
        relative_paths = sorted(_relative(path, root) for path in group)
        canonical_path = relative_paths[0]
        if len(relative_paths) == 2:
            first, second = relative_paths
            first_size = next(row[1] for row in before_snapshot if row[0] == first)
            second_size = next(row[1] for row in before_snapshot if row[0] == second)
            identical = first_size == second_size and hashes[first] == hashes[second]
            duplicate_rows.append(
                {
                    "physical_battery_id": physical_id,
                    "canonical_relative_path": canonical_path,
                    "duplicate_relative_path": second,
                    "canonical_byte_count": first_size,
                    "duplicate_byte_count": second_size,
                    "canonical_sha256": hashes[first],
                    "duplicate_sha256": hashes[second],
                    "identity_matches": identical,
                    "status": (
                        "canonicalized_identical_representation"
                        if identical
                        else "blocked_duplicate_conflict"
                    ),
                }
            )
            if not identical:
                raise NasaExtractedMetadataError(
                    f"Duplicate representation conflict for {physical_id}"
                )
        for relative_path in relative_paths:
            role = (
                "canonical"
                if relative_path == canonical_path
                else "duplicate_representation"
            )
            canonical_by_path[relative_path] = (role, canonical_path)
            path = root / relative_path
            identity_rows.append(
                {
                    "physical_battery_id": physical_id,
                    "relative_path": relative_path,
                    "byte_count": int(path.stat().st_size),
                    "sha256": hashes[relative_path],
                    "canonical_or_duplicate": role,
                    "canonical_relative_path": canonical_path,
                }
            )

    schema_rows: list[dict[str, object]] = []
    for path in mat_paths:
        relative_path = _relative(path, root)
        physical_id = _BATTERY_FILENAME.fullmatch(path.name).group(1).upper()
        role, canonical_path = canonical_by_path[relative_path]
        schema_rows.append(
            {
                "physical_battery_id": physical_id,
                "relative_path": relative_path,
                "byte_count": int(path.stat().st_size),
                "sha256": hashes[relative_path],
                **_schema_inventory(path),
                "canonical_or_duplicate": role,
                "canonical_relative_path": canonical_path,
                "outcome_values_loaded": False,
            }
        )
    readme_rows = [
        _readme_exposure(path, root, hashes[_relative(path, root)])
        for path in text_paths
    ]
    after_snapshot = _file_snapshot(root)
    return {
        "status": "passed",
        "top_level_directory_count": len(top_level_directories),
        "mat_file_count": len(mat_paths),
        "txt_file_count": len(text_paths),
        "unique_physical_battery_count": len(grouped),
        "duplicate_physical_battery_id_count": len(duplicate_ids),
        "duplicate_physical_battery_ids": list(duplicate_ids),
        "schema_readable_count": sum(
            row["status"] == "passed_schema_only" for row in schema_rows
        ),
        "schema_blocked_count": sum(
            row["status"] == "blocked_schema_unreadable" for row in schema_rows
        ),
        "mat_value_load_count": 0,
        "capacity_value_read_count": 0,
        "nasa_prepare_count": 0,
        "nasa_prediction_count": 0,
        "nasa_score_count": 0,
        "snl_content_open_count": 0,
        "evidence_role": EVIDENCE_ROLE,
        "readme_exposure_classification": EXPOSURE_CLASSIFICATION,
        "source_writeback_detected": before_snapshot != after_snapshot,
        "files": file_rows,
        "physical_identity_rows": identity_rows,
        "duplicate_rows": duplicate_rows,
        "mat_schema_rows": schema_rows,
        "readme_exposure_rows": readme_rows,
        "rights_review": _rights_review(),
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _output_manifest(output_directory: Path) -> tuple[Path, str]:
    path = output_directory / "output_manifest.json"
    generated_utc = _utc_now()
    entries = [
        {
            "relative_path": item.relative_to(output_directory).as_posix(),
            "sha256": _sha256_file(item),
            "byte_count": int(item.stat().st_size),
            "generated_utc": generated_utc,
        }
        for item in sorted(output_directory.iterdir())
        if item.is_file() and item != path
    ]
    _write_json(
        path,
        {
            "schema_version": 1,
            "generated_utc": generated_utc,
            "entries": entries,
            "manifest_self_hash_excluded": True,
        },
    )
    return path, _sha256_file(path)


def write_nasa_extracted_metadata_intake(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    adversarial_results: dict[str, object],
    test_results: dict[str, object],
) -> dict[str, object]:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"NASA metadata output already exists: {output}")
    audit = audit_nasa_extracted_metadata(source_directory)
    if audit["source_writeback_detected"]:
        raise NasaExtractedMetadataError("Source writeback detected")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    protocol = {
        key: audit[key]
        for key in (
            "status",
            "top_level_directory_count",
            "mat_file_count",
            "txt_file_count",
            "unique_physical_battery_count",
            "duplicate_physical_battery_id_count",
            "duplicate_physical_battery_ids",
            "schema_readable_count",
            "schema_blocked_count",
            "mat_value_load_count",
            "capacity_value_read_count",
            "nasa_prepare_count",
            "nasa_prediction_count",
            "nasa_score_count",
            "snl_content_open_count",
            "evidence_role",
            "readme_exposure_classification",
            "source_writeback_detected",
        )
    }
    protocol["protocol_id"] = "nasa_extracted_metadata_intake_v1_20260806"
    _write_json(output / "protocol_freeze.json", protocol)
    _write_json(
        output / "extracted_file_inventory.json",
        {"status": "passed", "files": audit["files"]},
    )
    identity_fields = (
        "physical_battery_id",
        "relative_path",
        "byte_count",
        "sha256",
        "canonical_or_duplicate",
        "canonical_relative_path",
    )
    _write_csv(
        output / "physical_battery_identity.csv",
        identity_fields,
        audit["physical_identity_rows"],
    )
    duplicate_fields = (
        "physical_battery_id",
        "canonical_relative_path",
        "duplicate_relative_path",
        "canonical_byte_count",
        "duplicate_byte_count",
        "canonical_sha256",
        "duplicate_sha256",
        "identity_matches",
        "status",
    )
    _write_csv(
        output / "duplicate_representation_manifest.csv",
        duplicate_fields,
        audit["duplicate_rows"],
    )
    _write_json(
        output / "mat_schema_inventory.json",
        {"status": "passed", "mat_files": audit["mat_schema_rows"]},
    )
    _write_json(
        output / "readme_exposure_log.json",
        {
            "status": "passed",
            "classification": EXPOSURE_CLASSIFICATION,
            "readmes": audit["readme_exposure_rows"],
        },
    )
    _write_json(output / "rights_review.json", audit["rights_review"])
    _write_json(output / "adversarial_results.json", adversarial_results)
    _write_json(output / "test_results.json", test_results)
    report = f"""# NASA extracted metadata intake V1 - 2026-08-06

Status: success

The extracted package contains {audit['mat_file_count']} MAT files representing
{audit['unique_physical_battery_count']} physical batteries and
{audit['txt_file_count']} README/TXT files. The four repeated physical IDs are
byte-identical representations and were canonicalized by relative-path order.

Only file metadata, streaming SHA-256, README/TXT text, 128-byte MATLAB headers,
and top-level `whosmat` schema metadata were accessed. No MATLAB array value,
capacity trajectory, prefix, prediction, or score was loaded or generated.

README exposure is classified as `{EXPOSURE_CLASSIFICATION}`. The evidence role
is development-only. Formal model execution remains blocked because the
dataset-specific license has not been explicitly resolved.
"""
    with (output / "intake_report.md").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(report)
    _, manifest_sha256 = _output_manifest(output)
    return {
        "status": "success",
        "audit": audit,
        "output_manifest_sha256": manifest_sha256,
    }
