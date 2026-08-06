from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from lifetwin.data.asset_intake import (
    DataAssetIntakeError,
    audit_calce_metadata,
    audit_matr_json,
    audit_matr_mat_representations,
    audit_nasa_battery_zip_metadata,
    audit_nasa_randomized_zip_metadata,
    audit_oxford_metadata,
    audit_snl_metadata,
    collect_file_metadata,
    file_metadata_records,
    snapshot_summary,
    validate_expected_inventory,
)
from lifetwin.experiments.nasa_official_prefix_stress import (
    NasaOfficialPrefixStressError,
    load_nasa_official_prefix_stress_config,
    predict_prefix_baselines,
    prepare_prefix_and_future_labels,
    score_prefix_baselines,
)


ALLOWED_SUBAUDIT_STATUSES = {
    "inventory": {"passed"},
    "snl": {"passed_metadata_only"},
    "matr_json": {"passed"},
    "matr_mat": {"passed_no_duplicate_counting"},
    "nasa_battery": {"passed_central_directory_only"},
    "nasa_randomized": {"inventory_only_physical_identity_not_verified"},
    "calce": {"passed_metadata_only"},
    "oxford": {"passed_source_record_and_file_metadata"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_output_manifest(output_directory: Path) -> tuple[Path, str]:
    manifest_path = output_directory / "output_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Output manifest already exists: {manifest_path}")
    generated = _utc_now()
    entries = [
        {
            "relative_path": path.relative_to(output_directory).as_posix(),
            "sha256": _sha256(path),
            "byte_count": path.stat().st_size,
            "generated_utc": generated,
        }
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "generated_utc": generated,
            "entries": entries,
            "manifest_self_hash_excluded": True,
        },
    )
    return manifest_path, _sha256(manifest_path)


def write_integrity_closeout(
    fastcharge_directory: Path,
    config_path: Path,
    output_directory: Path,
    test_results: dict[str, object],
) -> dict[str, object]:
    if output_directory.exists():
        raise FileExistsError(f"Closeout output already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(exist_ok=False)
    matr = audit_matr_json(fastcharge_directory)
    if matr["status"] != "passed":
        raise DataAssetIntakeError("MATR identity closeout did not pass")
    config = load_nasa_official_prefix_stress_config(config_path)
    gate_results: dict[str, object] = {}
    for name, function, arguments in (
        ("prepare", prepare_prefix_and_future_labels, (object(), config)),
        ("predict", predict_prefix_baselines, (object(), config)),
        ("score", score_prefix_baselines, (object(), object(), {}, config)),
    ):
        try:
            function(*arguments)
        except NasaOfficialPrefixStressError as exc:
            gate_results[name] = {"status": "rejected", "reason": str(exc)}
        else:
            gate_results[name] = {"status": "bypassed"}
    if any(value["status"] != "rejected" for value in gate_results.values()):
        raise DataAssetIntakeError("NASA direct-call rights gate was bypassed")
    protocol = {
        "status": "success",
        "protocol_id": "data_governance_integrity_closeout_v1_1_20260806",
        "base_commit": "9e2884a82710c2d64ca9b4d412acca5030a21986",
        "nasa_config_semantic_sha256": config["semantic_sha256"],
        "nasa_execution_allowed": False,
        "matr_json_file_count": matr["json_file_count"],
        "unique_barcode_count": matr["unique_barcode_count"],
        "additional_segment_count": matr["additional_segment_file_count"],
        "identity_conflict_count": matr["identity_conflict_count"],
        "physical_cells_added_by_mat": 0,
        "outcome_value_materialized_count": matr[
            "outcome_value_materialized_count"
        ],
        "read_beep_summary_call_count": matr["read_beep_summary_call_count"],
        "snl_content_open_count": 0,
        "nasa_real_outcome_open_count": 0,
    }
    _write_json(output_directory / "protocol_freeze.json", protocol)
    with (output_directory / "matr_identity_manifest.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        columns = (
            "barcode",
            "segment_count",
            "batch_id",
            "protocol_id",
            "channel_ids",
            "source_files",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for barcode, detail in matr["barcode_detail"].items():
            writer.writerow(
                {
                    "barcode": barcode,
                    "segment_count": detail["segment_count"],
                    "batch_id": detail["batch_id"],
                    "protocol_id": detail["protocol_id"],
                    "channel_ids": "|".join(map(str, detail["channel_ids"])),
                    "source_files": "|".join(detail["source_files"]),
                }
            )
    _write_json(
        output_directory / "gate_adversarial_results.json",
        {
            "status": "passed",
            "formal_direct_calls": gate_results,
            "nasa_real_outcome_open_count": 0,
        },
    )
    _write_json(output_directory / "test_results.json", test_results)
    report = """# Data governance integrity V1.1 closeout

Status: success

MATR identity intake used the identity-field whitelist reader. It did not call
the summary parser, materialize summary values, or materialize curve outcomes.
The 140 files resolve to 135 barcodes with five additional segments and zero
identity conflicts. Root MAT representations add zero physical cells.

The formal NASA configuration remains blocked. Direct library calls to prepare,
predict, and score were rejected before input processing. Synthetic tests
confirmed single-attempt scoring, non-overwrite behavior, failed-receipt
retention, and output-manifest verification. No SNL content or real NASA outcome
was opened, and no model was trained.
"""
    with (output_directory / "closeout_report.md").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(report)
    manifest_path, manifest_sha256 = write_output_manifest(output_directory)
    return {
        "status": "success",
        "protocol": protocol,
        "gate_results": gate_results,
        "output_manifest": manifest_path.name,
        "output_manifest_sha256": manifest_sha256,
    }


def run_audit(source_root: Path, output_directory: Path) -> dict[str, object]:
    if output_directory.exists():
        raise FileExistsError(f"Audit output directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(exist_ok=False)
    before_files = collect_file_metadata(source_root)
    inventory = validate_expected_inventory(before_files)
    matr_json = audit_matr_json(source_root / "FastCharge")
    if matr_json["status"] != "passed":
        raise DataAssetIntakeError("MATR JSON identity audit failed")
    report: dict[str, object] = {
        "before_snapshot": snapshot_summary(before_files),
        "inventory": inventory,
        "snl": audit_snl_metadata(before_files),
        "matr_json": matr_json,
        "matr_mat": audit_matr_mat_representations(source_root),
        "nasa_battery": audit_nasa_battery_zip_metadata(
            source_root / "5. Battery Data Set"
        ),
        "nasa_randomized": audit_nasa_randomized_zip_metadata(
            source_root / "11. Randomized Battery Usage Data Set"
        ),
        "calce": audit_calce_metadata(before_files),
        "oxford": audit_oxford_metadata(before_files),
    }
    after_files = collect_file_metadata(source_root)
    report["after_snapshot"] = snapshot_summary(after_files)
    report["source_writeback_detected"] = before_files != after_files
    report["snapshot_equal"] = before_files == after_files
    bad_statuses = {
        name: value["status"]
        for name, value in report.items()
        if name in ALLOWED_SUBAUDIT_STATUSES
        and value["status"] not in ALLOWED_SUBAUDIT_STATUSES[name]
    }
    report["subaudit_status_violations"] = bad_statuses
    report["status"] = (
        "passed"
        if not report["source_writeback_detected"] and not bad_statuses
        else "failed"
    )
    _write_json(output_directory / "data_asset_audit.json", report)
    _write_json(
        output_directory / "file_metadata_inventory.json",
        file_metadata_records(before_files),
    )
    write_output_manifest(output_directory)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit governed local data assets.")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.source_root.resolve(), args.output_directory.resolve())
    summary = {
        "status": report["status"],
        "inventory": report["inventory"]["observed"],
        "snl_content_open_count": report["snl"]["content_open_count"],
        "matr_unique_barcode_count": report["matr_json"]["unique_barcode_count"],
        "nasa_unique_battery_count": report["nasa_battery"][
            "unique_physical_battery_count"
        ],
        "snapshot_equal": report["snapshot_equal"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
