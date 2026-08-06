from __future__ import annotations

import argparse
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_audit(source_root: Path, output_directory: Path) -> dict[str, object]:
    before_files = collect_file_metadata(source_root)
    inventory = validate_expected_inventory(before_files)
    matr_json = audit_matr_json(source_root / "FastCharge")
    if matr_json["status"] != "passed":
        raise DataAssetIntakeError("MATR JSON identity audit failed")
    report = {
        "status": "passed",
        "source_writeback_detected": False,
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
    if before_files != after_files:
        raise DataAssetIntakeError("Source metadata changed during read-only audit")
    report["snapshot_equal"] = True
    _write_json(output_directory / "data_asset_audit.json", report)
    _write_json(
        output_directory / "file_metadata_inventory.json",
        file_metadata_records(before_files),
    )
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
