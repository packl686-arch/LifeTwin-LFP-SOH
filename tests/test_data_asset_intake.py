from __future__ import annotations

import builtins
from pathlib import Path
import re
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import lifetwin.data.asset_intake as intake


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _touch(path: Path, payload: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_lock_files_are_metadata_only_and_excluded(tmp_path: Path) -> None:
    _touch(tmp_path / "CALCE_A123" / "a.xlsx")
    _touch(tmp_path / "CALCE_A123" / "~$a.xlsx")
    files = intake.collect_file_metadata(tmp_path)
    summary = intake.snapshot_summary(files)
    assert summary["total_file_count"] == 2
    assert summary["lock_file_count"] == 1
    assert summary["valid_file_count"] == 1
    assert [row.name for row in files if row.is_lock_file] == ["~$a.xlsx"]


def _snl_fixture(root: Path) -> None:
    for index in range(30):
        key = f"SNL_18650_LFP_fixture_{index:02d}"
        _touch(root / "SNL LFP" / "SNL LFP" / f"{key}_cycle_data.csv")
        _touch(root / "SNL LFP" / "SNL LFP" / f"{key}_timeseries.csv")
    _touch(root / "SNL Metadata.xlsx")


def test_snl_audit_never_opens_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _snl_fixture(tmp_path)
    original_open = builtins.open
    original_path_open = Path.open

    def guarded_open(file, *args, **kwargs):
        if "SNL" in str(file):
            raise AssertionError("SNL content access attempted")
        return original_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        if "SNL" in str(self):
            raise AssertionError("SNL content access attempted")
        return original_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    files = intake.collect_file_metadata(tmp_path)
    audit = intake.audit_snl_metadata(files)
    assert audit["content_open_count"] == 0
    assert audit["content_hash_count"] == 0
    assert audit["pair_count"] == 30
    assert audit["policy"] == intake.SNL_POLICY


def test_snl_pairing_fails_closed_when_partner_is_missing(tmp_path: Path) -> None:
    _snl_fixture(tmp_path)
    (tmp_path / "SNL LFP" / "SNL LFP" / "SNL_18650_LFP_fixture_29_timeseries.csv").unlink()
    with pytest.raises(intake.DataAssetIntakeError, match="30 complete pairs"):
        intake.audit_snl_metadata(intake.collect_file_metadata(tmp_path))


def _record(path: Path, barcode: str, batch: str, protocol: str):
    return SimpleNamespace(
        barcode=barcode,
        batch_id=batch,
        protocol_id=protocol,
        channel_id=1,
        source_file=path.name,
        summary_rows=5,
    )


def test_matr_same_barcode_segments_merge_without_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [
        tmp_path / "a_structure.json",
        tmp_path / "b_structure.json",
        tmp_path / "c_structure.json",
    ]
    for path in paths:
        _touch(path)
    records = {
        "a_structure.json": _record(paths[0], "ONE", "B1", "P1"),
        "b_structure.json": _record(paths[1], "ONE", "B1", "P1"),
        "c_structure.json": _record(paths[2], "TWO", "B2", "P2"),
    }

    def fake_reader(path, *, hash_source):
        assert hash_source is False
        return records[Path(path).name]

    monkeypatch.setattr(intake, "read_beep_summary", fake_reader)
    audit = intake.audit_matr_json(tmp_path)
    assert audit["json_file_count"] == 3
    assert audit["unique_barcode_count"] == 2
    assert audit["multi_segment_barcode_count"] == 1
    assert audit["identity_conflict_count"] == 0
    assert audit["cycles_interpolated_read"] is False


def test_matr_conflicting_segment_identity_fails_audit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "a_structure.json"
    second = tmp_path / "b_structure.json"
    _touch(first)
    _touch(second)
    records = {
        first.name: _record(first, "ONE", "B1", "P1"),
        second.name: _record(second, "ONE", "B1", "P2"),
    }
    monkeypatch.setattr(
        intake,
        "read_beep_summary",
        lambda path, hash_source=False: records[Path(path).name],
    )
    audit = intake.audit_matr_json(tmp_path)
    assert audit["status"] == "failed"
    assert audit["identity_conflict_count"] == 1


def test_matr_mat_representations_add_zero_cells(tmp_path: Path) -> None:
    for index in range(5):
        header = (
            b"MATLAB 7.3 MAT-file" if index < 2 else b"MATLAB 5.0 MAT-file"
        )
        _touch(tmp_path / f"representation_{index}.mat", header.ljust(128, b" "))
    audit = intake.audit_matr_mat_representations(tmp_path)
    assert audit["mat_file_count"] == 5
    assert audit["physical_cells_added_by_mat"] == 0
    assert audit["future_outcomes_used_for_mapping"] is False


def _ordinary_ids() -> list[str]:
    return [
        "B0005",
        "B0006",
        "B0007",
        "B0018",
        *[f"B{index:04d}" for index in range(25, 35)],
        "B0036",
        *[f"B{index:04d}" for index in range(38, 57)],
    ]


def test_nasa_duplicate_ids_are_canonicalized_from_central_directory(
    tmp_path: Path,
) -> None:
    ids = _ordinary_ids()
    bundles = [ids[:6], ids[6:12], ids[12:18], ids[18:24], ids[24:29], ids[29:]]
    for index, group in enumerate(bundles, start=1):
        with ZipFile(tmp_path / f"bundle_{index}.zip", "w") as archive:
            for physical_id in group:
                archive.writestr(f"{physical_id}.mat", physical_id.encode())
            if index == 1:
                for physical_id in intake.NASA_DUPLICATE_IDS:
                    archive.writestr(f"duplicates/{physical_id}.mat", physical_id.encode())
    audit = intake.audit_nasa_battery_zip_metadata(tmp_path)
    assert audit["unique_physical_battery_count"] == 34
    assert set(audit["duplicate_physical_battery_ids"]) == set(
        intake.NASA_DUPLICATE_IDS
    )
    assert all(
        row["central_directory_identity_count"] == 1
        for row in audit["duplicate_physical_battery_ids"].values()
    )


def test_public_asset_files_have_no_local_paths_or_personal_information() -> None:
    paths = [
        PROJECT_ROOT / "docs/data_asset_register_20260806.csv",
        PROJECT_ROOT / "docs/data_asset_intake_20260806.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text)
    assert not re.search(r"[\w.+-]+@[\w.-]+", text)
    assert "users.noreply" not in text.casefold()
