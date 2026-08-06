from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import scipy.io

import lifetwin.data.nasa_extracted_metadata as extracted


def _physical_ids() -> list[str]:
    return [
        "B0005",
        "B0006",
        "B0007",
        "B0018",
        *[f"B{index:04d}" for index in range(25, 35)],
        "B0036",
        *[f"B{index:04d}" for index in range(38, 57)],
    ]


def _write_mat(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MATLAB 5.0 MAT-file".ljust(128, b" ") + payload)


@pytest.fixture
def extracted_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    directories = [tmp_path / f"{index}. bundle" for index in range(1, 7)]
    for directory in directories:
        directory.mkdir()
    for index, physical_id in enumerate(_physical_ids()):
        _write_mat(
            directories[index % len(directories)] / f"{physical_id}.mat",
            physical_id.encode("ascii"),
        )
    for index, physical_id in enumerate(extracted.EXPECTED_DUPLICATE_IDS):
        source = next(tmp_path.rglob(f"{physical_id}.mat"))
        _write_mat(directories[(index + 3) % 6] / source.name, physical_id.encode("ascii"))
    readme_text = (
        "End-of-life uses a 20% capacity fade stop condition at 1.4 Ah.\n"
        "A software crash caused an experimental anomaly.\n"
        "Please cite the repository publication; license details are not specified.\n"
    )
    for index in range(10):
        (directories[index % 6] / f"README_{index}.txt").write_text(
            readme_text, encoding="utf-8"
        )
    monkeypatch.setattr(
        extracted,
        "whosmat",
        lambda path: [(Path(path).stem, (1, 1), "struct")],
    )
    return tmp_path


def test_filename_identity_reduces_38_files_to_34_batteries(
    extracted_fixture: Path,
) -> None:
    audit = extracted.audit_nasa_extracted_metadata(extracted_fixture)
    assert audit["mat_file_count"] == 38
    assert audit["unique_physical_battery_count"] == 34
    assert audit["duplicate_physical_battery_id_count"] == 4
    assert tuple(audit["duplicate_physical_battery_ids"]) == (
        extracted.EXPECTED_DUPLICATE_IDS
    )
    assert audit["txt_file_count"] == 10


def test_identical_duplicate_representations_are_canonicalized(
    extracted_fixture: Path,
) -> None:
    audit = extracted.audit_nasa_extracted_metadata(extracted_fixture)
    assert len(audit["duplicate_rows"]) == 4
    for row in audit["duplicate_rows"]:
        assert row["canonical_byte_count"] == row["duplicate_byte_count"]
        assert row["canonical_sha256"] == row["duplicate_sha256"]
        assert row["identity_matches"] is True
        assert row["status"] == "canonicalized_identical_representation"
        assert row["canonical_relative_path"] < row["duplicate_relative_path"]


def test_duplicate_identity_conflict_fails_closed(extracted_fixture: Path) -> None:
    paths = sorted(extracted_fixture.rglob("B0025.mat"))
    paths[1].write_bytes(paths[1].read_bytes() + b"conflict")
    with pytest.raises(extracted.NasaExtractedMetadataError, match="conflict"):
        extracted.audit_nasa_extracted_metadata(extracted_fixture)


def test_mat_filename_without_battery_id_is_rejected(extracted_fixture: Path) -> None:
    path = next(extracted_fixture.rglob("B0005.mat"))
    path.rename(path.with_name("battery.mat"))
    with pytest.raises(extracted.NasaExtractedMetadataError, match="no physical"):
        extracted.audit_nasa_extracted_metadata(extracted_fixture)


def test_only_hash_header_and_whosmat_are_used(
    extracted_fixture: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenDataset:
        def __getitem__(self, _key):
            raise AssertionError("HDF5 Dataset value access attempted")

    monkeypatch.setitem(
        sys.modules,
        "h5py",
        SimpleNamespace(Dataset=ForbiddenDataset),
    )
    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("loadmat value access attempted")
        ),
    )
    audit = extracted.audit_nasa_extracted_metadata(extracted_fixture)
    assert audit["schema_readable_count"] == 38
    assert audit["schema_blocked_count"] == 0
    assert audit["mat_value_load_count"] == 0
    assert audit["capacity_value_read_count"] == 0
    assert all(
        row["outcome_values_loaded"] is False
        for row in audit["mat_schema_rows"]
    )


def test_unreadable_schema_is_recorded_without_value_fallback(
    extracted_fixture: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = next(extracted_fixture.rglob("B0005.mat"))

    def guarded_whosmat(path):
        if Path(path) == target:
            raise ValueError("synthetic unreadable schema")
        return [(Path(path).stem, (1, 1), "struct")]

    monkeypatch.setattr(extracted, "whosmat", guarded_whosmat)
    audit = extracted.audit_nasa_extracted_metadata(extracted_fixture)
    assert audit["schema_readable_count"] == 37
    assert audit["schema_blocked_count"] == 1


def test_readme_exposure_is_fixed_as_development_only(
    extracted_fixture: Path,
) -> None:
    audit = extracted.audit_nasa_extracted_metadata(extracted_fixture)
    assert audit["readme_exposure_classification"] == (
        "development_only_outcomes_and_protocol_structure_exposed"
    )
    assert audit["evidence_role"] == "development_only"
    assert len(audit["readme_exposure_rows"]) == 10
    assert all(
        row["model_or_threshold_selection_allowed"] is False
        for row in audit["readme_exposure_rows"]
    )


def test_output_is_non_overwriting_and_manifest_recomputes(
    extracted_fixture: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    result = extracted.write_nasa_extracted_metadata_intake(
        extracted_fixture,
        output,
        adversarial_results={"status": "passed"},
        test_results={"status": "passed"},
    )
    expected = {
        "protocol_freeze.json",
        "extracted_file_inventory.json",
        "physical_battery_identity.csv",
        "duplicate_representation_manifest.csv",
        "mat_schema_inventory.json",
        "readme_exposure_log.json",
        "rights_review.json",
        "adversarial_results.json",
        "test_results.json",
        "intake_report.md",
        "output_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest_path = output / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 10
    for entry in manifest["entries"]:
        path = output / entry["relative_path"]
        assert path.stat().st_size == entry["byte_count"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == result[
        "output_manifest_sha256"
    ]
    with pytest.raises(FileExistsError):
        extracted.write_nasa_extracted_metadata_intake(
            extracted_fixture,
            output,
            adversarial_results={"status": "passed"},
            test_results={"status": "passed"},
        )


def test_rights_review_remains_blocked(extracted_fixture: Path) -> None:
    rights = extracted.audit_nasa_extracted_metadata(extracted_fixture)[
        "rights_review"
    ]
    assert rights["formal_model_execution_allowed"] is False
    assert rights["dataset_specific_license_identifier"] is None
    assert rights["public_aggregate_result_release_confirmed"] is False
    assert rights["blocking_reason"] == (
        "dataset_specific_license_not_explicitly_resolved"
    )
