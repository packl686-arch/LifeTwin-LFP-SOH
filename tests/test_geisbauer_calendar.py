from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator
import uuid
import zipfile

import numpy as np
import pandas as pd
import pytest

import lifetwin.data.geisbauer_calendar as geisbauer


@pytest.fixture
def scratch_path() -> Iterator[Path]:
    prefix = Path(__file__).resolve().parent / f".geisbauer-test-{uuid.uuid4().hex}"
    try:
        yield prefix
    finally:
        for suffix in (".csv", ".zip"):
            candidate = prefix.with_suffix(suffix)
            if candidate.is_file():
                candidate.unlink()


def _synthetic_source_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for soc_pct, cells in geisbauer.GEISBAUER_CALENDAR_EXPECTED_COHORT.items():
        for cell_number in cells:
            initial_capacity = 1.45 + cell_number / 10_000.0
            for elapsed_days in geisbauer.GEISBAUER_CALENDAR_EXPECTED_DAYS:
                fade = elapsed_days * (0.00015 + soc_pct / 2_000_000.0)
                discharge_capacity = initial_capacity * (1.0 - fade)
                rows.append(
                    {
                        "State-of-Charge": soc_pct,
                        "Temperature": 60,
                        "Cell Identity Number": cell_number,
                        "Days Passed": elapsed_days,
                        "Discharge Capacity": discharge_capacity,
                        "Charge Capacity": discharge_capacity * 0.99,
                    }
                )
    return pd.DataFrame(rows)[list(geisbauer.GEISBAUER_CALENDAR_INPUT_COLUMNS)]


def _csv_payload(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\r\n",
        float_format="%.10g",
    ).encode("utf-8")


def _patch_member_identity(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    monkeypatch.setattr(
        geisbauer,
        "GEISBAUER_CALENDAR_MEMBER_SIZE_BYTES",
        len(payload),
    )
    monkeypatch.setattr(
        geisbauer,
        "GEISBAUER_CALENDAR_MEMBER_MD5",
        hashlib.md5(payload).hexdigest(),
    )
    monkeypatch.setattr(
        geisbauer,
        "GEISBAUER_CALENDAR_MEMBER_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def _write_pinned_csv(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame | None = None,
) -> Path:
    payload = _csv_payload(_synthetic_source_frame() if frame is None else frame)
    _patch_member_identity(monkeypatch, payload)
    path = scratch_path.with_suffix(".csv")
    path.write_bytes(payload)
    return path


def _load_synthetic_csv(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = _write_pinned_csv(scratch_path, monkeypatch, frame)
    return geisbauer.load_geisbauer_calendar_observations(path)


def test_frozen_source_metadata_and_claim_boundary() -> None:
    assert geisbauer.GEISBAUER_CALENDAR_DOI == "10.5281/zenodo.6685365"
    assert geisbauer.GEISBAUER_CALENDAR_LICENSE == "CC-BY-4.0"
    assert geisbauer.GEISBAUER_CALENDAR_ZIP_SIZE_BYTES == 79_545
    assert geisbauer.GEISBAUER_CALENDAR_ZIP_SHA256 == (
        "327ab07a89b3eb68ef422817f4bd75f50ab6fed399d458381c3b9f1d984fe155"
    )
    assert geisbauer.GEISBAUER_CALENDAR_MEMBER_SIZE_BYTES == 2_752
    assert geisbauer.GEISBAUER_CALENDAR_MEMBER_SHA256 == (
        "4b42216bc87bbb3cfeab18a307da1149b6ca4583898cc359e32c81d219bb03cb"
    )
    assert geisbauer.GEISBAUER_CALENDAR_EVIDENCE_ROLE == (
        "accelerated_external_stress_check"
    )
    assert geisbauer.GEISBAUER_CALENDAR_CLAIM_BOUNDARY == (
        "not_long_term_validation"
    )


def test_csv_normalizes_cell_level_trajectories_and_provenance(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, audit = _load_synthetic_csv(scratch_path, monkeypatch)

    assert tuple(observations.columns) == geisbauer.GEISBAUER_CALENDAR_OUTPUT_COLUMNS
    assert len(observations) == 75
    assert observations["cell_id"].nunique() == 15
    assert observations["condition_id"].nunique() == 3
    assert set(observations.groupby("cell_id").size()) == {5}
    assert set(observations.groupby("condition_id")["cell_id"].nunique()) == {5}
    assert set(observations["temperature_c"]) == {60.0}
    assert set(observations["storage_soc_fraction"]) == {0.2, 0.5, 1.0}
    assert set(observations["statistical_unit"]) == {
        "individual_physical_cell_trajectory"
    }
    assert set(observations["physical_replicates_aggregated"]) == {1}
    assert set(observations["evidence_role"]) == {
        "accelerated_external_stress_check"
    }
    assert not observations["long_term_validation_eligible"].any()
    assert set(observations["source_doi"]) == {"10.5281/zenodo.6685365"}
    assert set(observations["source_license"]) == {"CC-BY-4.0"}
    assert set(observations["source_url"]) == {
        "https://zenodo.org/records/6685365"
    }

    cell7 = observations.loc[observations["source_cell_number"] == 7]
    assert set(cell7["cell_id"]) == {"GEISBAUER_LFP_CELL_007"}
    assert set(cell7["test_id"]) == {
        "GEISBAUER_LFP_CELL_007_CALENDAR_TEST"
    }
    assert set(cell7["source_cell_id"]) == {"GEISBAUER_SOURCE_CELL_007"}
    assert cell7["elapsed_days"].tolist() == [0.0, 39.0, 59.0, 84.0, 120.0]
    assert cell7["checkup_index"].tolist() == [0, 1, 2, 3, 4]
    assert np.isclose(
        cell7.iloc[0]["capacity_retention_pct"],
        100.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        cell7.iloc[0]["capacity_loss_pct"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        cell7["capacity_retention_pct"],
        100.0 * cell7["capacity_ah"] / cell7.iloc[0]["capacity_ah"],
        rtol=0.0,
        atol=1e-12,
    )

    assert audit["status"] == "passed"
    assert audit["physical_cell_count"] == 15
    assert audit["condition_count"] == 3
    assert audit["guardrails"]["long_term_validation_eligible"] is False
    assert audit["guardrails"]["confirmation_claim_allowed"] is False
    assert "never long-term external validation" in audit["warning"]


def test_zip_and_extracted_csv_produce_the_same_canonical_fingerprint(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _csv_payload(_synthetic_source_frame())
    _patch_member_identity(monkeypatch, payload)

    zip_path = scratch_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(geisbauer.GEISBAUER_CALENDAR_ZIP_MEMBER, payload)
    archive_payload = zip_path.read_bytes()
    monkeypatch.setattr(
        geisbauer,
        "GEISBAUER_CALENDAR_ZIP_SIZE_BYTES",
        len(archive_payload),
    )
    monkeypatch.setattr(
        geisbauer,
        "GEISBAUER_CALENDAR_ZIP_MD5",
        hashlib.md5(archive_payload).hexdigest(),
    )
    monkeypatch.setattr(
        geisbauer,
        "GEISBAUER_CALENDAR_ZIP_SHA256",
        hashlib.sha256(archive_payload).hexdigest(),
    )
    csv_path = scratch_path.with_suffix(".csv")
    csv_path.write_bytes(payload)
    csv_observations, csv_audit = (
        geisbauer.load_geisbauer_calendar_observations(csv_path)
    )
    zip_observations, zip_audit = (
        geisbauer.load_geisbauer_calendar_observations(zip_path)
    )

    assert csv_audit["source"]["input_kind"] == "extracted_csv"
    assert zip_audit["source"]["input_kind"] == "zip_archive"
    assert zip_audit["source"]["archive_hash_verified"] is True
    assert geisbauer.geisbauer_calendar_observations_sha256(
        csv_observations
    ) == geisbauer.geisbauer_calendar_observations_sha256(zip_observations)


def test_canonical_fingerprint_is_invariant_to_row_order(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, audit = _load_synthetic_csv(scratch_path, monkeypatch)
    shuffled = observations.sample(frac=1.0, random_state=17).reset_index(drop=True)

    expected = audit["canonical_output_sha256"]
    assert geisbauer.geisbauer_calendar_observations_sha256(observations) == expected
    assert geisbauer.geisbauer_calendar_observations_sha256(shuffled) == expected


def test_source_member_hash_is_mandatory(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _csv_payload(_synthetic_source_frame())
    path = scratch_path.with_suffix(".csv")
    path.write_bytes(payload)
    monkeypatch.setattr(
        geisbauer,
        "GEISBAUER_CALENDAR_MEMBER_SIZE_BYTES",
        len(payload),
    )

    with pytest.raises(ValueError, match="CSV member MD5 mismatch"):
        geisbauer.load_geisbauer_calendar_observations(path)


@pytest.mark.parametrize("schema_case", ["missing", "extra", "reordered"])
def test_strict_six_column_schema(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_case: str,
) -> None:
    frame = _synthetic_source_frame()
    if schema_case == "missing":
        frame = frame.drop(columns=["Charge Capacity"])
    elif schema_case == "extra":
        frame["Unexpected"] = 1
    else:
        frame = frame[list(reversed(frame.columns))]
    path = _write_pinned_csv(scratch_path, monkeypatch, frame)

    with pytest.raises(ValueError, match="exact six-column schema and order"):
        geisbauer.load_geisbauer_calendar_observations(path)


@pytest.mark.parametrize(
    ("invalid_case", "message"),
    [
        ("exact_duplicate", "Exact duplicate"),
        ("duplicate_identity", "Duplicate Geisbauer physical-cell/checkup"),
        ("null", "must be non-null"),
        ("infinite", "must be finite"),
        ("nonpositive", "capacities must be positive"),
        ("cardinality", "must contain 75 rows"),
    ],
)
def test_rejects_duplicate_null_nonfinite_nonpositive_and_cardinality_drift(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
    message: str,
) -> None:
    frame = _synthetic_source_frame()
    if invalid_case == "exact_duplicate":
        frame.iloc[-1] = frame.iloc[0]
    elif invalid_case == "duplicate_identity":
        frame.loc[frame.index[-1], "Cell Identity Number"] = frame.iloc[0][
            "Cell Identity Number"
        ]
        frame.loc[frame.index[-1], "Days Passed"] = frame.iloc[0]["Days Passed"]
    elif invalid_case == "null":
        frame.loc[0, "Discharge Capacity"] = np.nan
    elif invalid_case == "infinite":
        frame.loc[0, "Discharge Capacity"] = np.inf
    elif invalid_case == "nonpositive":
        frame.loc[0, "Discharge Capacity"] = 0.0
    else:
        frame = frame.iloc[:-1].copy()
    path = _write_pinned_csv(scratch_path, monkeypatch, frame)

    with pytest.raises(ValueError, match=message):
        geisbauer.load_geisbauer_calendar_observations(path)


def test_rejects_checkup_axis_and_cohort_identity_drift(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_source_frame()
    frame.loc[frame.index[-1], "Days Passed"] = 121
    path = _write_pinned_csv(scratch_path, monkeypatch, frame)
    with pytest.raises(ValueError, match="checkups must be exactly"):
        geisbauer.load_geisbauer_calendar_observations(path)

    frame = _synthetic_source_frame()
    frame.loc[frame["Cell Identity Number"] == 118, "State-of-Charge"] = 50
    path = _write_pinned_csv(scratch_path, monkeypatch, frame)
    with pytest.raises(ValueError, match="20% SOC cohort"):
        geisbauer.load_geisbauer_calendar_observations(path)


def test_canonical_guardrail_rejects_long_term_relabeling(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, _ = _load_synthetic_csv(scratch_path, monkeypatch)

    relabeled = observations.copy()
    relabeled["long_term_validation_eligible"] = True
    with pytest.raises(ValueError, match="never assert long-term validation"):
        geisbauer.validate_geisbauer_calendar_observations(relabeled)

    relabeled = observations.copy()
    relabeled["evidence_role"] = "independent_long_term_validation"
    with pytest.raises(ValueError, match="canonical evidence_role"):
        geisbauer.validate_geisbauer_calendar_observations(relabeled)


def test_day_zero_roundoff_is_accepted(
    scratch_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, _ = _load_synthetic_csv(scratch_path, monkeypatch)
    day_zero = observations["checkup_index"] == 0
    observations.loc[day_zero, "capacity_retention_pct"] = np.nextafter(
        100.0,
        np.inf,
    )
    observations.loc[day_zero, "capacity_loss_pct"] = (
        100.0 - observations.loc[day_zero, "capacity_retention_pct"]
    )

    geisbauer.validate_geisbauer_calendar_observations(observations)
