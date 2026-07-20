from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


GEISBAUER_CALENDAR_DATASET_ID = "GEISBAUER_LFP_CALENDAR_2022"
GEISBAUER_CALENDAR_DOI = "10.5281/zenodo.6685365"
GEISBAUER_CALENDAR_SOURCE_URL = "https://zenodo.org/records/6685365"
GEISBAUER_CALENDAR_LICENSE = "CC-BY-4.0"
GEISBAUER_CALENDAR_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
GEISBAUER_CALENDAR_VERSION = "v3"
GEISBAUER_CALENDAR_EVIDENCE_ROLE = "accelerated_external_stress_check"
GEISBAUER_CALENDAR_CLAIM_BOUNDARY = "not_long_term_validation"
GEISBAUER_CALENDAR_STATISTICAL_UNIT = "individual_physical_cell_trajectory"
GEISBAUER_CALENDAR_REPLICATE_SEMANTICS = (
    "one_trajectory_per_physical_cell_no_aggregation"
)

GEISBAUER_CALENDAR_ZIP_NAME = (
    "Experimental Calendar Ageing Data for Lithium-Ion Battery Chemistries.zip"
)
GEISBAUER_CALENDAR_ZIP_MEMBER = (
    "Experimental Calendar Ageing Data for Lithium-Ion Battery Chemistries/"
    "CSV Format/LFP_Data.csv"
)
GEISBAUER_CALENDAR_ZIP_SIZE_BYTES = 79_545
GEISBAUER_CALENDAR_ZIP_MD5 = "2dae24f63fd0abcb08a8809f4156bcad"
GEISBAUER_CALENDAR_ZIP_SHA256 = (
    "327ab07a89b3eb68ef422817f4bd75f50ab6fed399d458381c3b9f1d984fe155"
)
GEISBAUER_CALENDAR_MEMBER_SIZE_BYTES = 2_752
GEISBAUER_CALENDAR_MEMBER_MD5 = "b8b7219ba9146da6d76c779f4d8c2e75"
GEISBAUER_CALENDAR_MEMBER_SHA256 = (
    "4b42216bc87bbb3cfeab18a307da1149b6ca4583898cc359e32c81d219bb03cb"
)
GEISBAUER_CALENDAR_OBSERVATIONS_SHA256 = (
    "e77981c702faec169ca6b8b0d44f3f5f6072654a4928273c2133eac0a8469e54"
)

GEISBAUER_CALENDAR_INPUT_COLUMNS = (
    "State-of-Charge",
    "Temperature",
    "Cell Identity Number",
    "Days Passed",
    "Discharge Capacity",
    "Charge Capacity",
)
GEISBAUER_CALENDAR_EXPECTED_DAYS = (0, 39, 59, 84, 120)
GEISBAUER_CALENDAR_EXPECTED_TEMPERATURE_C = 60
GEISBAUER_CALENDAR_EXPECTED_COHORT = {
    20: (7, 8, 71, 72, 118),
    50: (25, 26, 83, 84, 115),
    100: (43, 44, 95, 96, 112),
}

GEISBAUER_CALENDAR_OUTPUT_COLUMNS = (
    "dataset_id",
    "condition_id",
    "cell_id",
    "test_id",
    "source_cell_id",
    "source_cell_number",
    "temperature_c",
    "storage_soc_fraction",
    "elapsed_time_s",
    "elapsed_hours",
    "elapsed_days",
    "checkup_index",
    "capacity_ah",
    "charge_capacity_ah",
    "capacity_retention_pct",
    "capacity_loss_pct",
    "physical_replicates_aggregated",
    "replicate_semantics",
    "statistical_unit",
    "evidence_role",
    "long_term_validation_eligible",
    "claim_boundary",
    "source_doi",
    "source_url",
    "source_license",
    "source_license_url",
    "source_version",
    "source_archive_sha256",
    "source_member_sha256",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _md5_bytes(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()


def _verify_payload(
    payload: bytes,
    *,
    label: str,
    expected_size: int,
    expected_md5: str,
    expected_sha256: str,
) -> None:
    if len(payload) != expected_size:
        raise ValueError(
            f"Geisbauer {label} size mismatch: expected {expected_size}, "
            f"found {len(payload)}"
        )
    observed_md5 = _md5_bytes(payload)
    if observed_md5 != expected_md5:
        raise ValueError(
            f"Geisbauer {label} MD5 mismatch: expected {expected_md5}, "
            f"found {observed_md5}"
        )
    observed_sha256 = _sha256_bytes(payload)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Geisbauer {label} SHA-256 mismatch: expected {expected_sha256}, "
            f"found {observed_sha256}"
        )


def _read_source_member(source: Path) -> tuple[bytes, dict[str, object]]:
    if not source.is_file():
        raise FileNotFoundError(f"Geisbauer source does not exist: {source}")
    supplied = source.read_bytes()
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        _verify_payload(
            supplied,
            label="CSV member",
            expected_size=GEISBAUER_CALENDAR_MEMBER_SIZE_BYTES,
            expected_md5=GEISBAUER_CALENDAR_MEMBER_MD5,
            expected_sha256=GEISBAUER_CALENDAR_MEMBER_SHA256,
        )
        return supplied, {
            "input_kind": "extracted_csv",
            "provided_path": str(source.resolve()),
            "provided_size_bytes": len(supplied),
            "provided_sha256": _sha256_bytes(supplied),
            "archive_hash_verified": False,
            "member_hash_verified": True,
        }
    if suffix != ".zip":
        raise ValueError("Geisbauer source must be an extracted .csv or the pinned .zip")

    _verify_payload(
        supplied,
        label="ZIP archive",
        expected_size=GEISBAUER_CALENDAR_ZIP_SIZE_BYTES,
        expected_md5=GEISBAUER_CALENDAR_ZIP_MD5,
        expected_sha256=GEISBAUER_CALENDAR_ZIP_SHA256,
    )
    try:
        with zipfile.ZipFile(io.BytesIO(supplied)) as archive:
            member_count = archive.namelist().count(GEISBAUER_CALENDAR_ZIP_MEMBER)
            if member_count != 1:
                raise ValueError(
                    "Pinned Geisbauer ZIP must contain the LFP CSV member exactly once"
                )
            member_info = archive.getinfo(GEISBAUER_CALENDAR_ZIP_MEMBER)
            if member_info.is_dir():
                raise ValueError("Pinned Geisbauer LFP ZIP member cannot be a directory")
            member = archive.read(GEISBAUER_CALENDAR_ZIP_MEMBER)
    except zipfile.BadZipFile as exc:
        raise ValueError("Pinned Geisbauer source is not a readable ZIP archive") from exc

    _verify_payload(
        member,
        label="CSV member",
        expected_size=GEISBAUER_CALENDAR_MEMBER_SIZE_BYTES,
        expected_md5=GEISBAUER_CALENDAR_MEMBER_MD5,
        expected_sha256=GEISBAUER_CALENDAR_MEMBER_SHA256,
    )
    return member, {
        "input_kind": "zip_archive",
        "provided_path": str(source.resolve()),
        "provided_size_bytes": len(supplied),
        "provided_sha256": _sha256_bytes(supplied),
        "archive_hash_verified": True,
        "member_hash_verified": True,
    }


def _validate_raw_geisbauer_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    if tuple(raw.columns) != GEISBAUER_CALENDAR_INPUT_COLUMNS:
        raise ValueError(
            "Geisbauer LFP CSV must have the exact six-column schema and order: "
            f"{list(GEISBAUER_CALENDAR_INPUT_COLUMNS)}"
        )
    if raw.empty:
        raise ValueError("Geisbauer LFP CSV cannot be empty")
    if raw.isna().any().any():
        raise ValueError("Geisbauer LFP CSV values must be non-null")
    if raw.duplicated().any():
        raise ValueError("Exact duplicate Geisbauer LFP CSV rows are not allowed")

    numeric = raw.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Geisbauer LFP CSV values must be numeric")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Geisbauer LFP CSV numeric values must be finite")
    if numeric.duplicated().any():
        raise ValueError("Numerically duplicate Geisbauer LFP CSV rows are not allowed")

    integer_columns = (
        "State-of-Charge",
        "Temperature",
        "Cell Identity Number",
        "Days Passed",
    )
    for column in integer_columns:
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Geisbauer {column} values must be exact integers")
        numeric[column] = values.astype("int64")

    if (numeric["Cell Identity Number"] <= 0).any():
        raise ValueError("Geisbauer physical-cell numbers must be positive")
    if (numeric["Days Passed"] < 0).any():
        raise ValueError("Geisbauer elapsed days must be non-negative")
    if (numeric[["Discharge Capacity", "Charge Capacity"]] <= 0).any().any():
        raise ValueError("Geisbauer charge and discharge capacities must be positive")
    if not numeric["State-of-Charge"].between(0, 100).all():
        raise ValueError("Geisbauer storage SOC must be expressed as percent in [0, 100]")
    if set(numeric["Temperature"].astype(int)) != {
        GEISBAUER_CALENDAR_EXPECTED_TEMPERATURE_C
    }:
        raise ValueError("Geisbauer LFP accelerated cohort must remain fixed at 60 C")
    if numeric.duplicated(["Cell Identity Number", "Days Passed"]).any():
        raise ValueError("Duplicate Geisbauer physical-cell/checkup rows are not allowed")

    expected_cells = {
        cell
        for cohort in GEISBAUER_CALENDAR_EXPECTED_COHORT.values()
        for cell in cohort
    }
    observed_cells = set(numeric["Cell Identity Number"].astype(int))
    if observed_cells != expected_cells:
        raise ValueError(
            "Geisbauer physical-cell cohort mismatch: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )
    if len(numeric) != 75 or len(observed_cells) != 15:
        raise ValueError("Geisbauer LFP cohort must contain 75 rows from 15 cells")

    expected_days = list(GEISBAUER_CALENDAR_EXPECTED_DAYS)
    for soc_pct, expected_cohort in GEISBAUER_CALENDAR_EXPECTED_COHORT.items():
        cohort = numeric.loc[numeric["State-of-Charge"] == soc_pct]
        observed_cohort = set(cohort["Cell Identity Number"].astype(int))
        if observed_cohort != set(expected_cohort) or len(cohort) != 25:
            raise ValueError(
                f"Geisbauer {soc_pct}% SOC cohort must contain its five pinned cells"
            )
    if set(numeric["State-of-Charge"].astype(int)) != set(
        GEISBAUER_CALENDAR_EXPECTED_COHORT
    ):
        raise ValueError("Geisbauer LFP cohort must contain exactly three SOC levels")

    for cell_number, cell in numeric.groupby("Cell Identity Number", sort=True):
        ordered = cell.sort_values("Days Passed", kind="stable")
        if ordered["Days Passed"].astype(int).tolist() != expected_days:
            raise ValueError(
                "Geisbauer physical-cell checkups must be exactly "
                f"{expected_days}: cell {int(cell_number)}"
            )
        if ordered["State-of-Charge"].nunique() != 1:
            raise ValueError(
                f"Geisbauer storage SOC changes within cell {int(cell_number)}"
            )
        if ordered["Temperature"].nunique() != 1:
            raise ValueError(
                f"Geisbauer temperature changes within cell {int(cell_number)}"
            )

    return numeric.sort_values(
        ["Cell Identity Number", "Days Passed"],
        kind="stable",
    ).reset_index(drop=True)


def _condition_id(soc_pct: int) -> str:
    return f"GEISBAUER_LFP_T60_SOC{soc_pct}"


def _cell_id(cell_number: int) -> str:
    return f"GEISBAUER_LFP_CELL_{cell_number:03d}"


def _test_id(cell_number: int) -> str:
    return f"GEISBAUER_LFP_CELL_{cell_number:03d}_CALENDAR_TEST"


def _source_cell_id(cell_number: int) -> str:
    return f"GEISBAUER_SOURCE_CELL_{cell_number:03d}"


def validate_geisbauer_calendar_observations(observations: pd.DataFrame) -> None:
    """Validate the frozen, cell-level accelerated external stress cohort."""
    if tuple(observations.columns) != GEISBAUER_CALENDAR_OUTPUT_COLUMNS:
        raise ValueError("Geisbauer canonical output schema or column order changed")
    if observations.empty or len(observations) != 75:
        raise ValueError("Geisbauer canonical output must contain exactly 75 rows")
    if observations.isna().any().any():
        raise ValueError("Geisbauer canonical output values must be non-null")
    if observations.duplicated().any():
        raise ValueError("Exact duplicate Geisbauer canonical rows are not allowed")
    if observations.duplicated(["cell_id", "checkup_index"]).any():
        raise ValueError("Duplicate Geisbauer cell/checkup identities are not allowed")
    if observations.duplicated(["cell_id", "elapsed_days"]).any():
        raise ValueError("Duplicate Geisbauer cell/day identities are not allowed")

    numeric_columns = [
        "source_cell_number",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_time_s",
        "elapsed_hours",
        "elapsed_days",
        "checkup_index",
        "capacity_ah",
        "charge_capacity_ah",
        "capacity_retention_pct",
        "capacity_loss_pct",
        "physical_replicates_aggregated",
    ]
    numeric = observations[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError("Geisbauer canonical numeric values must be finite")
    if (numeric[["capacity_ah", "charge_capacity_ah"]] <= 0).any().any():
        raise ValueError("Geisbauer canonical capacities must be positive")
    if (numeric[["elapsed_time_s", "elapsed_hours", "elapsed_days"]] < 0).any().any():
        raise ValueError("Geisbauer canonical elapsed times must be non-negative")
    if not numeric["storage_soc_fraction"].between(0.0, 1.0).all():
        raise ValueError("Geisbauer canonical storage SOC must be a fraction")
    if set(numeric["temperature_c"].astype(int)) != {
        GEISBAUER_CALENDAR_EXPECTED_TEMPERATURE_C
    }:
        raise ValueError("Geisbauer canonical temperature must remain 60 C")
    if set(numeric["physical_replicates_aggregated"].astype(int)) != {1}:
        raise ValueError("Geisbauer physical-cell trajectories cannot be aggregated")
    if not np.allclose(
        numeric["elapsed_hours"].to_numpy(dtype=float),
        numeric["elapsed_days"].to_numpy(dtype=float) * 24.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Geisbauer elapsed-hour conversion is inconsistent")
    if not np.allclose(
        numeric["elapsed_time_s"].to_numpy(dtype=float),
        numeric["elapsed_days"].to_numpy(dtype=float) * 86_400.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Geisbauer elapsed-second conversion is inconsistent")
    if not np.allclose(
        numeric["capacity_loss_pct"].to_numpy(dtype=float),
        100.0 - numeric["capacity_retention_pct"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Geisbauer capacity loss must equal 100 minus retention")

    exact_provenance = {
        "dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
        "replicate_semantics": GEISBAUER_CALENDAR_REPLICATE_SEMANTICS,
        "statistical_unit": GEISBAUER_CALENDAR_STATISTICAL_UNIT,
        "evidence_role": GEISBAUER_CALENDAR_EVIDENCE_ROLE,
        "claim_boundary": GEISBAUER_CALENDAR_CLAIM_BOUNDARY,
        "source_doi": GEISBAUER_CALENDAR_DOI,
        "source_url": GEISBAUER_CALENDAR_SOURCE_URL,
        "source_license": GEISBAUER_CALENDAR_LICENSE,
        "source_license_url": GEISBAUER_CALENDAR_LICENSE_URL,
        "source_version": GEISBAUER_CALENDAR_VERSION,
        "source_archive_sha256": GEISBAUER_CALENDAR_ZIP_SHA256,
        "source_member_sha256": GEISBAUER_CALENDAR_MEMBER_SHA256,
    }
    for column, expected in exact_provenance.items():
        if set(observations[column].astype(str)) != {expected}:
            raise ValueError(f"Unexpected Geisbauer canonical {column}")
    if not observations["long_term_validation_eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and not bool(value)
    ).all():
        raise ValueError(
            "Geisbauer accelerated stress data can never assert long-term validation"
        )

    expected_cells = {
        cell
        for cohort in GEISBAUER_CALENDAR_EXPECTED_COHORT.values()
        for cell in cohort
    }
    observed_cells = set(numeric["source_cell_number"].astype(int))
    if observed_cells != expected_cells or observations["cell_id"].nunique() != 15:
        raise ValueError("Geisbauer canonical physical-cell cohort changed")

    for cell_number, cell in observations.groupby("source_cell_number", sort=True):
        number = int(cell_number)
        ordered = cell.sort_values("elapsed_days", kind="stable")
        if ordered["elapsed_days"].astype(int).tolist() != list(
            GEISBAUER_CALENDAR_EXPECTED_DAYS
        ):
            raise ValueError(f"Geisbauer canonical checkup axis changed for cell {number}")
        if ordered["checkup_index"].astype(int).tolist() != list(range(5)):
            raise ValueError(f"Geisbauer checkup indices changed for cell {number}")
        expected_identity = {
            "cell_id": _cell_id(number),
            "test_id": _test_id(number),
            "source_cell_id": _source_cell_id(number),
        }
        for column, expected in expected_identity.items():
            if set(ordered[column].astype(str)) != {expected}:
                raise ValueError(
                    f"Geisbauer stable {column} changed for physical cell {number}"
                )
        soc_pct = next(
            soc
            for soc, cohort in GEISBAUER_CALENDAR_EXPECTED_COHORT.items()
            if number in cohort
        )
        if set(ordered["condition_id"].astype(str)) != {_condition_id(soc_pct)}:
            raise ValueError(f"Geisbauer condition identity changed for cell {number}")
        if not np.allclose(
            ordered["storage_soc_fraction"].to_numpy(dtype=float),
            float(soc_pct) / 100.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Geisbauer storage SOC changed for cell {number}")
        initial_capacity = float(ordered.iloc[0]["capacity_ah"])
        expected_retention = (
            100.0 * ordered["capacity_ah"].to_numpy(dtype=float) / initial_capacity
        )
        if not np.allclose(
            ordered["capacity_retention_pct"].to_numpy(dtype=float),
            expected_retention,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Geisbauer within-cell capacity retention changed for cell {number}"
            )
        if not np.isclose(
            float(ordered.iloc[0]["capacity_retention_pct"]),
            100.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Geisbauer day-zero retention changed for cell {number}")
        if not np.isclose(
            float(ordered.iloc[0]["capacity_loss_pct"]),
            0.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Geisbauer day-zero loss changed for cell {number}")

    condition_counts = observations.groupby("condition_id")["cell_id"].nunique()
    if len(condition_counts) != 3 or not (condition_counts == 5).all():
        raise ValueError("Geisbauer canonical cohort must remain three groups of five cells")


def geisbauer_calendar_observations_sha256(observations: pd.DataFrame) -> str:
    """Return a row-order-invariant fingerprint of the canonical observation table."""
    validate_geisbauer_calendar_observations(observations)
    normalized = observations.sort_values(
        ["source_cell_number", "checkup_index"],
        kind="stable",
    )
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_geisbauer_calendar_observations(
    source: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the pinned Geisbauer LFP CSV or archive as a cell-level stress cohort."""
    source_path = Path(source)
    member, source_audit = _read_source_member(source_path)
    try:
        raw = pd.read_csv(io.BytesIO(member), encoding="utf-8-sig")
    except (UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError("Could not parse the pinned Geisbauer LFP CSV") from exc
    raw = _validate_raw_geisbauer_calendar(raw)

    source_cell_number = raw["Cell Identity Number"].astype("int64")
    soc_pct = raw["State-of-Charge"].astype("int64")
    elapsed_days = raw["Days Passed"].astype("int64")
    discharge_capacity = raw["Discharge Capacity"].astype(float)
    initial_discharge = discharge_capacity.groupby(source_cell_number).transform(
        "first"
    )

    observations = pd.DataFrame(
        {
            "dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
            "condition_id": soc_pct.map(_condition_id),
            "cell_id": source_cell_number.map(_cell_id),
            "test_id": source_cell_number.map(_test_id),
            "source_cell_id": source_cell_number.map(_source_cell_id),
            "source_cell_number": source_cell_number,
            "temperature_c": raw["Temperature"].astype(float),
            "storage_soc_fraction": soc_pct.astype(float) / 100.0,
            "elapsed_time_s": elapsed_days.astype(float) * 86_400.0,
            "elapsed_hours": elapsed_days.astype(float) * 24.0,
            "elapsed_days": elapsed_days.astype(float),
            "checkup_index": raw.groupby("Cell Identity Number", sort=False).cumcount(),
            "capacity_ah": discharge_capacity,
            "charge_capacity_ah": raw["Charge Capacity"].astype(float),
            "capacity_retention_pct": 100.0
            * discharge_capacity
            / initial_discharge,
        }
    )
    observations["capacity_loss_pct"] = (
        100.0 - observations["capacity_retention_pct"]
    )
    observations["physical_replicates_aggregated"] = 1
    observations["replicate_semantics"] = GEISBAUER_CALENDAR_REPLICATE_SEMANTICS
    observations["statistical_unit"] = GEISBAUER_CALENDAR_STATISTICAL_UNIT
    observations["evidence_role"] = GEISBAUER_CALENDAR_EVIDENCE_ROLE
    observations["long_term_validation_eligible"] = False
    observations["claim_boundary"] = GEISBAUER_CALENDAR_CLAIM_BOUNDARY
    observations["source_doi"] = GEISBAUER_CALENDAR_DOI
    observations["source_url"] = GEISBAUER_CALENDAR_SOURCE_URL
    observations["source_license"] = GEISBAUER_CALENDAR_LICENSE
    observations["source_license_url"] = GEISBAUER_CALENDAR_LICENSE_URL
    observations["source_version"] = GEISBAUER_CALENDAR_VERSION
    observations["source_archive_sha256"] = GEISBAUER_CALENDAR_ZIP_SHA256
    observations["source_member_sha256"] = GEISBAUER_CALENDAR_MEMBER_SHA256
    observations = observations[list(GEISBAUER_CALENDAR_OUTPUT_COLUMNS)].reset_index(
        drop=True
    )
    validate_geisbauer_calendar_observations(observations)
    fingerprint = geisbauer_calendar_observations_sha256(observations)

    audit: dict[str, object] = {
        "status": "passed",
        "dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
        "observation_count": len(observations),
        "physical_cell_count": int(observations["cell_id"].nunique()),
        "condition_count": int(observations["condition_id"].nunique()),
        "checkups_per_cell": sorted(
            observations.groupby("cell_id").size().unique().astype(int).tolist()
        ),
        "temperature_levels_c": [float(GEISBAUER_CALENDAR_EXPECTED_TEMPERATURE_C)],
        "storage_soc_levels": sorted(
            observations["storage_soc_fraction"].unique().astype(float).tolist()
        ),
        "elapsed_days": list(GEISBAUER_CALENDAR_EXPECTED_DAYS),
        "canonical_output_sha256": fingerprint,
        "source": {
            "doi": GEISBAUER_CALENDAR_DOI,
            "url": GEISBAUER_CALENDAR_SOURCE_URL,
            "license": GEISBAUER_CALENDAR_LICENSE,
            "license_url": GEISBAUER_CALENDAR_LICENSE_URL,
            "version": GEISBAUER_CALENDAR_VERSION,
            "archive_name": GEISBAUER_CALENDAR_ZIP_NAME,
            "archive_size_bytes": GEISBAUER_CALENDAR_ZIP_SIZE_BYTES,
            "archive_md5": GEISBAUER_CALENDAR_ZIP_MD5,
            "archive_sha256": GEISBAUER_CALENDAR_ZIP_SHA256,
            "member": GEISBAUER_CALENDAR_ZIP_MEMBER,
            "member_size_bytes": GEISBAUER_CALENDAR_MEMBER_SIZE_BYTES,
            "member_md5": GEISBAUER_CALENDAR_MEMBER_MD5,
            "member_sha256": GEISBAUER_CALENDAR_MEMBER_SHA256,
            **source_audit,
        },
        "guardrails": {
            "evidence_role": GEISBAUER_CALENDAR_EVIDENCE_ROLE,
            "statistical_unit": GEISBAUER_CALENDAR_STATISTICAL_UNIT,
            "long_term_validation_eligible": False,
            "confirmation_claim_allowed": False,
            "row_level_split_prohibited": True,
            "aggregate_cells_before_scoring": False,
        },
        "warning": (
            "This is a 120-day, 60 C accelerated external stress check with only "
            "three SOC conditions. It is never long-term external validation."
        ),
    }
    return observations, audit
