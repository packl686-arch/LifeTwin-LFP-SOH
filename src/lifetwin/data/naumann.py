from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NAUMANN_CALENDAR_DATASET_ID = "NAUMANN_LFP_CALENDAR_2021"
NAUMANN_CALENDAR_DOI = "10.17632/kxh42bfgtj.1"
NAUMANN_CALENDAR_SOURCE_URL = "https://data.mendeley.com/datasets/kxh42bfgtj/1"
NAUMANN_CALENDAR_LICENSE = "CC-BY-4.0"
NAUMANN_REPLICATE_SEMANTICS = "published_mean_of_3_physical_cells"
NAUMANN_STATISTICAL_UNIT = "temperature_soc_condition_mean_trajectory"

EXPECTED_CALENDAR_CONDITIONS = {
    (0.0, 0.5),
    (10.0, 0.5),
    (25.0, 0.0),
    (25.0, 0.5),
    (25.0, 1.0),
    (40.0, 0.0),
    (40.0, 0.125),
    (40.0, 0.25),
    (40.0, 0.375),
    (40.0, 0.5),
    (40.0, 0.625),
    (40.0, 0.75),
    (40.0, 0.875),
    (40.0, 1.0),
    (60.0, 0.0),
    (60.0, 0.5),
    (60.0, 1.0),
}

REQUIRED_CYCLE_SUMMARY_COLUMNS = {
    "test_id",
    "cell_id",
    "elapsed_time_s",
    "capacity_Ah",
    "capacity_retention_pct",
    "resistance_dc_ohm",
    "resistance_dc_pulse_duration_s",
    "resistance_dc_soc_pct",
}

REQUIRED_OBSERVATION_COLUMNS = {
    "dataset_id",
    "condition_id",
    "cell_id",
    "test_id",
    "temperature_c",
    "storage_soc_fraction",
    "elapsed_time_s",
    "elapsed_hours",
    "elapsed_days",
    "checkup_index",
    "capacity_ah",
    "capacity_retention_pct",
    "capacity_loss_pct",
    "resistance_dc_ohm",
    "resistance_growth_pct",
    "physical_replicates_aggregated",
    "replicate_semantics",
    "statistical_unit",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_bundle_sha256(paths: list[Path]) -> str:
    payload = [
        {
            "name": path.name,
            "sha256": _sha256(path),
        }
        for path in sorted(paths, key=lambda item: item.name)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cycle_summary(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported cycle-summary format: {path.suffix}")


def validate_naumann_calendar_observations(
    observations: pd.DataFrame,
    *,
    expected_condition_count: int | None = 17,
    expected_rows_per_condition: int | None = 35,
    require_published_grid: bool = True,
) -> None:
    """Validate condition-level calendar-aging trajectories without row leakage."""
    missing = sorted(REQUIRED_OBSERVATION_COLUMNS - set(observations.columns))
    if missing:
        raise ValueError(f"Missing Naumann calendar columns: {missing}")
    if observations.empty:
        raise ValueError("Naumann calendar observations cannot be empty")
    if observations[["condition_id", "test_id"]].isna().any().any():
        raise ValueError("Condition and test ids must be non-null")
    if observations.duplicated(["condition_id", "elapsed_time_s"]).any():
        raise ValueError("Duplicate condition/checkup observations are not allowed")
    if observations.duplicated(["condition_id", "checkup_index"]).any():
        raise ValueError("Duplicate condition/checkup indices are not allowed")
    if not (observations["condition_id"].astype(str) == observations["cell_id"].astype(str)).all():
        raise ValueError("Each logical cell_id must remain one condition-level trajectory")

    numeric_columns = [
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_time_s",
        "elapsed_hours",
        "elapsed_days",
        "capacity_ah",
        "capacity_retention_pct",
        "capacity_loss_pct",
        "resistance_dc_ohm",
        "resistance_growth_pct",
    ]
    numeric = observations[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Naumann calendar numeric values must be finite")
    if (numeric["elapsed_time_s"] < 0).any() or (numeric["elapsed_days"] < 0).any():
        raise ValueError("Elapsed time cannot be negative")
    if not np.allclose(
        numeric["elapsed_hours"].to_numpy(dtype=float),
        numeric["elapsed_time_s"].to_numpy(dtype=float) / 3600.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Elapsed-hour conversion is inconsistent with elapsed seconds")
    if not np.allclose(
        numeric["elapsed_days"].to_numpy(dtype=float),
        numeric["elapsed_time_s"].to_numpy(dtype=float) / 86400.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Elapsed-day conversion is inconsistent with elapsed seconds")
    if not np.allclose(
        numeric["capacity_loss_pct"].to_numpy(dtype=float),
        100.0 - numeric["capacity_retention_pct"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Capacity loss must equal 100 minus capacity retention")
    if not numeric["storage_soc_fraction"].between(0.0, 1.0).all():
        raise ValueError("Storage SOC must be expressed as a fraction in [0, 1]")
    if not numeric["temperature_c"].between(-40.0, 100.0).all():
        raise ValueError("Calendar-aging temperatures are outside the accepted range")
    if (numeric["capacity_ah"] <= 0).any() or (numeric["resistance_dc_ohm"] <= 0).any():
        raise ValueError("Capacity and resistance must be positive")

    dataset_ids = set(observations["dataset_id"].astype(str))
    if dataset_ids != {NAUMANN_CALENDAR_DATASET_ID}:
        raise ValueError(f"Unexpected Naumann dataset ids: {sorted(dataset_ids)}")
    if set(observations["replicate_semantics"].astype(str)) != {
        NAUMANN_REPLICATE_SEMANTICS
    }:
        raise ValueError("Replicate-mean semantics must remain explicit")
    if set(observations["statistical_unit"].astype(str)) != {
        NAUMANN_STATISTICAL_UNIT
    }:
        raise ValueError("The statistical unit must be the condition mean trajectory")
    if set(pd.to_numeric(observations["physical_replicates_aggregated"])) != {3}:
        raise ValueError("The public calendar trajectories aggregate three physical cells")

    grouped = observations.groupby("condition_id", sort=True)
    condition_count = grouped.ngroups
    if expected_condition_count is not None and condition_count != expected_condition_count:
        raise ValueError(
            f"Expected {expected_condition_count} calendar conditions, found {condition_count}"
        )
    row_counts = grouped.size()
    if expected_rows_per_condition is not None and not (
        row_counts == expected_rows_per_condition
    ).all():
        raise ValueError(
            "Unexpected checkup count by condition: "
            f"{row_counts.loc[row_counts != expected_rows_per_condition].to_dict()}"
        )

    condition_grid: set[tuple[float, float]] = set()
    for condition_id, condition in grouped:
        ordered = condition.sort_values("elapsed_time_s", kind="stable")
        elapsed = ordered["elapsed_time_s"].to_numpy(dtype=float)
        if elapsed[0] != 0.0 or np.any(np.diff(elapsed) <= 0):
            raise ValueError(f"Elapsed time must start at zero and increase for {condition_id}")
        checkup = pd.to_numeric(ordered["checkup_index"], errors="coerce").to_numpy(
            dtype=float
        )
        expected_checkup = np.arange(len(ordered), dtype=float)
        if not np.array_equal(checkup, expected_checkup):
            raise ValueError(
                "checkup_index must be the contiguous elapsed-time order for "
                f"{condition_id}"
            )
        if ordered["temperature_c"].nunique() != 1:
            raise ValueError(f"Temperature changes within condition {condition_id}")
        if ordered["storage_soc_fraction"].nunique() != 1:
            raise ValueError(f"Storage SOC changes within condition {condition_id}")
        initial = ordered.iloc[0]
        if not np.isclose(float(initial["capacity_retention_pct"]), 100.0, atol=1e-8):
            raise ValueError(f"Initial capacity retention is not 100% for {condition_id}")
        if not np.isclose(float(initial["capacity_loss_pct"]), 0.0, atol=1e-8):
            raise ValueError(f"Initial capacity loss is not zero for {condition_id}")
        condition_grid.add(
            (
                float(initial["temperature_c"]),
                float(initial["storage_soc_fraction"]),
            )
        )
    if len(condition_grid) != condition_count:
        raise ValueError("Temperature/SOC conditions must be unique")
    if require_published_grid and condition_grid != EXPECTED_CALENDAR_CONDITIONS:
        missing_grid = sorted(EXPECTED_CALENDAR_CONDITIONS - condition_grid)
        extra_grid = sorted(condition_grid - EXPECTED_CALENDAR_CONDITIONS)
        raise ValueError(
            f"Published condition grid mismatch: missing={missing_grid}, extra={extra_grid}"
        )


def load_naumann_calendar_observations(
    celljar_repository: str | Path,
    cycle_summary: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the canonical 17-condition Naumann calendar-aging observation table."""
    repository = Path(celljar_repository)
    cycle_summary_path = Path(cycle_summary)
    cell_directory = repository / "cells"
    test_directory = repository / "tests"
    if not cell_directory.is_dir() or not test_directory.is_dir():
        raise FileNotFoundError(f"Expected cells/ and tests/ under {repository}")
    if not cycle_summary_path.is_file():
        raise FileNotFoundError(f"Cycle summary does not exist: {cycle_summary_path}")

    test_paths = sorted(test_directory.glob("NAUMANN_CAL_*_TEST.json"))
    if len(test_paths) != 17:
        raise ValueError(f"Expected 17 Naumann calendar test files, found {len(test_paths)}")
    metadata_rows: list[dict[str, object]] = []
    metadata_paths: list[Path] = []
    for test_path in test_paths:
        test = _read_json(test_path)
        if test.get("test_type") != "calendar_aging":
            raise ValueError(f"Unexpected test type in {test_path.name}")
        cell_id = str(test["cell_id"])
        cell_path = cell_directory / f"{cell_id}.json"
        if not cell_path.is_file():
            raise FileNotFoundError(f"Missing cell metadata for {cell_id}")
        cell = _read_json(cell_path)
        description = str(test.get("protocol_description") or "")
        if "aggregated mean across 3 replicates" not in description:
            raise ValueError(f"Replicate aggregation is not explicit for {cell_id}")
        if test.get("source_doi") != NAUMANN_CALENDAR_DOI:
            raise ValueError(f"Unexpected calendar-aging DOI for {cell_id}")
        temperature_min = float(test["temperature_C_min"])
        temperature_max = float(test["temperature_C_max"])
        soc_min = float(test["soc_range_min"])
        soc_max = float(test["soc_range_max"])
        if temperature_min != temperature_max or soc_min != soc_max:
            raise ValueError(f"Condition is not fixed for {cell_id}")
        metadata_rows.append(
            {
                "test_id": str(test["test_id"]),
                "cell_id": cell_id,
                "condition_id": cell_id,
                "source_cell_id": str(cell["source_cell_id"]),
                "temperature_c": temperature_min,
                "storage_soc_fraction": soc_min,
                "nominal_capacity_ah": float(cell["nominal_capacity_Ah"]),
                "expected_checkups": int(test["num_cycles"]),
                "declared_duration_s": float(test["duration_s"]),
                "source_doi": str(test["source_doi"]),
                "source_url": str(test["source_url"]),
                "source_license": str(test["source_license"]),
                "source_license_url": str(test["source_license_url"]),
            }
        )
        metadata_paths.extend([test_path, cell_path])

    metadata = pd.DataFrame(metadata_rows)
    if metadata["test_id"].duplicated().any() or metadata["cell_id"].duplicated().any():
        raise ValueError("Naumann calendar metadata ids must be unique")
    grid = set(
        metadata[["temperature_c", "storage_soc_fraction"]].itertuples(
            index=False,
            name=None,
        )
    )
    if grid != EXPECTED_CALENDAR_CONDITIONS:
        raise ValueError("Naumann calendar metadata does not match the published grid")

    raw = _read_cycle_summary(cycle_summary_path)
    missing_columns = sorted(REQUIRED_CYCLE_SUMMARY_COLUMNS - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing cycle-summary columns: {missing_columns}")
    raw = raw.loc[raw["test_id"].isin(metadata["test_id"])].copy()
    observed_tests = set(raw["test_id"].astype(str))
    expected_tests = set(metadata["test_id"].astype(str))
    if observed_tests != expected_tests:
        raise ValueError(
            "Calendar test coverage mismatch: "
            f"missing={sorted(expected_tests - observed_tests)}, "
            f"extra={sorted(observed_tests - expected_tests)}"
        )
    raw = raw.merge(metadata, on=["test_id", "cell_id"], validate="many_to_one")
    counts = raw.groupby("test_id").size()
    expected_counts = metadata.set_index("test_id")["expected_checkups"]
    if not counts.equals(expected_counts.loc[counts.index]):
        raise ValueError("Cycle-summary checkup counts disagree with test metadata")

    raw = raw.sort_values(["condition_id", "elapsed_time_s"], kind="stable")
    raw["checkup_index"] = raw.groupby("condition_id", sort=False).cumcount()
    raw["elapsed_hours"] = pd.to_numeric(raw["elapsed_time_s"]) / 3600.0
    raw["elapsed_days"] = pd.to_numeric(raw["elapsed_time_s"]) / 86400.0
    raw["capacity_loss_pct"] = 100.0 - pd.to_numeric(
        raw["capacity_retention_pct"]
    )
    initial_resistance = raw.groupby("condition_id", sort=False)[
        "resistance_dc_ohm"
    ].transform("first")
    raw["resistance_growth_pct"] = (
        100.0 * (pd.to_numeric(raw["resistance_dc_ohm"]) / initial_resistance - 1.0)
    )
    raw["dataset_id"] = NAUMANN_CALENDAR_DATASET_ID
    raw["physical_replicates_aggregated"] = 3
    raw["replicate_semantics"] = NAUMANN_REPLICATE_SEMANTICS
    raw["statistical_unit"] = NAUMANN_STATISTICAL_UNIT
    raw["evidence_role"] = "public_calendar_aging_method_validation"

    output_columns = [
        "dataset_id",
        "condition_id",
        "cell_id",
        "test_id",
        "source_cell_id",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_time_s",
        "elapsed_hours",
        "elapsed_days",
        "checkup_index",
        "capacity_Ah",
        "capacity_retention_pct",
        "capacity_loss_pct",
        "resistance_dc_ohm",
        "resistance_growth_pct",
        "resistance_dc_pulse_duration_s",
        "resistance_dc_soc_pct",
        "nominal_capacity_ah",
        "physical_replicates_aggregated",
        "replicate_semantics",
        "statistical_unit",
        "evidence_role",
        "source_doi",
        "source_url",
        "source_license",
        "source_license_url",
    ]
    observations = raw[output_columns].rename(columns={"capacity_Ah": "capacity_ah"})
    observations = observations.reset_index(drop=True)
    validate_naumann_calendar_observations(observations)

    horizons = observations.groupby("condition_id")["elapsed_days"].max()
    audit: dict[str, object] = {
        "status": "passed",
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "observation_count": len(observations),
        "condition_count": int(observations["condition_id"].nunique()),
        "checkups_per_condition": sorted(
            observations.groupby("condition_id").size().unique().astype(int).tolist()
        ),
        "physical_replicates_per_condition": 3,
        "replicate_count": 3,
        "logical_trajectory_count": 17,
        "effective_n": 17,
        "physical_cell_count_in_source_experiment": 51,
        "aggregation_unit": "condition_mean",
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "replicate_semantics": NAUMANN_REPLICATE_SEMANTICS,
        "temperature_levels_c": sorted(
            observations["temperature_c"].unique().astype(float).tolist()
        ),
        "storage_soc_levels": sorted(
            observations["storage_soc_fraction"].unique().astype(float).tolist()
        ),
        "minimum_horizon_days": float(horizons.min()),
        "maximum_horizon_days": float(horizons.max()),
        "capacity_retention_range_pct": [
            float(observations["capacity_retention_pct"].min()),
            float(observations["capacity_retention_pct"].max()),
        ],
        "source": {
            "doi": NAUMANN_CALENDAR_DOI,
            "url": NAUMANN_CALENDAR_SOURCE_URL,
            "license": NAUMANN_CALENDAR_LICENSE,
            "cycle_summary_path": str(cycle_summary_path.resolve()),
            "cycle_summary_sha256": _sha256(cycle_summary_path),
            "celljar_metadata_bundle_sha256": _metadata_bundle_sha256(metadata_paths),
            "metadata_file_count": len(metadata_paths),
        },
        "guardrails": {
            "independent_units": 17,
            "row_level_split_prohibited": True,
            "row_level_bootstrap_prohibited": True,
            "treat_51_physical_replicates_as_observed_cells": False,
            "projection_beyond_observed_horizon_allowed": False,
        },
        "warning": (
            "These 17 trajectories are published condition means across three cells. "
            "They validate a public-data calendar-aging method, not Hithium product "
            "accuracy or 15-25 year extrapolation."
        ),
    }
    return observations, audit
