from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ATTIA_DATASET_ID = "ATTIA_2020_VALIDATION45"
ATTIA_CAMPAIGN_ID = "ATTIA_VALIDATION_2019_01_24"
ATTIA_CAMPAIGN_START_DATE = "2019-01-24"
ATTIA_CELL_PACKING_DATE = "2015-09-26"
ATTIA_DOI = "10.1038/s41586-020-1994-5"
ATTIA_SOURCE_URL = "https://data.matr.io/1/projects/5d80e633f405260001c0b60a"
ATTIA_AUTHOR_CODE_COMMIT = "0068fd0136bcd65884f5cd94b2b967c1ba73a668"
ATTIA_CELLJAR_COMMIT = "5c9601a027751c84ee8346f7c0ab9c6851330202"
ATTIA_TARGET_TIMESERIES_SHA256 = (
    "e27757440a4e476bc0208a2e55c94244f13d83fe0850fe9a8f1cf08aeb918ac7"
)
ATTIA_DATASET_SNAPSHOT_ID = (
    f"celljar_{ATTIA_CELLJAR_COMMIT}_timeseries_{ATTIA_TARGET_TIMESERIES_SHA256}"
)
ATTIA_LABEL_VERSION = (
    f"attia_final_results_commit_{ATTIA_AUTHOR_CODE_COMMIT}"
)
ATTIA_OUTCOME_SCHEMA_VERSION = "attia_validation45_outcomes_v2"
ATTIA_FINAL_RESULTS_SHA256 = (
    "94359f14e0066e1464e0e06d0ed3c4ceaccc8ea779b2fd492509cea02a22e5bb"
)
ATTIA_FINAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/chueh-ermon/"
    "battery-fast-charging-optimization/"
    f"{ATTIA_AUTHOR_CODE_COMMIT}/figures/fig4/final_results.csv"
)
ATTIA_AUTHORITY_STATUS = "authoritative_source_derived"
ATTIA_CROSSWALK_METHOD = (
    "celljar_source_id_plus_unique_cc1_cc2_cc3_cycle_life_tuple_to_author_final_results"
)
ATTIA_LABEL_STRATEGY = "author_final_results_cycle_life"
ATTIA_CYCLE_LIFE_DEFINITION = (
    "number of cycles until discharge capacity falls below 0.88 Ah "
    "(80% of 1.1 Ah nominal capacity)"
)

EXPECTED_CELL_IDS = {f"CLO_B4C{index}" for index in range(45)}
EXPECTED_AUTHOR_COLUMNS = ["C1", "C2", "C3", "R1", "R2", "R3", "R4", "R5"]

REQUIRED_TEST_KEYS = {
    "test_id",
    "cell_id",
    "test_type",
    "temperature_C_min",
    "temperature_C_max",
    "c_rate_discharge",
    "protocol_description",
    "num_cycles",
    "source_doi",
    "source_url",
    "source_license",
    "source_license_url",
}

REQUIRED_CELL_KEYS = {
    "cell_id",
    "source",
    "source_cell_id",
    "manufacturer",
    "model_number",
    "chemistry",
    "nominal_capacity_Ah",
    "nominal_voltage_V",
    "max_voltage_V",
    "min_voltage_V",
}

ATTIA_OUTCOME_COLUMNS = (
    "dataset_id",
    "dataset_snapshot_id",
    "campaign_id",
    "campaign_start_date",
    "cell_packing_date",
    "cell_id",
    "source_cell_id",
    "test_id",
    "protocol_id",
    "charge_cc1_c",
    "charge_cc2_c",
    "charge_cc3_c",
    "charge_cc4_c",
    "replicate_id",
    "cycle_life",
    "event_observed",
    "is_censored",
    "eol_soh_fraction",
    "eol_capacity_ah",
    "cycle_life_definition",
    "label_version",
    "label_strategy",
    "label_source_url",
    "author_code_commit",
    "author_final_results_sha256",
    "authority_status",
    "direct_author_cell_id_assertion",
    "crosswalk_method",
    "outcome_schema_version",
    "celljar_commit",
    "official_validation_cohort",
    "temperature_c",
    "discharge_c_rate",
    "nominal_capacity_ah",
    "nominal_voltage_v",
    "minimum_voltage_v",
    "maximum_voltage_v",
    "manufacturer",
    "model_number",
    "chemistry",
    "source_doi",
    "source_url",
    "source_license",
    "source_license_url",
)
REQUIRED_OUTCOME_COLUMNS = set(ATTIA_OUTCOME_COLUMNS)

FORBIDDEN_OUTCOME_COLUMNS = {
    "max_cycle",
    "max_cycle_number",
    "recorded_max_cycle",
    "source_max_cycle",
}

_PROTOCOL_PATTERN = re.compile(
    r"Charge policy \(BO-selected\):\s*"
    r"([0-9]+(?:\.[0-9]+)?)-"
    r"([0-9]+(?:\.[0-9]+)?)-"
    r"([0-9]+(?:\.[0-9]+)?)-"
    r"([0-9]+(?:\.[0-9]+)?)\."
)
_CELL_ID_PATTERN = re.compile(r"CLO_B4C([0-9]+)\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Attia metadata JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Attia metadata must be a JSON object: {path}")
    return value


def _require_keys(value: dict[str, Any], required: set[str], path: Path) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"Attia metadata missing required keys in {path.name}: {missing}"
        )


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"{field} must be a positive integer")
    return int(numeric)


def _canonical_rate(value: Any, *, field: str) -> tuple[float, str]:
    try:
        decimal = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite C-rate") from exc
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError(f"{field} must be a positive finite C-rate")
    canonical = format(decimal.normalize(), "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    return float(decimal), canonical


def _parse_protocol(description: str, *, source: str) -> dict[str, object]:
    matches = list(_PROTOCOL_PATTERN.finditer(description))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one BO-selected CC1-CC4 protocol in {source}"
        )
    numeric: list[float] = []
    canonical: list[str] = []
    for index, raw in enumerate(matches[0].groups(), start=1):
        rate, text = _canonical_rate(raw, field=f"CC{index}")
        numeric.append(rate)
        canonical.append(text)
    return {
        "charge_cc1_c": numeric[0],
        "charge_cc2_c": numeric[1],
        "charge_cc3_c": numeric[2],
        "charge_cc4_c": numeric[3],
        "protocol_prefix_key": "|".join(canonical[:3]),
        "protocol_id": "-".join(canonical),
    }


def _cell_order(cell_id: str) -> int:
    match = _CELL_ID_PATTERN.fullmatch(cell_id)
    if match is None:
        raise ValueError(f"Unexpected Attia cell id: {cell_id}")
    return int(match.group(1))


def _metadata_bundle_sha256(repository: Path, paths: list[Path]) -> str:
    payload = [
        {
            "path": path.relative_to(repository).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_attia_author_results(
    final_results: str | Path,
) -> tuple[pd.DataFrame, str]:
    """Load and explode the pinned author result table after byte-level validation."""
    path = Path(final_results)
    if not path.is_file():
        raise FileNotFoundError(f"Attia final-results file does not exist: {path}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != ATTIA_FINAL_RESULTS_SHA256:
        raise ValueError(
            "Attia final-results SHA-256 mismatch: "
            f"expected {ATTIA_FINAL_RESULTS_SHA256}, found {observed_sha256}"
        )

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Attia final-results file is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != EXPECTED_AUTHOR_COLUMNS:
        raise ValueError(
            "Unexpected Attia final-results columns: "
            f"expected {EXPECTED_AUTHOR_COLUMNS}, found {reader.fieldnames}"
        )
    source_rows = list(reader)
    if len(source_rows) != 9:
        raise ValueError(f"Expected 9 author protocol rows, found {len(source_rows)}")

    exploded: list[dict[str, object]] = []
    protocol_keys: list[str] = []
    for row_index, row in enumerate(source_rows, start=1):
        rates: list[float] = []
        canonical: list[str] = []
        for column in ("C1", "C2", "C3"):
            rate, text_rate = _canonical_rate(
                row.get(column),
                field=f"author row {row_index} {column}",
            )
            rates.append(rate)
            canonical.append(text_rate)
        protocol_key = "|".join(canonical)
        protocol_keys.append(protocol_key)
        for replicate_number in range(1, 6):
            replicate_id = f"R{replicate_number}"
            cycle_life = _positive_integer(
                row.get(replicate_id),
                field=f"author row {row_index} {replicate_id}",
            )
            exploded.append(
                {
                    "protocol_prefix_key": protocol_key,
                    "charge_cc1_c_author": rates[0],
                    "charge_cc2_c_author": rates[1],
                    "charge_cc3_c_author": rates[2],
                    "replicate_id": replicate_id,
                    "cycle_life": cycle_life,
                    "author_crosswalk_key": f"{protocol_key}|{cycle_life}",
                }
            )

    if len(set(protocol_keys)) != 9:
        raise ValueError("Attia author protocol keys must be unique")
    author = pd.DataFrame(exploded)
    if len(author) != 45:
        raise ValueError(f"Expected 45 exploded author outcomes, found {len(author)}")
    if author["author_crosswalk_key"].duplicated().any():
        duplicates = sorted(
            author.loc[
                author["author_crosswalk_key"].duplicated(keep=False),
                "author_crosswalk_key",
            ].unique()
        )
        raise ValueError(f"Duplicate Attia author outcome keys: {duplicates}")
    return author, observed_sha256


def _load_celljar_metadata(repository: Path) -> tuple[pd.DataFrame, list[Path]]:
    cell_directory = repository / "cells"
    test_directory = repository / "tests"
    if not cell_directory.is_dir() or not test_directory.is_dir():
        raise FileNotFoundError(f"Expected cells/ and tests/ under {repository}")

    test_paths = sorted(test_directory.glob("CLO_B4C*_CYCLING.json"))
    cell_paths = sorted(cell_directory.glob("CLO_B4C*.json"))
    if len(test_paths) != 45:
        raise ValueError(f"Expected 45 CLO test JSON files, found {len(test_paths)}")
    if len(cell_paths) != 45:
        raise ValueError(f"Expected 45 CLO cell JSON files, found {len(cell_paths)}")

    test_records: list[dict[str, object]] = []
    for path in test_paths:
        test = _read_json(path)
        _require_keys(test, REQUIRED_TEST_KEYS, path)
        test_records.append(
            {
                "test_path": path,
                "test_id": str(test["test_id"]),
                "cell_id": str(test["cell_id"]),
                "test": test,
            }
        )
    test_index = pd.DataFrame(test_records)
    if test_index["test_id"].duplicated().any():
        raise ValueError("Duplicate Attia test ids are not allowed")
    if test_index["cell_id"].duplicated().any():
        raise ValueError("Duplicate Attia test-to-cell keys are not allowed")
    observed_test_cells = set(test_index["cell_id"])
    if observed_test_cells != EXPECTED_CELL_IDS:
        raise ValueError(
            "Attia test-cell coverage mismatch: "
            f"missing={sorted(EXPECTED_CELL_IDS - observed_test_cells)}, "
            f"extra={sorted(observed_test_cells - EXPECTED_CELL_IDS)}"
        )

    cell_records: list[dict[str, object]] = []
    for path in cell_paths:
        cell = _read_json(path)
        _require_keys(cell, REQUIRED_CELL_KEYS, path)
        cell_records.append(
            {
                "cell_path": path,
                "cell_id": str(cell["cell_id"]),
                "cell": cell,
            }
        )
    cell_index = pd.DataFrame(cell_records)
    if cell_index["cell_id"].duplicated().any():
        raise ValueError("Duplicate Attia cell ids are not allowed")
    observed_cells = set(cell_index["cell_id"])
    if observed_cells != EXPECTED_CELL_IDS:
        raise ValueError(
            "Attia cell metadata coverage mismatch: "
            f"missing={sorted(EXPECTED_CELL_IDS - observed_cells)}, "
            f"extra={sorted(observed_cells - EXPECTED_CELL_IDS)}"
        )

    joined = test_index.merge(cell_index, on="cell_id", validate="one_to_one")
    rows: list[dict[str, object]] = []
    paths: list[Path] = []
    for record in joined.itertuples(index=False):
        test_path = Path(record.test_path)
        cell_path = Path(record.cell_path)
        test = record.test
        cell = record.cell
        cell_id = str(record.cell_id)
        cell_number = _cell_order(cell_id)
        expected_test_id = f"{cell_id}_CYCLING"
        if str(test["test_id"]) != expected_test_id:
            raise ValueError(f"Test id does not match cell id for {cell_id}")
        if test_path.name != f"{expected_test_id}.json":
            raise ValueError(f"Test filename does not match test id for {cell_id}")
        if cell_path.name != f"{cell_id}.json":
            raise ValueError(f"Cell filename does not match cell id for {cell_id}")
        if str(cell["source_cell_id"]) != f"b4c{cell_number}":
            raise ValueError(f"Unexpected source cell id for {cell_id}")
        if str(cell["source"]) != "CLO":
            raise ValueError(f"Unexpected source dataset for {cell_id}")
        if str(test["test_type"]) != "cycle_aging":
            raise ValueError(f"Unexpected test type for {cell_id}")
        if str(test["source_doi"]) != ATTIA_DOI:
            raise ValueError(f"Unexpected source DOI for {cell_id}")
        if str(test["source_url"]) != ATTIA_SOURCE_URL:
            raise ValueError(f"Unexpected source URL for {cell_id}")
        if str(cell["manufacturer"]) != "A123 Systems":
            raise ValueError(f"Unexpected cell manufacturer for {cell_id}")
        if str(cell["model_number"]) != "APR18650M1A":
            raise ValueError(f"Unexpected cell model for {cell_id}")
        if str(cell["chemistry"]) != "LFP":
            raise ValueError(f"Unexpected cell chemistry for {cell_id}")

        temperature_min = float(test["temperature_C_min"])
        temperature_max = float(test["temperature_C_max"])
        discharge_rate = float(test["c_rate_discharge"])
        nominal_capacity = float(cell["nominal_capacity_Ah"])
        if not (
            temperature_min == temperature_max == 30.0
            and discharge_rate == 4.0
            and nominal_capacity == 1.1
        ):
            raise ValueError(f"Unexpected Attia test condition for {cell_id}")
        protocol = _parse_protocol(
            str(test["protocol_description"]),
            source=test_path.name,
        )
        local_num_cycles = _positive_integer(
            test["num_cycles"],
            field=f"{cell_id} num_cycles crosswalk value",
        )
        rows.append(
            {
                "cell_order": cell_number,
                "cell_id": cell_id,
                "source_cell_id": str(cell["source_cell_id"]),
                "test_id": expected_test_id,
                "local_num_cycles_crosswalk_value": local_num_cycles,
                "local_crosswalk_key": (
                    f"{protocol['protocol_prefix_key']}|{local_num_cycles}"
                ),
                "temperature_c": temperature_min,
                "discharge_c_rate": discharge_rate,
                "nominal_capacity_ah": nominal_capacity,
                "nominal_voltage_v": float(cell["nominal_voltage_V"]),
                "maximum_voltage_v": float(cell["max_voltage_V"]),
                "minimum_voltage_v": float(cell["min_voltage_V"]),
                "manufacturer": str(cell["manufacturer"]),
                "model_number": str(cell["model_number"]),
                "chemistry": str(cell["chemistry"]),
                "source_doi": str(test["source_doi"]),
                "source_url": str(test["source_url"]),
                "source_license": str(test["source_license"]),
                "source_license_url": str(test["source_license_url"]),
                **protocol,
            }
        )
        paths.extend([test_path, cell_path])

    metadata = pd.DataFrame(rows)
    if metadata["local_crosswalk_key"].duplicated().any():
        duplicates = sorted(
            metadata.loc[
                metadata["local_crosswalk_key"].duplicated(keep=False),
                "local_crosswalk_key",
            ].unique()
        )
        raise ValueError(f"Duplicate Attia local crosswalk keys: {duplicates}")
    return metadata, paths


def validate_attia_outcome_pack(outcomes: pd.DataFrame) -> None:
    """Validate the 45-cell observed-event outcome contract."""
    missing = sorted(REQUIRED_OUTCOME_COLUMNS - set(outcomes.columns))
    if missing:
        raise ValueError(f"Missing Attia outcome columns: {missing}")
    forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS & set(outcomes.columns))
    if forbidden:
        raise ValueError(
            "Maximum recorded cycle is audit metadata, not an Attia outcome label: "
            f"{forbidden}"
        )
    unexpected = sorted(set(outcomes.columns) - REQUIRED_OUTCOME_COLUMNS)
    if unexpected:
        raise ValueError(f"Unexpected Attia outcome columns: {unexpected}")
    if len(outcomes) != 45:
        raise ValueError(f"Expected 45 Attia outcomes, found {len(outcomes)}")
    if outcomes[list(ATTIA_OUTCOME_COLUMNS)].isna().any().any():
        raise ValueError("Attia outcome pack cannot contain null values")
    for column in ("cell_id", "source_cell_id", "test_id"):
        if outcomes[column].isna().any() or outcomes[column].duplicated().any():
            raise ValueError(
                f"Attia outcome {column} values must be unique and non-null"
            )
    if set(outcomes["cell_id"].astype(str)) != EXPECTED_CELL_IDS:
        raise ValueError("Attia outcome pack does not contain the expected 45 cells")

    expected_constants: dict[str, object] = {
        "dataset_id": ATTIA_DATASET_ID,
        "dataset_snapshot_id": ATTIA_DATASET_SNAPSHOT_ID,
        "campaign_id": ATTIA_CAMPAIGN_ID,
        "event_observed": True,
        "is_censored": False,
        "eol_soh_fraction": 0.8,
        "eol_capacity_ah": 0.88,
        "cycle_life_definition": ATTIA_CYCLE_LIFE_DEFINITION,
        "label_version": ATTIA_LABEL_VERSION,
        "label_strategy": ATTIA_LABEL_STRATEGY,
        "label_source_url": ATTIA_FINAL_RESULTS_URL,
        "author_code_commit": ATTIA_AUTHOR_CODE_COMMIT,
        "author_final_results_sha256": ATTIA_FINAL_RESULTS_SHA256,
        "authority_status": ATTIA_AUTHORITY_STATUS,
        "direct_author_cell_id_assertion": False,
        "crosswalk_method": ATTIA_CROSSWALK_METHOD,
        "outcome_schema_version": ATTIA_OUTCOME_SCHEMA_VERSION,
        "celljar_commit": ATTIA_CELLJAR_COMMIT,
        "official_validation_cohort": True,
    }
    for column, expected in expected_constants.items():
        values = set(outcomes[column].tolist())
        if values != {expected}:
            raise ValueError(f"Unexpected Attia outcome values for {column}: {values}")

    numeric_columns = [
        "charge_cc1_c",
        "charge_cc2_c",
        "charge_cc3_c",
        "charge_cc4_c",
        "cycle_life",
    ]
    numeric = outcomes[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Attia outcome numeric values must be finite")
    if (numeric <= 0).any().any():
        raise ValueError("Attia outcome numeric values must be positive")
    life = numeric["cycle_life"]
    if not np.allclose(life, np.round(life), rtol=0.0, atol=0.0):
        raise ValueError("Attia cycle-life outcomes must be integer cycle counts")

    protocol_counts = outcomes.groupby("protocol_id").size()
    if len(protocol_counts) != 9 or set(protocol_counts.astype(int)) != {5}:
        raise ValueError(
            "Attia outcomes must contain nine protocols with five cells each"
        )
    for protocol_id, protocol in outcomes.groupby("protocol_id", sort=True):
        if set(protocol["replicate_id"].astype(str)) != {
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
        }:
            raise ValueError(f"Unexpected replicate ids for protocol {protocol_id}")
    crosswalk_columns = [
        "charge_cc1_c",
        "charge_cc2_c",
        "charge_cc3_c",
        "cycle_life",
    ]
    if outcomes.duplicated(crosswalk_columns).any():
        raise ValueError("Attia authoritative protocol/life keys must be unique")


def attia_outcome_artifact_sha256(outcomes: pd.DataFrame) -> str:
    """Hash the validated outcome content independently of row and CSV order."""
    validate_attia_outcome_pack(outcomes)
    normalized = outcomes[sorted(ATTIA_OUTCOME_COLUMNS)].sort_values(
        "cell_id",
        kind="stable",
    )
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_attia_outcome_pack(
    celljar_repository: str | Path,
    author_final_results: str | Path,
    *,
    label_strategy: str = ATTIA_LABEL_STRATEGY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build source-authoritative outcomes without exposing them to feature code."""
    if label_strategy != ATTIA_LABEL_STRATEGY:
        raise ValueError(
            "Attia labels must come from the pinned author final-results table; "
            "maximum recorded cycle labels are prohibited"
        )
    repository = Path(celljar_repository)
    author_path = Path(author_final_results)
    metadata, metadata_paths = _load_celljar_metadata(repository)
    author, observed_author_sha256 = load_attia_author_results(author_path)

    local_keys = set(metadata["local_crosswalk_key"].astype(str))
    author_keys = set(author["author_crosswalk_key"].astype(str))
    if local_keys != author_keys:
        raise ValueError(
            "Attia authoritative crosswalk mismatch: "
            f"local_without_author={sorted(local_keys - author_keys)}, "
            f"author_without_local={sorted(author_keys - local_keys)}. "
            "CellJar num_cycles is a crosswalk value only; recorded max cycle "
            "must never replace the author cycle-life label."
        )

    joined = metadata.merge(
        author,
        left_on="local_crosswalk_key",
        right_on="author_crosswalk_key",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 45:
        raise ValueError(f"Expected 45 matched Attia outcomes, found {len(joined)}")
    for column in ("charge_cc1_c", "charge_cc2_c", "charge_cc3_c"):
        author_column = f"{column}_author"
        if not np.allclose(
            joined[column].to_numpy(dtype=float),
            joined[author_column].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"Author and CellJar protocol rates disagree for {column}")

    joined["dataset_id"] = ATTIA_DATASET_ID
    joined["dataset_snapshot_id"] = ATTIA_DATASET_SNAPSHOT_ID
    joined["campaign_id"] = ATTIA_CAMPAIGN_ID
    joined["campaign_start_date"] = ATTIA_CAMPAIGN_START_DATE
    joined["cell_packing_date"] = ATTIA_CELL_PACKING_DATE
    joined["event_observed"] = True
    joined["is_censored"] = False
    joined["eol_soh_fraction"] = 0.8
    joined["eol_capacity_ah"] = 0.88
    joined["cycle_life_definition"] = ATTIA_CYCLE_LIFE_DEFINITION
    joined["label_version"] = ATTIA_LABEL_VERSION
    joined["label_strategy"] = ATTIA_LABEL_STRATEGY
    joined["label_source_url"] = ATTIA_FINAL_RESULTS_URL
    joined["author_code_commit"] = ATTIA_AUTHOR_CODE_COMMIT
    joined["author_final_results_sha256"] = observed_author_sha256
    joined["authority_status"] = ATTIA_AUTHORITY_STATUS
    joined["direct_author_cell_id_assertion"] = False
    joined["crosswalk_method"] = ATTIA_CROSSWALK_METHOD
    joined["outcome_schema_version"] = ATTIA_OUTCOME_SCHEMA_VERSION
    joined["celljar_commit"] = ATTIA_CELLJAR_COMMIT
    joined["official_validation_cohort"] = True

    outcomes = joined.sort_values("cell_order", kind="stable")[
        list(ATTIA_OUTCOME_COLUMNS)
    ]
    outcomes = outcomes.reset_index(drop=True)
    validate_attia_outcome_pack(outcomes)

    protocol_life = outcomes.groupby("protocol_id")["cycle_life"].agg(
        ["count", "mean", "min", "max"]
    )
    audit: dict[str, object] = {
        "status": "passed",
        "dataset_id": ATTIA_DATASET_ID,
        "campaign_id": ATTIA_CAMPAIGN_ID,
        "dataset_snapshot_id": ATTIA_DATASET_SNAPSHOT_ID,
        "label_version": ATTIA_LABEL_VERSION,
        "outcome_schema_version": ATTIA_OUTCOME_SCHEMA_VERSION,
        "cell_count": len(outcomes),
        "protocol_count": int(outcomes["protocol_id"].nunique()),
        "replicates_per_protocol": [
            int(value) for value in sorted(protocol_life["count"].astype(int).unique())
        ],
        "event_observed_count": int(outcomes["event_observed"].sum()),
        "right_censored_count": int(outcomes["is_censored"].sum()),
        "cycle_life_range": [
            int(outcomes["cycle_life"].min()),
            int(outcomes["cycle_life"].max()),
        ],
        "cycle_life_median": float(outcomes["cycle_life"].median()),
        "canonical_outcome_sha256": attia_outcome_artifact_sha256(outcomes),
        "crosswalk": {
            "method": ATTIA_CROSSWALK_METHOD,
            "author_protocol_rows": 9,
            "author_exploded_rows": len(author),
            "local_metadata_rows": len(metadata),
            "matched_rows": len(outcomes),
            "join_key": ["CC1", "CC2", "CC3", "cycle_life"],
            "celljar_num_cycles_role": "crosswalk_key_only_not_label_source",
            "direct_author_cell_id_assertion": False,
        },
        "source": {
            "doi": ATTIA_DOI,
            "dataset_url": ATTIA_SOURCE_URL,
            "author_final_results_path": str(author_path.resolve()),
            "author_final_results_url": ATTIA_FINAL_RESULTS_URL,
            "author_code_commit": ATTIA_AUTHOR_CODE_COMMIT,
            "author_final_results_sha256": observed_author_sha256,
            "celljar_repository": str(repository.resolve()),
            "celljar_metadata_bundle_sha256": _metadata_bundle_sha256(
                repository,
                metadata_paths,
            ),
            "metadata_file_count": len(metadata_paths),
        },
        "guardrails": {
            "label_strategy": ATTIA_LABEL_STRATEGY,
            "maximum_recorded_cycle_as_label_prohibited": True,
            "feature_pack_must_exclude_outcomes": True,
            "all_events_source_authoritative": True,
            "celljar_test_year_is_authoritative": False,
            "celljar_generic_censoring_description_is_authoritative": False,
        },
        "warning": (
            "This is a 45-cell, nine-protocol, all-observed validation campaign. "
            "It does not validate right-censoring behavior, independent-lab transfer, "
            "Hithium product accuracy, or 15-25 year storage extrapolation."
        ),
    }
    return outcomes, audit


def load_attia_label_free_metadata(
    celljar_repository: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the CLO test-to-cell map without reading target outcome values."""
    repository = Path(celljar_repository)
    cell_directory = repository / "cells"
    test_directory = repository / "tests"
    if not cell_directory.is_dir() or not test_directory.is_dir():
        raise FileNotFoundError(f"Expected cells/ and tests/ under {repository}")
    test_paths = sorted(test_directory.glob("CLO_B4C*_CYCLING.json"))
    if len(test_paths) != 45:
        raise ValueError(f"Expected 45 CLO test JSON files, found {len(test_paths)}")

    rows: list[dict[str, object]] = []
    metadata_paths: list[Path] = []
    for test_path in test_paths:
        test = _read_json(test_path)
        safe_required = REQUIRED_TEST_KEYS - {"num_cycles"}
        _require_keys(test, safe_required, test_path)
        cell_id = str(test["cell_id"])
        cell_path = cell_directory / f"{cell_id}.json"
        if not cell_path.is_file():
            raise FileNotFoundError(f"Missing Attia cell metadata for {cell_id}")
        cell = _read_json(cell_path)
        _require_keys(cell, REQUIRED_CELL_KEYS, cell_path)
        expected_test_id = f"{cell_id}_CYCLING"
        if str(test["test_id"]) != expected_test_id:
            raise ValueError(f"Test id does not match cell id for {cell_id}")
        cell_number = _cell_order(cell_id)
        if str(cell["source_cell_id"]) != f"b4c{cell_number}":
            raise ValueError(f"Unexpected source cell id for {cell_id}")
        if str(cell["source"]) != "CLO":
            raise ValueError(f"Unexpected source dataset for {cell_id}")
        if str(cell["model_number"]) != "APR18650M1A" or str(
            cell["chemistry"]
        ) != "LFP":
            raise ValueError(f"Unexpected target cell model or chemistry for {cell_id}")
        if str(test["test_type"]) != "cycle_aging":
            raise ValueError(f"Unexpected test type for {cell_id}")
        if float(test["temperature_C_min"]) != 30.0 or float(
            test["temperature_C_max"]
        ) != 30.0:
            raise ValueError(f"Unexpected test temperature for {cell_id}")
        if float(test["c_rate_discharge"]) != 4.0:
            raise ValueError(f"Unexpected discharge rate for {cell_id}")
        protocol = _parse_protocol(
            str(test["protocol_description"]),
            source=test_path.name,
        )
        rows.append(
            {
                "cell_order": cell_number,
                "cell_id": cell_id,
                "test_id": expected_test_id,
                "protocol_id": str(protocol["protocol_id"]),
                "batch_id": ATTIA_CAMPAIGN_ID,
                "source_cell_id": str(cell["source_cell_id"]),
                "source_protocol_id": str(protocol["protocol_id"]),
            }
        )
        metadata_paths.extend([test_path, cell_path])

    metadata = pd.DataFrame(rows).sort_values("cell_order", kind="stable")
    if set(metadata["cell_id"]) != EXPECTED_CELL_IDS:
        raise ValueError("Label-free Attia metadata does not cover the expected cohort")
    if metadata["cell_id"].duplicated().any() or metadata["test_id"].duplicated().any():
        raise ValueError("Label-free Attia identities must be unique")
    protocol_counts = metadata["protocol_id"].value_counts()
    if len(protocol_counts) != 9 or not (protocol_counts == 5).all():
        raise ValueError("Expected nine target protocols with five cells each")
    output = metadata.drop(columns="cell_order").reset_index(drop=True)
    forbidden = {
        "cycle_life",
        "is_censored",
        "event_observed",
        "num_cycles",
        "replicate_id",
    }
    if forbidden & set(output.columns):
        raise RuntimeError("Label-free metadata accidentally exposed target outcomes")
    audit: dict[str, object] = {
        "status": "passed",
        "dataset_id": ATTIA_DATASET_ID,
        "campaign_id": ATTIA_CAMPAIGN_ID,
        "cell_count": len(output),
        "protocol_count": int(output["protocol_id"].nunique()),
        "cells_per_protocol": sorted(protocol_counts.unique().astype(int).tolist()),
        "outcome_fields_emitted": [],
        "num_cycles_value_accessed": False,
        "celljar_metadata_bundle_sha256": _metadata_bundle_sha256(
            repository,
            metadata_paths,
        ),
        "metadata_file_count": len(metadata_paths),
        "warning": (
            "This map is label-free. Authoritative cycle-life outcomes must be built "
            "and joined only after target predictions are frozen and hashed."
        ),
    }
    return output, audit
