from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


# Official exclusions reproduced from the authors' Load Data.ipynb.
MATR_OFFICIAL_EXCLUSIONS: dict[str, str] = {
    "MATR_B1C8": "batch_1_did_not_reach_eol",
    "MATR_B1C10": "batch_1_did_not_reach_eol",
    "MATR_B1C12": "batch_1_did_not_reach_eol",
    "MATR_B1C13": "batch_1_did_not_reach_eol",
    "MATR_B1C22": "batch_1_did_not_reach_eol",
    "MATR_B3C2": "batch_3_noisy_or_incomplete",
    "MATR_B3C23": "batch_3_noisy_or_incomplete",
    "MATR_B3C32": "batch_3_noisy_or_incomplete",
    "MATR_B3C37": "batch_3_data_collection_problem",
    "MATR_B3C42": "batch_3_noisy_or_incomplete",
    "MATR_B3C43": "batch_3_noisy_or_incomplete",
}

MATR_CROSSWALK_REQUIRED_COLUMNS = {
    "paper_index",
    "dataset_id",
    "source_cell_id",
    "cell_id",
    "barcode",
    "batch_id",
    "batch_date",
    "paper_split",
    "cycle_life",
    "protocol_id",
    "continuation_source_cell_id",
    "continuation_added_cycles",
    "source_url",
    "source_pdf_sha256",
    "author_code_commit",
    "authority_status",
    "direct_author_assertion",
    "mapping_method",
}
MATR_SUPPLEMENTARY_PDF_SHA256 = (
    "5bd1e59d57daaf7778e42841c6aa0ffee6d286285d6968768ddb062fbe718a3c"
)
MATR_AUTHOR_CODE_COMMIT = "1ef13d27c66dc3d73affdaa008fbeba5687b2ea4"
MATR_CONTINUATIONS = {
    "MATR_B1C0": ("b2c7", 662),
    "MATR_B1C1": ("b2c8", 981),
    "MATR_B1C2": ("b2c9", 1060),
    "MATR_B1C3": ("b2c15", 208),
    "MATR_B1C4": ("b2c16", 482),
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _charge_policy(description: str) -> str:
    match = re.search(r"Charge policy:\s*(.+?)\.\s*4C discharge", description)
    if match:
        return match.group(1).strip().replace(" ", "")
    return description.strip()


def _batch_id(cell_id: str) -> str:
    match = re.match(r"MATR_B(\d+)C\d+$", cell_id)
    if not match:
        raise ValueError(f"Unexpected MATR cell id: {cell_id}")
    return f"MATR_BATCH_{match.group(1)}"


def _cell_order(cell_id: str) -> tuple[int, int]:
    match = re.match(r"MATR_B(\d+)C(\d+)$", cell_id)
    if not match:
        raise ValueError(f"Unexpected MATR cell id: {cell_id}")
    return int(match.group(1)), int(match.group(2))


def load_severson_crosswalk(path: str | Path) -> pd.DataFrame:
    """Load and validate the frozen Supplementary Table 9 identity crosswalk."""
    crosswalk_path = Path(path)
    crosswalk = pd.read_csv(crosswalk_path)
    missing = sorted(MATR_CROSSWALK_REQUIRED_COLUMNS - set(crosswalk.columns))
    if missing:
        raise ValueError(f"Missing authoritative crosswalk columns: {missing}")
    if len(crosswalk) != 124:
        raise ValueError(f"Expected 124 authoritative crosswalk rows, found {len(crosswalk)}")
    if crosswalk["cell_id"].duplicated().any():
        raise ValueError("Authoritative crosswalk contains duplicate cell ids")
    if crosswalk["barcode"].duplicated().any():
        raise ValueError("Authoritative crosswalk contains duplicate barcodes")
    non_null_columns = [
        "paper_index",
        "dataset_id",
        "source_cell_id",
        "cell_id",
        "barcode",
        "batch_id",
        "batch_date",
        "paper_split",
        "cycle_life",
        "protocol_id",
        "source_url",
        "source_pdf_sha256",
        "author_code_commit",
        "authority_status",
        "direct_author_assertion",
        "mapping_method",
    ]
    if crosswalk[non_null_columns].isna().any().any():
        raise ValueError("Authoritative crosswalk contains missing required values")
    cycle_life = pd.to_numeric(crosswalk["cycle_life"], errors="coerce")
    if cycle_life.isna().any() or (cycle_life <= 0).any():
        raise ValueError("Authoritative cycle-life labels must be positive numeric values")
    expected_indices = list(range(1, 125))
    if crosswalk["paper_index"].astype(int).tolist() != expected_indices:
        raise ValueError("Authoritative crosswalk paper_index must be ordered 1 through 124")
    expected_ids = {
        f"MATR_B{batch}C{cell}"
        for batch, size in ((1, 46), (2, 48), (3, 46))
        for cell in range(size)
        if f"MATR_B{batch}C{cell}" not in MATR_OFFICIAL_EXCLUSIONS
        and not (batch == 2 and cell in {7, 8, 9, 15, 16})
    }
    actual_ids = set(crosswalk["cell_id"])
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        unexpected_ids = sorted(actual_ids - expected_ids)
        raise ValueError(
            "Authoritative crosswalk cell ids disagree with official exclusions: "
            f"missing={missing_ids}, unexpected={unexpected_ids}"
        )
    expected_batch_counts = {"MATR_BATCH_1": 41, "MATR_BATCH_2": 43, "MATR_BATCH_3": 40}
    if crosswalk["batch_id"].value_counts().to_dict() != expected_batch_counts:
        raise ValueError("Unexpected authoritative batch counts")
    expected_split_counts = {"primary_test": 43, "train": 41, "secondary_test": 40}
    if crosswalk["paper_split"].value_counts().to_dict() != expected_split_counts:
        raise ValueError("Unexpected authoritative paper split counts")
    expected_splits = [
        (
            "primary_test"
            if paper_index <= 84 and (paper_index % 2 == 1 or paper_index == 84)
            else "train"
            if paper_index <= 84
            else "secondary_test"
        )
        for paper_index in expected_indices
    ]
    if crosswalk["paper_split"].tolist() != expected_splits:
        raise ValueError("Authoritative paper splits disagree with the author code")
    if set(crosswalk["source_pdf_sha256"]) != {MATR_SUPPLEMENTARY_PDF_SHA256}:
        raise ValueError("Unexpected Supplementary Information PDF SHA-256")
    if set(crosswalk["author_code_commit"]) != {MATR_AUTHOR_CODE_COMMIT}:
        raise ValueError("Unexpected author-code commit")
    if set(crosswalk["authority_status"]) != {"authoritative_source_derived"}:
        raise ValueError("Unexpected crosswalk authority status")
    if crosswalk["direct_author_assertion"].astype(bool).any():
        raise ValueError("Derived crosswalk cannot claim direct author assertion")
    continuation_rows = crosswalk.set_index("cell_id")
    for cell_id, (source_cell_id, added_cycles) in MATR_CONTINUATIONS.items():
        row = continuation_rows.loc[cell_id]
        if row["continuation_source_cell_id"] != source_cell_id:
            raise ValueError(f"Unexpected continuation source for {cell_id}")
        if int(row["continuation_added_cycles"]) != added_cycles:
            raise ValueError(f"Unexpected continuation length for {cell_id}")
    expected_tail = [
        "MATR_B3C38",
        "MATR_B3C39",
        "MATR_B3C40",
        "MATR_B3C41",
        "MATR_B3C44",
        "MATR_B3C45",
    ]
    if crosswalk.tail(6)["cell_id"].tolist() != expected_tail:
        raise ValueError("Authoritative Batch 3 tail identity check failed")
    crosswalk.attrs["sha256"] = _sha256(crosswalk_path)
    return crosswalk


def load_matr_metadata(
    celljar_repository: str | Path,
    authoritative_crosswalk: str | Path,
    *,
    official_cohort_only: bool = True,
) -> pd.DataFrame:
    """Load cell metadata and attach paper-authoritative identities and labels."""
    repository = Path(celljar_repository)
    cell_directory = repository / "cells"
    test_directory = repository / "tests"
    if not cell_directory.is_dir() or not test_directory.is_dir():
        raise FileNotFoundError(
            f"Expected celljar cells/ and tests/ under {repository}"
        )

    rows: list[dict[str, object]] = []
    for test_path in sorted(test_directory.glob("MATR_B*C*_CYCLING.json")):
        test = _read_json(test_path)
        cell_id = str(test["cell_id"])
        cell_path = cell_directory / f"{cell_id}.json"
        if not cell_path.exists():
            raise FileNotFoundError(f"Missing cell metadata for {cell_id}")
        cell = _read_json(cell_path)
        description = str(test.get("protocol_description") or "")

        rows.append(
            {
                "dataset_id": "MATR_SEVERSON_2019",
                "cell_id": cell_id,
                "source_cell_id": cell.get("source_cell_id"),
                "batch_id": _batch_id(cell_id),
                "celljar_protocol_id": _charge_policy(description),
                "source_num_cycles": float(test["num_cycles"]),
                "nominal_capacity_ah": float(cell["nominal_capacity_Ah"]),
                "c_rate_charge": float(test["c_rate_charge"]),
                "c_rate_discharge": float(test["c_rate_discharge"]),
                "temperature_c": float(test["temperature_C_max"]),
                "num_samples": int(test["n_samples"]),
                "source_doi": test.get("source_doi"),
                "source_license": test.get("source_license"),
            }
        )

    metadata = pd.DataFrame(rows)
    if len(metadata) != 135:
        raise ValueError(f"Expected 135 harmonized MATR cells, found {len(metadata)}")
    if metadata["cell_id"].duplicated().any():
        raise ValueError("MATR metadata contains duplicate cell ids")
    order = metadata["cell_id"].map(_cell_order)
    metadata["_batch_number"] = order.map(lambda value: value[0])
    metadata["_cell_number"] = order.map(lambda value: value[1])
    metadata = metadata.sort_values(["_batch_number", "_cell_number"]).reset_index(
        drop=True
    )

    crosswalk = load_severson_crosswalk(authoritative_crosswalk)
    crosswalk_sha256 = str(crosswalk.attrs["sha256"])
    source_id_by_cell = crosswalk.set_index("cell_id")["source_cell_id"]
    matched_source_ids = metadata.loc[
        metadata["cell_id"].isin(source_id_by_cell.index), ["cell_id", "source_cell_id"]
    ].set_index("cell_id")["source_cell_id"]
    if not matched_source_ids.equals(source_id_by_cell.loc[matched_source_ids.index]):
        raise ValueError("Celljar source ids disagree with the authoritative key order")
    authoritative_columns = [
        "cell_id",
        "paper_index",
        "barcode",
        "batch_date",
        "paper_split",
        "cycle_life",
        "protocol_id",
        "continuation_source_cell_id",
        "continuation_added_cycles",
        "source_url",
        "source_pdf_sha256",
        "author_code_commit",
        "authority_status",
        "direct_author_assertion",
        "mapping_method",
    ]
    metadata = metadata.merge(
        crosswalk[authoritative_columns],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    metadata["official_cohort"] = metadata["paper_index"].notna()
    metadata["exclusion_reason"] = metadata["cell_id"].map(MATR_OFFICIAL_EXCLUSIONS)
    continuation_ids = {"MATR_B2C7", "MATR_B2C8", "MATR_B2C9", "MATR_B2C15", "MATR_B2C16"}
    metadata.loc[metadata["cell_id"].isin(continuation_ids), "exclusion_reason"] = (
        "batch_2_continuation_merged_into_batch_1"
    )
    metadata["paper_split"] = metadata["paper_split"].fillna("excluded")
    metadata["protocol_id"] = metadata["protocol_id"].fillna(
        metadata["celljar_protocol_id"]
    )
    metadata["label_source"] = pd.NA
    metadata.loc[metadata["official_cohort"], "label_source"] = (
        "Severson et al. 2019 Supplementary Table 9"
    )
    metadata["authoritative_crosswalk_sha256"] = crosswalk_sha256
    metadata["cycle_life_definition"] = pd.NA
    metadata.loc[metadata["official_cohort"], "cycle_life_definition"] = (
        "first QDischarge below 0.88 Ah, otherwise stored cycle count plus one"
    )
    metadata["eol_soh_fraction"] = 0.8
    metadata["eol_capacity_ah"] = 0.88

    if int(metadata["official_cohort"].sum()) != 124:
        raise ValueError(
            f"Expected 124 official cells, found {int(metadata['official_cohort'].sum())}"
        )
    if metadata.loc[metadata["official_cohort"], "cycle_life"].isna().any():
        raise ValueError("Official cohort contains missing authoritative cycle-life labels")

    if official_cohort_only:
        metadata = metadata.loc[metadata["official_cohort"]].copy()
        if len(metadata) != 124:
            raise ValueError(f"Expected official 124-cell cohort, found {len(metadata)}")
    return metadata.drop(columns=["_batch_number", "_cell_number"]).reset_index(drop=True)


def matr_metadata_audit(metadata: pd.DataFrame) -> dict[str, object]:
    required = {
        "cell_id",
        "batch_id",
        "protocol_id",
        "cycle_life",
        "official_cohort",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing MATR audit columns: {missing}")
    life = metadata["cycle_life"]
    label_delta = life - metadata["source_num_cycles"]
    protocol_mismatch = metadata["protocol_id"] != metadata["celljar_protocol_id"]
    return {
        "cell_count": len(metadata),
        "batch_counts": dict(Counter(metadata["batch_id"])),
        "paper_split_counts": dict(Counter(metadata["paper_split"])),
        "protocol_count": int(metadata["protocol_id"].nunique()),
        "cycle_life_min": float(life.min()),
        "cycle_life_median": float(life.median()),
        "cycle_life_max": float(life.max()),
        "barcode_count": int(metadata["barcode"].nunique()),
        "authoritative_crosswalk_sha256": metadata[
            "authoritative_crosswalk_sha256"
        ].iloc[0],
        "label_source": sorted(metadata["label_source"].dropna().unique().tolist()),
        "source_num_cycles_difference_count": int((label_delta != 0).sum()),
        "source_num_cycles_differences": [
            {
                "cell_id": str(row.cell_id),
                "source_num_cycles": float(row.source_num_cycles),
                "authoritative_cycle_life": float(row.cycle_life),
                "difference": float(row.cycle_life - row.source_num_cycles),
            }
            for row in metadata.loc[
                label_delta != 0,
                ["cell_id", "source_num_cycles", "cycle_life"],
            ].itertuples(index=False)
        ],
        "celljar_protocol_difference_count": int(protocol_mismatch.sum()),
        "censoring_status": (
            "Supplementary Table 9 does not encode event status. Any downstream "
            "all-observed treatment must remain an explicit experiment assumption."
        ),
        "source_doi": sorted(metadata["source_doi"].dropna().unique().tolist()),
        "warning": (
            "Metadata audit and protocol-only probes are not early-health prediction results."
        ),
    }
