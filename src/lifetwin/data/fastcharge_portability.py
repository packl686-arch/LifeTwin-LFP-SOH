"""Authoritative identity mapping for the local MATR FastCharge summaries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.data.celljar import load_severson_crosswalk
from lifetwin.experiments.nasa_prefix_loco import canonical_frame_sha256


DATASET_ID = "MATR_FASTCHARGE_TABLE9_TRAJECTORY_V1"
CANONICAL_CYCLE_COLUMNS = (
    "dataset_id",
    "cell_id",
    "paper_split",
    "cycle_index",
    "discharge_capacity_ah",
    "internal_resistance_ohm",
    "temperature_max_c",
    "charge_time_s",
    "energy_efficiency",
)
TARGET_PREFIX_COLUMNS = (*CANONICAL_CYCLE_COLUMNS, "prefix_cycle")
_RAW_REQUIRED_COLUMNS = {
    "source_barcode",
    "cycle_index",
    "discharge_capacity_ah",
    "internal_resistance_ohm",
    "temperature_max_c",
    "charge_time_s",
    "energy_efficiency",
}


class FastChargePortabilityDataError(ValueError):
    """Raised when the FastCharge portability input violates its contract."""


def _experiment_value(config: Mapping[str, object], *keys: str) -> object:
    value: object = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise FastChargePortabilityDataError(
                f"Missing config path: {'.'.join(keys)}"
            )
        value = value[key]
    return value


def prepare_fastcharge_portability_cycles(
    raw_cycles: pd.DataFrame,
    authoritative_crosswalk: str | Path,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map raw BEEP barcodes to a frozen fixed-horizon Table 9 cohort."""
    missing_columns = sorted(_RAW_REQUIRED_COLUMNS - set(raw_cycles.columns))
    if missing_columns:
        raise FastChargePortabilityDataError(
            f"Raw FastCharge summary lacks columns: {missing_columns}"
        )
    source = raw_cycles.copy()
    if source["source_barcode"].isna().any():
        raise FastChargePortabilityDataError("Raw barcodes cannot be null")
    source["source_barcode"] = (
        source["source_barcode"].astype(str).str.strip().str.upper()
    )
    if source.duplicated(["source_barcode", "cycle_index"]).any():
        raise FastChargePortabilityDataError(
            "Raw FastCharge summary contains duplicate barcode cycles"
        )

    crosswalk = load_severson_crosswalk(authoritative_crosswalk)
    expected_crosswalk_hash = str(
        _experiment_value(config, "dataset", "authoritative_crosswalk_sha256")
    )
    if str(crosswalk.attrs["sha256"]) != expected_crosswalk_hash:
        raise FastChargePortabilityDataError("Authoritative crosswalk hash changed")
    identities = crosswalk.loc[:, ["cell_id", "barcode", "paper_split"]].rename(
        columns={"cell_id": "official_cell_id"}
    )
    identities["barcode"] = identities["barcode"].astype(str).str.upper()
    raw_barcodes = set(source["source_barcode"])
    missing_official = identities.loc[
        ~identities["barcode"].isin(raw_barcodes), "official_cell_id"
    ].tolist()
    expected_missing = [
        str(item["cell_id"])
        for item in _experiment_value(config, "dataset", "excluded_cells")
        if item["reason"] == "authoritative_barcode_missing_from_local_raw_transport"
    ]
    if sorted(missing_official) != sorted(expected_missing):
        raise FastChargePortabilityDataError(
            "Missing official raw barcodes differ from the frozen inventory"
        )

    mapped = source.merge(
        identities,
        left_on="source_barcode",
        right_on="barcode",
        how="inner",
        validate="many_to_one",
    )
    mapped["cell_id"] = mapped["official_cell_id"].astype(str)
    mapped["cycle_index"] = pd.to_numeric(mapped["cycle_index"], errors="coerce")
    if mapped["cycle_index"].isna().any():
        raise FastChargePortabilityDataError("Mapped cycle indices must be numeric")
    raw_index = mapped["cycle_index"].to_numpy(dtype=float)
    if not np.equal(raw_index, np.floor(raw_index)).all():
        raise FastChargePortabilityDataError("Mapped cycle indices must be integral")
    mapped["cycle_index"] = raw_index.astype(np.int64)
    score_end_cycle = int(
        _experiment_value(config, "split_and_firewall", "score_end_cycle")
    )
    support_exclusions: list[str] = []
    for cell_id, cell in mapped.groupby("cell_id", sort=True):
        support = sorted(
            cell.loc[cell["cycle_index"] <= score_end_cycle, "cycle_index"].tolist()
        )
        if support != list(range(1, score_end_cycle + 1)):
            support_exclusions.append(str(cell_id))
    expected_support_exclusions = [
        str(item["cell_id"])
        for item in _experiment_value(config, "dataset", "excluded_cells")
        if str(item["reason"]).startswith(
            "raw_summary_lacks_contiguous_support_through_fixed_cycle_"
        )
    ]
    if sorted(support_exclusions) != sorted(expected_support_exclusions):
        raise FastChargePortabilityDataError(
            "Fixed-horizon exclusions differ from the frozen support audit"
        )

    excluded = set(missing_official) | set(support_exclusions)
    mapped = mapped.loc[
        (~mapped["cell_id"].isin(excluded)) & (mapped["cycle_index"] <= score_end_cycle)
    ].copy()
    mapped["dataset_id"] = DATASET_ID
    canonical = mapped.loc[:, CANONICAL_CYCLE_COLUMNS].copy()
    canonical = canonical.sort_values(
        ["paper_split", "cell_id", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )
    string_columns = ("dataset_id", "cell_id", "paper_split")
    for column in string_columns:
        if canonical[column].isna().any():
            raise FastChargePortabilityDataError(f"{column} cannot be null")
        canonical[column] = canonical[column].astype(str)
    numeric_columns = tuple(
        column for column in CANONICAL_CYCLE_COLUMNS if column not in string_columns
    )
    numeric = canonical.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    dataset_config = _experiment_value(config, "dataset")
    if not isinstance(dataset_config, Mapping):
        raise FastChargePortabilityDataError("Dataset config must be an object")
    policy = dataset_config.get(
        "missing_value_policy",
        {
            "allowed_columns": [],
            "method": "within_cell_past_only_forward_fill",
            "leading_missing_value": "error",
            "discharge_capacity_imputation": "forbidden",
            "record_every_filled_coordinate": True,
        },
    )
    if not isinstance(policy, Mapping):
        raise FastChargePortabilityDataError("Missing-value policy must be an object")
    allowed_fill_columns = tuple(str(value) for value in policy["allowed_columns"])
    if policy.get("method") != "within_cell_past_only_forward_fill":
        raise FastChargePortabilityDataError("Unsupported missing-value policy")
    if "discharge_capacity_ah" in allowed_fill_columns:
        raise FastChargePortabilityDataError(
            "Discharge-capacity imputation is forbidden"
        )
    if not set(allowed_fill_columns).issubset(numeric_columns):
        raise FastChargePortabilityDataError(
            "Missing-value fill registry contains an unknown column"
        )
    missing_before = numeric.isna()
    disallowed_missing = missing_before.loc[
        :, [column for column in numeric_columns if column not in allowed_fill_columns]
    ]
    if disallowed_missing.any().any():
        raise FastChargePortabilityDataError(
            "Canonical FastCharge input is missing a non-imputable value"
        )
    imputation_records: list[dict[str, object]] = []
    for column in allowed_fill_columns:
        missing_rows = missing_before[column]
        for row_index in numeric.index[missing_rows]:
            imputation_records.append(
                {
                    "paper_split": str(canonical.at[row_index, "paper_split"]),
                    "cell_id": str(canonical.at[row_index, "cell_id"]),
                    "cycle_index": int(canonical.at[row_index, "cycle_index"]),
                    "column": column,
                }
            )
        numeric[column] = numeric.groupby(canonical["cell_id"], sort=False)[
            column
        ].ffill()
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise FastChargePortabilityDataError(
            "Canonical FastCharge model inputs must be finite"
        )
    canonical["cycle_index"] = numeric["cycle_index"].astype(np.int64)
    for column in set(numeric_columns) - {"cycle_index"}:
        canonical[column] = numeric[column].astype(float)
    if (
        (
            canonical[
                [
                    "discharge_capacity_ah",
                    "temperature_max_c",
                    "energy_efficiency",
                ]
            ]
            <= 0.0
        )
        .any()
        .any()
    ):
        raise FastChargePortabilityDataError(
            "Positive FastCharge model inputs must remain positive"
        )
    if (canonical["internal_resistance_ohm"] < 0.0).any():
        raise FastChargePortabilityDataError("Internal resistance cannot be negative")
    if (canonical["charge_time_s"] < 0.0).any():
        raise FastChargePortabilityDataError("Charge duration cannot be negative")
    expected_cell_count = int(
        _experiment_value(config, "dataset", "fixed_horizon_included_cell_count")
    )
    if canonical["cell_id"].nunique() != expected_cell_count:
        raise FastChargePortabilityDataError(
            "Canonical fixed-horizon cell count changed"
        )
    expected_split_counts = {
        str(_experiment_value(config, "split_and_firewall", "training_split")): int(
            _experiment_value(
                config,
                "split_and_firewall",
                "expected_training_cells",
            )
        ),
        **{
            str(key): int(value)
            for key, value in _experiment_value(
                config,
                "split_and_firewall",
                "expected_evaluation_cells_by_split",
            ).items()
        },
    }
    actual_split_counts = (
        canonical.groupby("paper_split", sort=True)["cell_id"].nunique().to_dict()
    )
    if actual_split_counts != expected_split_counts:
        raise FastChargePortabilityDataError(
            "Canonical split cell counts differ from the frozen protocol"
        )

    non_official_raw = sorted(raw_barcodes - set(identities["barcode"]))
    audit: dict[str, object] = {
        "schema_version": "lifetwin.fastcharge_portability_data_audit.v1",
        "dataset_id": DATASET_ID,
        "authoritative_crosswalk_sha256": expected_crosswalk_hash,
        "authority_status": "authoritative_source_derived_not_direct_author_assertion",
        "raw_unique_barcode_count": int(source["source_barcode"].nunique()),
        "official_crosswalk_cell_count": len(identities),
        "mapped_official_cell_count_before_support_filter": int(
            mapped["cell_id"].nunique() + len(support_exclusions)
        ),
        "included_fixed_horizon_cell_count": expected_cell_count,
        "included_split_counts": dict(sorted(actual_split_counts.items())),
        "missing_official_cells": sorted(missing_official),
        "fixed_horizon_support_exclusions": sorted(support_exclusions),
        "non_official_raw_barcode_count": len(non_official_raw),
        "non_official_raw_barcodes": non_official_raw,
        "cycle_row_count": len(canonical),
        "score_end_cycle": score_end_cycle,
        "canonical_cycle_sha256": canonical_frame_sha256(
            canonical,
            CANONICAL_CYCLE_COLUMNS,
        ),
        "paper_split_row_counts": dict(Counter(canonical["paper_split"])),
        "missing_value_policy": dict(policy),
        "past_only_forward_fill_count": len(imputation_records),
        "past_only_forward_fill_records": imputation_records,
        "forbidden_outcome_columns_removed_before_model_input": list(
            _experiment_value(config, "dataset", "forbidden_model_input_columns")
        ),
        "trajectory_suffix_scoring_performed": False,
        "claim_boundary": (
            "Identity/support audit only; no target suffix metric is computed here."
        ),
    }
    return canonical, audit


def build_fastcharge_prediction_inputs(
    cycles: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Separate complete training histories from evaluation target prefixes."""
    if tuple(cycles.columns) != CANONICAL_CYCLE_COLUMNS:
        raise FastChargePortabilityDataError("Canonical cycle columns changed")
    training_split = str(
        _experiment_value(config, "split_and_firewall", "training_split")
    )
    evaluation_splits = {
        str(value)
        for value in _experiment_value(
            config,
            "split_and_firewall",
            "evaluation_splits",
        )
    }
    training = cycles.loc[cycles["paper_split"] == training_split].copy()
    evaluation = cycles.loc[cycles["paper_split"].isin(evaluation_splits)].copy()
    prefixes: list[pd.DataFrame] = []
    for prefix_cycle in _experiment_value(
        config,
        "split_and_firewall",
        "prefix_cycles",
    ):
        prefix = evaluation.loc[evaluation["cycle_index"] <= int(prefix_cycle)].copy()
        prefix["prefix_cycle"] = int(prefix_cycle)
        prefixes.append(prefix.loc[:, TARGET_PREFIX_COLUMNS])
    target_prefixes = pd.concat(prefixes, ignore_index=True).sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )
    training = training.sort_values(
        ["cell_id", "cycle_index"], kind="stable", ignore_index=True
    )
    if set(training["cell_id"]) & set(target_prefixes["cell_id"]):
        raise FastChargePortabilityDataError(
            "Training and evaluation target identities overlap"
        )
    audit = {
        "schema_version": "lifetwin.fastcharge_prediction_input_audit.v1",
        "training_cell_count": int(training["cell_id"].nunique()),
        "target_cell_count": int(target_prefixes["cell_id"].nunique()),
        "training_row_count": len(training),
        "target_prefix_row_count": len(target_prefixes),
        "training_sha256": canonical_frame_sha256(
            training,
            CANONICAL_CYCLE_COLUMNS,
        ),
        "target_prefix_sha256": canonical_frame_sha256(
            target_prefixes,
            TARGET_PREFIX_COLUMNS,
        ),
        "target_future_rows_present": False,
    }
    return training, target_prefixes, audit


__all__ = [
    "CANONICAL_CYCLE_COLUMNS",
    "DATASET_ID",
    "FastChargePortabilityDataError",
    "TARGET_PREFIX_COLUMNS",
    "build_fastcharge_prediction_inputs",
    "prepare_fastcharge_portability_cycles",
]
