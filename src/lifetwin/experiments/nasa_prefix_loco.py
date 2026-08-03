"""Prefix-only NASA PCoE accelerated-cycling stress benchmark.

This module is intentionally separate from the LFP calendar-aging pipeline.  It
uses four public NASA Li-ion cell trajectories only to exercise chronological
prefix prediction and leave-one-physical-cell-out bookkeeping.  Prediction takes
an explicitly truncated prefix table; target suffix outcomes are accepted only by
the independent scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "lifetwin.nasa_prefix_loco.config.v1"
EXPERIMENT_ID = "nasa_pcoe_prefix_loco_accelerated_cycling_stress_v1"
DATASET_ID = "NASA_PCOE_LI_ION_AGING_DERIVED_CSV_V1"
EVIDENCE_ROLE = "cross_chemistry_accelerated_cycling_stress_only"
PREDICTION_MANIFEST_SCHEMA_VERSION = "lifetwin.nasa_prefix_loco.prediction_manifest.v1"
PREFIX_CYCLES = (20, 40, 60, 100)
PRIMARY_PREFIX_CYCLE = 40
SCORE_END_CYCLE = 132
MODEL_IDS = (
    "target_prefix_persistence",
    "target_prefix_linear",
    "target_prefix_sqrt_loss",
)
CELL_CUTOFFS = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5,
}

REQUIRED_CYCLE_COLUMNS = (
    "dataset_id",
    "cell_id",
    "cycle_index",
    "discharge_capacity_ah",
    "discharge_cutoff_voltage_v",
)
PREFIX_TABLE_COLUMNS = (*REQUIRED_CYCLE_COLUMNS, "prefix_cycle")
PREDICTION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "held_out_cell_id",
    "training_cell_ids",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "forecast_cycle",
    "predicted_capacity_retention_pct",
    "normalization_capacity_ah",
    "prefix_row_count",
    "target_prefix_sha256",
)
SCORE_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "held_out_cell_id",
    "training_cell_ids",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "future_observation_count",
    "trajectory_iae_pp_normalized_by_cycle_horizon",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
)

_ALLOWED_CLAIMS = (
    "chronological_prefix_prediction_software_validation",
    "descriptive_leave_one_cell_out_accelerated_cycling_stress_test",
)
_PROHIBITED_CLAIMS = (
    "lfp_chemistry_validation",
    "calendar_aging_validation",
    "fifteen_to_twenty_five_year_accuracy",
    "hithium_product_accuracy",
    "stationary_storage_field_validation",
    "inferential_significance_from_four_cells",
    "formal_uncertainty_coverage",
)


class NasaPrefixLocoError(ValueError):
    """Raised when the NASA prefix benchmark contract is violated."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NasaPrefixLocoError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NasaPrefixLocoError("Value is not canonical finite JSON") from exc
    return encoded.encode("ascii")


def canonical_json_sha256(value: object) -> str:
    """Hash a finite JSON value using the benchmark canonical codec."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash an ordered frame with explicit columns and stable scalar encoding."""
    if tuple(frame.columns) != tuple(columns):
        raise NasaPrefixLocoError("Frame column order does not match hash contract")
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        normalized: dict[str, object] = {}
        for column in columns:
            value = record[column]
            if isinstance(value, (np.integer,)):
                normalized[column] = int(value)
            elif isinstance(value, (np.floating,)):
                normalized[column] = float(value)
            elif isinstance(value, (np.bool_,)):
                normalized[column] = bool(value)
            else:
                normalized[column] = value
        records.append(normalized)
    return canonical_json_sha256(records)


def _exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise NasaPrefixLocoError(
            f"{context} keys changed: missing={missing}, extra={extra}"
        )


def validate_nasa_prefix_loco_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate and return a detached copy of the frozen benchmark config."""
    top_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "dataset",
        "design",
        "models",
        "metrics",
        "claim_boundaries",
    }
    _exact_keys(config, top_keys, "config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise NasaPrefixLocoError("Config schema version changed")
    if config["experiment_id"] != EXPERIMENT_ID:
        raise NasaPrefixLocoError("Experiment identity changed")
    if config["status"] != "descriptive_stress_test_frozen":
        raise NasaPrefixLocoError("Config is not frozen for descriptive use")

    dataset = config["dataset"]
    if not isinstance(dataset, Mapping):
        raise NasaPrefixLocoError("dataset must be an object")
    _exact_keys(
        dataset,
        {
            "dataset_id",
            "title",
            "chemistry",
            "aging_mode",
            "evidence_role",
            "conversion_provenance_status",
            "source_files_sha256",
            "cell_discharge_cutoff_voltage_v",
        },
        "dataset",
    )
    if dataset["dataset_id"] != DATASET_ID:
        raise NasaPrefixLocoError("Dataset identity changed")
    if dataset["chemistry"] != "unspecified_li_ion_not_lfp_evidence":
        raise NasaPrefixLocoError("Chemistry boundary changed")
    if dataset["aging_mode"] != "accelerated_cycle_aging":
        raise NasaPrefixLocoError("Aging-mode boundary changed")
    if dataset["evidence_role"] != EVIDENCE_ROLE:
        raise NasaPrefixLocoError("Evidence role changed")
    if dataset["conversion_provenance_status"] != "unverified_third_party_csv":
        raise NasaPrefixLocoError("Conversion provenance status changed")
    hashes = dataset["source_files_sha256"]
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "B0005.csv",
        "B0006.csv",
        "B0007.csv",
        "B0018.csv",
    }:
        raise NasaPrefixLocoError("Source-file hash registry changed")
    for filename, digest in hashes.items():
        if (
            not isinstance(filename, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise NasaPrefixLocoError("Source-file SHA256 must be lowercase hex")
    cutoffs = dataset["cell_discharge_cutoff_voltage_v"]
    if not isinstance(cutoffs, Mapping) or dict(cutoffs) != CELL_CUTOFFS:
        raise NasaPrefixLocoError("Cell cutoff-voltage registry changed")

    design = config["design"]
    if not isinstance(design, Mapping):
        raise NasaPrefixLocoError("design must be an object")
    _exact_keys(
        design,
        {
            "split",
            "cell_ids",
            "prefix_cycles",
            "primary_prefix_cycle",
            "score_end_cycle",
            "normalization",
            "target_future_outcomes_available_to_prediction",
            "training_cell_histories_used_by_models",
            "cell_weighting",
        },
        "design",
    )
    if design["split"] != "leave_one_physical_cell_out_four_folds":
        raise NasaPrefixLocoError("LOCO split changed")
    if tuple(design["cell_ids"]) != tuple(CELL_CUTOFFS):
        raise NasaPrefixLocoError("Frozen cell order changed")
    if tuple(design["prefix_cycles"]) != PREFIX_CYCLES:
        raise NasaPrefixLocoError("Prefix cycles changed")
    if design["primary_prefix_cycle"] != PRIMARY_PREFIX_CYCLE:
        raise NasaPrefixLocoError("Primary prefix changed")
    if design["score_end_cycle"] != SCORE_END_CYCLE:
        raise NasaPrefixLocoError("Common score endpoint changed")
    if design["normalization"] != "median_discharge_capacity_cycles_1_to_5":
        raise NasaPrefixLocoError("SOH normalization changed")
    if design["target_future_outcomes_available_to_prediction"] is not False:
        raise NasaPrefixLocoError("Target future outcomes must be unavailable")
    if design["training_cell_histories_used_by_models"] is not False:
        raise NasaPrefixLocoError("Frozen models must remain target-prefix-only")
    if design["cell_weighting"] != "equal_weight_per_held_out_cell":
        raise NasaPrefixLocoError("Cell weighting changed")

    models = config["models"]
    if not isinstance(models, Mapping):
        raise NasaPrefixLocoError("models must be an object")
    _exact_keys(
        models,
        {"model_ids", "prediction_clip_pct", "linear_slope_constraint"},
        "models",
    )
    if tuple(models["model_ids"]) != MODEL_IDS:
        raise NasaPrefixLocoError("Frozen model registry changed")
    if list(models["prediction_clip_pct"]) != [0.0, 110.0]:
        raise NasaPrefixLocoError("Prediction clipping bounds changed")
    if models["linear_slope_constraint"] != "non_positive":
        raise NasaPrefixLocoError("Linear slope constraint changed")

    metrics = config["metrics"]
    if not isinstance(metrics, Mapping):
        raise NasaPrefixLocoError("metrics must be an object")
    _exact_keys(
        metrics,
        {
            "primary",
            "secondary",
            "statistical_unit",
            "aggregation",
            "inference",
        },
        "metrics",
    )
    if metrics["primary"] != "trajectory_iae_pp_normalized_by_cycle_horizon":
        raise NasaPrefixLocoError("Primary metric changed")
    if tuple(metrics["secondary"]) != (
        "trajectory_mae_pp",
        "trajectory_rmse_pp",
        "endpoint_absolute_error_pp",
    ):
        raise NasaPrefixLocoError("Secondary metrics changed")
    if metrics["statistical_unit"] != "physical_cell_trajectory":
        raise NasaPrefixLocoError("Statistical unit changed")
    if metrics["aggregation"] != "equal_weight_across_four_held_out_cells":
        raise NasaPrefixLocoError("Metric aggregation changed")
    if metrics["inference"] != "descriptive_only_no_significance_test":
        raise NasaPrefixLocoError("Inference boundary changed")

    claims = config["claim_boundaries"]
    if not isinstance(claims, Mapping):
        raise NasaPrefixLocoError("claim_boundaries must be an object")
    _exact_keys(claims, {"allowed_claims", "prohibited_claims"}, "claims")
    if tuple(claims["allowed_claims"]) != _ALLOWED_CLAIMS:
        raise NasaPrefixLocoError("Allowed claims changed")
    if tuple(claims["prohibited_claims"]) != _PROHIBITED_CLAIMS:
        raise NasaPrefixLocoError("Prohibited claims changed")

    return json.loads(_canonical_json_bytes(config).decode("ascii"))


def load_nasa_prefix_loco_config(path: str | Path) -> dict[str, object]:
    """Load strict JSON and validate the frozen experiment configuration."""
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                NasaPrefixLocoError(f"Non-finite JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NasaPrefixLocoError("Cannot load NASA prefix config") from exc
    if not isinstance(value, Mapping):
        raise NasaPrefixLocoError("NASA prefix config must be a JSON object")
    return validate_nasa_prefix_loco_config(value)


def _validated_cycles(
    cycles: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    parsed = validate_nasa_prefix_loco_config(config)
    missing = sorted(set(REQUIRED_CYCLE_COLUMNS) - set(cycles.columns))
    if missing:
        raise NasaPrefixLocoError(f"Missing canonical cycle columns: {missing}")
    if cycles.empty:
        raise NasaPrefixLocoError("Canonical cycle table is empty")
    result = cycles.loc[:, REQUIRED_CYCLE_COLUMNS].copy()
    for column in ("dataset_id", "cell_id"):
        if result[column].isna().any():
            raise NasaPrefixLocoError(f"{column} cannot contain null values")
        result[column] = result[column].astype(str)
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise NasaPrefixLocoError("Canonical cycles contain an unexpected dataset")
    expected_cells = tuple(parsed["design"]["cell_ids"])
    if set(result["cell_id"]) != set(expected_cells):
        raise NasaPrefixLocoError(
            "Canonical cycles do not contain the exact four cells"
        )

    numeric = result.loc[
        :,
        ("cycle_index", "discharge_capacity_ah", "discharge_cutoff_voltage_v"),
    ].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise NasaPrefixLocoError(
            "Canonical cycle values must be finite numeric values"
        )
    raw_index = numeric["cycle_index"].to_numpy(dtype=float)
    if not np.equal(raw_index, np.floor(raw_index)).all() or (raw_index < 1).any():
        raise NasaPrefixLocoError("cycle_index must contain positive integers")
    result["cycle_index"] = raw_index.astype(np.int64)
    result["discharge_capacity_ah"] = numeric["discharge_capacity_ah"].astype(float)
    result["discharge_cutoff_voltage_v"] = numeric["discharge_cutoff_voltage_v"].astype(
        float
    )
    if (result["discharge_capacity_ah"] <= 0).any():
        raise NasaPrefixLocoError("Discharge capacity must be positive")
    if result.duplicated(["dataset_id", "cell_id", "cycle_index"]).any():
        raise NasaPrefixLocoError("Canonical cycles contain duplicate cell/cycle rows")

    cutoff_map = parsed["dataset"]["cell_discharge_cutoff_voltage_v"]
    for cell_id in expected_cells:
        cell = result.loc[result["cell_id"] == cell_id]
        cutoff = cell["discharge_cutoff_voltage_v"].to_numpy(dtype=float)
        expected_cutoff = float(cutoff_map[cell_id])
        if not np.allclose(cutoff, expected_cutoff, rtol=0.0, atol=1e-12):
            raise NasaPrefixLocoError(f"Cutoff voltage changed for {cell_id}")
        support = sorted(
            cell.loc[
                cell["cycle_index"] <= SCORE_END_CYCLE,
                "cycle_index",
            ].astype(int)
        )
        if support != list(range(1, SCORE_END_CYCLE + 1)):
            raise NasaPrefixLocoError(
                f"{cell_id} lacks contiguous cycle support through {SCORE_END_CYCLE}"
            )
    return result.sort_values(["cell_id", "cycle_index"], kind="stable").reset_index(
        drop=True
    )


def build_nasa_prefix_table(
    cycles: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Materialize the only table that the prediction function may receive."""
    ordered = _validated_cycles(cycles, config)
    frames: list[pd.DataFrame] = []
    for prefix_cycle in PREFIX_CYCLES:
        prefix = ordered.loc[ordered["cycle_index"] <= prefix_cycle].copy()
        prefix["prefix_cycle"] = prefix_cycle
        frames.append(prefix.loc[:, PREFIX_TABLE_COLUMNS])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["cell_id", "prefix_cycle", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )


def _validated_prefix_table(
    prefix_table: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    parsed = validate_nasa_prefix_loco_config(config)
    if tuple(prefix_table.columns) != PREFIX_TABLE_COLUMNS:
        raise NasaPrefixLocoError(
            "Prediction input must contain only the frozen prefix-table columns"
        )
    if prefix_table.empty:
        raise NasaPrefixLocoError("Prediction prefix table is empty")
    result = prefix_table.copy()
    numeric_columns = (
        "cycle_index",
        "discharge_capacity_ah",
        "discharge_cutoff_voltage_v",
        "prefix_cycle",
    )
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise NasaPrefixLocoError("Prefix table must contain finite numeric values")
    for column in ("cycle_index", "prefix_cycle"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise NasaPrefixLocoError(f"{column} must contain integers")
        result[column] = values.astype(np.int64)
    result["discharge_capacity_ah"] = numeric["discharge_capacity_ah"].astype(float)
    result["discharge_cutoff_voltage_v"] = numeric["discharge_cutoff_voltage_v"].astype(
        float
    )
    if set(result["dataset_id"].astype(str)) != {DATASET_ID}:
        raise NasaPrefixLocoError("Prefix table dataset identity changed")
    expected_cells = tuple(parsed["design"]["cell_ids"])
    if set(result["cell_id"].astype(str)) != set(expected_cells):
        raise NasaPrefixLocoError("Prefix table cell identities changed")
    if set(result["prefix_cycle"].astype(int)) != set(PREFIX_CYCLES):
        raise NasaPrefixLocoError("Prefix table does not contain all frozen prefixes")
    if (result["discharge_capacity_ah"] <= 0).any():
        raise NasaPrefixLocoError("Prefix capacities must be positive")
    if result.duplicated(["cell_id", "prefix_cycle", "cycle_index"]).any():
        raise NasaPrefixLocoError("Prefix table contains duplicate coordinates")

    cutoff_map = parsed["dataset"]["cell_discharge_cutoff_voltage_v"]
    for cell_id in expected_cells:
        for prefix_cycle in PREFIX_CYCLES:
            group = result.loc[
                (result["cell_id"].astype(str) == cell_id)
                & (result["prefix_cycle"] == prefix_cycle)
            ].sort_values("cycle_index", kind="stable")
            if group["cycle_index"].astype(int).tolist() != list(
                range(1, prefix_cycle + 1)
            ):
                raise NasaPrefixLocoError(
                    f"{cell_id} prefix {prefix_cycle} is not exactly truncated"
                )
            if (group["cycle_index"] > group["prefix_cycle"]).any():
                raise NasaPrefixLocoError(
                    "Prediction input contains target future rows"
                )
            if not np.allclose(
                group["discharge_cutoff_voltage_v"].to_numpy(dtype=float),
                float(cutoff_map[cell_id]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise NasaPrefixLocoError(f"Cutoff voltage changed for {cell_id}")
    return result.sort_values(
        ["cell_id", "prefix_cycle", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )


def _normalization_capacity(prefix: pd.DataFrame) -> float:
    first_five = prefix.loc[
        prefix["cycle_index"].between(1, 5), "discharge_capacity_ah"
    ].astype(float)
    if len(first_five) != 5:
        raise NasaPrefixLocoError("Normalization requires exact cycles 1 to 5")
    value = float(median(first_five.tolist()))
    if not math.isfinite(value) or value <= 0:
        raise NasaPrefixLocoError("Normalization capacity must be finite and positive")
    return value


def _linear_parameters(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x_values = [float(value) for value in x]
    y_values = [float(value) for value in y]
    x_mean = math.fsum(x_values) / len(x_values)
    y_mean = math.fsum(y_values) / len(y_values)
    denominator = math.fsum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        raise NasaPrefixLocoError("Linear prefix fit has no cycle support")
    slope = (
        math.fsum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )
    slope = min(float(slope), 0.0)
    return float(y_mean - slope * x_mean), slope


def _sqrt_loss_k(x: Sequence[float], retention: Sequence[float]) -> float:
    roots = [math.sqrt(max(float(value) - 1.0, 0.0)) for value in x]
    losses = [100.0 - float(value) for value in retention]
    denominator = math.fsum(value * value for value in roots)
    if denominator <= 0:
        raise NasaPrefixLocoError("Square-root prefix fit has no positive support")
    numerator = math.fsum(root * loss for root, loss in zip(roots, losses, strict=True))
    return max(float(numerator / denominator), 0.0)


def _clip_prediction(value: float) -> float:
    return float(min(max(value, 0.0), 110.0))


def _predict_one_prefix(
    prefix: pd.DataFrame,
    *,
    held_out_cell_id: str,
    prefix_cycle: int,
    training_cell_ids: str,
) -> list[dict[str, object]]:
    normalization = _normalization_capacity(prefix)
    cycle = prefix["cycle_index"].astype(float).tolist()
    retention = (
        100.0 * prefix["discharge_capacity_ah"].astype(float) / normalization
    ).tolist()
    prefix_hash = canonical_frame_sha256(
        prefix.loc[:, PREFIX_TABLE_COLUMNS].reset_index(drop=True),
        PREFIX_TABLE_COLUMNS,
    )
    linear_intercept, linear_slope = _linear_parameters(cycle, retention)
    sqrt_k = _sqrt_loss_k(cycle, retention)
    persistence = float(retention[-1])
    rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        for forecast_cycle in range(prefix_cycle + 1, SCORE_END_CYCLE + 1):
            if model_id == "target_prefix_persistence":
                prediction = persistence
            elif model_id == "target_prefix_linear":
                prediction = linear_intercept + linear_slope * forecast_cycle
            elif model_id == "target_prefix_sqrt_loss":
                prediction = 100.0 - sqrt_k * math.sqrt(forecast_cycle - 1.0)
            else:  # pragma: no cover - guarded by the frozen registry
                raise NasaPrefixLocoError(f"Unknown model: {model_id}")
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_id": DATASET_ID,
                    "held_out_cell_id": held_out_cell_id,
                    "training_cell_ids": training_cell_ids,
                    "prefix_cycle": prefix_cycle,
                    "score_end_cycle": SCORE_END_CYCLE,
                    "model_id": model_id,
                    "forecast_cycle": forecast_cycle,
                    "predicted_capacity_retention_pct": _clip_prediction(prediction),
                    "normalization_capacity_ah": normalization,
                    "prefix_row_count": len(prefix),
                    "target_prefix_sha256": prefix_hash,
                }
            )
    return rows


def predict_nasa_prefix_loco(
    prefix_table: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create predictions from an already truncated table and no suffix outcomes."""
    parsed = validate_nasa_prefix_loco_config(config)
    ordered = _validated_prefix_table(prefix_table, parsed)
    cell_ids = tuple(parsed["design"]["cell_ids"])
    rows: list[dict[str, object]] = []
    for held_out_cell_id in cell_ids:
        training_cell_ids = ";".join(
            cell_id for cell_id in cell_ids if cell_id != held_out_cell_id
        )
        for prefix_cycle in PREFIX_CYCLES:
            prefix = ordered.loc[
                (ordered["cell_id"].astype(str) == held_out_cell_id)
                & (ordered["prefix_cycle"] == prefix_cycle)
            ].sort_values("cycle_index", kind="stable")
            rows.extend(
                _predict_one_prefix(
                    prefix,
                    held_out_cell_id=held_out_cell_id,
                    prefix_cycle=prefix_cycle,
                    training_cell_ids=training_cell_ids,
                )
            )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    prediction_sha256 = canonical_frame_sha256(predictions, PREDICTION_COLUMNS)
    prefix_sha256 = canonical_frame_sha256(ordered, PREFIX_TABLE_COLUMNS)
    manifest: dict[str, object] = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "config_semantic_sha256": canonical_json_sha256(parsed),
        "prefix_bundle_sha256": prefix_sha256,
        "prediction_sha256": prediction_sha256,
        "prediction_row_count": len(predictions),
        "held_out_cell_ids": list(cell_ids),
        "prefix_cycles": list(PREFIX_CYCLES),
        "model_ids": list(MODEL_IDS),
        "score_end_cycle": SCORE_END_CYCLE,
        "target_future_outcomes_used": False,
        "training_cell_histories_used": False,
        "inference_scope": "descriptive_only_no_significance_test",
    }
    return predictions, manifest


def _validate_prediction_manifest(
    manifest: Mapping[str, object],
    *,
    config: Mapping[str, object],
    prefix_table: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "dataset_id",
        "evidence_role",
        "config_semantic_sha256",
        "prefix_bundle_sha256",
        "prediction_sha256",
        "prediction_row_count",
        "held_out_cell_ids",
        "prefix_cycles",
        "model_ids",
        "score_end_cycle",
        "target_future_outcomes_used",
        "training_cell_histories_used",
        "inference_scope",
    }
    _exact_keys(manifest, expected_keys, "prediction manifest")
    if manifest["schema_version"] != PREDICTION_MANIFEST_SCHEMA_VERSION:
        raise NasaPrefixLocoError("Prediction manifest schema changed")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise NasaPrefixLocoError("Prediction manifest experiment changed")
    if manifest["dataset_id"] != DATASET_ID:
        raise NasaPrefixLocoError("Prediction manifest dataset changed")
    if manifest["evidence_role"] != EVIDENCE_ROLE:
        raise NasaPrefixLocoError("Prediction evidence role changed")
    if manifest["config_semantic_sha256"] != canonical_json_sha256(config):
        raise NasaPrefixLocoError("Prediction config hash mismatch")
    if manifest["prefix_bundle_sha256"] != canonical_frame_sha256(
        prefix_table, PREFIX_TABLE_COLUMNS
    ):
        raise NasaPrefixLocoError("Prediction prefix hash mismatch")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise NasaPrefixLocoError("Prediction schema changed")
    if manifest["prediction_sha256"] != canonical_frame_sha256(
        predictions, PREDICTION_COLUMNS
    ):
        raise NasaPrefixLocoError("Prediction artifact hash mismatch")
    if manifest["prediction_row_count"] != len(predictions):
        raise NasaPrefixLocoError("Prediction row count mismatch")
    if tuple(manifest["held_out_cell_ids"]) != tuple(CELL_CUTOFFS):
        raise NasaPrefixLocoError("Prediction cells changed")
    if tuple(manifest["prefix_cycles"]) != PREFIX_CYCLES:
        raise NasaPrefixLocoError("Prediction prefixes changed")
    if tuple(manifest["model_ids"]) != MODEL_IDS:
        raise NasaPrefixLocoError("Prediction models changed")
    if manifest["score_end_cycle"] != SCORE_END_CYCLE:
        raise NasaPrefixLocoError("Prediction score endpoint changed")
    if manifest["target_future_outcomes_used"] is not False:
        raise NasaPrefixLocoError("Prediction claims target future access")
    if manifest["training_cell_histories_used"] is not False:
        raise NasaPrefixLocoError("Prediction claims training-history access")
    if manifest["inference_scope"] != "descriptive_only_no_significance_test":
        raise NasaPrefixLocoError("Prediction inference scope changed")


def _validated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise NasaPrefixLocoError("Prediction columns changed")
    result = predictions.copy()
    numeric_columns = (
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "predicted_capacity_retention_pct",
        "normalization_capacity_ah",
        "prefix_row_count",
    )
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise NasaPrefixLocoError("Prediction values must be finite")
    for column in (
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "prefix_row_count",
    ):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise NasaPrefixLocoError(f"Prediction {column} must be integral")
        result[column] = values.astype(np.int64)
    result["predicted_capacity_retention_pct"] = numeric[
        "predicted_capacity_retention_pct"
    ].astype(float)
    result["normalization_capacity_ah"] = numeric["normalization_capacity_ah"].astype(
        float
    )
    if set(result["experiment_id"].astype(str)) != {EXPERIMENT_ID}:
        raise NasaPrefixLocoError("Prediction experiment identity changed")
    if set(result["dataset_id"].astype(str)) != {DATASET_ID}:
        raise NasaPrefixLocoError("Prediction dataset identity changed")
    if set(result["held_out_cell_id"].astype(str)) != set(CELL_CUTOFFS):
        raise NasaPrefixLocoError("Prediction held-out cells changed")
    if set(result["prefix_cycle"].astype(int)) != set(PREFIX_CYCLES):
        raise NasaPrefixLocoError("Prediction prefix registry changed")
    if set(result["model_id"].astype(str)) != set(MODEL_IDS):
        raise NasaPrefixLocoError("Prediction model registry changed")
    expected_rows = (
        len(CELL_CUTOFFS)
        * len(MODEL_IDS)
        * sum(SCORE_END_CYCLE - prefix_cycle for prefix_cycle in PREFIX_CYCLES)
    )
    if len(result) != expected_rows:
        raise NasaPrefixLocoError("Prediction artifact cardinality changed")
    if not result["predicted_capacity_retention_pct"].between(0.0, 110.0).all():
        raise NasaPrefixLocoError("Predictions exceed frozen clipping bounds")
    if (result["normalization_capacity_ah"] <= 0).any():
        raise NasaPrefixLocoError("Prediction normalization capacity must be positive")
    valid_hash = result["target_prefix_sha256"].map(
        lambda value: isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    if not valid_hash.all():
        raise NasaPrefixLocoError("Prediction target-prefix hash is invalid")
    if result.duplicated(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"]
    ).any():
        raise NasaPrefixLocoError("Prediction coordinates are duplicated")
    for held_out_cell_id in CELL_CUTOFFS:
        expected_training = ";".join(
            cell_id for cell_id in CELL_CUTOFFS if cell_id != held_out_cell_id
        )
        for prefix_cycle in PREFIX_CYCLES:
            for model_id in MODEL_IDS:
                group = result.loc[
                    (result["held_out_cell_id"].astype(str) == held_out_cell_id)
                    & (result["prefix_cycle"] == prefix_cycle)
                    & (result["model_id"].astype(str) == model_id)
                ].sort_values("forecast_cycle", kind="stable")
                if group["forecast_cycle"].astype(int).tolist() != list(
                    range(prefix_cycle + 1, SCORE_END_CYCLE + 1)
                ):
                    raise NasaPrefixLocoError("Prediction forecast coordinates changed")
                if set(group["training_cell_ids"].astype(str)) != {expected_training}:
                    raise NasaPrefixLocoError("LOCO training-cell identity changed")
                if set(group["score_end_cycle"].astype(int)) != {SCORE_END_CYCLE}:
                    raise NasaPrefixLocoError("Prediction endpoint changed")
                if set(group["prefix_row_count"].astype(int)) != {prefix_cycle}:
                    raise NasaPrefixLocoError("Prediction prefix row count changed")
                if group["target_prefix_sha256"].nunique() != 1:
                    raise NasaPrefixLocoError(
                        "Prediction prefix hash changed within group"
                    )
                if group["normalization_capacity_ah"].nunique() != 1:
                    raise NasaPrefixLocoError(
                        "Prediction normalization changed within group"
                    )
    return result.sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )


def score_nasa_prefix_loco(
    cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Link held-out suffix truth only after verifying the prediction commitment."""
    parsed = validate_nasa_prefix_loco_config(config)
    ordered_cycles = _validated_cycles(cycles, parsed)
    prefix_table = build_nasa_prefix_table(ordered_cycles, parsed)
    ordered_predictions = _validated_predictions(predictions)
    _validate_prediction_manifest(
        prediction_manifest,
        config=parsed,
        prefix_table=prefix_table,
        predictions=ordered_predictions,
    )

    truth_frames: list[pd.DataFrame] = []
    for cell_id in CELL_CUTOFFS:
        cell = ordered_cycles.loc[
            (ordered_cycles["cell_id"] == cell_id)
            & (ordered_cycles["cycle_index"] <= SCORE_END_CYCLE)
        ].sort_values("cycle_index", kind="stable")
        normalization = float(median(cell["discharge_capacity_ah"].iloc[:5].tolist()))
        truth = pd.DataFrame(
            {
                "held_out_cell_id": cell_id,
                "forecast_cycle": cell["cycle_index"].astype(int),
                "observed_capacity_retention_pct": (
                    100.0 * cell["discharge_capacity_ah"].astype(float) / normalization
                ),
            }
        )
        truth_frames.append(truth)
    truth_table = pd.concat(truth_frames, ignore_index=True)
    linked = ordered_predictions.merge(
        truth_table,
        on=["held_out_cell_id", "forecast_cycle"],
        how="left",
        validate="many_to_one",
    )
    if linked["observed_capacity_retention_pct"].isna().any():
        raise NasaPrefixLocoError("Prediction coordinates could not be linked to truth")

    score_rows: list[dict[str, object]] = []
    group_columns = ["held_out_cell_id", "prefix_cycle", "model_id"]
    for (cell_id, prefix_cycle, model_id), group in linked.groupby(
        group_columns, sort=True
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        forecast_cycle = group["forecast_cycle"].to_numpy(dtype=float)
        observed = group["observed_capacity_retention_pct"].to_numpy(dtype=float)
        predicted = group["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        error = predicted - observed
        absolute_error = np.abs(error)
        horizon = float(forecast_cycle[-1] - forecast_cycle[0])
        iae = (
            float(np.trapezoid(absolute_error, forecast_cycle) / horizon)
            if horizon > 0
            else float(absolute_error[0])
        )
        score_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "held_out_cell_id": str(cell_id),
                "training_cell_ids": str(group["training_cell_ids"].iloc[0]),
                "prefix_cycle": int(prefix_cycle),
                "score_end_cycle": SCORE_END_CYCLE,
                "model_id": str(model_id),
                "future_observation_count": len(group),
                "trajectory_iae_pp_normalized_by_cycle_horizon": iae,
                "trajectory_mae_pp": float(np.mean(absolute_error)),
                "trajectory_rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                "endpoint_absolute_error_pp": float(absolute_error[-1]),
            }
        )
    scores = pd.DataFrame(score_rows, columns=SCORE_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id"],
        kind="stable",
        ignore_index=True,
    )
    metric_columns = list(SCORE_COLUMNS[-4:])
    aggregate = (
        scores.groupby(["prefix_cycle", "model_id"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    aggregate_records = json.loads(
        _canonical_json_bytes(aggregate.to_dict(orient="records")).decode("ascii")
    )
    summary: dict[str, object] = {
        "schema_version": "lifetwin.nasa_prefix_loco.score_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "prediction_sha256": str(prediction_manifest["prediction_sha256"]),
        "score_sha256": canonical_frame_sha256(scores, SCORE_COLUMNS),
        "fold_count": len(CELL_CUTOFFS),
        "prefix_cycles": list(PREFIX_CYCLES),
        "primary_prefix_cycle": PRIMARY_PREFIX_CYCLE,
        "score_end_cycle": SCORE_END_CYCLE,
        "model_ids": list(MODEL_IDS),
        "cell_weighting": "equal_weight_per_held_out_cell",
        "inference_scope": "descriptive_only_no_significance_test",
        "aggregate_metrics": aggregate_records,
        "allowed_claims": list(_ALLOWED_CLAIMS),
        "prohibited_claims": list(_PROHIBITED_CLAIMS),
    }
    return scores, summary


__all__ = [
    "CELL_CUTOFFS",
    "DATASET_ID",
    "EVIDENCE_ROLE",
    "EXPERIMENT_ID",
    "MODEL_IDS",
    "NasaPrefixLocoError",
    "PREDICTION_COLUMNS",
    "PREFIX_CYCLES",
    "PREFIX_TABLE_COLUMNS",
    "PRIMARY_PREFIX_CYCLE",
    "REQUIRED_CYCLE_COLUMNS",
    "SCORE_COLUMNS",
    "SCORE_END_CYCLE",
    "build_nasa_prefix_table",
    "canonical_frame_sha256",
    "canonical_json_sha256",
    "load_nasa_prefix_loco_config",
    "predict_nasa_prefix_loco",
    "score_nasa_prefix_loco",
    "validate_nasa_prefix_loco_config",
]
