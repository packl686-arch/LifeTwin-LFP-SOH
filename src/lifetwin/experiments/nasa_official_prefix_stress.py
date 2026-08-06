from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from lifetwin.data.asset_intake import NASA_CHEMISTRY, NASA_DUPLICATE_IDS, NASA_ROLE


BASELINE_IDS = (
    "target_prefix_persistence",
    "nonpositive_linear_trend",
    "constrained_sqrt_loss_trend",
)
PREFIX_COLUMNS = (
    "physical_battery_id",
    "partition",
    "source_bundle",
    "experiment_batch",
    "temperature_c",
    "discharge_cutoff_v",
    "dynamic_condition",
    "prefix_cycle",
    "cycle_index",
    "capacity_retention_pct",
)
LABEL_COLUMNS = (
    "physical_battery_id",
    "partition",
    "source_bundle",
    "experiment_batch",
    "temperature_c",
    "discharge_cutoff_v",
    "dynamic_condition",
    "prefix_cycle",
    "cycle_index",
    "observed_capacity_retention_pct",
)
PREDICTION_COLUMNS = (
    "physical_battery_id",
    "partition",
    "source_bundle",
    "experiment_batch",
    "temperature_c",
    "discharge_cutoff_v",
    "dynamic_condition",
    "prefix_cycle",
    "cycle_index",
    "model_id",
    "predicted_capacity_retention_pct",
)
FORBIDDEN_PREDICTION_INPUT_TOKENS = (
    "future",
    "suffix",
    "label",
    "cycle_life",
    "final_capacity",
    "end_of_life",
)


class NasaOfficialPrefixStressError(ValueError):
    pass


def canonical_json_sha256(value: dict[str, object]) -> str:
    payload = copy.deepcopy(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_frame_sha256(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    payload = selected.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_nasa_official_prefix_stress_config(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_nasa_official_prefix_stress_config(value)


def _partition_map(config: dict[str, object]) -> dict[str, str]:
    partitions = config["partitions"]
    assert isinstance(partitions, dict)
    mapping: dict[str, str] = {}
    for partition in ("training", "validation", "locked_test"):
        raw_ids = partitions[partition]
        if not isinstance(raw_ids, list):
            raise NasaOfficialPrefixStressError("Partition IDs must be arrays")
        for raw_id in raw_ids:
            physical_id = str(raw_id)
            if physical_id in mapping:
                raise NasaOfficialPrefixStressError(
                    f"Physical battery crosses partitions: {physical_id}"
                )
            mapping[physical_id] = partition
    return mapping


def validate_nasa_official_prefix_stress_config(
    config: dict[str, object],
) -> dict[str, object]:
    if config.get("dataset", {}).get("chemistry") != NASA_CHEMISTRY:
        raise NasaOfficialPrefixStressError("NASA chemistry claim is not permitted")
    if config.get("dataset", {}).get("task_role") != NASA_ROLE:
        raise NasaOfficialPrefixStressError("NASA task role is not permitted")
    if tuple(config.get("baselines", [])) != BASELINE_IDS:
        raise NasaOfficialPrefixStressError("Baseline registry changed")
    if config.get("prefix_cycles") != [20, 40, 60, 100]:
        raise NasaOfficialPrefixStressError("Prefix registry changed")
    if int(config.get("score_end_cycle", 0)) != 200:
        raise NasaOfficialPrefixStressError("Score horizon changed")
    if int(config.get("minimum_future_observations", 0)) != 20:
        raise NasaOfficialPrefixStressError("Future support rule changed")
    mapping = _partition_map(config)
    if len(mapping) != 34:
        raise NasaOfficialPrefixStressError("Expected 34 physical batteries")
    for physical_id in NASA_DUPLICATE_IDS:
        if physical_id not in mapping:
            raise NasaOfficialPrefixStressError("Duplicate battery grouping is incomplete")
    firewall = config.get("future_label_firewall", {})
    required_firewall = {
        "prepare_writes_separate_prefix_and_label_tables": True,
        "predict_receives_label_path": False,
        "predict_receives_suffix_values": False,
        "prediction_written_before_score": True,
        "prediction_manifest_sha256_required": True,
    }
    if not isinstance(firewall, dict) or any(
        firewall.get(key) is not value for key, value in required_firewall.items()
    ):
        raise NasaOfficialPrefixStressError("Future-label firewall changed")
    expected_hash = config.get("semantic_sha256")
    if expected_hash not in {None, "PENDING_FIRST_COMMIT_FREEZE"}:
        observed_hash = canonical_json_sha256(config)
        if expected_hash != observed_hash:
            raise NasaOfficialPrefixStressError("Frozen config semantic hash mismatch")
    return config


def ensure_execution_authorized(config: dict[str, object]) -> None:
    validate_nasa_official_prefix_stress_config(config)
    rights = config.get("rights_gate", {})
    if not isinstance(rights, dict) or rights.get("execution_allowed") is not True:
        raise NasaOfficialPrefixStressError(
            "Official NASA scoring is blocked pending dataset-specific rights review"
        )


def _validate_cycle_table(cycles: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    required = {
        "physical_battery_id",
        "cycle_index",
        "capacity_ah",
        "source_bundle",
        "experiment_batch",
        "temperature_c",
        "discharge_cutoff_v",
        "dynamic_condition",
    }
    missing = sorted(required - set(cycles.columns))
    if missing:
        raise NasaOfficialPrefixStressError(f"Cycle table missing columns: {missing}")
    frame = cycles.loc[:, sorted(required)].copy()
    frame["physical_battery_id"] = frame["physical_battery_id"].astype(str)
    frame["cycle_index"] = pd.to_numeric(frame["cycle_index"], errors="raise")
    frame["capacity_ah"] = pd.to_numeric(frame["capacity_ah"], errors="raise")
    if frame[["cycle_index", "capacity_ah"]].isna().any().any():
        raise NasaOfficialPrefixStressError("Cycle index and capacity must be complete")
    if (frame["capacity_ah"] <= 0).any():
        raise NasaOfficialPrefixStressError("Capacity must be positive")
    if frame.duplicated(["physical_battery_id", "cycle_index"]).any():
        raise NasaOfficialPrefixStressError("Duplicate physical-battery cycle identity")
    partition_map = _partition_map(config)
    unknown = sorted(set(frame["physical_battery_id"]) - set(partition_map))
    if unknown:
        raise NasaOfficialPrefixStressError(f"Unfrozen physical batteries: {unknown}")
    frame["partition"] = frame["physical_battery_id"].map(partition_map)
    return frame.sort_values(
        ["physical_battery_id", "cycle_index"], kind="stable"
    ).reset_index(drop=True)


def prepare_prefix_and_future_labels(
    cycles: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config = validate_nasa_official_prefix_stress_config(config)
    frame = _validate_cycle_table(cycles, config)
    prefixes: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    max_cycle = int(config["score_end_cycle"])
    for physical_id, group in frame.groupby("physical_battery_id", sort=True):
        group = group.sort_values("cycle_index", kind="stable")
        initial = float(group.iloc[: min(5, len(group))]["capacity_ah"].median())
        normalized = group.copy()
        normalized["capacity_retention_pct"] = normalized["capacity_ah"] / initial * 100
        for prefix_cycle in config["prefix_cycles"]:
            prefix = normalized.loc[normalized["cycle_index"] <= prefix_cycle].copy()
            if len(prefix) != prefix_cycle or int(prefix["cycle_index"].max()) != prefix_cycle:
                continue
            prefix["prefix_cycle"] = prefix_cycle
            prefixes.append(prefix.loc[:, PREFIX_COLUMNS])
            future = normalized.loc[
                (normalized["cycle_index"] > prefix_cycle)
                & (normalized["cycle_index"] <= max_cycle)
            ].copy()
            future["prefix_cycle"] = prefix_cycle
            future["observed_capacity_retention_pct"] = future[
                "capacity_retention_pct"
            ]
            labels.append(future.loc[:, LABEL_COLUMNS])
    if not prefixes:
        raise NasaOfficialPrefixStressError("No eligible physical-battery prefixes")
    prefix_table = pd.concat(prefixes, ignore_index=True).sort_values(
        ["physical_battery_id", "prefix_cycle", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    label_table = (
        pd.concat(labels, ignore_index=True)
        if labels
        else pd.DataFrame(columns=LABEL_COLUMNS)
    ).sort_values(
        ["physical_battery_id", "prefix_cycle", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    audit = {
        "physical_battery_count": frame["physical_battery_id"].nunique(),
        "prefix_group_count": prefix_table.groupby(
            ["physical_battery_id", "prefix_cycle"]
        ).ngroups,
        "prefix_sha256": canonical_frame_sha256(prefix_table, PREFIX_COLUMNS),
        "future_label_sha256": canonical_frame_sha256(label_table, LABEL_COLUMNS),
        "split_unit": "physical_battery_id",
        "future_labels_separate": True,
    }
    return prefix_table, label_table, audit


def _fit_nonpositive_linear(cycle: np.ndarray, retention: np.ndarray) -> float:
    centered = cycle - cycle.mean()
    denominator = float(np.dot(centered, centered))
    slope = 0.0 if denominator == 0 else float(np.dot(centered, retention - retention.mean()) / denominator)
    return min(0.0, slope)


def _fit_sqrt_loss(cycle: np.ndarray, retention: np.ndarray) -> float:
    root = np.sqrt(cycle)
    centered = root - root.mean()
    denominator = float(np.dot(centered, centered))
    slope = 0.0 if denominator == 0 else float(np.dot(centered, retention - retention.mean()) / denominator)
    return min(0.0, slope)


def predict_prefix_baselines(
    prefix_table: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = validate_nasa_official_prefix_stress_config(config)
    forbidden = [
        column
        for column in prefix_table.columns
        if any(token in column.casefold() for token in FORBIDDEN_PREDICTION_INPUT_TOKENS)
    ]
    if forbidden:
        raise NasaOfficialPrefixStressError(
            f"Prediction input contains future-label columns: {forbidden}"
        )
    if tuple(prefix_table.columns) != PREFIX_COLUMNS:
        raise NasaOfficialPrefixStressError("Prediction input schema changed")
    prefix_table = prefix_table.sort_values(
        ["physical_battery_id", "prefix_cycle", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    end_cycle = int(config["score_end_cycle"])
    for (physical_id, prefix_cycle), group in prefix_table.groupby(
        ["physical_battery_id", "prefix_cycle"], sort=True
    ):
        group = group.sort_values("cycle_index", kind="stable")
        if len(group) != int(prefix_cycle) or int(group["cycle_index"].max()) != int(
            prefix_cycle
        ):
            raise NasaOfficialPrefixStressError("Prefix is not exactly truncated")
        cycle = group["cycle_index"].to_numpy(dtype=float)
        retention = group["capacity_retention_pct"].to_numpy(dtype=float)
        last = float(retention[-1])
        linear_slope = _fit_nonpositive_linear(cycle, retention)
        sqrt_slope = _fit_sqrt_loss(cycle, retention)
        metadata = group.iloc[-1]
        for target_cycle in range(int(prefix_cycle) + 1, end_cycle + 1):
            values = {
                "target_prefix_persistence": last,
                "nonpositive_linear_trend": last
                + linear_slope * (target_cycle - int(prefix_cycle)),
                "constrained_sqrt_loss_trend": last
                + sqrt_slope
                * (np.sqrt(target_cycle) - np.sqrt(int(prefix_cycle))),
            }
            for model_id, prediction in values.items():
                rows.append(
                    {
                        "physical_battery_id": physical_id,
                        "partition": metadata["partition"],
                        "source_bundle": metadata["source_bundle"],
                        "experiment_batch": metadata["experiment_batch"],
                        "temperature_c": metadata["temperature_c"],
                        "discharge_cutoff_v": metadata["discharge_cutoff_v"],
                        "dynamic_condition": metadata["dynamic_condition"],
                        "prefix_cycle": int(prefix_cycle),
                        "cycle_index": target_cycle,
                        "model_id": model_id,
                        "predicted_capacity_retention_pct": float(
                            np.clip(prediction, 0.0, 110.0)
                        ),
                    }
                )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["physical_battery_id", "prefix_cycle", "model_id", "cycle_index"],
        kind="stable",
    ).reset_index(drop=True)
    manifest = {
        "config_sha256": canonical_json_sha256(config),
        "prefix_input_sha256": canonical_frame_sha256(prefix_table, PREFIX_COLUMNS),
        "prediction_sha256": canonical_frame_sha256(
            predictions, PREDICTION_COLUMNS
        ),
        "prediction_row_count": len(predictions),
        "future_outcomes_used": False,
        "future_label_path_received": False,
        "score_executed": False,
    }
    return predictions, manifest


def _metric_row(group: pd.DataFrame) -> dict[str, float | int]:
    error = (
        group["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        - group["observed_capacity_retention_pct"].to_numpy(dtype=float)
    )
    absolute = np.abs(error)
    return {
        "future_observation_count": len(group),
        "trajectory_mae_pp": float(np.mean(absolute)),
        "trajectory_iae_pp_normalized_by_cycle_horizon": float(np.mean(absolute)),
        "trajectory_rmse_pp": float(np.sqrt(np.mean(error**2))),
        "endpoint_absolute_error_pp": float(absolute[-1]),
    }


def _grouped_descriptive(scores: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keys, group in scores.groupby(columns + ["model_id"], dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(columns + ["model_id"], keys)}
        row.update(
            {
                "cell_prefix_count": len(group),
                "physical_battery_count": group["physical_battery_id"].nunique(),
                "trajectory_mae_pp": float(group["trajectory_mae_pp"].mean()),
                "trajectory_rmse_pp": float(group["trajectory_rmse_pp"].mean()),
                "endpoint_absolute_error_pp": float(
                    group["endpoint_absolute_error_pp"].mean()
                ),
                "inference": "descriptive_only",
            }
        )
        rows.append(row)
    return rows


def score_prefix_baselines(
    future_labels: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: dict[str, object],
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = validate_nasa_official_prefix_stress_config(config)
    if tuple(future_labels.columns) != LABEL_COLUMNS:
        raise NasaOfficialPrefixStressError("Future-label schema changed")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise NasaOfficialPrefixStressError("Prediction schema changed")
    ordered_predictions = predictions.sort_values(
        ["physical_battery_id", "prefix_cycle", "model_id", "cycle_index"],
        kind="stable",
    ).reset_index(drop=True)
    observed_hash = canonical_frame_sha256(ordered_predictions, PREDICTION_COLUMNS)
    if manifest.get("prediction_sha256") != observed_hash:
        raise NasaOfficialPrefixStressError("Prediction artifact hash mismatch")
    if manifest.get("future_outcomes_used") is not False:
        raise NasaOfficialPrefixStressError("Prediction manifest violates firewall")
    labels = future_labels.sort_values(
        ["physical_battery_id", "prefix_cycle", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    join_keys = [
        "physical_battery_id",
        "partition",
        "source_bundle",
        "experiment_batch",
        "temperature_c",
        "discharge_cutoff_v",
        "dynamic_condition",
        "prefix_cycle",
        "cycle_index",
    ]
    linked = ordered_predictions.merge(
        labels,
        on=join_keys,
        how="inner",
        validate="many_to_one",
    )
    minimum = int(config["minimum_future_observations"])
    rows: list[dict[str, object]] = []
    candidate_groups = ordered_predictions.groupby(
        ["physical_battery_id", "prefix_cycle"], sort=True
    ).ngroups
    supported_groups = 0
    for keys, group in linked.groupby(
        ["physical_battery_id", "prefix_cycle", "model_id"], sort=True
    ):
        if len(group) < minimum:
            continue
        supported_groups += int(keys[2] == BASELINE_IDS[0])
        metadata = group.iloc[0]
        row = {
            "physical_battery_id": keys[0],
            "prefix_cycle": int(keys[1]),
            "model_id": keys[2],
            "partition": metadata["partition"],
            "source_bundle": metadata["source_bundle"],
            "experiment_batch": metadata["experiment_batch"],
            "temperature_c": metadata["temperature_c"],
            "discharge_cutoff_v": metadata["discharge_cutoff_v"],
            "dynamic_condition": metadata["dynamic_condition"],
            "domain_group": (
                "development_reference"
                if metadata["partition"] == "training"
                else "held_out_auxiliary"
            ),
            "prediction_status": "accepted",
            "refusal_fraction": 0.0,
        }
        row.update(_metric_row(group.sort_values("cycle_index", kind="stable")))
        rows.append(row)
    scores = pd.DataFrame(rows).sort_values(
        ["physical_battery_id", "prefix_cycle", "model_id"], kind="stable"
    ).reset_index(drop=True)
    unsupported_fraction = (
        1.0 - supported_groups / candidate_groups if candidate_groups else 1.0
    )
    primary = scores.loc[scores["partition"] == "locked_test"]
    per_cell_prefix = (
        primary.groupby(
            ["physical_battery_id", "prefix_cycle", "model_id"], sort=True
        )["trajectory_mae_pp"]
        .mean()
        .reset_index()
    )
    aggregate = (
        per_cell_prefix.groupby("model_id", sort=True)["trajectory_mae_pp"]
        .mean()
        .to_dict()
    )
    summary = {
        "status": "scored_once",
        "prediction_sha256": observed_hash,
        "score_sha256": canonical_frame_sha256(scores, scores.columns),
        "primary_scope": "locked_test",
        "primary_metric": "trajectory_mae_pp_equal_cell_then_equal_prefix",
        "primary_metrics_by_model": aggregate,
        "unsupported_cell_prefix_fraction": unsupported_fraction,
        "prediction_interval_coverage_fraction": None,
        "mean_interval_width_pp": None,
        "interval_reason": "no_prefrozen_prediction_interval",
        "refusal_fraction": 0.0,
        "grouped_metrics": {
            "physical_battery": _grouped_descriptive(
                scores, ["physical_battery_id"]
            ),
            "prefix": _grouped_descriptive(scores, ["prefix_cycle"]),
            "source_bundle": _grouped_descriptive(scores, ["source_bundle"]),
            "experiment_batch": _grouped_descriptive(scores, ["experiment_batch"]),
            "temperature": _grouped_descriptive(scores, ["temperature_c"]),
            "cutoff_or_dynamic": _grouped_descriptive(
                scores, ["discharge_cutoff_v", "dynamic_condition"]
            ),
            "domain": _grouped_descriptive(scores, ["domain_group"]),
            "acceptance": _grouped_descriptive(scores, ["prediction_status"]),
        },
        "inference": "descriptive_only_no_significance_test",
    }
    return scores, summary
