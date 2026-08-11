"""Outcome-free measurement qualification and stability gating for V8.

Stage A learns only repeat-measurement noise. Stage B perturbs an already
available residual prefix and decides whether the frozen V7 correction is
stable enough to issue. Neither API accepts future capacity outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from lifetwin.experiments.fastcharge_v5_pairwise import (
    FastChargeV5PairwiseError,
)
from lifetwin.experiments.fastcharge_v7_prefix_robustness import (
    frozen_gate_update,
)


REQUIRED_MEASUREMENT_COLUMNS = {
    "record_role",
    "physical_cell_id",
    "landmark_cycle",
    "repeat_index",
    "retention_pct",
    "tester_id",
    "temperature_chamber_id",
    "measurement_date",
    "reference_channel_id",
    "bridge_id",
}
REQUIRED_NOISE_LEDGER_COLUMNS = {
    "tester_id",
    "temperature_chamber_id",
    "model_id",
    "distribution",
    "degrees_of_freedom",
    "scale_pp",
    "physical_cell_count",
    "residual_count",
}
FORBIDDEN_OUTCOME_COLUMNS = {
    "future_retention_pct",
    "future_capacity_ah",
    "future_soh_pct",
    "suffix_retention_pct",
    "observed_future_retention_pct",
    "target_trajectory_mae_pp",
}


@dataclass(frozen=True)
class NoiseCandidate:
    model_id: str
    distribution: str
    degrees_of_freedom: float | None = None


@dataclass(frozen=True)
class MeasurementNoiseModel:
    model_id: str
    distribution: str
    scale_pp: float
    tester_id: str
    temperature_chamber_id: str
    degrees_of_freedom: float | None = None


@dataclass(frozen=True)
class StabilityIssuanceRequest:
    issuance_id: str
    cell_id: str
    manufacturing_batch_id: str
    tester_id: str
    temperature_chamber_id: str
    history_cycles: np.ndarray
    history_observed_retention_pct: np.ndarray
    history_previous_v5_center_pct: np.ndarray
    future_cycles: np.ndarray
    previous_v5_center_pct: np.ndarray
    current_v5_center_pct: np.ndarray


STABILITY_REQUEST_FIELDS = {
    "schema_version",
    "issuance_id",
    "cell_id",
    "manufacturing_batch_id",
    "tester_id",
    "temperature_chamber_id",
    "history_cycles",
    "history_observed_retention_pct",
    "history_previous_v5_center_pct",
    "future_cycles",
    "previous_v5_center_pct",
    "current_v5_center_pct",
}


def validate_measurement_frame(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    """Validate the Stage A repeatability-only measurement contract."""

    missing = REQUIRED_MEASUREMENT_COLUMNS - set(frame.columns)
    if missing:
        raise FastChargeV5PairwiseError(
            f"V8 measurement input is missing columns: {sorted(missing)}"
        )
    forbidden = FORBIDDEN_OUTCOME_COLUMNS & set(frame.columns)
    if forbidden:
        raise FastChargeV5PairwiseError(
            f"V8 measurement input contains future outcomes: {sorted(forbidden)}"
        )
    unregistered = set(frame.columns) - REQUIRED_MEASUREMENT_COLUMNS
    if unregistered:
        raise FastChargeV5PairwiseError(
            "V8 measurement input contains unregistered columns: "
            f"{sorted(unregistered)}"
        )
    data = frame.copy()
    for column in (
        "record_role",
        "physical_cell_id",
        "tester_id",
        "temperature_chamber_id",
        "reference_channel_id",
        "bridge_id",
    ):
        data[column] = data[column].fillna("").astype(str).str.strip()
    for column in ("landmark_cycle", "repeat_index", "retention_pct"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["measurement_date"] = pd.to_datetime(
        data["measurement_date"], errors="coerce", utc=True
    )
    numeric = data[["landmark_cycle", "repeat_index", "retention_pct"]].to_numpy(
        dtype=float
    )
    if len(data) == 0 or not np.isfinite(numeric).all():
        raise FastChargeV5PairwiseError(
            "V8 measurement input is empty or numerically invalid"
        )
    if data["measurement_date"].isna().any():
        raise FastChargeV5PairwiseError(
            "V8 measurement dates must be valid ISO timestamps"
        )
    if not np.equal(data["landmark_cycle"], np.floor(data["landmark_cycle"])).all():
        raise FastChargeV5PairwiseError(
            "V8 measurement landmark cycles must be integers"
        )
    if (
        not np.equal(data["repeat_index"], np.floor(data["repeat_index"])).all()
        or (data["repeat_index"] < 0).any()
    ):
        raise FastChargeV5PairwiseError(
            "V8 measurement repeat indices must be nonnegative integers"
        )
    if ((data["retention_pct"] <= 0.0) | (data["retention_pct"] > 150.0)).any():
        raise FastChargeV5PairwiseError(
            "V8 retention measurements must lie in (0, 150] percent"
        )
    allowed = set(config["measurement_contract"]["allowed_record_roles"])
    unknown = set(data["record_role"]) - allowed
    if unknown:
        raise FastChargeV5PairwiseError(
            f"V8 measurement input has unknown roles: {sorted(unknown)}"
        )
    if (data["tester_id"] == "").any() or (data["temperature_chamber_id"] == "").any():
        raise FastChargeV5PairwiseError(
            "V8 tester and chamber identities are mandatory"
        )
    _validate_role_identities(data)
    if data.duplicated(
        [
            "record_role",
            "physical_cell_id",
            "reference_channel_id",
            "bridge_id",
            "landmark_cycle",
            "repeat_index",
            "tester_id",
            "temperature_chamber_id",
            "measurement_date",
        ]
    ).any():
        raise FastChargeV5PairwiseError(
            "V8 measurement input contains duplicate measurement coordinates"
        )
    return data.sort_values(
        [
            "record_role",
            "tester_id",
            "temperature_chamber_id",
            "physical_cell_id",
            "reference_channel_id",
            "bridge_id",
            "landmark_cycle",
            "measurement_date",
            "repeat_index",
        ],
        kind="stable",
        ignore_index=True,
    )


def _validate_role_identities(data: pd.DataFrame) -> None:
    role_requirements = {
        "cell_repeat": ("physical_cell_id",),
        "daily_reference": ("reference_channel_id",),
        "tester_bridge": ("physical_cell_id", "bridge_id"),
    }
    for role, columns in role_requirements.items():
        rows = data.loc[data["record_role"] == role]
        if len(rows) == 0:
            raise FastChargeV5PairwiseError(
                f"V8 measurement input is missing required role: {role}"
            )
        for column in columns:
            if (rows[column] == "").any():
                raise FastChargeV5PairwiseError(f"V8 {role} rows require {column}")


def noise_candidates(config: Mapping[str, object]) -> tuple[NoiseCandidate, ...]:
    candidates = tuple(
        NoiseCandidate(
            model_id=str(item["model_id"]),
            distribution=str(item["distribution"]),
            degrees_of_freedom=(
                float(item["degrees_of_freedom"])
                if "degrees_of_freedom" in item
                else None
            ),
        )
        for item in config["noise_model_selection"]["candidate_families"]
    )
    identifiers = [item.model_id for item in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise FastChargeV5PairwiseError("V8 noise candidate identifiers must be unique")
    return candidates


def characterize_measurement_noise(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit a repeatability ledger and apply outcome-free quality gates."""

    data = validate_measurement_frame(frame, config)
    contract = config["measurement_contract"]
    repeats = data.loc[data["record_role"] == "cell_repeat"].copy()
    _validate_cell_repeat_coverage(repeats, contract)
    residuals = _repeat_residuals(repeats)
    candidate_scores = _crossfit_noise_candidates(residuals, config)
    selected = _select_noise_candidate(candidate_scores, config)
    ledger = _fit_noise_ledger(residuals, selected, config)
    quality = _quality_gate_summary(data, repeats, ledger, config)
    quality["selected_noise_model_id"] = selected.model_id
    quality["noise_candidate_scores"] = [
        {
            "model_id": str(row.model_id),
            "leave_one_cell_out_log_score": float(row.leave_one_cell_out_log_score),
            "valid_fold_count": int(row.valid_fold_count),
            "passed_all_folds": bool(row.passed_all_folds),
        }
        for row in candidate_scores.itertuples(index=False)
    ]
    return candidate_scores, ledger, quality


def _validate_cell_repeat_coverage(
    repeats: pd.DataFrame, contract: Mapping[str, object]
) -> None:
    minimum_cells = int(contract["minimum_physical_cell_count"])
    if int(repeats["physical_cell_id"].nunique()) < minimum_cells:
        raise FastChargeV5PairwiseError(
            "V8 measurement characterization has too few physical cells"
        )
    required_landmarks = {int(value) for value in contract["required_landmarks"]}
    minimum_repeats = int(contract["minimum_repeats_per_cell_landmark"])
    for cell_id, cell in repeats.groupby("physical_cell_id", sort=True):
        observed = set(cell["landmark_cycle"].astype(int))
        if not required_landmarks.issubset(observed):
            raise FastChargeV5PairwiseError(
                f"V8 cell {cell_id} lacks a required landmark"
            )
        for landmark in sorted(required_landmarks):
            rows = cell.loc[cell["landmark_cycle"] == landmark]
            if len(rows) < minimum_repeats:
                raise FastChargeV5PairwiseError(
                    f"V8 cell {cell_id} has insufficient repeats at {landmark}"
                )
            groups = rows[["tester_id", "temperature_chamber_id"]].drop_duplicates()
            if len(groups) != 1:
                raise FastChargeV5PairwiseError(
                    "V8 cell-landmark repeats must share one tester/chamber"
                )
            if rows["repeat_index"].duplicated().any():
                raise FastChargeV5PairwiseError(
                    "V8 cell-landmark repeat indices must be unique"
                )


def _repeat_residuals(repeats: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "physical_cell_id",
        "landmark_cycle",
        "tester_id",
        "temperature_chamber_id",
    ]
    result = repeats.copy()
    result["repeat_center_pct"] = result.groupby(group_columns, sort=True)[
        "retention_pct"
    ].transform("mean")
    result["repeat_count"] = result.groupby(group_columns, sort=True)[
        "retention_pct"
    ].transform("size")
    result["repeat_residual_pp"] = result["retention_pct"] - result["repeat_center_pct"]
    # Centered repeat residuals have variance sigma^2 * (1 - 1 / n).  Undo that
    # shrinkage so the ledger represents one future measurement error, not the
    # artificially quieter residual around a mean containing that measurement.
    result["measurement_error_proxy_pp"] = result["repeat_residual_pp"] / np.sqrt(
        1.0 - 1.0 / result["repeat_count"]
    )
    return result


def _crossfit_noise_candidates(
    residuals: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    contract = config["measurement_contract"]
    minimum_cells = int(contract["minimum_cells_per_noise_group"])
    minimum_residuals = int(contract["minimum_residuals_per_noise_group"])
    group_fields = list(contract["noise_group_fields"])
    cell_ids = sorted(residuals["physical_cell_id"].unique())
    rows: list[dict[str, object]] = []
    for candidate in noise_candidates(config):
        score = 0.0
        valid_folds = 0
        passed = True
        for held_out in cell_ids:
            fit = residuals.loc[residuals["physical_cell_id"] != held_out]
            held = residuals.loc[residuals["physical_cell_id"] == held_out]
            fold_score = 0.0
            fold_valid = True
            for group_key, held_group in held.groupby(group_fields, sort=True):
                normalized_key = _normalize_group_key(group_key)
                fit_group = _match_group(fit, group_fields, normalized_key)
                if (
                    int(fit_group["physical_cell_id"].nunique()) < minimum_cells
                    or len(fit_group) < minimum_residuals
                ):
                    fold_valid = False
                    break
                values = fit_group["measurement_error_proxy_pp"].to_numpy(dtype=float)
                scale = _fit_scale(values, candidate, config)
                held_values = held_group["measurement_error_proxy_pp"].to_numpy(
                    dtype=float
                )
                fold_score += float(
                    np.sum(_noise_logpdf(held_values, candidate, scale))
                )
            if not fold_valid:
                passed = False
                continue
            score += fold_score
            valid_folds += 1
        rows.append(
            {
                "model_id": candidate.model_id,
                "distribution": candidate.distribution,
                "degrees_of_freedom": candidate.degrees_of_freedom,
                "leave_one_cell_out_log_score": score,
                "valid_fold_count": valid_folds,
                "expected_fold_count": len(cell_ids),
                "passed_all_folds": bool(passed and valid_folds == len(cell_ids)),
            }
        )
    return pd.DataFrame(rows).sort_values("model_id", kind="stable", ignore_index=True)


def _normalize_group_key(value: object) -> tuple[object, ...]:
    return value if isinstance(value, tuple) else (value,)


def _match_group(
    frame: pd.DataFrame,
    fields: list[str],
    key: tuple[object, ...],
) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for field, value in zip(fields, key, strict=True):
        mask &= frame[field].to_numpy() == value
    return frame.loc[mask]


def _fit_scale(
    values: np.ndarray,
    candidate: NoiseCandidate,
    config: Mapping[str, object],
) -> float:
    variance = float(np.mean(np.square(values)))
    if candidate.distribution == "gaussian":
        scale = np.sqrt(variance)
    elif candidate.distribution == "student_t":
        degrees = candidate.degrees_of_freedom
        if degrees is None or degrees <= 2.0:
            raise FastChargeV5PairwiseError(
                "V8 Student-t noise candidates require df > 2"
            )
        scale = np.sqrt(variance * (degrees - 2.0) / degrees)
    else:
        raise FastChargeV5PairwiseError(
            f"Unknown V8 noise distribution: {candidate.distribution}"
        )
    return max(
        float(scale),
        float(config["noise_model_selection"]["minimum_scale_pp"]),
    )


def _noise_logpdf(
    values: np.ndarray, candidate: NoiseCandidate, scale: float
) -> np.ndarray:
    if candidate.distribution == "gaussian":
        return norm.logpdf(values, loc=0.0, scale=scale)
    if candidate.distribution == "student_t":
        assert candidate.degrees_of_freedom is not None
        return t.logpdf(values / scale, df=candidate.degrees_of_freedom) - np.log(scale)
    raise FastChargeV5PairwiseError(
        f"Unknown V8 noise distribution: {candidate.distribution}"
    )


def _select_noise_candidate(
    scores: pd.DataFrame, config: Mapping[str, object]
) -> NoiseCandidate:
    eligible = scores.loc[scores["passed_all_folds"]].copy()
    if len(eligible) == 0:
        raise FastChargeV5PairwiseError(
            "No V8 noise candidate passed every physical-cell fold"
        )
    best = float(eligible["leave_one_cell_out_log_score"].max())
    tolerance = float(config["noise_model_selection"]["tie_tolerance_log_score"])
    tied = set(
        eligible.loc[
            eligible["leave_one_cell_out_log_score"] >= best - tolerance,
            "model_id",
        ]
    )
    selected_id = next(
        model_id
        for model_id in config["noise_model_selection"]["tie_break_order"]
        if model_id in tied
    )
    return next(
        item for item in noise_candidates(config) if item.model_id == selected_id
    )


def _fit_noise_ledger(
    residuals: pd.DataFrame,
    selected: NoiseCandidate,
    config: Mapping[str, object],
) -> pd.DataFrame:
    group_fields = list(config["measurement_contract"]["noise_group_fields"])
    rows: list[dict[str, object]] = []
    for group_key, group in residuals.groupby(group_fields, sort=True):
        key = _normalize_group_key(group_key)
        scale = _fit_scale(
            group["measurement_error_proxy_pp"].to_numpy(dtype=float),
            selected,
            config,
        )
        row: dict[str, object] = {
            field: str(value) for field, value in zip(group_fields, key, strict=True)
        }
        row.update(
            {
                "model_id": selected.model_id,
                "distribution": selected.distribution,
                "degrees_of_freedom": selected.degrees_of_freedom,
                "scale_pp": scale,
                "physical_cell_count": int(group["physical_cell_id"].nunique()),
                "residual_count": len(group),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        group_fields, kind="stable", ignore_index=True
    )


def _quality_gate_summary(
    data: pd.DataFrame,
    repeats: pd.DataFrame,
    ledger: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, object]:
    thresholds = config["drift_quality_gates"]
    repeat_slopes: list[float] = []
    for _, group in repeats.groupby(
        [
            "physical_cell_id",
            "landmark_cycle",
            "tester_id",
            "temperature_chamber_id",
        ],
        sort=True,
    ):
        x = group["repeat_index"].to_numpy(dtype=float)
        y = group["retention_pct"].to_numpy(dtype=float)
        repeat_slopes.append(float(np.polyfit(x, y, 1)[0]))
    median_abs_slope = float(np.median(np.abs(repeat_slopes)))
    repeat_order_passed = bool(
        median_abs_slope
        <= float(thresholds["maximum_median_absolute_repeat_order_slope_pp_per_repeat"])
    )

    references = data.loc[data["record_role"] == "daily_reference"].copy()
    reference_group_fields = [
        "reference_channel_id",
        "tester_id",
        "temperature_chamber_id",
        "measurement_date",
    ]
    reference_counts = references.groupby(reference_group_fields, sort=True).size()
    minimum_reference_repeats = int(
        thresholds["minimum_daily_reference_repeats_per_group"]
    )
    reference_means = (
        references.groupby(reference_group_fields, sort=True)["retention_pct"]
        .mean()
        .rename("daily_mean")
        .reset_index()
    )
    reference_means["channel_center"] = reference_means.groupby(
        [
            "reference_channel_id",
            "tester_id",
            "temperature_chamber_id",
        ],
        sort=True,
    )["daily_mean"].transform("median")
    reference_means["absolute_shift"] = np.abs(
        reference_means["daily_mean"] - reference_means["channel_center"]
    )
    maximum_reference_shift = float(reference_means["absolute_shift"].max())
    expected_noise_groups = {
        (str(row.tester_id), str(row.temperature_chamber_id))
        for row in ledger[["tester_id", "temperature_chamber_id"]].itertuples(
            index=False
        )
    }
    reference_group_counts = (
        reference_counts.rename("repeat_count")
        .reset_index()
        .groupby(["tester_id", "temperature_chamber_id"], sort=True)
        .size()
    )
    observed_reference_groups = {
        (str(tester), str(chamber)) for tester, chamber in reference_group_counts.index
    }
    reference_groups_complete = expected_noise_groups.issubset(
        observed_reference_groups
    )
    minimum_reference_group_count = min(
        (int(reference_group_counts.get(group, 0)) for group in expected_noise_groups),
        default=0,
    )
    reference_passed = bool(
        reference_groups_complete
        and minimum_reference_group_count
        >= int(thresholds["minimum_daily_reference_tester_date_groups"])
        and bool((reference_counts >= minimum_reference_repeats).all())
        and maximum_reference_shift
        <= float(thresholds["maximum_absolute_daily_reference_shift_pp"])
    )

    bridges = data.loc[data["record_role"] == "tester_bridge"].copy()
    bridge_means = (
        bridges.groupby(["bridge_id", "tester_id"], sort=True)["retention_pct"]
        .mean()
        .rename("tester_mean")
        .reset_index()
    )
    bridge_summary = bridge_means.groupby("bridge_id", sort=True).agg(
        tester_count=("tester_id", "nunique"),
        minimum=("tester_mean", "min"),
        maximum=("tester_mean", "max"),
    )
    bridge_summary["range_pp"] = bridge_summary["maximum"] - bridge_summary["minimum"]
    maximum_bridge_range = float(bridge_summary["range_pp"].max())
    expected_testers = {tester for tester, _ in expected_noise_groups}
    observed_bridge_testers = set(bridge_means["tester_id"].astype(str))
    bridge_testers_complete = expected_testers.issubset(observed_bridge_testers)
    tester_neighbors = {tester: set() for tester in expected_testers}
    for _, bridge_group in bridge_means.groupby("bridge_id", sort=True):
        testers = set(bridge_group["tester_id"].astype(str)) & expected_testers
        for tester in testers:
            tester_neighbors[tester].update(testers - {tester})
    reached: set[str] = set()
    frontier = [next(iter(expected_testers))] if expected_testers else []
    while frontier:
        tester = frontier.pop()
        if tester in reached:
            continue
        reached.add(tester)
        frontier.extend(sorted(tester_neighbors[tester] - reached))
    bridge_graph_connected = reached == expected_testers
    bridge_passed = bool(
        len(bridge_summary) >= int(thresholds["minimum_bridge_id_count"])
        and bridge_testers_complete
        and bridge_graph_connected
        and bool(
            (
                bridge_summary["tester_count"]
                >= int(thresholds["minimum_testers_per_bridge"])
            ).all()
        )
        and maximum_bridge_range <= float(thresholds["maximum_tester_bridge_range_pp"])
    )

    contract = config["measurement_contract"]
    ledger_groups_passed = bool(
        (
            ledger["physical_cell_count"]
            >= int(contract["minimum_cells_per_noise_group"])
        ).all()
        and (
            ledger["residual_count"]
            >= int(contract["minimum_residuals_per_noise_group"])
        ).all()
    )
    all_passed = bool(
        repeat_order_passed
        and reference_passed
        and bridge_passed
        and ledger_groups_passed
    )
    return {
        "measurement_quality_passed": all_passed,
        "physical_cell_count": int(repeats["physical_cell_id"].nunique()),
        "noise_group_count": len(ledger),
        "repeat_order": {
            "median_absolute_slope_pp_per_repeat": median_abs_slope,
            "passed": repeat_order_passed,
        },
        "daily_reference": {
            "tester_date_group_count": len(reference_counts),
            "minimum_tester_date_group_count_per_noise_group": (
                minimum_reference_group_count
            ),
            "all_noise_groups_present": reference_groups_complete,
            "maximum_absolute_shift_pp": maximum_reference_shift,
            "passed": reference_passed,
        },
        "tester_bridge": {
            "bridge_id_count": len(bridge_summary),
            "all_noise_group_testers_present": bridge_testers_complete,
            "tester_graph_connected": bridge_graph_connected,
            "maximum_range_pp": maximum_bridge_range,
            "passed": bridge_passed,
        },
        "noise_group_support_passed": ledger_groups_passed,
    }


def measurement_noise_model(
    ledger: pd.DataFrame,
    *,
    tester_id: str,
    temperature_chamber_id: str,
) -> MeasurementNoiseModel | None:
    validated = validate_noise_ledger(ledger)
    rows = validated.loc[
        (validated["tester_id"] == tester_id)
        & (validated["temperature_chamber_id"] == temperature_chamber_id)
    ]
    if len(rows) == 0:
        return None
    if len(rows) != 1:
        raise FastChargeV5PairwiseError(
            "V8 noise ledger contains duplicate tester/chamber mappings"
        )
    row = rows.iloc[0]
    degrees = row.get("degrees_of_freedom")
    return MeasurementNoiseModel(
        model_id=str(row["model_id"]),
        distribution=str(row["distribution"]),
        scale_pp=float(row["scale_pp"]),
        tester_id=tester_id,
        temperature_chamber_id=temperature_chamber_id,
        degrees_of_freedom=(
            float(degrees) if degrees is not None and pd.notna(degrees) else None
        ),
    )


def validate_noise_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Validate the frozen Stage A tester/chamber noise mapping."""

    missing = REQUIRED_NOISE_LEDGER_COLUMNS - set(ledger.columns)
    extra = set(ledger.columns) - REQUIRED_NOISE_LEDGER_COLUMNS
    if missing or extra:
        raise FastChargeV5PairwiseError(
            "V8 noise ledger schema mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    data = ledger.copy()
    for column in (
        "tester_id",
        "temperature_chamber_id",
        "model_id",
        "distribution",
    ):
        data[column] = data[column].fillna("").astype(str).str.strip()
        if (data[column] == "").any():
            raise FastChargeV5PairwiseError(
                f"V8 noise ledger requires nonempty {column}"
            )
    for column in (
        "degrees_of_freedom",
        "scale_pp",
        "physical_cell_count",
        "residual_count",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if (
        len(data) == 0
        or not np.isfinite(
            data[["scale_pp", "physical_cell_count", "residual_count"]].to_numpy(
                dtype=float
            )
        ).all()
    ):
        raise FastChargeV5PairwiseError(
            "V8 noise ledger is empty or numerically invalid"
        )
    if (data["scale_pp"] <= 0.0).any():
        raise FastChargeV5PairwiseError("V8 noise ledger scales must be positive")
    for column in ("physical_cell_count", "residual_count"):
        if (data[column] <= 0).any() or not np.equal(
            data[column], np.floor(data[column])
        ).all():
            raise FastChargeV5PairwiseError(
                f"V8 noise ledger {column} values must be positive integers"
            )
    gaussian = data["distribution"] == "gaussian"
    student = data["distribution"] == "student_t"
    if not (gaussian | student).all():
        raise FastChargeV5PairwiseError(
            "V8 noise ledger contains an unsupported distribution"
        )
    degrees = data["degrees_of_freedom"]
    if degrees.loc[gaussian].notna().any() or (
        degrees.loc[student].isna().any() or (degrees.loc[student] <= 2.0).any()
    ):
        raise FastChargeV5PairwiseError(
            "V8 noise ledger degrees of freedom do not match the distribution"
        )
    if data.duplicated(["tester_id", "temperature_chamber_id"]).any():
        raise FastChargeV5PairwiseError(
            "V8 noise ledger contains duplicate tester/chamber mappings"
        )
    return data.sort_values(
        ["tester_id", "temperature_chamber_id"], kind="stable", ignore_index=True
    )


def validate_stability_request(
    request: Mapping[str, object], candidate: Mapping[str, object]
) -> StabilityIssuanceRequest:
    """Validate an exact outcome-free P60-to-P100 issuance request."""

    fields = set(request)
    missing = STABILITY_REQUEST_FIELDS - fields
    extra = fields - STABILITY_REQUEST_FIELDS
    if missing or extra:
        raise FastChargeV5PairwiseError(
            "V8 stability request schema mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if request["schema_version"] != (
        "lifetwin.fastcharge_v8.measurement_stability.request.v1"
    ):
        raise FastChargeV5PairwiseError(
            "V8 stability request schema version is unsupported"
        )
    identities = {
        field: str(request[field]).strip()
        for field in (
            "issuance_id",
            "cell_id",
            "manufacturing_batch_id",
            "tester_id",
            "temperature_chamber_id",
        )
    }
    if any(not value for value in identities.values()):
        raise FastChargeV5PairwiseError(
            "V8 stability request identities must be nonempty"
        )
    try:
        arrays = {
            field: np.asarray(request[field], dtype=float).reshape(-1)
            for field in (
                "history_cycles",
                "history_observed_retention_pct",
                "history_previous_v5_center_pct",
                "future_cycles",
                "previous_v5_center_pct",
                "current_v5_center_pct",
            )
        }
    except (TypeError, ValueError) as error:
        raise FastChargeV5PairwiseError(
            "V8 stability request arrays must be numeric"
        ) from error
    if not all(len(value) and np.isfinite(value).all() for value in arrays.values()):
        raise FastChargeV5PairwiseError(
            "V8 stability request arrays must be finite and nonempty"
        )
    transition = candidate["eligible_transition"]
    previous_prefix = int(transition["previous_prefix_cycle"])
    current_prefix = int(transition["current_prefix_cycle"])
    score_end = int(transition["score_end_cycle"])
    expected_history = np.arange(previous_prefix + 1, current_prefix + 1)
    expected_future = np.arange(current_prefix + 1, score_end + 1)
    if not np.array_equal(arrays["history_cycles"], expected_history):
        raise FastChargeV5PairwiseError("V8 stability request history support changed")
    if not np.array_equal(arrays["future_cycles"], expected_future):
        raise FastChargeV5PairwiseError(
            "V8 stability request future coordinate support changed"
        )
    if not (
        arrays["history_cycles"].shape
        == arrays["history_observed_retention_pct"].shape
        == arrays["history_previous_v5_center_pct"].shape
    ):
        raise FastChargeV5PairwiseError(
            "V8 stability request history arrays have inconsistent lengths"
        )
    if not (
        arrays["future_cycles"].shape
        == arrays["previous_v5_center_pct"].shape
        == arrays["current_v5_center_pct"].shape
    ):
        raise FastChargeV5PairwiseError(
            "V8 stability request future arrays have inconsistent lengths"
        )
    return StabilityIssuanceRequest(
        **identities,
        **arrays,
    )


def measurement_stability_update(
    history_cycles: np.ndarray,
    history_residuals: np.ndarray,
    future_cycles: np.ndarray,
    previous_future_center: np.ndarray,
    current_future_center: np.ndarray,
    candidate: Mapping[str, object],
    stability_config: Mapping[str, object],
    noise_model: MeasurementNoiseModel | None,
    *,
    protocol_sha256: str,
    cell_id: str,
    measurement_quality_passed: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Issue a correction only when noise-resampled V7 decisions are stable."""

    if len(protocol_sha256) != 64:
        raise FastChargeV5PairwiseError(
            "V8 stability issuance requires a canonical protocol SHA-256"
        )
    try:
        int(protocol_sha256, 16)
    except ValueError as error:
        raise FastChargeV5PairwiseError(
            "V8 stability issuance requires a canonical protocol SHA-256"
        ) from error
    if not cell_id.strip():
        raise FastChargeV5PairwiseError(
            "V8 stability issuance requires a physical-cell identity"
        )
    if (
        stability_config["stable_correction"]
        != "pointwise_median_of_measurement_resampled_effective_corrections"
    ):
        raise FastChargeV5PairwiseError(
            "V8 stability correction rule changed from the registered median"
        )
    if stability_config["seed_derivation"] != (
        "sha256(protocol_hash|cell_id|draw_index)_first_64_bits"
    ):
        raise FastChargeV5PairwiseError(
            "V8 stability seed derivation changed from the registered rule"
        )
    history_x = np.asarray(history_cycles, dtype=float).reshape(-1)
    history_y = np.asarray(history_residuals, dtype=float).reshape(-1)
    future_x = np.asarray(future_cycles, dtype=float).reshape(-1)
    previous = np.asarray(previous_future_center, dtype=float).reshape(-1)
    current = np.asarray(current_future_center, dtype=float).reshape(-1)
    arrays = (history_x, history_y, future_x, previous, current)
    if not all(len(value) and np.isfinite(value).all() for value in arrays):
        raise FastChargeV5PairwiseError(
            "V8 stability issuance arrays must be finite and nonempty"
        )
    if history_x.shape != history_y.shape or not (
        future_x.shape == previous.shape == current.shape
    ):
        raise FastChargeV5PairwiseError(
            "V8 stability issuance array lengths are inconsistent"
        )
    base_correction, base_active, base_diagnostics = frozen_gate_update(
        history_x,
        history_y,
        future_x,
        previous,
        current,
        candidate,
    )
    zero = np.zeros_like(current)
    reasons: list[str] = []
    if not base_active:
        reasons.append("unperturbed_v7_gate_inactive")
    if not measurement_quality_passed:
        reasons.append("measurement_quality_failed")
    if noise_model is None:
        reasons.append("noise_mapping_missing")
    if reasons:
        return zero, {
            "quality_activated": False,
            "reasons": reasons,
            "draw_count": 0,
            "unperturbed_v7_activated": base_active,
            "measurement_resampled_activation_probability": None,
            "measurement_resampled_correction_sign_probability": None,
            "p95_endpoint_effective_correction_deviation_pp": None,
            "noise_model_id": noise_model.model_id if noise_model else None,
            "noise_scale_pp": noise_model.scale_pp if noise_model else None,
            "base_diagnostics": base_diagnostics,
        }
    assert noise_model is not None
    _validate_noise_model(noise_model)
    draw_count = int(stability_config["draw_count"])
    if draw_count <= 0:
        raise FastChargeV5PairwiseError(
            "V8 measurement stability draw count must be positive"
        )
    corrections: list[np.ndarray] = []
    activations: list[bool] = []
    endpoint_sign_matches: list[bool] = []
    base_endpoint = float(base_correction[-1])
    base_sign = np.sign(base_endpoint)
    for draw_index in range(draw_count):
        rng = _stability_rng(protocol_sha256, cell_id, draw_index)
        noise = _draw_measurement_noise(rng, noise_model, len(history_y))
        correction, active, _ = frozen_gate_update(
            history_x,
            history_y + noise,
            future_x,
            previous,
            current,
            candidate,
        )
        corrections.append(correction)
        activations.append(active)
        endpoint_sign_matches.append(
            bool(active and np.sign(float(correction[-1])) == base_sign)
        )
    matrix = np.vstack(corrections)
    activation_probability = float(np.mean(activations))
    sign_probability = float(np.mean(endpoint_sign_matches))
    endpoint_deviation = np.abs(matrix[:, -1] - base_endpoint)
    p95_deviation = float(np.quantile(endpoint_deviation, 0.95))
    if activation_probability < float(
        stability_config["minimum_measurement_resampled_activation_probability"]
    ):
        reasons.append("activation_probability_below_threshold")
    if sign_probability < float(
        stability_config["minimum_measurement_resampled_correction_sign_probability"]
    ):
        reasons.append("correction_sign_probability_below_threshold")
    if p95_deviation > float(
        stability_config["maximum_p95_endpoint_effective_correction_deviation_pp"]
    ):
        reasons.append("endpoint_correction_deviation_above_threshold")
    activated = not reasons
    stable_correction = np.median(matrix, axis=0) if activated else zero
    return stable_correction, {
        "quality_activated": activated,
        "reasons": reasons,
        "draw_count": draw_count,
        "unperturbed_v7_activated": base_active,
        "measurement_resampled_activation_probability": activation_probability,
        "measurement_resampled_correction_sign_probability": sign_probability,
        "p95_endpoint_effective_correction_deviation_pp": p95_deviation,
        "noise_model_id": noise_model.model_id,
        "noise_distribution": noise_model.distribution,
        "noise_scale_pp": noise_model.scale_pp,
        "noise_degrees_of_freedom": noise_model.degrees_of_freedom,
        "base_diagnostics": base_diagnostics,
    }


def cohort_readiness_decision(
    issuances: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Decide whether a committed outcome-free cohort is large enough to open."""

    if not issuances:
        raise FastChargeV5PairwiseError(
            "V8 cohort readiness requires at least one issuance"
        )
    rows: list[dict[str, object]] = []
    for issuance in issuances:
        if issuance.get("schema_version") != (
            "lifetwin.fastcharge_v8.measurement_stability.issuance_result.v1"
        ):
            raise FastChargeV5PairwiseError(
                "V8 cohort contains an unsupported issuance result"
            )
        identities = {
            field: str(issuance.get(field, "")).strip()
            for field in (
                "issuance_id",
                "cell_id",
                "manufacturing_batch_id",
            )
        }
        if any(not value for value in identities.values()):
            raise FastChargeV5PairwiseError(
                "V8 cohort issuance identities must be nonempty"
            )
        if (
            issuance.get("future_outcomes_read") is not False
            or issuance.get("model_accuracy_evidence_created") is not False
            or issuance.get("v5_champion_changed") is not False
        ):
            raise FastChargeV5PairwiseError(
                "V8 cohort issuance violates the pre-outcome claim boundary"
            )
        stability = issuance.get("stability")
        if not isinstance(stability, dict) or not isinstance(
            stability.get("quality_activated"), bool
        ):
            raise FastChargeV5PairwiseError(
                "V8 cohort issuance lacks a Boolean stability decision"
            )
        active = bool(stability["quality_activated"])
        expected_decision = (
            "v8_stable_correction_issued" if active else "exact_v5_fallback_issued"
        )
        if issuance.get("decision") != expected_decision or issuance.get(
            "exact_v5_fallback"
        ) is not (not active):
            raise FastChargeV5PairwiseError(
                "V8 cohort issuance decision fields are inconsistent"
            )
        if active and issuance.get("measurement_quality_passed") is not True:
            raise FastChargeV5PairwiseError(
                "V8 cohort activated despite failed measurement quality"
            )
        hashes = {
            field: str(issuance.get(field, ""))
            for field in (
                "config_sha256",
                "candidate_sha256",
                "measurement_quality_decision_sha256",
                "noise_ledger_sha256",
                "forecast_correction_sha256",
            )
        }
        if any(not _is_sha256(value) for value in hashes.values()):
            raise FastChargeV5PairwiseError(
                "V8 cohort issuance contains a noncanonical SHA-256"
            )
        rows.append({**identities, **hashes, "quality_activated": active})

    frame = pd.DataFrame(rows)
    if frame["issuance_id"].duplicated().any() or frame["cell_id"].duplicated().any():
        raise FastChargeV5PairwiseError(
            "V8 cohort contains duplicate issuance or physical-cell identities"
        )
    for field in (
        "config_sha256",
        "candidate_sha256",
        "measurement_quality_decision_sha256",
        "noise_ledger_sha256",
    ):
        if frame[field].nunique() != 1:
            raise FastChargeV5PairwiseError(
                f"V8 cohort mixes incompatible {field} values"
            )

    stage_b = protocol["stage_b_outcome_free_stability_issuance"]
    stage_c = protocol["stage_c_single_open_blind_scoring"]
    primary = stage_c["required_primary_endpoints"]
    batch_gates = stage_c["required_batch_endpoints"]
    physical_cell_count = len(frame)
    manufacturing_batch_count = int(frame["manufacturing_batch_id"].nunique())
    active_rows = frame.loc[frame["quality_activated"]]
    stable_activation_count = len(active_rows)
    stable_activation_coverage = stable_activation_count / physical_cell_count
    activated_batch_count = int(active_rows["manufacturing_batch_id"].nunique())
    reasons: list[str] = []
    if physical_cell_count < int(stage_b["minimum_new_physical_cell_count"]):
        reasons.append("physical_cell_count_below_threshold")
    if manufacturing_batch_count < int(stage_b["minimum_manufacturing_batch_count"]):
        reasons.append("manufacturing_batch_count_below_threshold")
    if stable_activation_count < int(
        stage_b["minimum_stable_activation_count_before_outcome_opening"]
    ):
        reasons.append("stable_activation_count_below_threshold")
    if stable_activation_coverage < float(
        primary["minimum_stable_activation_coverage"]
    ):
        reasons.append("stable_activation_coverage_below_threshold")
    if activated_batch_count < int(batch_gates["minimum_activated_batch_count"]):
        reasons.append("activated_batch_count_below_threshold")
    return {
        "stage_c_outcome_opening_authorized": not reasons,
        "reasons": reasons,
        "physical_cell_count": physical_cell_count,
        "manufacturing_batch_count": manufacturing_batch_count,
        "stable_activation_count": stable_activation_count,
        "stable_activation_coverage": stable_activation_coverage,
        "activated_batch_count": activated_batch_count,
        "config_sha256": str(frame["config_sha256"].iloc[0]),
        "candidate_sha256": str(frame["candidate_sha256"].iloc[0]),
        "measurement_quality_decision_sha256": str(
            frame["measurement_quality_decision_sha256"].iloc[0]
        ),
        "noise_ledger_sha256": str(frame["noise_ledger_sha256"].iloc[0]),
    }


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_noise_model(model: MeasurementNoiseModel) -> None:
    if not np.isfinite(model.scale_pp) or model.scale_pp <= 0.0:
        raise FastChargeV5PairwiseError(
            "V8 measurement noise scale must be finite and positive"
        )
    if model.distribution == "gaussian":
        return
    if (
        model.distribution == "student_t"
        and model.degrees_of_freedom is not None
        and model.degrees_of_freedom > 2.0
    ):
        return
    raise FastChargeV5PairwiseError(
        "V8 measurement noise model is unsupported or incomplete"
    )


def _stability_rng(
    protocol_sha256: str,
    cell_id: str,
    draw_index: int,
) -> np.random.Generator:
    material = (f"{protocol_sha256}|{cell_id}|{draw_index}").encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed)


def _draw_measurement_noise(
    rng: np.random.Generator,
    model: MeasurementNoiseModel,
    count: int,
) -> np.ndarray:
    if model.distribution == "gaussian":
        return rng.normal(0.0, model.scale_pp, size=count)
    assert model.degrees_of_freedom is not None
    return rng.standard_t(model.degrees_of_freedom, size=count) * model.scale_pp


__all__ = [
    "MeasurementNoiseModel",
    "NoiseCandidate",
    "StabilityIssuanceRequest",
    "characterize_measurement_noise",
    "cohort_readiness_decision",
    "measurement_noise_model",
    "measurement_stability_update",
    "noise_candidates",
    "validate_noise_ledger",
    "validate_stability_request",
    "validate_measurement_frame",
]
