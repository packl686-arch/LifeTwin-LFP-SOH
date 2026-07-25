"""Pure training, calibration, and state codecs for frozen V0.15.

This module never reads a dataset or accepts a dataframe/path.  The orchestration
layer must first construct the explicitly named numeric arrays and must enforce
the physical truth-file firewall.  In particular, no formal V2 generator is
invoked here.

The frozen documents contain a superficially broader ``98% finite center``
validity threshold as well as the center-fit rule saying that *any* nonfinite
input makes fitting inconclusive.  The implementation records and follows the
more specific fit rule: center development is exactly 600 complete clusters by
eight horizons, with every library, sqrt, and latent-target value finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_auc_score

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    canonical_json_bytes,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    FORECAST_DIMENSION,
    PREFIX_FEATURE_NAMES,
    ConformalExpansionState,
    IsotonicState,
    LogisticRiskState,
    StandardizerState,
    V2ModelError,
    fit_center_blend_beta,
    fit_conformal_expansion,
    fit_isotonic_state,
    fit_logistic_risk_state,
    simultaneous_nonconformity_scores,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
    PLACEBO_FEATURE_NAMES,
    VISIBLE_STRESS_FEATURE_NAMES,
    FrozenLabelFreeState,
    V015PipelineError,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256,
    FROZEN_PROTOCOL_ID,
)


CENTER_DEVELOPMENT_COUNT = 600
RISK_DEVELOPMENT_COUNT = 600
CALIBRATION_COUNT = 900
MINIMUM_CLASS_COUNT = 60
CATASTROPHIC_ERROR_THRESHOLD_PP = 5.0
CENTER_RIDGE_PENALTY = 0.01
CONFORMAL_COVERAGE = 0.90
CONFORMAL_ORDER_STATISTIC_INDEX = 811
CENTER_COMPLETENESS_INTERPRETATION = "exactly_600_complete_rows_all_8_horizons_finite"
MEAN_BASELINE_IDS = (
    "target_prefix_persistence",
    "target_prefix_sqrt_time",
    "target_prefix_bounded_power_law",
)
FROZEN_SOFTWARE_VERSIONS = (
    ("numpy", "2.5.1"),
    ("python", "3.12.13"),
    ("scikit-learn", "1.9.0"),
    ("scipy", "1.18.0"),
)

_MODEL_STATE_KEYS = frozenset(
    {
        "protocol_id",
        "config_sha256",
        "center_state",
        "risk_states",
        "calibration_state",
        "comparator_states",
        "feature_orders",
        "input_byte_hashes",
        "software_versions",
        "created_utc",
    }
)
_CENTER_STATE_KEYS = frozenset(
    {
        "beta",
        "development_cluster_count",
        "forecast_horizon_count",
        "ridge_penalty",
        "completeness_rule",
    }
)
_RISK_STATE_KEYS = frozenset(
    {
        "development_cluster_count",
        "eligible_cluster_count",
        "positive_label_count",
        "negative_label_count",
        "catastrophic_threshold_pp",
        "prefix_only",
        "visible_stress",
    }
)
_COMPARATOR_STATE_KEYS = frozenset(
    {
        "placebo_8",
        "arm_a_plus_s_plan",
        "strongest_single_feature",
    }
)
_SINGLE_FEATURE_KEYS = frozenset(
    {"feature_name", "danger_orientation", "oriented_empirical_auroc"}
)
_CALIBRATION_STATE_KEYS = frozenset(
    {
        "calibration_cluster_count",
        "positive_label_count",
        "negative_label_count",
        "prefix_only_isotonic",
        "visible_stress_isotonic",
        "conformal",
        "selected_mean_baseline",
        "mean_baseline_iae_pp",
    }
)
_LOGISTIC_KEYS = frozenset(
    {"feature_names", "standardizer", "intercept", "coefficients"}
)
_STANDARDIZER_KEYS = frozenset({"mean", "scale", "zero_variance"})
_ISOTONIC_KEYS = frozenset({"x_thresholds", "y_thresholds"})
_CONFORMAL_KEYS = frozenset(
    {
        "coverage",
        "calibration_count",
        "order_statistic_index",
        "expansion_pp",
    }
)
_FEATURE_ORDER_KEYS = frozenset(
    {"prefix_only", "visible_stress", "placebo_8", "arm_a_plus_s_plan"}
)
_INPUT_HASH_PHASE_KEYS = frozenset(
    {"center_development", "risk_development", "calibration"}
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_CONTRACT = load_artifact_contract()


class V015TrainingError(ValueError):
    """Raised when a frozen training, calibration, or codec rule is violated."""


@dataclass(frozen=True)
class CenterDevelopmentState:
    beta: float
    development_cluster_count: int = CENTER_DEVELOPMENT_COUNT
    forecast_horizon_count: int = FORECAST_DIMENSION
    ridge_penalty: float = CENTER_RIDGE_PENALTY
    completeness_rule: str = CENTER_COMPLETENESS_INTERPRETATION


@dataclass(frozen=True)
class RiskDevelopmentState:
    prefix_only_risk: LogisticRiskState
    visible_stress_risk: LogisticRiskState
    placebo_risk: LogisticRiskState
    arm_a_plus_s_plan_risk: LogisticRiskState
    strongest_single_feature_name: str
    strongest_single_feature_orientation: int
    strongest_single_feature_auroc: float
    development_cluster_count: int
    eligible_cluster_count: int
    positive_label_count: int
    negative_label_count: int


@dataclass(frozen=True)
class CalibrationDevelopmentState:
    prefix_only_isotonic: IsotonicState
    visible_stress_isotonic: IsotonicState
    conformal: ConformalExpansionState
    selected_mean_baseline: str
    mean_baseline_iae_pp: tuple[tuple[str, float], ...]
    calibration_cluster_count: int
    positive_label_count: int
    negative_label_count: int

    def baseline_iae_by_id(self) -> dict[str, float]:
        return dict(self.mean_baseline_iae_pp)


@dataclass(frozen=True)
class FrozenTrainingState:
    center: CenterDevelopmentState
    risk: RiskDevelopmentState
    calibration: CalibrationDevelopmentState


@dataclass(frozen=True)
class DecodedModelState:
    training_state: FrozenTrainingState
    frozen_label_free_state: FrozenLabelFreeState
    input_byte_hashes: dict[str, dict[str, str]]
    software_versions: dict[str, str]
    created_utc: str


def _numeric_matrix(
    values: Sequence[Sequence[float]],
    *,
    name: str,
    rows: int,
    columns: int,
    require_finite: bool = True,
) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V015TrainingError(f"{name} must be a numeric matrix") from exc
    if matrix.shape != (rows, columns):
        raise V015TrainingError(
            f"{name} must have shape ({rows}, {columns}), got {matrix.shape}"
        )
    if require_finite and not np.isfinite(matrix).all():
        raise V015TrainingError(f"{name} must contain only finite values")
    return matrix


def _numeric_vector(
    values: Sequence[float],
    *,
    name: str,
    length: int,
    require_finite: bool = True,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V015TrainingError(f"{name} must be a numeric vector") from exc
    if vector.shape != (length,):
        raise V015TrainingError(f"{name} must contain exactly {length} values")
    if require_finite and not np.isfinite(vector).all():
        raise V015TrainingError(f"{name} must contain only finite values")
    return vector


def _strict_boolean_vector(
    values: Sequence[bool], *, name: str, length: int
) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (length,) or any(
        not isinstance(value, (bool, np.bool_)) for value in raw
    ):
        raise V015TrainingError(f"{name} must contain exactly {length} strict booleans")
    return raw.astype(bool)


def _canonical_row_order(*aligned: np.ndarray) -> np.ndarray:
    """Return an ID-free lexicographic order for aligned training rows."""

    if not aligned:
        raise V015TrainingError("Canonical training order requires arrays")
    row_count = aligned[0].shape[0]
    columns: list[np.ndarray] = []
    for index, values in enumerate(aligned):
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] != row_count:
            raise V015TrainingError(
                f"Canonical training array {index} is not row-aligned"
            )
        try:
            flattened = array.reshape(row_count, -1).astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise V015TrainingError(
                "Canonical training arrays must be numeric or boolean"
            ) from exc
        columns.extend(flattened[:, column] for column in range(flattened.shape[1]))
    # np.lexsort uses the last key as primary.  Every relevant predictor,
    # target, eligibility bit, and comparator row is included, so any residual
    # tie consists of byte-equivalent numerical training content.
    return np.lexsort(tuple(reversed(columns)))


def _class_counts(labels: np.ndarray, *, stage: str) -> tuple[int, int]:
    positive = int(np.count_nonzero(labels == 1))
    negative = int(np.count_nonzero(labels == 0))
    if positive < MINIMUM_CLASS_COUNT or negative < MINIMUM_CLASS_COUNT:
        raise V015TrainingError(
            f"{stage} requires at least {MINIMUM_CLASS_COUNT} labels per "
            f"class; observed positive={positive}, negative={negative}"
        )
    return positive, negative


def _validate_complete_structure_features(
    prefix: np.ndarray, selected: np.ndarray, *, stage: str
) -> None:
    successful = prefix[selected, 0]
    failures = prefix[selected, 1]
    if (
        np.any(successful != np.rint(successful))
        or np.any((successful < 2.0) | (successful > 7.0))
        or np.any(failures != 7.0 - successful)
    ):
        raise V015TrainingError(
            f"{stage} contains an invalid successful/failure family count"
        )


def fit_center_development_state(
    *,
    library_forecasts_pct: Sequence[Sequence[float]],
    sqrt_forecasts_pct: Sequence[Sequence[float]],
    latent_targets_pct: Sequence[Sequence[float]],
) -> CenterDevelopmentState:
    """Fit beta from exactly 600 complete eight-horizon development rows."""

    library = _numeric_matrix(
        library_forecasts_pct,
        name="library_forecasts_pct",
        rows=CENTER_DEVELOPMENT_COUNT,
        columns=FORECAST_DIMENSION,
    )
    sqrt = _numeric_matrix(
        sqrt_forecasts_pct,
        name="sqrt_forecasts_pct",
        rows=CENTER_DEVELOPMENT_COUNT,
        columns=FORECAST_DIMENSION,
    )
    targets = _numeric_matrix(
        latent_targets_pct,
        name="latent_targets_pct",
        rows=CENTER_DEVELOPMENT_COUNT,
        columns=FORECAST_DIMENSION,
    )
    order = _canonical_row_order(library, sqrt, targets)
    library = library[order]
    sqrt = sqrt[order]
    targets = targets[order]
    try:
        beta = fit_center_blend_beta(
            library,
            sqrt,
            targets,
            ridge_penalty=CENTER_RIDGE_PENALTY,
        )
    except V2ModelError as exc:
        raise V015TrainingError("frozen center fitting failed") from exc
    state = CenterDevelopmentState(beta=beta)
    _parse_center(_center_payload(state))
    return state


def _empirical_single_feature_selection(
    features: np.ndarray, labels: np.ndarray
) -> tuple[str, int, float]:
    winner_name = PREFIX_FEATURE_NAMES[0]
    winner_orientation = 1
    winner_auroc = -math.inf
    for index, name in enumerate(PREFIX_FEATURE_NAMES):
        try:
            forward = float(roc_auc_score(labels, features[:, index]))
        except ValueError as exc:
            raise V015TrainingError(
                f"empirical AUROC is undefined for feature {name!r}"
            ) from exc
        orientation = 1 if forward >= 0.5 else -1
        oriented = forward if orientation == 1 else 1.0 - forward
        if not math.isfinite(oriented):
            raise V015TrainingError("single-feature AUROC must be finite")
        # A strict comparison preserves feature-list order on exact ties.
        if oriented > winner_auroc:
            winner_name = name
            winner_orientation = orientation
            winner_auroc = oriented
    return winner_name, winner_orientation, winner_auroc


def fit_risk_development_state(
    *,
    prefix_features: Sequence[Sequence[float]],
    visible_stress_features: Sequence[Sequence[float]],
    placebo_features: Sequence[Sequence[float]],
    planned_stress_index: Sequence[float],
    frozen_center_25y_pct: Sequence[float],
    latent_target_25y_pct: Sequence[float],
    common_pool_eligible: Sequence[bool],
) -> RiskDevelopmentState:
    """Fit all four frozen heads on the one common eligible D-risk pool."""

    prefix = _numeric_matrix(
        prefix_features,
        name="prefix_features",
        rows=RISK_DEVELOPMENT_COUNT,
        columns=len(PREFIX_FEATURE_NAMES),
        require_finite=False,
    )
    visible = _numeric_matrix(
        visible_stress_features,
        name="visible_stress_features",
        rows=RISK_DEVELOPMENT_COUNT,
        columns=8,
        require_finite=False,
    )
    placebo = _numeric_matrix(
        placebo_features,
        name="placebo_features",
        rows=RISK_DEVELOPMENT_COUNT,
        columns=8,
        require_finite=False,
    )
    planned = _numeric_vector(
        planned_stress_index,
        name="planned_stress_index",
        length=RISK_DEVELOPMENT_COUNT,
        require_finite=False,
    )
    center_25y = _numeric_vector(
        frozen_center_25y_pct,
        name="frozen_center_25y_pct",
        length=RISK_DEVELOPMENT_COUNT,
        require_finite=False,
    )
    latent_25y = _numeric_vector(
        latent_target_25y_pct,
        name="latent_target_25y_pct",
        length=RISK_DEVELOPMENT_COUNT,
    )
    eligible = _strict_boolean_vector(
        common_pool_eligible,
        name="common_pool_eligible",
        length=RISK_DEVELOPMENT_COUNT,
    )
    if not np.any(eligible):
        raise V015TrainingError("risk development common pool is empty")
    if not np.isfinite(center_25y[eligible]).all():
        raise V015TrainingError(
            "frozen_center_25y_pct is nonfinite inside the common eligible pool"
        )
    for name, values in (
        ("prefix_features", prefix),
        ("visible_stress_features", visible),
        ("placebo_features", placebo),
        ("planned_stress_index", planned),
    ):
        if not np.isfinite(values[eligible]).all():
            raise V015TrainingError(
                f"{name} is nonfinite inside the common eligible pool"
            )
    order = _canonical_row_order(
        prefix,
        visible,
        placebo,
        planned,
        center_25y,
        latent_25y,
        eligible,
    )
    prefix = prefix[order]
    visible = visible[order]
    placebo = placebo[order]
    planned = planned[order]
    center_25y = center_25y[order]
    latent_25y = latent_25y[order]
    eligible = eligible[order]
    _validate_complete_structure_features(
        prefix, eligible, stage="risk development common eligible pool"
    )

    labels = (
        np.abs(center_25y[eligible] - latent_25y[eligible])
        >= CATASTROPHIC_ERROR_THRESHOLD_PP
    ).astype(np.int64)
    positive, negative = _class_counts(labels, stage="risk development")
    prefix_eligible = prefix[eligible]
    visible_eligible = np.column_stack((prefix_eligible, visible[eligible]))
    placebo_eligible = np.column_stack((prefix_eligible, placebo[eligible]))
    planned_eligible = np.column_stack((prefix_eligible, planned[eligible]))
    try:
        prefix_state = fit_logistic_risk_state(
            prefix_eligible,
            labels,
            feature_names=PREFIX_FEATURE_NAMES,
        )
        visible_state = fit_logistic_risk_state(
            visible_eligible,
            labels,
            feature_names=VISIBLE_STRESS_FEATURE_NAMES,
        )
        placebo_state = fit_logistic_risk_state(
            placebo_eligible,
            labels,
            feature_names=PLACEBO_FEATURE_NAMES,
        )
        planned_state = fit_logistic_risk_state(
            planned_eligible,
            labels,
            feature_names=ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
        )
    except V2ModelError as exc:
        raise V015TrainingError("frozen logistic fitting failed") from exc
    name, orientation, auroc = _empirical_single_feature_selection(
        prefix_eligible, labels
    )
    state = RiskDevelopmentState(
        prefix_only_risk=prefix_state,
        visible_stress_risk=visible_state,
        placebo_risk=placebo_state,
        arm_a_plus_s_plan_risk=planned_state,
        strongest_single_feature_name=name,
        strongest_single_feature_orientation=orientation,
        strongest_single_feature_auroc=auroc,
        development_cluster_count=RISK_DEVELOPMENT_COUNT,
        eligible_cluster_count=int(np.count_nonzero(eligible)),
        positive_label_count=positive,
        negative_label_count=negative,
    )
    risk_payload, comparator_payload = _risk_payloads(state)
    _parse_risk(risk_payload, comparator_payload)
    return state


def fit_calibration_development_state(
    *,
    risk_state: RiskDevelopmentState,
    prefix_features: Sequence[Sequence[float]],
    visible_stress_features: Sequence[Sequence[float]],
    frozen_center_25y_pct: Sequence[float],
    latent_targets_pct: Sequence[Sequence[float]],
    base_interval_lower_pct: Sequence[Sequence[float]],
    base_interval_upper_pct: Sequence[Sequence[float]],
    mean_baseline_forecasts_pct: Mapping[str, Sequence[Sequence[float]]],
) -> CalibrationDevelopmentState:
    """Fit both isotonic maps, baseline selection, and 900/811 conformal."""

    risk_payload, comparator_payload = _risk_payloads(risk_state)
    _parse_risk(risk_payload, comparator_payload)
    prefix = _numeric_matrix(
        prefix_features,
        name="prefix_features",
        rows=CALIBRATION_COUNT,
        columns=len(PREFIX_FEATURE_NAMES),
    )
    visible = _numeric_matrix(
        visible_stress_features,
        name="visible_stress_features",
        rows=CALIBRATION_COUNT,
        columns=8,
    )
    _validate_complete_structure_features(
        prefix,
        np.ones(CALIBRATION_COUNT, dtype=bool),
        stage="calibration",
    )
    center_25y = _numeric_vector(
        frozen_center_25y_pct,
        name="frozen_center_25y_pct",
        length=CALIBRATION_COUNT,
    )
    targets = _numeric_matrix(
        latent_targets_pct,
        name="latent_targets_pct",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
    )
    lower = _numeric_matrix(
        base_interval_lower_pct,
        name="base_interval_lower_pct",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
    )
    upper = _numeric_matrix(
        base_interval_upper_pct,
        name="base_interval_upper_pct",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
    )
    if np.any(lower > upper):
        raise V015TrainingError("calibration base intervals must be ordered")

    observed_baselines = set(mean_baseline_forecasts_pct)
    if observed_baselines != set(MEAN_BASELINE_IDS):
        raise V015TrainingError(
            "mean_baseline_forecasts_pct must contain exactly the three "
            "frozen baseline IDs"
        )
    baseline_forecasts: dict[str, np.ndarray] = {}
    for model_id in MEAN_BASELINE_IDS:
        baseline_forecasts[model_id] = _numeric_matrix(
            mean_baseline_forecasts_pct[model_id],
            name=f"mean_baseline_forecasts_pct[{model_id!r}]",
            rows=CALIBRATION_COUNT,
            columns=FORECAST_DIMENSION,
        )

    order = _canonical_row_order(
        prefix,
        visible,
        center_25y,
        targets,
        lower,
        upper,
        *(baseline_forecasts[model_id] for model_id in MEAN_BASELINE_IDS),
    )
    prefix = prefix[order]
    visible = visible[order]
    center_25y = center_25y[order]
    targets = targets[order]
    lower = lower[order]
    upper = upper[order]
    baseline_forecasts = {
        model_id: baseline_forecasts[model_id][order] for model_id in MEAN_BASELINE_IDS
    }
    try:
        arm_a = risk_state.prefix_only_risk.decision_function(prefix)
        arm_b = risk_state.visible_stress_risk.decision_function(
            np.column_stack((prefix, visible))
        )
    except V2ModelError as exc:
        raise V015TrainingError("frozen calibration risk evaluation failed") from exc

    baseline_iae: dict[str, float] = {}
    for model_id in MEAN_BASELINE_IDS:
        forecast = baseline_forecasts[model_id]
        per_cluster_iae = np.trapezoid(
            np.abs(forecast - targets),
            x=np.asarray(FORECAST_DAYS, dtype=np.float64),
            axis=1,
        ) / (FORECAST_DAYS[-1] - FORECAST_DAYS[0])
        baseline_iae[model_id] = float(np.mean(per_cluster_iae))
    selected = min(baseline_iae, key=lambda item: (baseline_iae[item], item))

    labels = (
        np.abs(center_25y - targets[:, -1]) >= CATASTROPHIC_ERROR_THRESHOLD_PP
    ).astype(np.int64)
    positive, negative = _class_counts(labels, stage="calibration")
    try:
        prefix_isotonic = fit_isotonic_state(arm_a, labels)
        visible_isotonic = fit_isotonic_state(arm_b, labels)
        nonconformity = simultaneous_nonconformity_scores(lower, upper, targets)
        conformal = fit_conformal_expansion(
            nonconformity,
            coverage=CONFORMAL_COVERAGE,
        )
    except V2ModelError as exc:
        raise V015TrainingError("frozen calibration fitting failed") from exc
    if (
        conformal.calibration_count != CALIBRATION_COUNT
        or conformal.order_statistic_index != CONFORMAL_ORDER_STATISTIC_INDEX
    ):
        raise V015TrainingError("conformal state differs from frozen 900/811")
    state = CalibrationDevelopmentState(
        prefix_only_isotonic=prefix_isotonic,
        visible_stress_isotonic=visible_isotonic,
        conformal=conformal,
        selected_mean_baseline=selected,
        mean_baseline_iae_pp=tuple(
            (model_id, baseline_iae[model_id]) for model_id in MEAN_BASELINE_IDS
        ),
        calibration_cluster_count=CALIBRATION_COUNT,
        positive_label_count=positive,
        negative_label_count=negative,
    )
    _parse_calibration(_calibration_payload(state))
    return state


def construct_frozen_label_free_state(
    center: CenterDevelopmentState,
    risk: RiskDevelopmentState,
    calibration: CalibrationDevelopmentState,
) -> FrozenLabelFreeState:
    """Construct and fully validate the state consumed by prediction."""

    try:
        return FrozenLabelFreeState(
            center_beta=center.beta,
            prefix_only_risk=risk.prefix_only_risk,
            visible_stress_risk=risk.visible_stress_risk,
            placebo_risk=risk.placebo_risk,
            arm_a_plus_s_plan_risk=risk.arm_a_plus_s_plan_risk,
            strongest_single_feature_name=(risk.strongest_single_feature_name),
            strongest_single_feature_orientation=(
                risk.strongest_single_feature_orientation
            ),
            prefix_only_isotonic=calibration.prefix_only_isotonic,
            visible_stress_isotonic=calibration.visible_stress_isotonic,
            conformal=calibration.conformal,
        )
    except V015PipelineError as exc:
        raise V015TrainingError("frozen label-free state is invalid") from exc


def _zero_logistic_probe(feature_names: tuple[str, ...]) -> LogisticRiskState:
    dimension = len(feature_names)
    return LogisticRiskState(
        feature_names=feature_names,
        standardizer=StandardizerState(
            mean=(0.0,) * dimension,
            scale=(1.0,) * dimension,
            zero_variance=(True,) * dimension,
        ),
        intercept=0.0,
        coefficients=(0.0,) * dimension,
    )


def make_probe_state(center_beta: float) -> FrozenLabelFreeState:
    """Return a validated nonformal state for label-free feature extraction.

    This convenience state has zero risk heads and a zero conformal expansion.
    It exists only because the monolithic prediction pipeline requires a full
    state while D-center/D-risk orchestration is still constructing that state.
    It must never be serialized as ``model_state.json`` or used for issuance.
    """

    if not math.isfinite(center_beta) or not 0.0 <= center_beta <= 1.0:
        raise V015TrainingError("probe center_beta must be finite and in [0, 1]")
    isotonic = IsotonicState(
        x_thresholds=(-1.0, 1.0),
        y_thresholds=(0.0, 1.0),
    )
    try:
        return FrozenLabelFreeState(
            center_beta=float(center_beta),
            prefix_only_risk=_zero_logistic_probe(PREFIX_FEATURE_NAMES),
            visible_stress_risk=_zero_logistic_probe(VISIBLE_STRESS_FEATURE_NAMES),
            placebo_risk=_zero_logistic_probe(PLACEBO_FEATURE_NAMES),
            arm_a_plus_s_plan_risk=_zero_logistic_probe(
                ARM_A_PLUS_S_PLAN_FEATURE_NAMES
            ),
            strongest_single_feature_name=PREFIX_FEATURE_NAMES[0],
            strongest_single_feature_orientation=1,
            prefix_only_isotonic=isotonic,
            visible_stress_isotonic=isotonic,
            conformal=ConformalExpansionState(
                coverage=CONFORMAL_COVERAGE,
                calibration_count=CALIBRATION_COUNT,
                order_statistic_index=CONFORMAL_ORDER_STATISTIC_INDEX,
                expansion_pp=0.0,
            ),
        )
    except V015PipelineError as exc:
        raise V015TrainingError("probe state is invalid") from exc


def _require_exact_keys(
    value: object, expected: frozenset[str], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise V015TrainingError(f"{context} must be a JSON object")
    observed = set(value)
    if observed != expected:
        raise V015TrainingError(
            f"{context} keys changed: observed={sorted(observed)}, "
            f"expected={sorted(expected)}"
        )
    return value


def _real(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V015TrainingError(f"{context} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise V015TrainingError(f"{context} must be finite")
    return result


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise V015TrainingError(f"{context} must be an integer")
    return int(value)


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise V015TrainingError(f"{context} must be a nonempty string")
    return value


def _real_list(
    value: object, *, context: str, length: int | None = None
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise V015TrainingError(f"{context} must be a JSON array")
    result = tuple(
        _real(item, context=f"{context}[{index}]") for index, item in enumerate(value)
    )
    if length is not None and len(result) != length:
        raise V015TrainingError(f"{context} must contain exactly {length} values")
    return result


def _strict_bool_list(value: object, *, context: str, length: int) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, bool) for item in value)
    ):
        raise V015TrainingError(
            f"{context} must contain exactly {length} strict booleans"
        )
    return tuple(value)


def _utc_string(value: object, *, context: str = "created_utc") -> str:
    text = _string(value, context=context)
    if not _UTC_PATTERN.fullmatch(text):
        raise V015TrainingError(f"{context} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise V015TrainingError(f"{context} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise V015TrainingError(f"{context} must use UTC")
    return text


def _validate_hash_mapping(
    value: object, *, context: str, require_nonempty: bool = True
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise V015TrainingError(f"{context} must be a JSON object")
    if require_nonempty and not value:
        raise V015TrainingError(f"{context} must not be empty")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not _IDENTIFIER_PATTERN.fullmatch(key)
            or not isinstance(digest, str)
            or not _SHA256_PATTERN.fullmatch(digest)
        ):
            raise V015TrainingError(
                f"{context} must map safe input IDs to lowercase SHA256"
            )
        result[key] = digest
    return dict(sorted(result.items()))


def _validate_input_byte_hashes(value: object) -> dict[str, dict[str, str]]:
    phases = _require_exact_keys(
        value, _INPUT_HASH_PHASE_KEYS, context="input_byte_hashes"
    )
    return {
        phase: _validate_hash_mapping(
            phases[phase], context=f"input_byte_hashes.{phase}"
        )
        for phase in sorted(_INPUT_HASH_PHASE_KEYS)
    }


def default_software_versions() -> dict[str, str]:
    """Return the exact frozen runtime or fail before state construction."""

    actual = {
        "numpy": np.__version__,
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    expected = dict(FROZEN_SOFTWARE_VERSIONS)
    if actual != expected:
        raise V015TrainingError(
            f"training runtime differs from freeze: "
            f"observed={actual}, expected={expected}"
        )
    return expected


def _validate_software_versions(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise V015TrainingError("software_versions must be a JSON object")
    expected = dict(FROZEN_SOFTWARE_VERSIONS)
    if dict(value) != expected:
        raise V015TrainingError(
            f"software_versions differs from frozen runtime: "
            f"observed={dict(value)}, expected={expected}"
        )
    return expected


def _logistic_payload(state: LogisticRiskState) -> dict[str, Any]:
    return {
        "feature_names": list(state.feature_names),
        "standardizer": {
            "mean": list(state.standardizer.mean),
            "scale": list(state.standardizer.scale),
            "zero_variance": list(state.standardizer.zero_variance),
        },
        "intercept": state.intercept,
        "coefficients": list(state.coefficients),
    }


def _parse_logistic(
    value: object, *, context: str, expected_names: tuple[str, ...]
) -> LogisticRiskState:
    payload = _require_exact_keys(value, _LOGISTIC_KEYS, context=context)
    names_raw = payload["feature_names"]
    if (
        not isinstance(names_raw, list)
        or any(not isinstance(name, str) for name in names_raw)
        or tuple(names_raw) != expected_names
    ):
        raise V015TrainingError(f"{context}.feature_names changed")
    dimension = len(expected_names)
    standardizer_payload = _require_exact_keys(
        payload["standardizer"],
        _STANDARDIZER_KEYS,
        context=f"{context}.standardizer",
    )
    mean = _real_list(
        standardizer_payload["mean"],
        context=f"{context}.standardizer.mean",
        length=dimension,
    )
    scale = _real_list(
        standardizer_payload["scale"],
        context=f"{context}.standardizer.scale",
        length=dimension,
    )
    zero = _strict_bool_list(
        standardizer_payload["zero_variance"],
        context=f"{context}.standardizer.zero_variance",
        length=dimension,
    )
    coefficients = _real_list(
        payload["coefficients"],
        context=f"{context}.coefficients",
        length=dimension,
    )
    if any(item <= 0.0 for item in scale):
        raise V015TrainingError(f"{context}.standardizer.scale must be positive")
    for index, is_zero in enumerate(zero):
        if is_zero and (scale[index] != 1.0 or abs(coefficients[index]) > 1e-12):
            raise V015TrainingError(
                f"{context} zero-variance state violates the freeze"
            )
    return LogisticRiskState(
        feature_names=expected_names,
        standardizer=StandardizerState(
            mean=mean,
            scale=scale,
            zero_variance=zero,
        ),
        intercept=_real(payload["intercept"], context=f"{context}.intercept"),
        coefficients=coefficients,
    )


def _isotonic_payload(state: IsotonicState) -> dict[str, Any]:
    return {
        "x_thresholds": list(state.x_thresholds),
        "y_thresholds": list(state.y_thresholds),
    }


def _parse_isotonic(value: object, *, context: str) -> IsotonicState:
    payload = _require_exact_keys(value, _ISOTONIC_KEYS, context=context)
    x = _real_list(payload["x_thresholds"], context=f"{context}.x_thresholds")
    y = _real_list(payload["y_thresholds"], context=f"{context}.y_thresholds")
    if (
        len(x) < 2
        or len(x) != len(y)
        or any(right <= left for left, right in zip(x, x[1:]))
        or any(right < left for left, right in zip(y, y[1:]))
        or any(item < 0.0 or item > 1.0 for item in y)
    ):
        raise V015TrainingError(f"{context} is not a frozen isotonic map")
    return IsotonicState(x_thresholds=x, y_thresholds=y)


def _conformal_payload(state: ConformalExpansionState) -> dict[str, Any]:
    return {
        "coverage": state.coverage,
        "calibration_count": state.calibration_count,
        "order_statistic_index": state.order_statistic_index,
        "expansion_pp": state.expansion_pp,
    }


def _parse_conformal(value: object) -> ConformalExpansionState:
    payload = _require_exact_keys(
        value, _CONFORMAL_KEYS, context="calibration_state.conformal"
    )
    coverage = _real(
        payload["coverage"], context="calibration_state.conformal.coverage"
    )
    count = _integer(
        payload["calibration_count"],
        context="calibration_state.conformal.calibration_count",
    )
    index = _integer(
        payload["order_statistic_index"],
        context="calibration_state.conformal.order_statistic_index",
    )
    expansion = _real(
        payload["expansion_pp"],
        context="calibration_state.conformal.expansion_pp",
    )
    if (
        coverage != CONFORMAL_COVERAGE
        or count != CALIBRATION_COUNT
        or index != CONFORMAL_ORDER_STATISTIC_INDEX
        or expansion < 0.0
    ):
        raise V015TrainingError("conformal state differs from frozen 900/811")
    return ConformalExpansionState(
        coverage=coverage,
        calibration_count=count,
        order_statistic_index=index,
        expansion_pp=expansion,
    )


def _center_payload(state: CenterDevelopmentState) -> dict[str, Any]:
    return {
        "beta": state.beta,
        "development_cluster_count": state.development_cluster_count,
        "forecast_horizon_count": state.forecast_horizon_count,
        "ridge_penalty": state.ridge_penalty,
        "completeness_rule": state.completeness_rule,
    }


def _parse_center(value: object) -> CenterDevelopmentState:
    payload = _require_exact_keys(value, _CENTER_STATE_KEYS, context="center_state")
    beta = _real(payload["beta"], context="center_state.beta")
    cluster_count = _integer(
        payload["development_cluster_count"],
        context="center_state.development_cluster_count",
    )
    horizon_count = _integer(
        payload["forecast_horizon_count"],
        context="center_state.forecast_horizon_count",
    )
    ridge = _real(payload["ridge_penalty"], context="center_state.ridge_penalty")
    completeness = _string(
        payload["completeness_rule"], context="center_state.completeness_rule"
    )
    if (
        not 0.0 <= beta <= 1.0
        or cluster_count != CENTER_DEVELOPMENT_COUNT
        or horizon_count != FORECAST_DIMENSION
        or ridge != CENTER_RIDGE_PENALTY
        or completeness != CENTER_COMPLETENESS_INTERPRETATION
    ):
        raise V015TrainingError("center_state differs from the frozen rule")
    return CenterDevelopmentState(beta=beta)


def _risk_payloads(
    state: RiskDevelopmentState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    risk = {
        "development_cluster_count": state.development_cluster_count,
        "eligible_cluster_count": state.eligible_cluster_count,
        "positive_label_count": state.positive_label_count,
        "negative_label_count": state.negative_label_count,
        "catastrophic_threshold_pp": CATASTROPHIC_ERROR_THRESHOLD_PP,
        "prefix_only": _logistic_payload(state.prefix_only_risk),
        "visible_stress": _logistic_payload(state.visible_stress_risk),
    }
    comparators = {
        "placebo_8": _logistic_payload(state.placebo_risk),
        "arm_a_plus_s_plan": _logistic_payload(state.arm_a_plus_s_plan_risk),
        "strongest_single_feature": {
            "feature_name": state.strongest_single_feature_name,
            "danger_orientation": state.strongest_single_feature_orientation,
            "oriented_empirical_auroc": (state.strongest_single_feature_auroc),
        },
    }
    return risk, comparators


def _feature_orders_payload() -> dict[str, list[str]]:
    return {
        "prefix_only": list(PREFIX_FEATURE_NAMES),
        "visible_stress": list(VISIBLE_STRESS_FEATURE_NAMES),
        "placebo_8": list(PLACEBO_FEATURE_NAMES),
        "arm_a_plus_s_plan": list(ARM_A_PLUS_S_PLAN_FEATURE_NAMES),
    }


def _parse_feature_orders(value: object) -> None:
    payload = _require_exact_keys(value, _FEATURE_ORDER_KEYS, context="feature_orders")
    expected = _feature_orders_payload()
    for head, names in expected.items():
        observed = payload[head]
        if (
            not isinstance(observed, list)
            or any(not isinstance(item, str) for item in observed)
            or observed != names
        ):
            raise V015TrainingError(f"feature_orders.{head} changed")


def _parse_risk(risk_value: object, comparator_value: object) -> RiskDevelopmentState:
    risk = _require_exact_keys(risk_value, _RISK_STATE_KEYS, context="risk_states")
    comparators = _require_exact_keys(
        comparator_value,
        _COMPARATOR_STATE_KEYS,
        context="comparator_states",
    )
    cluster_count = _integer(
        risk["development_cluster_count"],
        context="risk_states.development_cluster_count",
    )
    eligible_count = _integer(
        risk["eligible_cluster_count"],
        context="risk_states.eligible_cluster_count",
    )
    positive = _integer(
        risk["positive_label_count"],
        context="risk_states.positive_label_count",
    )
    negative = _integer(
        risk["negative_label_count"],
        context="risk_states.negative_label_count",
    )
    threshold = _real(
        risk["catastrophic_threshold_pp"],
        context="risk_states.catastrophic_threshold_pp",
    )
    if (
        cluster_count != RISK_DEVELOPMENT_COUNT
        or not 0 < eligible_count <= cluster_count
        or positive + negative != eligible_count
        or positive < MINIMUM_CLASS_COUNT
        or negative < MINIMUM_CLASS_COUNT
        or threshold != CATASTROPHIC_ERROR_THRESHOLD_PP
    ):
        raise V015TrainingError("risk development counts or threshold changed")
    single = _require_exact_keys(
        comparators["strongest_single_feature"],
        _SINGLE_FEATURE_KEYS,
        context="comparator_states.strongest_single_feature",
    )
    feature_name = _string(
        single["feature_name"],
        context="comparator_states.strongest_single_feature.feature_name",
    )
    orientation = _integer(
        single["danger_orientation"],
        context=("comparator_states.strongest_single_feature.danger_orientation"),
    )
    auroc = _real(
        single["oriented_empirical_auroc"],
        context=("comparator_states.strongest_single_feature.oriented_empirical_auroc"),
    )
    if (
        feature_name not in PREFIX_FEATURE_NAMES
        or orientation not in {-1, 1}
        or not 0.5 <= auroc <= 1.0
    ):
        raise V015TrainingError("strongest single-feature state is invalid")
    return RiskDevelopmentState(
        prefix_only_risk=_parse_logistic(
            risk["prefix_only"],
            context="risk_states.prefix_only",
            expected_names=PREFIX_FEATURE_NAMES,
        ),
        visible_stress_risk=_parse_logistic(
            risk["visible_stress"],
            context="risk_states.visible_stress",
            expected_names=VISIBLE_STRESS_FEATURE_NAMES,
        ),
        placebo_risk=_parse_logistic(
            comparators["placebo_8"],
            context="comparator_states.placebo_8",
            expected_names=PLACEBO_FEATURE_NAMES,
        ),
        arm_a_plus_s_plan_risk=_parse_logistic(
            comparators["arm_a_plus_s_plan"],
            context="comparator_states.arm_a_plus_s_plan",
            expected_names=ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
        ),
        strongest_single_feature_name=feature_name,
        strongest_single_feature_orientation=orientation,
        strongest_single_feature_auroc=auroc,
        development_cluster_count=cluster_count,
        eligible_cluster_count=eligible_count,
        positive_label_count=positive,
        negative_label_count=negative,
    )


def _calibration_payload(
    state: CalibrationDevelopmentState,
) -> dict[str, Any]:
    return {
        "calibration_cluster_count": state.calibration_cluster_count,
        "positive_label_count": state.positive_label_count,
        "negative_label_count": state.negative_label_count,
        "prefix_only_isotonic": _isotonic_payload(state.prefix_only_isotonic),
        "visible_stress_isotonic": _isotonic_payload(state.visible_stress_isotonic),
        "conformal": _conformal_payload(state.conformal),
        "selected_mean_baseline": state.selected_mean_baseline,
        "mean_baseline_iae_pp": state.baseline_iae_by_id(),
    }


def _parse_calibration(value: object) -> CalibrationDevelopmentState:
    payload = _require_exact_keys(
        value, _CALIBRATION_STATE_KEYS, context="calibration_state"
    )
    count = _integer(
        payload["calibration_cluster_count"],
        context="calibration_state.calibration_cluster_count",
    )
    positive = _integer(
        payload["positive_label_count"],
        context="calibration_state.positive_label_count",
    )
    negative = _integer(
        payload["negative_label_count"],
        context="calibration_state.negative_label_count",
    )
    if (
        count != CALIBRATION_COUNT
        or positive + negative != count
        or positive < MINIMUM_CLASS_COUNT
        or negative < MINIMUM_CLASS_COUNT
    ):
        raise V015TrainingError("calibration counts changed")
    baseline_value = payload["mean_baseline_iae_pp"]
    if not isinstance(baseline_value, Mapping) or set(baseline_value) != set(
        MEAN_BASELINE_IDS
    ):
        raise V015TrainingError("calibration_state.mean_baseline_iae_pp keys changed")
    baseline_iae = {
        model_id: _real(
            baseline_value[model_id],
            context=f"calibration_state.mean_baseline_iae_pp.{model_id}",
        )
        for model_id in MEAN_BASELINE_IDS
    }
    if any(value < 0.0 for value in baseline_iae.values()):
        raise V015TrainingError("mean baseline IAE cannot be negative")
    selected = _string(
        payload["selected_mean_baseline"],
        context="calibration_state.selected_mean_baseline",
    )
    expected_selected = min(baseline_iae, key=lambda item: (baseline_iae[item], item))
    if selected != expected_selected:
        raise V015TrainingError(
            "selected mean baseline is not the frozen metric/lexical winner"
        )
    return CalibrationDevelopmentState(
        prefix_only_isotonic=_parse_isotonic(
            payload["prefix_only_isotonic"],
            context="calibration_state.prefix_only_isotonic",
        ),
        visible_stress_isotonic=_parse_isotonic(
            payload["visible_stress_isotonic"],
            context="calibration_state.visible_stress_isotonic",
        ),
        conformal=_parse_conformal(payload["conformal"]),
        selected_mean_baseline=selected,
        mean_baseline_iae_pp=tuple(
            (model_id, baseline_iae[model_id]) for model_id in MEAN_BASELINE_IDS
        ),
        calibration_cluster_count=count,
        positive_label_count=positive,
        negative_label_count=negative,
    )


def _validate_protocol_identity(payload: Mapping[str, Any], *, context: str) -> None:
    if payload.get("protocol_id") != FROZEN_PROTOCOL_ID:
        raise V015TrainingError(f"{context} protocol_id changed")
    if payload.get("config_sha256") != FROZEN_CONFIG_BYTE_SHA256:
        raise V015TrainingError(
            f"{context} config_sha256 must be the frozen config byte hash"
        )


def build_model_state_payload(
    training_state: FrozenTrainingState,
    *,
    center_development_input_hashes: Mapping[str, str],
    risk_development_input_hashes: Mapping[str, str],
    calibration_input_hashes: Mapping[str, str],
    software_versions: Mapping[str, str],
    created_utc: str,
) -> dict[str, Any]:
    """Build the exact nested ``model_state.json`` payload."""

    # This validates every numeric state before it can be serialized.
    construct_frozen_label_free_state(
        training_state.center,
        training_state.risk,
        training_state.calibration,
    )
    risk_payload, comparator_payload = _risk_payloads(training_state.risk)
    payload: dict[str, Any] = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "center_state": _center_payload(training_state.center),
        "risk_states": risk_payload,
        "calibration_state": _calibration_payload(training_state.calibration),
        "comparator_states": comparator_payload,
        "feature_orders": _feature_orders_payload(),
        "input_byte_hashes": {
            "center_development": _validate_hash_mapping(
                center_development_input_hashes,
                context="center_development_input_hashes",
            ),
            "risk_development": _validate_hash_mapping(
                risk_development_input_hashes,
                context="risk_development_input_hashes",
            ),
            "calibration": _validate_hash_mapping(
                calibration_input_hashes,
                context="calibration_input_hashes",
            ),
        },
        "software_versions": _validate_software_versions(software_versions),
        "created_utc": _utc_string(created_utc),
    }
    # Parse the produced object as a guard against malformed manually-created
    # dataclass instances or a future payload/decoder drift.
    validate_model_state_payload(payload)
    return payload


def validate_model_state_payload(payload: object) -> DecodedModelState:
    """Validate every top-level and nested model-state field."""

    top = _require_exact_keys(payload, _MODEL_STATE_KEYS, context="model_state.json")
    if _MODEL_STATE_KEYS != _CONTRACT.json_keys("model_state.json"):
        raise V015TrainingError("training codec and frozen artifact allowlist disagree")
    _validate_protocol_identity(top, context="model_state.json")
    _parse_feature_orders(top["feature_orders"])
    center = _parse_center(top["center_state"])
    risk = _parse_risk(top["risk_states"], top["comparator_states"])
    calibration = _parse_calibration(top["calibration_state"])
    frozen_state = construct_frozen_label_free_state(center, risk, calibration)
    input_hashes = _validate_input_byte_hashes(top["input_byte_hashes"])
    versions = _validate_software_versions(top["software_versions"])
    created = _utc_string(top["created_utc"])
    return DecodedModelState(
        training_state=FrozenTrainingState(
            center=center,
            risk=risk,
            calibration=calibration,
        ),
        frozen_label_free_state=frozen_state,
        input_byte_hashes=input_hashes,
        software_versions=versions,
        created_utc=created,
    )


def serialize_model_state_json(
    training_state: FrozenTrainingState,
    *,
    center_development_input_hashes: Mapping[str, str],
    risk_development_input_hashes: Mapping[str, str],
    calibration_input_hashes: Mapping[str, str],
    software_versions: Mapping[str, str],
    created_utc: str,
) -> bytes:
    """Serialize canonical deterministic ``model_state.json`` bytes."""

    payload = build_model_state_payload(
        training_state,
        center_development_input_hashes=center_development_input_hashes,
        risk_development_input_hashes=risk_development_input_hashes,
        calibration_input_hashes=calibration_input_hashes,
        software_versions=software_versions,
        created_utc=created_utc,
    )
    return canonical_json_bytes(payload)


def _reject_json_constant(token: str) -> None:
    raise V015TrainingError(f"model_state.json contains {token}")


def deserialize_model_state_json(raw: bytes) -> DecodedModelState:
    """Decode canonical bytes and reject semantic or byte-level tampering."""

    if not isinstance(raw, bytes):
        raise V015TrainingError("model_state.json input must be bytes")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V015TrainingError("model_state.json is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise V015TrainingError("model_state.json root must be an object")
    try:
        canonical = canonical_json_bytes(payload)
    except Exception as exc:
        raise V015TrainingError("model_state.json is not finite JSON") from exc
    if canonical != raw:
        raise V015TrainingError("model_state.json bytes are not canonical")
    decoded = validate_model_state_payload(payload)
    normalized = build_model_state_payload(
        decoded.training_state,
        center_development_input_hashes=(
            decoded.input_byte_hashes["center_development"]
        ),
        risk_development_input_hashes=(decoded.input_byte_hashes["risk_development"]),
        calibration_input_hashes=decoded.input_byte_hashes["calibration"],
        software_versions=decoded.software_versions,
        created_utc=decoded.created_utc,
    )
    if canonical_json_bytes(normalized) != raw:
        raise V015TrainingError(
            "model_state.json is canonical JSON but not canonical frozen state"
        )
    return decoded


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_manifest_allowlist(
    payload: object, *, filename: str
) -> Mapping[str, Any]:
    expected = _CONTRACT.json_keys(filename)
    value = _require_exact_keys(payload, expected, context=filename)
    _validate_protocol_identity(value, context=filename)
    _utc_string(value["created_utc"], context=f"{filename}.created_utc")
    return value


def _validate_digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise V015TrainingError(f"{context} must be lowercase SHA256")
    return value


def _center_commitment_payload(
    state: CenterDevelopmentState,
) -> dict[str, Any]:
    return _center_payload(state)


def _risk_commitment_payload(
    state: RiskDevelopmentState,
) -> dict[str, Any]:
    risk, comparators = _risk_payloads(state)
    return {
        "risk_states": risk,
        "comparator_states": comparators,
        "feature_orders": _feature_orders_payload(),
    }


def _isotonic_commitment_payload(
    state: CalibrationDevelopmentState,
) -> dict[str, Any]:
    return {
        "prefix_only_isotonic": _isotonic_payload(state.prefix_only_isotonic),
        "visible_stress_isotonic": _isotonic_payload(state.visible_stress_isotonic),
    }


def center_state_sha256(state: CenterDevelopmentState) -> str:
    """Hash the exact center substate committed before D-risk labels exist."""

    payload = _center_commitment_payload(state)
    _parse_center(payload)
    return _sha256_payload(payload)


def risk_state_sha256(state: RiskDevelopmentState) -> str:
    """Hash both primary heads, comparators, and frozen feature orders."""

    payload = _risk_commitment_payload(state)
    _parse_feature_orders(payload["feature_orders"])
    _parse_risk(payload["risk_states"], payload["comparator_states"])
    return _sha256_payload(payload)


def isotonic_state_sha256(state: CalibrationDevelopmentState) -> str:
    """Hash the two frozen isotonic calibration maps together."""

    _parse_calibration(_calibration_payload(state))
    return _sha256_payload(_isotonic_commitment_payload(state))


def conformal_state_sha256(state: CalibrationDevelopmentState) -> str:
    """Hash the frozen simultaneous conformal expansion substate."""

    _parse_calibration(_calibration_payload(state))
    return _sha256_payload(_conformal_payload(state.conformal))


def build_training_manifest(
    *,
    center_development_input_hashes: Mapping[str, str],
    risk_development_input_hashes: Mapping[str, str],
    center_state: CenterDevelopmentState,
    risk_state: RiskDevelopmentState,
    created_utc: str,
) -> dict[str, Any]:
    """Build the exact training manifest after D-center and D-risk."""

    _parse_center(_center_commitment_payload(center_state))
    risk_payload, comparator_payload = _risk_payloads(risk_state)
    _parse_risk(risk_payload, comparator_payload)
    payload: dict[str, Any] = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "center_development_input_hashes": _validate_hash_mapping(
            center_development_input_hashes,
            context="center_development_input_hashes",
        ),
        "risk_development_input_hashes": _validate_hash_mapping(
            risk_development_input_hashes,
            context="risk_development_input_hashes",
        ),
        "opened_truth_files": [
            "center_development_truth.csv",
            "risk_development_truth.csv",
        ],
        "forbidden_v1_evidence_matches": [],
        "center_state_sha256": center_state_sha256(center_state),
        "risk_state_sha256": risk_state_sha256(risk_state),
        "created_utc": _utc_string(created_utc),
    }
    validate_training_manifest(payload)
    verify_training_manifest_state_hashes(
        payload, center_state=center_state, risk_state=risk_state
    )
    return payload


def validate_training_manifest(payload: object) -> None:
    value = _validate_manifest_allowlist(payload, filename="training_manifest.json")
    _validate_hash_mapping(
        value["center_development_input_hashes"],
        context="training_manifest.center_development_input_hashes",
    )
    _validate_hash_mapping(
        value["risk_development_input_hashes"],
        context="training_manifest.risk_development_input_hashes",
    )
    if value["opened_truth_files"] != [
        "center_development_truth.csv",
        "risk_development_truth.csv",
    ]:
        raise V015TrainingError(
            "training manifest opened_truth_files changed phase scope/order"
        )
    if value["forbidden_v1_evidence_matches"] != []:
        raise V015TrainingError("training manifest records forbidden V1 evidence reuse")
    _validate_digest(
        value["center_state_sha256"],
        context="training_manifest.center_state_sha256",
    )
    _validate_digest(
        value["risk_state_sha256"],
        context="training_manifest.risk_state_sha256",
    )


def verify_training_manifest_state_hashes(
    payload: object,
    *,
    center_state: CenterDevelopmentState,
    risk_state: RiskDevelopmentState,
) -> None:
    """Bind a structurally valid training manifest to the supplied states."""

    validate_training_manifest(payload)
    value = payload
    if not isinstance(value, Mapping):
        raise V015TrainingError("training_manifest.json must be an object")
    if value["center_state_sha256"] != center_state_sha256(center_state):
        raise V015TrainingError("training manifest center state hash mismatch")
    if value["risk_state_sha256"] != risk_state_sha256(risk_state):
        raise V015TrainingError("training manifest risk state hash mismatch")


def build_calibration_manifest(
    *,
    calibration_input_hashes: Mapping[str, str],
    calibration_state: CalibrationDevelopmentState,
    created_utc: str,
) -> dict[str, Any]:
    """Build the exact calibration manifest with cumulative opened truth."""

    _parse_calibration(_calibration_payload(calibration_state))
    payload: dict[str, Any] = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "config_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "calibration_input_hashes": _validate_hash_mapping(
            calibration_input_hashes,
            context="calibration_input_hashes",
        ),
        "opened_truth_files": [
            "calibration_truth.csv",
            "center_development_truth.csv",
            "risk_development_truth.csv",
        ],
        "isotonic_state_sha256": isotonic_state_sha256(calibration_state),
        "conformal_state_sha256": conformal_state_sha256(calibration_state),
        "selected_mean_baseline": (calibration_state.selected_mean_baseline),
        "created_utc": _utc_string(created_utc),
    }
    validate_calibration_manifest(payload)
    verify_calibration_manifest_state_hashes(
        payload, calibration_state=calibration_state
    )
    return payload


def validate_calibration_manifest(payload: object) -> None:
    value = _validate_manifest_allowlist(payload, filename="calibration_manifest.json")
    _validate_hash_mapping(
        value["calibration_input_hashes"],
        context="calibration_manifest.calibration_input_hashes",
    )
    if value["opened_truth_files"] != [
        "calibration_truth.csv",
        "center_development_truth.csv",
        "risk_development_truth.csv",
    ]:
        raise V015TrainingError(
            "calibration manifest opened_truth_files changed phase scope/order"
        )
    _validate_digest(
        value["isotonic_state_sha256"],
        context="calibration_manifest.isotonic_state_sha256",
    )
    _validate_digest(
        value["conformal_state_sha256"],
        context="calibration_manifest.conformal_state_sha256",
    )
    selected = _string(
        value["selected_mean_baseline"],
        context="calibration_manifest.selected_mean_baseline",
    )
    if selected not in MEAN_BASELINE_IDS:
        raise V015TrainingError("calibration manifest selected_mean_baseline changed")


def verify_calibration_manifest_state_hashes(
    payload: object,
    *,
    calibration_state: CalibrationDevelopmentState,
) -> None:
    """Bind a structurally valid calibration manifest to its frozen state."""

    validate_calibration_manifest(payload)
    value = payload
    if not isinstance(value, Mapping):
        raise V015TrainingError("calibration_manifest.json must be an object")
    if value["isotonic_state_sha256"] != isotonic_state_sha256(calibration_state):
        raise V015TrainingError("calibration manifest isotonic state hash mismatch")
    if value["conformal_state_sha256"] != conformal_state_sha256(calibration_state):
        raise V015TrainingError("calibration manifest conformal state hash mismatch")
    if value["selected_mean_baseline"] != calibration_state.selected_mean_baseline:
        raise V015TrainingError("calibration manifest selected mean baseline mismatch")
