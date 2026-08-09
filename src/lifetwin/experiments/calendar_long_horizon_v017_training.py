"""V2.2 calibration-pool split without changing the frozen V0.15 code.

The V2.2 rule separates the rows used for probability calibration from the
rows used for mean-baseline selection and simultaneous conformal calibration.
The isotonic maps share one label-free hard-eligibility mask.  Baseline and
conformal calculations retain the complete 900-row calibration source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    FORECAST_DIMENSION,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    V2ModelError,
    coordinatewise_weighted_quantile,
    family_balanced_support,
    fit_conformal_expansion,
    fit_isotonic_state,
    simultaneous_nonconformity_scores,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    DECLARED_STRUCTURE_FAMILIES,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import (
    CALIBRATION_COUNT,
    CATASTROPHIC_ERROR_THRESHOLD_PP,
    CONFORMAL_COVERAGE,
    CONFORMAL_ORDER_STATISTIC_INDEX,
    MEAN_BASELINE_IDS,
    MINIMUM_CLASS_COUNT,
    CalibrationDevelopmentState,
    RiskDevelopmentState,
)
from lifetwin.experiments.calendar_long_horizon_v017_signals import (
    V022CalibrationTerminalInconclusive,
)


V022_MINIMUM_ELIGIBLE_FRACTION = 0.95
V022_MINIMUM_ELIGIBLE_COUNT = math.ceil(
    V022_MINIMUM_ELIGIBLE_FRACTION * CALIBRATION_COUNT
)
_MAXIMUM_STRUCTURE_FAMILY_COUNT = 7
_MASK_HASH_DOMAIN = b"lifetwin-v022-calibration-mask-v2\0"
_ROW_HASH_DOMAIN = b"lifetwin-v022-calibration-row-v2\0"
_SUPPORT_HASH_DOMAIN = b"lifetwin-v022-structural-support-v1\0"
_V022_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLUSTER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LABEL_FREE_INELIGIBILITY_REASONS = frozenset(
    {
        "invalid_prefix_grid_or_observations",
        "invalid_forecast_coordinates",
        "insufficient_structure_families",
        "nonfinite_center_forecast",
        "nonfinite_prefix_features",
        "nonfinite_real_operating_fields",
        "nonfinite_placebo_operating_fields",
        "nonfinite_real_operating_features",
        "nonfinite_placebo_features",
        "nonfinite_primary_risk_scores",
    }
)


class V022CalibrationError(ValueError):
    """Raised when V2.2 calibration inputs violate their typed contract."""


@dataclass(frozen=True)
class V022CommittedMaskRow:
    """One canonical, identity-bound row in a pretruth mask commitment."""

    cluster_id: str
    label_free_row_sha256: str
    structural_support_sha256: str
    successful_structure_family_ids: tuple[str, ...]
    eligible: bool
    ineligibility_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if _CLUSTER_ID.fullmatch(self.cluster_id) is None:
            raise ValueError("Committed cluster_id is invalid")
        for value in (self.label_free_row_sha256, self.structural_support_sha256):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("Committed row digest is invalid")
        if (
            not isinstance(self.successful_structure_family_ids, tuple)
            or self.successful_structure_family_ids
            != tuple(sorted(set(self.successful_structure_family_ids)))
            or not set(self.successful_structure_family_ids).issubset(
                DECLARED_STRUCTURE_FAMILIES
            )
        ):
            raise ValueError("Committed structure-family IDs are invalid")
        if not isinstance(self.eligible, bool):
            raise TypeError("Committed eligible flag must be a strict boolean")
        if (
            not isinstance(self.ineligibility_reasons, tuple)
            or self.ineligibility_reasons
            != tuple(sorted(set(self.ineligibility_reasons)))
            or not set(self.ineligibility_reasons).issubset(
                _LABEL_FREE_INELIGIBILITY_REASONS
            )
        ):
            raise ValueError("Committed ineligibility reasons are invalid")
        if self.eligible == bool(self.ineligibility_reasons):
            raise ValueError("Committed eligibility and reasons disagree")


@dataclass(frozen=True)
class V022PretruthMaskCommitment:
    """Immutable exact-row mask commitment created without calibration truth."""

    protocol_id: str
    source_calibration_count: int
    rows: tuple[V022CommittedMaskRow, ...]
    eligibility_mask_sha256: str

    def __post_init__(self) -> None:
        if self.protocol_id != _V022_PROTOCOL_ID:
            raise ValueError("Pretruth mask protocol_id is invalid")
        if self.source_calibration_count != CALIBRATION_COUNT:
            raise ValueError("Pretruth mask source count must be 900")
        if (
            not isinstance(self.rows, tuple)
            or len(self.rows) != CALIBRATION_COUNT
            or self.rows != tuple(sorted(self.rows, key=lambda row: row.cluster_id))
            or len({row.cluster_id for row in self.rows}) != CALIBRATION_COUNT
        ):
            raise ValueError("Pretruth mask rows must be 900 unique canonical rows")
        if _SHA256.fullmatch(self.eligibility_mask_sha256) is None:
            raise ValueError("Pretruth mask digest is invalid")

    @property
    def eligible_count(self) -> int:
        return sum(row.eligible for row in self.rows)

    def canonical_bytes(self) -> bytes:
        payload = {
            "schema_version": "1.0.0",
            "protocol_id": self.protocol_id,
            "source_calibration_count": self.source_calibration_count,
            "risk_isotonic_eligible_count": self.eligible_count,
            "eligibility_mask_sha256": self.eligibility_mask_sha256,
            "rows": [
                {
                    "cluster_id": row.cluster_id,
                    "label_free_row_sha256": row.label_free_row_sha256,
                    "structural_support_sha256": row.structural_support_sha256,
                    "successful_structure_family_ids": list(
                        row.successful_structure_family_ids
                    ),
                    "eligible": row.eligible,
                    "ineligibility_reasons": list(row.ineligibility_reasons),
                }
                for row in self.rows
            ],
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"{encoded}\n".encode("ascii")

    @property
    def canonical_byte_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class V022CalibrationAudit:
    """Immutable evidence for the split isotonic/conformal calibration pools."""

    source_calibration_count: int
    risk_isotonic_eligible_count: int
    risk_isotonic_ineligible_zero_family_count: int
    risk_isotonic_ineligible_one_family_count: int
    risk_isotonic_ineligible_other_count: int
    risk_isotonic_positive_label_count: int
    risk_isotonic_negative_label_count: int
    mean_baseline_count: int
    conformal_calibration_count: int
    conformal_order_statistic_index: int
    eligibility_mask_sha256: str


@dataclass(frozen=True)
class _ValidatedPretruthEvidence:
    commitment: V022PretruthMaskCommitment
    cluster_ids: tuple[str, ...]
    eligible: np.ndarray
    ineligibility_reasons: tuple[tuple[str, ...], ...]
    family_counts: np.ndarray
    prefix_features: np.ndarray
    real_stress_features: np.ndarray
    center_forecasts: np.ndarray
    raw_prefix_scores: np.ndarray
    raw_visible_scores: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    baseline_forecasts: Mapping[str, np.ndarray]


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
        raise V022CalibrationError(f"{name} must be a numeric matrix") from exc
    if matrix.shape != (rows, columns):
        raise V022CalibrationError(
            f"{name} must have shape ({rows}, {columns}), got {matrix.shape}"
        )
    if require_finite and not np.isfinite(matrix).all():
        raise V022CalibrationError(f"{name} must contain only finite values")
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
        raise V022CalibrationError(f"{name} must be a numeric vector") from exc
    if vector.shape != (length,):
        raise V022CalibrationError(f"{name} must contain exactly {length} values")
    if require_finite and not np.isfinite(vector).all():
        raise V022CalibrationError(f"{name} must contain only finite values")
    return vector


def _strict_family_counts(values: Sequence[int]) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (CALIBRATION_COUNT,) or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise V022CalibrationError(
            "successful_structure_family_count must contain exactly 900 strict integers"
        )
    counts = raw.astype(np.int64)
    if np.any((counts < 0) | (counts > _MAXIMUM_STRUCTURE_FAMILY_COUNT)):
        raise V022CalibrationError(
            "successful_structure_family_count must be in [0, 7]"
        )
    return counts


def _strict_text_vector(
    values: Sequence[str],
    *,
    name: str,
    pattern: re.Pattern[str],
    require_unique: bool = False,
) -> tuple[str, ...]:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (CALIBRATION_COUNT,) or any(
        not isinstance(value, str) or pattern.fullmatch(value) is None for value in raw
    ):
        raise V022CalibrationError(f"{name} must contain exactly 900 valid strings")
    result = tuple(str(value) for value in raw)
    if require_unique and len(set(result)) != CALIBRATION_COUNT:
        raise V022CalibrationError(f"{name} must contain 900 unique values")
    return result


def _require_source_count(**row_aligned: object) -> None:
    mismatches: list[str] = []
    for name, values in row_aligned.items():
        try:
            observed = len(values)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            observed = None
        if observed != CALIBRATION_COUNT:
            mismatches.append(f"{name}={observed}")
    if mismatches:
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_SOURCE_COUNT_NOT_900",
            "V2.2 calibration source count must be 900 for every aligned "
            f"input; observed {', '.join(mismatches)}",
        )


def _row_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f8").reshape(-1).tobytes(order="C")


def _update_sized(hasher: Any, raw: bytes) -> None:
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)


def _support_row(
    value: object,
    *,
    row_index: int,
) -> tuple[tuple[str, ...], str, np.ndarray, np.ndarray]:
    if not isinstance(value, Mapping):
        raise V022CalibrationError(
            f"structural_family_supports_pct[{row_index}] must be a mapping"
        )
    normalized: dict[str, tuple[tuple[float, ...], ...]] = {}
    for family_id, raw_vectors in value.items():
        if (
            not isinstance(family_id, str)
            or family_id not in DECLARED_STRUCTURE_FAMILIES
        ):
            raise V022CalibrationError(
                f"structural_family_supports_pct[{row_index}] has an undeclared family"
            )
        try:
            matrix = np.asarray(raw_vectors, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise V022CalibrationError(
                f"structural_family_supports_pct[{row_index}][{family_id!r}] "
                "must be numeric"
            ) from exc
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 1
            or matrix.shape[1] != FORECAST_DIMENSION
            or not np.isfinite(matrix).all()
        ):
            raise V022CalibrationError(
                f"structural_family_supports_pct[{row_index}][{family_id!r}] "
                "must contain finite eight-horizon support"
            )
        normalized[family_id] = tuple(
            tuple(float(number) for number in vector) for vector in matrix
        )

    family_ids = tuple(sorted(normalized))
    support_hasher = hashlib.sha256()
    support_hasher.update(_SUPPORT_HASH_DOMAIN)
    for family_id in family_ids:
        _update_sized(support_hasher, family_id.encode("ascii"))
        family_vectors, family_weights = family_balanced_support(
            {family_id: normalized[family_id]}
        )
        support_hasher.update(struct.pack("<Q", len(family_vectors)))
        for vector, weight in zip(family_vectors, family_weights, strict=True):
            _update_sized(
                support_hasher,
                _row_bytes(np.asarray(vector, dtype=np.float64)),
            )
            support_hasher.update(struct.pack("<d", float(weight)))

    if not family_ids:
        return (
            family_ids,
            support_hasher.hexdigest(),
            np.full(FORECAST_DIMENSION, np.nan),
            np.full(FORECAST_DIMENSION, np.nan),
        )

    support_vectors, support_weights = family_balanced_support(normalized)
    lower = np.asarray(
        coordinatewise_weighted_quantile(support_vectors, support_weights, 0.05),
        dtype=np.float64,
    )
    upper = np.asarray(
        coordinatewise_weighted_quantile(support_vectors, support_weights, 0.95),
        dtype=np.float64,
    )
    return family_ids, support_hasher.hexdigest(), lower, upper


def _verify_primary_scores(
    *,
    risk_state: RiskDevelopmentState,
    prefix: np.ndarray,
    real_stress: np.ndarray,
    supplied_prefix: np.ndarray,
    supplied_visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_ready = np.isfinite(prefix).all(axis=1)
    visible_ready = prefix_ready & np.isfinite(real_stress).all(axis=1)
    expected_prefix = np.full(CALIBRATION_COUNT, np.nan, dtype=np.float64)
    expected_visible = np.full(CALIBRATION_COUNT, np.nan, dtype=np.float64)
    try:
        expected_prefix[prefix_ready] = risk_state.prefix_only_risk.decision_function(
            prefix[prefix_ready]
        )
        expected_visible[visible_ready] = (
            risk_state.visible_stress_risk.decision_function(
                np.column_stack((prefix[visible_ready], real_stress[visible_ready]))
            )
        )
    except (V2ModelError, ValueError) as exc:
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_RISK_SCORE_NONFINITE",
            "Frozen primary risk scores could not be reproduced before truth access",
        ) from exc

    for name, ready, expected, supplied in (
        ("raw_prefix_risk_scores", prefix_ready, expected_prefix, supplied_prefix),
        ("raw_visible_risk_scores", visible_ready, expected_visible, supplied_visible),
    ):
        if np.any(~np.isnan(supplied[~ready])):
            raise V022CalibrationError(
                f"{name} must use canonical NaN where its feature vector is "
                "not evaluable"
            )
        if (
            not np.isfinite(expected[ready]).all()
            or not np.isfinite(supplied[ready]).all()
            or np.any(expected[ready] != supplied[ready])
        ):
            raise V022CalibrationError(
                f"{name} differs from the frozen risk-state evaluation"
            )
    return expected_prefix, expected_visible


def _derive_pretruth_evidence(
    *,
    risk_state: RiskDevelopmentState,
    cluster_ids: Sequence[str],
    arm_a_predictor_content_sha256: Sequence[str],
    arm_b_predictor_content_sha256: Sequence[str],
    placebo_predictor_content_sha256: Sequence[str],
    prefix_days: Sequence[Sequence[float]],
    prefix_observations_pct: Sequence[Sequence[float]],
    forecast_days: Sequence[Sequence[float]],
    real_operating_fields: Sequence[Sequence[float]],
    placebo_operating_fields: Sequence[Sequence[float]],
    real_stress_features: Sequence[Sequence[float]],
    placebo_features: Sequence[Sequence[float]],
    successful_structure_family_count: Sequence[int],
    structural_family_supports_pct: Sequence[Mapping[str, Sequence[Sequence[float]]]],
    frozen_center_forecasts_pct: Sequence[Sequence[float]],
    prefix_features: Sequence[Sequence[float]],
    raw_prefix_risk_scores: Sequence[float],
    raw_visible_risk_scores: Sequence[float],
    base_interval_lower_pct: Sequence[Sequence[float]],
    base_interval_upper_pct: Sequence[Sequence[float]],
    mean_baseline_forecasts_pct: Mapping[str, Sequence[Sequence[float]]],
) -> _ValidatedPretruthEvidence:
    if not isinstance(risk_state, RiskDevelopmentState):
        raise V022CalibrationError("risk_state must be a RiskDevelopmentState")
    _require_source_count(
        cluster_ids=cluster_ids,
        arm_a_predictor_content_sha256=arm_a_predictor_content_sha256,
        arm_b_predictor_content_sha256=arm_b_predictor_content_sha256,
        placebo_predictor_content_sha256=placebo_predictor_content_sha256,
        prefix_days=prefix_days,
        prefix_observations_pct=prefix_observations_pct,
        forecast_days=forecast_days,
        real_operating_fields=real_operating_fields,
        placebo_operating_fields=placebo_operating_fields,
        real_stress_features=real_stress_features,
        placebo_features=placebo_features,
        successful_structure_family_count=successful_structure_family_count,
        structural_family_supports_pct=structural_family_supports_pct,
        frozen_center_forecasts_pct=frozen_center_forecasts_pct,
        prefix_features=prefix_features,
        raw_prefix_risk_scores=raw_prefix_risk_scores,
        raw_visible_risk_scores=raw_visible_risk_scores,
        base_interval_lower_pct=base_interval_lower_pct,
        base_interval_upper_pct=base_interval_upper_pct,
    )
    identifiers = _strict_text_vector(
        cluster_ids,
        name="cluster_ids",
        pattern=_CLUSTER_ID,
        require_unique=True,
    )
    predictor_hashes = (
        _strict_text_vector(
            arm_a_predictor_content_sha256,
            name="arm_a_predictor_content_sha256",
            pattern=_SHA256,
        ),
        _strict_text_vector(
            arm_b_predictor_content_sha256,
            name="arm_b_predictor_content_sha256",
            pattern=_SHA256,
        ),
        _strict_text_vector(
            placebo_predictor_content_sha256,
            name="placebo_predictor_content_sha256",
            pattern=_SHA256,
        ),
    )
    family_counts = _strict_family_counts(successful_structure_family_count)
    prefix_day_matrix = _numeric_matrix(
        prefix_days,
        name="prefix_days",
        rows=CALIBRATION_COUNT,
        columns=len(PREFIX_DAYS),
        require_finite=False,
    )
    prefix_observations = _numeric_matrix(
        prefix_observations_pct,
        name="prefix_observations_pct",
        rows=CALIBRATION_COUNT,
        columns=len(PREFIX_DAYS),
        require_finite=False,
    )
    forecast_day_matrix = _numeric_matrix(
        forecast_days,
        name="forecast_days",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
        require_finite=False,
    )
    real_operating = _numeric_matrix(
        real_operating_fields,
        name="real_operating_fields",
        rows=CALIBRATION_COUNT,
        columns=len(REAL_OPERATING_FIELDS),
        require_finite=False,
    )
    placebo_operating = _numeric_matrix(
        placebo_operating_fields,
        name="placebo_operating_fields",
        rows=CALIBRATION_COUNT,
        columns=len(PLACEBO_FIELDS),
        require_finite=False,
    )
    real_stress = _numeric_matrix(
        real_stress_features,
        name="real_stress_features",
        rows=CALIBRATION_COUNT,
        columns=len(REAL_OPERATING_FIELDS),
        require_finite=False,
    )
    placebo_feature_matrix = _numeric_matrix(
        placebo_features,
        name="placebo_features",
        rows=CALIBRATION_COUNT,
        columns=len(PLACEBO_FIELDS),
        require_finite=False,
    )
    prefix = _numeric_matrix(
        prefix_features,
        name="prefix_features",
        rows=CALIBRATION_COUNT,
        columns=len(PREFIX_FEATURE_NAMES),
        require_finite=False,
    )
    center = _numeric_matrix(
        frozen_center_forecasts_pct,
        name="frozen_center_forecasts_pct",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
        require_finite=False,
    )
    raw_prefix = _numeric_vector(
        raw_prefix_risk_scores,
        name="raw_prefix_risk_scores",
        length=CALIBRATION_COUNT,
        require_finite=False,
    )
    raw_visible = _numeric_vector(
        raw_visible_risk_scores,
        name="raw_visible_risk_scores",
        length=CALIBRATION_COUNT,
        require_finite=False,
    )
    lower = _numeric_matrix(
        base_interval_lower_pct,
        name="base_interval_lower_pct",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
        require_finite=False,
    )
    upper = _numeric_matrix(
        base_interval_upper_pct,
        name="base_interval_upper_pct",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
        require_finite=False,
    )
    if not isinstance(mean_baseline_forecasts_pct, Mapping) or set(
        mean_baseline_forecasts_pct
    ) != set(MEAN_BASELINE_IDS):
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_BASELINE_INCOMPLETE",
            "mean_baseline_forecasts_pct must contain exactly the three "
            "frozen baseline IDs",
        )
    try:
        baseline_forecasts = {
            model_id: _numeric_matrix(
                mean_baseline_forecasts_pct[model_id],
                name=f"mean_baseline_forecasts_pct[{model_id!r}]",
                rows=CALIBRATION_COUNT,
                columns=FORECAST_DIMENSION,
            )
            for model_id in MEAN_BASELINE_IDS
        }
    except V022CalibrationError as exc:
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_BASELINE_INCOMPLETE",
            "A required 900-row mean-baseline trajectory is incomplete",
        ) from exc

    finite_count_columns = np.isfinite(prefix[:, :2]).all(axis=1)
    if np.any(
        finite_count_columns
        & (
            (prefix[:, 0] != family_counts)
            | (prefix[:, 1] != _MAXIMUM_STRUCTURE_FAMILY_COUNT - family_counts)
        )
    ):
        raise V022CalibrationError(
            "prefix family-count features differ from committed family support"
        )
    raw_prefix, raw_visible = _verify_primary_scores(
        risk_state=risk_state,
        prefix=prefix,
        real_stress=real_stress,
        supplied_prefix=raw_prefix,
        supplied_visible=raw_visible,
    )

    support_rows: list[tuple[tuple[str, ...], str, np.ndarray, np.ndarray]] = []
    for row_index, raw_support in enumerate(structural_family_supports_pct):
        support_rows.append(_support_row(raw_support, row_index=row_index))
    derived_counts = np.asarray(
        [len(family_ids) for family_ids, _, _, _ in support_rows],
        dtype=np.int64,
    )
    if np.any(derived_counts != family_counts):
        raise V022CalibrationError(
            "successful_structure_family_count differs from committed support"
        )
    zero_family = family_counts == 0
    if np.any(zero_family):
        indices = tuple(int(index) for index in np.flatnonzero(zero_family))
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_ZERO_FAMILY_NO_BAND",
            "A zero-family row cannot supply a finite calibration band",
            offending_row_indices=indices,
        )
    _terminal_nonfinite_band(lower, upper)
    derived_lower = np.vstack([row[2] for row in support_rows])
    derived_upper = np.vstack([row[3] for row in support_rows])
    if not np.array_equal(lower, derived_lower) or not np.array_equal(
        upper, derived_upper
    ):
        raise V022CalibrationError(
            "base intervals are not the exact family-balanced support quantiles"
        )

    expected_prefix_days = np.asarray(PREFIX_DAYS, dtype=np.float64)
    expected_forecast_days = np.asarray(FORECAST_DAYS, dtype=np.float64)
    prefix_grid_ok = np.isfinite(prefix_day_matrix).all(axis=1) & np.isfinite(
        prefix_observations
    ).all(axis=1)
    prefix_grid_ok &= np.all(prefix_day_matrix == expected_prefix_days, axis=1)
    forecast_grid_ok = np.isfinite(forecast_day_matrix).all(axis=1)
    forecast_grid_ok &= np.all(forecast_day_matrix == expected_forecast_days, axis=1)
    center_finite = np.isfinite(center).all(axis=1)
    prefix_finite = np.isfinite(prefix).all(axis=1)
    real_operating_finite = np.isfinite(real_operating).all(axis=1)
    placebo_operating_finite = np.isfinite(placebo_operating).all(axis=1)
    real_stress_finite = np.isfinite(real_stress).all(axis=1)
    placebo_features_finite = np.isfinite(placebo_feature_matrix).all(axis=1)
    primary_scores_finite = np.isfinite(raw_prefix) & np.isfinite(raw_visible)

    reason_rows: list[tuple[str, ...]] = []
    eligible = np.empty(CALIBRATION_COUNT, dtype=bool)
    for index in range(CALIBRATION_COUNT):
        reasons: list[str] = []
        if not prefix_grid_ok[index]:
            reasons.append("invalid_prefix_grid_or_observations")
        if not forecast_grid_ok[index]:
            reasons.append("invalid_forecast_coordinates")
        if family_counts[index] < 2:
            reasons.append("insufficient_structure_families")
        if not center_finite[index]:
            reasons.append("nonfinite_center_forecast")
        if not prefix_finite[index]:
            reasons.append("nonfinite_prefix_features")
        if not real_operating_finite[index]:
            reasons.append("nonfinite_real_operating_fields")
        if not placebo_operating_finite[index]:
            reasons.append("nonfinite_placebo_operating_fields")
        if not real_stress_finite[index]:
            reasons.append("nonfinite_real_operating_features")
        if not placebo_features_finite[index]:
            reasons.append("nonfinite_placebo_features")
        if not primary_scores_finite[index]:
            reasons.append("nonfinite_primary_risk_scores")
        canonical_reasons = tuple(sorted(reasons))
        reason_rows.append(canonical_reasons)
        eligible[index] = not canonical_reasons

    committed_rows: list[V022CommittedMaskRow] = []
    for index, cluster_id in enumerate(identifiers):
        row_hasher = hashlib.sha256()
        row_hasher.update(_ROW_HASH_DOMAIN)
        for text in (
            cluster_id,
            predictor_hashes[0][index],
            predictor_hashes[1][index],
            predictor_hashes[2][index],
            support_rows[index][1],
            *support_rows[index][0],
            *reason_rows[index],
        ):
            _update_sized(row_hasher, text.encode("ascii"))
        row_hasher.update(
            struct.pack(
                "<qB",
                int(family_counts[index]),
                int(eligible[index]),
            )
        )
        for values in (
            prefix_day_matrix[index],
            prefix_observations[index],
            forecast_day_matrix[index],
            real_operating[index],
            placebo_operating[index],
            real_stress[index],
            placebo_feature_matrix[index],
            center[index],
            prefix[index],
            raw_prefix[index : index + 1],
            raw_visible[index : index + 1],
            lower[index],
            upper[index],
            *(baseline_forecasts[model_id][index] for model_id in MEAN_BASELINE_IDS),
        ):
            _update_sized(row_hasher, _row_bytes(np.asarray(values)))
        committed_rows.append(
            V022CommittedMaskRow(
                cluster_id=cluster_id,
                label_free_row_sha256=row_hasher.hexdigest(),
                structural_support_sha256=support_rows[index][1],
                successful_structure_family_ids=support_rows[index][0],
                eligible=bool(eligible[index]),
                ineligibility_reasons=reason_rows[index],
            )
        )

    canonical_rows = tuple(sorted(committed_rows, key=lambda row: row.cluster_id))
    commitment_hasher = hashlib.sha256()
    commitment_hasher.update(_MASK_HASH_DOMAIN)
    commitment_hasher.update(struct.pack("<Q", CALIBRATION_COUNT))
    for row in canonical_rows:
        _update_sized(commitment_hasher, row.cluster_id.encode("ascii"))
        commitment_hasher.update(bytes.fromhex(row.label_free_row_sha256))
        commitment_hasher.update(struct.pack("<B", int(row.eligible)))
    commitment = V022PretruthMaskCommitment(
        protocol_id=_V022_PROTOCOL_ID,
        source_calibration_count=CALIBRATION_COUNT,
        rows=canonical_rows,
        eligibility_mask_sha256=commitment_hasher.hexdigest(),
    )
    if commitment.eligible_count < V022_MINIMUM_ELIGIBLE_COUNT:
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_RISK_ELIGIBLE_BELOW_855",
            "V2.2 isotonic calibration requires at least "
            f"{V022_MINIMUM_ELIGIBLE_COUNT} eligible rows; observed "
            f"{commitment.eligible_count}",
        )
    return _ValidatedPretruthEvidence(
        commitment=commitment,
        cluster_ids=identifiers,
        eligible=eligible,
        ineligibility_reasons=tuple(reason_rows),
        family_counts=family_counts,
        prefix_features=prefix,
        real_stress_features=real_stress,
        center_forecasts=center,
        raw_prefix_scores=raw_prefix,
        raw_visible_scores=raw_visible,
        lower=lower,
        upper=upper,
        baseline_forecasts=MappingProxyType(baseline_forecasts),
    )


def derive_calibration_mask_commitment_v022(
    *,
    risk_state: RiskDevelopmentState,
    cluster_ids: Sequence[str],
    arm_a_predictor_content_sha256: Sequence[str],
    arm_b_predictor_content_sha256: Sequence[str],
    placebo_predictor_content_sha256: Sequence[str],
    prefix_days: Sequence[Sequence[float]],
    prefix_observations_pct: Sequence[Sequence[float]],
    forecast_days: Sequence[Sequence[float]],
    real_operating_fields: Sequence[Sequence[float]],
    placebo_operating_fields: Sequence[Sequence[float]],
    real_stress_features: Sequence[Sequence[float]],
    placebo_features: Sequence[Sequence[float]],
    successful_structure_family_count: Sequence[int],
    structural_family_supports_pct: Sequence[Mapping[str, Sequence[Sequence[float]]]],
    frozen_center_forecasts_pct: Sequence[Sequence[float]],
    prefix_features: Sequence[Sequence[float]],
    raw_prefix_risk_scores: Sequence[float],
    raw_visible_risk_scores: Sequence[float],
    base_interval_lower_pct: Sequence[Sequence[float]],
    base_interval_upper_pct: Sequence[Sequence[float]],
    mean_baseline_forecasts_pct: Mapping[str, Sequence[Sequence[float]]],
) -> V022PretruthMaskCommitment:
    """Derive the exact calibration mask from label-free evidence only."""

    evidence = _derive_pretruth_evidence(
        risk_state=risk_state,
        cluster_ids=cluster_ids,
        arm_a_predictor_content_sha256=arm_a_predictor_content_sha256,
        arm_b_predictor_content_sha256=arm_b_predictor_content_sha256,
        placebo_predictor_content_sha256=placebo_predictor_content_sha256,
        prefix_days=prefix_days,
        prefix_observations_pct=prefix_observations_pct,
        forecast_days=forecast_days,
        real_operating_fields=real_operating_fields,
        placebo_operating_fields=placebo_operating_fields,
        real_stress_features=real_stress_features,
        placebo_features=placebo_features,
        successful_structure_family_count=successful_structure_family_count,
        structural_family_supports_pct=structural_family_supports_pct,
        frozen_center_forecasts_pct=frozen_center_forecasts_pct,
        prefix_features=prefix_features,
        raw_prefix_risk_scores=raw_prefix_risk_scores,
        raw_visible_risk_scores=raw_visible_risk_scores,
        base_interval_lower_pct=base_interval_lower_pct,
        base_interval_upper_pct=base_interval_upper_pct,
        mean_baseline_forecasts_pct=mean_baseline_forecasts_pct,
    )
    return evidence.commitment


def calibration_eligibility_mask_sha256_v022(
    commitment: V022PretruthMaskCommitment,
) -> str:
    """Return the digest of an already-derived immutable pretruth commitment."""

    if not isinstance(commitment, V022PretruthMaskCommitment):
        raise TypeError("commitment must be a V022PretruthMaskCommitment")
    return commitment.eligibility_mask_sha256


def _terminal_nonfinite_band(lower: np.ndarray, upper: np.ndarray) -> None:
    nonfinite = ~np.isfinite(lower).all(axis=1) | ~np.isfinite(upper).all(axis=1)
    if np.any(nonfinite):
        indices = tuple(int(index) for index in np.flatnonzero(nonfinite))
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_BAND_NONFINITE_OR_UNORDERED",
            "At least one of the 900 conformal source rows has a nonfinite band",
            offending_row_indices=indices,
        )
    unordered = np.any(lower > upper, axis=1)
    if np.any(unordered):
        indices = tuple(int(index) for index in np.flatnonzero(unordered))
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_BAND_NONFINITE_OR_UNORDERED",
            "At least one of the 900 conformal source rows has an unordered band",
            offending_row_indices=indices,
        )


def _align_calibration_targets(
    *,
    label_free_cluster_ids: tuple[str, ...],
    latent_target_cluster_ids: Sequence[str],
    latent_target_forecast_days: Sequence[Sequence[float]],
    latent_targets_pct: Sequence[Sequence[float]],
) -> np.ndarray:
    """Join truth rows to label-free rows by exact opaque ID and horizon grid."""

    _require_source_count(
        latent_target_cluster_ids=latent_target_cluster_ids,
        latent_target_forecast_days=latent_target_forecast_days,
        latent_targets_pct=latent_targets_pct,
    )
    target_ids = _strict_text_vector(
        latent_target_cluster_ids,
        name="latent_target_cluster_ids",
        pattern=_CLUSTER_ID,
        require_unique=True,
    )
    expected_ids = set(label_free_cluster_ids)
    if set(target_ids) != expected_ids:
        raise V022CalibrationError(
            "Calibration truth and label-free cluster ID sets differ"
        )
    target_days = _numeric_matrix(
        latent_target_forecast_days,
        name="latent_target_forecast_days",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
    )
    expected_days = np.asarray(FORECAST_DAYS, dtype=np.float64)
    if not np.all(target_days == expected_days):
        raise V022CalibrationError(
            "Calibration truth does not contain the exact forecast grid"
        )
    targets = _numeric_matrix(
        latent_targets_pct,
        name="latent_targets_pct",
        rows=CALIBRATION_COUNT,
        columns=FORECAST_DIMENSION,
    )
    row_by_id = {cluster_id: index for index, cluster_id in enumerate(target_ids)}
    return targets[
        np.asarray(
            [row_by_id[cluster_id] for cluster_id in label_free_cluster_ids],
            dtype=np.int64,
        )
    ]


def fit_calibration_development_state_v022(
    *,
    pretruth_commitment: V022PretruthMaskCommitment,
    risk_state: RiskDevelopmentState,
    cluster_ids: Sequence[str],
    arm_a_predictor_content_sha256: Sequence[str],
    arm_b_predictor_content_sha256: Sequence[str],
    placebo_predictor_content_sha256: Sequence[str],
    prefix_days: Sequence[Sequence[float]],
    prefix_observations_pct: Sequence[Sequence[float]],
    forecast_days: Sequence[Sequence[float]],
    real_operating_fields: Sequence[Sequence[float]],
    placebo_operating_fields: Sequence[Sequence[float]],
    real_stress_features: Sequence[Sequence[float]],
    placebo_features: Sequence[Sequence[float]],
    successful_structure_family_count: Sequence[int],
    structural_family_supports_pct: Sequence[Mapping[str, Sequence[Sequence[float]]]],
    frozen_center_forecasts_pct: Sequence[Sequence[float]],
    prefix_features: Sequence[Sequence[float]],
    raw_prefix_risk_scores: Sequence[float],
    raw_visible_risk_scores: Sequence[float],
    base_interval_lower_pct: Sequence[Sequence[float]],
    base_interval_upper_pct: Sequence[Sequence[float]],
    mean_baseline_forecasts_pct: Mapping[str, Sequence[Sequence[float]]],
    latent_target_cluster_ids: Sequence[str],
    latent_target_forecast_days: Sequence[Sequence[float]],
    latent_targets_pct: Sequence[Sequence[float]],
) -> tuple[CalibrationDevelopmentState, V022CalibrationAudit]:
    """Verify a pretruth mask, then fit isotonic and conformal calibration."""

    evidence = _derive_pretruth_evidence(
        risk_state=risk_state,
        cluster_ids=cluster_ids,
        arm_a_predictor_content_sha256=arm_a_predictor_content_sha256,
        arm_b_predictor_content_sha256=arm_b_predictor_content_sha256,
        placebo_predictor_content_sha256=placebo_predictor_content_sha256,
        prefix_days=prefix_days,
        prefix_observations_pct=prefix_observations_pct,
        forecast_days=forecast_days,
        real_operating_fields=real_operating_fields,
        placebo_operating_fields=placebo_operating_fields,
        real_stress_features=real_stress_features,
        placebo_features=placebo_features,
        successful_structure_family_count=successful_structure_family_count,
        structural_family_supports_pct=structural_family_supports_pct,
        frozen_center_forecasts_pct=frozen_center_forecasts_pct,
        prefix_features=prefix_features,
        raw_prefix_risk_scores=raw_prefix_risk_scores,
        raw_visible_risk_scores=raw_visible_risk_scores,
        base_interval_lower_pct=base_interval_lower_pct,
        base_interval_upper_pct=base_interval_upper_pct,
        mean_baseline_forecasts_pct=mean_baseline_forecasts_pct,
    )
    if not isinstance(pretruth_commitment, V022PretruthMaskCommitment):
        raise V022CalibrationError(
            "pretruth_commitment must be a V022PretruthMaskCommitment"
        )
    if evidence.commitment != pretruth_commitment:
        raise V022CalibrationError(
            "Pretruth mask commitment differs from the current label-free evidence"
        )
    targets = _align_calibration_targets(
        label_free_cluster_ids=evidence.cluster_ids,
        latent_target_cluster_ids=latent_target_cluster_ids,
        latent_target_forecast_days=latent_target_forecast_days,
        latent_targets_pct=latent_targets_pct,
    )
    eligible = evidence.eligible.copy()
    family_counts = evidence.family_counts.copy()
    prefix = evidence.prefix_features.copy()
    visible = evidence.real_stress_features.copy()
    center_25y = evidence.center_forecasts[:, -1].copy()
    arm_a = evidence.raw_prefix_scores.copy()
    arm_b = evidence.raw_visible_scores.copy()
    lower = evidence.lower.copy()
    upper = evidence.upper.copy()
    baseline_forecasts = {
        model_id: evidence.baseline_forecasts[model_id].copy()
        for model_id in MEAN_BASELINE_IDS
    }
    eligible_count = int(np.count_nonzero(eligible))
    order = np.argsort(np.asarray(evidence.cluster_ids, dtype=str), kind="stable")
    prefix = prefix[order]
    visible = visible[order]
    family_counts = family_counts[order]
    eligible = eligible[order]
    center_25y = center_25y[order]
    arm_a = arm_a[order]
    arm_b = arm_b[order]
    targets = targets[order]
    lower = lower[order]
    upper = upper[order]
    baseline_forecasts = {
        model_id: baseline_forecasts[model_id][order] for model_id in MEAN_BASELINE_IDS
    }

    labels = (
        np.abs(center_25y - targets[:, -1]) >= CATASTROPHIC_ERROR_THRESHOLD_PP
    ).astype(np.int64)
    eligible_labels = labels[eligible]
    eligible_positive = int(np.count_nonzero(eligible_labels == 1))
    eligible_negative = int(np.count_nonzero(eligible_labels == 0))
    if (
        eligible_positive < MINIMUM_CLASS_COUNT
        or eligible_negative < MINIMUM_CLASS_COUNT
    ):
        reason_code = (
            "CALIBRATION_RISK_POSITIVE_BELOW_60"
            if eligible_positive < MINIMUM_CLASS_COUNT
            else "CALIBRATION_RISK_NEGATIVE_BELOW_60"
        )
        raise V022CalibrationTerminalInconclusive(
            reason_code,
            "V2.2 isotonic calibration requires at least "
            f"{MINIMUM_CLASS_COUNT} labels per class in the shared eligible "
            f"pool; observed positive={eligible_positive}, "
            f"negative={eligible_negative}",
        )

    try:
        selected_arm_a = arm_a[eligible]
        selected_arm_b = arm_b[eligible]
        if (
            not np.isfinite(selected_arm_a).all()
            or not np.isfinite(selected_arm_b).all()
        ):
            raise V022CalibrationTerminalInconclusive(
                "CALIBRATION_RISK_SCORE_NONFINITE",
                "A selected calibration risk score is nonfinite",
            )
        prefix_isotonic = fit_isotonic_state(selected_arm_a, eligible_labels)
        visible_isotonic = fit_isotonic_state(selected_arm_b, eligible_labels)
        if (
            len(prefix_isotonic.x_thresholds) < 2
            or len(visible_isotonic.x_thresholds) < 2
        ):
            raise V022CalibrationTerminalInconclusive(
                "CALIBRATION_ISOTONIC_FIT_UNDEFINED",
                "A V2.2 isotonic map has fewer than two score thresholds",
            )
    except V022CalibrationTerminalInconclusive:
        raise
    except V2ModelError as exc:
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_ISOTONIC_FIT_UNDEFINED",
            "V2.2 isotonic fitting is undefined on the shared eligible pool",
        ) from exc

    baseline_iae: dict[str, float] = {}
    forecast_days = np.asarray(FORECAST_DAYS, dtype=np.float64)
    for model_id in MEAN_BASELINE_IDS:
        per_cluster_iae = np.trapezoid(
            np.abs(baseline_forecasts[model_id] - targets),
            x=forecast_days,
            axis=1,
        ) / (FORECAST_DAYS[-1] - FORECAST_DAYS[0])
        baseline_iae[model_id] = float(np.mean(per_cluster_iae))
    if not all(math.isfinite(value) for value in baseline_iae.values()):
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_BASELINE_INCOMPLETE",
            "A 900-row mean-baseline IAE is nonfinite",
        )
    selected = min(baseline_iae, key=lambda item: (baseline_iae[item], item))

    try:
        nonconformity = simultaneous_nonconformity_scores(lower, upper, targets)
        if (
            nonconformity.shape != (CALIBRATION_COUNT,)
            or not np.isfinite(nonconformity).all()
        ):
            raise V022CalibrationTerminalInconclusive(
                "CALIBRATION_CONFORMAL_SCORE_NONFINITE",
                "The 900-row conformal score vector is incomplete or nonfinite",
            )
        conformal = fit_conformal_expansion(
            nonconformity,
            coverage=CONFORMAL_COVERAGE,
        )
    except V022CalibrationTerminalInconclusive:
        raise
    except V2ModelError as exc:
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_CONFORMAL_FIT_UNDEFINED",
            "V2.2 conformal calibration is undefined on the 900-row source",
        ) from exc
    if (
        conformal.calibration_count != CALIBRATION_COUNT
        or conformal.order_statistic_index != CONFORMAL_ORDER_STATISTIC_INDEX
    ):
        raise V022CalibrationTerminalInconclusive(
            "CALIBRATION_CONFORMAL_COUNT_NOT_900",
            "V2.2 conformal state differs from the 900/811 rule",
        )

    source_positive = int(np.count_nonzero(labels == 1))
    source_negative = int(np.count_nonzero(labels == 0))
    state = CalibrationDevelopmentState(
        prefix_only_isotonic=prefix_isotonic,
        visible_stress_isotonic=visible_isotonic,
        conformal=conformal,
        selected_mean_baseline=selected,
        mean_baseline_iae_pp=tuple(
            (model_id, baseline_iae[model_id]) for model_id in MEAN_BASELINE_IDS
        ),
        calibration_cluster_count=CALIBRATION_COUNT,
        positive_label_count=source_positive,
        negative_label_count=source_negative,
    )
    audit = V022CalibrationAudit(
        source_calibration_count=CALIBRATION_COUNT,
        risk_isotonic_eligible_count=eligible_count,
        risk_isotonic_ineligible_zero_family_count=int(
            np.count_nonzero((~eligible) & (family_counts == 0))
        ),
        risk_isotonic_ineligible_one_family_count=int(
            np.count_nonzero((~eligible) & (family_counts == 1))
        ),
        risk_isotonic_ineligible_other_count=int(
            np.count_nonzero((~eligible) & (family_counts >= 2))
        ),
        risk_isotonic_positive_label_count=eligible_positive,
        risk_isotonic_negative_label_count=eligible_negative,
        mean_baseline_count=CALIBRATION_COUNT,
        conformal_calibration_count=conformal.calibration_count,
        conformal_order_statistic_index=conformal.order_statistic_index,
        eligibility_mask_sha256=pretruth_commitment.eligibility_mask_sha256,
    )
    return state, audit


__all__ = [
    "V022CalibrationAudit",
    "V022CalibrationError",
    "V022CalibrationTerminalInconclusive",
    "V022CommittedMaskRow",
    "V022PretruthMaskCommitment",
    "V022_MINIMUM_ELIGIBLE_COUNT",
    "V022_MINIMUM_ELIGIBLE_FRACTION",
    "calibration_eligibility_mask_sha256_v022",
    "derive_calibration_mask_commitment_v022",
    "fit_calibration_development_state_v022",
]
