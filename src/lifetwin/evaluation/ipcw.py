from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np


def _time_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _boolean_vector(
    values: Sequence[bool] | np.ndarray,
    *,
    row_count: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != row_count:
        raise ValueError(f"{name} must be one-dimensional and match rows")
    if result.dtype != np.bool_:
        raise ValueError(f"{name} must contain boolean values")
    return result.astype(bool, copy=False)


class IPCWUnavailableError(ValueError):
    """Raised when censoring support cannot identify an IPCW score."""


@dataclass(frozen=True)
class ReverseKaplanMeierCensoring:
    event_times: tuple[float, ...]
    survival_after_time: tuple[float, ...]
    risk_count: tuple[int, ...]
    censor_event_count: tuple[int, ...]
    row_count: int
    censored_count: int
    max_followup: float
    reference_sha256: str
    source_splits: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        observed_time: Sequence[float] | np.ndarray,
        is_censored: Sequence[bool] | np.ndarray,
        *,
        reference_ids: Sequence[object] | np.ndarray,
        source_splits: Sequence[str] | np.ndarray,
    ) -> ReverseKaplanMeierCensoring:
        time = _time_vector(observed_time, name="observed_time")
        censored = _boolean_vector(
            is_censored,
            row_count=len(time),
            name="is_censored",
        )
        identities = np.asarray(reference_ids, dtype=object)
        splits = np.asarray(source_splits, dtype=object)
        if identities.ndim != 1 or len(identities) != len(time):
            raise ValueError("reference_ids must be one-dimensional and match rows")
        if splits.ndim != 1 or len(splits) != len(time):
            raise ValueError("source_splits must be one-dimensional and match rows")
        normalized_identities = tuple(str(value).strip() for value in identities)
        if any(
            value is None
            or normalized in {"", "nan", "<NA>", "None"}
            for value, normalized in zip(
                identities, normalized_identities, strict=True
            )
        ):
            raise ValueError("reference_ids cannot contain null or empty values")
        if len(set(normalized_identities)) != len(identities):
            raise ValueError("reference_ids must be unique")
        normalized_splits = tuple(str(value).strip() for value in splits)
        if any(
            value is None
            or normalized in {"", "nan", "<NA>", "None"}
            for value, normalized in zip(splits, normalized_splits, strict=True)
        ):
            raise ValueError("source_splits cannot contain null or empty values")
        split_values = tuple(sorted(set(normalized_splits)))
        unexpected_splits = sorted(set(split_values) - {"train", "validation"})
        if unexpected_splits:
            raise ValueError(
                "Censoring reference rows may use only train/validation; found "
                f"{unexpected_splits}"
            )

        unique_times = np.unique(time)
        survival = 1.0
        survival_values: list[float] = []
        risk_values: list[int] = []
        censor_values: list[int] = []
        for event_time in unique_times:
            at_risk = int(np.sum(time >= event_time))
            censor_events = int(np.sum((time == event_time) & censored))
            if censor_events:
                survival *= 1 - censor_events / at_risk
            survival_values.append(float(survival))
            risk_values.append(at_risk)
            censor_values.append(censor_events)

        reference_rows = sorted(
            (
                normalized_identity,
                format(float(duration), ".17g"),
                bool(censor_flag),
                str(split),
            )
            for normalized_identity, duration, censor_flag, split in zip(
                normalized_identities,
                time,
                censored,
                normalized_splits,
                strict=True,
            )
        )
        payload = json.dumps(
            reference_rows,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return cls(
            event_times=tuple(float(value) for value in unique_times),
            survival_after_time=tuple(survival_values),
            risk_count=tuple(risk_values),
            censor_event_count=tuple(censor_values),
            row_count=len(time),
            censored_count=int(censored.sum()),
            max_followup=float(time.max()),
            reference_sha256=hashlib.sha256(payload).hexdigest(),
            source_splits=split_values,
        )

    def survival_at(
        self,
        evaluation_time: float | Sequence[float] | np.ndarray,
        *,
        left_limit: bool = False,
    ) -> np.ndarray:
        values = np.asarray(evaluation_time, dtype=float)
        scalar = values.ndim == 0
        if scalar:
            values = values.reshape(1)
        if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("evaluation_time must contain finite non-negative values")
        times = np.asarray(self.event_times, dtype=float)
        survival = np.asarray(self.survival_after_time, dtype=float)
        side = "left" if left_limit else "right"
        indices = np.searchsorted(times, values, side=side) - 1
        result = np.ones(len(values), dtype=float)
        available = indices >= 0
        result[available] = survival[indices[available]]
        return result[0:1] if scalar else result

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenIPCWPolicy:
    censoring_model: ReverseKaplanMeierCensoring
    evaluation_times: tuple[float, ...]
    min_censor_survival: float
    max_weight: float
    min_effective_sample_size: float
    min_effective_sample_fraction: float
    min_event_count: int
    min_alive_count: int
    max_clipped_fraction: float
    policy_sha256: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["censoring_model"] = self.censoring_model.to_dict()
        return result


def freeze_ipcw_policy(
    censoring_model: ReverseKaplanMeierCensoring,
    evaluation_times: Sequence[float] | np.ndarray,
    *,
    min_censor_survival: float = 0.10,
    max_weight: float = 10.0,
    min_effective_sample_size: float = 30.0,
    min_effective_sample_fraction: float = 0.50,
    min_event_count: int = 5,
    min_alive_count: int = 5,
    max_clipped_fraction: float = 0.01,
) -> FrozenIPCWPolicy:
    grid = _time_vector(evaluation_times, name="evaluation_times")
    if len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("evaluation_times must be strictly increasing with at least two points")
    if grid[-1] > censoring_model.max_followup:
        raise IPCWUnavailableError(
            "evaluation_times exceed the censoring reference follow-up support"
        )
    censor_survival = censoring_model.survival_at(grid)
    if (censor_survival <= 0).any():
        raise IPCWUnavailableError("Censoring survival is zero on the evaluation grid")
    thresholds = (
        min_censor_survival,
        max_weight,
        min_effective_sample_size,
        min_effective_sample_fraction,
        max_clipped_fraction,
    )
    if any(not math.isfinite(value) for value in thresholds):
        raise ValueError("IPCW policy thresholds must be finite")
    if not 0 < min_censor_survival <= 1:
        raise ValueError("min_censor_survival must lie in (0, 1]")
    if max_weight <= 0 or min_effective_sample_size < 0:
        raise ValueError("max_weight must be positive and min ESS non-negative")
    if not 0 <= min_effective_sample_fraction <= 1:
        raise ValueError("min_effective_sample_fraction must lie in [0, 1]")
    if min_event_count < 0 or min_alive_count < 0:
        raise ValueError("minimum event/alive counts cannot be negative")
    if not 0 <= max_clipped_fraction <= 1:
        raise ValueError("max_clipped_fraction must lie in [0, 1]")
    serializable = {
        "censoring_model": censoring_model.to_dict(),
        "evaluation_times": [float(value) for value in grid],
        "min_censor_survival": min_censor_survival,
        "max_weight": max_weight,
        "min_effective_sample_size": min_effective_sample_size,
        "min_effective_sample_fraction": min_effective_sample_fraction,
        "min_event_count": min_event_count,
        "min_alive_count": min_alive_count,
        "max_clipped_fraction": max_clipped_fraction,
    }
    payload = json.dumps(
        serializable,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return FrozenIPCWPolicy(
        censoring_model=censoring_model,
        evaluation_times=tuple(float(value) for value in grid),
        min_censor_survival=float(min_censor_survival),
        max_weight=float(max_weight),
        min_effective_sample_size=float(min_effective_sample_size),
        min_effective_sample_fraction=float(min_effective_sample_fraction),
        min_event_count=int(min_event_count),
        min_alive_count=int(min_alive_count),
        max_clipped_fraction=float(max_clipped_fraction),
        policy_sha256=hashlib.sha256(payload).hexdigest(),
    )


@dataclass(frozen=True)
class IPCWTimePointEvaluation:
    evaluation_time: float
    raw_brier_score: float
    clipped_brier_score: float
    censor_survival: float
    event_count: int
    alive_count: int
    zero_contribution_censored_count: int
    max_raw_weight: float
    weight_sum_per_test_row: float
    effective_sample_size: float
    effective_sample_fraction: float
    clipped_count: int
    clipped_fraction: float
    gate_status: str
    gate_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IPCWBrierEvaluation:
    status: str
    test_row_count: int
    integrated_brier_score: float
    clipped_integrated_brier_score: float
    integration_start: float
    integration_end: float
    policy_sha256: str
    time_points: tuple[IPCWTimePointEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["time_points"] = [item.to_dict() for item in self.time_points]
        return result


def evaluate_ipcw_brier(
    observed_time: Sequence[float] | np.ndarray,
    is_censored: Sequence[bool] | np.ndarray,
    survival_probabilities: Sequence[Sequence[float]] | np.ndarray,
    policy: FrozenIPCWPolicy,
) -> IPCWBrierEvaluation:
    time = _time_vector(observed_time, name="observed_time")
    censored = _boolean_vector(
        is_censored,
        row_count=len(time),
        name="is_censored",
    )
    prediction = np.asarray(survival_probabilities, dtype=float)
    expected_shape = (len(time), len(policy.evaluation_times))
    if prediction.shape != expected_shape:
        raise ValueError(
            f"survival_probabilities must have shape {expected_shape}, found {prediction.shape}"
        )
    if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("survival_probabilities must be finite and lie in [0, 1]")
    if (np.diff(prediction, axis=1) > 1e-12).any():
        raise ValueError("survival probabilities cannot increase over time")

    test_count = len(time)
    event_observed = ~censored
    evaluations: list[IPCWTimePointEvaluation] = []
    raw_scores: list[float] = []
    clipped_scores: list[float] = []
    for column, evaluation_time in enumerate(policy.evaluation_times):
        event_mask = event_observed & (time <= evaluation_time)
        alive_mask = time > evaluation_time
        censored_zero_mask = censored & (time <= evaluation_time)
        raw_weights = np.zeros(test_count, dtype=float)
        if event_mask.any():
            event_survival = policy.censoring_model.survival_at(
                time[event_mask],
                left_limit=True,
            )
            if (event_survival <= 0).any():
                raise IPCWUnavailableError(
                    "Censoring survival is zero at an observed event left limit"
                )
            raw_weights[event_mask] = 1 / event_survival
        censor_survival = float(
            policy.censoring_model.survival_at(float(evaluation_time))[0]
        )
        if alive_mask.any():
            if censor_survival <= 0:
                raise IPCWUnavailableError(
                    "Censoring survival is zero for an alive IPCW contribution"
                )
            raw_weights[alive_mask] = 1 / censor_survival
        truth_survival = alive_mask.astype(float)
        squared_error = (truth_survival - prediction[:, column]) ** 2
        raw_score = float(np.sum(raw_weights * squared_error) / test_count)
        clipped_weights = np.minimum(raw_weights, policy.max_weight)
        clipped_score = float(np.sum(clipped_weights * squared_error) / test_count)
        active_weights = raw_weights[raw_weights > 0]
        if len(active_weights) == 0:
            raise IPCWUnavailableError("No active IPCW contributions at an evaluation time")
        effective_sample_size = float(
            active_weights.sum() ** 2 / np.sum(active_weights**2)
        )
        clipped_count = int(np.sum(raw_weights > policy.max_weight))
        clipped_fraction = clipped_count / test_count
        reasons: list[str] = []
        if censor_survival < policy.min_censor_survival:
            reasons.append("censor survival below policy minimum")
        if effective_sample_size < policy.min_effective_sample_size:
            reasons.append("effective sample size below policy minimum")
        if effective_sample_size / test_count < policy.min_effective_sample_fraction:
            reasons.append("effective sample fraction below policy minimum")
        if int(event_mask.sum()) < policy.min_event_count:
            reasons.append("observed event count below policy minimum")
        if int(alive_mask.sum()) < policy.min_alive_count:
            reasons.append("alive count below policy minimum")
        if clipped_fraction > policy.max_clipped_fraction:
            reasons.append("clipped weight fraction exceeds policy maximum")
        evaluations.append(
            IPCWTimePointEvaluation(
                evaluation_time=float(evaluation_time),
                raw_brier_score=raw_score,
                clipped_brier_score=clipped_score,
                censor_survival=censor_survival,
                event_count=int(event_mask.sum()),
                alive_count=int(alive_mask.sum()),
                zero_contribution_censored_count=int(censored_zero_mask.sum()),
                max_raw_weight=float(raw_weights.max()),
                weight_sum_per_test_row=float(raw_weights.sum() / test_count),
                effective_sample_size=effective_sample_size,
                effective_sample_fraction=effective_sample_size / test_count,
                clipped_count=clipped_count,
                clipped_fraction=clipped_fraction,
                gate_status="passed" if not reasons else "failed",
                gate_reasons=tuple(reasons),
            )
        )
        raw_scores.append(raw_score)
        clipped_scores.append(clipped_score)

    grid = np.asarray(policy.evaluation_times, dtype=float)
    span = float(grid[-1] - grid[0])
    integrated = float(np.trapezoid(raw_scores, grid) / span)
    clipped_integrated = float(np.trapezoid(clipped_scores, grid) / span)
    return IPCWBrierEvaluation(
        status=(
            "passed"
            if all(item.gate_status == "passed" for item in evaluations)
            else "failed"
        ),
        test_row_count=test_count,
        integrated_brier_score=integrated,
        clipped_integrated_brier_score=clipped_integrated,
        integration_start=float(grid[0]),
        integration_end=float(grid[-1]),
        policy_sha256=policy.policy_sha256,
        time_points=tuple(evaluations),
    )
