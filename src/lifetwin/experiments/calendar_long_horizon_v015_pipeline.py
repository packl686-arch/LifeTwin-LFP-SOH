"""Label-free orchestration for the frozen V0.15 prediction pipeline.

The public prediction entry point accepts only the five committed predictor
tables and an already-frozen model state.  It deliberately has no argument for
truth data, filesystem paths, truth families, or matched-pair side labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEYS,
    V015FitError,
    frozen_parameter_metadata,
    parse_canonical_parameters_json,
    recompute_variant_commitment,
    validate_frozen_variant_keys,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    canonicalize_frame,
    load_artifact_contract,
    predictor_content_hashes,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    ConformalExpansionState,
    IsotonicState,
    LogisticRiskState,
    VariantSummary,
    blend_center_forecast,
    build_library_forecast,
    coordinatewise_weighted_quantile,
    expand_intervals,
    extract_prefix_features,
    rank_for_issuance,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
    stress_index,
)


DECLARED_STRUCTURE_FAMILIES = (
    "target_prefix_persistence",
    "target_prefix_sqrt_time",
    "target_prefix_bounded_power_law",
    "target_prefix_saturating_plus_slow",
    "target_prefix_dual_power",
    "target_prefix_late_knee_prior_grid",
    "target_prefix_early_activation_plus_power",
)
VISIBLE_STRESS_FEATURE_NAMES = PREFIX_FEATURE_NAMES + REAL_OPERATING_FIELDS
PLACEBO_FEATURE_NAMES = PREFIX_FEATURE_NAMES + PLACEBO_FIELDS
ARM_A_PLUS_S_PLAN_FEATURE_NAMES = PREFIX_FEATURE_NAMES + ("planned_stress_index",)
PRIMARY_ARMS = ("prefix_only", "visible_stress")
PRIMARY_ISSUE_COUNTS = {"test": 950, "audit": 475}

_INPUT_FILENAMES = (
    "prefix_pack.csv",
    "forecast_coordinates.csv",
    "operating_pack.csv",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
)
_IDENTITY_COLUMNS = ("protocol_id", "partition", "cluster_id")
_FORECAST_BOUNDS_PCT = (40.0, 105.0)
_MAXIMUM_PREFIX_RMSE_PP = 1.0
_MAXIMUM_PREFIX_RESIDUAL_PP = 1.5
_FORMULA_ABSOLUTE_TOLERANCE = 1e-12
_CONTRACT = load_artifact_contract()


class V015PipelineError(ValueError):
    """Raised when label-free predictor inputs or state violate V0.15."""


@dataclass(frozen=True)
class FrozenLabelFreeState:
    """Outcome-trained state frozen before a prediction partition is opened."""

    center_beta: float
    prefix_only_risk: LogisticRiskState
    visible_stress_risk: LogisticRiskState
    placebo_risk: LogisticRiskState
    arm_a_plus_s_plan_risk: LogisticRiskState
    strongest_single_feature_name: str
    strongest_single_feature_orientation: int
    prefix_only_isotonic: IsotonicState
    visible_stress_isotonic: IsotonicState
    conformal: ConformalExpansionState

    def __post_init__(self) -> None:
        if not math.isfinite(self.center_beta) or not 0.0 <= self.center_beta <= 1.0:
            raise V015PipelineError("center_beta must be finite and in [0, 1]")
        if self.prefix_only_risk.feature_names != PREFIX_FEATURE_NAMES:
            raise V015PipelineError("Arm A feature order differs from the freeze")
        if self.visible_stress_risk.feature_names != VISIBLE_STRESS_FEATURE_NAMES:
            raise V015PipelineError("Arm B feature order differs from the freeze")
        if self.placebo_risk.feature_names != PLACEBO_FEATURE_NAMES:
            raise V015PipelineError("Placebo feature order differs from the freeze")
        if self.arm_a_plus_s_plan_risk.feature_names != ARM_A_PLUS_S_PLAN_FEATURE_NAMES:
            raise V015PipelineError(
                "Arm-A-plus-S_plan feature order differs from the freeze"
            )
        if self.strongest_single_feature_name not in PREFIX_FEATURE_NAMES:
            raise V015PipelineError(
                "Strongest single feature is not a frozen Arm-A feature"
            )
        if self.strongest_single_feature_orientation not in {-1, 1}:
            raise V015PipelineError(
                "Strongest single-feature orientation must be -1 or 1"
            )
        for name, state in (
            ("prefix_only", self.prefix_only_risk),
            ("visible_stress", self.visible_stress_risk),
            ("placebo_8", self.placebo_risk),
            ("arm_a_plus_s_plan", self.arm_a_plus_s_plan_risk),
        ):
            dimension = len(state.feature_names)
            mean = np.asarray(state.standardizer.mean, dtype=float)
            scale = np.asarray(state.standardizer.scale, dtype=float)
            zero = np.asarray(state.standardizer.zero_variance)
            coefficients = np.asarray(state.coefficients, dtype=float)
            if (
                mean.shape != (dimension,)
                or scale.shape != (dimension,)
                or zero.shape != (dimension,)
                or coefficients.shape != (dimension,)
                or not all(
                    isinstance(value, (bool, np.bool_))
                    for value in state.standardizer.zero_variance
                )
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or np.any(scale <= 0.0)
                or not np.isfinite(coefficients).all()
                or not math.isfinite(state.intercept)
            ):
                raise V015PipelineError(f"{name} logistic state is malformed")
            zero_mask = zero.astype(bool)
            if np.any(scale[zero_mask] != 1.0) or np.any(
                np.abs(coefficients[zero_mask]) > 1e-12
            ):
                raise V015PipelineError(f"{name} zero-variance feature state changed")
        for name, state in (
            ("prefix_only", self.prefix_only_isotonic),
            ("visible_stress", self.visible_stress_isotonic),
        ):
            x = np.asarray(state.x_thresholds, dtype=float)
            y = np.asarray(state.y_thresholds, dtype=float)
            if (
                x.ndim != 1
                or len(x) < 2
                or x.shape != y.shape
                or not np.isfinite(x).all()
                or not np.isfinite(y).all()
                or np.any(np.diff(x) <= 0.0)
                or np.any(np.diff(y) < 0.0)
                or np.any((y < 0.0) | (y > 1.0))
            ):
                raise V015PipelineError(f"{name} isotonic state is malformed")
        if (
            self.conformal.coverage != 0.90
            or self.conformal.calibration_count != 900
            or self.conformal.order_statistic_index != 811
            or not math.isfinite(self.conformal.expansion_pp)
            or self.conformal.expansion_pp < 0.0
        ):
            raise V015PipelineError(
                "Conformal state differs from the frozen 900/811 rule"
            )


@dataclass(frozen=True)
class PrimaryArmRanking:
    """Full-pool ranks with ``None`` for a common-pool abstention."""

    prefix_only_ranks: tuple[int | None, ...]
    visible_stress_ranks: tuple[int | None, ...]
    prefix_only_issued: tuple[bool, ...]
    visible_stress_issued: tuple[bool, ...]


@dataclass(frozen=True)
class LabelFreePipelineResult:
    """Recomputed label-free artifacts plus an auditable feature table."""

    prediction_bundle: pd.DataFrame
    feature_bundle: pd.DataFrame
    primary_risk_bundle: pd.DataFrame
    decision_bundle: pd.DataFrame
    predictor_content_bundle: pd.DataFrame


def _validate_dependency_contract(contract: FrozenArtifactContract) -> None:
    if contract.protocol_id != FROZEN_PROTOCOL_ID:
        raise V015PipelineError("Protocol IDs differ across V0.15 modules")
    if contract.prefix_days != PREFIX_DAYS:
        raise V015PipelineError("Prefix grid differs across V0.15 modules")
    if contract.forecast_days != FORECAST_DAYS:
        raise V015PipelineError("Forecast grid differs across V0.15 modules")
    frozen_families = tuple(dict.fromkeys(key[0] for key in FROZEN_VARIANT_KEYS))
    if frozen_families != DECLARED_STRUCTURE_FAMILIES:
        raise V015PipelineError(
            "Structure-family order differs from the exact variant contract"
        )


def _canonical_input(
    frame: pd.DataFrame,
    filename: str,
    contract: FrozenArtifactContract,
) -> pd.DataFrame:
    try:
        return canonicalize_frame(
            frame,
            contract.csv_schema(filename),
            contract,
            formal=False,
        )
    except V015ArtifactError as exc:
        raise V015PipelineError(str(exc)) from exc


def _strict_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise V015PipelineError(f"{context} must be a strict boolean")
    return bool(value)


def _finite_number(value: object, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise V015PipelineError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise V015PipelineError(f"{context} must be finite")
    return result


def _require_exact_grid(
    frame: pd.DataFrame,
    *,
    value_column: str,
    expected: Sequence[float],
    context: str,
) -> pd.DataFrame:
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(float)
    if (
        len(values) != len(expected)
        or not np.isfinite(values).all()
        or tuple(sorted(float(value) for value in values)) != tuple(expected)
    ):
        raise V015PipelineError(f"{context} does not contain the frozen grid")
    return frame.sort_values(value_column, kind="stable").reset_index(drop=True)


def _parse_parameter_payload(value: object) -> dict[str, float]:
    try:
        return parse_canonical_parameters_json(value)
    except V015FitError as exc:
        raise V015PipelineError(str(exc)) from exc


def _parameter_metadata(
    model_id: str,
    variant_id: str,
    parameters: Mapping[str, float],
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float, float], ...],
    float,
]:
    try:
        return frozen_parameter_metadata(model_id, variant_id, parameters)
    except V015FitError as exc:
        raise V015PipelineError(str(exc)) from exc


def _require_formula_match(
    committed: float,
    recomputed: float,
    *,
    context: str,
) -> None:
    if not math.isclose(
        committed,
        recomputed,
        rel_tol=0.0,
        abs_tol=_FORMULA_ABSOLUTE_TOLERANCE,
    ):
        raise V015PipelineError(f"{context} differs from frozen formula recomputation")


def _recomputed_credible(
    diagnostic: pd.Series,
    raw_forecast: np.ndarray,
) -> bool:
    status = str(diagnostic["fit_status"])
    if status not in {"succeeded", "failed"}:
        raise V015PipelineError("fit_status must be succeeded or failed")
    declared = _strict_bool(diagnostic["credible_variant"], context="credible_variant")
    if status == "failed":
        if declared:
            raise V015PipelineError("A failed variant was declared credible")
        if not np.isnan(raw_forecast).all():
            raise V015PipelineError(
                "A failed variant must contain only empty forecasts"
            )
        for name in (
            "prefix_rmse_pp",
            "prefix_max_abs_residual_pp",
            "parameter_boundary_hit_fraction",
        ):
            try:
                value = float(diagnostic[name])
            except (TypeError, ValueError) as exc:
                raise V015PipelineError(f"A failed variant has invalid {name}") from exc
            if not math.isnan(value):
                raise V015PipelineError(f"A failed variant must contain empty {name}")
        return False

    rmse = _finite_number(diagnostic["prefix_rmse_pp"], context="prefix_rmse_pp")
    residual = _finite_number(
        diagnostic["prefix_max_abs_residual_pp"],
        context="prefix_max_abs_residual_pp",
    )
    recomputed = bool(
        rmse <= _MAXIMUM_PREFIX_RMSE_PP
        and residual <= _MAXIMUM_PREFIX_RESIDUAL_PP
        and np.isfinite(raw_forecast).all()
        and np.all(raw_forecast >= _FORECAST_BOUNDS_PCT[0])
        and np.all(raw_forecast <= _FORECAST_BOUNDS_PCT[1])
    )
    if declared != recomputed:
        raise V015PipelineError(
            "credible_variant differs from the frozen credibility rule"
        )
    return recomputed


def _tie_hash(arm: str, content_hash: str) -> str:
    material = f"{FROZEN_PROTOCOL_ID}|{arm}|{content_hash}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def rank_primary_arms(
    *,
    prefix_only_scores: Sequence[float],
    visible_stress_scores: Sequence[float],
    prefix_only_hashes: Sequence[str],
    visible_stress_hashes: Sequence[str],
    hard_eligible: Sequence[bool],
    issue_count: int,
) -> PrimaryArmRanking:
    """Rank both heads on one byte-identical eligibility mask and count."""

    prefix_scores = np.asarray(prefix_only_scores, dtype=np.float64)
    visible_scores = np.asarray(visible_stress_scores, dtype=np.float64)
    eligible = np.asarray(hard_eligible)
    hashes_a = tuple(prefix_only_hashes)
    hashes_b = tuple(visible_stress_hashes)
    size = prefix_scores.size
    if (
        prefix_scores.ndim != 1
        or visible_scores.ndim != 1
        or visible_scores.size != size
        or eligible.ndim != 1
        or eligible.size != size
        or len(hashes_a) != size
        or len(hashes_b) != size
    ):
        raise V015PipelineError("Primary ranking inputs have inconsistent lengths")
    if not all(isinstance(value, (bool, np.bool_)) for value in eligible):
        raise V015PipelineError("hard_eligible must contain strict booleans")
    indices = np.flatnonzero(eligible.astype(bool))
    if issue_count > len(indices):
        raise V015PipelineError(
            "The common hard-eligibility pool is smaller than the issue count"
        )
    if (
        not np.isfinite(prefix_scores[indices]).all()
        or not np.isfinite(visible_scores[indices]).all()
    ):
        raise V015PipelineError("An eligible primary risk score is nonfinite")

    ranking_a = rank_for_issuance(
        prefix_scores[indices],
        tuple(_tie_hash("prefix_only", hashes_a[index]) for index in indices),
        issue_count,
    )
    ranking_b = rank_for_issuance(
        visible_scores[indices],
        tuple(_tie_hash("visible_stress", hashes_b[index]) for index in indices),
        issue_count,
    )
    ranks_a: list[int | None] = [None] * size
    ranks_b: list[int | None] = [None] * size
    issued_a = [False] * size
    issued_b = [False] * size
    for local, global_index in enumerate(indices):
        ranks_a[int(global_index)] = ranking_a.ranks[local]
        ranks_b[int(global_index)] = ranking_b.ranks[local]
        issued_a[int(global_index)] = ranking_a.issued[local]
        issued_b[int(global_index)] = ranking_b.issued[local]
    if sum(issued_a) != issue_count or sum(issued_b) != issue_count:
        raise V015PipelineError("Primary arms did not issue the same fixed count")
    return PrimaryArmRanking(
        tuple(ranks_a),
        tuple(ranks_b),
        tuple(issued_a),
        tuple(issued_b),
    )


def _cluster_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["partition"].astype(str),
            frame["cluster_id"].astype(str),
            strict=True,
        )
    )


def _validate_cluster_alignment(frames: Sequence[pd.DataFrame]) -> None:
    expected = _cluster_keys(frames[0])
    if not expected:
        raise V015PipelineError("Predictor inputs contain no clusters")
    cluster_partitions: dict[str, set[str]] = {}
    for partition, cluster_id in expected:
        cluster_partitions.setdefault(cluster_id, set()).add(partition)
    if any(len(partitions) != 1 for partitions in cluster_partitions.values()):
        raise V015PipelineError("An opaque cluster ID is reused across partitions")
    for frame in frames[1:]:
        if _cluster_keys(frame) != expected:
            raise V015PipelineError("Predictor input cluster sets differ")


def _select_baseline(
    forecasts: Mapping[tuple[str, str], np.ndarray],
    diagnostics: Mapping[tuple[str, str], pd.Series],
    model_id: str,
) -> np.ndarray:
    keys = sorted(key for key in forecasts if key[0] == model_id)
    successful = [
        forecasts[key]
        for key in keys
        if str(diagnostics[key]["fit_status"]) == "succeeded"
        and np.isfinite(forecasts[key]).all()
    ]
    if not successful:
        raise V015PipelineError(f"{model_id} has no finite succeeded forecast")
    first = successful[0]
    if any(not np.array_equal(first, candidate) for candidate in successful[1:]):
        raise V015PipelineError(f"{model_id} has conflicting baseline variants")
    return first


def _abstention_reasons(
    *,
    successful_family_count: int,
    center: Sequence[float],
    prefix_features: Sequence[float],
    real_operating: Sequence[float],
    placebo_operating: Sequence[float],
) -> tuple[bool, str]:
    reasons: list[str] = []
    if successful_family_count < 2:
        reasons.append("insufficient_structure_families")
    if not np.isfinite(np.asarray(center, dtype=float)).all():
        reasons.append("nonfinite_center_forecast")
    if not np.isfinite(np.asarray(prefix_features, dtype=float)).all():
        reasons.append("nonfinite_prefix_features")
    if not np.isfinite(np.asarray(real_operating, dtype=float)).all():
        reasons.append("nonfinite_real_operating_features")
    if not np.isfinite(np.asarray(placebo_operating, dtype=float)).all():
        reasons.append("nonfinite_placebo_features")
    return not reasons, ";".join(reasons)


def recompute_label_free_pipeline(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    member_fit_diagnostics: pd.DataFrame,
    member_forecast_bundle: pd.DataFrame,
    state: FrozenLabelFreeState,
) -> LabelFreePipelineResult:
    """Recompute every primary label-free prediction and policy input."""

    contract = _CONTRACT
    _validate_dependency_contract(contract)
    inputs = [
        _canonical_input(frame, filename, contract)
        for frame, filename in zip(
            (
                prefix_pack,
                forecast_coordinates,
                operating_pack,
                member_fit_diagnostics,
                member_forecast_bundle,
            ),
            _INPUT_FILENAMES,
            strict=True,
        )
    ]
    prefix, coordinates, operating, diagnostics, member_forecasts = inputs
    _validate_cluster_alignment(inputs)

    prediction_records: list[dict[str, object]] = []
    feature_records: list[dict[str, object]] = []
    risk_records: list[dict[str, object]] = []
    content_records: list[dict[str, object]] = []

    diagnostics_by_cluster = {
        key: group.reset_index(drop=True)
        for key, group in diagnostics.groupby(["partition", "cluster_id"], sort=True)
    }
    forecasts_by_cluster = {
        key: group.reset_index(drop=True)
        for key, group in member_forecasts.groupby(
            ["partition", "cluster_id"], sort=True
        )
    }
    prefix_by_cluster = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in prefix.groupby(
            ["partition", "cluster_id"], sort=True
        )
    }
    coordinates_by_cluster = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in coordinates.groupby(
            ["partition", "cluster_id"], sort=True
        )
    }
    operating_by_cluster = operating.set_index(["partition", "cluster_id"])
    if not operating_by_cluster.index.is_unique:
        raise V015PipelineError("Operating pack contains duplicate cluster keys")

    ordered_keys = sorted(_cluster_keys(prefix))
    for partition, cluster_id in ordered_keys:
        prefix_rows = _require_exact_grid(
            prefix_by_cluster[(partition, cluster_id)],
            value_column="prefix_day",
            expected=PREFIX_DAYS,
            context=f"{partition}/{cluster_id} prefix",
        )
        coordinate_rows = _require_exact_grid(
            coordinates_by_cluster[(partition, cluster_id)],
            value_column="forecast_day",
            expected=FORECAST_DAYS,
            context=f"{partition}/{cluster_id} coordinates",
        )
        observed = pd.to_numeric(
            prefix_rows["observed_retention_pct"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(observed).all():
            raise V015PipelineError("Prefix observations must be finite")

        operating_row = operating_by_cluster.loc[(partition, cluster_id)]
        operating_values = tuple(
            _finite_number(operating_row[name], context=name)
            for name in REAL_OPERATING_FIELDS
        )
        placebo_values = tuple(
            _finite_number(operating_row[name], context=name) for name in PLACEBO_FIELDS
        )
        try:
            hashes = predictor_content_hashes(
                prefix_rows, coordinate_rows, operating_row
            )
        except V015ArtifactError as exc:
            raise V015PipelineError(str(exc)) from exc

        diagnostic_group = diagnostics_by_cluster[(partition, cluster_id)]
        forecast_group = forecasts_by_cluster[(partition, cluster_id)]
        diagnostic_key_rows = tuple(
            zip(
                diagnostic_group["model_id"].astype(str),
                diagnostic_group["variant_id"].astype(str),
                strict=True,
            )
        )
        forecast_key_rows = tuple(
            forecast_group.loc[:, ["model_id", "variant_id"]]
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        try:
            validate_frozen_variant_keys(
                diagnostic_key_rows,
                context=f"{partition}/{cluster_id} diagnostics",
            )
            validate_frozen_variant_keys(
                forecast_key_rows,
                context=f"{partition}/{cluster_id} forecasts",
            )
        except V015FitError as exc:
            raise V015PipelineError(str(exc)) from exc
        diagnostic_keys = set(diagnostic_key_rows)
        forecast_keys = set(forecast_key_rows)
        if diagnostic_keys != forecast_keys or len(forecast_group) != len(
            FROZEN_VARIANT_KEYS
        ) * len(FORECAST_DAYS):
            raise V015PipelineError(
                "Diagnostic and member-forecast exact 86 variant sets differ"
            )

        diagnostic_map: dict[tuple[str, str], pd.Series] = {}
        raw_forecast_map: dict[tuple[str, str], np.ndarray] = {}
        credible_by_family: dict[str, list[VariantSummary]] = {}
        for key in sorted(diagnostic_keys):
            model_id, variant_id = key
            diagnostic_matches = diagnostic_group.loc[
                diagnostic_group["model_id"].astype(str).eq(model_id)
                & diagnostic_group["variant_id"].astype(str).eq(variant_id)
            ]
            if len(diagnostic_matches) != 1:
                raise V015PipelineError("Variant diagnostics are not unique")
            diagnostic = diagnostic_matches.iloc[0]
            parameters = _parse_parameter_payload(diagnostic["parameters_json"])
            forecast_rows = _require_exact_grid(
                forecast_group.loc[
                    forecast_group["model_id"].astype(str).eq(model_id)
                    & forecast_group["variant_id"].astype(str).eq(variant_id)
                ],
                value_column="forecast_day",
                expected=FORECAST_DAYS,
                context=f"{partition}/{cluster_id}/{model_id}/{variant_id}",
            )
            raw_forecast = pd.to_numeric(
                forecast_rows["raw_forecast_retention_pct"], errors="coerce"
            ).to_numpy(float)
            expected_hash = hashes.arm_a
            diagnostic_hashes = set(
                diagnostic_matches["canonical_prefix_content_sha256"].astype(str)
            )
            forecast_hashes = set(
                forecast_rows["canonical_prefix_content_sha256"].astype(str)
            )
            if diagnostic_hashes != {expected_hash} or forecast_hashes != {
                expected_hash
            }:
                raise V015PipelineError(
                    "Committed prefix-content hash differs from recomputation"
                )
            status = str(diagnostic["fit_status"])
            if status not in {"succeeded", "failed"}:
                raise V015PipelineError("fit_status must be succeeded or failed")
            if status == "failed":
                if parameters:
                    raise V015PipelineError(
                        "A failed variant retained fitted parameters"
                    )
                parameter_values: tuple[tuple[str, float], ...] = ()
                parameter_bounds: tuple[tuple[str, float, float], ...] = ()
            else:
                (
                    parameter_values,
                    parameter_bounds,
                    recomputed_boundary_fraction,
                ) = _parameter_metadata(model_id, variant_id, parameters)
                committed_boundary_fraction = _finite_number(
                    diagnostic["parameter_boundary_hit_fraction"],
                    context="parameter_boundary_hit_fraction",
                )
                _require_formula_match(
                    committed_boundary_fraction,
                    recomputed_boundary_fraction,
                    context="parameter_boundary_hit_fraction",
                )
                try:
                    recomputed = recompute_variant_commitment(
                        model_id=model_id,
                        variant_id=variant_id,
                        parameters=parameters,
                        prefix_days=prefix_rows["prefix_day"].to_numpy(float),
                        observed_retention_pct=observed,
                        forecast_days=forecast_rows["forecast_day"].to_numpy(float),
                    )
                except V015FitError as exc:
                    raise V015PipelineError(str(exc)) from exc
                committed_rmse = _finite_number(
                    diagnostic["prefix_rmse_pp"],
                    context="prefix_rmse_pp",
                )
                committed_max_residual = _finite_number(
                    diagnostic["prefix_max_abs_residual_pp"],
                    context="prefix_max_abs_residual_pp",
                )
                _require_formula_match(
                    committed_rmse,
                    recomputed.prefix_rmse_pp,
                    context="prefix_rmse_pp",
                )
                _require_formula_match(
                    committed_max_residual,
                    recomputed.prefix_max_abs_residual_pp,
                    context="prefix_max_abs_residual_pp",
                )
                recomputed_forecast = np.asarray(
                    recomputed.forecast_retention_pct, dtype=np.float64
                )
                if not np.allclose(
                    raw_forecast,
                    recomputed_forecast,
                    rtol=0.0,
                    atol=_FORMULA_ABSOLUTE_TOLERANCE,
                    equal_nan=False,
                ):
                    raise V015PipelineError(
                        "raw_forecast_retention_pct differs from frozen "
                        "formula recomputation"
                    )
            credible = _recomputed_credible(diagnostic, raw_forecast)
            diagnostic_map[key] = diagnostic
            raw_forecast_map[key] = raw_forecast
            if credible:
                credible_by_family.setdefault(model_id, []).append(
                    VariantSummary(
                        forecast=tuple(float(value) for value in raw_forecast),
                        prefix_rmse_pp=_finite_number(
                            diagnostic["prefix_rmse_pp"],
                            context="prefix_rmse_pp",
                        ),
                        parameter_values=parameter_values,
                        parameter_bounds=parameter_bounds,
                    )
                )

        sqrt_forecast = _select_baseline(
            raw_forecast_map,
            diagnostic_map,
            "target_prefix_sqrt_time",
        )
        bounded_forecast = _select_baseline(
            raw_forecast_map,
            diagnostic_map,
            "target_prefix_bounded_power_law",
        )
        family_vectors = {
            model_id: tuple(variant.forecast for variant in variants)
            for model_id, variants in credible_by_family.items()
        }
        library = build_library_forecast(family_vectors, sqrt_forecast)
        center = blend_center_forecast(
            sqrt_forecast, library.forecast, state.center_beta
        )

        if library.support_vectors:
            base_lower = coordinatewise_weighted_quantile(
                library.support_vectors, library.support_weights, 0.05
            )
            base_upper = coordinatewise_weighted_quantile(
                library.support_vectors, library.support_weights, 0.95
            )
            calibrated_lower_array, calibrated_upper_array = expand_intervals(
                (base_lower,),
                (base_upper,),
                state.conformal.expansion_pp,
            )
            calibrated_lower = calibrated_lower_array[0]
            calibrated_upper = calibrated_upper_array[0]
        else:
            base_lower = (math.nan,) * len(FORECAST_DAYS)
            base_upper = (math.nan,) * len(FORECAST_DAYS)
            calibrated_lower = np.full(len(FORECAST_DAYS), math.nan)
            calibrated_upper = np.full(len(FORECAST_DAYS), math.nan)

        if credible_by_family:
            prefix_feature_vector = extract_prefix_features(
                prefix_days=PREFIX_DAYS,
                observed_retention_pct=observed,
                family_variants=credible_by_family,
                sqrt_forecast=sqrt_forecast,
                center_forecast=center,
                declared_family_count=len(DECLARED_STRUCTURE_FAMILIES),
            )
            prefix_features = prefix_feature_vector.values
        else:
            prefix_features = (math.nan,) * len(PREFIX_FEATURE_NAMES)
        arm_a_features = tuple(float(value) for value in prefix_features)
        arm_b_features = arm_a_features + operating_values
        placebo_features = arm_a_features + placebo_values
        planned_stress = stress_index(*operating_values[4:])
        arm_a_plus_s_plan_features = arm_a_features + (planned_stress,)
        credible_vectors = [
            np.asarray(variant.forecast, dtype=float)
            for variants in credible_by_family.values()
            for variant in variants
        ]
        if credible_vectors:
            credible_matrix = np.vstack(credible_vectors)
            v1_max_envelope = float(
                np.max(
                    np.max(credible_matrix, axis=0) - np.min(credible_matrix, axis=0)
                )
            )
        else:
            v1_max_envelope = math.nan
        best_prefix_rmse = arm_a_features[
            PREFIX_FEATURE_NAMES.index("best_prefix_rmse_pp")
        ]
        center_sqrt_difference = abs(center[-1] - sqrt_forecast[-1])
        strongest_feature = arm_a_features[
            PREFIX_FEATURE_NAMES.index(state.strongest_single_feature_name)
        ]
        all_features_finite = bool(
            np.isfinite(
                (
                    *arm_b_features,
                    *placebo_values,
                    planned_stress,
                    v1_max_envelope,
                    center_sqrt_difference,
                    strongest_feature,
                )
            ).all()
        )
        hard_eligible, abstention_reasons = _abstention_reasons(
            successful_family_count=library.successful_family_count,
            center=center,
            prefix_features=arm_a_features,
            real_operating=operating_values,
            placebo_operating=placebo_values,
        )

        if all_features_finite:
            raw_a = float(
                state.prefix_only_risk.decision_function((arm_a_features,))[0]
            )
            raw_b = float(
                state.visible_stress_risk.decision_function((arm_b_features,))[0]
            )
            raw_placebo = float(
                state.placebo_risk.decision_function((placebo_features,))[0]
            )
            raw_arm_a_plus_s_plan = float(
                state.arm_a_plus_s_plan_risk.decision_function(
                    (arm_a_plus_s_plan_features,)
                )[0]
            )
            raw_strongest = float(
                state.strongest_single_feature_orientation * strongest_feature
            )
            raw_planned_stress = float(planned_stress)
            raw_prefix_rmse = float(best_prefix_rmse)
            raw_v1_envelope = float(v1_max_envelope)
            raw_center_sqrt = float(center_sqrt_difference)
            calibrated_a = float(state.prefix_only_isotonic.predict((raw_a,))[0])
            calibrated_b = float(state.visible_stress_isotonic.predict((raw_b,))[0])
        else:
            (
                raw_a,
                raw_b,
                raw_placebo,
                raw_arm_a_plus_s_plan,
                raw_strongest,
                raw_planned_stress,
                raw_prefix_rmse,
                raw_v1_envelope,
                raw_center_sqrt,
                calibrated_a,
                calibrated_b,
            ) = (math.nan,) * 11

        identity = {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "partition": partition,
            "cluster_id": cluster_id,
        }
        for index, day in enumerate(FORECAST_DAYS):
            prediction_records.append(
                {
                    **identity,
                    "forecast_day": day,
                    "center_forecast_pct": center[index],
                    "sqrt_time_forecast_pct": float(sqrt_forecast[index]),
                    "bounded_power_forecast_pct": float(bounded_forecast[index]),
                    "base_interval_lower_pct": base_lower[index],
                    "base_interval_upper_pct": base_upper[index],
                    "calibrated_interval_lower_pct": calibrated_lower[index],
                    "calibrated_interval_upper_pct": calibrated_upper[index],
                    "canonical_prefix_content_sha256": hashes.arm_a,
                }
            )
        feature_records.append(
            {
                **identity,
                "hard_eligible": hard_eligible,
                "all_features_finite": all_features_finite,
                "abstention_reasons": abstention_reasons,
                **dict(zip(PREFIX_FEATURE_NAMES, arm_a_features, strict=True)),
                **dict(zip(REAL_OPERATING_FIELDS, operating_values, strict=True)),
                **dict(zip(PLACEBO_FIELDS, placebo_values, strict=True)),
            }
        )
        for score_id, raw, calibrated, content_hash in (
            ("prefix_only", raw_a, calibrated_a, hashes.arm_a),
            ("visible_stress", raw_b, calibrated_b, hashes.arm_b),
            ("placebo_8", raw_placebo, math.nan, hashes.placebo),
            (
                "arm_a_plus_s_plan",
                raw_arm_a_plus_s_plan,
                math.nan,
                hashes.arm_b,
            ),
            (
                "strongest_single_feature",
                raw_strongest,
                math.nan,
                hashes.arm_a,
            ),
            (
                "planned_stress_only",
                raw_planned_stress,
                math.nan,
                hashes.arm_b,
            ),
            ("prefix_rmse_only", raw_prefix_rmse, math.nan, hashes.arm_a),
            (
                "v1_max_envelope_only",
                raw_v1_envelope,
                math.nan,
                hashes.arm_a,
            ),
            (
                "center_sqrt_abs_difference_only",
                raw_center_sqrt,
                math.nan,
                hashes.arm_a,
            ),
        ):
            risk_records.append(
                {
                    **identity,
                    "score_id": score_id,
                    "raw_risk_score": raw,
                    "calibrated_catastrophic_probability": calibrated,
                    "all_features_finite": all_features_finite,
                    "successful_structure_family_count": (
                        library.successful_family_count
                    ),
                    "fit_failure_count": (
                        len(DECLARED_STRUCTURE_FAMILIES)
                        - library.successful_family_count
                    ),
                    "effective_unique_shape_count": (
                        arm_a_features[
                            PREFIX_FEATURE_NAMES.index("effective_unique_shape_count")
                        ]
                    ),
                    "canonical_predictor_content_sha256": content_hash,
                }
            )
        content_records.append(
            {
                **identity,
                "random_policy_content_sha256": hashes.random_policy,
                "arm_a_content_sha256": hashes.arm_a,
                "arm_b_content_sha256": hashes.arm_b,
                "placebo_content_sha256": hashes.placebo,
            }
        )

    feature_bundle = (
        pd.DataFrame(feature_records)
        .sort_values(["partition", "cluster_id"], kind="stable")
        .reset_index(drop=True)
    )
    primary_risk = (
        pd.DataFrame(risk_records)
        .sort_values(["partition", "cluster_id", "score_id"], kind="stable")
        .reset_index(drop=True)
    )
    content_bundle = (
        pd.DataFrame(content_records)
        .sort_values(["partition", "cluster_id"], kind="stable")
        .reset_index(drop=True)
    )
    for partition in PRIMARY_ISSUE_COUNTS:
        ordinary = content_bundle.loc[content_bundle["partition"].eq(partition)]
        for column in (
            "random_policy_content_sha256",
            "arm_b_content_sha256",
            "placebo_content_sha256",
        ):
            if ordinary[column].duplicated().any():
                raise V015PipelineError(
                    f"{partition} contains duplicate predictor content: {column}"
                )

    decision_records: list[dict[str, object]] = []
    for partition, feature_group in feature_bundle.groupby("partition", sort=True):
        feature_group = feature_group.sort_values(
            "cluster_id", kind="stable"
        ).reset_index(drop=True)
        risks = primary_risk.loc[primary_risk["partition"].eq(partition)].pivot(
            index="cluster_id", columns="score_id", values="raw_risk_score"
        )
        contents = content_bundle.loc[
            content_bundle["partition"].eq(partition)
        ].set_index("cluster_id")
        cluster_ids = feature_group["cluster_id"].astype(str).tolist()
        if set(risks.index.astype(str)) != set(cluster_ids):
            raise V015PipelineError("Primary risk rows are incomplete")
        risks = risks.loc[cluster_ids]
        contents = contents.loc[cluster_ids]
        if partition in PRIMARY_ISSUE_COUNTS:
            ranking = rank_primary_arms(
                prefix_only_scores=risks["prefix_only"].to_numpy(float),
                visible_stress_scores=risks["visible_stress"].to_numpy(float),
                prefix_only_hashes=contents["arm_a_content_sha256"].astype(str),
                visible_stress_hashes=contents["arm_b_content_sha256"].astype(str),
                hard_eligible=feature_group["hard_eligible"].tolist(),
                issue_count=PRIMARY_ISSUE_COUNTS[partition],
            )
        else:
            empty_ranks: tuple[int | None, ...] = (None,) * len(feature_group)
            empty_issued = (False,) * len(feature_group)
            ranking = PrimaryArmRanking(
                empty_ranks,
                empty_ranks,
                empty_issued,
                empty_issued,
            )
        for index, row in feature_group.iterrows():
            cluster_id = str(row["cluster_id"])
            for arm, ranks, issued, hash_column in (
                (
                    "prefix_only",
                    ranking.prefix_only_ranks,
                    ranking.prefix_only_issued,
                    "arm_a_content_sha256",
                ),
                (
                    "visible_stress",
                    ranking.visible_stress_ranks,
                    ranking.visible_stress_issued,
                    "arm_b_content_sha256",
                ),
            ):
                decision_records.append(
                    {
                        "protocol_id": FROZEN_PROTOCOL_ID,
                        "partition": partition,
                        "cluster_id": cluster_id,
                        "arm": arm,
                        "raw_risk_score": float(risks.loc[cluster_id, arm]),
                        "hard_eligible": bool(row["hard_eligible"]),
                        "issuance_rank": ranks[index],
                        "issued": issued[index],
                        "abstention_reasons": row["abstention_reasons"],
                        "canonical_predictor_content_sha256": str(
                            contents.loc[cluster_id, hash_column]
                        ),
                    }
                )

    prediction_bundle = pd.DataFrame(prediction_records)
    decision_bundle = pd.DataFrame(decision_records)
    prediction_bundle = _canonical_input(
        prediction_bundle, "prediction_bundle.csv", contract
    )
    primary_risk = _canonical_input(primary_risk, "risk_bundle.csv", contract)
    decision_bundle = _canonical_input(decision_bundle, "decision_bundle.csv", contract)
    return LabelFreePipelineResult(
        prediction_bundle=prediction_bundle,
        feature_bundle=feature_bundle,
        primary_risk_bundle=primary_risk,
        decision_bundle=decision_bundle,
        predictor_content_bundle=content_bundle,
    )
