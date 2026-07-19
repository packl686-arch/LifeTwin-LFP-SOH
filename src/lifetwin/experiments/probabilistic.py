from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from lifetwin.evaluation.probabilistic import (
    ConformalCalibrationError,
    conformal_log_radius,
    evaluate_group_prediction_interval,
    evaluate_prediction_interval,
    evaluate_quantile,
    group_max_conformal_log_radius,
    log_symmetric_interval,
)
from lifetwin.data.split import assert_group_isolation
from lifetwin.models.baselines import regression_metrics
from lifetwin.models.probabilistic import LogNormalAFT
from lifetwin.experiments.model_selection import select_lognormal_aft_l2


def run_probabilistic_lifetime_experiment(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    label_column: str = "cycle_life",
    censor_column: str = "is_censored",
    conformal_coverage: float = 0.8,
    l2_penalty: float = 1e-4,
    identity_column: str = "cell_id",
    group_isolation_columns: Sequence[str] = ("cell_id",),
    conformal_group_column: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit on train and evaluate a frozen log-normal AFT on test.

    Validation rows are used only to calibrate a symmetric split-conformal
    interval. Test outcomes never affect fitting or calibration.
    """
    features = list(feature_columns)
    required = {
        label_column,
        censor_column,
        identity_column,
        "split",
        *features,
        *group_isolation_columns,
    }
    if conformal_group_column is not None:
        required.add(conformal_group_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing probabilistic experiment columns: {missing}")
    if not features:
        raise ValueError("At least one feature column is required")
    if frame.empty:
        raise ValueError("Experiment frame is empty")

    working = frame.copy()
    if working[censor_column].isna().any() or working[censor_column].dtype != bool:
        raise ValueError(f"{censor_column} must contain non-null boolean values")
    numeric = working[[*features, label_column]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Features and lifetime labels must contain finite numeric values")
    if (numeric[label_column] <= 0).any():
        raise ValueError("Lifetime labels must be positive")
    working[[*features, label_column]] = numeric

    allowed_splits = {"train", "validation", "test"}
    if working["split"].isna().any():
        raise ValueError("split cannot contain null values")
    unexpected_splits = sorted(set(working["split"]) - allowed_splits)
    if unexpected_splits:
        raise ValueError(f"Unexpected split values: {unexpected_splits}")
    if working[identity_column].isna().any():
        raise ValueError(f"{identity_column} cannot contain null values")
    duplicated_identity = working[identity_column].duplicated(keep=False)
    if duplicated_identity.any():
        raise ValueError(
            f"Expected one lifetime row per {identity_column}; found "
            f"{working.loc[duplicated_identity, identity_column].nunique()} duplicates"
        )
    for group_column in group_isolation_columns:
        assert_group_isolation(working, [group_column])

    train = working.loc[working["split"] == "train"]
    validation = working.loc[working["split"] == "validation"]
    test = working.loc[working["split"] == "test"]
    if train.empty or test.empty:
        raise ValueError("Both train and test splits are required")

    model = LogNormalAFT(l2_penalty=l2_penalty).fit(
        train[features].to_numpy(dtype=float),
        train[label_column].to_numpy(dtype=float),
        is_censored=train[censor_column].to_numpy(dtype=bool),
    )
    test_features = test[features].to_numpy(dtype=float)
    test_lifetime = test[label_column].to_numpy(dtype=float)
    test_censored = test[censor_column].to_numpy(dtype=bool)
    test_observed = ~test_censored
    if not test_observed.any():
        raise ValueError("Test split needs at least one observed EOL for point metrics")
    quantile_levels = np.array([0.1, 0.5, 0.9])
    quantiles = model.predict_quantile(test_features, quantile_levels)
    median = quantiles[:, 1]

    evaluation: dict[str, object] = {
        "fit": {
            **model.fit_summary_.to_dict(),
            "sigma_log_life": model.sigma_,
            "intercept": model.intercept_,
            "coefficients": dict(zip(features, model.coef_, strict=True)),
            "l2_penalty": l2_penalty,
        },
        "test": {
            "row_count": len(test),
            "observed_event_count": int(test_observed.sum()),
            "right_censored_count": int(test_censored.sum()),
            "median_point_metrics_observed_only": regression_metrics(
                test_lifetime[test_observed], median[test_observed]
            ).to_dict(),
            "mean_negative_log_likelihood": model.negative_log_likelihood(
                test_features,
                test_lifetime,
                is_censored=test_censored,
            ),
            "quantiles_observed_only": {
                f"q{int(level * 100):02d}": evaluate_quantile(
                    test_lifetime,
                    quantiles[:, index],
                    float(level),
                    is_censored=test_censored,
                ).to_dict()
                for index, level in enumerate(quantile_levels)
            },
            "raw_central_80_interval_observed_only": evaluate_prediction_interval(
                test_lifetime,
                quantiles[:, 0],
                quantiles[:, 2],
                nominal_coverage=0.8,
                is_censored=test_censored,
            ).to_dict(),
        },
        "censoring_note": (
            "Likelihood includes right-censored rows. Point, pinball, and interval "
            "metrics explicitly exclude them and are not IPCW-corrected."
        ),
        "calibration_gate_eligible": bool(
            not test_censored.any()
            and (validation.empty or not validation[censor_column].any())
        ),
    }

    conformal_lower = np.full(len(test), np.nan)
    conformal_upper = np.full(len(test), np.nan)
    validation_has_censoring = bool(
        not validation.empty and validation[censor_column].any()
    )
    test_has_censoring = bool(test_censored.any())
    if (
        not validation.empty
        and not validation_has_censoring
        and not test_has_censoring
    ):
        validation_median = model.predict_median(
            validation[features].to_numpy(dtype=float)
        )
        try:
            if conformal_group_column is None:
                radius = conformal_log_radius(
                    validation[label_column].to_numpy(dtype=float),
                    validation_median,
                    coverage=conformal_coverage,
                    is_censored=validation[censor_column].to_numpy(dtype=bool),
                )
                calibration_method = "cell_absolute_log_residual"
                calibration_group_count = None
                calibration_estimand = (
                    "marginal observed-cell coverage under row exchangeability"
                )
            else:
                radius = group_max_conformal_log_radius(
                    validation[label_column].to_numpy(dtype=float),
                    validation_median,
                    validation[conformal_group_column].to_numpy(),
                    coverage=conformal_coverage,
                    is_censored=validation[censor_column].to_numpy(dtype=bool),
                )
                calibration_method = "group_maximum_absolute_log_residual"
                calibration_group_count = int(
                    validation[conformal_group_column].nunique()
                )
                calibration_estimand = (
                    "simultaneous observed-cell coverage for a new exchangeable group"
                )
            conformal_lower, conformal_upper = log_symmetric_interval(median, radius)
            conformal_evaluation: dict[str, object] = {
                "status": "available",
                "coverage": conformal_coverage,
                "method": calibration_method,
                "estimand": calibration_estimand,
                "log_radius": radius,
                "calibration_row_count": len(validation),
                "calibration_group_count": calibration_group_count,
                "calibration_observed_event_count": int(
                    (~validation[censor_column]).sum()
                ),
                "test_interval_observed_only": evaluate_prediction_interval(
                    test_lifetime,
                    conformal_lower,
                    conformal_upper,
                    nominal_coverage=conformal_coverage,
                    is_censored=test_censored,
                ).to_dict(),
            }
            if conformal_group_column is not None:
                conformal_evaluation["test_simultaneous_group_coverage"] = (
                    evaluate_group_prediction_interval(
                        test_lifetime,
                        conformal_lower,
                        conformal_upper,
                        test[conformal_group_column].to_numpy(),
                        nominal_coverage=conformal_coverage,
                        is_censored=test_censored,
                    ).to_dict()
                )
            evaluation["conformal"] = conformal_evaluation
        except ConformalCalibrationError as error:
            evaluation["conformal"] = {
                "status": "unavailable",
                "reason": str(error),
            }
    else:
        if validation_has_censoring or test_has_censoring:
            reason = (
                "Nominal conformal coverage is disabled when calibration or test "
                "rows are right-censored; censored-aware calibration is not implemented"
            )
        else:
            reason = "No validation split with observed EOL events"
        evaluation["conformal"] = {
            "status": "unavailable",
            "reason": reason,
        }

    identity_columns = [
        column
        for column in ("dataset_id", "cell_id", "batch_id", "protocol_id")
        if column in test
    ]
    predictions = test[identity_columns].reset_index(drop=True).copy()
    predictions["observed_time"] = test_lifetime
    predictions["is_censored"] = test_censored
    predictions["p10"] = quantiles[:, 0]
    predictions["p50"] = median
    predictions["p90"] = quantiles[:, 2]
    predictions["conformal_lower_80"] = conformal_lower
    predictions["conformal_upper_80"] = conformal_upper
    return evaluation, predictions


def run_tuned_probabilistic_lifetime_experiment(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    l2_candidates: Sequence[float],
    group_column: str = "protocol_id",
    label_column: str = "cycle_life",
    censor_column: str = "is_censored",
    conformal_coverage: float = 0.8,
    inner_cv_folds: int = 5,
    inner_cv_seed: int = 42,
    identity_column: str = "cell_id",
    group_isolation_columns: Sequence[str] = ("cell_id",),
    conformal_group_column: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Tune only on outer-training groups, then run the frozen experiment."""
    if "split" not in frame:
        raise ValueError("Missing probabilistic experiment column: split")
    outer_train = frame.loc[frame["split"] == "train"]
    selection = select_lognormal_aft_l2(
        outer_train,
        feature_columns=feature_columns,
        l2_candidates=l2_candidates,
        group_column=group_column,
        label_column=label_column,
        censor_column=censor_column,
        n_splits=inner_cv_folds,
        seed=inner_cv_seed,
    )
    evaluation, predictions = run_probabilistic_lifetime_experiment(
        frame,
        feature_columns=feature_columns,
        label_column=label_column,
        censor_column=censor_column,
        conformal_coverage=conformal_coverage,
        l2_penalty=selection.selected_l2_penalty,
        identity_column=identity_column,
        group_isolation_columns=group_isolation_columns,
        conformal_group_column=conformal_group_column,
    )
    evaluation["model_selection"] = selection.to_dict()
    return evaluation, predictions
