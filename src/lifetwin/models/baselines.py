from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> RegressionMetrics:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional with equal shape")
    if len(truth) == 0:
        raise ValueError("Cannot evaluate an empty prediction")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite truth and predictions")
    if (truth == 0).any():
        raise ValueError("MAPE is undefined for zero-valued targets")

    error = prediction - truth
    return RegressionMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        mape=float(np.mean(np.abs(error / truth)) * 100),
        sample_count=len(truth),
    )


def bootstrap_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    groups: Sequence[object] | None = None,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, object]:
    """Return point metrics with a deterministic percentile group-bootstrap CI."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    point = regression_metrics(truth, prediction)
    if groups is None:
        group_codes = np.arange(len(truth), dtype=int)
        group_count = len(truth)
        method = "sample_percentile"
    else:
        group_values = pd.Series(groups, copy=False)
        if len(group_values) != len(truth):
            raise ValueError("groups must have the same length as y_true")
        if group_values.isna().any():
            raise ValueError("Bootstrap groups cannot contain null values")
        group_codes, unique_groups = pd.factorize(group_values, sort=False)
        group_count = len(unique_groups)
        if group_count < 2:
            raise ValueError("Grouped bootstrap requires at least two groups")
        method = "group_percentile"

    group_indices = [np.flatnonzero(group_codes == code) for code in range(group_count)]
    rng = np.random.default_rng(random_state)
    bootstrap_values = {
        "mae": np.empty(n_resamples, dtype=float),
        "rmse": np.empty(n_resamples, dtype=float),
        "mape": np.empty(n_resamples, dtype=float),
    }
    for sample_number in range(n_resamples):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        sampled_indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        sampled = regression_metrics(
            truth[sampled_indices], prediction[sampled_indices]
        )
        bootstrap_values["mae"][sample_number] = sampled.mae
        bootstrap_values["rmse"][sample_number] = sampled.rmse
        bootstrap_values["mape"][sample_number] = sampled.mape

    tail = (1 - confidence_level) / 2
    intervals = {
        name: {
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1 - tail)),
        }
        for name, values in bootstrap_values.items()
    }
    interval_key = f"confidence_interval_{confidence_level * 100:g}"
    return {
        **point.to_dict(),
        interval_key: intervals,
        "bootstrap": {
            "method": method,
            "resamples": n_resamples,
            "group_count": group_count,
            "confidence_level": confidence_level,
            "random_state": random_state,
        },
    }


def _numeric_feature_columns(
    frame: pd.DataFrame,
    *,
    label_column: str,
    excluded: Sequence[str] = (),
) -> list[str]:
    blocked = {
        label_column,
        "split",
        "dataset_id",
        "cell_id",
        "batch_id",
        "protocol_id",
        "label_source",
        "is_censored",
        "eol_threshold",
        *excluded,
    }
    columns = [
        column
        for column in frame.select_dtypes(include=["number"]).columns
        if column not in blocked
    ]
    if not columns:
        raise ValueError("No numeric feature columns remain after exclusions")
    return columns


def run_baselines(
    frame: pd.DataFrame,
    *,
    label_column: str = "cycle_life",
    feature_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] = (),
    elastic_net_alpha: float = 0.02,
    elastic_net_l1_ratio: float = 0.5,
    target_transform: str = "identity",
    bootstrap_resamples: int = 0,
    bootstrap_group_column: str | None = None,
    bootstrap_random_state: int = 42,
) -> dict[str, dict[str, dict[str, object]]]:
    """Fit median, linear, and Elastic Net baselines on a frozen split column."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to run baselines") from exc

    required = {label_column, "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")

    features = list(feature_columns) if feature_columns else _numeric_feature_columns(
        frame, label_column=label_column
    )
    categorical = list(categorical_columns)
    missing_features = sorted(set([*features, *categorical]) - set(frame.columns))
    if missing_features:
        raise ValueError(f"Missing requested feature columns: {missing_features}")
    overlap = set(features) & set(categorical)
    if overlap:
        raise ValueError(f"Features cannot be both numeric and categorical: {sorted(overlap)}")
    train = frame.loc[frame["split"] == "train"]
    if train.empty:
        raise ValueError("Training split is empty")

    train_target = train[label_column].to_numpy(dtype=float)
    if target_transform == "identity":
        model_target = train_target
    elif target_transform == "log10":
        if (train_target <= 0).any():
            raise ValueError("log10 target transform requires positive targets")
        model_target = np.log10(train_target)
    else:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    median_prediction = float(np.median(train_target))
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples cannot be negative")
    if bootstrap_group_column is not None and bootstrap_group_column not in frame:
        raise ValueError(
            f"Missing bootstrap group column: {bootstrap_group_column}"
        )

    def make_preprocessor() -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers: list[tuple[str, object, list[str]]] = [
            ("numeric", numeric_pipeline, features)
        ]
        if categorical:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical))
        return ColumnTransformer(transformers=transformers)

    linear_regression = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", LinearRegression()),
        ]
    )
    elastic_net = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                ElasticNet(
                    alpha=elastic_net_alpha,
                    l1_ratio=elastic_net_l1_ratio,
                    max_iter=20000,
                    random_state=42,
                ),
            ),
        ]
    )
    model_columns = [*features, *categorical]
    linear_regression.fit(train[model_columns], model_target)
    elastic_net.fit(train[model_columns], model_target)

    output: dict[str, dict[str, dict[str, object]]] = {
        "median": {},
        "linear_regression": {},
        "elastic_net": {},
    }
    for split_name in ("validation", "test"):
        split = frame.loc[frame["split"] == split_name]
        if split.empty:
            continue
        truth = split[label_column].to_numpy(dtype=float)
        median = np.full(len(split), median_prediction)
        linear = linear_regression.predict(split[model_columns])
        elastic = elastic_net.predict(split[model_columns])
        if target_transform == "log10":
            linear = np.power(10.0, linear)
            elastic = np.power(10.0, elastic)
        groups = (
            split[bootstrap_group_column].to_numpy()
            if bootstrap_group_column is not None
            else None
        )

        def evaluate(prediction: np.ndarray) -> dict[str, object]:
            if bootstrap_resamples:
                return bootstrap_regression_metrics(
                    truth,
                    prediction,
                    groups=groups,
                    n_resamples=bootstrap_resamples,
                    random_state=bootstrap_random_state,
                )
            return regression_metrics(truth, prediction).to_dict()

        output["median"][split_name] = evaluate(median)
        output["linear_regression"][split_name] = evaluate(linear)
        output["elastic_net"][split_name] = evaluate(elastic)
    return output
