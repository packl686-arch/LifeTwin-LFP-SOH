from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.calendar_long_horizon_v019_numeric_contract import (
    V024NumericContractError,
    validate_pipeline_numeric_contract,
)


def _outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clusters = ("eligible", "abstained")
    features = pd.DataFrame(
        {
            "partition": ["test", "test"],
            "cluster_id": list(clusters),
            "hard_eligible": [True, False],
            "all_features_finite": [True, False],
            "feature_a": [1.0, np.nan],
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "partition": "test",
                "cluster_id": cluster,
                "forecast_day": 365.25,
                "center_forecast_pct": 95.0,
                "sqrt_time_forecast_pct": 94.0,
                "bounded_power_forecast_pct": 93.0,
                "base_interval_lower_pct": 90.0,
                "base_interval_upper_pct": 98.0,
                "calibrated_interval_lower_pct": 89.0,
                "calibrated_interval_upper_pct": 99.0,
            }
            for cluster in clusters
        ]
    )
    score_ids = (
        "prefix_only",
        "visible_stress",
        "placebo_8",
        "arm_a_plus_s_plan",
        "strongest_single_feature",
        "planned_stress_only",
        "prefix_rmse_only",
        "v1_max_envelope_only",
        "center_sqrt_abs_difference_only",
    )
    risk = pd.DataFrame(
        [
            {
                "partition": "test",
                "cluster_id": cluster,
                "score_id": score_id,
                "raw_risk_score": 0.1 if cluster == "eligible" else np.nan,
                "calibrated_catastrophic_probability": (
                    0.2
                    if cluster == "eligible"
                    and score_id in {"prefix_only", "visible_stress"}
                    else np.nan
                ),
                "all_features_finite": cluster == "eligible",
                "successful_structure_family_count": 6,
                "fit_failure_count": 0,
                "effective_unique_shape_count": 6.0,
            }
            for cluster in clusters
            for score_id in score_ids
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "partition": "test",
                "cluster_id": cluster,
                "arm": arm,
                "raw_risk_score": 0.1 if cluster == "eligible" else np.nan,
                "hard_eligible": cluster == "eligible",
                "issuance_rank": 1 if cluster == "eligible" else np.nan,
                "issued": cluster == "eligible",
            }
            for cluster in clusters
            for arm in ("prefix_only", "visible_stress")
        ]
    )
    return predictions, features, risk, decisions


def _validate(
    prediction: pd.DataFrame,
    features: pd.DataFrame,
    risk: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    validate_pipeline_numeric_contract(
        prediction_bundle=prediction,
        feature_bundle=features,
        risk_bundle=risk,
        decision_bundle=decisions,
        primary_issue_counts={"test": 1},
    )


def test_complete_output_numeric_contract_accepts_exact_structural_nan() -> None:
    _validate(*_outputs())


@pytest.mark.parametrize(
    ("target", "column", "row", "value", "message"),
    (
        ("prediction", "center_forecast_pct", 0, np.inf, "fully finite"),
        ("features", "all_features_finite", 1, True, "does not match"),
        ("risk", "raw_risk_score", 9, 0.0, "all_features_finite"),
        ("decisions", "issuance_rank", 2, 2.0, "ineligible decision has a rank"),
    ),
)
def test_complete_output_numeric_contract_rejects_mask_corruption(
    target: str,
    column: str,
    row: int,
    value: object,
    message: str,
) -> None:
    prediction, features, risk, decisions = _outputs()
    frames = {
        "prediction": prediction,
        "features": features,
        "risk": risk,
        "decisions": decisions,
    }
    frames[target].loc[row, column] = value
    with pytest.raises(V024NumericContractError, match=message):
        _validate(prediction, features, risk, decisions)
