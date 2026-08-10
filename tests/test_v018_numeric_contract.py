from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.calendar_long_horizon_v017_contract import (
    load_v022_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v017_partition import (
    V022PartitionContractError,
    canonicalize_partition_output,
)
from lifetwin.experiments.calendar_long_horizon_v018_numeric_contract import (
    V023NumericContractError,
    validate_risk_bundle_numeric_contract,
)


def _synthetic_risk_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = load_v022_contract_view().artifacts
    schema = contract.csv_schema("risk_bundle.csv")
    risks: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    for cluster_index in range(600):
        all_finite = cluster_index % 11 != 0
        hard_eligible = all_finite and cluster_index % 7 != 0
        cluster_id = f"synthetic-cluster-{cluster_index:04d}"
        features.append(
            {
                "partition": "center_development",
                "cluster_id": cluster_id,
                "hard_eligible": hard_eligible,
                "all_features_finite": all_finite,
            }
        )
        for score_index, score_id in enumerate(schema.required_score_ids):
            primary = score_id in {"prefix_only", "visible_stress"}
            risks.append(
                {
                    "protocol_id": contract.protocol_id,
                    "partition": "center_development",
                    "cluster_id": cluster_id,
                    "score_id": score_id,
                    "raw_risk_score": (
                        float(score_index + 1) if all_finite else math.nan
                    ),
                    "calibrated_catastrophic_probability": (
                        0.25 if primary and hard_eligible else math.nan
                    ),
                    "all_features_finite": all_finite,
                    "successful_structure_family_count": 2,
                    "fit_failure_count": 4,
                    "effective_unique_shape_count": 2.0,
                    "canonical_predictor_content_sha256": (
                        f"{cluster_index * 9 + score_index + 1:064x}"
                    ),
                }
            )
    risk = pd.DataFrame(risks, columns=schema.columns).sort_values(
        list(schema.key), kind="stable"
    )
    return risk.reset_index(drop=True), pd.DataFrame(features)


def test_v022_blanket_finite_gate_reproduces_structural_nan_failure() -> None:
    risk, _ = _synthetic_risk_and_features()
    contract = load_v022_contract_view().artifacts
    with pytest.raises(
        V022PartitionContractError,
        match="risk_bundle.csv contains a nonfinite numeric value",
    ):
        canonicalize_partition_output(
            risk,
            filename="risk_bundle.csv",
            partition="center_development",
            required_rows=5_400,
            contract=contract,
        )


def test_v023_contract_accepts_only_exact_structural_nan_masks() -> None:
    risk, features = _synthetic_risk_and_features()
    validate_risk_bundle_numeric_contract(risk, features)

    primary = risk["score_id"].eq("prefix_only")
    eligible = primary & risk["cluster_id"].eq("synthetic-cluster-0001")
    broken = risk.copy()
    broken.loc[eligible, "calibrated_catastrophic_probability"] = np.nan
    with pytest.raises(V023NumericContractError, match="eligibility mask"):
        validate_risk_bundle_numeric_contract(broken, features)

    nonprimary = risk["score_id"].eq("placebo_8")
    broken = risk.copy()
    broken.loc[nonprimary, "calibrated_catastrophic_probability"] = 0.0
    with pytest.raises(V023NumericContractError, match="eligibility mask"):
        validate_risk_bundle_numeric_contract(broken, features)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("raw_risk_score", np.inf, "infinity"),
        ("calibrated_catastrophic_probability", -0.1, "eligibility mask"),
        ("effective_unique_shape_count", np.nan, "must be finite"),
    ],
)
def test_v023_contract_rejects_true_numeric_corruption(
    column: str,
    value: float,
    message: str,
) -> None:
    risk, features = _synthetic_risk_and_features()
    eligible = risk["cluster_id"].eq("synthetic-cluster-0001") & risk[
        "score_id"
    ].eq("prefix_only")
    risk.loc[eligible, column] = value
    with pytest.raises(V023NumericContractError, match=message):
        validate_risk_bundle_numeric_contract(risk, features)


def test_v023_contract_rejects_cross_table_mask_drift() -> None:
    risk, features = _synthetic_risk_and_features()
    features.loc[0, "all_features_finite"] = not bool(
        features.loc[0, "all_features_finite"]
    )
    with pytest.raises(V023NumericContractError, match="masks differ"):
        validate_risk_bundle_numeric_contract(risk, features)
