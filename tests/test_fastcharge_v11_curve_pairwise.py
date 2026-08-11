from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lifetwin.experiments.fastcharge_v11_curve_pairwise as v11
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


def _features() -> pd.DataFrame:
    rows = []
    for index in range(6):
        rows.append(
            {
                "cell_id": f"MATR_B{1 + index % 2}C{index}",
                "early_cycle": 10,
                "late_cycle": 100,
                "log10_delta_q_variance": -5.0 + 0.1 * index,
                "delta_q_min_ah": -0.02 + 0.001 * index,
                "delta_q_mean_ah": 0.002 * index,
                "delta_q_abs_area_ah_v": 0.01 + 0.001 * index,
                "delta_q_skewness": -0.3 + 0.05 * index,
                "delta_q_kurtosis": -1.0 + 0.1 * index,
            }
        )
    return pd.DataFrame(rows, columns=v11.CURVE_INPUT_COLUMNS)


def test_curve_feature_firewall_rejects_unregistered_label_column() -> None:
    bad = _features().assign(cycle_life=1000)
    with pytest.raises(FastChargeV5PairwiseError, match="columns changed"):
        v11.validate_curve_features(bad, required_cell_ids=bad["cell_id"])


def test_curve_feature_contract_requires_cycle_10_to_100() -> None:
    bad = _features()
    bad.loc[0, "late_cycle"] = 101
    with pytest.raises(FastChargeV5PairwiseError, match="cycle-10-to-100"):
        v11.validate_curve_features(bad, required_cell_ids=bad["cell_id"])


def test_robust_scaler_uses_only_registered_fit_cells() -> None:
    features = _features()
    fit_ids = features["cell_id"].iloc[:4].tolist()
    scaler = v11.fit_curve_scaler(features, fit_ids)
    first = v11.transform_curve_features(features, scaler, fit_ids)
    changed = features.copy()
    changed.loc[
        changed["cell_id"] == features["cell_id"].iloc[5], v11.CURVE_FEATURE_IDS
    ] = 1e9
    replay = v11.fit_curve_scaler(changed, fit_ids)
    second = v11.transform_curve_features(changed, replay, fit_ids)
    assert np.array_equal(scaler.median, replay.median)
    assert np.array_equal(scaler.scale, replay.scale)
    for cell_id in fit_ids:
        assert np.array_equal(first[cell_id], second[cell_id])


def test_curve_feature_validation_is_deterministic_and_filters_extra_cells() -> None:
    features = _features()
    required = features["cell_id"].iloc[:4].tolist()
    first = v11.validate_curve_features(features, required_cell_ids=required)
    second = v11.validate_curve_features(
        features.sample(frac=1.0, random_state=4), required_cell_ids=required
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["cell_id"].tolist() == sorted(required)
