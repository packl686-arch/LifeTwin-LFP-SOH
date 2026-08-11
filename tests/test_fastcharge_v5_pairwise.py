from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lifetwin.experiments.fastcharge_v5_pairwise as v5
from lifetwin.experiments.fastcharge_safe_prior_v2 import (
    _core_config,
    load_fastcharge_safe_prior_v2_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG = PROJECT_ROOT / "configs/experiments/fastcharge_lfp_safe_prior_v2.json"


@pytest.fixture(scope="module")
def core_config() -> dict[str, object]:
    return _core_config(load_fastcharge_safe_prior_v2_config(V2_CONFIG))


def _synthetic_cells(count: int = 5, support: int = 30) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_index in range(count):
        for cycle in range(1, support + 1):
            rows.append(
                {
                    "cell_id": f"MATR_B{1 + cell_index % 2}C{cell_index}",
                    "cycle_index": cycle,
                    "discharge_capacity_ah": (
                        1.1
                        - (0.00012 + 0.00001 * cell_index) * cycle
                        - 0.00002 * np.sqrt(cycle) * cell_index
                    ),
                    "internal_resistance_ohm": (
                        0.015 + 0.000002 * cycle * (cell_index + 1)
                    ),
                    "temperature_max_c": 34.0 + 0.01 * cycle + cell_index,
                    "charge_time_s": 700.0 + cycle * (cell_index + 1),
                    "energy_efficiency": 0.9 - 0.00001 * cycle * cell_index,
                }
            )
    return pd.DataFrame(rows)


def test_pair_matrix_is_antisymmetric_and_uses_only_fit_cells(
    core_config: dict[str, object],
) -> None:
    cells = v5._validated_cells(_synthetic_cells(4), required_support=30)
    fit_ids = sorted(cells)[:3]
    held_out = sorted(cells)[3:]
    fit = {cell_id: cells[cell_id] for cell_id in fit_ids}
    features, target, audit = v5.build_pairwise_training_matrix(
        fit,
        20,
        30,
        core_config,
        anchor_stride=5,
    )
    assert features.shape == (18, 4 * len(v5.DESCRIPTOR_IDS))
    assert target.shape == (18,)
    assert np.allclose(features[0::2], -features[1::2])
    assert np.allclose(target[0::2], -target[1::2])
    v5.assert_pair_fold_firewall(fit_ids, held_out, audit)

    attacked = dict(audit)
    attacked["reference_cell_ids"] = [*fit_ids, held_out[0]]
    with pytest.raises(v5.FastChargeV5PairwiseError, match="fit-cell firewall"):
        v5.assert_pair_fold_firewall(fit_ids, held_out, attacked)


def test_weighted_median_and_aggregation_are_deterministic() -> None:
    trajectories = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [100.0, 30.0]], dtype=float
    )
    weights = np.asarray([0.4, 0.35, 0.25])
    assert np.array_equal(v5.weighted_median(trajectories, weights), [2.0, 20.0])
    assert np.array_equal(
        v5.aggregate_reference_trajectories(
            trajectories, weights, "weighted_median"
        ),
        [2.0, 20.0],
    )
    mean = v5.aggregate_reference_trajectories(
        trajectories, weights, "weighted_mean"
    )
    assert np.allclose(mean, [26.1, 18.5])


def test_prediction_is_invariant_to_unavailable_target_suffix(
    core_config: dict[str, object],
) -> None:
    cells = v5._validated_cells(_synthetic_cells(10), required_support=30)
    target_id = sorted(cells)[-1]
    references = {
        cell_id: cell for cell_id, cell in cells.items() if cell_id != target_id
    }
    features, target, _ = v5.build_pairwise_training_matrix(
        references,
        20,
        30,
        core_config,
        anchor_stride=5,
    )
    estimator = v5.make_estimator(
        v5.ModelSpec("test_ridge", "ridge", (("alpha", 1.0),)),
        pairwise=True,
    ).fit(features, target)
    prefix = cells[target_id].loc[cells[target_id]["cycle_index"] <= 20]
    prediction_a, audit_a = v5.predict_pairwise_trajectory(
        estimator,
        prefix,
        references,
        20,
        30,
        core_config,
        aggregation="weighted_mean",
        neighbor_count=8,
    )
    attacked_full = cells[target_id].copy()
    attacked_full.loc[attacked_full["cycle_index"] > 20, "discharge_capacity_ah"] = 0.1
    attacked_prefix = attacked_full.loc[attacked_full["cycle_index"] <= 20]
    prediction_b, audit_b = v5.predict_pairwise_trajectory(
        estimator,
        attacked_prefix,
        references,
        20,
        30,
        core_config,
        aggregation="weighted_mean",
        neighbor_count=8,
    )
    assert np.array_equal(prediction_a, prediction_b)
    assert audit_a == audit_b
    assert target_id not in audit_a["reference_cell_ids"]


def test_deterministic_folds_keep_physical_cells_disjoint() -> None:
    cell_ids = [f"MATR_B{batch}C{index}" for batch in (1, 2) for index in range(5)]
    folds = v5.deterministic_cell_folds(cell_ids, fold_count=5)
    assert len(folds) == 5
    observed: list[str] = []
    for fit, held_out in folds:
        assert len(held_out) == 2
        assert not set(fit) & set(held_out)
        observed.extend(held_out)
    assert sorted(observed) == sorted(cell_ids)


def test_finite_sample_quantile_and_interval_score() -> None:
    quantile, level = v5.finite_sample_absolute_quantile(
        np.arange(1.0, 11.0), coverage=0.8
    )
    assert level == 0.9
    assert quantile == 10.0
    score = v5.weighted_interval_score(
        np.asarray([0.0, -2.0, 3.0]),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        coverage=0.8,
    )
    assert np.allclose(score, [2.0 / 15.0, 22.0 / 15.0, 37.0 / 15.0])
