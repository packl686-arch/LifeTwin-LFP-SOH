from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from lifetwin.data.snl import DATASET_ID, RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import canonical_json_sha256
from lifetwin.experiments.private_dual_clock_post_outcome_audit import (
    CELL_AUDIT_COLUMNS,
    CONDITION_AUDIT_COLUMNS,
    audit_private_dual_clock_v3,
)
from lifetwin.experiments.private_dual_clock_prior_v3 import (
    DECISION_COLUMNS as V3_DECISION_COLUMNS,
    PREDICTION_COLUMNS as V3_PREDICTION_COLUMNS,
    SCORE_COLUMNS as V3_SCORE_COLUMNS,
    PrivateDualClockPriorV3Error,
    default_private_dual_clock_prior_v3_config,
    predict_private_dual_clock_prior_capsule,
    predict_private_dual_clock_prior_v3,
    score_private_dual_clock_prior_v3,
    train_private_dual_clock_prior_capsule,
    _fit_dual,
    _predict_dual,
)
from lifetwin.experiments.private_dual_clock_uncertainty_audit import (
    CELL_UNCERTAINTY_COLUMNS,
    CONDITION_UNCERTAINTY_COLUMNS,
    audit_private_dual_clock_uncertainty,
)
from lifetwin.experiments.private_cycle_prior_v2 import (
    DECISION_COLUMNS,
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    PrivateCyclePriorV2Error,
    default_private_cycle_prior_v2_config,
    predict_private_cycle_prior_capsule,
    predict_private_cycle_prior_v2,
    score_private_cycle_prior_v2,
    train_private_cycle_prior_capsule,
)
from lifetwin.experiments.snl_rpt_loco import (
    REFERENCE_COLUMNS,
    TARGET_PREFIX_COLUMNS,
    TARGET_TRUTH_COLUMNS,
)
from lifetwin.models.hierarchical_cycle_prior import (
    BasisKernelPrior,
    DualClockKernelPrior,
    PowerConditionPrior,
    fit_basis_kernel_prior,
    fit_dual_clock_kernel_prior,
    fit_power_condition_prior,
    predict_basis_kernel_prior,
    predict_dual_clock_kernel_prior,
    predict_power_condition_prior,
)


def _trajectories() -> pd.DataFrame:
    records = []
    conditions = [
        ("C1", 15.0, 1.0, 1.0, 2.1, 0.25),
        ("C2", 25.0, 0.6, 1.0, 2.8, 0.35),
        ("C3", 35.0, 1.0, 2.0, 4.0, 0.55),
        ("C4", 25.0, 0.2, 3.0, 3.2, 0.45),
    ]
    for condition_id, temperature, dod, rate, sqrt_rate, linear_rate in conditions:
        for replicate in ("a", "b"):
            replicate_shift = 0.08 if replicate == "b" else -0.08
            for visit, efc in enumerate((0.0, 100.0, 200.0, 300.0, 400.0, 500.0)):
                scaled = efc / 1000.0
                fade = (
                    (sqrt_rate + replicate_shift) * np.sqrt(scaled)
                    + linear_rate * scaled
                )
                records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "cell_id": f"CELL_{condition_id}_{replicate}",
                        "condition_id": condition_id,
                        "temperature_c": temperature,
                        "min_soc_pct": 0.0,
                        "max_soc_pct": dod * 100.0,
                        "dod_fraction": dod,
                        "charge_c_rate": 0.5,
                        "discharge_c_rate": rate,
                        "visit_index": visit,
                        "elapsed_days": efc / 4.0,
                        "equivalent_full_cycles": efc,
                        "capacity_ah": 1.1 * (100.0 - fade) / 100.0,
                        "capacity_retention_pct": 100.0 - fade,
                        "rpt_cycle_count": 3,
                    }
                )
    return pd.DataFrame(records, columns=RPT_TRAJECTORY_COLUMNS).sort_values(
        ["condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _trajectories()
    references = []
    prefixes = []
    truths = []
    for outer in sorted(data["condition_id"].unique()):
        reference = data.loc[data["condition_id"] != outer].copy()
        reference.insert(0, "outer_condition_id", outer)
        references.append(reference.loc[:, REFERENCE_COLUMNS])
        target = data.loc[data["condition_id"] == outer].copy()
        truth = target.copy()
        truth.insert(0, "outer_condition_id", outer)
        truths.append(truth.loc[:, TARGET_TRUTH_COLUMNS])
        prefix = target.loc[target["visit_index"] < 3].copy()
        prefix.insert(0, "outer_condition_id", outer)
        prefix["landmark_visit_count"] = 3
        prefixes.append(prefix.loc[:, TARGET_PREFIX_COLUMNS])
    return (
        pd.concat(references, ignore_index=True).sort_values(
            ["outer_condition_id", "condition_id", "cell_id", "visit_index"],
            kind="stable",
            ignore_index=True,
        ),
        pd.concat(prefixes, ignore_index=True).sort_values(
            ["outer_condition_id", "cell_id", "visit_index"],
            kind="stable",
            ignore_index=True,
        ),
        pd.concat(truths, ignore_index=True).sort_values(
            ["outer_condition_id", "cell_id", "visit_index"],
            kind="stable",
            ignore_index=True,
        ),
    )


def _config() -> dict[str, object]:
    config = default_private_cycle_prior_v2_config()
    config["landmark_visit_counts"] = [3]
    config["score_end_equivalent_full_cycles"] = 500.0
    config["power_family"].update(
        {
            "exponents": [0.5, 0.7],
            "ridge_alphas": [1.0],
            "prefix_rate_weights": [0.0, 0.1],
        }
    )
    config["basis_family"].update(
        {
            "basis_exponent_pairs": [[0.3, 1.0]],
            "kernel_gammas": [1.0],
            "coefficient_shrinkages": [1.0, 10.0],
            "anchor_weights": [0.5, 1.0],
        }
    )
    config["blend"]["power_weights"] = [0.0, 0.5, 1.0]
    config["uncertainty"]["horizon_bins_efc"] = [0.0, 200.0, 500.0]
    return config


def _v3_config() -> dict[str, object]:
    config = default_private_dual_clock_prior_v3_config()
    config["landmark_visit_counts"] = [3]
    config["score_end_equivalent_full_cycles"] = 500.0
    config["dual_clock_family"].update(
        {
            "time_exponents": [0.3, 0.5],
            "cycle_exponents": [1.0],
            "kernel_gammas": [0.3, 1.0],
            "coefficient_shrinkages": [1.0, 10.0],
            "anchor_weights": [0.5, 1.0],
        }
    )
    config["uncertainty"]["horizon_bins_efc"] = [0.0, 200.0, 500.0]
    return config


def test_hierarchical_prior_roundtrip_and_prediction() -> None:
    data = _trajectories()
    references = data.loc[data["condition_id"] != "C4"]
    prefix = data.loc[data["cell_id"] == "CELL_C4_a"].iloc[:3]
    forecast = np.asarray([250.0, 350.0, 500.0])
    power = fit_power_condition_prior(
        references, exponent=0.7, alpha=1.0
    )
    basis = fit_basis_kernel_prior(
        references, basis_exponents=(0.3, 1.0), gamma=1.0
    )
    power_replay = PowerConditionPrior.from_dict(power.to_dict())
    basis_replay = BasisKernelPrior.from_dict(basis.to_dict())
    np.testing.assert_allclose(
        predict_power_condition_prior(
            prefix,
            forecast,
            power,
            prefix_rate_weight=0.1,
        ),
        predict_power_condition_prior(
            prefix,
            forecast,
            power_replay,
            prefix_rate_weight=0.1,
        ),
    )
    np.testing.assert_allclose(
        predict_basis_kernel_prior(
            prefix,
            forecast,
            basis,
            shrinkage=10.0,
            anchor_weight=0.5,
        ),
        predict_basis_kernel_prior(
            prefix,
            forecast,
            basis_replay,
            shrinkage=10.0,
            anchor_weight=0.5,
        ),
    )


def test_private_v2_prediction_firewall_scoring_and_capsule() -> None:
    references, prefixes, truth = _inputs()
    config = _config()
    predictions, decisions, manifest = predict_private_cycle_prior_v2(
        references, prefixes, config
    )
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert tuple(decisions.columns) == DECISION_COLUMNS
    assert manifest["target_truth_argument_accepted"] is False
    assert manifest["private_only"] is True

    attacked_truth = truth.copy()
    attacked_truth.loc[
        attacked_truth["visit_index"] >= 3, "capacity_retention_pct"
    ] -= 20.0
    replay, replay_decisions, replay_manifest = predict_private_cycle_prior_v2(
        references, prefixes, config
    )
    pd.testing.assert_frame_equal(predictions, replay)
    pd.testing.assert_frame_equal(decisions, replay_decisions)
    assert manifest == replay_manifest
    scores, summary = score_private_cycle_prior_v2(
        truth, predictions, decisions, manifest, config
    )
    attacked_scores, _ = score_private_cycle_prior_v2(
        attacked_truth, predictions, decisions, manifest, config
    )
    assert tuple(scores.columns) == SCORE_COLUMNS
    assert summary["private_only"] is True
    assert not scores["trajectory_iae_pp"].equals(
        attacked_scores["trajectory_iae_pp"]
    )

    capsule = train_private_cycle_prior_capsule(
        _trajectories(),
        config,
        training_identity={"fixture": True},
    )
    cell_prefix = _trajectories().loc[
        _trajectories()["cell_id"] == "CELL_C1_a"
    ].iloc[:3]
    output, metadata = predict_private_cycle_prior_capsule(
        cell_prefix,
        [300.0, 400.0, 500.0],
        capsule,
    )
    assert len(output) == 3
    assert metadata["evidence_status"] == "supported"
    assert capsule["raw_training_rows_in_capsule"] is False

    attacked_capsule = deepcopy(capsule)
    attacked_capsule["landmark_models"]["3"]["power_blend_weight"] = 0.123
    assert canonical_json_sha256(
        {key: value for key, value in attacked_capsule.items() if key != "capsule_content_sha256"}
    ) != attacked_capsule["capsule_content_sha256"]
    with pytest.raises(PrivateCyclePriorV2Error, match="capsule content changed"):
        predict_private_cycle_prior_capsule(
            cell_prefix,
            [300.0],
            attacked_capsule,
        )


def test_private_v2_scorer_rejects_prediction_tampering() -> None:
    references, prefixes, truth = _inputs()
    config = _config()
    predictions, decisions, manifest = predict_private_cycle_prior_v2(
        references, prefixes, config
    )
    attacked = predictions.copy()
    attacked.loc[0, "predicted_capacity_retention_pct"] += 0.1
    with pytest.raises(PrivateCyclePriorV2Error, match="changed after freeze"):
        score_private_cycle_prior_v2(
            truth,
            attacked,
            decisions,
            manifest,
            config,
        )


def test_dual_clock_prior_roundtrip_and_explicit_schedule() -> None:
    data = _trajectories()
    references = data.loc[data["condition_id"] != "C4"]
    prefix = data.loc[data["cell_id"] == "CELL_C4_a"].iloc[:3]
    model = fit_dual_clock_kernel_prior(
        references,
        time_exponent=0.3,
        cycle_exponent=1.0,
        gamma=0.3,
    )
    replay = DualClockKernelPrior.from_dict(model.to_dict())
    forecast = np.asarray([300.0, 400.0, 500.0])
    inferred = predict_dual_clock_kernel_prior(
        prefix,
        forecast,
        model,
        shrinkage=1.0,
        anchor_weight=0.5,
    )
    explicit = predict_dual_clock_kernel_prior(
        prefix,
        forecast,
        replay,
        shrinkage=1.0,
        anchor_weight=0.5,
        forecast_elapsed_days=forecast / 4.0,
    )
    np.testing.assert_allclose(inferred, explicit)


def test_dual_clock_fast_path_matches_validated_path() -> None:
    data = _trajectories()
    cases = [
        ("C4", "CELL_C4_a", 3, 0.3, 1.0, 0.3, 1.0, 0.75),
        ("C3", "CELL_C3_b", 4, 0.5, 0.7, 1.0, 10.0, 0.5),
    ]
    for held, cell_id, landmark, time_exp, cycle_exp, gamma, shrinkage, anchor in cases:
        references = data.loc[data["condition_id"] != held]
        prefix = data.loc[data["cell_id"] == cell_id].iloc[:landmark]
        forecast = np.asarray([450.0, 500.0])
        hyperparameters = {
            "time_exponent": time_exp,
            "cycle_exponent": cycle_exp,
            "gamma": gamma,
            "shrinkage": shrinkage,
            "anchor_weight": anchor,
        }
        fast_model = _fit_dual(references, hyperparameters)
        validated_model = fit_dual_clock_kernel_prior(
            references,
            time_exponent=time_exp,
            cycle_exponent=cycle_exp,
            gamma=gamma,
        )
        assert fast_model == validated_model
        np.testing.assert_allclose(
            _predict_dual(prefix, forecast, fast_model, hyperparameters),
            predict_dual_clock_kernel_prior(
                prefix,
                forecast,
                validated_model,
                shrinkage=shrinkage,
                anchor_weight=anchor,
            ),
            rtol=0.0,
            atol=1e-12,
        )


def test_private_v3_firewall_scoring_capsule_and_tamper_rejection() -> None:
    references, prefixes, truth = _inputs()
    config = _v3_config()
    predictions, decisions, manifest = predict_private_dual_clock_prior_v3(
        references, prefixes, config
    )
    assert tuple(predictions.columns) == V3_PREDICTION_COLUMNS
    assert tuple(decisions.columns) == V3_DECISION_COLUMNS
    assert manifest["target_truth_argument_accepted"] is False
    assert manifest["future_elapsed_days_argument_accepted"] is False

    scores, summary = score_private_dual_clock_prior_v3(
        truth, predictions, decisions, manifest, config
    )
    assert tuple(scores.columns) == V3_SCORE_COLUMNS
    assert summary["private_only"] is True
    attacked_truth = truth.copy()
    attacked_truth.loc[
        attacked_truth["visit_index"] >= 3, "capacity_retention_pct"
    ] -= 15.0
    attacked_scores, _ = score_private_dual_clock_prior_v3(
        attacked_truth, predictions, decisions, manifest, config
    )
    assert not scores["trajectory_iae_pp"].equals(
        attacked_scores["trajectory_iae_pp"]
    )
    replay, replay_decisions, replay_manifest = (
        predict_private_dual_clock_prior_v3(references, prefixes, config)
    )
    pd.testing.assert_frame_equal(predictions, replay)
    pd.testing.assert_frame_equal(decisions, replay_decisions)
    assert manifest == replay_manifest

    capsule = train_private_dual_clock_prior_capsule(
        _trajectories(), config, training_identity={"fixture": True}
    )
    prefix = _trajectories().loc[
        _trajectories()["cell_id"] == "CELL_C1_a"
    ].iloc[:3]
    output, metadata = predict_private_dual_clock_prior_capsule(
        prefix, [300.0, 400.0, 500.0], capsule
    )
    assert len(output) == 3
    assert metadata["future_schedule_source"] == "constant_prefix_efc_per_day"
    assert capsule["raw_training_rows_in_capsule"] is False
    explicit_days = np.asarray([75.0, 100.0, 125.0])
    explicit_output, explicit_metadata = predict_private_dual_clock_prior_capsule(
        prefix,
        [300.0, 400.0, 500.0],
        capsule,
        forecast_elapsed_days=explicit_days,
    )
    np.testing.assert_allclose(
        explicit_output["forecast_elapsed_days"], explicit_days
    )
    assert explicit_metadata["future_schedule_source"] == (
        "explicit_forecast_elapsed_days"
    )
    with pytest.raises(
        PrivateDualClockPriorV3Error, match="elapsed days must be beyond"
    ):
        predict_private_dual_clock_prior_capsule(
            prefix,
            [300.0],
            capsule,
            forecast_elapsed_days=[40.0],
        )

    attacked_predictions = predictions.copy()
    attacked_predictions.loc[0, "predicted_capacity_retention_pct"] += 0.1
    with pytest.raises(PrivateDualClockPriorV3Error, match="changed after freeze"):
        score_private_dual_clock_prior_v3(
            truth,
            attacked_predictions,
            decisions,
            manifest,
            config,
        )

    attacked_capsule = deepcopy(capsule)
    attacked_capsule["landmark_models"]["3"]["hyperparameters"][
        "anchor_weight"
    ] = 0.123
    with pytest.raises(PrivateDualClockPriorV3Error, match="capsule content changed"):
        predict_private_dual_clock_prior_capsule(
            prefix, [300.0], attacked_capsule
        )


def test_private_v3_post_outcome_failure_audit_is_structured() -> None:
    references, prefixes, truth = _inputs()
    config = _v3_config()
    predictions, decisions, manifest = predict_private_dual_clock_prior_v3(
        references, prefixes, config
    )
    scores, _ = score_private_dual_clock_prior_v3(
        truth, predictions, decisions, manifest, config
    )
    cells, conditions, summary = audit_private_dual_clock_v3(
        truth, predictions, decisions, scores, config
    )
    assert tuple(cells.columns) == CELL_AUDIT_COLUMNS
    assert tuple(conditions.columns) == CONDITION_AUDIT_COLUMNS
    assert len(cells) == truth["cell_id"].nunique()
    assert len(conditions) == truth["outer_condition_id"].nunique()
    assert set(cells["risk_flags"])
    assert summary["private_only"] is True
    assert summary["evidence_role"] == "outcome_exposed_failure_analysis"


def test_private_dual_clock_uncertainty_audit_is_reference_calibrated() -> None:
    references, prefixes, truth = _inputs()
    config = _v3_config()
    predictions, decisions, manifest = predict_private_dual_clock_prior_v3(
        references, prefixes, config
    )
    cells, conditions, summary = audit_private_dual_clock_uncertainty(
        references, truth, predictions, decisions, manifest, config
    )
    assert tuple(cells.columns) == CELL_UNCERTAINTY_COLUMNS
    assert tuple(conditions.columns) == CONDITION_UNCERTAINTY_COLUMNS
    assert cells["pointwise_interval_coverage"].between(0.0, 1.0).all()
    assert (cells["mean_full_interval_width_pp"] > 0.0).all()
    assert summary["formal_interval_coverage_claim"] is False
    assert summary["public_release_permitted"] is False

    changed_truth = truth.copy()
    future = changed_truth["visit_index"] >= 3
    changed_truth.loc[future, "capacity_retention_pct"] -= 5.0
    changed, _, _ = audit_private_dual_clock_uncertainty(
        references,
        changed_truth,
        predictions,
        decisions,
        manifest,
        config,
    )
    outcome_free_columns = [
        "outer_condition_id",
        "cell_id",
        "landmark_visit_count",
        "mean_full_interval_width_pp",
        "prefix_linear_residual_rms_pp",
        "prefix_residual_abstention_threshold_pp",
        "issued",
        "abstention_reason",
    ]
    pd.testing.assert_frame_equal(
        cells.loc[:, outcome_free_columns],
        changed.loc[:, outcome_free_columns],
        check_exact=True,
    )
    assert not cells["pointwise_interval_coverage"].equals(
        changed["pointwise_interval_coverage"]
    )
