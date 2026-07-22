from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_synthetic as synthetic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v1.json"
)


@pytest.fixture()
def protocol() -> synthetic.ValidatedSyntheticProtocol:
    return synthetic.load_frozen_protocol_config(CONFIG_PATH)


def test_frozen_config_has_both_exact_commitments(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    raw = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == synthetic.FROZEN_CONFIG_BYTE_SHA256
    assert protocol.config_sha256 == synthetic.FROZEN_CONFIG_CANONICAL_SHA256


@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        ("single_power", {"a": 0.5, "b": 0.6}),
        (
            "dual_power",
            {"a1": 0.4, "b1": 0.5, "a2": 0.1, "b2": 1.1},
        ),
        (
            "saturating_plus_slow",
            {
                "a_sat": 2.0,
                "tau_sat_days": 300.0,
                "b_sat": 1.0,
                "a_slow": 0.2,
                "b_slow": 0.5,
            },
        ),
        (
            "early_activation_plus_power",
            {
                "a": 0.5,
                "b": 0.6,
                "activation_amplitude_pp": 0.7,
                "tau_rise_days": 10.0,
                "tau_decay_days": 200.0,
            },
        ),
        (
            "late_knee",
            {
                "a": 0.5,
                "b": 0.6,
                "k_pp_per_day": 0.002,
                "t_knee_days": 2500.0,
                "w_days": 90.0,
            },
        ),
    ],
)
def test_truth_families_are_finite_and_exactly_normalized_at_day_zero(
    family: str,
    parameters: dict[str, float],
) -> None:
    values = synthetic.evaluate_truth_retention(
        family,
        parameters,
        [0.0, 730.0, 9131.25],
    )
    assert values[0] == 100.0
    assert np.isfinite(values).all()


def test_truth_seed_matches_frozen_sha_derivation_and_uses_root(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    root = dict(protocol.partition_seed_roots)["test"]
    material = (
        f"{protocol.protocol_id}|{root}|test|single_power|7|truth_parameters"
    )
    expected = int(hashlib.sha256(material.encode()).hexdigest()[:16], 16) % (
        2**63 - 1
    )
    observed = synthetic.derive_truth_stream_seed(
        protocol.protocol_id,
        root,
        "test",
        "single_power",
        7,
        "truth_parameters",
    )
    changed_root = synthetic.derive_truth_stream_seed(
        protocol.protocol_id,
        root + 1,
        "test",
        "single_power",
        7,
        "truth_parameters",
    )
    assert observed == expected
    assert changed_root != observed


def test_seedless_sobol_starts_are_the_declared_unscrambled_points() -> None:
    lower = np.array([0.0, 10.0])
    upper = np.array([2.0, 20.0])
    first = synthetic._sobol_starts(lower, upper, count=16)
    second = synthetic._sobol_starts(lower, upper, count=16)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], lower)
    np.testing.assert_array_equal(first[1], np.array([1.0, 15.0]))


def test_matched_pair_is_one_shot_deterministic_opaque_and_shared_noise(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    first = synthetic.generate_matched_pair_packs(
        protocol, zero_based_pair_index=19
    )
    repeated = synthetic.generate_matched_pair_packs(
        protocol, zero_based_pair_index=19
    )
    pd.testing.assert_frame_equal(first.prefix_pack, repeated.prefix_pack)
    pd.testing.assert_frame_equal(
        first.matched_prefix_pairs, repeated.matched_prefix_pairs
    )
    record = first.matched_prefix_pairs.iloc[0]
    for value in (record.left_cluster_id, record.right_cluster_id):
        assert all(
            token not in value
            for token in ("single", "knee", "left", "right", "family", "side")
        )
    assert record.latent_prefix_rmse_pp <= 0.1
    assert record.latent_prefix_max_abs_difference_pp <= 0.1
    assert record.truth_separation_25y_pp >= 5.0
    ids = first.prefix_pack["cluster_id"].unique()
    left = (
        first.prefix_pack.loc[first.prefix_pack["cluster_id"].eq(ids[0])]
        .sort_values("prefix_day")["observed_retention_pct"]
        .to_numpy()
    )
    right = (
        first.prefix_pack.loc[first.prefix_pack["cluster_id"].eq(ids[1])]
        .sort_values("prefix_day")["observed_retention_pct"]
        .to_numpy()
    )
    np.testing.assert_array_equal(left, right)


def test_matched_opaque_assignment_is_deterministically_exchange_symmetric(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    root = dict(protocol.partition_seed_roots)[synthetic.MATCHED_PARTITION]
    left_uses_first_pool_member = []
    observed_ids: set[str] = set()
    for pair_index in range(200):
        left_id, right_id = synthetic.derive_matched_opaque_cluster_ids(
            protocol, pair_index
        )
        first_material = (
            f"{protocol.protocol_id}|{root}|opaque_cluster_pool|{2 * pair_index}"
        ).encode()
        first_pool_id = "c_" + hashlib.sha256(first_material).hexdigest()[:32]
        left_uses_first_pool_member.append(left_id == first_pool_id)
        observed_ids.update((left_id, right_id))
    assert sum(left_uses_first_pool_member) == 100
    assert len(observed_ids) == 400
    assert synthetic.derive_matched_opaque_cluster_ids(protocol, 19) == (
        synthetic.derive_matched_opaque_cluster_ids(protocol, 19)
    )


def _decision_inputs(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    layout = {
        "development": 2,
        "calibration": 3,
        "test": 3,
        "audit": 3,
        synthetic.MATCHED_PARTITION: 2,
    }
    global_index = 0
    for partition, count in layout.items():
        for index in range(count):
            cluster_id = f"opaque_{partition}_{index}"
            slope = 0.01 * (global_index + 1)
            cluster_prefix = pd.DataFrame(
                [
                    {
                        "protocol_id": protocol.protocol_id,
                        "partition": partition,
                        "cluster_id": cluster_id,
                        "prefix_day": day,
                        "observed_retention_pct": 100.0
                        - slope * np.sqrt(day / protocol.time_scale_days),
                    }
                    for day in protocol.prefix_days
                ],
                columns=synthetic.PREFIX_COLUMNS,
            )
            prefix_rows.extend(cluster_prefix.to_dict(orient="records"))
            prefix_hash = synthetic.canonical_prefix_content_sha256(cluster_prefix)
            spread = float(index + 1)
            for day in protocol.forecast_days:
                prediction_rows.append(
                    {
                        "protocol_id": protocol.protocol_id,
                        "partition": partition,
                        "cluster_id": cluster_id,
                        "forecast_day": day,
                        "candidate_point_forecast_pct": 90.0,
                        "persistence_forecast_pct": 91.0,
                        "sqrt_time_forecast_pct": 90.5,
                        "bounded_power_forecast_pct": 90.0,
                        "structure_envelope_lower_pct": 90.0 - spread,
                        "structure_envelope_upper_pct": 90.0 + spread,
                        "canonical_prefix_content_sha256": prefix_hash,
                    }
                )
            for model_id in synthetic.STRUCTURE_MEMBER_IDS:
                diagnostic_rows.append(
                    {
                        "protocol_id": protocol.protocol_id,
                        "partition": partition,
                        "cluster_id": cluster_id,
                        "model_id": model_id,
                        "variant_id": "only",
                        "fit_status": "succeeded",
                        "credible_variant": True,
                        "prefix_rmse_pp": 0.1,
                        "prefix_max_abs_residual_pp": 0.2,
                        "forecast_min_pct": 80.0,
                        "forecast_max_pct": 95.0,
                        "canonical_prefix_content_sha256": prefix_hash,
                    }
                )
            global_index += 1
    return (
        pd.DataFrame(prefix_rows, columns=synthetic.PREFIX_COLUMNS),
        pd.DataFrame(prediction_rows, columns=synthetic.PREDICTION_COLUMNS),
        pd.DataFrame(diagnostic_rows, columns=synthetic.MEMBER_DIAGNOSTIC_COLUMNS),
    )


def test_only_test_and_audit_are_issued_and_calibration_is_rank_only(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    _, predictions, diagnostics = _decision_inputs(protocol)
    result = synthetic.build_disagreement_decisions(
        predictions, diagnostics, protocol
    )
    decisions = result.decision_bundle
    issued = decisions.loc[decisions["primary_issued"]]
    assert set(issued["partition"]) == {"test", "audit"}
    assert len(issued.loc[issued["partition"].eq("test")]) == 3
    assert len(issued.loc[issued["partition"].eq("audit")]) == 3
    calibration = decisions.loc[decisions["partition"].eq("calibration")]
    assert calibration["primary_issued"].eq(False).all()
    assert sorted(calibration["primary_issuance_rank"].astype(int)) == [1, 2, 3]
    for partition in ("development", synthetic.MATCHED_PARTITION):
        subset = decisions.loc[decisions["partition"].eq(partition)]
        assert subset["primary_issuance_rank"].isna().all()
        assert subset["primary_issued"].eq(False).all()
    assert result.target_issue_count == 750


def test_main_scorer_rejects_truncated_batch_before_touching_truth(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, predictions, diagnostics = _decision_inputs(protocol)
    decisions = synthetic.build_disagreement_decisions(
        predictions, diagnostics, protocol
    ).decision_bundle
    coordinates = predictions.loc[
        :, list(synthetic.FORECAST_COORDINATE_COLUMNS)
    ].copy()
    artifacts = {
        "prefix": (prefix, synthetic.PREFIX_COLUMNS),
        "prediction": (predictions, synthetic.PREDICTION_COLUMNS),
        "decision": (decisions, synthetic.DECISION_COLUMNS),
        "coordinates": (coordinates, synthetic.FORECAST_COORDINATE_COLUMNS),
        "diagnostics": (diagnostics, synthetic.MEMBER_DIAGNOSTIC_COLUMNS),
    }
    paths = {
        name: Path(f"v014_{name}.csv") for name in artifacts
    }
    hashes: dict[str, str] = {}
    raw_by_path: dict[Path, bytes] = {}
    for name, (frame, columns) in artifacts.items():
        raw = synthetic.canonical_csv_bytes(frame, columns=columns)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        raw_by_path[paths[name]] = raw
    opened_paths: list[Path] = []

    def committed_read(path: Path) -> bytes:
        opened_paths.append(path)
        return raw_by_path[path]

    monkeypatch.setattr(Path, "read_bytes", committed_read)
    missing_truth = Path("v014_sealed_truth_must_not_be_opened.csv")
    with pytest.raises(synthetic.SyntheticProtocolError, match="exactly 500"):
        synthetic.score_frozen_predictions(
            paths["prefix"],
            paths["prediction"],
            paths["decision"],
            paths["coordinates"],
            paths["diagnostics"],
            missing_truth,
            protocol,
            expected_prefix_sha256=hashes["prefix"],
            expected_prediction_sha256=hashes["prediction"],
            expected_decision_sha256=hashes["decision"],
            expected_forecast_coordinates_sha256=hashes["coordinates"],
            expected_member_diagnostics_sha256=hashes["diagnostics"],
            expected_truth_sha256="0" * 64,
        )
    assert missing_truth not in opened_paths


def test_exact_bundle_schema_rejects_truth_alias(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    prefix = pd.DataFrame(
        [
            {
                "protocol_id": protocol.protocol_id,
                "partition": "test",
                "cluster_id": "opaque",
                "prefix_day": day,
                "observed_retention_pct": 100.0,
                "truth_family": "single_power",
            }
            for day in protocol.prefix_days
        ]
    )
    coordinates = pd.DataFrame(
        [
            {
                "protocol_id": protocol.protocol_id,
                "partition": "test",
                "cluster_id": "opaque",
                "forecast_day": day,
            }
            for day in protocol.forecast_days
        ],
        columns=synthetic.FORECAST_COORDINATE_COLUMNS,
    )
    with pytest.raises(synthetic.SyntheticProtocolError, match="unknown or missing"):
        synthetic.build_label_free_predictions(prefix, coordinates, protocol)


def test_matched_audit_marks_threshold_unavailable_when_calibration_is_short(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    mapping = synthetic.generate_all_matched_pair_packs(
        protocol
    ).matched_prefix_pairs
    rows: list[dict[str, object]] = []
    for rank in range(1, 250):
        rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "partition": "calibration",
                "cluster_id": f"calibration_{rank}",
                "canonical_prefix_content_sha256": hashlib.sha256(
                    f"calibration_{rank}".encode("ascii")
                ).hexdigest(),
                "credible_structure_family_count": 2,
                "fit_failure_count": 0,
                "best_prefix_rmse_pp": 0.1,
                "disagreement_score_pp": float(rank),
                "hard_eligible": True,
                "primary_issuance_rank": rank,
                "primary_issued": False,
                "abstention_reasons": "calibration_rank_only",
            }
        )
    matched_ids = sorted(
        set(mapping["left_cluster_id"].astype(str))
        | set(mapping["right_cluster_id"].astype(str))
    )
    for index, cluster_id in enumerate(matched_ids):
        rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "partition": synthetic.MATCHED_PARTITION,
                "cluster_id": cluster_id,
                "canonical_prefix_content_sha256": hashlib.sha256(
                    f"matched_{index}".encode("ascii")
                ).hexdigest(),
                "credible_structure_family_count": 2,
                "fit_failure_count": 0,
                "best_prefix_rmse_pp": 0.1,
                "disagreement_score_pp": 1000.0,
                "hard_eligible": True,
                "primary_issuance_rank": None,
                "primary_issued": False,
                "abstention_reasons": "matched_audit_only",
            }
        )
    decisions = pd.DataFrame(rows, columns=synthetic.DECISION_COLUMNS).sort_values(
        ["partition", "cluster_id"], kind="stable"
    ).reset_index(drop=True)
    decision_bytes = synthetic.canonical_csv_bytes(
        decisions, columns=synthetic.DECISION_COLUMNS
    )
    verified_score = synthetic.FrozenScoreResult(
        point_scores=pd.DataFrame(),
        trajectory_scores=pd.DataFrame(),
        prediction_sha256="0" * 64,
        decision_sha256=hashlib.sha256(decision_bytes).hexdigest(),
        prefix_sha256="1" * 64,
        forecast_coordinates_sha256="2" * 64,
        member_diagnostics_sha256="3" * 64,
        truth_sha256="4" * 64,
        verified_decision_bytes=decision_bytes,
        _verification_token=synthetic._VERIFIED_SCORE_TOKEN,
    )

    result = synthetic.evaluate_matched_pair_rejection(
        verified_score, mapping, protocol
    )

    assert result.qualified_pair_count == 200
    assert result.calibration_disagreement_threshold_pp is None
    assert result.both_rejected_pair_count == 0
    assert result.both_rejected_fraction == 0.0
    assert result.pair_scores["both_members_rejected"].eq(False).all()
