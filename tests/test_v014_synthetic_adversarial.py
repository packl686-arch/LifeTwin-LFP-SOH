from __future__ import annotations

from copy import deepcopy
import json
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
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return synthetic.validate_protocol_config(config)


def _predictor_packs(
    protocol: synthetic.ValidatedSyntheticProtocol,
    *,
    partition: str,
    cluster_slopes: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows: list[dict[str, object]] = []
    coordinate_rows: list[dict[str, object]] = []
    for cluster_id, slope in cluster_slopes.items():
        for day in protocol.prefix_days:
            prefix_rows.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "prefix_day": day,
                    "observed_retention_pct": 100.0
                    - slope * np.sqrt(day / protocol.time_scale_days),
                }
            )
        for day in protocol.forecast_days:
            coordinate_rows.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "forecast_day": day,
                }
            )
    return (
        pd.DataFrame(prefix_rows, columns=synthetic.PREFIX_COLUMNS),
        pd.DataFrame(
            coordinate_rows,
            columns=synthetic.FORECAST_COORDINATE_COLUMNS,
        ),
    )


def _stub_structure_variants(
    prefix_days: object,
    observed_retention_pct: object,
    forecast_days: object,
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> tuple[synthetic.CandidateVariant, ...]:
    del prefix_days, protocol
    observed = np.asarray(observed_retention_pct, dtype=float)
    forecast = np.asarray(forecast_days, dtype=float)
    eligible = float(observed[-1]) > 99.8
    variants: list[synthetic.CandidateVariant] = []
    for index, model_id in enumerate(synthetic.STRUCTURE_MEMBER_IDS):
        is_bounded = model_id == "target_prefix_bounded_power_law"
        rmse = 0.01 + 0.01 * index if eligible or is_bounded else 1.0
        values = np.full(len(forecast), 99.0 - 0.2 * index)
        variants.append(
            synthetic.CandidateVariant(
                model_id=model_id,
                variant_id=f"stub_{index}",
                parameters=(),
                prefix_rmse_pp=rmse,
                prefix_max_absolute_residual_pp=0.1,
                forecast_retention_pct=tuple(float(value) for value in values),
                fit_succeeded=True,
            )
        )
    return tuple(variants)


def _normalized_prediction_values(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.drop(columns=["cluster_id"])
        .sort_values("forecast_day", kind="stable")
        .reset_index(drop=True)
    )


def _single_power_truth_pack(
    protocol: synthetic.ValidatedSyntheticProtocol,
    *,
    partition: str,
    cluster_id: str,
) -> pd.DataFrame:
    parameters = {"a": 0.5, "b": 0.5}
    latent = synthetic.evaluate_truth_retention(
        "single_power",
        parameters,
        protocol.forecast_days,
        time_scale_days=protocol.time_scale_days,
    )
    parameter_json = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
    )
    return pd.DataFrame(
        {
            "protocol_id": protocol.protocol_id,
            "partition": partition,
            "cluster_id": cluster_id,
            "truth_family": "single_power",
            "truth_parameters_json": parameter_json,
            "forecast_day": protocol.forecast_days,
            "latent_retention_pct": latent,
            "noisy_retention_pct": latent,
        },
        columns=synthetic.TRUTH_PACK_COLUMNS,
    )


def test_frozen_protocol_rejects_even_semantically_plausible_mutation() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed = deepcopy(config)
    changed["truth_generation"]["truth_families"][0]["parameters"]["a"][
        "maximum"
    ] = 1.2001

    with pytest.raises(synthetic.SyntheticProtocolError, match="frozen v1"):
        synthetic.validate_protocol_config(changed)


def test_truth_stream_seed_uses_partition_root(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    roots = dict(protocol.partition_seed_roots)
    first = synthetic.derive_truth_stream_seed(
        protocol.protocol_id,
        roots["test"],
        "test",
        "single_power",
        0,
        "truth_parameters",
    )
    repeated = synthetic.derive_truth_stream_seed(
        protocol.protocol_id,
        roots["test"],
        "test",
        "single_power",
        0,
        "truth_parameters",
    )
    changed_root = synthetic.derive_truth_stream_seed(
        protocol.protocol_id,
        roots["test"] + 1,
        "test",
        "single_power",
        0,
        "truth_parameters",
    )

    assert first == repeated
    assert first != changed_root


def test_predictor_rejects_oracle_columns_and_is_id_order_invariant(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        synthetic,
        "fit_structure_family_variants",
        _stub_structure_variants,
    )
    prefix, coordinates = _predictor_packs(
        protocol,
        partition="test",
        cluster_slopes={"opaque_a": 0.05},
    )

    poisoned = prefix.assign(truth_family="late_knee")
    with pytest.raises(
        synthetic.SyntheticProtocolError,
        match="unknown or missing columns",
    ):
        synthetic.build_label_free_predictions(poisoned, coordinates, protocol)

    original = synthetic.build_label_free_predictions(
        prefix.sample(frac=1.0, random_state=11),
        coordinates.sample(frac=1.0, random_state=12),
        protocol,
    )
    relabeled_prefix = prefix.copy()
    relabeled_coordinates = coordinates.copy()
    relabeled_prefix["cluster_id"] = "unrelated_label"
    relabeled_coordinates["cluster_id"] = "unrelated_label"
    relabeled = synthetic.build_label_free_predictions(
        relabeled_prefix,
        relabeled_coordinates,
        protocol,
    )

    pd.testing.assert_frame_equal(
        _normalized_prediction_values(original.prediction_bundle),
        _normalized_prediction_values(relabeled.prediction_bundle),
        check_exact=True,
    )


def test_hard_ineligibility_never_fills_ranked_quota(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        synthetic,
        "fit_structure_family_variants",
        _stub_structure_variants,
    )
    prefix, coordinates = _predictor_packs(
        protocol,
        partition="test",
        cluster_slopes={
            "only_eligible": 0.05,
            "ineligible_1": 0.30,
            "ineligible_2": 0.40,
            "ineligible_3": 0.50,
        },
    )
    predictions = synthetic.build_label_free_predictions(
        prefix, coordinates, protocol
    )
    decisions = synthetic.build_disagreement_decisions(
        predictions.prediction_bundle,
        predictions.member_diagnostics,
        protocol,
    )

    assert decisions.target_issue_count == 500
    assert decisions.actual_issue_count == 1
    issued = decisions.decision_bundle.loc[
        decisions.decision_bundle["primary_issued"]
    ]
    assert issued["cluster_id"].tolist() == ["only_eligible"]
    assert decisions.decision_bundle.loc[
        ~decisions.decision_bundle["hard_eligible"], "primary_issued"
    ].eq(False).all()


def test_identical_matched_prefixes_are_scored_without_primary_ranking(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        synthetic,
        "fit_structure_family_variants",
        _stub_structure_variants,
    )
    prefix, coordinates = _predictor_packs(
        protocol,
        partition=synthetic.MATCHED_PARTITION,
        cluster_slopes={"pair_left": 0.05, "pair_right": 0.05},
    )
    predictions = synthetic.build_label_free_predictions(
        prefix, coordinates, protocol
    )
    decisions = synthetic.build_disagreement_decisions(
        predictions.prediction_bundle,
        predictions.member_diagnostics,
        protocol,
    )

    assert decisions.decision_bundle[
        "canonical_prefix_content_sha256"
    ].nunique() == 1
    assert decisions.decision_bundle["primary_issuance_rank"].isna().all()
    assert decisions.decision_bundle["primary_issued"].eq(False).all()
    assert decisions.actual_issue_count == 0


def test_matched_counterexamples_are_deterministic_dense_and_complete(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    first = synthetic.generate_matched_pair_packs(
        protocol,
        zero_based_pair_index=0,
    )
    repeated = synthetic.generate_matched_pair_packs(
        protocol,
        zero_based_pair_index=0,
    )
    pd.testing.assert_frame_equal(first.prefix_pack, repeated.prefix_pack)
    pd.testing.assert_frame_equal(
        first.matched_prefix_pairs,
        repeated.matched_prefix_pairs,
    )

    left_spec, right_spec = first.truth_specs
    dense_days = np.arange(0.0, protocol.primary_prefix_end_day + 1.0)
    left_latent = synthetic.evaluate_truth_retention(
        left_spec.family_id,
        left_spec.parameter_map(),
        dense_days,
        time_scale_days=protocol.time_scale_days,
    )
    right_latent = synthetic.evaluate_truth_retention(
        right_spec.family_id,
        right_spec.parameter_map(),
        dense_days,
        time_scale_days=protocol.time_scale_days,
    )
    difference = right_latent - left_latent
    assert np.sqrt(np.mean(np.square(difference))) <= 0.1
    assert np.max(np.abs(difference)) <= 0.1

    observed_by_id = {
        str(cluster_id): group.sort_values("prefix_day")[
            "observed_retention_pct"
        ].to_numpy(dtype=float)
        for cluster_id, group in first.prefix_pack.groupby("cluster_id")
    }
    declared_days = np.asarray(protocol.prefix_days, dtype=float)
    left_declared = synthetic.evaluate_truth_retention(
        left_spec.family_id,
        left_spec.parameter_map(),
        declared_days,
        time_scale_days=protocol.time_scale_days,
    )
    right_declared = synthetic.evaluate_truth_retention(
        right_spec.family_id,
        right_spec.parameter_map(),
        declared_days,
        time_scale_days=protocol.time_scale_days,
    )
    left_noise = observed_by_id[left_spec.cluster_id] - left_declared
    right_noise = observed_by_id[right_spec.cluster_id] - right_declared
    np.testing.assert_allclose(left_noise, right_noise, rtol=0.0, atol=1e-12)

    all_pairs = synthetic.generate_all_matched_pair_packs(protocol)
    mapping = all_pairs.matched_prefix_pairs
    assert len(mapping) == 200
    assert all_pairs.prefix_pack["cluster_id"].nunique() == 400
    assert mapping["pair_id"].nunique() == 200
    assert set(mapping["left_cluster_id"]).isdisjoint(
        set(mapping["right_cluster_id"])
    )
    assert mapping["latent_prefix_rmse_pp"].le(0.1).all()
    assert mapping["latent_prefix_max_abs_difference_pp"].le(0.1).all()
    assert mapping["truth_separation_25y_pp"].ge(5.0).all()
    assert mapping["max_forecast_truth_separation_pp"].ge(5.0).all()


def test_scorer_rejects_rehashed_decision_attack_before_opening_truth(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        synthetic,
        "fit_structure_family_variants",
        _stub_structure_variants,
    )
    prefix, coordinates = _predictor_packs(
        protocol,
        partition="test",
        cluster_slopes={"target": 0.05},
    )
    predictions = synthetic.build_label_free_predictions(
        prefix, coordinates, protocol
    )
    decisions = synthetic.build_disagreement_decisions(
        predictions.prediction_bundle,
        predictions.member_diagnostics,
        protocol,
    )
    attacked = decisions.decision_bundle.copy()
    attacked["primary_issued"] = False
    attacked["abstention_reasons"] = "attacker_reduced_coverage"
    attacked_hash = synthetic.canonical_csv_sha256(
        attacked,
        columns=synthetic.DECISION_COLUMNS,
    )
    prefix_path = Path("prefix_pack.csv")
    prediction_path = Path("prediction_bundle.csv")
    decision_path = Path("decision_bundle.csv")
    coordinates_path = Path("forecast_coordinates.csv")
    diagnostics_path = Path("member_fit_diagnostics.csv")
    sealed_truth_path = Path("sealed_truth_must_not_be_opened.csv")
    opened_paths: list[Path] = []
    artifacts = {
        prefix_path: synthetic.canonical_csv_bytes(
            prefix,
            columns=synthetic.PREFIX_COLUMNS,
        ),
        prediction_path: synthetic.canonical_csv_bytes(
            predictions.prediction_bundle,
            columns=synthetic.PREDICTION_COLUMNS,
        ),
        decision_path: synthetic.canonical_csv_bytes(
            attacked,
            columns=synthetic.DECISION_COLUMNS,
        ),
        coordinates_path: synthetic.canonical_csv_bytes(
            coordinates,
            columns=synthetic.FORECAST_COORDINATE_COLUMNS,
        ),
        diagnostics_path: synthetic.canonical_csv_bytes(
            predictions.member_diagnostics,
            columns=synthetic.MEMBER_DIAGNOSTIC_COLUMNS,
        ),
    }

    def committed_read(path: Path) -> bytes:
        opened_paths.append(path)
        return artifacts[path]

    monkeypatch.setattr(Path, "read_bytes", committed_read)

    with pytest.raises(
        synthetic.SyntheticProtocolError,
        match="do not reproduce",
    ):
        synthetic.score_frozen_predictions(
            prefix_path,
            prediction_path,
            decision_path,
            coordinates_path,
            diagnostics_path,
            sealed_truth_path,
            protocol,
            expected_prefix_sha256=synthetic.canonical_csv_sha256(
                prefix,
                columns=synthetic.PREFIX_COLUMNS,
            ),
            expected_prediction_sha256=predictions.prediction_sha256,
            expected_decision_sha256=attacked_hash,
            expected_forecast_coordinates_sha256=synthetic.canonical_csv_sha256(
                coordinates,
                columns=synthetic.FORECAST_COORDINATE_COLUMNS,
            ),
            expected_member_diagnostics_sha256=(
                predictions.member_diagnostics_sha256
            ),
            expected_truth_sha256="0" * 64,
        )
    assert sealed_truth_path not in opened_paths


def test_incomplete_scoring_batch_is_rejected_before_truth_is_opened(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        synthetic,
        "fit_structure_family_variants",
        _stub_structure_variants,
    )
    prefix, coordinates = _predictor_packs(
        protocol,
        partition="test",
        cluster_slopes={"target": 0.05},
    )
    predictions = synthetic.build_label_free_predictions(
        prefix, coordinates, protocol
    )
    decisions = synthetic.build_disagreement_decisions(
        predictions.prediction_bundle,
        predictions.member_diagnostics,
        protocol,
    )
    prefix_path = Path("prefix_pack.csv")
    prediction_path = Path("prediction_bundle.csv")
    decision_path = Path("decision_bundle.csv")
    coordinates_path = Path("forecast_coordinates.csv")
    diagnostics_path = Path("member_fit_diagnostics.csv")
    sealed_truth_path = Path("sealed_truth.csv")
    opened_paths: list[Path] = []
    artifacts = {
        prefix_path: synthetic.canonical_csv_bytes(
            prefix,
            columns=synthetic.PREFIX_COLUMNS,
        ),
        prediction_path: synthetic.canonical_csv_bytes(
            predictions.prediction_bundle,
            columns=synthetic.PREDICTION_COLUMNS,
        ),
        decision_path: synthetic.canonical_csv_bytes(
            decisions.decision_bundle,
            columns=synthetic.DECISION_COLUMNS,
        ),
        coordinates_path: synthetic.canonical_csv_bytes(
            coordinates,
            columns=synthetic.FORECAST_COORDINATE_COLUMNS,
        ),
        diagnostics_path: synthetic.canonical_csv_bytes(
            predictions.member_diagnostics,
            columns=synthetic.MEMBER_DIAGNOSTIC_COLUMNS,
        ),
    }

    def committed_read(path: Path) -> bytes:
        opened_paths.append(path)
        return artifacts[path]

    monkeypatch.setattr(Path, "read_bytes", committed_read)
    common = {
        "expected_prefix_sha256": synthetic.canonical_csv_sha256(
            prefix,
            columns=synthetic.PREFIX_COLUMNS,
        ),
        "expected_prediction_sha256": predictions.prediction_sha256,
        "expected_decision_sha256": decisions.decision_sha256,
        "expected_forecast_coordinates_sha256": synthetic.canonical_csv_sha256(
            coordinates,
            columns=synthetic.FORECAST_COORDINATE_COLUMNS,
        ),
        "expected_member_diagnostics_sha256": (
            predictions.member_diagnostics_sha256
        ),
    }

    with pytest.raises(
        synthetic.SyntheticProtocolError,
        match="every complete protocol partition",
    ):
        synthetic.score_frozen_predictions(
            prefix_path,
            prediction_path,
            decision_path,
            coordinates_path,
            diagnostics_path,
            sealed_truth_path,
            protocol,
            expected_truth_sha256="0" * 64,
            **common,
        )
    assert sealed_truth_path not in opened_paths


def test_prefix_raw_byte_tamper_is_rejected_before_other_artifacts_or_truth(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, _ = _predictor_packs(
        protocol,
        partition="test",
        cluster_slopes={"target": 0.05},
    )
    attacked = prefix.copy()
    attacked.loc[attacked["prefix_day"].eq(730.0), "observed_retention_pct"] -= 1.0
    prefix_path = Path("attacked_prefix_pack.csv")
    truth_path = Path("sealed_truth_must_not_be_opened.csv")
    opened_paths: list[Path] = []

    def committed_read(path: Path) -> bytes:
        opened_paths.append(path)
        if path != prefix_path:
            raise AssertionError("Scorer crossed the prefix commitment firewall")
        return synthetic.canonical_csv_bytes(
            attacked, columns=synthetic.PREFIX_COLUMNS
        )

    monkeypatch.setattr(Path, "read_bytes", committed_read)
    with pytest.raises(synthetic.SyntheticProtocolError, match="prefix_pack bytes"):
        synthetic.score_frozen_predictions(
            prefix_path,
            Path("prediction.csv"),
            Path("decision.csv"),
            Path("coordinates.csv"),
            Path("diagnostics.csv"),
            truth_path,
            protocol,
            expected_prefix_sha256=synthetic.canonical_csv_sha256(
                prefix, columns=synthetic.PREFIX_COLUMNS
            ),
            expected_prediction_sha256="0" * 64,
            expected_decision_sha256="0" * 64,
            expected_forecast_coordinates_sha256="0" * 64,
            expected_member_diagnostics_sha256="0" * 64,
            expected_truth_sha256="0" * 64,
        )
    assert opened_paths == [prefix_path]
    assert truth_path not in opened_paths


@pytest.mark.parametrize("attack", ["missing_coordinate", "changed_content"])
def test_prefix_grid_and_recomputed_content_hash_are_enforced(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    monkeypatch.setattr(
        synthetic,
        "fit_structure_family_variants",
        _stub_structure_variants,
    )
    prefix, coordinates = _predictor_packs(
        protocol,
        partition="test",
        cluster_slopes={"target": 0.05},
    )
    predictions = synthetic.build_label_free_predictions(
        prefix, coordinates, protocol
    )
    decisions = synthetic.build_disagreement_decisions(
        predictions.prediction_bundle,
        predictions.member_diagnostics,
        protocol,
    )
    attacked = prefix.copy()
    if attack == "missing_coordinate":
        attacked = attacked.loc[~attacked["prefix_day"].eq(90.0)].reset_index(
            drop=True
        )
        expected_message = "prefix grid is incomplete"
    else:
        attacked.loc[
            attacked["prefix_day"].eq(730.0), "observed_retention_pct"
        ] -= 0.25
        expected_message = "Committed prefix hash differs"

    with pytest.raises(synthetic.SyntheticProtocolError, match=expected_message):
        synthetic._validate_prediction_and_decision_freeze(
            attacked,
            predictions.prediction_bundle,
            decisions.decision_bundle,
            coordinates,
            predictions.member_diagnostics,
            protocol,
            expected_prefix_sha256=synthetic.canonical_csv_sha256(
                attacked, columns=synthetic.PREFIX_COLUMNS
            ),
            expected_prediction_sha256=predictions.prediction_sha256,
            expected_decision_sha256=decisions.decision_sha256,
            expected_forecast_coordinates_sha256=synthetic.canonical_csv_sha256(
                coordinates,
                columns=synthetic.FORECAST_COORDINATE_COLUMNS,
            ),
            expected_member_diagnostics_sha256=(
                predictions.member_diagnostics_sha256
            ),
        )


def test_matched_audit_rejects_an_unverified_decision_dataframe(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    arbitrary_decisions = pd.DataFrame(columns=synthetic.DECISION_COLUMNS)
    sealed_mapping = pd.DataFrame(columns=synthetic.MATCHED_PAIR_COLUMNS)
    with pytest.raises(
        synthetic.SyntheticProtocolError,
        match="decision capability from the frozen scorer",
    ):
        synthetic.evaluate_matched_pair_rejection(
            arbitrary_decisions,  # type: ignore[arg-type]
            sealed_mapping,
            protocol,
        )


def test_truth_byte_commitment_is_checked_after_label_free_firewall(
    protocol: synthetic.ValidatedSyntheticProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate the truth-byte check from the separate full-batch contract."""
    monkeypatch.setattr(
        synthetic,
        "fit_structure_family_variants",
        _stub_structure_variants,
    )
    prefix, coordinates = _predictor_packs(
        protocol,
        partition="test",
        cluster_slopes={"target": 0.05},
    )
    predictions = synthetic.build_label_free_predictions(
        prefix, coordinates, protocol
    )
    decisions = synthetic.build_disagreement_decisions(
        predictions.prediction_bundle,
        predictions.member_diagnostics,
        protocol,
    )
    truth = _single_power_truth_pack(
        protocol,
        partition="test",
        cluster_id="target",
    )
    prefix_path = Path("prefix_pack.csv")
    prediction_path = Path("prediction_bundle.csv")
    decision_path = Path("decision_bundle.csv")
    coordinates_path = Path("forecast_coordinates.csv")
    diagnostics_path = Path("member_fit_diagnostics.csv")
    truth_path = Path("sealed_truth.csv")
    artifacts = {
        prefix_path: synthetic.canonical_csv_bytes(
            prefix,
            columns=synthetic.PREFIX_COLUMNS,
        ),
        prediction_path: synthetic.canonical_csv_bytes(
            predictions.prediction_bundle,
            columns=synthetic.PREDICTION_COLUMNS,
        ),
        decision_path: synthetic.canonical_csv_bytes(
            decisions.decision_bundle,
            columns=synthetic.DECISION_COLUMNS,
        ),
        coordinates_path: synthetic.canonical_csv_bytes(
            coordinates,
            columns=synthetic.FORECAST_COORDINATE_COLUMNS,
        ),
        diagnostics_path: synthetic.canonical_csv_bytes(
            predictions.member_diagnostics,
            columns=synthetic.MEMBER_DIAGNOSTIC_COLUMNS,
        ),
        truth_path: synthetic.canonical_csv_bytes(
            truth,
            columns=synthetic.TRUTH_PACK_COLUMNS,
        ),
    }
    observed_hashes = tuple(
        synthetic.canonical_csv_sha256(frame, columns=columns)
        for frame, columns in (
            (prefix, synthetic.PREFIX_COLUMNS),
            (predictions.prediction_bundle, synthetic.PREDICTION_COLUMNS),
            (decisions.decision_bundle, synthetic.DECISION_COLUMNS),
            (coordinates, synthetic.FORECAST_COORDINATE_COLUMNS),
            (predictions.member_diagnostics, synthetic.MEMBER_DIAGNOSTIC_COLUMNS),
        )
    )
    monkeypatch.setattr(
        synthetic,
        "_validate_prediction_and_decision_freeze",
        lambda *args, **kwargs: observed_hashes,
    )
    monkeypatch.setattr(Path, "read_bytes", lambda path: artifacts[path])
    with pytest.raises(
        synthetic.SyntheticProtocolError,
        match="truth_pack bytes",
    ):
        synthetic.score_frozen_predictions(
            prefix_path,
            prediction_path,
            decision_path,
            coordinates_path,
            diagnostics_path,
            truth_path,
            protocol,
            expected_prefix_sha256=synthetic.canonical_csv_sha256(
                prefix,
                columns=synthetic.PREFIX_COLUMNS,
            ),
            expected_prediction_sha256=predictions.prediction_sha256,
            expected_decision_sha256=decisions.decision_sha256,
            expected_forecast_coordinates_sha256=(
                synthetic.canonical_csv_sha256(
                    coordinates,
                    columns=synthetic.FORECAST_COORDINATE_COLUMNS,
                )
            ),
            expected_member_diagnostics_sha256=(
                predictions.member_diagnostics_sha256
            ),
            expected_truth_sha256="0" * 64,
        )
