from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_io as v015_io


@pytest.fixture(scope="module")
def contract() -> v015_io.FrozenArtifactContract:
    return v015_io.load_artifact_contract()


def _prefix_frame(
    contract: v015_io.FrozenArtifactContract, *, reverse: bool = False
) -> pd.DataFrame:
    rows = [
        {
            "protocol_id": contract.protocol_id,
            "partition": "test",
            "cluster_id": "fixture-c1",
            "prefix_day": 0.0,
            "observed_retention_pct": 100.0,
        },
        {
            "protocol_id": contract.protocol_id,
            "partition": "test",
            "cluster_id": "fixture-c1",
            "prefix_day": 7.0,
            "observed_retention_pct": 99.875,
        },
    ]
    if reverse:
        rows.reverse()
    return pd.DataFrame(rows, columns=contract.csv_schema("prefix_pack.csv").columns)


def _fixture_frame(
    filename: str, contract: v015_io.FrozenArtifactContract
) -> pd.DataFrame:
    schema = contract.csv_schema(filename)
    base: dict[str, object] = {
        "protocol_id": contract.protocol_id,
        "partition": "test",
        "cluster_id": "fixture-c1",
        "prefix_day": 0.0,
        "observed_retention_pct": 100.0,
        "forecast_day": 1095.75,
        "past_mean_temperature_c": 25.0,
        "past_mean_soc_fraction": 0.5,
        "past_mean_dod_fraction": 0.6,
        "past_efc_per_year": 250.0,
        "planned_mean_temperature_c": 30.0,
        "planned_mean_soc_fraction": 0.6,
        "planned_mean_dod_fraction": 0.7,
        "planned_efc_per_year": 300.0,
        "model_id": "fixture-model",
        "variant_id": "fixture-variant",
        "parameters_json": "{}",
        "fit_status": "success",
        "credible_variant": True,
        "prefix_rmse_pp": 0.1,
        "prefix_max_abs_residual_pp": 0.2,
        "parameter_boundary_hit_fraction": 0.0,
        "canonical_prefix_content_sha256": "1" * 64,
        "raw_forecast_retention_pct": 97.0,
        "center_forecast_pct": 97.0,
        "sqrt_time_forecast_pct": 97.2,
        "bounded_power_forecast_pct": 96.9,
        "base_interval_lower_pct": 80.0,
        "base_interval_upper_pct": 100.0,
        "calibrated_interval_lower_pct": 75.0,
        "calibrated_interval_upper_pct": 105.0,
        "score_id": "prefix_only",
        "raw_risk_score": -0.25,
        "calibrated_catastrophic_probability": 0.2,
        "all_features_finite": True,
        "successful_structure_family_count": 5,
        "fit_failure_count": 2,
        "effective_unique_shape_count": 4,
        "canonical_predictor_content_sha256": "2" * 64,
        "arm": "prefix_only",
        "hard_eligible": True,
        "issuance_rank": 1,
        "issued": True,
        "abstention_reasons": "",
    }
    for index in range(1, 9):
        base[f"placebo_control_{index}"] = index / 10
    return pd.DataFrame(
        [{column: base[column] for column in schema.columns}],
        columns=schema.columns,
    )


def _truth_fixture(
    filename: str, contract: v015_io.FrozenArtifactContract
) -> pd.DataFrame:
    schema = contract.csv_schema(filename)
    partition = schema.expected_partition
    if partition in {"intrinsic_matched_pairs", "stress_plan_matched_pairs"}:
        cluster_ids = [f"left-{partition}", f"right-{partition}"]
    else:
        cluster_ids = [f"fixture-{partition}"]
    rows = [
        {
            "protocol_id": contract.protocol_id,
            "partition": partition,
            "cluster_id": cluster_id,
            "truth_family": "single_power",
            "truth_parameters_json": '{"a":0.5,"b":0.5}',
            "gamma": 0.1,
            "forecast_day": 1095.75,
            "latent_retention_pct": 98.0,
            "noisy_retention_pct": 97.9,
        }
        for cluster_id in cluster_ids
    ]
    return pd.DataFrame(rows, columns=schema.columns)


def _sealed_fixture(
    filename: str, contract: v015_io.FrozenArtifactContract
) -> pd.DataFrame:
    if filename in contract.truth_filenames:
        return _truth_fixture(filename, contract)
    schema = contract.csv_schema(filename)
    partition = schema.expected_partition
    row = {
        "protocol_id": contract.protocol_id,
        "pair_partition": partition,
        "pair_id": f"pair-{partition}",
        "left_cluster_id": f"left-{partition}",
        "right_cluster_id": f"right-{partition}",
        "construction_family": "single_power",
        "left_side_code": "left",
        "right_side_code": "right",
        "latent_prefix_rmse_pp": 0.0,
        "latent_prefix_max_abs_difference_pp": 0.0,
        "truth_separation_25y_pp": 5.0,
    }
    return pd.DataFrame([row], columns=schema.columns)


def _model_state(
    contract: v015_io.FrozenArtifactContract,
) -> dict[str, object]:
    return {
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_byte_sha256,
        "center_state": {"beta": 0.5},
        "risk_states": {},
        "calibration_state": {},
        "comparator_states": {},
        "feature_orders": {},
        "input_byte_hashes": {},
        "software_versions": {},
        "created_utc": "2026-07-23T00:00:00Z",
    }


def _prediction_frames(
    contract: v015_io.FrozenArtifactContract,
) -> dict[str, pd.DataFrame]:
    filenames = [
        "prefix_pack.csv",
        "forecast_coordinates.csv",
        "operating_pack.csv",
        "member_fit_diagnostics.csv",
        "member_forecast_bundle.csv",
        "prediction_bundle.csv",
        "risk_bundle.csv",
        "decision_bundle.csv",
    ]
    return {filename: _fixture_frame(filename, contract) for filename in filenames}


def _sealed_frames(
    contract: v015_io.FrozenArtifactContract,
) -> dict[str, pd.DataFrame]:
    return {
        filename: _sealed_fixture(filename, contract)
        for filename in contract.sealed_filenames
    }


def _exposure_event(
    contract: v015_io.FrozenArtifactContract,
    *,
    created_utc: str,
    phase: str,
    opened_truth_files: list[str] | None = None,
) -> dict[str, object]:
    return {
        "attempt_id": "fixture-attempt",
        "created_utc": created_utc,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "config_byte_sha256": contract.config_byte_sha256,
        "phase": phase,
        "truth_commitments_byte_sha256": None,
        "prediction_commitment_byte_sha256": None,
        "opened_truth_files": opened_truth_files or [],
        "exit_status": "running",
        "message": "fixture only",
    }


def test_contract_expands_exact_frozen_schemas(
    contract: v015_io.FrozenArtifactContract,
) -> None:
    assert contract.protocol_id == "synthetic_long_horizon_identifiability_v2"
    assert len(contract.truth_filenames) == 7
    assert contract.csv_schema("prefix_pack.csv").required_rows == 71_400
    assert (
        contract.csv_schema("intrinsic_matched_truth.csv").expected_partition
        == "intrinsic_matched_pairs"
    )
    assert (
        contract.csv_schema("stress_plan_matched_truth.csv").expected_partition
        == "stress_plan_matched_pairs"
    )
    assert contract.csv_schema("risk_bundle.csv").required_score_ids == (
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
    assert (
        contract.config_byte_sha256
        == hashlib.sha256(contract.config_path.read_bytes()).hexdigest()
    )


def test_contract_rejects_any_config_byte_drift(
    tmp_path: Path, contract: v015_io.FrozenArtifactContract
) -> None:
    changed = json.loads(contract.config_path.read_text(encoding="utf-8"))
    changed["status"] = "tampered"
    path = tmp_path / "tampered-v2.json"
    path.write_text(
        json.dumps(changed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(v015_io.V015ArtifactError, match="byte hash"):
        v015_io.load_artifact_contract(path)


def test_canonical_csv_is_reorder_invariant_but_disk_order_is_strict(
    tmp_path: Path, contract: v015_io.FrozenArtifactContract
) -> None:
    schema = contract.csv_schema("prefix_pack.csv")
    forward = _prefix_frame(contract)
    reverse = _prefix_frame(contract, reverse=True)
    assert v015_io.canonical_csv_bytes(
        forward, schema, contract, formal=False
    ) == v015_io.canonical_csv_bytes(reverse, schema, contract, formal=False)

    path = tmp_path / "prefix_pack.csv"
    v015_io.write_canonical_csv(path, reverse, contract, formal=False)
    loaded = v015_io.read_canonical_csv(path, contract, formal=False)
    assert loaded["prefix_day"].tolist() == [0.0, 7.0]

    noncanonical = reverse.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    bad_path = tmp_path / "bad" / "prefix_pack.csv"
    bad_path.parent.mkdir()
    bad_path.write_bytes(noncanonical)
    with pytest.raises(v015_io.V015ArtifactError, match="canonical CSV"):
        v015_io.read_canonical_csv(bad_path, contract, formal=False)


def test_csv_rejects_unknown_columns_duplicate_keys_and_wrong_protocol(
    contract: v015_io.FrozenArtifactContract,
) -> None:
    schema = contract.csv_schema("prefix_pack.csv")
    unknown = _prefix_frame(contract).assign(oracle_truth=91.0)
    with pytest.raises(v015_io.V015ArtifactError, match="columns differ"):
        v015_io.canonical_csv_bytes(unknown, schema, contract, formal=False)

    duplicate = pd.concat(
        [_prefix_frame(contract).iloc[[0]], _prefix_frame(contract).iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(v015_io.V015ArtifactError, match="duplicate key"):
        v015_io.canonical_csv_bytes(duplicate, schema, contract, formal=False)

    wrong = _prefix_frame(contract)
    wrong.loc[:, "protocol_id"] = "unfrozen"
    with pytest.raises(v015_io.V015ArtifactError, match="protocol_id"):
        v015_io.canonical_csv_bytes(wrong, schema, contract, formal=False)


def test_csv_rejects_na_blank_keys_and_non_boolean_flags(
    contract: v015_io.FrozenArtifactContract,
) -> None:
    prefix_schema = contract.csv_schema("prefix_pack.csv")
    missing_id = _prefix_frame(contract)
    missing_id.loc[0, "cluster_id"] = None
    with pytest.raises(v015_io.V015ArtifactError, match="key column.*contains NA"):
        v015_io.canonical_csv_bytes(missing_id, prefix_schema, contract, formal=True)

    blank_id = _prefix_frame(contract)
    blank_id.loc[:, "cluster_id"] = "   "
    with pytest.raises(v015_io.V015ArtifactError, match="whitespace-only"):
        v015_io.canonical_csv_bytes(blank_id, prefix_schema, contract, formal=True)

    missing_day = _prefix_frame(contract)
    missing_day.loc[0, "prefix_day"] = float("nan")
    with pytest.raises(v015_io.V015ArtifactError, match="key column.*contains NA"):
        v015_io.canonical_csv_bytes(missing_day, prefix_schema, contract, formal=True)

    diagnostics = _fixture_frame("member_fit_diagnostics.csv", contract)
    diagnostics["credible_variant"] = pd.Series([1], dtype=object)
    with pytest.raises(v015_io.V015ArtifactError, match="strict booleans"):
        v015_io.canonical_csv_bytes(
            diagnostics,
            contract.csv_schema("member_fit_diagnostics.csv"),
            contract,
            formal=False,
        )


def test_truth_fields_must_be_constant_within_cluster(
    contract: v015_io.FrozenArtifactContract,
) -> None:
    filename = "test_truth.csv"
    first = _truth_fixture(filename, contract)
    second = first.copy()
    second.loc[:, "forecast_day"] = 1461.0
    second.loc[:, "gamma"] = 0.2
    tampered = pd.concat([first, second], ignore_index=True)
    with pytest.raises(v015_io.V015ArtifactError, match="changes gamma"):
        v015_io.canonical_csv_bytes(
            tampered,
            contract.csv_schema(filename),
            contract,
            formal=False,
        )


def test_canonical_json_rejects_unknown_keys_and_noncanonical_disk_order(
    tmp_path: Path, contract: v015_io.FrozenArtifactContract
) -> None:
    path = tmp_path / "model_state.json"
    payload = _model_state(contract)
    v015_io.write_canonical_json(path, payload, contract)
    assert v015_io.read_canonical_json(path, contract) == payload

    with pytest.raises(v015_io.V015ArtifactError, match="allowlist"):
        v015_io.write_canonical_json(
            tmp_path / "other" / "model_state.json",
            {**payload, "oracle": True},
            contract,
        )

    raw_noncanonical = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    bad = tmp_path / "bad" / "model_state.json"
    bad.parent.mkdir()
    bad.write_bytes(raw_noncanonical)
    with pytest.raises(v015_io.V015ArtifactError, match="canonical JSON"):
        v015_io.read_canonical_json(bad, contract)


def test_exclusive_create_does_not_overwrite_a_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "prediction_commitment.json"
    original_open = Path.open
    raced = False

    def racing_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        nonlocal raced
        if path == target and mode == "xb" and not raced:
            raced = True
            with builtins.open(path, "wb") as handle:
                handle.write(b"racing-writer")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        v015_io._atomic_create(target, b"formal-commitment")
    assert target.read_bytes() == b"racing-writer"


def test_formal_state_validation_rejects_allowlisted_but_fake_model_state(
    tmp_path: Path, contract: v015_io.FrozenArtifactContract
) -> None:
    root = tmp_path / "label-free"
    root.mkdir()
    v015_io.write_canonical_json(
        root / "model_state.json", _model_state(contract), contract
    )
    with pytest.raises(v015_io.V015ArtifactError, match="semantic"):
        v015_io._validate_frozen_state_artifacts(root, contract, formal=True)


def test_truth_commitments_keep_truth_in_a_disjoint_root(
    tmp_path: Path, contract: v015_io.FrozenArtifactContract
) -> None:
    label_free = tmp_path / "label-free"
    sealed = tmp_path / "sealed"
    label_free.mkdir()
    sealed.mkdir()
    for filename in contract.sealed_filenames:
        v015_io.write_canonical_csv(
            sealed / filename,
            _sealed_fixture(filename, contract),
            contract,
            formal=False,
        )

    commitment_path = label_free / "truth_commitments.json"
    payload = v015_io.create_truth_commitments(
        sealed_truth_root=sealed,
        commitment_path=commitment_path,
        contract=contract,
        created_utc="2026-07-23T00:00:00Z",
        formal=False,
    )
    assert [item["path"] for item in payload["files"]] == list(
        contract.sealed_filenames
    )
    assert not any((label_free / name).exists() for name in contract.sealed_filenames)

    with pytest.raises(v015_io.V015ArtifactError, match="not explicitly authorized"):
        v015_io.verify_sealed_truth_files(
            commitment_path=commitment_path,
            sealed_truth_root=sealed,
            contract=contract,
            truth_access_authorized=False,
            formal=False,
        )
    verified = v015_io.verify_sealed_truth_files(
        commitment_path=commitment_path,
        sealed_truth_root=sealed,
        contract=contract,
        truth_access_authorized=True,
        formal=False,
    )
    assert len(verified) == 9

    with pytest.raises(v015_io.V015ArtifactError, match="disjoint"):
        v015_io.assert_separate_truth_roots(label_free, label_free / "truth")


def test_exposure_log_appends_without_rewriting_prior_bytes(
    tmp_path: Path, contract: v015_io.FrozenArtifactContract
) -> None:
    path = tmp_path / "exposure_log.jsonl"
    first = _exposure_event(
        contract, created_utc="2026-07-23T00:00:00Z", phase="generation_started"
    )
    second = _exposure_event(
        contract,
        created_utc="2026-07-23T00:01:00Z",
        phase="truth_committed",
    )
    opened_files = sorted(
        ["intrinsic_matched_pairs.csv", "intrinsic_matched_truth.csv"]
    )
    third = _exposure_event(
        contract,
        created_utc="2026-07-23T00:02:00Z",
        phase="sealed_truth_opened",
        opened_truth_files=opened_files,
    )
    v015_io.append_exposure_event(path, first, contract)
    prefix = path.read_bytes()
    v015_io.append_exposure_event(path, second, contract)
    v015_io.append_exposure_event(path, third, contract)
    assert path.read_bytes().startswith(prefix)
    assert [event["phase"] for event in v015_io.read_exposure_log(path, contract)] == [
        "generation_started",
        "truth_committed",
        "sealed_truth_opened",
    ]

    with pytest.raises(v015_io.V015ArtifactError, match="forgot"):
        v015_io.append_exposure_event(
            path,
            _exposure_event(
                contract,
                created_utc="2026-07-23T00:03:00Z",
                phase="invalid_forget",
            ),
            contract,
        )

    with pytest.raises(v015_io.V015ArtifactError, match="allowlist"):
        v015_io.append_exposure_event(
            path, {**third, "oracle_result": "pass"}, contract
        )

    path.write_bytes(path.read_bytes().replace(b'"phase":', b'"phase" :', 1))
    with pytest.raises(v015_io.V015ArtifactError, match="canonical JSONL"):
        v015_io.read_exposure_log(path, contract)


def test_prediction_bundle_validator_rejects_cluster_and_variant_misalignment(
    contract: v015_io.FrozenArtifactContract,
) -> None:
    frames = _prediction_frames(contract)
    v015_io.validate_prediction_artifact_bundle(
        frames,
        contract,
        formal=False,
        expected_variant_keys=[("fixture-model", "fixture-variant")],
    )

    wrong_cluster = {name: frame.copy() for name, frame in frames.items()}
    wrong_cluster["operating_pack.csv"].loc[:, "cluster_id"] = "fixture-c2"
    with pytest.raises(v015_io.V015ArtifactError, match="cluster set differs"):
        v015_io.validate_prediction_artifact_bundle(
            wrong_cluster, contract, formal=False
        )

    wrong_variant = {name: frame.copy() for name, frame in frames.items()}
    wrong_variant["member_forecast_bundle.csv"].loc[:, "variant_id"] = "other"
    with pytest.raises(v015_io.V015ArtifactError, match="variant coordinates"):
        v015_io.validate_prediction_artifact_bundle(
            wrong_variant, contract, formal=False
        )

    with pytest.raises(v015_io.V015ArtifactError, match="expected variant set"):
        v015_io.validate_prediction_artifact_bundle(
            frames,
            contract,
            formal=False,
            expected_variant_keys=[("fixture-model", "missing")],
        )


def test_sealed_bundle_validator_rejects_mapping_omissions_and_id_reuse(
    contract: v015_io.FrozenArtifactContract,
) -> None:
    frames = _sealed_frames(contract)
    v015_io.validate_sealed_truth_bundle(frames, contract, formal=False)

    missing_member = {name: frame.copy() for name, frame in frames.items()}
    missing_member["intrinsic_matched_truth.csv"] = (
        missing_member["intrinsic_matched_truth.csv"].iloc[[0]].reset_index(drop=True)
    )
    with pytest.raises(v015_io.V015ArtifactError, match="does not cover every member"):
        v015_io.validate_sealed_truth_bundle(missing_member, contract, formal=False)

    reused_id = {name: frame.copy() for name, frame in frames.items()}
    reused_id["risk_development_truth.csv"].loc[:, "cluster_id"] = reused_id[
        "center_development_truth.csv"
    ].loc[0, "cluster_id"]
    with pytest.raises(v015_io.V015ArtifactError, match="across partitions"):
        v015_io.validate_sealed_truth_bundle(reused_id, contract, formal=False)


def test_predictor_hashes_ignore_ids_and_row_order_but_bind_visible_values() -> None:
    prefix = pd.DataFrame(
        {
            "protocol_id": ["p", "p"],
            "partition": ["test", "test"],
            "cluster_id": ["left", "left"],
            "prefix_day": [0.0, 7.0],
            "observed_retention_pct": [100.0, 99.9],
        }
    )
    forecast = pd.DataFrame(
        {
            "protocol_id": ["p", "p"],
            "partition": ["test", "test"],
            "cluster_id": ["left", "left"],
            "forecast_day": [1461.0, 1095.75],
        }
    )
    operating = {
        "past_mean_temperature_c": 25.0,
        "past_mean_soc_fraction": 0.5,
        "past_mean_dod_fraction": 0.6,
        "past_efc_per_year": 250.0,
        "planned_mean_temperature_c": 30.0,
        "planned_mean_soc_fraction": 0.6,
        "planned_mean_dod_fraction": 0.7,
        "planned_efc_per_year": 300.0,
        **{f"placebo_control_{index}": index / 10 for index in range(1, 9)},
    }
    first = v015_io.predictor_content_hashes(
        prefix, forecast, operating, enforce_frozen_counts=False
    )
    first_payloads = v015_io.predictor_content_payloads(
        prefix, forecast, operating, enforce_frozen_counts=False
    )
    assert hashlib.sha256(first_payloads.arm_a).hexdigest() == first.arm_a
    assert hashlib.sha256(first_payloads.arm_b).hexdigest() == first.arm_b
    assert hashlib.sha256(first_payloads.placebo).hexdigest() == first.placebo
    assert first_payloads.random_policy == first_payloads.arm_a

    relabeled = prefix.iloc[::-1].copy()
    relabeled.loc[:, "cluster_id"] = "right"
    second = v015_io.predictor_content_hashes(
        relabeled,
        forecast.iloc[::-1],
        operating,
        enforce_frozen_counts=False,
    )
    assert first == second
    assert first.random_policy == first.arm_a

    changed_real = {**operating, "planned_mean_temperature_c": 31.0}
    third = v015_io.predictor_content_hashes(
        prefix, forecast, changed_real, enforce_frozen_counts=False
    )
    assert third.arm_a == first.arm_a
    assert third.arm_b != first.arm_b
    assert third.placebo == first.placebo

    changed_placebo = {**operating, "placebo_control_8": -0.8}
    fourth = v015_io.predictor_content_hashes(
        prefix, forecast, changed_placebo, enforce_frozen_counts=False
    )
    assert fourth.arm_a == first.arm_a
    assert fourth.arm_b == first.arm_b
    assert fourth.placebo != first.placebo


def test_prediction_commitment_detects_post_commitment_tampering(
    tmp_path: Path, contract: v015_io.FrozenArtifactContract
) -> None:
    label_free = tmp_path / "label-free"
    label_free.mkdir()
    v015_io.write_canonical_json(
        label_free / "model_state.json", _model_state(contract), contract
    )
    csv_names = [
        "prefix_pack.csv",
        "forecast_coordinates.csv",
        "operating_pack.csv",
        "member_fit_diagnostics.csv",
        "member_forecast_bundle.csv",
        "prediction_bundle.csv",
        "risk_bundle.csv",
        "decision_bundle.csv",
    ]
    for filename in csv_names:
        v015_io.write_canonical_csv(
            label_free / filename,
            _fixture_frame(filename, contract),
            contract,
            formal=False,
        )

    commitment_path = label_free / "prediction_commitment.json"
    payload = v015_io.create_prediction_commitment(
        label_free_root=label_free,
        commitment_path=commitment_path,
        contract=contract,
        created_utc="2026-07-23T00:02:00Z",
        formal=False,
    )
    assert payload["sealed_truth_opened_before_commitment"] is False
    v015_io.verify_prediction_commitment(
        commitment_path=commitment_path,
        label_free_root=label_free,
        contract=contract,
        formal=False,
    )

    prediction_path = label_free / "prediction_bundle.csv"
    prediction = v015_io.read_canonical_csv(prediction_path, contract, formal=False)
    prediction.loc[:, "center_forecast_pct"] = 96
    prediction_path.write_bytes(
        v015_io.canonical_csv_bytes(
            prediction,
            contract.csv_schema("prediction_bundle.csv"),
            contract,
            formal=False,
        )
    )
    with pytest.raises(v015_io.V015ArtifactError, match="commitment"):
        v015_io.verify_prediction_commitment(
            commitment_path=commitment_path,
            label_free_root=label_free,
            contract=contract,
            formal=False,
        )
