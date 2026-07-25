from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v015_generation as generation,
)
from lifetwin.experiments.calendar_long_horizon_v015_generation import (
    OrdinaryPackRecord,
    V015GenerationError,
    assemble_generated_artifact_frames,
    assert_generation_destinations_available,
    concatenate_member_packs,
    generate_frozen_v015_artifacts,
    prepare_generated_artifacts,
    validate_ordinary_family_counts,
    validate_predictor_content_collision_policy,
)
from lifetwin.experiments.calendar_long_horizon_v015_firewall import (
    AttemptProgress,
    FormalAttemptIdentity,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    DEFAULT_V2_CONFIG_PATH,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FORECAST_COORDINATE_COLUMNS,
    INTRINSIC_MATCHED_PARTITION,
    MATCHED_PAIR_COLUMNS,
    OPERATING_COLUMNS,
    PREFIX_COLUMNS,
    STRESS_PLAN_MATCHED_PARTITION,
    TRUTH_COLUMNS,
    GeneratedMemberPacks,
    MatchedPairPacks,
    load_frozen_protocol_config,
)


def _member_pack(
    partition: str,
    cluster_id: str,
    *,
    prefix_offset: float,
    planned_temperature_c: float = 27.5,
    placebo_offset: float = 0.0,
    family: str = "single_power",
) -> GeneratedMemberPacks:
    contract = load_artifact_contract()
    common = {
        "protocol_id": contract.protocol_id,
        "partition": partition,
        "cluster_id": cluster_id,
    }
    prefix = pd.DataFrame(
        [
            {
                **common,
                "prefix_day": day,
                "observed_retention_pct": (
                    100.0 if day == 0.0 else 100.0 - 0.001 * day - prefix_offset
                ),
            }
            for day in contract.prefix_days
        ],
        columns=PREFIX_COLUMNS,
    )
    forecast = pd.DataFrame(
        [
            {
                **common,
                "forecast_day": day,
            }
            for day in contract.forecast_days
        ],
        columns=FORECAST_COORDINATE_COLUMNS,
    )
    operating_values = {
        "past_mean_temperature_c": 27.5,
        "past_mean_soc_fraction": 0.55,
        "past_mean_dod_fraction": 0.55,
        "past_efc_per_year": 275.0,
        "planned_mean_temperature_c": planned_temperature_c,
        "planned_mean_soc_fraction": 0.55,
        "planned_mean_dod_fraction": 0.55,
        "planned_efc_per_year": 275.0,
        **{
            f"placebo_control_{index}": placebo_offset + index / 100.0
            for index in range(1, 9)
        },
    }
    operating = pd.DataFrame(
        [{**common, **operating_values}],
        columns=OPERATING_COLUMNS,
    )
    truth = pd.DataFrame(
        [
            {
                **common,
                "truth_family": family,
                "truth_parameters_json": '{"a":0.5,"b":0.6}',
                "gamma": 0.1,
                "forecast_day": day,
                "latent_retention_pct": 98.0 - index,
                "noisy_retention_pct": 97.9 - index,
            }
            for index, day in enumerate(contract.forecast_days)
        ],
        columns=TRUTH_COLUMNS,
    )
    return GeneratedMemberPacks(
        prefix_pack=prefix,
        forecast_coordinates=forecast,
        operating_pack=operating,
        truth_pack=truth,
    )


def _matched_pair(
    partition: str,
    left: GeneratedMemberPacks,
    right: GeneratedMemberPacks,
    *,
    pair_id: str,
) -> MatchedPairPacks:
    combined = concatenate_member_packs((left, right))
    left_id = str(left.operating_pack.iloc[0]["cluster_id"])
    right_id = str(right.operating_pack.iloc[0]["cluster_id"])
    mapping = pd.DataFrame(
        [
            {
                "protocol_id": load_artifact_contract().protocol_id,
                "pair_partition": partition,
                "pair_id": pair_id,
                "left_cluster_id": left_id,
                "right_cluster_id": right_id,
                "construction_family": "hand_fixture",
                "left_side_code": "left",
                "right_side_code": "right",
                "latent_prefix_rmse_pp": 0.0,
                "latent_prefix_max_abs_difference_pp": 0.0,
                "truth_separation_25y_pp": 6.0,
            }
        ],
        columns=MATCHED_PAIR_COLUMNS,
    )
    return MatchedPairPacks(
        prefix_pack=combined.prefix_pack,
        forecast_coordinates=combined.forecast_coordinates,
        operating_pack=combined.operating_pack,
        truth_pack=combined.truth_pack,
        matched_pairs=mapping,
    )


def _hand_frames():
    protocol = load_frozen_protocol_config(DEFAULT_V2_CONFIG_PATH)
    ordinary = tuple(
        OrdinaryPackRecord(
            partition=partition,
            family_id="single_power",
            zero_based_index=0,
            packs=_member_pack(
                partition,
                f"c_{partition}",
                prefix_offset=index / 10.0,
                placebo_offset=index / 20.0,
            ),
        )
        for index, partition in enumerate(
            (
                "center_development",
                "risk_development",
                "calibration",
                "test",
                "audit",
            )
        )
    )
    intrinsic = _matched_pair(
        INTRINSIC_MATCHED_PARTITION,
        _member_pack(
            INTRINSIC_MATCHED_PARTITION,
            "c_intrinsic_left",
            prefix_offset=1.0,
            family="intrinsic_single_power",
        ),
        _member_pack(
            INTRINSIC_MATCHED_PARTITION,
            "c_intrinsic_right",
            prefix_offset=1.0,
            family="intrinsic_piecewise_linear_knee",
        ),
        pair_id="p_intrinsic",
    )
    stress = _matched_pair(
        STRESS_PLAN_MATCHED_PARTITION,
        _member_pack(
            STRESS_PLAN_MATCHED_PARTITION,
            "c_stress_low",
            prefix_offset=2.0,
            planned_temperature_c=20.0,
        ),
        _member_pack(
            STRESS_PLAN_MATCHED_PARTITION,
            "c_stress_high",
            prefix_offset=2.0,
            planned_temperature_c=35.0,
        ),
        pair_id="p_stress",
    )
    frames = assemble_generated_artifact_frames(
        ordinary_records=ordinary,
        intrinsic_pairs=(intrinsic,),
        stress_plan_pairs=(stress,),
        protocol=protocol,
        formal=False,
    )
    return protocol, frames


def test_member_pack_aggregation_checks_alignment_and_duplicate_members() -> None:
    first = _member_pack("test", "c_first", prefix_offset=0.1)
    second = _member_pack("test", "c_second", prefix_offset=0.2)
    combined = concatenate_member_packs((second, first))
    assert len(combined.prefix_pack) == 24
    assert len(combined.forecast_coordinates) == 16
    assert len(combined.operating_pack) == 2
    assert len(combined.truth_pack) == 16
    assert set(combined.operating_pack["cluster_id"]) == {"c_first", "c_second"}

    with pytest.raises(V015GenerationError, match="more than one member pack"):
        concatenate_member_packs((first, first))


def test_ordinary_family_count_validator_uses_coordinate_counts() -> None:
    pack = _member_pack("fixture", "c_one", prefix_offset=0.0)
    records = (
        OrdinaryPackRecord("fixture", "family_a", 0, pack),
        OrdinaryPackRecord("fixture", "family_a", 1, pack),
    )
    validate_ordinary_family_counts(
        records,
        {"fixture": {"family_a": 2}},
    )
    with pytest.raises(V015GenerationError, match="family counts"):
        validate_ordinary_family_counts(
            records,
            {"fixture": {"family_a": 1}},
        )


def test_hand_bundle_prevalidates_all_bytes_and_commits_nine_sealed_files() -> None:
    protocol, frames = _hand_frames()
    contract = load_artifact_contract()
    prepared = prepare_generated_artifacts(
        frames,
        protocol=protocol,
        contract=contract,
        created_utc="2026-07-23T01:02:03Z",
        formal=False,
    )
    assert tuple(prepared.sealed_bytes) == contract.sealed_filenames
    assert len(prepared.sealed_bytes) == 9
    files = prepared.truth_commitment_payload["files"]
    assert isinstance(files, list)
    assert [item["path"] for item in files] == list(contract.sealed_filenames)
    for item in files:
        raw = prepared.sealed_bytes[item["path"]]
        assert item["byte_count"] == len(raw)
        assert item["byte_sha256"] == hashlib.sha256(raw).hexdigest()
    assert tuple(prepared.label_free_bytes) == (
        "prefix_pack.csv",
        "forecast_coordinates.csv",
        "operating_pack.csv",
        "truth_commitments.json",
    )


def test_test_predictor_content_must_be_unique() -> None:
    _, frames = _hand_frames()
    duplicate = _member_pack(
        "test",
        "c_test_duplicate",
        prefix_offset=0.3,
        placebo_offset=0.15,
    )
    prefix = pd.concat(
        [frames.label_free["prefix_pack.csv"], duplicate.prefix_pack],
        ignore_index=True,
    )
    forecast = pd.concat(
        [
            frames.label_free["forecast_coordinates.csv"],
            duplicate.forecast_coordinates,
        ],
        ignore_index=True,
    )
    operating = pd.concat(
        [frames.label_free["operating_pack.csv"], duplicate.operating_pack],
        ignore_index=True,
    )
    broken = type(frames)(
        label_free={
            "prefix_pack.csv": prefix,
            "forecast_coordinates.csv": forecast,
            "operating_pack.csv": operating,
        },
        sealed=frames.sealed,
    )
    with pytest.raises(V015GenerationError, match="Duplicate predictor content"):
        validate_predictor_content_collision_policy(broken)


def test_test_and_audit_predictor_content_is_unique_across_partitions() -> None:
    _, frames = _hand_frames()
    prefix = frames.label_free["prefix_pack.csv"].copy()
    operating = frames.label_free["operating_pack.csv"].copy()

    test_prefix = (
        prefix.loc[prefix["partition"].eq("test")]
        .sort_values("prefix_day", kind="stable")["observed_retention_pct"]
        .to_numpy()
    )
    audit_mask = prefix["partition"].eq("audit")
    prefix.loc[audit_mask, "observed_retention_pct"] = test_prefix

    predictor_columns = [
        column
        for column in operating.columns
        if column not in {"protocol_id", "partition", "cluster_id"}
    ]
    test_operating = operating.loc[
        operating["partition"].eq("test"), predictor_columns
    ].iloc[0]
    operating.loc[operating["partition"].eq("audit"), predictor_columns] = (
        test_operating.to_numpy()
    )
    broken = type(frames)(
        label_free={
            **frames.label_free,
            "prefix_pack.csv": prefix,
            "operating_pack.csv": operating,
        },
        sealed=frames.sealed,
    )
    with pytest.raises(V015GenerationError, match="Duplicate predictor content"):
        validate_predictor_content_collision_policy(broken)


def test_matched_repeats_are_exactly_the_declared_pairs() -> None:
    _, frames = _hand_frames()
    assert validate_predictor_content_collision_policy(frames) > 0

    operating = frames.label_free["operating_pack.csv"].copy()
    mask = operating["cluster_id"].eq("c_intrinsic_right")
    operating.loc[mask, "planned_mean_temperature_c"] = 31.0
    broken = type(frames)(
        label_free={
            **frames.label_free,
            "operating_pack.csv": operating,
        },
        sealed=frames.sealed,
    )
    with pytest.raises(V015GenerationError, match="repeats"):
        validate_predictor_content_collision_policy(broken)


def test_destination_preflight_rejects_nested_and_partial_roots(
    tmp_path: Path,
) -> None:
    contract = load_artifact_contract()
    label = tmp_path / "label"
    sealed = tmp_path / "sealed"
    assert_generation_destinations_available(
        label_free_root=label,
        sealed_truth_root=sealed,
        contract=contract,
    )
    label.mkdir()
    (label / "prefix_pack.csv").write_bytes(b"partial")
    with pytest.raises(V015GenerationError, match="partial prior output"):
        assert_generation_destinations_available(
            label_free_root=label,
            sealed_truth_root=sealed,
            contract=contract,
        )
    assert not sealed.exists()

    with pytest.raises(V015GenerationError, match="disjoint"):
        assert_generation_destinations_available(
            label_free_root=tmp_path / "outer",
            sealed_truth_root=tmp_path / "outer" / "sealed",
            contract=contract,
        )


def test_formal_entry_has_no_seed_or_protocol_override_surface() -> None:
    parameters = inspect.signature(generate_frozen_v015_artifacts).parameters
    assert tuple(parameters) == ("label_free_root", "sealed_truth_root")
    assert all(
        name not in parameters
        for name in ("seed", "config", "family", "partition", "row_count")
    )


def test_formal_entry_checks_environment_before_plan_or_seed_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GuardStopped(RuntimeError):
        pass

    observed_roots: list[Path] = []

    def stop_at_guard(repo_root: str | Path) -> None:
        observed_roots.append(Path(repo_root))
        raise GuardStopped

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("formal plan or config was reached before the guard")

    monkeypatch.setattr(
        generation,
        "verify_formal_environment",
        stop_at_guard,
    )
    monkeypatch.setattr(
        generation,
        "load_frozen_protocol_config",
        forbidden,
    )
    monkeypatch.setattr(
        generation,
        "audit_generation_coordinate_plan",
        forbidden,
    )
    with pytest.raises(GuardStopped):
        generation.generate_frozen_v015_artifacts(
            label_free_root=Path("unused-label-root"),
            sealed_truth_root=Path("unused-sealed-root"),
        )
    assert observed_roots == [generation._PROJECT_ROOT]


def test_formal_entry_requires_checkpointed_ledger_before_plan_or_seed_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LedgerStopped(RuntimeError):
        pass

    contract = load_artifact_contract()
    environment = SimpleNamespace(
        git_commit="a" * 40,
        config_byte_sha256=contract.config_byte_sha256,
    )
    protocol = SimpleNamespace(protocol_id=contract.protocol_id)

    def stop_at_ledger(**kwargs: object) -> None:
        assert kwargs["environment"] is environment
        raise LedgerStopped

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("formal plan or seed use preceded the ledger check")

    monkeypatch.setattr(generation, "verify_formal_environment", lambda _: environment)
    monkeypatch.setattr(generation, "load_frozen_protocol_config", lambda _: protocol)
    monkeypatch.setattr(generation, "load_artifact_contract", lambda: contract)
    monkeypatch.setattr(generation, "_require_pre_generation_ledger", stop_at_ledger)
    monkeypatch.setattr(generation, "audit_generation_coordinate_plan", forbidden)
    with pytest.raises(LedgerStopped):
        generation.generate_frozen_v015_artifacts(
            label_free_root=Path("unused-label-root"),
            sealed_truth_root=Path("unused-sealed-root"),
        )


def test_pre_generation_ledger_identity_must_match_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_artifact_contract()
    label_root = tmp_path / "label"
    label_root.mkdir()
    (label_root / "exposure_log.jsonl").write_bytes(b"fixture\n")
    identity = FormalAttemptIdentity(
        "attempt-one", "a" * 40, contract.config_byte_sha256
    )
    state = AttemptProgress(
        identity=identity,
        completed_phase="before_generation",
        pending_phase=None,
        truth_commitments_byte_sha256=None,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(),
        terminal_failed=False,
    )
    monkeypatch.setattr(
        generation,
        "validate_formal_exposure_log",
        lambda *_: {identity.attempt_id: state},
    )
    matching = SimpleNamespace(
        git_commit=identity.git_commit,
        config_byte_sha256=contract.config_byte_sha256,
    )
    assert (
        generation._require_pre_generation_ledger(
            label_free_root=label_root,
            contract=contract,
            environment=matching,
        )
        == identity.attempt_id
    )

    changed = SimpleNamespace(
        git_commit="b" * 40,
        config_byte_sha256=contract.config_byte_sha256,
    )
    with pytest.raises(V015GenerationError, match="identity"):
        generation._require_pre_generation_ledger(
            label_free_root=label_root,
            contract=contract,
            environment=changed,
        )

    prior_identity = FormalAttemptIdentity(
        "failed-old-attempt", "b" * 40, contract.config_byte_sha256
    )
    failed_prior = AttemptProgress(
        identity=prior_identity,
        completed_phase="",
        pending_phase=None,
        truth_commitments_byte_sha256=None,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(),
        terminal_failed=True,
    )
    monkeypatch.setattr(
        generation,
        "validate_formal_exposure_log",
        lambda *_: {
            prior_identity.attempt_id: failed_prior,
            identity.attempt_id: state,
        },
    )
    with pytest.raises(V015GenerationError, match="prior formal attempt.*different"):
        generation._require_pre_generation_ledger(
            label_free_root=label_root,
            contract=contract,
            environment=matching,
        )

    committed_prior_identity = FormalAttemptIdentity(
        "committed-old-attempt",
        identity.git_commit,
        contract.config_byte_sha256,
    )
    committed_prior = AttemptProgress(
        identity=committed_prior_identity,
        completed_phase="truth_committed",
        pending_phase=None,
        truth_commitments_byte_sha256="c" * 64,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(),
        terminal_failed=True,
    )
    monkeypatch.setattr(
        generation,
        "validate_formal_exposure_log",
        lambda *_: {
            committed_prior_identity.attempt_id: committed_prior,
            identity.attempt_id: state,
        },
    )
    with pytest.raises(V015GenerationError, match="never regenerated"):
        generation._require_pre_generation_ledger(
            label_free_root=label_root,
            contract=contract,
            environment=matching,
        )
