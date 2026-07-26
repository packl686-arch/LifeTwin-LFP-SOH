from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_training as v015
from lifetwin.experiments import (
    calendar_long_horizon_v016_provenance as provenance,
)
from lifetwin.experiments import calendar_long_horizon_v016_state as state_codec
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    DECLARED_STRUCTURE_FAMILIES,
)
from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    V021ContractView,
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v016_pipeline import (
    V021PipelineError,
    recompute_label_free_pipeline_v021,
)
from lifetwin.experiments.calendar_long_horizon_v016_training import (
    V021CalibrationAudit,
    V021CommittedMaskRow,
    V021PretruthMaskCommitment,
)


_CREATED = "2026-07-26T03:00:00Z"


@lru_cache(maxsize=1)
def _contract() -> V021ContractView:
    return load_v021_contract_view()


@lru_cache(maxsize=1)
def _training_state() -> v015.FrozenTrainingState:
    probe = v015.make_probe_state(0.5)
    center = v015.CenterDevelopmentState(beta=0.5)
    risk = v015.RiskDevelopmentState(
        prefix_only_risk=probe.prefix_only_risk,
        visible_stress_risk=probe.visible_stress_risk,
        placebo_risk=probe.placebo_risk,
        arm_a_plus_s_plan_risk=probe.arm_a_plus_s_plan_risk,
        strongest_single_feature_name=probe.strongest_single_feature_name,
        strongest_single_feature_orientation=1,
        strongest_single_feature_auroc=0.5,
        development_cluster_count=600,
        eligible_cluster_count=600,
        positive_label_count=300,
        negative_label_count=300,
    )
    calibration = v015.CalibrationDevelopmentState(
        prefix_only_isotonic=probe.prefix_only_isotonic,
        visible_stress_isotonic=probe.visible_stress_isotonic,
        conformal=probe.conformal,
        selected_mean_baseline="target_prefix_persistence",
        mean_baseline_iae_pp=(
            ("target_prefix_persistence", 1.0),
            ("target_prefix_sqrt_time", 2.0),
            ("target_prefix_bounded_power_law", 3.0),
        ),
        calibration_cluster_count=900,
        positive_label_count=450,
        negative_label_count=450,
    )
    return v015.FrozenTrainingState(center, risk, calibration)


@lru_cache(maxsize=1)
def _mask_commitment() -> V021PretruthMaskCommitment:
    rows: list[V021CommittedMaskRow] = []
    for index in range(900):
        ineligible = index == 899
        family_ids = (
            (DECLARED_STRUCTURE_FAMILIES[0],)
            if ineligible
            else (
                DECLARED_STRUCTURE_FAMILIES[0],
                DECLARED_STRUCTURE_FAMILIES[1],
            )
        )
        rows.append(
            V021CommittedMaskRow(
                cluster_id=f"c_{index:04d}",
                label_free_row_sha256=hashlib.sha256(
                    f"row-{index}".encode("ascii")
                ).hexdigest(),
                structural_support_sha256=hashlib.sha256(
                    f"support-{index}".encode("ascii")
                ).hexdigest(),
                successful_structure_family_ids=family_ids,
                eligible=not ineligible,
                ineligibility_reasons=(
                    ("insufficient_structure_families",) if ineligible else ()
                ),
            )
        )
    return state_codec.build_calibration_mask_commitment_v021(rows=tuple(rows))


def _hashes(letter: str) -> dict[str, str]:
    return {"fixture_array": letter * 64}


def _calibration_hashes() -> dict[str, str]:
    commitment = _mask_commitment()
    return {
        "calibration_arrays": "c" * 64,
        "calibration_mask_commitment.json": hashlib.sha256(
            state_codec.serialize_calibration_mask_commitment_json_v021(commitment)
        ).hexdigest(),
    }


def _fresh_input_bytes() -> dict[str, dict[str, bytes]]:
    mask_raw = state_codec.serialize_calibration_mask_commitment_json_v021(
        _mask_commitment()
    )
    return {
        "center_development": {"center_arrays.bin": b"fresh-v021-center\n"},
        "risk_development": {"risk_arrays.bin": b"fresh-v021-risk\n"},
        "calibration": {
            "calibration_arrays.bin": b"fresh-v021-calibration\n",
            "calibration_mask_commitment.json": mask_raw,
        },
    }


def _provenance_kwargs() -> dict[str, object]:
    inputs = _fresh_input_bytes()
    return {
        "contract_view": _contract(),
        "attempt_id": "v021-unit-state",
        "training_state": _training_state(),
        "calibration_audit": _audit(),
        "mask_commitment": _mask_commitment(),
        "generation_plan_commitment_bytes": b"fresh-generation-plan\n",
        "truth_commitment_bytes": b"fresh-truth-commitment\n",
        "actual_analysis_hash_ledger_commitment_bytes": (
            b"fresh-actual-analysis-hash-ledger\n"
        ),
        "label_free_fit_commitment_bytes": b"fresh-fit-commitment\n",
        "center_state_commitment_bytes": b"fresh-center-state\n",
        "risk_state_commitment_bytes": b"fresh-risk-state\n",
        "calibration_state_commitment_bytes": b"fresh-calibration-state\n",
        "center_development_input_bytes": inputs["center_development"],
        "risk_development_input_bytes": inputs["risk_development"],
        "calibration_input_bytes": inputs["calibration"],
    }


@lru_cache(maxsize=1)
def _provenance() -> provenance.V021TrainingProvenanceEnvelope:
    return provenance._issue_v021_training_provenance_from_fresh_bytes(
        **_provenance_kwargs()
    )


def _audit() -> V021CalibrationAudit:
    commitment = _mask_commitment()
    return V021CalibrationAudit(
        source_calibration_count=900,
        risk_isotonic_eligible_count=899,
        risk_isotonic_ineligible_zero_family_count=0,
        risk_isotonic_ineligible_one_family_count=1,
        risk_isotonic_ineligible_other_count=0,
        risk_isotonic_positive_label_count=449,
        risk_isotonic_negative_label_count=450,
        mean_baseline_count=900,
        conformal_calibration_count=900,
        conformal_order_statistic_index=811,
        eligibility_mask_sha256=commitment.eligibility_mask_sha256,
    )


def _compact(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _pretty(payload: dict[str, object]) -> bytes:
    return v015.canonical_json_bytes(payload)


def _model_raw() -> bytes:
    return state_codec.serialize_model_state_json_v021(
        _provenance(),
        software_versions=v015.default_software_versions(),
        created_utc=_CREATED,
    )


def _validated_model() -> provenance.V021ValidatedModelStateEnvelope:
    return state_codec.deserialize_model_state_json_v021(
        _model_raw(),
        provenance_envelope=_provenance(),
    )


def _committed_model() -> provenance.V021CommittedModelStateEnvelope:
    raw = _model_raw()
    artifacts = {"model_state.json": raw}
    payload = {
        "protocol_id": V021_PROTOCOL_ID,
        "config_sha256": _contract().artifacts.config_byte_sha256,
        "git_commit": "1" * 40,
        "files": [
            {
                "path": "model_state.json",
                "row_count": 1,
                "byte_count": len(raw),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
        "created_utc": _CREATED,
    }
    commitment_raw = _pretty(payload)
    return provenance._issue_committed_model_state_envelope_v021(
        validated_model_state=_validated_model(),
        model_state_commitment_bytes=commitment_raw,
        ledger_model_state_commitment_byte_sha256=hashlib.sha256(
            commitment_raw
        ).hexdigest(),
        committed_state_artifact_bytes=artifacts,
    )


def _assert_identity_only(v2: dict[str, object], v021: dict[str, object]) -> None:
    differences = {key for key in v2 if key not in v021 or v2[key] != v021[key]} | {
        key for key in v021 if key not in v2
    }
    assert differences == {"protocol_id", "config_sha256"}
    assert v2["protocol_id"] == v015.FROZEN_PROTOCOL_ID
    assert v2["config_sha256"] == v015.FROZEN_CONFIG_BYTE_SHA256
    assert v021["protocol_id"] == V021_PROTOCOL_ID
    assert v021["config_sha256"] == _contract().artifacts.config_byte_sha256


def test_model_and_inherited_manifests_change_only_external_identity() -> None:
    training = _training_state()
    fresh_hashes = _provenance().input_byte_hashes
    v2_model = v015.build_model_state_payload(
        training,
        center_development_input_hashes=fresh_hashes["center_development"],
        risk_development_input_hashes=fresh_hashes["risk_development"],
        calibration_input_hashes=fresh_hashes["calibration"],
        software_versions=v015.default_software_versions(),
        created_utc=_CREATED,
    )
    v021_model = json.loads(_model_raw())
    _assert_identity_only(v2_model, v021_model)

    v2_training = v015.build_training_manifest(
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        center_state=training.center,
        risk_state=training.risk,
        created_utc=_CREATED,
    )
    v021_training = state_codec.build_training_manifest_payload_v021(
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        center_state=training.center,
        risk_state=training.risk,
        created_utc=_CREATED,
        contract_view=_contract(),
    )
    _assert_identity_only(v2_training, v021_training)

    v2_calibration = v015.build_calibration_manifest(
        calibration_input_hashes=_calibration_hashes(),
        calibration_state=training.calibration,
        created_utc=_CREATED,
    )
    v021_calibration = state_codec.build_calibration_manifest_payload_v021(
        calibration_input_hashes=_calibration_hashes(),
        calibration_state=training.calibration,
        created_utc=_CREATED,
        mask_commitment=_mask_commitment(),
        contract_view=_contract(),
    )
    _assert_identity_only(v2_calibration, v021_calibration)


def test_model_state_roundtrip_preserves_exact_v2_numeric_dataclasses() -> None:
    raw = _model_raw()
    envelope = state_codec.deserialize_model_state_json_v021(
        raw,
        provenance_envelope=_provenance(),
    )
    decoded = envelope.decoded_model_state
    assert decoded.training_state == _training_state()
    assert decoded.frozen_label_free_state == v015.construct_frozen_label_free_state(
        _training_state().center,
        _training_state().risk,
        _training_state().calibration,
    )
    assert decoded.input_byte_hashes == _provenance().input_byte_hashes
    assert envelope.model_state_byte_sha256 == hashlib.sha256(raw).hexdigest()
    assert raw == _pretty(json.loads(raw))


def test_v015_state_identity_relabel_and_fake_hashes_cannot_cross_codec() -> None:
    with pytest.raises(state_codec.V021StateCodecError, match="factory-issued"):
        state_codec.serialize_model_state_json_v021(
            _training_state(),
            software_versions=v015.default_software_versions(),
            created_utc=_CREATED,
        )

    relabeled = json.loads(_model_raw())
    relabeled["center_state"]["beta"] = 0.6
    with pytest.raises(state_codec.V021StateCodecError, match="training provenance"):
        state_codec.deserialize_model_state_json_v021(
            _pretty(relabeled),
            provenance_envelope=_provenance(),
        )

    fake_hashes = json.loads(_model_raw())
    fake_hashes["input_byte_hashes"]["center_development"]["center_arrays.bin"] = (
        "0" * 64
    )
    with pytest.raises(state_codec.V021StateCodecError, match="fresh bytes"):
        state_codec.deserialize_model_state_json_v021(
            _pretty(fake_hashes),
            provenance_envelope=_provenance(),
        )


def test_formal_pipeline_requires_committed_not_decoded_or_validated_state() -> None:
    empty = pd.DataFrame()
    for forbidden in (
        _training_state(),
        v015.construct_frozen_label_free_state(
            _training_state().center,
            _training_state().risk,
            _training_state().calibration,
        ),
        _validated_model(),
    ):
        with pytest.raises(V021PipelineError, match="IO-issued"):
            recompute_label_free_pipeline_v021(
                prefix_pack=empty,
                forecast_coordinates=empty,
                operating_pack=empty,
                member_fit_diagnostics=empty,
                member_forecast_bundle=empty,
                model_state_envelope=forbidden,
                contract=_contract().artifacts,
                formal=True,
            )
    assert _committed_model().model_state_commitment_artifact_byte_sha256


def test_provenance_is_private_immutable_and_derived_from_exact_bytes() -> None:
    issued = _provenance()
    inputs = _fresh_input_bytes()
    assert issued.input_byte_hashes == {
        phase: {
            name: hashlib.sha256(raw).hexdigest() for name, raw in phase_inputs.items()
        }
        for phase, phase_inputs in inputs.items()
    }
    with pytest.raises(AttributeError, match="immutable"):
        issued._attempt_id = "v021-forged"  # type: ignore[misc]
    with pytest.raises(TypeError):
        provenance.V021TrainingProvenanceEnvelope()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        provenance.V021ValidatedModelStateEnvelope()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        provenance.V021CommittedModelStateEnvelope()  # type: ignore[call-arg]

    bad = _provenance_kwargs()
    bad["calibration_input_bytes"] = {
        "calibration_arrays.bin": b"fresh-v021-calibration\n"
    }
    with pytest.raises(
        provenance.V021ProvenanceError,
        match="committed mask bytes",
    ):
        provenance._issue_v021_training_provenance_from_fresh_bytes(**bad)


def test_committed_model_envelope_requires_exact_ledger_phase_hash() -> None:
    raw = _model_raw()
    payload = {
        "protocol_id": V021_PROTOCOL_ID,
        "config_sha256": _contract().artifacts.config_byte_sha256,
        "git_commit": "1" * 40,
        "files": [
            {
                "path": "model_state.json",
                "row_count": 1,
                "byte_count": len(raw),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
        "created_utc": _CREATED,
    }
    with pytest.raises(
        provenance.V021ProvenanceError,
        match="Ledger model-state phase hash",
    ):
        provenance._issue_committed_model_state_envelope_v021(
            validated_model_state=_validated_model(),
            model_state_commitment_bytes=_pretty(payload),
            ledger_model_state_commitment_byte_sha256="0" * 64,
            committed_state_artifact_bytes={"model_state.json": raw},
        )


def test_training_and_calibration_manifests_roundtrip_and_bind_state() -> None:
    training = _training_state()
    training_raw = state_codec.serialize_training_manifest_json_v021(
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        center_state=training.center,
        risk_state=training.risk,
        created_utc=_CREATED,
        contract_view=_contract(),
    )
    assert (
        state_codec.deserialize_training_manifest_json_v021(
            training_raw,
            center_state=training.center,
            risk_state=training.risk,
            contract_view=_contract(),
        )["protocol_id"]
        == V021_PROTOCOL_ID
    )

    calibration_raw = state_codec.serialize_calibration_manifest_json_v021(
        calibration_input_hashes=_calibration_hashes(),
        calibration_state=training.calibration,
        created_utc=_CREATED,
        mask_commitment=_mask_commitment(),
        contract_view=_contract(),
    )
    decoded = state_codec.deserialize_calibration_manifest_json_v021(
        calibration_raw,
        calibration_state=training.calibration,
        mask_commitment=_mask_commitment(),
        contract_view=_contract(),
    )
    assert decoded["calibration_input_hashes"] == _calibration_hashes()


def test_calibration_input_hashes_must_include_exact_mask_bytes() -> None:
    training = _training_state()
    with pytest.raises(state_codec.V021StateCodecError, match="factory-issued"):
        state_codec.serialize_model_state_json_v021(
            training,
            software_versions=v015.default_software_versions(),
            created_utc=_CREATED,
        )
    bad = _calibration_hashes()
    bad["calibration_mask_commitment.json"] = "0" * 64
    with pytest.raises(state_codec.V021StateCodecError, match="incorrect"):
        state_codec.serialize_calibration_manifest_json_v021(
            calibration_input_hashes=bad,
            calibration_state=training.calibration,
            created_utc=_CREATED,
            mask_commitment=_mask_commitment(),
            contract_view=_contract(),
        )


def test_mask_commitment_codec_recomputes_count_order_and_digest() -> None:
    commitment = _mask_commitment()
    raw = state_codec.serialize_calibration_mask_commitment_json_v021(commitment)
    assert (
        state_codec.deserialize_calibration_mask_commitment_json_v021(raw) == commitment
    )
    assert commitment.eligible_count == 899

    reordered = json.loads(raw)
    reordered["rows"][0], reordered["rows"][1] = (
        reordered["rows"][1],
        reordered["rows"][0],
    )
    with pytest.raises(state_codec.V021StateCodecError, match="order"):
        state_codec.deserialize_calibration_mask_commitment_json_v021(
            _compact(reordered)
        )

    changed_row = json.loads(raw)
    changed_row["rows"][0]["label_free_row_sha256"] = "f" * 64
    with pytest.raises(state_codec.V021StateCodecError, match="digest"):
        state_codec.deserialize_calibration_mask_commitment_json_v021(
            _compact(changed_row)
        )

    changed_count = json.loads(raw)
    changed_count["risk_isotonic_eligible_count"] = 900
    with pytest.raises(state_codec.V021StateCodecError, match="eligible count"):
        state_codec.deserialize_calibration_mask_commitment_json_v021(
            _compact(changed_count)
        )


def test_calibration_population_audit_roundtrips_and_cross_validates() -> None:
    training = _training_state()
    raw = state_codec.serialize_calibration_population_audit_json_v021(
        provenance_envelope=_provenance(),
        created_utc=_CREATED,
    )
    decoded = state_codec.deserialize_calibration_population_audit_json_v021(
        raw,
        provenance_envelope=_provenance(),
    )
    assert decoded.audit == _audit()
    assert len(decoded.eligibility_mask) == 900
    assert sum(decoded.eligibility_mask) == 899
    assert decoded.eligibility_mask_cluster_ids == tuple(
        row.cluster_id for row in _mask_commitment().rows
    )
    assert decoded.isotonic_state_sha256 == v015.isotonic_state_sha256(
        training.calibration
    )
    assert decoded.conformal_state_sha256 == v015.conformal_state_sha256(
        training.calibration
    )
    payload = json.loads(raw)
    assert set(state_codec._REQUIRED_AUDIT_COUNT_FIELDS).issubset(payload)
    assert payload["config_sha256"] == _contract().artifacts.config_byte_sha256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__("isotonic_state_sha256", "0" * 64),
            "isotonic state hash",
        ),
        (
            lambda payload: payload.__setitem__(
                "calibration_mask_commitment_byte_sha256", "0" * 64
            ),
            "mask byte hash",
        ),
        (
            lambda payload: payload.__setitem__(
                "risk_isotonic_positive_label_count", 448
            ),
            "risk counts",
        ),
        (
            lambda payload: payload["eligibility_mask"].__setitem__(0, False),
            "mask rows or order",
        ),
        (
            lambda payload: payload["eligibility_mask_cluster_ids"].reverse(),
            "mask rows or order",
        ),
    ],
)
def test_calibration_population_audit_rejects_tampering(
    mutate: object,
    message: str,
) -> None:
    raw = state_codec.serialize_calibration_population_audit_json_v021(
        provenance_envelope=_provenance(),
        created_utc=_CREATED,
    )
    payload = json.loads(raw)
    assert callable(mutate)
    mutate(payload)
    with pytest.raises(state_codec.V021StateCodecError, match=message):
        state_codec.deserialize_calibration_population_audit_json_v021(
            _pretty(payload),
            provenance_envelope=_provenance(),
        )


def test_calibration_audit_rejects_plausible_60_839_label_substitution() -> None:
    raw = state_codec.serialize_calibration_population_audit_json_v021(
        provenance_envelope=_provenance(),
        created_utc=_CREATED,
    )
    payload = json.loads(raw)
    payload["risk_isotonic_positive_label_count"] = 60
    payload["risk_isotonic_negative_label_count"] = 839
    with pytest.raises(
        state_codec.V021StateCodecError,
        match="training-issued evidence",
    ):
        state_codec.deserialize_calibration_population_audit_json_v021(
            _pretty(payload),
            provenance_envelope=_provenance(),
        )


def test_canonical_decoders_reject_duplicate_keys_nan_extra_keys_and_identity() -> None:
    raw = _model_raw()
    current_hash = _contract().artifacts.config_byte_sha256
    duplicate = raw.replace(
        b"{\n",
        (b'{\n  "config_sha256": "' + current_hash.encode("ascii") + b'",\n'),
        1,
    )
    with pytest.raises(state_codec.V021StateCodecError, match="Duplicate JSON key"):
        state_codec.deserialize_model_state_json_v021(
            duplicate,
            provenance_envelope=_provenance(),
        )

    nonfinite = json.loads(raw)
    nonfinite["center_state"]["beta"] = float("nan")
    nonfinite_raw = (
        json.dumps(nonfinite, allow_nan=True, indent=2, sort_keys=True) + "\n"
    ).encode()
    with pytest.raises(state_codec.V021StateCodecError, match="Nonfinite"):
        state_codec.deserialize_model_state_json_v021(
            nonfinite_raw,
            provenance_envelope=_provenance(),
        )

    extra = json.loads(raw)
    extra["unexpected"] = 1
    with pytest.raises(state_codec.V021StateCodecError, match="keys changed"):
        state_codec.deserialize_model_state_json_v021(
            _pretty(extra),
            provenance_envelope=_provenance(),
        )

    wrong_identity = json.loads(raw)
    wrong_identity["config_sha256"] = "0" * 64
    with pytest.raises(state_codec.V021StateCodecError, match="amendment byte hash"):
        state_codec.deserialize_model_state_json_v021(
            _pretty(wrong_identity),
            provenance_envelope=_provenance(),
        )

    mask_raw = state_codec.serialize_calibration_mask_commitment_json_v021(
        _mask_commitment()
    )
    duplicate_mask = mask_raw.replace(
        b"{",
        b'{"schema_version":"1.0.0",',
        1,
    )
    with pytest.raises(state_codec.V021StateCodecError, match="Duplicate JSON key"):
        state_codec.deserialize_calibration_mask_commitment_json_v021(duplicate_mask)


def test_state_module_has_no_filesystem_or_outcome_reader_capability() -> None:
    source_path = Path(inspect.getsourcefile(state_codec) or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        fragment in module
        for module in imported_modules
        for fragment in ("pathlib", "pandas", "generation", "scoring")
    )

    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_attributes.isdisjoint(
        {"open", "read_bytes", "read_text", "write_bytes", "write_text"}
    )
    assert called_names.isdisjoint({"open", "setattr"})

    forbidden_parameters = {
        "path",
        "dataframe",
        "truth",
        "labels",
        "targets",
        "generator",
        "reader",
    }
    for name in state_codec.__all__:
        value = getattr(state_codec, name)
        if not callable(value) or inspect.isclass(value):
            continue
        parameters = inspect.signature(value).parameters
        assert not any(
            token in parameter.lower()
            for parameter in parameters
            for token in forbidden_parameters
        )
