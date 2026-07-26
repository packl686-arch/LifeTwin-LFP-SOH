from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from typing import Any, Callable, Mapping

import pytest

from lifetwin.experiments import calendar_long_horizon_v015_training as v015
from lifetwin.experiments import (
    calendar_long_horizon_v016_prediction_capsule as capsule,
)
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
from lifetwin.experiments.calendar_long_horizon_v016_training import (
    V021CalibrationAudit,
    V021CommittedMaskRow,
    V021PretruthMaskCommitment,
)


_CREATED_UTC = "2026-07-26T03:00:00Z"


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
    rows = []
    for index in range(900):
        is_last = index == 899
        rows.append(
            V021CommittedMaskRow(
                cluster_id=f"c_{index:04d}",
                label_free_row_sha256=hashlib.sha256(
                    f"row-{index}".encode("ascii")
                ).hexdigest(),
                structural_support_sha256=hashlib.sha256(
                    f"support-{index}".encode("ascii")
                ).hexdigest(),
                successful_structure_family_ids=(
                    (DECLARED_STRUCTURE_FAMILIES[0],)
                    if is_last
                    else DECLARED_STRUCTURE_FAMILIES[:2]
                ),
                eligible=not is_last,
                ineligibility_reasons=(
                    ("insufficient_structure_families",) if is_last else ()
                ),
            )
        )
    return state_codec.build_calibration_mask_commitment_v021(rows=tuple(rows))


def _calibration_audit() -> V021CalibrationAudit:
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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _input_hashes(values: Mapping[str, bytes]) -> dict[str, str]:
    return {name: _sha256(raw) for name, raw in values.items()}


def _compact_json(payload: object) -> bytes:
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


@dataclass(frozen=True)
class _TrainingChainFixture:
    payloads: Mapping[str, Mapping[str, Any]]
    raw_by_name: Mapping[str, bytes]
    decoded: capsule.DecodedPredictionState
    mask: Any
    config_sha256: str


@lru_cache(maxsize=1)
def _training_chain() -> _TrainingChainFixture:
    contract = _contract()
    config_sha256 = contract.artifacts.config_byte_sha256
    state = _training_state()
    mask_commitment = _mask_commitment()
    mask_raw = state_codec.serialize_calibration_mask_commitment_json_v021(
        mask_commitment
    )
    center_inputs = {"center_arrays.bin": b"center fixture bytes\n"}
    risk_inputs = {"risk_arrays.bin": b"risk fixture bytes\n"}
    calibration_inputs = {
        "calibration_arrays.bin": b"calibration fixture bytes\n",
        "calibration_mask_commitment.json": mask_raw,
    }
    center_input_hashes = _input_hashes(center_inputs)
    risk_input_hashes = _input_hashes(risk_inputs)
    calibration_input_hashes = _input_hashes(calibration_inputs)

    training_raw = state_codec.serialize_training_manifest_json_v021(
        center_development_input_hashes=center_input_hashes,
        risk_development_input_hashes=risk_input_hashes,
        center_state=state.center,
        risk_state=state.risk,
        created_utc=_CREATED_UTC,
        contract_view=contract,
    )
    center_raw = v015.canonical_json_bytes(
        {
            "protocol_id": V021_PROTOCOL_ID,
            "config_sha256": config_sha256,
            "state_kind": "center_development",
            "center_state_sha256": v015.center_state_sha256(state.center),
            "center_beta": state.center.beta,
            "development_cluster_count": state.center.development_cluster_count,
            "forecast_horizon_count": state.center.forecast_horizon_count,
            "ridge_penalty": state.center.ridge_penalty,
            "completeness_rule": state.center.completeness_rule,
            "input_byte_hashes": center_input_hashes,
            "created_utc": _CREATED_UTC,
        }
    )
    risk_raw = v015.canonical_json_bytes(
        {
            "protocol_id": V021_PROTOCOL_ID,
            "config_sha256": config_sha256,
            "state_kind": "risk_development",
            "center_checkpoint_byte_sha256": _sha256(center_raw),
            "training_manifest_byte_sha256": _sha256(training_raw),
            "risk_state_sha256": v015.risk_state_sha256(state.risk),
            "development_cluster_count": state.risk.development_cluster_count,
            "eligible_cluster_count": state.risk.eligible_cluster_count,
            "positive_label_count": state.risk.positive_label_count,
            "negative_label_count": state.risk.negative_label_count,
            "input_byte_hashes": risk_input_hashes,
            "created_utc": _CREATED_UTC,
        }
    )
    calibration_raw = state_codec.serialize_calibration_manifest_json_v021(
        calibration_input_hashes=calibration_input_hashes,
        calibration_state=state.calibration,
        created_utc=_CREATED_UTC,
        mask_commitment=mask_commitment,
        contract_view=contract,
    )
    training_provenance = provenance._issue_v021_training_provenance_from_fresh_bytes(
        contract_view=contract,
        attempt_id="v021-capsule-semantics",
        training_state=state,
        calibration_audit=_calibration_audit(),
        mask_commitment=mask_commitment,
        generation_plan_commitment_bytes=b"generation plan fixture\n",
        truth_commitment_bytes=b"truth commitment fixture\n",
        actual_analysis_hash_ledger_commitment_bytes=(
            b"analysis hash ledger fixture\n"
        ),
        label_free_fit_commitment_bytes=b"fit commitment fixture\n",
        center_state_commitment_bytes=center_raw,
        risk_state_commitment_bytes=risk_raw,
        calibration_state_commitment_bytes=calibration_raw,
        center_development_input_bytes=center_inputs,
        risk_development_input_bytes=risk_inputs,
        calibration_input_bytes=calibration_inputs,
    )
    audit_raw = state_codec.serialize_calibration_population_audit_json_v021(
        provenance_envelope=training_provenance,
        created_utc=_CREATED_UTC,
    )
    model_raw = state_codec.serialize_model_state_json_v021(
        training_provenance,
        software_versions=v015.default_software_versions(),
        created_utc=_CREATED_UTC,
    )
    raw_by_name = {
        "center_state_checkpoint.json": center_raw,
        "risk_state_checkpoint.json": risk_raw,
        "training_manifest.json": training_raw,
        "calibration_mask_commitment.json": mask_raw,
        "calibration_manifest.json": calibration_raw,
        "calibration_population_audit.json": audit_raw,
        "model_state.json": model_raw,
    }
    return _TrainingChainFixture(
        payloads={
            name: json.loads(raw)
            for name, raw in raw_by_name.items()
            if name != "calibration_mask_commitment.json"
        },
        raw_by_name=raw_by_name,
        decoded=capsule.decode_prediction_state(
            model_raw,
            expected_config_sha256=config_sha256,
        ),
        mask=capsule._decode_mask_commitment(mask_raw),
        config_sha256=config_sha256,
    )


def test_capsule_mask_decoder_matches_official_codec() -> None:
    raw = state_codec.serialize_calibration_mask_commitment_json_v021(
        _mask_commitment()
    )
    official = state_codec.deserialize_calibration_mask_commitment_json_v021(raw)
    safe = capsule._decode_mask_commitment(raw)

    assert safe.cluster_ids == tuple(row.cluster_id for row in official.rows)
    assert safe.eligible == tuple(row.eligible for row in official.rows)
    assert safe.family_counts == tuple(
        len(row.successful_structure_family_ids) for row in official.rows
    )
    assert safe.eligibility_mask_sha256 == official.eligibility_mask_sha256


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["rows"].__setitem__(
            slice(0, 2), reversed(payload["rows"][:2])
        ),
        lambda payload: payload["rows"][0].__setitem__(
            "label_free_row_sha256", "f" * 64
        ),
        lambda payload: payload.__setitem__("risk_isotonic_eligible_count", 900),
    ],
    ids=("row-order", "row-digest", "eligible-count"),
)
def test_capsule_mask_decoder_rejects_tampering(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    raw = state_codec.serialize_calibration_mask_commitment_json_v021(
        _mask_commitment()
    )
    payload = json.loads(raw)
    mutate(payload)
    tampered = _compact_json(payload)

    with pytest.raises(capsule.V021PredictionCapsuleError):
        capsule._decode_mask_commitment(tampered)
    with pytest.raises(state_codec.V021StateCodecError):
        state_codec.deserialize_calibration_mask_commitment_json_v021(tampered)


def test_capsule_accepts_official_training_chain_semantics() -> None:
    fixture = _training_chain()

    capsule._verify_training_chain_semantics(
        payloads=fixture.payloads,
        raw_by_name=fixture.raw_by_name,
        decoded=fixture.decoded,
        mask=fixture.mask,
        config_sha256=fixture.config_sha256,
    )


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    [
        (
            "center_state_checkpoint.json",
            lambda payload: payload.__setitem__("center_state_sha256", "0" * 64),
            "Center checkpoint semantics changed",
        ),
        (
            "risk_state_checkpoint.json",
            lambda payload: payload.__setitem__(
                "center_checkpoint_byte_sha256", "0" * 64
            ),
            "Risk checkpoint semantics changed",
        ),
        (
            "risk_state_checkpoint.json",
            lambda payload: payload.__setitem__(
                "training_manifest_byte_sha256", "0" * 64
            ),
            "Risk checkpoint semantics changed",
        ),
        (
            "training_manifest.json",
            lambda payload: payload.__setitem__("risk_state_sha256", "0" * 64),
            "Training manifest semantics changed",
        ),
        (
            "calibration_manifest.json",
            lambda payload: payload.__setitem__("isotonic_state_sha256", "0" * 64),
            "Calibration manifest semantics changed",
        ),
        (
            "calibration_population_audit.json",
            lambda payload: payload.__setitem__("conformal_state_sha256", "0" * 64),
            "Calibration population audit semantics changed",
        ),
    ],
)
def test_capsule_training_chain_rejects_broken_semantic_hash_links(
    filename: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    fixture = _training_chain()
    payloads = deepcopy(fixture.payloads)
    mutate(payloads[filename])

    with pytest.raises(capsule.V021PredictionCapsuleError, match=message):
        capsule._verify_training_chain_semantics(
            payloads=payloads,
            raw_by_name=fixture.raw_by_name,
            decoded=fixture.decoded,
            mask=fixture.mask,
            config_sha256=fixture.config_sha256,
        )
