"""Sealed, pure-memory provenance capabilities for the V2.1 formal path.

The V2.1 numerical state intentionally reuses frozen V2 dataclasses.  Their
Python type therefore cannot prove that the values were produced by a fresh
V2.1 attempt.  This module supplies that missing distinction: only package-
private issuers can wrap numeric state, and the wrapper binds the complete
V2.1 commitment chain plus hashes computed from exact input bytes.

There is no filesystem, dataframe, truth reader, generator, or scoring surface
here.  Callers cannot construct either public envelope class directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from typing import Any, Mapping

from lifetwin.experiments import calendar_long_horizon_v015_training as _v015
from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    V021ContractView,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v016_training import (
    V021CalibrationAudit,
    V021PretruthMaskCommitment,
)


class V021ProvenanceError(ValueError):
    """Raised when a value cannot prove fresh V2.1 provenance."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID = re.compile(r"^v021-[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_INPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,159}$")
_ISSUER_KEY = object()
_TRAINING_DOMAIN = b"lifetwin-v021-training-provenance-v1\0"
_MODEL_DOMAIN = b"lifetwin-v021-model-provenance-v1\0"
_COMMITTED_MODEL_DOMAIN = b"lifetwin-v021-committed-model-provenance-v1\0"
_MASK_INPUT_NAME = "calibration_mask_commitment.json"
_COMMITMENT_NAMES = (
    "generation_plan",
    "truth",
    "actual_analysis_hash_ledger",
    "label_free_fit",
    "center_state",
    "risk_state",
    "calibration_mask",
    "calibration_state",
)


def _canonical_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise V021ProvenanceError("Provenance payload is not canonical JSON") from exc
    return encoded + b"\n"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _update_sized(hasher: Any, raw: bytes) -> None:
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)


def _exact_raw(raw: object, *, context: str) -> bytes:
    if type(raw) is not bytes or not raw:
        raise V021ProvenanceError(f"{context} must be nonempty exact bytes")
    return raw


def _hash_exact_inputs(
    values: object,
    *,
    context: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping) or not values:
        raise V021ProvenanceError(f"{context} must be a nonempty byte mapping")
    items: list[tuple[str, str]] = []
    for name, raw in values.items():
        if (
            not isinstance(name, str)
            or _INPUT_NAME.fullmatch(name) is None
            or name.startswith("/")
            or ".." in name.split("/")
        ):
            raise V021ProvenanceError(f"{context} contains an invalid input name")
        items.append(
            (
                name,
                _sha256(_exact_raw(raw, context=f"{context}.{name}")),
            )
        )
    canonical = tuple(sorted(items))
    if len({name for name, _ in canonical}) != len(canonical):
        raise V021ProvenanceError(f"{context} contains duplicate input names")
    return canonical


def _validate_preverified_input_hashes(
    values: object,
    *,
    context: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping) or not values:
        raise V021ProvenanceError(f"{context} must be a nonempty hash mapping")
    items: list[tuple[str, str]] = []
    for name, digest in values.items():
        if (
            not isinstance(name, str)
            or _INPUT_NAME.fullmatch(name) is None
            or name.startswith("/")
            or ".." in name.split("/")
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise V021ProvenanceError(f"{context} contains an invalid entry")
        items.append((name, digest))
    return tuple(sorted(items))


def _audit_payload(audit: V021CalibrationAudit) -> dict[str, object]:
    return {
        "source_calibration_count": audit.source_calibration_count,
        "risk_isotonic_eligible_count": audit.risk_isotonic_eligible_count,
        "risk_isotonic_ineligible_zero_family_count": (
            audit.risk_isotonic_ineligible_zero_family_count
        ),
        "risk_isotonic_ineligible_one_family_count": (
            audit.risk_isotonic_ineligible_one_family_count
        ),
        "risk_isotonic_ineligible_other_count": (
            audit.risk_isotonic_ineligible_other_count
        ),
        "risk_isotonic_positive_label_count": (
            audit.risk_isotonic_positive_label_count
        ),
        "risk_isotonic_negative_label_count": (
            audit.risk_isotonic_negative_label_count
        ),
        "mean_baseline_count": audit.mean_baseline_count,
        "conformal_calibration_count": audit.conformal_calibration_count,
        "conformal_order_statistic_index": audit.conformal_order_statistic_index,
        "eligibility_mask_sha256": audit.eligibility_mask_sha256,
    }


def _require_contract(contract_view: object) -> V021ContractView:
    if type(contract_view) is not V021ContractView:
        raise V021ProvenanceError(
            "contract_view must be an exact validated V021ContractView"
        )
    view = contract_view
    config_hash = view.artifacts.config_byte_sha256
    if (
        view.protocol.protocol_id != V021_PROTOCOL_ID
        or view.artifacts.protocol_id != V021_PROTOCOL_ID
        or view.protocol.config_sha256 != config_hash
        or not isinstance(config_hash, str)
        or _SHA256.fullmatch(config_hash) is None
    ):
        raise V021ProvenanceError("Validated V2.1 contract identity changed")
    return view


def _input_mapping(items: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(items)


def _validate_numeric_state(
    training_state: object,
    *,
    center_inputs: tuple[tuple[str, str], ...],
    risk_inputs: tuple[tuple[str, str], ...],
    calibration_inputs: tuple[tuple[str, str], ...],
) -> tuple[_v015.FrozenTrainingState, dict[str, Any]]:
    if type(training_state) is not _v015.FrozenTrainingState:
        raise V021ProvenanceError(
            "training_state must be an exact frozen numeric state"
        )
    try:
        payload = _v015.build_model_state_payload(
            training_state,
            center_development_input_hashes=_input_mapping(center_inputs),
            risk_development_input_hashes=_input_mapping(risk_inputs),
            calibration_input_hashes=_input_mapping(calibration_inputs),
            software_versions=_v015.default_software_versions(),
            created_utc="1970-01-01T00:00:00Z",
        )
    except (TypeError, ValueError) as exc:
        raise V021ProvenanceError("Numeric training state is invalid") from exc
    return training_state, payload


def _training_payload(
    *,
    protocol_id: str,
    config_sha256: str,
    attempt_id: str,
    commitments: tuple[tuple[str, str], ...],
    state_hashes: tuple[tuple[str, str], ...],
    audit: V021CalibrationAudit,
    input_hashes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": protocol_id,
        "config_sha256": config_sha256,
        "attempt_id": attempt_id,
        "commitment_byte_sha256": dict(commitments),
        "numeric_state_sha256": dict(state_hashes),
        "calibration_audit": _audit_payload(audit),
        "fresh_input_byte_sha256": {
            phase: dict(values) for phase, values in input_hashes
        },
    }


def _training_digest(payload: Mapping[str, object]) -> str:
    hasher = hashlib.sha256()
    hasher.update(_TRAINING_DOMAIN)
    _update_sized(hasher, _canonical_bytes(payload))
    return hasher.hexdigest()


class V021TrainingProvenanceEnvelope:
    """Immutable capability proving the complete pre-model V2.1 chain."""

    __slots__ = (
        "_audit",
        "_attempt_id",
        "_commitments",
        "_config_sha256",
        "_input_hashes",
        "_issuer_key",
        "_mask_commitment",
        "_protocol_id",
        "_provenance_sha256",
        "_state_hashes",
        "_training_state",
    )

    def __init__(
        self,
        *,
        _issuer_key: object,
        contract_view: V021ContractView,
        attempt_id: str,
        training_state: _v015.FrozenTrainingState,
        audit: V021CalibrationAudit,
        mask_commitment: V021PretruthMaskCommitment,
        commitments: tuple[tuple[str, str], ...],
        state_hashes: tuple[tuple[str, str], ...],
        input_hashes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
        provenance_sha256: str,
    ) -> None:
        if (
            _issuer_key is not _ISSUER_KEY
            or type(self) is not V021TrainingProvenanceEnvelope
        ):
            raise TypeError(
                "V021TrainingProvenanceEnvelope is issued only by the V2.1 "
                "training provenance factory"
            )
        object.__setattr__(self, "_issuer_key", _issuer_key)
        object.__setattr__(self, "_protocol_id", V021_PROTOCOL_ID)
        object.__setattr__(
            self,
            "_config_sha256",
            contract_view.artifacts.config_byte_sha256,
        )
        object.__setattr__(self, "_attempt_id", attempt_id)
        object.__setattr__(self, "_training_state", training_state)
        object.__setattr__(self, "_audit", audit)
        object.__setattr__(self, "_mask_commitment", mask_commitment)
        object.__setattr__(self, "_commitments", commitments)
        object.__setattr__(self, "_state_hashes", state_hashes)
        object.__setattr__(self, "_input_hashes", input_hashes)
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("V2.1 provenance envelopes are immutable")

    @property
    def protocol_id(self) -> str:
        return self._protocol_id

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def training_state(self) -> _v015.FrozenTrainingState:
        return self._training_state

    @property
    def calibration_audit(self) -> V021CalibrationAudit:
        return self._audit

    @property
    def mask_commitment(self) -> V021PretruthMaskCommitment:
        return self._mask_commitment

    @property
    def commitment_byte_sha256(self) -> dict[str, str]:
        return dict(self._commitments)

    @property
    def numeric_state_sha256(self) -> dict[str, str]:
        return dict(self._state_hashes)

    @property
    def input_byte_hashes(self) -> dict[str, dict[str, str]]:
        return {phase: dict(values) for phase, values in self._input_hashes}

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256


class V021ValidatedModelStateEnvelope:
    """Immutable capability proving bytes, but not yet their formal commit."""

    __slots__ = (
        "_decoded",
        "_issuer_key",
        "_model_state_byte_sha256",
        "_provenance_sha256",
        "_training_provenance",
    )

    def __init__(
        self,
        *,
        _issuer_key: object,
        training_provenance: V021TrainingProvenanceEnvelope,
        decoded: _v015.DecodedModelState,
        model_state_byte_sha256: str,
        provenance_sha256: str,
    ) -> None:
        if (
            _issuer_key is not _ISSUER_KEY
            or type(self) is not V021ValidatedModelStateEnvelope
        ):
            raise TypeError(
                "V021ValidatedModelStateEnvelope is issued only by the "
                "validated V2.1 model-state codec"
            )
        object.__setattr__(self, "_issuer_key", _issuer_key)
        object.__setattr__(self, "_training_provenance", training_provenance)
        object.__setattr__(self, "_decoded", decoded)
        object.__setattr__(
            self,
            "_model_state_byte_sha256",
            model_state_byte_sha256,
        )
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("V2.1 model-state envelopes are immutable")

    @property
    def protocol_id(self) -> str:
        return self._training_provenance.protocol_id

    @property
    def config_sha256(self) -> str:
        return self._training_provenance.config_sha256

    @property
    def attempt_id(self) -> str:
        return self._training_provenance.attempt_id

    @property
    def decoded_model_state(self) -> _v015.DecodedModelState:
        return self._decoded

    @property
    def training_provenance(self) -> V021TrainingProvenanceEnvelope:
        return self._training_provenance

    @property
    def model_state_byte_sha256(self) -> str:
        return self._model_state_byte_sha256

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256


class V021CommittedModelStateEnvelope:
    """Formal prediction capability issued after artifact and ledger commit."""

    __slots__ = (
        "_issuer_key",
        "_ledger_model_state_commitment_byte_sha256",
        "_model_state_commitment_artifact_byte_sha256",
        "_provenance_sha256",
        "_validated_model_state",
    )

    def __init__(
        self,
        *,
        _issuer_key: object,
        validated_model_state: V021ValidatedModelStateEnvelope,
        model_state_commitment_artifact_byte_sha256: str,
        ledger_model_state_commitment_byte_sha256: str,
        provenance_sha256: str,
    ) -> None:
        if (
            _issuer_key is not _ISSUER_KEY
            or type(self) is not V021CommittedModelStateEnvelope
        ):
            raise TypeError(
                "V021CommittedModelStateEnvelope is issued only after exact "
                "artifact and ledger commitment validation"
            )
        object.__setattr__(self, "_issuer_key", _issuer_key)
        object.__setattr__(self, "_validated_model_state", validated_model_state)
        object.__setattr__(
            self,
            "_model_state_commitment_artifact_byte_sha256",
            model_state_commitment_artifact_byte_sha256,
        )
        object.__setattr__(
            self,
            "_ledger_model_state_commitment_byte_sha256",
            ledger_model_state_commitment_byte_sha256,
        )
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("V2.1 committed model envelopes are immutable")

    @property
    def protocol_id(self) -> str:
        return self._validated_model_state.protocol_id

    @property
    def config_sha256(self) -> str:
        return self._validated_model_state.config_sha256

    @property
    def attempt_id(self) -> str:
        return self._validated_model_state.attempt_id

    @property
    def validated_model_state(self) -> V021ValidatedModelStateEnvelope:
        return self._validated_model_state

    @property
    def model_state_commitment_artifact_byte_sha256(self) -> str:
        return self._model_state_commitment_artifact_byte_sha256

    @property
    def ledger_model_state_commitment_byte_sha256(self) -> str:
        return self._ledger_model_state_commitment_byte_sha256

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256


def _require_training_envelope(
    value: object,
) -> V021TrainingProvenanceEnvelope:
    if (
        type(value) is not V021TrainingProvenanceEnvelope
        or value._issuer_key is not _ISSUER_KEY
    ):
        raise V021ProvenanceError(
            "An exact factory-issued V021TrainingProvenanceEnvelope is required"
        )
    payload = _training_payload(
        protocol_id=value.protocol_id,
        config_sha256=value.config_sha256,
        attempt_id=value.attempt_id,
        commitments=value._commitments,
        state_hashes=value._state_hashes,
        audit=value.calibration_audit,
        input_hashes=value._input_hashes,
    )
    if _training_digest(payload) != value.provenance_sha256:
        raise V021ProvenanceError("Training provenance envelope digest changed")
    return value


def _issue_v021_training_provenance_from_fresh_bytes(
    *,
    contract_view: V021ContractView,
    attempt_id: str,
    training_state: _v015.FrozenTrainingState,
    calibration_audit: V021CalibrationAudit,
    mask_commitment: V021PretruthMaskCommitment,
    generation_plan_commitment_bytes: bytes,
    truth_commitment_bytes: bytes,
    actual_analysis_hash_ledger_commitment_bytes: bytes,
    label_free_fit_commitment_bytes: bytes,
    center_state_commitment_bytes: bytes,
    risk_state_commitment_bytes: bytes,
    calibration_state_commitment_bytes: bytes,
    center_development_input_bytes: Mapping[str, bytes],
    risk_development_input_bytes: Mapping[str, bytes],
    calibration_input_bytes: Mapping[str, bytes],
) -> V021TrainingProvenanceEnvelope:
    """Package-private issuer used immediately on trusted training output."""

    view = _require_contract(contract_view)
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise V021ProvenanceError("attempt_id is not a canonical V2.1 attempt ID")

    center_inputs = _hash_exact_inputs(
        center_development_input_bytes,
        context="center_development_input_bytes",
    )
    risk_inputs = _hash_exact_inputs(
        risk_development_input_bytes,
        context="risk_development_input_bytes",
    )
    calibration_inputs = _hash_exact_inputs(
        calibration_input_bytes,
        context="calibration_input_bytes",
    )

    # Import lazily to keep the provenance module independent of codec import
    # order while still using the codec's exact mask/audit validators.
    from lifetwin.experiments import (  # noqa: PLC0415
        calendar_long_horizon_v016_state as _state_codec,
    )

    try:
        validated_mask = _state_codec.validate_calibration_mask_commitment_v021(
            mask_commitment
        )
        validated_audit = _state_codec._validate_calibration_audit(
            calibration_audit,
            mask_commitment=validated_mask,
            calibration_state=training_state.calibration,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise V021ProvenanceError(
            "Calibration output cannot be bound to the V2.1 mask"
        ) from exc
    mask_raw = _state_codec.serialize_calibration_mask_commitment_json_v021(
        validated_mask
    )
    observed_mask_inputs = dict(calibration_inputs)
    if observed_mask_inputs.get(_MASK_INPUT_NAME) != _sha256(mask_raw):
        raise V021ProvenanceError(
            "Exact calibration inputs do not contain the committed mask bytes"
        )

    validated_state, numeric_payload = _validate_numeric_state(
        training_state,
        center_inputs=center_inputs,
        risk_inputs=risk_inputs,
        calibration_inputs=calibration_inputs,
    )
    calibration_state_hash = _sha256(
        _canonical_bytes(numeric_payload["calibration_state"])
    )
    audit_hash = _sha256(_canonical_bytes(_audit_payload(validated_audit)))
    state_hashes = (
        ("center_state", _v015.center_state_sha256(validated_state.center)),
        ("risk_state", _v015.risk_state_sha256(validated_state.risk)),
        ("calibration_state", calibration_state_hash),
        ("calibration_audit", audit_hash),
    )
    commitment_raw = {
        "generation_plan": _exact_raw(
            generation_plan_commitment_bytes,
            context="generation_plan_commitment_bytes",
        ),
        "truth": _exact_raw(
            truth_commitment_bytes,
            context="truth_commitment_bytes",
        ),
        "actual_analysis_hash_ledger": _exact_raw(
            actual_analysis_hash_ledger_commitment_bytes,
            context="actual_analysis_hash_ledger_commitment_bytes",
        ),
        "label_free_fit": _exact_raw(
            label_free_fit_commitment_bytes,
            context="label_free_fit_commitment_bytes",
        ),
        "center_state": _exact_raw(
            center_state_commitment_bytes,
            context="center_state_commitment_bytes",
        ),
        "risk_state": _exact_raw(
            risk_state_commitment_bytes,
            context="risk_state_commitment_bytes",
        ),
        "calibration_mask": mask_raw,
        "calibration_state": _exact_raw(
            calibration_state_commitment_bytes,
            context="calibration_state_commitment_bytes",
        ),
    }
    commitments = tuple(
        (name, _sha256(commitment_raw[name])) for name in _COMMITMENT_NAMES
    )
    input_hashes = (
        ("center_development", center_inputs),
        ("risk_development", risk_inputs),
        ("calibration", calibration_inputs),
    )
    payload = _training_payload(
        protocol_id=V021_PROTOCOL_ID,
        config_sha256=view.artifacts.config_byte_sha256,
        attempt_id=attempt_id,
        commitments=commitments,
        state_hashes=state_hashes,
        audit=validated_audit,
        input_hashes=input_hashes,
    )
    digest = _training_digest(payload)
    return V021TrainingProvenanceEnvelope(
        _issuer_key=_ISSUER_KEY,
        contract_view=view,
        attempt_id=attempt_id,
        training_state=validated_state,
        audit=validated_audit,
        mask_commitment=validated_mask,
        commitments=commitments,
        state_hashes=state_hashes,
        input_hashes=input_hashes,
        provenance_sha256=digest,
    )


def _rehydrate_v021_training_provenance_after_strict_io(
    *,
    contract_view: V021ContractView,
    attempt_id: str,
    training_state: _v015.FrozenTrainingState,
    calibration_audit: V021CalibrationAudit,
    mask_commitment: V021PretruthMaskCommitment,
    generation_plan_commitment_bytes: bytes,
    truth_commitment_bytes: bytes,
    actual_analysis_hash_ledger_commitment_bytes: bytes,
    label_free_fit_commitment_bytes: bytes,
    center_state_commitment_bytes: bytes,
    risk_state_commitment_bytes: bytes,
    calibration_state_commitment_bytes: bytes,
    verified_input_byte_hashes: Mapping[str, Mapping[str, str]],
) -> V021TrainingProvenanceEnvelope:
    """Private rehydration hook for the strict, truth-incapable IO verifier.

    Unlike the fresh training issuer, this hook cannot reread opened training
    truth.  Its caller must have verified each declared truth hash through the
    ledger-bound ``truth_commitments.json`` and every label-free hash against a
    direct physical file.  It remains private and returns the same sealed type,
    so no public model or prediction API accepts caller-reported hashes.
    """

    view = _require_contract(contract_view)
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise V021ProvenanceError("attempt_id is not a canonical V2.1 attempt ID")
    if not isinstance(verified_input_byte_hashes, Mapping) or set(
        verified_input_byte_hashes
    ) != {"center_development", "risk_development", "calibration"}:
        raise V021ProvenanceError("Verified input phase registry changed")
    center_inputs = _validate_preverified_input_hashes(
        verified_input_byte_hashes["center_development"],
        context="verified center inputs",
    )
    risk_inputs = _validate_preverified_input_hashes(
        verified_input_byte_hashes["risk_development"],
        context="verified risk inputs",
    )
    calibration_inputs = _validate_preverified_input_hashes(
        verified_input_byte_hashes["calibration"],
        context="verified calibration inputs",
    )

    from lifetwin.experiments import (  # noqa: PLC0415
        calendar_long_horizon_v016_state as _state_codec,
    )

    try:
        validated_mask = _state_codec.validate_calibration_mask_commitment_v021(
            mask_commitment
        )
        validated_audit = _state_codec._validate_calibration_audit(
            calibration_audit,
            mask_commitment=validated_mask,
            calibration_state=training_state.calibration,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise V021ProvenanceError(
            "Calibration output cannot be rebound to the V2.1 mask"
        ) from exc
    mask_raw = _state_codec.serialize_calibration_mask_commitment_json_v021(
        validated_mask
    )
    if dict(calibration_inputs).get(_MASK_INPUT_NAME) != _sha256(mask_raw):
        raise V021ProvenanceError(
            "Verified calibration inputs do not bind the committed mask bytes"
        )
    validated_state, numeric_payload = _validate_numeric_state(
        training_state,
        center_inputs=center_inputs,
        risk_inputs=risk_inputs,
        calibration_inputs=calibration_inputs,
    )
    state_hashes = (
        ("center_state", _v015.center_state_sha256(validated_state.center)),
        ("risk_state", _v015.risk_state_sha256(validated_state.risk)),
        (
            "calibration_state",
            _sha256(_canonical_bytes(numeric_payload["calibration_state"])),
        ),
        (
            "calibration_audit",
            _sha256(_canonical_bytes(_audit_payload(validated_audit))),
        ),
    )
    commitment_raw = {
        "generation_plan": _exact_raw(
            generation_plan_commitment_bytes,
            context="generation_plan_commitment_bytes",
        ),
        "truth": _exact_raw(
            truth_commitment_bytes,
            context="truth_commitment_bytes",
        ),
        "actual_analysis_hash_ledger": _exact_raw(
            actual_analysis_hash_ledger_commitment_bytes,
            context="actual_analysis_hash_ledger_commitment_bytes",
        ),
        "label_free_fit": _exact_raw(
            label_free_fit_commitment_bytes,
            context="label_free_fit_commitment_bytes",
        ),
        "center_state": _exact_raw(
            center_state_commitment_bytes,
            context="center_state_commitment_bytes",
        ),
        "risk_state": _exact_raw(
            risk_state_commitment_bytes,
            context="risk_state_commitment_bytes",
        ),
        "calibration_mask": mask_raw,
        "calibration_state": _exact_raw(
            calibration_state_commitment_bytes,
            context="calibration_state_commitment_bytes",
        ),
    }
    commitments = tuple(
        (name, _sha256(commitment_raw[name])) for name in _COMMITMENT_NAMES
    )
    input_hashes = (
        ("center_development", center_inputs),
        ("risk_development", risk_inputs),
        ("calibration", calibration_inputs),
    )
    payload = _training_payload(
        protocol_id=V021_PROTOCOL_ID,
        config_sha256=view.artifacts.config_byte_sha256,
        attempt_id=attempt_id,
        commitments=commitments,
        state_hashes=state_hashes,
        audit=validated_audit,
        input_hashes=input_hashes,
    )
    return V021TrainingProvenanceEnvelope(
        _issuer_key=_ISSUER_KEY,
        contract_view=view,
        attempt_id=attempt_id,
        training_state=validated_state,
        audit=validated_audit,
        mask_commitment=validated_mask,
        commitments=commitments,
        state_hashes=state_hashes,
        input_hashes=input_hashes,
        provenance_sha256=_training_digest(payload),
    )


def _issue_validated_model_state_envelope_v021(
    *,
    training_provenance: V021TrainingProvenanceEnvelope,
    decoded: _v015.DecodedModelState,
    raw_model_state: bytes,
) -> V021ValidatedModelStateEnvelope:
    provenance = _require_training_envelope(training_provenance)
    if type(decoded) is not _v015.DecodedModelState:
        raise V021ProvenanceError("Decoded model state has an unexpected type")
    raw = _exact_raw(raw_model_state, context="raw_model_state")
    model_hash = _sha256(raw)
    hasher = hashlib.sha256()
    hasher.update(_MODEL_DOMAIN)
    hasher.update(bytes.fromhex(provenance.provenance_sha256))
    hasher.update(bytes.fromhex(model_hash))
    digest = hasher.hexdigest()
    return V021ValidatedModelStateEnvelope(
        _issuer_key=_ISSUER_KEY,
        training_provenance=provenance,
        decoded=decoded,
        model_state_byte_sha256=model_hash,
        provenance_sha256=digest,
    )


def _require_validated_model_state_envelope(
    value: object,
) -> V021ValidatedModelStateEnvelope:
    if (
        type(value) is not V021ValidatedModelStateEnvelope
        or value._issuer_key is not _ISSUER_KEY
    ):
        raise V021ProvenanceError(
            "An exact codec-issued V021ValidatedModelStateEnvelope is required"
        )
    provenance = _require_training_envelope(value.training_provenance)
    model_hash = value.model_state_byte_sha256
    if _SHA256.fullmatch(model_hash) is None:
        raise V021ProvenanceError("Model-state commitment is invalid")
    hasher = hashlib.sha256()
    hasher.update(_MODEL_DOMAIN)
    hasher.update(bytes.fromhex(provenance.provenance_sha256))
    hasher.update(bytes.fromhex(model_hash))
    if hasher.hexdigest() != value.provenance_sha256:
        raise V021ProvenanceError("Model-state provenance envelope digest changed")
    decoded = value.decoded_model_state
    if (
        type(decoded) is not _v015.DecodedModelState
        or decoded.training_state != provenance.training_state
        or decoded.input_byte_hashes != provenance.input_byte_hashes
    ):
        raise V021ProvenanceError("Decoded model state left its provenance chain")
    return value


def _decode_model_commitment(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise V021ProvenanceError(
                    f"Duplicate model-state commitment key: {key}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                V021ProvenanceError(f"Nonfinite model-state commitment value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V021ProvenanceError(
            "model_state_commitment.json is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict) or _v015.canonical_json_bytes(payload) != raw:
        raise V021ProvenanceError("model_state_commitment.json is not canonical JSON")
    return payload


def _issue_committed_model_state_envelope_v021(
    *,
    validated_model_state: V021ValidatedModelStateEnvelope,
    model_state_commitment_bytes: bytes,
    ledger_model_state_commitment_byte_sha256: str,
    committed_state_artifact_bytes: Mapping[str, bytes],
) -> V021CommittedModelStateEnvelope:
    """Package-private issuer after IO verifies the model-state phase."""

    validated = _require_validated_model_state_envelope(validated_model_state)
    commitment_raw = _exact_raw(
        model_state_commitment_bytes,
        context="model_state_commitment_bytes",
    )
    commitment_hash = _sha256(commitment_raw)
    if (
        not isinstance(ledger_model_state_commitment_byte_sha256, str)
        or _SHA256.fullmatch(ledger_model_state_commitment_byte_sha256) is None
        or ledger_model_state_commitment_byte_sha256 != commitment_hash
    ):
        raise V021ProvenanceError(
            "Ledger model-state phase hash does not match commitment bytes"
        )
    payload = _decode_model_commitment(commitment_raw)
    if set(payload) != {
        "protocol_id",
        "config_sha256",
        "git_commit",
        "files",
        "created_utc",
    }:
        raise V021ProvenanceError("Model-state commitment schema changed")
    if (
        payload["protocol_id"] != validated.protocol_id
        or payload["config_sha256"] != validated.config_sha256
    ):
        raise V021ProvenanceError("Model-state commitment identity changed")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise V021ProvenanceError("Model-state commitment files are missing")
    if not isinstance(committed_state_artifact_bytes, Mapping):
        raise V021ProvenanceError(
            "committed_state_artifact_bytes must be an exact byte mapping"
        )
    exact_artifacts: dict[str, bytes] = {}
    for name, raw in committed_state_artifact_bytes.items():
        if (
            not isinstance(name, str)
            or _INPUT_NAME.fullmatch(name) is None
            or "/" in name
        ):
            raise V021ProvenanceError("Committed state artifact name is invalid")
        exact_artifacts[name] = _exact_raw(
            raw,
            context=f"committed_state_artifact_bytes.{name}",
        )
    observed_names: list[str] = []
    for index, raw_entry in enumerate(files):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
            "path",
            "row_count",
            "byte_count",
            "byte_sha256",
        }:
            raise V021ProvenanceError(
                f"Model-state commitment file entry {index} changed"
            )
        name = raw_entry["path"]
        if (
            not isinstance(name, str)
            or name not in exact_artifacts
            or name in observed_names
        ):
            raise V021ProvenanceError("Model-state commitment file registry changed")
        artifact = exact_artifacts[name]
        if (
            type(raw_entry["row_count"]) is not int
            or raw_entry["row_count"] != 1
            or type(raw_entry["byte_count"]) is not int
            or raw_entry["byte_count"] != len(artifact)
            or raw_entry["byte_sha256"] != _sha256(artifact)
        ):
            raise V021ProvenanceError(f"Committed state artifact {name} changed")
        observed_names.append(name)
    if set(observed_names) != set(exact_artifacts):
        raise V021ProvenanceError(
            "Model-state commitment does not cover every supplied state artifact"
        )
    if (
        "model_state.json" not in exact_artifacts
        or _sha256(exact_artifacts["model_state.json"])
        != validated.model_state_byte_sha256
    ):
        raise V021ProvenanceError(
            "Model-state commitment does not bind the validated model bytes"
        )

    hasher = hashlib.sha256()
    hasher.update(_COMMITTED_MODEL_DOMAIN)
    hasher.update(bytes.fromhex(validated.provenance_sha256))
    hasher.update(bytes.fromhex(commitment_hash))
    digest = hasher.hexdigest()
    return V021CommittedModelStateEnvelope(
        _issuer_key=_ISSUER_KEY,
        validated_model_state=validated,
        model_state_commitment_artifact_byte_sha256=commitment_hash,
        ledger_model_state_commitment_byte_sha256=commitment_hash,
        provenance_sha256=digest,
    )


def _require_committed_model_state_envelope(
    value: object,
) -> V021CommittedModelStateEnvelope:
    if (
        type(value) is not V021CommittedModelStateEnvelope
        or value._issuer_key is not _ISSUER_KEY
    ):
        raise V021ProvenanceError(
            "An exact IO-issued V021CommittedModelStateEnvelope is required"
        )
    validated = _require_validated_model_state_envelope(value.validated_model_state)
    commitment_hash = value.model_state_commitment_artifact_byte_sha256
    if (
        _SHA256.fullmatch(commitment_hash) is None
        or value.ledger_model_state_commitment_byte_sha256 != commitment_hash
    ):
        raise V021ProvenanceError("Committed model-state phase binding changed")
    hasher = hashlib.sha256()
    hasher.update(_COMMITTED_MODEL_DOMAIN)
    hasher.update(bytes.fromhex(validated.provenance_sha256))
    hasher.update(bytes.fromhex(commitment_hash))
    if hasher.hexdigest() != value.provenance_sha256:
        raise V021ProvenanceError("Committed model-state envelope digest changed")
    return value


def _extract_label_free_state_for_formal_v021(
    value: object,
    *,
    config_sha256: str,
) -> _v015.FrozenLabelFreeState:
    committed = _require_committed_model_state_envelope(value)
    envelope = committed.validated_model_state
    if (
        envelope.protocol_id != V021_PROTOCOL_ID
        or envelope.config_sha256 != config_sha256
    ):
        raise V021ProvenanceError(
            "Model-state provenance does not match the active V2.1 contract"
        )
    return envelope.decoded_model_state.frozen_label_free_state


__all__ = [
    "V021CommittedModelStateEnvelope",
    "V021ProvenanceError",
    "V021TrainingProvenanceEnvelope",
    "V021ValidatedModelStateEnvelope",
]
