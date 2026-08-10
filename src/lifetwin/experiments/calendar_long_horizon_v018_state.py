"""Pure-memory V2.3 state, manifest, and calibration-audit codecs.

The inherited V2 numeric codec remains the sole validator for fitted numeric
state.  V2.3 translation is performed on detached in-memory objects and changes
only ``protocol_id`` and ``config_sha256``.  This module has no artifact-path,
dataframe, generation, or outcome-reading surface.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import re
import struct
from typing import Any, Mapping

from lifetwin.experiments import calendar_long_horizon_v015_training as _v015
from lifetwin.experiments.calendar_long_horizon_v018_contract import (
    V023ContractView,
)
from lifetwin.experiments.calendar_long_horizon_v018_protocol import (
    V023_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v018_provenance import (
    V023ProvenanceError,
    V023TrainingProvenanceEnvelope,
    V023ValidatedModelStateEnvelope,
    _issue_validated_model_state_envelope_v023,
    _require_training_envelope,
)
from lifetwin.experiments.calendar_long_horizon_v018_training import (
    V023_MINIMUM_ELIGIBLE_COUNT,
    V023CalibrationAudit,
    V023CommittedMaskRow,
    V023PretruthMaskCommitment,
)


class V023StateCodecError(ValueError):
    """Raised when V2.3 state or manifest bytes fail closed."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_IDENTITY_FIELDS = frozenset({"protocol_id", "config_sha256"})
_MASK_INPUT_NAME = "calibration_mask_commitment.json"
_MASK_SCHEMA_VERSION = "1.0.0"
_AUDIT_SCHEMA_VERSION = "1.0.0"
_MASK_HASH_DOMAIN = b"lifetwin-v023-calibration-mask-v2\0"
_MASK_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "source_calibration_count",
        "risk_isotonic_eligible_count",
        "eligibility_mask_sha256",
        "rows",
    }
)
_MASK_ROW_KEYS = frozenset(
    {
        "cluster_id",
        "label_free_row_sha256",
        "structural_support_sha256",
        "successful_structure_family_ids",
        "eligible",
        "ineligibility_reasons",
    }
)
_REQUIRED_AUDIT_COUNT_FIELDS = (
    "source_calibration_count",
    "risk_isotonic_eligible_count",
    "risk_isotonic_ineligible_zero_family_count",
    "risk_isotonic_ineligible_one_family_count",
    "risk_isotonic_ineligible_other_count",
    "risk_isotonic_positive_label_count",
    "risk_isotonic_negative_label_count",
    "mean_baseline_count",
    "conformal_calibration_count",
    "conformal_order_statistic_index",
)
_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "config_sha256",
        *_REQUIRED_AUDIT_COUNT_FIELDS,
        "eligibility_mask_cluster_ids",
        "eligibility_mask",
        "eligibility_mask_sha256",
        "calibration_mask_commitment_byte_sha256",
        "isotonic_state_sha256",
        "conformal_state_sha256",
        "selected_mean_baseline",
        "created_utc",
    }
)


@dataclass(frozen=True, slots=True)
class DecodedV023CalibrationPopulationAudit:
    """Validated in-memory view of ``calibration_population_audit.json``."""

    protocol_id: str
    config_sha256: str
    audit: V023CalibrationAudit
    eligibility_mask_cluster_ids: tuple[str, ...]
    eligibility_mask: tuple[bool, ...]
    calibration_mask_commitment_byte_sha256: str
    isotonic_state_sha256: str
    conformal_state_sha256: str
    selected_mean_baseline: str
    created_utc: str


def _require_contract_identity(contract_view: object) -> tuple[str, str]:
    if type(contract_view) is not V023ContractView:
        raise V023StateCodecError(
            "contract_view must be an exact validated V023ContractView"
        )
    view = contract_view
    if (
        view.protocol.protocol_id != V023_PROTOCOL_ID
        or view.artifacts.protocol_id != V023_PROTOCOL_ID
    ):
        raise V023StateCodecError("Validated V2.3 protocol identity changed")
    amendment_hash = view.artifacts.config_byte_sha256
    if (
        not isinstance(amendment_hash, str)
        or _SHA256.fullmatch(amendment_hash) is None
        or view.protocol.config_sha256 != amendment_hash
    ):
        raise V023StateCodecError("Validated V2.3 amendment byte hash changed")
    return V023_PROTOCOL_ID, amendment_hash


def _require_bound_identity(
    *,
    protocol_id: object,
    config_sha256: object,
) -> tuple[str, str]:
    """Validate the scalar identity retained by a sealed provenance envelope."""

    if protocol_id != V023_PROTOCOL_ID:
        raise V023StateCodecError("Validated V2.3 protocol identity changed")
    if not isinstance(config_sha256, str) or _SHA256.fullmatch(config_sha256) is None:
        raise V023StateCodecError("Validated V2.3 amendment byte hash changed")
    return V023_PROTOCOL_ID, config_sha256


def _exact_object(
    value: object,
    *,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise V023StateCodecError(f"{context} must be a JSON object")
    if set(value) != expected:
        raise V023StateCodecError(
            f"{context} keys changed: observed={sorted(value)}, "
            f"expected={sorted(expected)}"
        )
    return value


def _digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V023StateCodecError(f"{context} must be a lowercase SHA256")
    return value


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V023StateCodecError(f"{context} must be an integer")
    return value


def _created_utc(value: object) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise V023StateCodecError("created_utc must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise V023StateCodecError("created_utc is not a valid timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise V023StateCodecError("created_utc must use UTC")
    return value


def _canonical_pretty_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return _v015.canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Payload is not finite canonical JSON") from exc


def _canonical_compact_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise V023StateCodecError("Payload is not finite canonical JSON") from exc
    return encoded + b"\n"


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V023StateCodecError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise V023StateCodecError(f"Nonfinite JSON constant is forbidden: {token}")


def _decode_canonical_object(
    raw: bytes,
    *,
    filename: str,
    compact: bool,
) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise V023StateCodecError(f"{filename} input must be exact bytes")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V023StateCodecError(f"{filename} is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise V023StateCodecError(f"{filename} root must be an object")
    canonical = (
        _canonical_compact_bytes(decoded)
        if compact
        else _canonical_pretty_bytes(decoded)
    )
    if canonical != raw:
        raise V023StateCodecError(f"{filename} bytes are not canonical")
    return decoded


def _require_current_identity(
    payload: Mapping[str, Any],
    *,
    contract_view: V023ContractView,
    context: str,
) -> None:
    protocol_id, amendment_hash = _require_contract_identity(contract_view)
    if payload.get("protocol_id") != protocol_id:
        raise V023StateCodecError(f"{context} protocol_id changed")
    if payload.get("config_sha256") != amendment_hash:
        raise V023StateCodecError(
            f"{context} config_sha256 is not the validated amendment byte hash"
        )


def _require_current_bound_identity(
    payload: Mapping[str, Any],
    *,
    protocol_id: object,
    config_sha256: object,
    context: str,
) -> None:
    validated_protocol_id, validated_config_sha256 = _require_bound_identity(
        protocol_id=protocol_id,
        config_sha256=config_sha256,
    )
    if payload.get("protocol_id") != validated_protocol_id:
        raise V023StateCodecError(f"{context} protocol_id changed")
    if payload.get("config_sha256") != validated_config_sha256:
        raise V023StateCodecError(
            f"{context} config_sha256 is not the validated amendment byte hash"
        )


def _translate_identity(
    payload: Mapping[str, Any],
    *,
    source_protocol_id: str,
    source_config_sha256: str,
    target_protocol_id: str,
    target_config_sha256: str,
    context: str,
) -> dict[str, Any]:
    if payload.get("protocol_id") != source_protocol_id:
        raise V023StateCodecError(f"{context} source protocol identity changed")
    if payload.get("config_sha256") != source_config_sha256:
        raise V023StateCodecError(f"{context} source config identity changed")
    translated = dict(payload)
    translated["protocol_id"] = target_protocol_id
    translated["config_sha256"] = target_config_sha256
    changed = {
        key
        for key in translated
        if key not in payload or translated[key] != payload[key]
    }
    if changed != _IDENTITY_FIELDS or set(translated) != set(payload):
        raise V023StateCodecError(
            f"{context} translation changed fields other than identity"
        )
    return translated


def _v2_to_v023(
    payload: Mapping[str, Any],
    *,
    contract_view: V023ContractView,
    context: str,
) -> dict[str, Any]:
    protocol_id, amendment_hash = _require_contract_identity(contract_view)
    return _translate_identity(
        payload,
        source_protocol_id=_v015.FROZEN_PROTOCOL_ID,
        source_config_sha256=_v015.FROZEN_CONFIG_BYTE_SHA256,
        target_protocol_id=protocol_id,
        target_config_sha256=amendment_hash,
        context=context,
    )


def _v023_to_v2(
    payload: Mapping[str, Any],
    *,
    contract_view: V023ContractView,
    context: str,
) -> dict[str, Any]:
    protocol_id, amendment_hash = _require_contract_identity(contract_view)
    return _translate_identity(
        payload,
        source_protocol_id=protocol_id,
        source_config_sha256=amendment_hash,
        target_protocol_id=_v015.FROZEN_PROTOCOL_ID,
        target_config_sha256=_v015.FROZEN_CONFIG_BYTE_SHA256,
        context=context,
    )


def _v2_to_v023_bound(
    payload: Mapping[str, Any],
    *,
    protocol_id: object,
    config_sha256: object,
    context: str,
) -> dict[str, Any]:
    validated_protocol_id, validated_config_sha256 = _require_bound_identity(
        protocol_id=protocol_id,
        config_sha256=config_sha256,
    )
    return _translate_identity(
        payload,
        source_protocol_id=_v015.FROZEN_PROTOCOL_ID,
        source_config_sha256=_v015.FROZEN_CONFIG_BYTE_SHA256,
        target_protocol_id=validated_protocol_id,
        target_config_sha256=validated_config_sha256,
        context=context,
    )


def _v023_to_v2_bound(
    payload: Mapping[str, Any],
    *,
    protocol_id: object,
    config_sha256: object,
    context: str,
) -> dict[str, Any]:
    validated_protocol_id, validated_config_sha256 = _require_bound_identity(
        protocol_id=protocol_id,
        config_sha256=config_sha256,
    )
    return _translate_identity(
        payload,
        source_protocol_id=validated_protocol_id,
        source_config_sha256=validated_config_sha256,
        target_protocol_id=_v015.FROZEN_PROTOCOL_ID,
        target_config_sha256=_v015.FROZEN_CONFIG_BYTE_SHA256,
        context=context,
    )


def _update_sized(hasher: Any, payload: bytes) -> None:
    hasher.update(struct.pack("<Q", len(payload)))
    hasher.update(payload)


def _mask_digest(
    rows: tuple[V023CommittedMaskRow, ...],
    *,
    source_count: int,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_MASK_HASH_DOMAIN)
    hasher.update(struct.pack("<Q", source_count))
    for row in rows:
        _update_sized(hasher, row.cluster_id.encode("ascii"))
        hasher.update(bytes.fromhex(row.label_free_row_sha256))
        hasher.update(struct.pack("<B", int(row.eligible)))
    return hasher.hexdigest()


def _validate_committed_row(
    row: object,
    *,
    index: int,
) -> V023CommittedMaskRow:
    if type(row) is not V023CommittedMaskRow:
        raise V023StateCodecError(f"Mask row {index} has an invalid type")
    try:
        normalized = V023CommittedMaskRow(
            cluster_id=row.cluster_id,
            label_free_row_sha256=row.label_free_row_sha256,
            structural_support_sha256=row.structural_support_sha256,
            successful_structure_family_ids=row.successful_structure_family_ids,
            eligible=row.eligible,
            ineligibility_reasons=row.ineligibility_reasons,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError(f"Mask row {index} is invalid") from exc
    if normalized != row:
        raise V023StateCodecError(f"Mask row {index} is not canonical")
    return normalized


def validate_calibration_mask_commitment_v023(
    commitment: V023PretruthMaskCommitment,
) -> V023PretruthMaskCommitment:
    """Recompute every count, order constraint, and mask digest."""

    if type(commitment) is not V023PretruthMaskCommitment:
        raise V023StateCodecError(
            "commitment must be an exact V023PretruthMaskCommitment"
        )
    rows = tuple(
        _validate_committed_row(row, index=index)
        for index, row in enumerate(commitment.rows)
    )
    if (
        commitment.protocol_id != V023_PROTOCOL_ID
        or commitment.source_calibration_count != _v015.CALIBRATION_COUNT
        or len(rows) != commitment.source_calibration_count
    ):
        raise V023StateCodecError("Mask protocol identity or source count changed")
    identifiers = tuple(row.cluster_id for row in rows)
    if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(
        identifiers
    ):
        raise V023StateCodecError("Mask rows are reordered or duplicated")
    eligible_count = sum(row.eligible for row in rows)
    if eligible_count < V023_MINIMUM_ELIGIBLE_COUNT:
        raise V023StateCodecError("Mask eligible count is below the V2.3 minimum")
    expected_digest = _mask_digest(
        rows, source_count=commitment.source_calibration_count
    )
    if commitment.eligibility_mask_sha256 != expected_digest:
        raise V023StateCodecError("Mask eligibility digest does not match its rows")
    try:
        normalized = V023PretruthMaskCommitment(
            protocol_id=commitment.protocol_id,
            source_calibration_count=commitment.source_calibration_count,
            rows=rows,
            eligibility_mask_sha256=expected_digest,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Mask commitment is invalid") from exc
    if normalized != commitment or normalized.eligible_count != eligible_count:
        raise V023StateCodecError("Mask commitment is not canonical")
    return normalized


def build_calibration_mask_commitment_v023(
    *,
    rows: tuple[V023CommittedMaskRow, ...],
) -> V023PretruthMaskCommitment:
    """Build the immutable commitment from already-computed label-free rows."""

    if not isinstance(rows, tuple) or any(
        type(row) is not V023CommittedMaskRow for row in rows
    ):
        raise V023StateCodecError("rows must be a tuple of committed mask rows")
    canonical_rows = tuple(sorted(rows, key=lambda row: row.cluster_id))
    digest = _mask_digest(canonical_rows, source_count=len(canonical_rows))
    try:
        commitment = V023PretruthMaskCommitment(
            protocol_id=V023_PROTOCOL_ID,
            source_calibration_count=len(canonical_rows),
            rows=canonical_rows,
            eligibility_mask_sha256=digest,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Committed mask rows are invalid") from exc
    return validate_calibration_mask_commitment_v023(commitment)


def serialize_calibration_mask_commitment_json_v023(
    commitment: V023PretruthMaskCommitment,
) -> bytes:
    """Return the strict compact canonical commitment bytes."""

    validated = validate_calibration_mask_commitment_v023(commitment)
    raw = validated.canonical_bytes()
    if deserialize_calibration_mask_commitment_json_v023(raw) != validated:
        raise V023StateCodecError("Mask commitment failed its roundtrip")
    return raw


def deserialize_calibration_mask_commitment_json_v023(
    raw: bytes,
) -> V023PretruthMaskCommitment:
    """Decode canonical bytes and recompute the mask from all ordered rows."""

    payload = _decode_canonical_object(
        raw,
        filename=_MASK_INPUT_NAME,
        compact=True,
    )
    top = _exact_object(
        payload,
        expected=_MASK_TOP_LEVEL_KEYS,
        context=_MASK_INPUT_NAME,
    )
    if top["schema_version"] != _MASK_SCHEMA_VERSION:
        raise V023StateCodecError("Mask commitment schema changed")
    if top["protocol_id"] != V023_PROTOCOL_ID:
        raise V023StateCodecError("Mask commitment protocol_id changed")
    source_count = _integer(
        top["source_calibration_count"],
        context="mask source_calibration_count",
    )
    declared_eligible = _integer(
        top["risk_isotonic_eligible_count"],
        context="mask risk_isotonic_eligible_count",
    )
    declared_digest = _digest(
        top["eligibility_mask_sha256"],
        context="mask eligibility_mask_sha256",
    )
    row_payloads = top["rows"]
    if not isinstance(row_payloads, list):
        raise V023StateCodecError("Mask rows must be a JSON array")
    rows: list[V023CommittedMaskRow] = []
    for index, value in enumerate(row_payloads):
        row = _exact_object(
            value,
            expected=_MASK_ROW_KEYS,
            context=f"mask rows[{index}]",
        )
        family_ids = row["successful_structure_family_ids"]
        reasons = row["ineligibility_reasons"]
        if not isinstance(family_ids, list) or any(
            not isinstance(item, str) for item in family_ids
        ):
            raise V023StateCodecError(f"mask rows[{index}] family IDs are invalid")
        if not isinstance(reasons, list) or any(
            not isinstance(item, str) for item in reasons
        ):
            raise V023StateCodecError(f"mask rows[{index}] reasons are invalid")
        if not isinstance(row["eligible"], bool):
            raise V023StateCodecError(
                f"mask rows[{index}] eligible must be a strict boolean"
            )
        try:
            rows.append(
                V023CommittedMaskRow(
                    cluster_id=row["cluster_id"],
                    label_free_row_sha256=row["label_free_row_sha256"],
                    structural_support_sha256=row["structural_support_sha256"],
                    successful_structure_family_ids=tuple(family_ids),
                    eligible=row["eligible"],
                    ineligibility_reasons=tuple(reasons),
                )
            )
        except (TypeError, ValueError) as exc:
            raise V023StateCodecError(f"mask rows[{index}] is invalid") from exc
    try:
        commitment = V023PretruthMaskCommitment(
            protocol_id=top["protocol_id"],
            source_calibration_count=source_count,
            rows=tuple(rows),
            eligibility_mask_sha256=declared_digest,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Mask row count or order is invalid") from exc
    validated = validate_calibration_mask_commitment_v023(commitment)
    if declared_eligible != validated.eligible_count:
        raise V023StateCodecError("Declared mask eligible count is false")
    if validated.canonical_bytes() != raw:
        raise V023StateCodecError("Mask commitment does not roundtrip canonically")
    return validated


def _mask_byte_sha256(commitment: V023PretruthMaskCommitment) -> str:
    raw = serialize_calibration_mask_commitment_json_v023(commitment)
    return hashlib.sha256(raw).hexdigest()


def _require_mask_input_hash(
    calibration_input_hashes: object,
    *,
    mask_commitment: V023PretruthMaskCommitment,
) -> None:
    if not isinstance(calibration_input_hashes, Mapping):
        raise V023StateCodecError("calibration_input_hashes must be an object")
    expected = _mask_byte_sha256(mask_commitment)
    if _MASK_INPUT_NAME not in calibration_input_hashes:
        raise V023StateCodecError(
            "calibration_input_hashes must bind calibration_mask_commitment.json"
        )
    if calibration_input_hashes[_MASK_INPUT_NAME] != expected:
        raise V023StateCodecError(
            "calibration_mask_commitment.json input hash is incorrect"
        )


def build_model_state_payload_v023(
    provenance_envelope: V023TrainingProvenanceEnvelope,
    *,
    software_versions: Mapping[str, str],
    created_utc: str,
) -> dict[str, Any]:
    """Build state only from a factory-issued fresh V2.3 provenance chain."""

    try:
        provenance = _require_training_envelope(provenance_envelope)
    except V023ProvenanceError as exc:
        raise V023StateCodecError(str(exc)) from exc
    input_hashes = provenance.input_byte_hashes
    try:
        inherited = _v015.build_model_state_payload(
            provenance.training_state,
            center_development_input_hashes=input_hashes["center_development"],
            risk_development_input_hashes=input_hashes["risk_development"],
            calibration_input_hashes=input_hashes["calibration"],
            software_versions=software_versions,
            created_utc=created_utc,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Inherited model state is invalid") from exc
    return _v2_to_v023_bound(
        inherited,
        protocol_id=provenance.protocol_id,
        config_sha256=provenance.config_sha256,
        context="model_state.json",
    )


def serialize_model_state_json_v023(
    provenance_envelope: V023TrainingProvenanceEnvelope,
    *,
    software_versions: Mapping[str, str],
    created_utc: str,
) -> bytes:
    """Serialize ``model_state.json`` from sealed provenance, never raw hashes."""

    payload = build_model_state_payload_v023(
        provenance_envelope,
        software_versions=software_versions,
        created_utc=created_utc,
    )
    return _canonical_pretty_bytes(payload)


def deserialize_model_state_json_v023(
    raw: bytes,
    *,
    provenance_envelope: V023TrainingProvenanceEnvelope,
) -> V023ValidatedModelStateEnvelope:
    """Validate bytes against sealed provenance and issue a formal capability."""

    try:
        provenance = _require_training_envelope(provenance_envelope)
    except V023ProvenanceError as exc:
        raise V023StateCodecError(str(exc)) from exc
    payload = _decode_canonical_object(
        raw,
        filename="model_state.json",
        compact=False,
    )
    _require_current_bound_identity(
        payload,
        protocol_id=provenance.protocol_id,
        config_sha256=provenance.config_sha256,
        context="model_state.json",
    )
    inherited = _v023_to_v2_bound(
        payload,
        protocol_id=provenance.protocol_id,
        config_sha256=provenance.config_sha256,
        context="model_state.json",
    )
    try:
        decoded = _v015.validate_model_state_payload(inherited)
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError(
            f"V2.3 model state is numerically invalid: {exc}"
        ) from exc
    if decoded.training_state != provenance.training_state:
        raise V023StateCodecError(
            "V2.3 numeric state does not match its training provenance"
        )
    if decoded.input_byte_hashes != provenance.input_byte_hashes:
        raise V023StateCodecError(
            "V2.3 input hashes do not match fresh bytes in its provenance"
        )
    normalized = build_model_state_payload_v023(
        provenance,
        software_versions=decoded.software_versions,
        created_utc=decoded.created_utc,
    )
    if _canonical_pretty_bytes(normalized) != raw:
        raise V023StateCodecError("V2.3 model state is not canonical numeric state")
    try:
        return _issue_validated_model_state_envelope_v023(
            training_provenance=provenance,
            decoded=decoded,
            raw_model_state=raw,
        )
    except V023ProvenanceError as exc:
        raise V023StateCodecError(str(exc)) from exc


def build_training_manifest_payload_v023(
    *,
    center_development_input_hashes: Mapping[str, str],
    risk_development_input_hashes: Mapping[str, str],
    center_state: _v015.CenterDevelopmentState,
    risk_state: _v015.RiskDevelopmentState,
    created_utc: str,
    contract_view: V023ContractView,
) -> dict[str, Any]:
    """Build the inherited training manifest with V2.3 external identity."""

    try:
        inherited = _v015.build_training_manifest(
            center_development_input_hashes=center_development_input_hashes,
            risk_development_input_hashes=risk_development_input_hashes,
            center_state=center_state,
            risk_state=risk_state,
            created_utc=created_utc,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Inherited training manifest is invalid") from exc
    return _v2_to_v023(
        inherited,
        contract_view=contract_view,
        context="training_manifest.json",
    )


def serialize_training_manifest_json_v023(
    *,
    center_development_input_hashes: Mapping[str, str],
    risk_development_input_hashes: Mapping[str, str],
    center_state: _v015.CenterDevelopmentState,
    risk_state: _v015.RiskDevelopmentState,
    created_utc: str,
    contract_view: V023ContractView,
) -> bytes:
    return _canonical_pretty_bytes(
        build_training_manifest_payload_v023(
            center_development_input_hashes=center_development_input_hashes,
            risk_development_input_hashes=risk_development_input_hashes,
            center_state=center_state,
            risk_state=risk_state,
            created_utc=created_utc,
            contract_view=contract_view,
        )
    )


def deserialize_training_manifest_json_v023(
    raw: bytes,
    *,
    center_state: _v015.CenterDevelopmentState,
    risk_state: _v015.RiskDevelopmentState,
    contract_view: V023ContractView,
) -> dict[str, Any]:
    payload = _decode_canonical_object(
        raw,
        filename="training_manifest.json",
        compact=False,
    )
    _require_current_identity(
        payload,
        contract_view=contract_view,
        context="training_manifest.json",
    )
    inherited = _v023_to_v2(
        payload,
        contract_view=contract_view,
        context="training_manifest.json",
    )
    try:
        _v015.validate_training_manifest(inherited)
        _v015.verify_training_manifest_state_hashes(
            inherited,
            center_state=center_state,
            risk_state=risk_state,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("V2.3 training manifest is invalid") from exc
    normalized = build_training_manifest_payload_v023(
        center_development_input_hashes=inherited["center_development_input_hashes"],
        risk_development_input_hashes=inherited["risk_development_input_hashes"],
        center_state=center_state,
        risk_state=risk_state,
        created_utc=inherited["created_utc"],
        contract_view=contract_view,
    )
    if _canonical_pretty_bytes(normalized) != raw:
        raise V023StateCodecError("V2.3 training manifest is not canonical state")
    return payload


def build_calibration_manifest_payload_v023(
    *,
    calibration_input_hashes: Mapping[str, str],
    calibration_state: _v015.CalibrationDevelopmentState,
    created_utc: str,
    mask_commitment: V023PretruthMaskCommitment,
    contract_view: V023ContractView,
) -> dict[str, Any]:
    """Build the inherited calibration manifest with an explicit mask input."""

    _require_mask_input_hash(
        calibration_input_hashes,
        mask_commitment=mask_commitment,
    )
    try:
        inherited = _v015.build_calibration_manifest(
            calibration_input_hashes=calibration_input_hashes,
            calibration_state=calibration_state,
            created_utc=created_utc,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Inherited calibration manifest is invalid") from exc
    return _v2_to_v023(
        inherited,
        contract_view=contract_view,
        context="calibration_manifest.json",
    )


def serialize_calibration_manifest_json_v023(
    *,
    calibration_input_hashes: Mapping[str, str],
    calibration_state: _v015.CalibrationDevelopmentState,
    created_utc: str,
    mask_commitment: V023PretruthMaskCommitment,
    contract_view: V023ContractView,
) -> bytes:
    return _canonical_pretty_bytes(
        build_calibration_manifest_payload_v023(
            calibration_input_hashes=calibration_input_hashes,
            calibration_state=calibration_state,
            created_utc=created_utc,
            mask_commitment=mask_commitment,
            contract_view=contract_view,
        )
    )


def deserialize_calibration_manifest_json_v023(
    raw: bytes,
    *,
    calibration_state: _v015.CalibrationDevelopmentState,
    mask_commitment: V023PretruthMaskCommitment,
    contract_view: V023ContractView,
) -> dict[str, Any]:
    payload = _decode_canonical_object(
        raw,
        filename="calibration_manifest.json",
        compact=False,
    )
    _require_current_identity(
        payload,
        contract_view=contract_view,
        context="calibration_manifest.json",
    )
    inherited = _v023_to_v2(
        payload,
        contract_view=contract_view,
        context="calibration_manifest.json",
    )
    try:
        _v015.validate_calibration_manifest(inherited)
        _v015.verify_calibration_manifest_state_hashes(
            inherited,
            calibration_state=calibration_state,
        )
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("V2.3 calibration manifest is invalid") from exc
    _require_mask_input_hash(
        inherited["calibration_input_hashes"],
        mask_commitment=mask_commitment,
    )
    normalized = build_calibration_manifest_payload_v023(
        calibration_input_hashes=inherited["calibration_input_hashes"],
        calibration_state=calibration_state,
        created_utc=inherited["created_utc"],
        mask_commitment=mask_commitment,
        contract_view=contract_view,
    )
    if _canonical_pretty_bytes(normalized) != raw:
        raise V023StateCodecError("V2.3 calibration manifest is not canonical state")
    return payload


def _validate_calibration_audit(
    audit: V023CalibrationAudit,
    *,
    mask_commitment: V023PretruthMaskCommitment,
    calibration_state: _v015.CalibrationDevelopmentState,
) -> V023CalibrationAudit:
    if type(audit) is not V023CalibrationAudit:
        raise V023StateCodecError("audit must be an exact V023CalibrationAudit")
    validated_mask = validate_calibration_mask_commitment_v023(mask_commitment)
    if tuple(field.name for field in fields(audit)) != (
        *_REQUIRED_AUDIT_COUNT_FIELDS,
        "eligibility_mask_sha256",
    ):
        raise V023StateCodecError("V2.3 calibration audit field registry changed")
    counts = {
        name: _integer(getattr(audit, name), context=f"audit.{name}")
        for name in _REQUIRED_AUDIT_COUNT_FIELDS
    }
    source_count = counts["source_calibration_count"]
    eligible_count = counts["risk_isotonic_eligible_count"]
    zero_count = counts["risk_isotonic_ineligible_zero_family_count"]
    one_count = counts["risk_isotonic_ineligible_one_family_count"]
    other_count = counts["risk_isotonic_ineligible_other_count"]
    positive_count = counts["risk_isotonic_positive_label_count"]
    negative_count = counts["risk_isotonic_negative_label_count"]
    if (
        source_count != _v015.CALIBRATION_COUNT
        or eligible_count != validated_mask.eligible_count
        or eligible_count < V023_MINIMUM_ELIGIBLE_COUNT
        or zero_count < 0
        or one_count < 0
        or other_count < 0
        or eligible_count + zero_count + one_count + other_count != source_count
        or positive_count < _v015.MINIMUM_CLASS_COUNT
        or negative_count < _v015.MINIMUM_CLASS_COUNT
        or positive_count + negative_count != eligible_count
    ):
        raise V023StateCodecError("V2.3 calibration audit risk counts disagree")
    expected_zero = sum(
        not row.eligible and len(row.successful_structure_family_ids) == 0
        for row in validated_mask.rows
    )
    expected_one = sum(
        not row.eligible and len(row.successful_structure_family_ids) == 1
        for row in validated_mask.rows
    )
    expected_other = sum(
        not row.eligible and len(row.successful_structure_family_ids) >= 2
        for row in validated_mask.rows
    )
    if (zero_count, one_count, other_count) != (
        expected_zero,
        expected_one,
        expected_other,
    ):
        raise V023StateCodecError("V2.3 ineligible-family counts disagree with mask")
    if (
        calibration_state.calibration_cluster_count != source_count
        or calibration_state.positive_label_count
        + calibration_state.negative_label_count
        != source_count
        or counts["mean_baseline_count"] != source_count
        or counts["conformal_calibration_count"]
        != calibration_state.conformal.calibration_count
        or counts["conformal_order_statistic_index"]
        != calibration_state.conformal.order_statistic_index
        or audit.eligibility_mask_sha256 != validated_mask.eligibility_mask_sha256
    ):
        raise V023StateCodecError("V2.3 calibration audit state bindings disagree")
    try:
        _v015.isotonic_state_sha256(calibration_state)
        _v015.conformal_state_sha256(calibration_state)
    except (TypeError, ValueError) as exc:
        raise V023StateCodecError("Calibration state is invalid") from exc
    return audit


def build_calibration_population_audit_payload_v023(
    *,
    provenance_envelope: V023TrainingProvenanceEnvelope,
    created_utc: str,
) -> dict[str, Any]:
    """Build the audit from the immutable calibration-stage provenance."""

    try:
        provenance = _require_training_envelope(provenance_envelope)
    except V023ProvenanceError as exc:
        raise V023StateCodecError(str(exc)) from exc
    audit = provenance.calibration_audit
    calibration_state = provenance.training_state.calibration
    mask_commitment = provenance.mask_commitment
    _validate_calibration_audit(
        audit,
        mask_commitment=mask_commitment,
        calibration_state=calibration_state,
    )
    protocol_id, amendment_hash = _require_bound_identity(
        protocol_id=provenance.protocol_id,
        config_sha256=provenance.config_sha256,
    )
    validated_mask = validate_calibration_mask_commitment_v023(mask_commitment)
    payload: dict[str, Any] = {
        "schema_version": _AUDIT_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "config_sha256": amendment_hash,
        **{name: getattr(audit, name) for name in _REQUIRED_AUDIT_COUNT_FIELDS},
        "eligibility_mask_cluster_ids": [row.cluster_id for row in validated_mask.rows],
        "eligibility_mask": [row.eligible for row in validated_mask.rows],
        "eligibility_mask_sha256": validated_mask.eligibility_mask_sha256,
        "calibration_mask_commitment_byte_sha256": _mask_byte_sha256(validated_mask),
        "isotonic_state_sha256": _v015.isotonic_state_sha256(calibration_state),
        "conformal_state_sha256": _v015.conformal_state_sha256(calibration_state),
        "selected_mean_baseline": calibration_state.selected_mean_baseline,
        "created_utc": _created_utc(created_utc),
    }
    _validate_calibration_population_audit_payload(
        payload,
        provenance_envelope=provenance,
    )
    return payload


def _validate_calibration_population_audit_payload(
    payload: object,
    *,
    provenance_envelope: V023TrainingProvenanceEnvelope,
) -> DecodedV023CalibrationPopulationAudit:
    try:
        provenance = _require_training_envelope(provenance_envelope)
    except V023ProvenanceError as exc:
        raise V023StateCodecError(str(exc)) from exc
    calibration_state = provenance.training_state.calibration
    mask_commitment = provenance.mask_commitment
    top = _exact_object(
        payload,
        expected=_AUDIT_KEYS,
        context="calibration_population_audit.json",
    )
    if top["schema_version"] != _AUDIT_SCHEMA_VERSION:
        raise V023StateCodecError("Calibration audit schema changed")
    _require_current_bound_identity(
        top,
        protocol_id=provenance.protocol_id,
        config_sha256=provenance.config_sha256,
        context="calibration_population_audit.json",
    )
    audit = V023CalibrationAudit(
        **{
            name: _integer(top[name], context=f"calibration audit {name}")
            for name in _REQUIRED_AUDIT_COUNT_FIELDS
        },
        eligibility_mask_sha256=_digest(
            top["eligibility_mask_sha256"],
            context="calibration audit eligibility_mask_sha256",
        ),
    )
    _validate_calibration_audit(
        audit,
        mask_commitment=mask_commitment,
        calibration_state=calibration_state,
    )
    if audit != provenance.calibration_audit:
        raise V023StateCodecError(
            "Calibration audit counts differ from the training-issued evidence"
        )
    validated_mask = validate_calibration_mask_commitment_v023(mask_commitment)
    identifiers = top["eligibility_mask_cluster_ids"]
    mask = top["eligibility_mask"]
    if not isinstance(identifiers, list) or any(
        not isinstance(item, str) for item in identifiers
    ):
        raise V023StateCodecError("Calibration audit mask IDs are invalid")
    if not isinstance(mask, list) or any(not isinstance(item, bool) for item in mask):
        raise V023StateCodecError("Calibration audit mask is invalid")
    expected_identifiers = tuple(row.cluster_id for row in validated_mask.rows)
    expected_mask = tuple(row.eligible for row in validated_mask.rows)
    if tuple(identifiers) != expected_identifiers or tuple(mask) != expected_mask:
        raise V023StateCodecError(
            "Calibration audit mask rows or order differ from commitment"
        )
    mask_byte_hash = _digest(
        top["calibration_mask_commitment_byte_sha256"],
        context="calibration audit mask commitment byte hash",
    )
    if mask_byte_hash != _mask_byte_sha256(validated_mask):
        raise V023StateCodecError("Calibration audit mask byte hash mismatch")
    isotonic_hash = _digest(
        top["isotonic_state_sha256"],
        context="calibration audit isotonic state hash",
    )
    conformal_hash = _digest(
        top["conformal_state_sha256"],
        context="calibration audit conformal state hash",
    )
    if isotonic_hash != _v015.isotonic_state_sha256(calibration_state):
        raise V023StateCodecError("Calibration audit isotonic state hash mismatch")
    if conformal_hash != _v015.conformal_state_sha256(calibration_state):
        raise V023StateCodecError("Calibration audit conformal state hash mismatch")
    selected = top["selected_mean_baseline"]
    if (
        not isinstance(selected, str)
        or selected != calibration_state.selected_mean_baseline
    ):
        raise V023StateCodecError("Calibration audit selected baseline mismatch")
    created = _created_utc(top["created_utc"])
    return DecodedV023CalibrationPopulationAudit(
        protocol_id=V023_PROTOCOL_ID,
        config_sha256=provenance.config_sha256,
        audit=audit,
        eligibility_mask_cluster_ids=expected_identifiers,
        eligibility_mask=expected_mask,
        calibration_mask_commitment_byte_sha256=mask_byte_hash,
        isotonic_state_sha256=isotonic_hash,
        conformal_state_sha256=conformal_hash,
        selected_mean_baseline=selected,
        created_utc=created,
    )


def serialize_calibration_population_audit_json_v023(
    *,
    provenance_envelope: V023TrainingProvenanceEnvelope,
    created_utc: str,
) -> bytes:
    payload = build_calibration_population_audit_payload_v023(
        provenance_envelope=provenance_envelope,
        created_utc=created_utc,
    )
    return _canonical_pretty_bytes(payload)


def deserialize_calibration_population_audit_json_v023(
    raw: bytes,
    *,
    provenance_envelope: V023TrainingProvenanceEnvelope,
) -> DecodedV023CalibrationPopulationAudit:
    payload = _decode_canonical_object(
        raw,
        filename="calibration_population_audit.json",
        compact=False,
    )
    decoded = _validate_calibration_population_audit_payload(
        payload,
        provenance_envelope=provenance_envelope,
    )
    normalized = build_calibration_population_audit_payload_v023(
        provenance_envelope=provenance_envelope,
        created_utc=decoded.created_utc,
    )
    if _canonical_pretty_bytes(normalized) != raw:
        raise V023StateCodecError("Calibration population audit is not canonical state")
    return decoded


__all__ = [
    "DecodedV023CalibrationPopulationAudit",
    "V023StateCodecError",
    "build_calibration_manifest_payload_v023",
    "build_calibration_mask_commitment_v023",
    "build_calibration_population_audit_payload_v023",
    "build_model_state_payload_v023",
    "build_training_manifest_payload_v023",
    "deserialize_calibration_manifest_json_v023",
    "deserialize_calibration_mask_commitment_json_v023",
    "deserialize_calibration_population_audit_json_v023",
    "deserialize_model_state_json_v023",
    "deserialize_training_manifest_json_v023",
    "serialize_calibration_manifest_json_v023",
    "serialize_calibration_mask_commitment_json_v023",
    "serialize_calibration_population_audit_json_v023",
    "serialize_model_state_json_v023",
    "serialize_training_manifest_json_v023",
    "validate_calibration_mask_commitment_v023",
]
