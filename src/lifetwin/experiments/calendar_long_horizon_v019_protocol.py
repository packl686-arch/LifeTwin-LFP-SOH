"""Static, non-generative validator for the preregistered V2.4 amendment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


V024_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_4"
V024_ONLY_ATTEMPT_ID = "v024-formal-20260810-a1"
V024_ALLOWED_DESIGN_STATUSES = ("implementation_frozen",)
V024_SOURCE_CALIBRATION_COUNT = 900
V024_MINIMUM_RISK_ISOTONIC_ELIGIBLE_COUNT = 855
V024_MINIMUM_CALIBRATION_CLASS_COUNT = 60
V024_CONFORMAL_COUNT = 900
V024_CONFORMAL_ORDER_STATISTIC_INDEX = 811
V2_SEED_ROOTS = tuple(range(202607230101, 202607230114))
V021_SEED_ROOTS = tuple(range(202607260201, 202607260214))
V022_SEED_ROOTS = tuple(range(202608090201, 202608090214))
V023_SEED_ROOTS = tuple(range(202608100301, 202608100314))
V024_EXPECTED_SEED_ROOTS = MappingProxyType(
    {
        "center_development": 202608100401,
        "risk_development": 202608100402,
        "calibration": 202608100403,
        "test": 202608100404,
        "audit": 202608100405,
        "novel_mechanism_test": 202608100406,
        "novel_mechanism_audit": 202608100407,
        "intrinsic_matched_pairs": 202608100408,
        "stress_plan_matched_pairs": 202608100409,
        "random_rankings": 202608100410,
        "bootstrap": 202608100411,
        "stress_permutations": 202608100412,
        "placebo_covariate": 202608100413,
    }
)
V024_AMENDMENT_BYTE_SHA256 = (
    "24e4e08f10337080212be8932c9c7b696faf281221d6e39b0a33b0cd2d7cb28f"
)
V024_AMENDMENT_SEMANTIC_SHA256 = (
    "cf449e4440ee4ae73fa9310a237dd515b1cc65d96e314412e0c54bf19a58032e"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V024_AMENDMENT_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_4_amendment.json"
)
_EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "title",
        "design_date",
        "design_witnesses",
        "authorization",
        "attempt_registry",
        "immutable_history",
        "exposure_disclosure",
        "scientific_inheritance",
        "fresh_generation",
        "path_isolation",
        "whole_bundle_contract",
        "partition_contract",
        "capability_architecture",
        "numeric_output_contract",
        "lifecycle_order",
        "terminal_rules",
        "pre_result_test_gate",
        "freeze_requirements",
        "claim_boundary",
    }
)
_EXPECTED_WHOLE_ROWS = MappingProxyType(
    {
        "prefix_pack.csv": 71_400,
        "forecast_coordinates.csv": 47_600,
        "operating_pack.csv": 5_950,
        "member_fit_diagnostics.csv": 511_700,
        "member_forecast_bundle.csv": 4_093_600,
    }
)
_EXPECTED_TERMINAL_FILES = (
    "terminal_attempt_record.json",
    "terminal_artifact_manifest.json",
    "terminal_exposure_log_snapshot.jsonl",
)


class V024ProtocolError(ValueError):
    """Raised when the preregistered V2.4 bytes or invariants drift."""


@dataclass(frozen=True, slots=True)
class ValidatedV024Design:
    """Validated, non-executable V2.4 amendment identity."""

    protocol_id: str
    status: str
    config_path: Path
    config_byte_sha256: str
    seed_roots: Mapping[str, int]
    raw: Mapping[str, Any]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise V024ProtocolError(f"{context} must be a JSON object")
    return value


def _validate_payload(
    payload: object, *, path: Path, raw: bytes
) -> ValidatedV024Design:
    top = _object(payload, context="V2.4 amendment")
    if set(top) != _EXPECTED_TOP_LEVEL_KEYS:
        raise V024ProtocolError("V2.4 top-level registry changed")
    if hashlib.sha256(raw).hexdigest() != V024_AMENDMENT_BYTE_SHA256:
        raise V024ProtocolError("V2.4 amendment byte commitment changed")
    canonical = json.dumps(
        top,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != V024_AMENDMENT_SEMANTIC_SHA256:
        raise V024ProtocolError("V2.4 amendment semantic commitment changed")
    if top.get("schema_version") != "lifetwin_synthetic_long_horizon_v2_4/1.0.0":
        raise V024ProtocolError("V2.4 schema version changed")
    if top.get("protocol_id") != V024_PROTOCOL_ID:
        raise V024ProtocolError("V2.4 protocol identity changed")
    status = top.get("status")
    if status not in V024_ALLOWED_DESIGN_STATUSES:
        raise V024ProtocolError("V2.4 design status changed")
    if top.get("design_witnesses") != ["Jincheng Liu"]:
        raise V024ProtocolError("V2.4 design witness changed")

    authorization = _object(top.get("authorization"), context="authorization")
    if (
        authorization.get("formal_execution_authorized_after_all_freeze_gates")
        is not True
        or authorization.get("generation_before_freeze_forbidden") is not True
        or authorization.get("additional_confirmation_after_freeze_required")
        is not False
    ):
        raise V024ProtocolError("V2.4 execution authorization boundary changed")

    attempt = _object(top.get("attempt_registry"), context="attempt_registry")
    if (
        attempt.get("only_attempt_id") != V024_ONLY_ATTEMPT_ID
        or attempt.get("maximum_attempts") != 1
        or attempt.get("a2_or_replacement_attempt_forbidden") is not True
        or attempt.get("retry_for_better_result_forbidden") is not True
    ):
        raise V024ProtocolError("V2.4 one-shot attempt registry changed")

    history = _object(top.get("immutable_history"), context="immutable_history")
    if (
        history.get("v014_status") != "failure"
        or history.get("v015_status") != "inconclusive_not_success"
        or history.get("v021_attempt_id") != "v021-formal-20260808-a1"
        or history.get("v021_may_be_reclassified_repaired_resumed_or_rerun")
        is not False
        or history.get("v022_attempt_id") != "v022-formal-20260809-a1"
        or history.get("v022_scientific_status") != "void"
        or history.get("v022_reason_code") != "INTEGRITY_PARTITION_CONTRACT_MISMATCH"
        or history.get("v022_prediction_commitment_existed") is not False
        or history.get("v022_opened_truth_files") != []
        or history.get("v022_may_be_reclassified_repaired_resumed_or_rerun")
        is not False
        or history.get("v023_attempt_id") != "v023-formal-20260810-a1"
        or history.get("v023_scientific_status") != "void"
        or history.get("v023_classification_mode") != "proven_integrity"
        or history.get("v023_reason_code") != "INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH"
        or history.get("v023_last_completed_phase") != "fresh_generation_committed"
        or history.get("v023_attempted_phase") != "label_free_fit_committed"
        or history.get("v023_fit_commitment_file_existed_as_partial_artifact")
        is not True
        or history.get("v023_registered_fit_commitment") is not None
        or history.get("v023_prediction_commitment_existed") is not False
        or history.get("v023_opened_truth_files") != []
        or history.get("v023_may_be_reclassified_repaired_resumed_or_rerun")
        is not False
    ):
        raise V024ProtocolError("Immutable predecessor history changed")

    fresh = _object(top.get("fresh_generation"), context="fresh_generation")
    if (
        fresh.get("generation_has_started") is not False
        or fresh.get("seed_consumed") is not False
        or fresh.get("sealed_truth_created_or_opened") is not False
        or fresh.get("pilot_or_test_generation_forbidden") is not True
    ):
        raise V024ProtocolError("V2.4 result-free generation status changed")
    roots = _object(fresh.get("seed_roots"), context="fresh_generation.seed_roots")
    if dict(roots) != dict(V024_EXPECTED_SEED_ROOTS):
        raise V024ProtocolError("V2.4 seed roots changed")
    if set(roots.values()).intersection(
        (*V2_SEED_ROOTS, *V021_SEED_ROOTS, *V022_SEED_ROOTS, *V023_SEED_ROOTS)
    ):
        raise V024ProtocolError("V2.4 seed roots collide with a predecessor")

    whole = _object(top.get("whole_bundle_contract"), context="whole_bundle_contract")
    if (
        whole.get("validation_must_precede_partition_slicing") is not True
        or whole.get("formal_false_forbidden") is not True
        or whole.get("required_tables") != dict(_EXPECTED_WHOLE_ROWS)
    ):
        raise V024ProtocolError("Whole-bundle formal contract changed")

    capability = _object(top.get("capability_architecture"), context="capability")
    if (
        capability.get("route")
        != "whole_bundle_formal_validation_then_capability_bound_partition_slice"
        or capability.get("whole_bundle_capability") != "WholeBundleValidated"
        or capability.get("partition_capability") != "ValidatedPartitionView"
    ):
        raise V024ProtocolError("V2.4 capability architecture changed")

    numeric = _object(top.get("numeric_output_contract"), context="numeric contract")
    if (
        numeric.get("infinity_allowed") is not False
        or "fill" not in str(numeric.get("policy", "")).lower()
        or "structural-nan" not in str(numeric.get("policy", "")).lower()
    ):
        raise V024ProtocolError("V2.4 numeric output contract changed")
    member_fit = _object(numeric.get("member_fit"), context="member-fit contract")
    if set(member_fit) != {
        "status_registry",
        "succeeded_mask",
        "failed_mask",
        "forbidden",
    }:
        raise V024ProtocolError("V2.4 member-fit contract changed")
    atomicity = _object(
        numeric.get("fit_commitment_atomicity"),
        context="fit commitment atomicity",
    )
    if set(atomicity) != {
        "write_order",
        "ledger_order",
        "failure_rule",
        "terminal_manifest_rule",
    }:
        raise V024ProtocolError("V2.4 fit commitment atomicity changed")

    terminal = _object(top.get("terminal_rules"), context="terminal_rules")
    if terminal.get("known_partition_error_may_use_unknown_default") is not False:
        raise V024ProtocolError("Known partition failures may not use unknown_default")
    if (
        tuple(terminal.get("terminal_registry_exact_files", ()))
        != _EXPECTED_TERMINAL_FILES
    ):
        raise V024ProtocolError("V2.4 terminal registry changed")
    if (
        terminal.get("prediction_and_terminal_registries_mutually_exclusive")
        is not True
    ):
        raise V024ProtocolError("V2.4 registry exclusivity changed")
    if "INTEGRITY_MEMBER_FIT_NUMERIC_CONTRACT_MISMATCH" not in tuple(
        terminal.get("registered_v024_integrity_codes", ())
    ):
        raise V024ProtocolError("V2.4 member-fit terminal code changed")

    return ValidatedV024Design(
        protocol_id=V024_PROTOCOL_ID,
        status=str(status),
        config_path=path,
        config_byte_sha256=V024_AMENDMENT_BYTE_SHA256,
        seed_roots=MappingProxyType({str(k): int(v) for k, v in roots.items()}),
        raw=_deep_freeze(top),
    )


def load_v024_design(
    path: str | Path = DEFAULT_V024_AMENDMENT_PATH,
) -> ValidatedV024Design:
    """Load and validate V2.4 design bytes without consuming a seed."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V024ProtocolError("Cannot read V2.4 amendment JSON") from exc
    return _validate_payload(payload, path=config_path, raw=raw)


__all__ = [
    "DEFAULT_V024_AMENDMENT_PATH",
    "V021_SEED_ROOTS",
    "V022_SEED_ROOTS",
    "V023_SEED_ROOTS",
    "V024_ALLOWED_DESIGN_STATUSES",
    "V024_AMENDMENT_BYTE_SHA256",
    "V024_AMENDMENT_SEMANTIC_SHA256",
    "V024_EXPECTED_SEED_ROOTS",
    "V024_ONLY_ATTEMPT_ID",
    "V024_PROTOCOL_ID",
    "V024ProtocolError",
    "V2_SEED_ROOTS",
    "ValidatedV024Design",
    "load_v024_design",
]
