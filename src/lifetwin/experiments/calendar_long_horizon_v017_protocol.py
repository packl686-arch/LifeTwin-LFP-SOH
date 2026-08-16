"""Static, non-generative validator for the preregistered V2.2 amendment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


V022_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_2"
V022_ONLY_ATTEMPT_ID = "v022-formal-20260809-a1"
V022_ALLOWED_DESIGN_STATUSES = ("implementation_frozen",)
V022_SOURCE_CALIBRATION_COUNT = 900
V022_MINIMUM_RISK_ISOTONIC_ELIGIBLE_COUNT = 855
V022_MINIMUM_CALIBRATION_CLASS_COUNT = 60
V022_CONFORMAL_COUNT = 900
V022_CONFORMAL_ORDER_STATISTIC_INDEX = 811
V2_SEED_ROOTS = tuple(range(202607230101, 202607230114))
V021_SEED_ROOTS = tuple(range(202607260201, 202607260214))
V022_EXPECTED_SEED_ROOTS = MappingProxyType(
    {
        "center_development": 202608090201,
        "risk_development": 202608090202,
        "calibration": 202608090203,
        "test": 202608090204,
        "audit": 202608090205,
        "novel_mechanism_test": 202608090206,
        "novel_mechanism_audit": 202608090207,
        "intrinsic_matched_pairs": 202608090208,
        "stress_plan_matched_pairs": 202608090209,
        "random_rankings": 202608090210,
        "bootstrap": 202608090211,
        "stress_permutations": 202608090212,
        "placebo_covariate": 202608090213,
    }
)
V022_AMENDMENT_BYTE_SHA256 = (
    "aaadd5b9d5436d6ccfa08806250f0a48bef93e04446d0c089cb2eb5cf8ce0f29"
)
V022_AMENDMENT_SEMANTIC_SHA256 = (
    "88f6e32067b9637a931b149e9950f2a690c3e4558effbde6a5002cba8bd5b6a2"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V022_AMENDMENT_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_2_amendment.json"
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


class V022ProtocolError(ValueError):
    """Raised when the preregistered V2.2 bytes or invariants drift."""


@dataclass(frozen=True, slots=True)
class ValidatedV022Design:
    """Validated, non-executable V2.2 amendment identity."""

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
        raise V022ProtocolError(f"{context} must be a JSON object")
    return value


def _validate_payload(payload: object, *, path: Path, raw: bytes) -> ValidatedV022Design:
    top = _object(payload, context="V2.2 amendment")
    if set(top) != _EXPECTED_TOP_LEVEL_KEYS:
        raise V022ProtocolError("V2.2 top-level registry changed")
    if hashlib.sha256(raw).hexdigest() != V022_AMENDMENT_BYTE_SHA256:
        raise V022ProtocolError("V2.2 amendment byte commitment changed")
    canonical = json.dumps(
        top,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != V022_AMENDMENT_SEMANTIC_SHA256:
        raise V022ProtocolError("V2.2 amendment semantic commitment changed")
    if top.get("schema_version") != "lifetwin_synthetic_long_horizon_v2_2/1.0.0":
        raise V022ProtocolError("V2.2 schema version changed")
    if top.get("protocol_id") != V022_PROTOCOL_ID:
        raise V022ProtocolError("V2.2 protocol identity changed")
    status = top.get("status")
    if status not in V022_ALLOWED_DESIGN_STATUSES:
        raise V022ProtocolError("V2.2 design status changed")
    if top.get("design_witnesses") != ["Jincheng Liu"]:
        raise V022ProtocolError("V2.2 design witness changed")

    authorization = _object(top.get("authorization"), context="authorization")
    if (
        authorization.get("formal_execution_authorized_after_all_freeze_gates")
        is not True
        or authorization.get("generation_before_freeze_forbidden") is not True
        or authorization.get("additional_confirmation_after_freeze_required") is not False
    ):
        raise V022ProtocolError("V2.2 execution authorization boundary changed")

    attempt = _object(top.get("attempt_registry"), context="attempt_registry")
    if (
        attempt.get("only_attempt_id") != V022_ONLY_ATTEMPT_ID
        or attempt.get("maximum_attempts") != 1
        or attempt.get("a2_or_replacement_attempt_forbidden") is not True
        or attempt.get("retry_for_better_result_forbidden") is not True
    ):
        raise V022ProtocolError("V2.2 one-shot attempt registry changed")

    history = _object(top.get("immutable_history"), context="immutable_history")
    if (
        history.get("v014_status") != "failure"
        or history.get("v015_status") != "inconclusive_not_success"
        or history.get("v021_attempt_id") != "v021-formal-20260808-a1"
        or history.get("v021_may_be_reclassified_repaired_resumed_or_rerun") is not False
    ):
        raise V022ProtocolError("Immutable predecessor history changed")

    fresh = _object(top.get("fresh_generation"), context="fresh_generation")
    if (
        fresh.get("generation_has_started") is not False
        or fresh.get("seed_consumed") is not False
        or fresh.get("sealed_truth_created_or_opened") is not False
        or fresh.get("pilot_or_test_generation_forbidden") is not True
    ):
        raise V022ProtocolError("V2.2 result-free generation status changed")
    roots = _object(fresh.get("seed_roots"), context="fresh_generation.seed_roots")
    if dict(roots) != dict(V022_EXPECTED_SEED_ROOTS):
        raise V022ProtocolError("V2.2 seed roots changed")
    if set(roots.values()).intersection((*V2_SEED_ROOTS, *V021_SEED_ROOTS)):
        raise V022ProtocolError("V2.2 seed roots collide with a predecessor")

    whole = _object(top.get("whole_bundle_contract"), context="whole_bundle_contract")
    if (
        whole.get("validation_must_precede_partition_slicing") is not True
        or whole.get("formal_false_forbidden") is not True
        or whole.get("required_tables") != dict(_EXPECTED_WHOLE_ROWS)
    ):
        raise V022ProtocolError("Whole-bundle formal contract changed")

    capability = _object(top.get("capability_architecture"), context="capability")
    if (
        capability.get("route")
        != "whole_bundle_formal_validation_then_capability_bound_partition_slice"
        or capability.get("whole_bundle_capability") != "WholeBundleValidated"
        or capability.get("partition_capability") != "ValidatedPartitionView"
    ):
        raise V022ProtocolError("V2.2 capability architecture changed")

    terminal = _object(top.get("terminal_rules"), context="terminal_rules")
    if terminal.get("known_partition_error_may_use_unknown_default") is not False:
        raise V022ProtocolError("Known partition failures may not use unknown_default")
    if tuple(terminal.get("terminal_registry_exact_files", ())) != _EXPECTED_TERMINAL_FILES:
        raise V022ProtocolError("V2.2 terminal registry changed")
    if terminal.get("prediction_and_terminal_registries_mutually_exclusive") is not True:
        raise V022ProtocolError("V2.2 registry exclusivity changed")

    return ValidatedV022Design(
        protocol_id=V022_PROTOCOL_ID,
        status=str(status),
        config_path=path,
        config_byte_sha256=V022_AMENDMENT_BYTE_SHA256,
        seed_roots=MappingProxyType({str(k): int(v) for k, v in roots.items()}),
        raw=_deep_freeze(top),
    )


def load_v022_design(
    path: str | Path = DEFAULT_V022_AMENDMENT_PATH,
) -> ValidatedV022Design:
    """Load and validate V2.2 design bytes without consuming a seed."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V022ProtocolError("Cannot read V2.2 amendment JSON") from exc
    return _validate_payload(payload, path=config_path, raw=raw)


__all__ = [
    "DEFAULT_V022_AMENDMENT_PATH",
    "V021_SEED_ROOTS",
    "V022_ALLOWED_DESIGN_STATUSES",
    "V022_AMENDMENT_BYTE_SHA256",
    "V022_AMENDMENT_SEMANTIC_SHA256",
    "V022_EXPECTED_SEED_ROOTS",
    "V022_ONLY_ATTEMPT_ID",
    "V022_PROTOCOL_ID",
    "V022ProtocolError",
    "V2_SEED_ROOTS",
    "ValidatedV022Design",
    "load_v022_design",
]
