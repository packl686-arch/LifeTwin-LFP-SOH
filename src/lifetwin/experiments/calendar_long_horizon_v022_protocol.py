"""Static, non-generative validator for the preregistered V2.7 amendment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from lifetwin.experiments.calendar_long_horizon_v019_protocol import (
    V021_SEED_ROOTS,
    V022_SEED_ROOTS,
    V023_SEED_ROOTS,
    V024_EXPECTED_SEED_ROOTS,
    V2_SEED_ROOTS,
)
from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
    INPUT_FILENAMES_BY_STAGE,
)
from lifetwin.experiments.calendar_long_horizon_v020_protocol import (
    V025_EXPECTED_SEED_ROOTS,
)
from lifetwin.experiments.calendar_long_horizon_v021_protocol import (
    V026_EXPECTED_SEED_ROOTS,
)


V027_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_7"
V027_ONLY_ATTEMPT_ID = "v027-formal-20260813-a1"
V027_DESIGN_STATUS = "implementation_frozen"
V027_EXPECTED_SEED_ROOTS = MappingProxyType(
    {
        "center_development": 202608130701,
        "risk_development": 202608130702,
        "calibration": 202608130703,
        "test": 202608130704,
        "audit": 202608130705,
        "novel_mechanism_test": 202608130706,
        "novel_mechanism_audit": 202608130707,
        "intrinsic_matched_pairs": 202608130708,
        "stress_plan_matched_pairs": 202608130709,
        "random_rankings": 202608130710,
        "bootstrap": 202608130711,
        "stress_permutations": 202608130712,
        "placebo_covariate": 202608130713,
    }
)
V027_AMENDMENT_BYTE_SHA256 = (
    "5669638e854d15dd0873ee863c93635f3f287753fa0b823c708f7e12a2c3d6b2"
)
V027_AMENDMENT_SEMANTIC_SHA256 = (
    "d9e25ea634ff5bae3c03c6dbb0a329e994e480db80ef0621479a19023747c9cf"
)
V027_PREREG_BYTE_SHA256 = (
    "ffd99eb9019e8cb86d148a94d697d6f3241701f179d0511ae848cf06d9ad63f8"
)
V027_REQUIREMENTS_BYTE_SHA256 = (
    "9261baf3be841996c357aa44ef815de2eaebb3a051f97de433ce67ce49047c6a"
)
V027_DESIGN_FREEZE_COMMIT = "906877fbf627bb32ac9a22f12caea1a63125b3f7"
V027_FIXED_CORE_COMMIT = "43ca947a02b15ab373422a7b066a361ed711cd1b"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V027_AMENDMENT_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_7_amendment.json"
)
_EXPECTED_KEYS = {
    "schema_version",
    "protocol_id",
    "implementation_profile",
    "status",
    "design_date",
    "base_contract",
    "change_scope",
    "attempt_registry",
    "fresh_generation",
    "path_isolation",
    "exposure_contract_fix",
    "partition_contract_view",
    "checkpoint_registry_contract",
    "scientific_inheritance",
    "predecessor_terminal",
    "lifecycle_order",
    "terminal_rules",
    "freeze_requirements",
    "claim_boundary",
}


class V027ProtocolError(ValueError):
    """Raised when the result-blind V2.7 amendment bytes drift."""


@dataclass(frozen=True, slots=True)
class ValidatedV027Design:
    protocol_id: str
    status: str
    config_path: Path
    config_byte_sha256: str
    config_semantic_sha256: str
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


def _object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise V027ProtocolError(f"{context} must be a JSON object")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V027ProtocolError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V027ProtocolError(f"Nonfinite JSON constant: {token}")


def load_v027_design(
    path: str | Path = DEFAULT_V027_AMENDMENT_PATH,
) -> ValidatedV027Design:
    """Validate committed V2.7 bytes without deriving or consuming a seed."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V027ProtocolError("Cannot read V2.7 amendment JSON") from exc
    top = _object(payload, context="V2.7 amendment")
    canonical = json.dumps(
        top,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if (
        set(top) != _EXPECTED_KEYS
        or hashlib.sha256(raw).hexdigest() != V027_AMENDMENT_BYTE_SHA256
        or hashlib.sha256(canonical).hexdigest() != V027_AMENDMENT_SEMANTIC_SHA256
        or top.get("schema_version") != "lifetwin_synthetic_long_horizon_v2_7/1.0.0"
        or top.get("protocol_id") != V027_PROTOCOL_ID
        or top.get("implementation_profile") != "v0.22"
        or top.get("status") != V027_DESIGN_STATUS
    ):
        raise V027ProtocolError("V2.7 amendment identity or commitment changed")

    base = _object(top.get("base_contract"), context="base_contract")
    if (
        base.get("protocol_id") != "synthetic_long_horizon_identifiability_v2_6"
        or base.get("amendment_byte_sha256")
        != "6784cace2f2d3f4f561ef8abdbde580d4800343787748c52fa7280af7b4ddb81"
        or base.get("amendment_semantic_sha256")
        != "fd090ca56e3d0ad2c91fe442e272a928d6e778571b4e93a4e45a28780124cc54"
        or base.get("fixed_core_commit") != V027_FIXED_CORE_COMMIT
    ):
        raise V027ProtocolError("V2.7 base contract changed")

    attempt = _object(top.get("attempt_registry"), context="attempt_registry")
    if (
        attempt.get("only_attempt_id") != V027_ONLY_ATTEMPT_ID
        or attempt.get("maximum_attempts") != 1
        or attempt.get("a2_or_replacement_attempt_forbidden") is not True
        or attempt.get("formal_generation_before_freeze_forbidden") is not True
    ):
        raise V027ProtocolError("V2.7 attempt registry changed")

    fresh = _object(top.get("fresh_generation"), context="fresh_generation")
    roots = _object(fresh.get("seed_roots"), context="seed_roots")
    if (
        fresh.get("generation_has_started") is not False
        or fresh.get("seed_consumed") is not False
        or fresh.get("sealed_truth_created_or_opened") is not False
        or fresh.get("pilot_or_test_generation_forbidden") is not True
        or roots != dict(V027_EXPECTED_SEED_ROOTS)
    ):
        raise V027ProtocolError("V2.7 result-blind seed registry changed")
    predecessors = {
        *V2_SEED_ROOTS,
        *V021_SEED_ROOTS,
        *V022_SEED_ROOTS,
        *V023_SEED_ROOTS,
        *V024_EXPECTED_SEED_ROOTS.values(),
        *V025_EXPECTED_SEED_ROOTS.values(),
        *V026_EXPECTED_SEED_ROOTS.values(),
    }
    if predecessors.intersection(roots.values()) or len(set(roots.values())) != 13:
        raise V027ProtocolError("V2.7 seed roots collide or are not unique")

    checkpoint = _object(
        top.get("checkpoint_registry_contract"),
        context="checkpoint_registry_contract",
    )
    for stage, names in INPUT_FILENAMES_BY_STAGE.items():
        if checkpoint.get(f"{stage}_count") != len(names):
            raise V027ProtocolError(f"V2.7 {stage} checkpoint registry changed")
    if checkpoint.get("failure_rule") is None:
        raise V027ProtocolError("V2.7 checkpoint failure rule is absent")

    isolation = _object(top.get("path_isolation"), context="path_isolation")
    for role in ("label_free", "sealed_truth", "score", "termination"):
        expected = f"artifacts/{V027_ONLY_ATTEMPT_ID}-{role.replace('_', '-')}"
        if isolation.get(f"{role}_root") != expected:
            raise V027ProtocolError(f"V2.7 {role} root changed")
    if (
        isolation.get("formal_roots_must_be_absent_before_launch") is not True
        or isolation.get("resolved_roots_must_be_pairwise_distinct") is not True
    ):
        raise V027ProtocolError("V2.7 path isolation weakened")

    inheritance = _object(
        top.get("scientific_inheritance"), context="scientific_inheritance"
    )
    if len(inheritance.get("unchanged", ())) != 10:
        raise V027ProtocolError("V2.7 scientific inheritance changed")
    terminal = _object(top.get("terminal_rules"), context="terminal_rules")
    if (
        terminal.get("known_integrity_contract_mismatch_is_void") is not True
        or terminal.get("unknown_exception_preserves_unknown_default") is not True
        or terminal.get("prediction_and_terminal_registries_mutually_exclusive")
        is not True
    ):
        raise V027ProtocolError("V2.7 terminal rules changed")
    partition_view = _object(
        top.get("partition_contract_view"), context="partition_contract_view"
    )
    if (
        partition_view.get("naked_artifacts_fail_closed") is not True
        or partition_view.get("resolver_relaxation_forbidden") is not True
    ):
        raise V027ProtocolError("V2.7 authenticated-view contract changed")
    exposure = _object(
        top.get("exposure_contract_fix"), context="exposure_contract_fix"
    )
    if (
        exposure.get("base_config_hash_substitution_forbidden") is not True
        or exposure.get("ledger_schema_and_phase_order_unchanged") is not True
        or exposure.get("all_phase_and_failure_append_callers_use_artifact_contract")
        is not True
    ):
        raise V027ProtocolError("V2.7 exposure contract fix changed")
    predecessor = _object(
        top.get("predecessor_terminal"), context="predecessor_terminal"
    )
    if (
        predecessor.get("attempt_id") != "v026-formal-20260812-a1"
        or predecessor.get("reason_code") != "UNKNOWN_PRE_PREDICTION_EXCEPTION"
        or predecessor.get("scientific_status") != "unclassified_terminal_not_success"
        or predecessor.get("opened_truth_files")
        != ["center_development_truth.csv", "risk_development_truth.csv"]
        or predecessor.get("prediction_commitment_created") is not False
        or predecessor.get("score_created") is not False
        or predecessor.get("immutable_and_not_reusable") is not True
    ):
        raise V027ProtocolError("V2.6 terminal history changed")

    return ValidatedV027Design(
        protocol_id=V027_PROTOCOL_ID,
        status=V027_DESIGN_STATUS,
        config_path=config_path,
        config_byte_sha256=V027_AMENDMENT_BYTE_SHA256,
        config_semantic_sha256=V027_AMENDMENT_SEMANTIC_SHA256,
        seed_roots=MappingProxyType({str(k): int(v) for k, v in roots.items()}),
        raw=_deep_freeze(top),
    )


__all__ = [
    "DEFAULT_V027_AMENDMENT_PATH",
    "V027_AMENDMENT_BYTE_SHA256",
    "V027_AMENDMENT_SEMANTIC_SHA256",
    "V027_DESIGN_FREEZE_COMMIT",
    "V027_DESIGN_STATUS",
    "V027_EXPECTED_SEED_ROOTS",
    "V027_FIXED_CORE_COMMIT",
    "V027_ONLY_ATTEMPT_ID",
    "V027_PREREG_BYTE_SHA256",
    "V027_PROTOCOL_ID",
    "V027_REQUIREMENTS_BYTE_SHA256",
    "V027ProtocolError",
    "ValidatedV027Design",
    "load_v027_design",
]
