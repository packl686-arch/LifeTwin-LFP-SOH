"""Static, non-generative validator for the preregistered V2.6 amendment."""

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


V026_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_6"
V026_ONLY_ATTEMPT_ID = "v026-formal-20260812-a1"
V026_DESIGN_STATUS = "implementation_frozen"
V026_EXPECTED_SEED_ROOTS = MappingProxyType(
    {
        "center_development": 202608120601,
        "risk_development": 202608120602,
        "calibration": 202608120603,
        "test": 202608120604,
        "audit": 202608120605,
        "novel_mechanism_test": 202608120606,
        "novel_mechanism_audit": 202608120607,
        "intrinsic_matched_pairs": 202608120608,
        "stress_plan_matched_pairs": 202608120609,
        "random_rankings": 202608120610,
        "bootstrap": 202608120611,
        "stress_permutations": 202608120612,
        "placebo_covariate": 202608120613,
    }
)
V026_AMENDMENT_BYTE_SHA256 = (
    "6784cace2f2d3f4f561ef8abdbde580d4800343787748c52fa7280af7b4ddb81"
)
V026_AMENDMENT_SEMANTIC_SHA256 = (
    "fd090ca56e3d0ad2c91fe442e272a928d6e778571b4e93a4e45a28780124cc54"
)
V026_PREREG_BYTE_SHA256 = (
    "5250732d88918f2409b8cd06f84657c127d9f7222f25ca2e3d9ce0f73a32c23d"
)
V026_REQUIREMENTS_BYTE_SHA256 = (
    "7c80ceca777636afa26024cfc217ad855d1b141383c062de7b38e58d19fc692b"
)
V026_DESIGN_FREEZE_COMMIT = "2a364a827af3a96eea1cb89ccb770cba48e86aa2"
V026_FIXED_CORE_COMMIT = "044259e28dd0f37fb4dfc0ad12f7a4993a83f38b"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V026_AMENDMENT_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_6_amendment.json"
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
    "partition_contract_view",
    "checkpoint_registry_contract",
    "scientific_inheritance",
    "predecessor_terminal",
    "lifecycle_order",
    "terminal_rules",
    "freeze_requirements",
    "claim_boundary",
}


class V026ProtocolError(ValueError):
    """Raised when the result-blind V2.6 amendment bytes drift."""


@dataclass(frozen=True, slots=True)
class ValidatedV026Design:
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
        raise V026ProtocolError(f"{context} must be a JSON object")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V026ProtocolError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V026ProtocolError(f"Nonfinite JSON constant: {token}")


def load_v026_design(
    path: str | Path = DEFAULT_V026_AMENDMENT_PATH,
) -> ValidatedV026Design:
    """Validate committed V2.6 bytes without deriving or consuming a seed."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V026ProtocolError("Cannot read V2.6 amendment JSON") from exc
    top = _object(payload, context="V2.6 amendment")
    canonical = json.dumps(
        top,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if (
        set(top) != _EXPECTED_KEYS
        or hashlib.sha256(raw).hexdigest() != V026_AMENDMENT_BYTE_SHA256
        or hashlib.sha256(canonical).hexdigest() != V026_AMENDMENT_SEMANTIC_SHA256
        or top.get("schema_version") != "lifetwin_synthetic_long_horizon_v2_6/1.0.0"
        or top.get("protocol_id") != V026_PROTOCOL_ID
        or top.get("implementation_profile") != "v0.21"
        or top.get("status") != V026_DESIGN_STATUS
    ):
        raise V026ProtocolError("V2.6 amendment identity or commitment changed")

    base = _object(top.get("base_contract"), context="base_contract")
    if (
        base.get("protocol_id") != "synthetic_long_horizon_identifiability_v2_5"
        or base.get("amendment_byte_sha256")
        != "4fb7fb0394fd91f772c51d290f03bfb0f8e6fc73a379acedf72874a65d34f119"
        or base.get("amendment_semantic_sha256")
        != "1a1e5e52be338f84c1bfe29b41883e673ffef0d9d6e6be6756fdbeca5497f43c"
        or base.get("fixed_core_commit") != V026_FIXED_CORE_COMMIT
    ):
        raise V026ProtocolError("V2.6 base contract changed")

    attempt = _object(top.get("attempt_registry"), context="attempt_registry")
    if (
        attempt.get("only_attempt_id") != V026_ONLY_ATTEMPT_ID
        or attempt.get("maximum_attempts") != 1
        or attempt.get("a2_or_replacement_attempt_forbidden") is not True
        or attempt.get("formal_generation_before_freeze_forbidden") is not True
    ):
        raise V026ProtocolError("V2.6 attempt registry changed")

    fresh = _object(top.get("fresh_generation"), context="fresh_generation")
    roots = _object(fresh.get("seed_roots"), context="seed_roots")
    if (
        fresh.get("generation_has_started") is not False
        or fresh.get("seed_consumed") is not False
        or fresh.get("sealed_truth_created_or_opened") is not False
        or fresh.get("pilot_or_test_generation_forbidden") is not True
        or roots != dict(V026_EXPECTED_SEED_ROOTS)
    ):
        raise V026ProtocolError("V2.6 result-blind seed registry changed")
    predecessors = {
        *V2_SEED_ROOTS,
        *V021_SEED_ROOTS,
        *V022_SEED_ROOTS,
        *V023_SEED_ROOTS,
        *V024_EXPECTED_SEED_ROOTS.values(),
        *V025_EXPECTED_SEED_ROOTS.values(),
    }
    if predecessors.intersection(roots.values()) or len(set(roots.values())) != 13:
        raise V026ProtocolError("V2.6 seed roots collide or are not unique")

    checkpoint = _object(
        top.get("checkpoint_registry_contract"),
        context="checkpoint_registry_contract",
    )
    for stage, names in INPUT_FILENAMES_BY_STAGE.items():
        if checkpoint.get(f"{stage}_count") != len(names):
            raise V026ProtocolError(f"V2.6 {stage} checkpoint registry changed")
    if checkpoint.get("failure_rule") is None:
        raise V026ProtocolError("V2.6 checkpoint failure rule is absent")

    isolation = _object(top.get("path_isolation"), context="path_isolation")
    for role in ("label_free", "sealed_truth", "score", "termination"):
        expected = f"artifacts/{V026_ONLY_ATTEMPT_ID}-{role.replace('_', '-')}"
        if isolation.get(f"{role}_root") != expected:
            raise V026ProtocolError(f"V2.6 {role} root changed")
    if (
        isolation.get("formal_roots_must_be_absent_before_launch") is not True
        or isolation.get("resolved_roots_must_be_pairwise_distinct") is not True
    ):
        raise V026ProtocolError("V2.6 path isolation weakened")

    inheritance = _object(
        top.get("scientific_inheritance"), context="scientific_inheritance"
    )
    if len(inheritance.get("unchanged", ())) != 10:
        raise V026ProtocolError("V2.6 scientific inheritance changed")
    terminal = _object(top.get("terminal_rules"), context="terminal_rules")
    if (
        terminal.get("partition_capability_mismatch_is_integrity_void") is not True
        or terminal.get("known_contract_error_may_use_unknown_default") is not False
        or terminal.get("prediction_and_terminal_registries_mutually_exclusive")
        is not True
    ):
        raise V026ProtocolError("V2.6 terminal rules changed")
    partition_view = _object(
        top.get("partition_contract_view"), context="partition_contract_view"
    )
    if (
        partition_view.get("naked_artifacts_fail_closed") is not True
        or partition_view.get("resolver_relaxation_forbidden") is not True
    ):
        raise V026ProtocolError("V2.6 authenticated-view contract changed")
    predecessor = _object(
        top.get("predecessor_terminal"), context="predecessor_terminal"
    )
    if (
        predecessor.get("attempt_id") != "v025-formal-20260812-a1"
        or predecessor.get("reason_code") != "INTEGRITY_PARTITION_CAPABILITY_MISMATCH"
        or predecessor.get("immutable_and_not_reusable") is not True
    ):
        raise V026ProtocolError("V2.5 terminal history changed")

    return ValidatedV026Design(
        protocol_id=V026_PROTOCOL_ID,
        status=V026_DESIGN_STATUS,
        config_path=config_path,
        config_byte_sha256=V026_AMENDMENT_BYTE_SHA256,
        config_semantic_sha256=V026_AMENDMENT_SEMANTIC_SHA256,
        seed_roots=MappingProxyType({str(k): int(v) for k, v in roots.items()}),
        raw=_deep_freeze(top),
    )


__all__ = [
    "DEFAULT_V026_AMENDMENT_PATH",
    "V026_AMENDMENT_BYTE_SHA256",
    "V026_AMENDMENT_SEMANTIC_SHA256",
    "V026_DESIGN_FREEZE_COMMIT",
    "V026_DESIGN_STATUS",
    "V026_EXPECTED_SEED_ROOTS",
    "V026_FIXED_CORE_COMMIT",
    "V026_ONLY_ATTEMPT_ID",
    "V026_PREREG_BYTE_SHA256",
    "V026_PROTOCOL_ID",
    "V026_REQUIREMENTS_BYTE_SHA256",
    "V026ProtocolError",
    "ValidatedV026Design",
    "load_v026_design",
]
