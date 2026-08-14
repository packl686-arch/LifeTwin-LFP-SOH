"""Static, non-generative validator for the preregistered V2.10 amendment."""

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
from lifetwin.experiments.calendar_long_horizon_v022_protocol import (
    V027_EXPECTED_SEED_ROOTS,
)
from lifetwin.experiments.calendar_long_horizon_v023_protocol import (
    V028_EXPECTED_SEED_ROOTS,
)
from lifetwin.experiments.calendar_long_horizon_v024_protocol import (
    V029_EXPECTED_SEED_ROOTS,
)


V030_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_10"
V030_ONLY_ATTEMPT_ID = "v030-formal-20260814-a1"
V030_DESIGN_STATUS = "implementation_frozen"
V030_EXPECTED_SEED_ROOTS = MappingProxyType(
    {
        "center_development": 202608141001,
        "risk_development": 202608141002,
        "calibration": 202608141003,
        "test": 202608141004,
        "audit": 202608141005,
        "novel_mechanism_test": 202608141006,
        "novel_mechanism_audit": 202608141007,
        "intrinsic_matched_pairs": 202608141008,
        "stress_plan_matched_pairs": 202608141009,
        "random_rankings": 202608141010,
        "bootstrap": 202608141011,
        "stress_permutations": 202608141012,
        "placebo_covariate": 202608141013,
    }
)
V030_AMENDMENT_BYTE_SHA256 = (
    "a728bb0a688b0a6f09a6a788dd38a13b03e8c3497b3d16f3054c9261dc2251ba"
)
V030_AMENDMENT_SEMANTIC_SHA256 = (
    "1524b607a1eb88b37dddab132d1476024407ec03734c57095ffe3da72d5f78c6"
)
V030_PREREG_BYTE_SHA256 = (
    "03292c2acda7084b25d9d9a0eea314f1f4cb6d319f2af623c61fa414051d728e"
)
V030_REQUIREMENTS_BYTE_SHA256 = (
    "bfd057b1538c4d5c1d9fc3079529c209d623cc9f618ca079583bd76e8d48315c"
)
V030_DESIGN_FREEZE_COMMIT = "de546c4e6afe8473637c2318727ffd7e82b6b4de"
V030_FIXED_CORE_COMMIT = "6bcf826b365e5d483f03ee4a2617a57f533c0f6b"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V030_AMENDMENT_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_10_amendment.json"
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
    "prediction_capsule_identity_handoff_fix",
    "result_blind_development_evidence",
    "partition_contract_view",
    "checkpoint_registry_contract",
    "scientific_inheritance",
    "predecessor_terminal",
    "lifecycle_order",
    "terminal_rules",
    "freeze_requirements",
    "claim_boundary",
}


class V030ProtocolError(ValueError):
    """Raised when the result-blind V2.10 amendment bytes drift."""


@dataclass(frozen=True, slots=True)
class ValidatedV030Design:
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
        raise V030ProtocolError(f"{context} must be a JSON object")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V030ProtocolError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V030ProtocolError(f"Nonfinite JSON constant: {token}")


def load_v030_design(
    path: str | Path = DEFAULT_V030_AMENDMENT_PATH,
) -> ValidatedV030Design:
    """Validate committed V2.10 bytes without deriving or consuming a seed."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V030ProtocolError("Cannot read V2.10 amendment JSON") from exc
    top = _object(payload, context="V2.10 amendment")
    canonical = json.dumps(
        top,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if (
        set(top) != _EXPECTED_KEYS
        or hashlib.sha256(raw).hexdigest() != V030_AMENDMENT_BYTE_SHA256
        or hashlib.sha256(canonical).hexdigest() != V030_AMENDMENT_SEMANTIC_SHA256
        or top.get("schema_version") != "lifetwin_synthetic_long_horizon_v2_10/1.0.0"
        or top.get("protocol_id") != V030_PROTOCOL_ID
        or top.get("implementation_profile") != "v0.25"
        or top.get("status") != V030_DESIGN_STATUS
    ):
        raise V030ProtocolError("V2.10 amendment identity or commitment changed")

    base = _object(top.get("base_contract"), context="base_contract")
    if (
        base.get("protocol_id") != "synthetic_long_horizon_identifiability_v2_9"
        or base.get("amendment_byte_sha256")
        != "175e9765c290f6c2718c8881bbf1324ba62ecbe2d2d71d2083a589025a743c8c"
        or base.get("amendment_semantic_sha256")
        != "0a606caad5e03cebacfdbe2df48d01d884c9485d0eb65de539d2804bcbceb8ee"
        or base.get("fixed_core_commit") != V030_FIXED_CORE_COMMIT
    ):
        raise V030ProtocolError("V2.10 base contract changed")

    attempt = _object(top.get("attempt_registry"), context="attempt_registry")
    if (
        attempt.get("only_attempt_id") != V030_ONLY_ATTEMPT_ID
        or attempt.get("maximum_attempts") != 1
        or attempt.get("a2_or_replacement_attempt_forbidden") is not True
        or attempt.get("formal_generation_before_freeze_forbidden") is not True
    ):
        raise V030ProtocolError("V2.10 attempt registry changed")

    fresh = _object(top.get("fresh_generation"), context="fresh_generation")
    roots = _object(fresh.get("seed_roots"), context="seed_roots")
    if (
        fresh.get("generation_has_started") is not False
        or fresh.get("seed_consumed") is not False
        or fresh.get("sealed_truth_created_or_opened") is not False
        or fresh.get("pilot_or_test_generation_forbidden") is not True
        or roots != dict(V030_EXPECTED_SEED_ROOTS)
    ):
        raise V030ProtocolError("V2.10 result-blind seed registry changed")
    predecessors = {
        *V2_SEED_ROOTS,
        *V021_SEED_ROOTS,
        *V022_SEED_ROOTS,
        *V023_SEED_ROOTS,
        *V024_EXPECTED_SEED_ROOTS.values(),
        *V025_EXPECTED_SEED_ROOTS.values(),
        *V026_EXPECTED_SEED_ROOTS.values(),
        *V027_EXPECTED_SEED_ROOTS.values(),
        *V028_EXPECTED_SEED_ROOTS.values(),
        *V029_EXPECTED_SEED_ROOTS.values(),
    }
    if predecessors.intersection(roots.values()) or len(set(roots.values())) != 13:
        raise V030ProtocolError("V2.10 seed roots collide or are not unique")

    checkpoint = _object(
        top.get("checkpoint_registry_contract"),
        context="checkpoint_registry_contract",
    )
    for stage, names in INPUT_FILENAMES_BY_STAGE.items():
        if checkpoint.get(f"{stage}_count") != len(names):
            raise V030ProtocolError(f"V2.10 {stage} checkpoint registry changed")
    if checkpoint.get("failure_rule") is None:
        raise V030ProtocolError("V2.10 checkpoint failure rule is absent")

    isolation = _object(top.get("path_isolation"), context="path_isolation")
    for role in ("label_free", "sealed_truth", "score", "termination"):
        expected = f"artifacts/{V030_ONLY_ATTEMPT_ID}-{role.replace('_', '-')}"
        if isolation.get(f"{role}_root") != expected:
            raise V030ProtocolError(f"V2.10 {role} root changed")
    if (
        isolation.get("formal_roots_must_be_absent_before_launch") is not True
        or isolation.get("resolved_roots_must_be_pairwise_distinct") is not True
    ):
        raise V030ProtocolError("V2.10 path isolation weakened")

    inheritance = _object(
        top.get("scientific_inheritance"), context="scientific_inheritance"
    )
    if len(inheritance.get("unchanged", ())) != 11:
        raise V030ProtocolError("V2.10 scientific inheritance changed")
    terminal = _object(top.get("terminal_rules"), context="terminal_rules")
    if (
        terminal.get("known_integrity_contract_mismatch_is_void") is not True
        or terminal.get("unknown_exception_preserves_unknown_default") is not True
        or terminal.get("prediction_and_terminal_registries_mutually_exclusive")
        is not True
    ):
        raise V030ProtocolError("V2.10 terminal rules changed")
    partition_view = _object(
        top.get("partition_contract_view"), context="partition_contract_view"
    )
    if (
        partition_view.get("naked_artifacts_fail_closed") is not True
        or partition_view.get("resolver_relaxation_forbidden") is not True
    ):
        raise V030ProtocolError("V2.10 authenticated-view contract changed")
    handoff = _object(
        top.get("prediction_capsule_identity_handoff_fix"),
        context="prediction_capsule_identity_handoff_fix",
    )
    if (
        handoff.get("required_constructor_keyword") != "protocol_id"
        or handoff.get("development_fix_commit") != V030_FIXED_CORE_COMMIT
        or handoff.get("scientific_outputs_unchanged") is not True
        or handoff.get("file_hash_and_membership_checks_unchanged") is not True
        or handoff.get("truth_capability_added") is not False
    ):
        raise V030ProtocolError("V2.10 prediction identity handoff changed")
    evidence = _object(
        top.get("result_blind_development_evidence"),
        context="result_blind_development_evidence",
    )
    if (
        evidence.get("missing_protocol_keyword_typeerror_reproduced") is not True
        or evidence.get("isolated_loader_handoff_regression_passed") is not True
        or evidence.get("prediction_lifecycle_regressions_passed") != 123
        or evidence.get("parent_child_capability_boundary_regressions_passed") != 15
        or evidence.get("formal_attempt_or_outcome_created") is not False
        or evidence.get("sealed_raw_or_formal_outputs_read") is not False
    ):
        raise V030ProtocolError("V2.10 result-blind development evidence changed")
    predecessor = _object(
        top.get("predecessor_terminal"), context="predecessor_terminal"
    )
    if (
        predecessor.get("attempt_id") != "v029-formal-20260814-a1"
        or predecessor.get("reason_code") != "UNKNOWN_PRE_PREDICTION_EXCEPTION"
        or predecessor.get("scientific_status") != "unclassified_terminal_not_success"
        or predecessor.get("last_completed_phase") != "model_state_committed"
        or predecessor.get("pending_phase") != "prediction_started"
        or predecessor.get("prediction_commitment_created") is not False
        or predecessor.get("score_created") is not False
        or predecessor.get("immutable_and_not_reusable") is not True
    ):
        raise V030ProtocolError("V2.9 terminal history changed")

    return ValidatedV030Design(
        protocol_id=V030_PROTOCOL_ID,
        status=V030_DESIGN_STATUS,
        config_path=config_path,
        config_byte_sha256=V030_AMENDMENT_BYTE_SHA256,
        config_semantic_sha256=V030_AMENDMENT_SEMANTIC_SHA256,
        seed_roots=MappingProxyType({str(k): int(v) for k, v in roots.items()}),
        raw=_deep_freeze(top),
    )


__all__ = [
    "DEFAULT_V030_AMENDMENT_PATH",
    "V030_AMENDMENT_BYTE_SHA256",
    "V030_AMENDMENT_SEMANTIC_SHA256",
    "V030_DESIGN_FREEZE_COMMIT",
    "V030_DESIGN_STATUS",
    "V030_EXPECTED_SEED_ROOTS",
    "V030_FIXED_CORE_COMMIT",
    "V030_ONLY_ATTEMPT_ID",
    "V030_PREREG_BYTE_SHA256",
    "V030_PROTOCOL_ID",
    "V030_REQUIREMENTS_BYTE_SHA256",
    "V030ProtocolError",
    "ValidatedV030Design",
    "load_v030_design",
]
