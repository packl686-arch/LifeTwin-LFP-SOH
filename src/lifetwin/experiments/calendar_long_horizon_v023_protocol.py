"""Static, non-generative validator for the preregistered V2.8 amendment."""

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


V028_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_8"
V028_ONLY_ATTEMPT_ID = "v028-formal-20260814-a1"
V028_DESIGN_STATUS = "implementation_frozen"
V028_EXPECTED_SEED_ROOTS = MappingProxyType(
    {
        "center_development": 202608140801,
        "risk_development": 202608140802,
        "calibration": 202608140803,
        "test": 202608140804,
        "audit": 202608140805,
        "novel_mechanism_test": 202608140806,
        "novel_mechanism_audit": 202608140807,
        "intrinsic_matched_pairs": 202608140808,
        "stress_plan_matched_pairs": 202608140809,
        "random_rankings": 202608140810,
        "bootstrap": 202608140811,
        "stress_permutations": 202608140812,
        "placebo_covariate": 202608140813,
    }
)
V028_AMENDMENT_BYTE_SHA256 = (
    "b5e93bac3e744cd6bfff09edf437f6a17d37c20fa0382f9546b8459dce740a1a"
)
V028_AMENDMENT_SEMANTIC_SHA256 = (
    "a27124fbb86307b8c02f5e7a011e7941a7a114e6acec5df041a7f0cbed6e99f1"
)
V028_PREREG_BYTE_SHA256 = (
    "a487b4ac6544bbb80d5ddaf71a3e86a85548cf957865e35ad6c9d0734ba22d3a"
)
V028_REQUIREMENTS_BYTE_SHA256 = (
    "0619ac43d21e48d3f78554b9e3d25ec270974f1fa987653951242748491534f5"
)
V028_DESIGN_FREEZE_COMMIT = "7c0018346831ef76d5079e98bc8db4b884a8e83b"
V028_FIXED_CORE_COMMIT = "411a6676e4f40defd16ea0403712c957833887a7"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V028_AMENDMENT_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_8_amendment.json"
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
    "risk_score_reduction_fix",
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


class V028ProtocolError(ValueError):
    """Raised when the result-blind V2.8 amendment bytes drift."""


@dataclass(frozen=True, slots=True)
class ValidatedV028Design:
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
        raise V028ProtocolError(f"{context} must be a JSON object")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V028ProtocolError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V028ProtocolError(f"Nonfinite JSON constant: {token}")


def load_v028_design(
    path: str | Path = DEFAULT_V028_AMENDMENT_PATH,
) -> ValidatedV028Design:
    """Validate committed V2.8 bytes without deriving or consuming a seed."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V028ProtocolError("Cannot read V2.8 amendment JSON") from exc
    top = _object(payload, context="V2.8 amendment")
    canonical = json.dumps(
        top,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if (
        set(top) != _EXPECTED_KEYS
        or hashlib.sha256(raw).hexdigest() != V028_AMENDMENT_BYTE_SHA256
        or hashlib.sha256(canonical).hexdigest() != V028_AMENDMENT_SEMANTIC_SHA256
        or top.get("schema_version") != "lifetwin_synthetic_long_horizon_v2_8/1.0.0"
        or top.get("protocol_id") != V028_PROTOCOL_ID
        or top.get("implementation_profile") != "v0.23"
        or top.get("status") != V028_DESIGN_STATUS
    ):
        raise V028ProtocolError("V2.8 amendment identity or commitment changed")

    base = _object(top.get("base_contract"), context="base_contract")
    if (
        base.get("protocol_id") != "synthetic_long_horizon_identifiability_v2_7"
        or base.get("amendment_byte_sha256")
        != "5669638e854d15dd0873ee863c93635f3f287753fa0b823c708f7e12a2c3d6b2"
        or base.get("amendment_semantic_sha256")
        != "d9e25ea634ff5bae3c03c6dbb0a329e994e480db80ef0621479a19023747c9cf"
        or base.get("fixed_core_commit") != V028_FIXED_CORE_COMMIT
    ):
        raise V028ProtocolError("V2.8 base contract changed")

    attempt = _object(top.get("attempt_registry"), context="attempt_registry")
    if (
        attempt.get("only_attempt_id") != V028_ONLY_ATTEMPT_ID
        or attempt.get("maximum_attempts") != 1
        or attempt.get("a2_or_replacement_attempt_forbidden") is not True
        or attempt.get("formal_generation_before_freeze_forbidden") is not True
    ):
        raise V028ProtocolError("V2.8 attempt registry changed")

    fresh = _object(top.get("fresh_generation"), context="fresh_generation")
    roots = _object(fresh.get("seed_roots"), context="seed_roots")
    if (
        fresh.get("generation_has_started") is not False
        or fresh.get("seed_consumed") is not False
        or fresh.get("sealed_truth_created_or_opened") is not False
        or fresh.get("pilot_or_test_generation_forbidden") is not True
        or roots != dict(V028_EXPECTED_SEED_ROOTS)
    ):
        raise V028ProtocolError("V2.8 result-blind seed registry changed")
    predecessors = {
        *V2_SEED_ROOTS,
        *V021_SEED_ROOTS,
        *V022_SEED_ROOTS,
        *V023_SEED_ROOTS,
        *V024_EXPECTED_SEED_ROOTS.values(),
        *V025_EXPECTED_SEED_ROOTS.values(),
        *V026_EXPECTED_SEED_ROOTS.values(),
        *V027_EXPECTED_SEED_ROOTS.values(),
    }
    if predecessors.intersection(roots.values()) or len(set(roots.values())) != 13:
        raise V028ProtocolError("V2.8 seed roots collide or are not unique")

    checkpoint = _object(
        top.get("checkpoint_registry_contract"),
        context="checkpoint_registry_contract",
    )
    for stage, names in INPUT_FILENAMES_BY_STAGE.items():
        if checkpoint.get(f"{stage}_count") != len(names):
            raise V028ProtocolError(f"V2.8 {stage} checkpoint registry changed")
    if checkpoint.get("failure_rule") is None:
        raise V028ProtocolError("V2.8 checkpoint failure rule is absent")

    isolation = _object(top.get("path_isolation"), context="path_isolation")
    for role in ("label_free", "sealed_truth", "score", "termination"):
        expected = f"artifacts/{V028_ONLY_ATTEMPT_ID}-{role.replace('_', '-')}"
        if isolation.get(f"{role}_root") != expected:
            raise V028ProtocolError(f"V2.8 {role} root changed")
    if (
        isolation.get("formal_roots_must_be_absent_before_launch") is not True
        or isolation.get("resolved_roots_must_be_pairwise_distinct") is not True
    ):
        raise V028ProtocolError("V2.8 path isolation weakened")

    inheritance = _object(
        top.get("scientific_inheritance"), context="scientific_inheritance"
    )
    if len(inheritance.get("unchanged", ())) != 10:
        raise V028ProtocolError("V2.8 scientific inheritance changed")
    terminal = _object(top.get("terminal_rules"), context="terminal_rules")
    if (
        terminal.get("known_integrity_contract_mismatch_is_void") is not True
        or terminal.get("unknown_exception_preserves_unknown_default") is not True
        or terminal.get("prediction_and_terminal_registries_mutually_exclusive")
        is not True
    ):
        raise V028ProtocolError("V2.8 terminal rules changed")
    partition_view = _object(
        top.get("partition_contract_view"), context="partition_contract_view"
    )
    if (
        partition_view.get("naked_artifacts_fail_closed") is not True
        or partition_view.get("resolver_relaxation_forbidden") is not True
    ):
        raise V028ProtocolError("V2.8 authenticated-view contract changed")
    reduction = _object(
        top.get("risk_score_reduction_fix"), context="risk_score_reduction_fix"
    )
    if (
        reduction.get("formula_dtype_threshold_and_tolerance_unchanged") is not True
        or reduction.get("row_and_batch_scores_bit_exact") is not True
        or reduction.get("primary_score_recomputation_remains_fail_closed") is not True
        or reduction.get("development_fix_commit") != V028_FIXED_CORE_COMMIT
    ):
        raise V028ProtocolError("V2.8 risk-score reduction fix changed")
    evidence = _object(
        top.get("result_blind_development_evidence"),
        context="result_blind_development_evidence",
    )
    if (
        evidence.get(
            "formal_cardinality_whole_calibration_recompute_and_repeated_mask_passed"
        )
        is not True
        or evidence.get("model_state_and_prediction_capsule_boundaries_verified")
        is not False
        or evidence.get("formal_attempt_or_outcome_created") is not False
        or evidence.get("large_fixture_and_operator_paths_are_not_tracked") is not True
    ):
        raise V028ProtocolError("V2.8 result-blind development evidence changed")
    predecessor = _object(
        top.get("predecessor_terminal"), context="predecessor_terminal"
    )
    if (
        predecessor.get("attempt_id") != "v027-formal-20260813-a1"
        or predecessor.get("reason_code") != "UNKNOWN_PRE_PREDICTION_EXCEPTION"
        or predecessor.get("scientific_status") != "unclassified_terminal_not_success"
        or predecessor.get("opened_truth_files")
        != ["center_development_truth.csv", "risk_development_truth.csv"]
        or predecessor.get("prediction_commitment_created") is not False
        or predecessor.get("score_created") is not False
        or predecessor.get("immutable_and_not_reusable") is not True
    ):
        raise V028ProtocolError("V2.7 terminal history changed")

    return ValidatedV028Design(
        protocol_id=V028_PROTOCOL_ID,
        status=V028_DESIGN_STATUS,
        config_path=config_path,
        config_byte_sha256=V028_AMENDMENT_BYTE_SHA256,
        config_semantic_sha256=V028_AMENDMENT_SEMANTIC_SHA256,
        seed_roots=MappingProxyType({str(k): int(v) for k, v in roots.items()}),
        raw=_deep_freeze(top),
    )


__all__ = [
    "DEFAULT_V028_AMENDMENT_PATH",
    "V028_AMENDMENT_BYTE_SHA256",
    "V028_AMENDMENT_SEMANTIC_SHA256",
    "V028_DESIGN_FREEZE_COMMIT",
    "V028_DESIGN_STATUS",
    "V028_EXPECTED_SEED_ROOTS",
    "V028_FIXED_CORE_COMMIT",
    "V028_ONLY_ATTEMPT_ID",
    "V028_PREREG_BYTE_SHA256",
    "V028_PROTOCOL_ID",
    "V028_REQUIREMENTS_BYTE_SHA256",
    "V028ProtocolError",
    "ValidatedV028Design",
    "load_v028_design",
]
