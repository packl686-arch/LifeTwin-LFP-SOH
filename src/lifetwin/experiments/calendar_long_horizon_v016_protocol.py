"""Static validator for the V2.1 design amendment.

This module validates design bytes only. It has no generator, optimizer, model
fit, prediction, scoring, or truth-reading entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


V021_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_1"
V021_ALLOWED_DESIGN_STATUSES = (
    "design_candidate_preimplementation",
    "implementation_candidate_unfrozen",
    "implementation_frozen",
)
V021_SOURCE_CALIBRATION_COUNT = 900
V021_MINIMUM_RISK_ISOTONIC_ELIGIBLE_COUNT = 855
V021_MINIMUM_CALIBRATION_CLASS_COUNT = 60
V021_CONFORMAL_COUNT = 900
V021_CONFORMAL_ORDER_STATISTIC_INDEX = 811
V2_SEED_ROOTS = tuple(range(202607230101, 202607230114))
V021_EXPECTED_SEED_ROOTS = MappingProxyType(
    {
        "center_development": 202607260201,
        "risk_development": 202607260202,
        "calibration": 202607260203,
        "test": 202607260204,
        "audit": 202607260205,
        "novel_mechanism_test": 202607260206,
        "novel_mechanism_audit": 202607260207,
        "intrinsic_matched_pairs": 202607260208,
        "stress_plan_matched_pairs": 202607260209,
        "random_rankings": 202607260210,
        "bootstrap": 202607260211,
        "stress_permutations": 202607260212,
        "placebo_covariate": 202607260213,
    }
)
BASE_CONFIG_BYTE_SHA256 = (
    "27dc7f89178f73779a52068c1878df26c9686faa7433686e60ba6496b6705796"
)
BASE_PREREGISTRATION_BYTE_SHA256 = (
    "c1dee9f9b4ef134b1a52e9a51300c591e790c10a0e97b3fe6c15eb441b2c09f0"
)
TERMINATED_ATTEMPT_MANIFEST_SHA256 = (
    "5b2b2d300653d070ed107b67a1a11b4edc10a0d33b2bad491ef1be784e0f4b09"
)
V021_AMENDMENT_SEMANTIC_SHA256 = (
    "2dd77ef9f9393cc982fb370f3ed8e5f7d1753a0f7f311d5b2cc24d01e2acbbde"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V021_AMENDMENT_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_1_amendment.json"
)
_BASE_CONFIG_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2.json"
)
_BASE_PREREGISTRATION_PATH = (
    _PROJECT_ROOT / "reports" / "synthetic_long_horizon_identifiability_prereg_v2.md"
)
_TERMINATED_ATTEMPT_MANIFEST_PATH = (
    _PROJECT_ROOT
    / "showcase"
    / "evidence_v015"
    / "synthetic_long_horizon_identifiability_v2"
    / "formal_attempt_termination_manifest.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "title",
        "design_date",
        "design_witnesses",
        "base_protocol",
        "exposure_disclosure",
        "evidence_reuse_policy",
        "fresh_generation",
        "calibration_population_split",
        "artifact_registries",
        "terminal_reason_codes",
        "unchanged_v2_decision_rules",
        "implementation_and_freeze_requirements",
    }
)
_EXPECTED_SCORED_ARTIFACTS = frozenset(
    {
        "point_scores.csv",
        "trajectory_scores.csv",
        "family_metrics.csv",
        "matched_pair_scores.csv",
        "bootstrap_replicates.csv",
        "random_ranking_metrics.csv",
        "stress_permutation_metrics.csv",
        "negative_control_metrics.json",
        "score_report.json",
        "run_manifest.json",
    }
)
_EXPECTED_TERMINAL_ARTIFACTS = frozenset(
    {
        "terminal_attempt_record.json",
        "terminal_artifact_manifest.json",
        "terminal_exposure_log_snapshot.jsonl",
    }
)


class V021ProtocolError(ValueError):
    """Raised when the design amendment drifts from its declared boundaries."""


@dataclass(frozen=True)
class ValidatedV021Design:
    """Validated, non-executable V2.1 amendment identity."""

    protocol_id: str
    status: str
    config_path: Path
    config_byte_sha256: str
    seed_roots: Mapping[str, int]
    raw: Mapping[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise V021ProtocolError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    value: object,
    *,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, Any]:
    result = _object(value, context=context)
    if set(result) != expected:
        raise V021ProtocolError(
            f"{context} keys changed: observed={sorted(result)}, "
            f"expected={sorted(expected)}"
        )
    return result


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V021ProtocolError(f"{context} must be an integer")
    return value


def _string_set(value: object, *, context: str) -> frozenset[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise V021ProtocolError(f"{context} must be a nonempty-string array")
    if len(value) != len(set(value)):
        raise V021ProtocolError(f"{context} contains duplicates")
    return frozenset(value)


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _validate_predecessor_hashes(base: Mapping[str, Any]) -> None:
    expected = {
        "config_byte_sha256": BASE_CONFIG_BYTE_SHA256,
        "preregistration_byte_sha256": BASE_PREREGISTRATION_BYTE_SHA256,
        "terminated_attempt_manifest_sha256": (TERMINATED_ATTEMPT_MANIFEST_SHA256),
    }
    for key, expected_hash in expected.items():
        observed = base.get(key)
        if observed != expected_hash or _SHA256.fullmatch(str(observed)) is None:
            raise V021ProtocolError(f"base_protocol.{key} changed")
    actual = {
        "config_byte_sha256": _sha256(_BASE_CONFIG_PATH),
        "preregistration_byte_sha256": _sha256(_BASE_PREREGISTRATION_PATH),
        "terminated_attempt_manifest_sha256": _sha256(
            _TERMINATED_ATTEMPT_MANIFEST_PATH
        ),
    }
    if actual != expected:
        raise V021ProtocolError("Predecessor evidence bytes changed")


def _validate_fresh_generation(value: object, *, status: str) -> Mapping[str, int]:
    generation = _object(value, context="fresh_generation")
    if generation.get("generation_has_started") is not False:
        raise V021ProtocolError("V2.1 generation must remain unstarted")
    if generation.get("implementation_freeze_required_before_generation") is not True:
        raise V021ProtocolError("Implementation freeze gate was relaxed")
    if status == "design_candidate_preimplementation":
        expected_implementation = False
    else:
        expected_implementation = True
    if generation.get("implementation_exists") is not expected_implementation:
        raise V021ProtocolError("Status and implementation_exists disagree")
    roots = _object(generation.get("seed_roots"), context="fresh_generation.seed_roots")
    if roots != V021_EXPECTED_SEED_ROOTS:
        raise V021ProtocolError("Fresh seed roots changed")
    integer_roots = {
        key: _integer(value, context=f"fresh_generation.seed_roots.{key}")
        for key, value in roots.items()
    }
    if len(set(integer_roots.values())) != len(integer_roots):
        raise V021ProtocolError("Fresh seed roots collide with each other")
    if set(integer_roots.values()).intersection(V2_SEED_ROOTS):
        raise V021ProtocolError("Fresh seed roots collide with V2")
    return MappingProxyType(integer_roots)


def _validate_calibration_policy(value: object) -> None:
    split = _object(value, context="calibration_population_split")
    if split.get("source_calibration_count") != V021_SOURCE_CALIBRATION_COUNT:
        raise V021ProtocolError("Source calibration count changed")
    risk = _object(
        split.get("risk_isotonic"),
        context="calibration_population_split.risk_isotonic",
    )
    expected_risk = {
        "minimum_eligible_count": V021_MINIMUM_RISK_ISOTONIC_ELIGIBLE_COUNT,
        "minimum_positive_labels": V021_MINIMUM_CALIBRATION_CLASS_COUNT,
        "minimum_negative_labels": V021_MINIMUM_CALIBRATION_CLASS_COUNT,
        "same_mask_for_both_primary_heads": True,
        "arm_specific_exclusion_or_refill_forbidden": True,
        "class_counts_are_within_mask": True,
    }
    for key, expected in expected_risk.items():
        if risk.get(key) != expected:
            raise V021ProtocolError(f"risk_isotonic.{key} changed")
    exact_mask = risk.get("exact_mask")
    if not isinstance(exact_mask, list) or len(exact_mask) != 8:
        raise V021ProtocolError("risk_isotonic exact mask must have eight clauses")

    baseline = _object(
        split.get("mean_baseline"),
        context="calibration_population_split.mean_baseline",
    )
    if baseline.get("denominator") != V021_SOURCE_CALIBRATION_COUNT:
        raise V021ProtocolError("Mean-baseline denominator changed")
    if baseline.get("row_exclusion_forbidden") is not True:
        raise V021ProtocolError("Mean-baseline row exclusion was enabled")

    conformal = _object(
        split.get("conformal"),
        context="calibration_population_split.conformal",
    )
    expected_conformal = {
        "denominator": V021_CONFORMAL_COUNT,
        "required_finite_scores": V021_CONFORMAL_COUNT,
        "order_statistic_index_one_based": (V021_CONFORMAL_ORDER_STATISTIC_INDEX),
        "coverage": 0.9,
    }
    for key, expected in expected_conformal.items():
        if conformal.get(key) != expected:
            raise V021ProtocolError(f"conformal.{key} changed")
    if "Do not synthesize" not in str(conformal.get("zero_family_rule")):
        raise V021ProtocolError("Zero-family no-imputation rule changed")
    if "sole successful family" not in str(conformal.get("one_family_rule")):
        raise V021ProtocolError("One-family genuine-support rule changed")


def _validate_artifact_registries(value: object) -> None:
    registries = _object(value, context="artifact_registries")
    scored = _object(registries.get("scored"), context="artifact_registries.scored")
    terminal = _object(
        registries.get("terminal_pre_prediction"),
        context="artifact_registries.terminal_pre_prediction",
    )
    scored_files = _string_set(
        scored.get("filenames"),
        context="artifact_registries.scored.filenames",
    )
    terminal_files = _string_set(
        terminal.get("filenames"),
        context="artifact_registries.terminal_pre_prediction.filenames",
    )
    if scored_files != _EXPECTED_SCORED_ARTIFACTS:
        raise V021ProtocolError("Scored artifact registry changed")
    if terminal_files != _EXPECTED_TERMINAL_ARTIFACTS:
        raise V021ProtocolError("Terminal artifact registry changed")
    if scored_files.intersection(terminal_files):
        raise V021ProtocolError("Scored and terminal registries overlap")


def _validate_reason_registries(value: object) -> None:
    reasons = _object(value, context="terminal_reason_codes")
    groups = {}
    for group in ("declared_inconclusive", "integrity_void", "interruption", "unknown"):
        groups[group] = _string_set(
            reasons.get(group),
            context=f"terminal_reason_codes.{group}",
        )
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if groups[left].intersection(groups[right]):
                raise V021ProtocolError("Terminal reason registries overlap")
    if not all(item.startswith("INTEGRITY_") for item in groups["integrity_void"]):
        raise V021ProtocolError("Integrity reason prefix changed")
    if groups["unknown"] != {"UNKNOWN_PRE_PREDICTION_EXCEPTION"}:
        raise V021ProtocolError("Unknown-exception registry changed")
    if reasons.get("free_text_is_not_a_reason_code") is not True:
        raise V021ProtocolError("Free-text reason codes were enabled")


def _amendment_semantic_sha256(payload: Mapping[str, Any]) -> str:
    """Hash every design field except the paired implementation-status switch."""

    normalized = json.loads(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    normalized["status"] = "<IMPLEMENTATION_STATUS>"
    normalized["fresh_generation"]["implementation_exists"] = "<IMPLEMENTATION_EXISTS>"
    canonical = json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _validate_v021_payload(
    decoded: object,
    *,
    config_path: Path,
    raw_bytes: bytes,
) -> ValidatedV021Design:
    """Validate already-decoded design bytes without any experimental action."""

    top = _exact_keys(decoded, expected=_TOP_LEVEL_KEYS, context="V2.1 amendment")
    if top.get("schema_version") != "1.0.0":
        raise V021ProtocolError("V2.1 schema version changed")
    if top.get("protocol_id") != V021_PROTOCOL_ID:
        raise V021ProtocolError("V2.1 protocol ID changed")
    status = top.get("status")
    if status not in V021_ALLOWED_DESIGN_STATUSES:
        raise V021ProtocolError("V2.1 implementation status is not recognized")
    witnesses = top.get("design_witnesses")
    if witnesses != ["Jincheng Liu"]:
        raise V021ProtocolError("V2.1 design witness changed")

    base = _object(top.get("base_protocol"), context="base_protocol")
    if base.get("protocol_id") != "synthetic_long_horizon_identifiability_v2":
        raise V021ProtocolError("Base protocol ID changed")
    if base.get("terminated_attempt_status") != "inconclusive_not_success":
        raise V021ProtocolError("Terminated V2 disposition changed")
    _validate_predecessor_hashes(base)

    exposure = _object(top.get("exposure_disclosure"), context="exposure_disclosure")
    if exposure.get("prediction_commitment_existed") is not False:
        raise V021ProtocolError("Exposure disclosure added a prediction commitment")
    if exposure.get("score_package_existed") is not False:
        raise V021ProtocolError("Exposure disclosure added a score package")
    if len(exposure.get("opened_development_truth", ())) != 3:
        raise V021ProtocolError("Opened development-truth disclosure changed")
    if len(exposure.get("unopened_heldout_truth", ())) != 6:
        raise V021ProtocolError("Unopened heldout-truth disclosure changed")

    reuse = _object(top.get("evidence_reuse_policy"), context="evidence_reuse_policy")
    forbidden = reuse.get("forbidden_v2_reuse")
    if not isinstance(forbidden, list) or len(forbidden) < 5:
        raise V021ProtocolError("V2 row/state reuse prohibition was weakened")

    roots = _validate_fresh_generation(top.get("fresh_generation"), status=str(status))
    _validate_calibration_policy(top.get("calibration_population_split"))
    _validate_artifact_registries(top.get("artifact_registries"))
    _validate_reason_registries(top.get("terminal_reason_codes"))
    if _amendment_semantic_sha256(top) != V021_AMENDMENT_SEMANTIC_SHA256:
        raise V021ProtocolError(
            "V2.1 amendment semantic content changed outside the paired "
            "implementation-status switch"
        )

    return ValidatedV021Design(
        protocol_id=V021_PROTOCOL_ID,
        status=str(status),
        config_path=config_path,
        config_byte_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        seed_roots=roots,
        raw=_deep_freeze_json(top),
    )


def load_v021_design(
    path: str | Path = DEFAULT_V021_AMENDMENT_PATH,
) -> ValidatedV021Design:
    """Load and validate the non-executable V2.1 amendment."""

    config_path = Path(path).resolve()
    try:
        raw_bytes = config_path.read_bytes()
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V021ProtocolError("Cannot read V2.1 amendment JSON") from exc
    return _validate_v021_payload(
        decoded,
        config_path=config_path,
        raw_bytes=raw_bytes,
    )
