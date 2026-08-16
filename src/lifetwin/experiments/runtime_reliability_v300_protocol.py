"""Static validator for the preregistered V3.0 runtime-reliability study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


V300_PROTOCOL_ID = "lifetwin_structure_fit_runtime_reliability_v3_0"
V300_ONLY_ATTEMPT_ID = "v300-formal-20260815-a1"
V300_DESIGN_STATUS = "preregistered_pre_implementation"
V300_DESIGN_COMMIT = "e418d9916a014ee3b8ba416081ca5f0f90d09a06"
V300_PROTOCOL_COMMIT = "942acfa9b221da0d34d4411a76cb37c584293c1f"
V300_RUNTIME_HARDENING_COMMIT = "f4d067348ca9985d5f55601b09f9648256fbe7b1"
V300_FORMAL_SEED_ROOT = 202608153001
V300_DEVELOPMENT_SEED_ROOT = 31_000_000
V300_CONFIG_BYTE_SHA256 = (
    "b30dbe205923632daa4e68df8e6be222be7be6e7c74c28f6936743ae44f168b3"
)
V300_CONFIG_SEMANTIC_SHA256 = (
    "2c0c7c5e6361e0ace37f3149191c01a7a54685c4d546187d25800808dabdd9ee"
)
V300_PREREG_BYTE_SHA256 = (
    "3e601b0719929d431e58864b8ca7ad45d62662135b7ce6888b92c7bd97687caf"
)
V300_REQUIREMENTS_BYTE_SHA256 = (
    "6affd64bf8464f5449d3c70685504ff661f50933f8adb4d95abcbdf4aedafe85"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V300_CONFIG_PATH = (
    _PROJECT_ROOT / "configs" / "experiments" / "runtime_reliability_v3_0.json"
)
DEFAULT_V300_PREREG_PATH = (
    _PROJECT_ROOT / "reports" / "runtime_reliability_v3_0_preregistration.md"
)
DEFAULT_V300_REQUIREMENTS_PATH = _PROJECT_ROOT / "requirements" / "v300-formal.txt"
DEFAULT_V300_FREEZE_RECORD_PATH = (
    _PROJECT_ROOT / "reports" / "runtime_reliability_v3_0_freeze_record.json"
)

V300_EXPECTED_JOBS = (
    ("reference-96-serial", 96, 1),
    ("reference-96-parallel-a", 96, 6),
    ("reference-96-parallel-b", 96, 6),
    ("scale-1024-a", 1024, 6),
    ("scale-1024-b", 1024, 6),
    ("full-5950-a", 5950, 6),
    ("full-5950-b", 5950, 6),
)
V300_HASH_EQUIVALENCE_GROUPS = (
    (
        "reference-96-serial",
        "reference-96-parallel-a",
        "reference-96-parallel-b",
    ),
    ("scale-1024-a", "scale-1024-b"),
    ("full-5950-a", "full-5950-b"),
)
V300_EXPECTED_FAILURE_PHASES = MappingProxyType(
    {
        "pool_startup": "process_pool_construction",
        "worker_submission": "worker_submission",
        "worker_completion_wait": "worker_completion_wait",
        "worker_exception": "worker_future_result",
        "broken_process_pool": "broken_process_pool",
        "invalid_worker_output": "worker_output_validation",
        "executor_shutdown": "process_pool_shutdown",
        "verified_bundle_io": None,
    }
)

_EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "implementation_profile",
    "status",
    "design_date",
    "design_basis",
    "independence_contract",
    "attempt_registry",
    "seed_registry",
    "environment",
    "formal_paths",
    "workload",
    "primary_gates",
    "output_contract",
    "terminal_rules",
    "freeze_requirements",
    "claim_boundary",
}


class V300ProtocolError(ValueError):
    """Raised when the frozen V3.0 protocol identity or content drifts."""


@dataclass(frozen=True, slots=True)
class V300Job:
    job_id: str
    clusters: int
    workers: int


@dataclass(frozen=True, slots=True)
class V300PrimaryGates:
    full_scale_elapsed_seconds_maximum: int
    peak_process_tree_working_set_bytes_maximum: int
    peak_process_tree_private_bytes_maximum: int
    minimum_available_physical_memory_bytes_minimum: int
    resource_sampling_error_count_maximum: int
    peak_worker_process_count_maximum: int


@dataclass(frozen=True, slots=True)
class ValidatedV300Design:
    protocol_id: str
    status: str
    config_path: Path
    config_byte_sha256: str
    config_semantic_sha256: str
    jobs: tuple[V300Job, ...]
    hash_equivalence_groups: tuple[tuple[str, ...], ...]
    formal_seed_root: int
    development_seed_root: int
    attempt_root: Path
    authorization_record: Path
    primary_gates: V300PrimaryGates
    raw: Mapping[str, Any]


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V300ProtocolError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise V300ProtocolError(f"Nonfinite JSON constant: {token}")


def _object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise V300ProtocolError(f"{context} must be a JSON object")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _repo_path(value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise V300ProtocolError(f"{context} must be a nonempty POSIX-style path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise V300ProtocolError(f"{context} must stay inside the repository")
    resolved = (_PROJECT_ROOT / relative).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise V300ProtocolError(f"{context} escapes the repository") from exc
    return resolved


def load_v300_design(
    path: str | Path = DEFAULT_V300_CONFIG_PATH,
) -> ValidatedV300Design:
    """Load V3.0 without importing NumPy or consuming either seed namespace."""

    config_path = Path(path).resolve()
    try:
        raw_bytes = config_path.read_bytes()
        payload = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except V300ProtocolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V300ProtocolError("Cannot read V3.0 protocol JSON") from exc
    top = _object(payload, context="V3.0 protocol")
    canonical = json.dumps(
        top,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    semantic_hash = hashlib.sha256(canonical).hexdigest()
    if (
        set(top) != _EXPECTED_TOP_LEVEL_KEYS
        or raw_hash != V300_CONFIG_BYTE_SHA256
        or semantic_hash != V300_CONFIG_SEMANTIC_SHA256
        or top.get("schema_version") != "lifetwin_runtime_reliability_v3_0/1.0.0"
        or top.get("protocol_id") != V300_PROTOCOL_ID
        or top.get("implementation_profile") != "v3.0-runtime"
        or top.get("status") != V300_DESIGN_STATUS
    ):
        raise V300ProtocolError("V3.0 protocol identity or committed bytes changed")

    basis = _object(top.get("design_basis"), context="design_basis")
    if (
        basis.get("design_commit") != V300_DESIGN_COMMIT
        or basis.get("runtime_hardening_commit") != V300_RUNTIME_HARDENING_COMMIT
        or basis.get("design_record_byte_sha256")
        != "c5d702ea755fd2064fe4982403184e2f19b7e590aacb6543384010e3d2152900"
    ):
        raise V300ProtocolError("V3.0 design basis changed")

    attempt = _object(top.get("attempt_registry"), context="attempt_registry")
    if (
        attempt.get("only_attempt_id") != V300_ONLY_ATTEMPT_ID
        or attempt.get("maximum_attempts") != 1
        or attempt.get("a2_or_replacement_attempt_forbidden") is not True
        or attempt.get("partial_attempt_consumes_attempt") is not True
        or attempt.get("authorization_must_postdate_freeze") is not True
    ):
        raise V300ProtocolError("V3.0 one-shot attempt contract changed")

    seeds = _object(top.get("seed_registry"), context="seed_registry")
    if (
        seeds.get("formal_seed_root") != V300_FORMAL_SEED_ROOT
        or seeds.get("development_seed_root") != V300_DEVELOPMENT_SEED_ROOT
        or seeds.get("formal_seed_consumed") is not False
        or seeds.get("formal_seed_consumption_before_freeze_forbidden") is not True
        or seeds.get("formal_seed_consumption_by_tests_or_preflight_forbidden")
        is not True
    ):
        raise V300ProtocolError("V3.0 seed firewall changed")

    paths = _object(top.get("formal_paths"), context="formal_paths")
    attempt_root = _repo_path(paths.get("attempt_root"), context="attempt_root")
    authorization_record = _repo_path(
        paths.get("authorization_record"), context="authorization_record"
    )
    if (
        attempt_root == authorization_record
        or attempt_root in authorization_record.parents
        or paths.get("attempt_root_must_be_absent_before_authorized_launch") is not True
    ):
        raise V300ProtocolError("V3.0 path isolation changed")

    workload = _object(top.get("workload"), context="workload")
    jobs_raw = workload.get("jobs")
    if not isinstance(jobs_raw, list):
        raise V300ProtocolError("V3.0 jobs must be a JSON array")
    jobs: list[V300Job] = []
    for value in jobs_raw:
        job = _object(value, context="workload job")
        if set(job) != {"job_id", "clusters", "workers"}:
            raise V300ProtocolError("V3.0 job fields changed")
        jobs.append(
            V300Job(
                job_id=job["job_id"],
                clusters=job["clusters"],
                workers=job["workers"],
            )
        )
    observed_jobs = tuple((job.job_id, job.clusters, job.workers) for job in jobs)
    groups_raw = workload.get("hash_equivalence_groups")
    if not isinstance(groups_raw, list):
        raise V300ProtocolError("V3.0 hash groups must be a JSON array")
    groups = tuple(tuple(group) for group in groups_raw)
    failure_phases = _object(
        workload.get("failure_matrix_expected_runtime_phases"),
        context="failure_matrix_expected_runtime_phases",
    )
    if (
        observed_jobs != V300_EXPECTED_JOBS
        or groups != V300_HASH_EQUIVALENCE_GROUPS
        or failure_phases != dict(V300_EXPECTED_FAILURE_PHASES)
        or workload.get("suite") != "mixed"
        or workload.get("cluster_id_prefix") != "v300-formal-runtime"
        or workload.get("repeat_inside_each_child") != 1
    ):
        raise V300ProtocolError("V3.0 workload changed")

    gates_raw = _object(top.get("primary_gates"), context="primary_gates")
    gates = V300PrimaryGates(
        full_scale_elapsed_seconds_maximum=gates_raw.get(
            "full_scale_elapsed_seconds_maximum"
        ),
        peak_process_tree_working_set_bytes_maximum=gates_raw.get(
            "peak_process_tree_working_set_bytes_maximum"
        ),
        peak_process_tree_private_bytes_maximum=gates_raw.get(
            "peak_process_tree_private_bytes_maximum"
        ),
        minimum_available_physical_memory_bytes_minimum=gates_raw.get(
            "minimum_available_physical_memory_bytes_minimum"
        ),
        resource_sampling_error_count_maximum=gates_raw.get(
            "resource_sampling_error_count_maximum"
        ),
        peak_worker_process_count_maximum=gates_raw.get(
            "peak_worker_process_count_maximum"
        ),
    )
    if gates != V300PrimaryGates(
        full_scale_elapsed_seconds_maximum=7200,
        peak_process_tree_working_set_bytes_maximum=2_415_919_104,
        peak_process_tree_private_bytes_maximum=2_952_790_016,
        minimum_available_physical_memory_bytes_minimum=1_073_741_824,
        resource_sampling_error_count_maximum=0,
        peak_worker_process_count_maximum=6,
    ):
        raise V300ProtocolError("V3.0 primary thresholds changed")
    required_boolean_gates = (
        "all_normal_jobs_exit_zero",
        "all_normal_job_stderr_empty",
        "all_progress_records_terminal",
        "all_worker_exit_code_lists_empty",
        "exact_hash_equivalence_within_groups",
        "failure_matrix_exact_phase_set_required",
    )
    if any(gates_raw.get(name) is not True for name in required_boolean_gates):
        raise V300ProtocolError("V3.0 conjunctive primary gates weakened")

    output = _object(top.get("output_contract"), context="output_contract")
    if (
        output.get("child_probe_script") != "scripts/diagnose_v210_fit_spawn.py"
        or output.get("fault_matrix_script")
        != "scripts/diagnose_v210_failure_matrix.py"
        or output.get("result_blind_wrapper") != "scripts/run_result_blind_python.ps1"
        or output.get("formal_runner") != "scripts/run_runtime_reliability_v300.py"
        or output.get("terminal_record_is_atomic") is not True
    ):
        raise V300ProtocolError("V3.0 output contract changed")

    terminal = _object(top.get("terminal_rules"), context="terminal_rules")
    if (
        terminal.get("exactly_one_terminal_disposition") is not True
        or terminal.get("terminal_disposition_consumes_attempt") is not True
        or terminal.get("partial_artifacts_retained") is not True
    ):
        raise V300ProtocolError("V3.0 terminal rules changed")

    return ValidatedV300Design(
        protocol_id=V300_PROTOCOL_ID,
        status=V300_DESIGN_STATUS,
        config_path=config_path,
        config_byte_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        jobs=tuple(jobs),
        hash_equivalence_groups=groups,
        formal_seed_root=V300_FORMAL_SEED_ROOT,
        development_seed_root=V300_DEVELOPMENT_SEED_ROOT,
        attempt_root=attempt_root,
        authorization_record=authorization_record,
        primary_gates=gates,
        raw=_deep_freeze(top),
    )


__all__ = [
    "DEFAULT_V300_CONFIG_PATH",
    "DEFAULT_V300_FREEZE_RECORD_PATH",
    "DEFAULT_V300_PREREG_PATH",
    "DEFAULT_V300_REQUIREMENTS_PATH",
    "V300_CONFIG_BYTE_SHA256",
    "V300_CONFIG_SEMANTIC_SHA256",
    "V300_DEVELOPMENT_SEED_ROOT",
    "V300_DESIGN_COMMIT",
    "V300_DESIGN_STATUS",
    "V300_EXPECTED_FAILURE_PHASES",
    "V300_EXPECTED_JOBS",
    "V300_FORMAL_SEED_ROOT",
    "V300_HASH_EQUIVALENCE_GROUPS",
    "V300_ONLY_ATTEMPT_ID",
    "V300_PREREG_BYTE_SHA256",
    "V300_PROTOCOL_COMMIT",
    "V300_PROTOCOL_ID",
    "V300_REQUIREMENTS_BYTE_SHA256",
    "V300_RUNTIME_HARDENING_COMMIT",
    "V300Job",
    "V300PrimaryGates",
    "V300ProtocolError",
    "ValidatedV300Design",
    "load_v300_design",
]
