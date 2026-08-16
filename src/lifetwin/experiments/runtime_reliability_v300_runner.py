"""One-shot orchestration and attestation for the V3.0 runtime study."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

from lifetwin.experiments.runtime_reliability_v300_protocol import (
    DEFAULT_V300_PREREG_PATH,
    DEFAULT_V300_REQUIREMENTS_PATH,
    V300_CONFIG_BYTE_SHA256,
    V300_CONFIG_SEMANTIC_SHA256,
    V300_DESIGN_COMMIT,
    V300_EXPECTED_FAILURE_PHASES,
    V300_ONLY_ATTEMPT_ID,
    V300_PREREG_BYTE_SHA256,
    V300_PROTOCOL_COMMIT,
    V300_PROTOCOL_ID,
    V300_REQUIREMENTS_BYTE_SHA256,
    V300Job,
    ValidatedV300Design,
    load_v300_design,
)


V300_FREEZE_METADATA_PATHS = frozenset(
    {
        "reports/runtime_reliability_v3_0_implementation_audit.md",
        "reports/runtime_reliability_v3_0_nonformal_validation.json",
        "reports/runtime_reliability_v3_0_freeze_record.json",
        "reports/runtime_reliability_v3_0_freeze_bundle.json",
    }
)
_EXPECTED_PYTHON = (3, 12, 13)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_FULL_GIT_HASH = re.compile(r"[0-9a-f]{40}")
_AUTHORIZATION_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}")
_NORMAL_HASH_KEYS = {
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
}
_NORMAL_RECORD_KEYS = {
    "payload",
    "progress",
    "exit_manifest",
    "wrapper_exit_code",
    "stderr_byte_count",
}
_NORMAL_PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "phase",
    "clusters",
    "workers",
    "repeat",
    "suite",
    "elapsed_seconds",
    "repeat_elapsed_seconds",
    "diagnostic_rows",
    "forecast_rows",
    "hashes",
    "runtime_failure_telemetry",
    "worker_exit_codes",
    "resource_telemetry",
    "formal_inputs_used",
    "formal_rows_opened",
    "formal_seeds_used",
    "sealed_truth_opened",
}
_RESOURCE_KEYS = {
    "backend",
    "sample_count",
    "sampling_error_count",
    "peak_process_tree_process_count",
    "peak_worker_process_count",
    "peak_process_tree_working_set_bytes",
    "peak_process_tree_private_bytes",
    "minimum_available_physical_memory_bytes",
    "disk_free_start_bytes",
    "disk_free_end_bytes",
}
_EXIT_MANIFEST_KEYS = {
    "schema_version",
    "wrapper_status",
    "process_exit_code",
    "timed_out",
    "launch_exception_class",
    "started_utc",
    "finished_utc",
    "elapsed_seconds",
    "script_sha256",
    "stdout_byte_count",
    "stdout_sha256",
    "stderr_byte_count",
    "stderr_sha256",
}
_FORBIDDEN_RESULT_KEYS = {
    "cluster_id",
    "observed_retention_pct",
    "forecast_day",
    "exception_message",
    "process_id",
    "sealed_truth",
    "score",
}
_DETERMINISTIC_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class V300RuntimeError(RuntimeError):
    """Base class for a V3.0 orchestration failure."""


class V300IntegrityError(V300RuntimeError):
    """Raised before interpretation when integrity cannot be established."""


class V300ExecutionError(V300RuntimeError):
    """Raised when an authorized child process cannot yield a valid result."""


@dataclass(frozen=True, slots=True)
class V300PreflightReport:
    status: str
    protocol_id: str
    attempt_id: str
    freeze_commit: str
    implementation_commit: str
    git_clean: bool
    environment_verified: bool
    requirements_verified: bool
    formal_root_absent: bool
    authorization_status: str
    available_physical_memory_bytes: int
    disk_free_bytes: int
    powershell_executable: str


@dataclass(frozen=True, slots=True)
class V300Evaluation:
    disposition: str
    passed: bool
    gates: Mapping[str, bool]
    failed_gates: tuple[str, ...]


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise V300IntegrityError("Atomic output staging path already exists")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(payload)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise V300IntegrityError("Exclusive record already exists") from exc


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise V300IntegrityError("Cannot hash required file") from exc


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V300IntegrityError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    del token
    raise V300IntegrityError("JSON contains a nonfinite constant")


def _json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except V300IntegrityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise V300IntegrityError(f"{context} is not strict JSON") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise V300IntegrityError(f"{context} is not a JSON object")
    return value


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise V300IntegrityError(f"Cannot read {context}") from exc
    return _json_object(raw, context=context)


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise V300IntegrityError("Git attestation command failed")
    return completed.stdout.strip()


def source_tree_sha256(repo_root: Path, paths: Iterable[str]) -> str:
    """Hash an ordered path-and-byte stream for a declared source set."""

    digest = hashlib.sha256()
    for relative in sorted(paths):
        if not relative or "\\" in relative or ".." in Path(relative).parts:
            raise V300IntegrityError("Invalid source-tree path")
        path = repo_root / relative
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise V300IntegrityError("Cannot read bound source file") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_blob_sha256(repo_root: Path, revision: str, path: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{revision}:{path}"),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise V300IntegrityError("Cannot read bound Git blob")
    return _sha256_bytes(completed.stdout)


def verify_v300_freeze(repo_root: Path) -> tuple[str, str]:
    """Verify the direct-child freeze and every implementation source hash."""

    root = repo_root.resolve()
    design = load_v300_design()
    record_path = root / "reports" / "runtime_reliability_v3_0_freeze_record.json"
    record = _read_json_object(record_path, context="V3.0 freeze record")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "review_status",
        "created_utc",
        "design_commit",
        "protocol_commit",
        "implementation_commit",
        "implementation_parent_commit",
        "direct_child_attestation_required",
        "config_byte_sha256",
        "config_semantic_sha256",
        "preregistration_byte_sha256",
        "requirements_byte_sha256",
        "implementation_paths",
        "implementation_source_byte_hashes",
        "implementation_source_tree_sha256",
        "freeze_metadata_paths",
        "formal_seed_consumed_before_freeze",
        "formal_attempt_created_before_freeze",
        "authorization_created_before_freeze",
        "formal_root_created_before_freeze",
        "nonformal_validation_status",
    }
    if (
        set(record) != expected_keys
        or record.get("schema_version") != "lifetwin_v300_freeze_record/1.0.0"
        or record.get("protocol_id") != V300_PROTOCOL_ID
        or record.get("review_status") != "implementation_frozen"
        or record.get("design_commit") != V300_DESIGN_COMMIT
        or record.get("protocol_commit") != V300_PROTOCOL_COMMIT
        or record.get("implementation_parent_commit") != V300_PROTOCOL_COMMIT
        or record.get("direct_child_attestation_required") is not True
        or record.get("config_byte_sha256") != V300_CONFIG_BYTE_SHA256
        or record.get("config_semantic_sha256") != V300_CONFIG_SEMANTIC_SHA256
        or record.get("preregistration_byte_sha256") != V300_PREREG_BYTE_SHA256
        or record.get("requirements_byte_sha256") != V300_REQUIREMENTS_BYTE_SHA256
        or record.get("formal_seed_consumed_before_freeze") is not False
        or record.get("formal_attempt_created_before_freeze") is not False
        or record.get("authorization_created_before_freeze") is not False
        or record.get("formal_root_created_before_freeze") is not False
        or record.get("nonformal_validation_status") != "passed"
    ):
        raise V300IntegrityError("V3.0 freeze record identity changed")
    implementation = record.get("implementation_commit")
    if (
        not isinstance(implementation, str)
        or _FULL_GIT_HASH.fullmatch(implementation) is None
    ):
        raise V300IntegrityError("V3.0 implementation commit is invalid")
    if _git(root, "rev-parse", f"{implementation}^") != V300_PROTOCOL_COMMIT:
        raise V300IntegrityError("V3.0 P-to-I topology is not direct")
    head = _git(root, "rev-parse", "HEAD")
    if _git(root, "rev-parse", "HEAD^") != implementation:
        raise V300IntegrityError("V3.0 I-to-F topology is not direct")
    changed = frozenset(
        line
        for line in _git(root, "diff", "--name-only", implementation, head).splitlines()
        if line
    )
    declared_metadata = record.get("freeze_metadata_paths")
    if (
        not isinstance(declared_metadata, list)
        or frozenset(declared_metadata) != V300_FREEZE_METADATA_PATHS
        or changed != V300_FREEZE_METADATA_PATHS
    ):
        raise V300IntegrityError("V3.0 freeze metadata allowlist changed")
    implementation_paths = record.get("implementation_paths")
    source_hashes = record.get("implementation_source_byte_hashes")
    if (
        not isinstance(implementation_paths, list)
        or not implementation_paths
        or not isinstance(source_hashes, dict)
        or set(source_hashes) != set(implementation_paths)
        or any(not isinstance(path, str) for path in implementation_paths)
    ):
        raise V300IntegrityError("V3.0 implementation source registry is invalid")
    changed_implementation = {
        line
        for line in _git(
            root,
            "diff",
            "--name-only",
            V300_PROTOCOL_COMMIT,
            implementation,
        ).splitlines()
        if line
    }
    if changed_implementation != set(implementation_paths):
        raise V300IntegrityError("V3.0 P-to-I implementation allowlist changed")
    for relative in implementation_paths:
        expected = source_hashes.get(relative)
        if not isinstance(expected, str) or _HEX_SHA256.fullmatch(expected) is None:
            raise V300IntegrityError("V3.0 implementation source hash is invalid")
        if (
            _sha256_file(root / relative) != expected
            or _git_blob_sha256(root, implementation, relative) != expected
        ):
            raise V300IntegrityError("V3.0 implementation source bytes drifted")
    if source_tree_sha256(root, implementation_paths) != record.get(
        "implementation_source_tree_sha256"
    ):
        raise V300IntegrityError("V3.0 implementation source tree drifted")
    validation_path = (
        root / "reports" / "runtime_reliability_v3_0_nonformal_validation.json"
    )
    validation = _read_json_object(
        validation_path,
        context="V3.0 nonformal validation",
    )
    boundary = validation.get("boundary")
    if (
        validation.get("schema_version") != "lifetwin_v300_nonformal_validation/1.0.0"
        or validation.get("protocol_id") != V300_PROTOCOL_ID
        or validation.get("implementation_commit") != implementation
        or validation.get("status") != "passed"
        or not isinstance(boundary, dict)
        or boundary.get("formal_inputs_used") is not False
        or boundary.get("formal_rows_opened") is not False
        or boundary.get("formal_seeds_used") is not False
        or boundary.get("sealed_truth_opened") is not False
        or boundary.get("formal_attempt_created") is not False
    ):
        raise V300IntegrityError("V3.0 nonformal validation boundary changed")
    audit_path = root / "reports" / "runtime_reliability_v3_0_implementation_audit.md"
    bundle = _read_json_object(
        root / "reports" / "runtime_reliability_v3_0_freeze_bundle.json",
        context="V3.0 freeze bundle",
    )
    if (
        set(bundle)
        != {
            "schema_version",
            "protocol_id",
            "review_status",
            "protocol_commit",
            "implementation_commit",
            "config_byte_sha256",
            "config_semantic_sha256",
            "preregistration_byte_sha256",
            "requirements_byte_sha256",
            "implementation_source_tree_sha256",
            "freeze_record_sha256",
            "implementation_audit_sha256",
            "nonformal_validation_sha256",
            "formal_seed_consumed_before_freeze",
            "formal_attempt_created_before_freeze",
            "authorization_created_before_freeze",
            "formal_root_created_before_freeze",
        }
        or bundle.get("schema_version") != "lifetwin_v300_freeze_bundle/1.0.0"
        or bundle.get("protocol_id") != V300_PROTOCOL_ID
        or bundle.get("review_status") != "implementation_frozen"
        or bundle.get("protocol_commit") != V300_PROTOCOL_COMMIT
        or bundle.get("implementation_commit") != implementation
        or bundle.get("config_byte_sha256") != V300_CONFIG_BYTE_SHA256
        or bundle.get("config_semantic_sha256") != V300_CONFIG_SEMANTIC_SHA256
        or bundle.get("preregistration_byte_sha256") != V300_PREREG_BYTE_SHA256
        or bundle.get("requirements_byte_sha256") != V300_REQUIREMENTS_BYTE_SHA256
        or bundle.get("implementation_source_tree_sha256")
        != record.get("implementation_source_tree_sha256")
        or bundle.get("freeze_record_sha256") != _sha256_file(record_path)
        or bundle.get("implementation_audit_sha256") != _sha256_file(audit_path)
        or bundle.get("nonformal_validation_sha256") != _sha256_file(validation_path)
        or bundle.get("formal_seed_consumed_before_freeze") is not False
        or bundle.get("formal_attempt_created_before_freeze") is not False
        or bundle.get("authorization_created_before_freeze") is not False
        or bundle.get("formal_root_created_before_freeze") is not False
    ):
        raise V300IntegrityError("V3.0 freeze bundle identity changed")
    if design.config_byte_sha256 != V300_CONFIG_BYTE_SHA256:
        raise V300IntegrityError("V3.0 design attestation drifted")
    if _sha256_file(root / DEFAULT_V300_PREREG_PATH.relative_to(root)) != (
        V300_PREREG_BYTE_SHA256
    ):
        raise V300IntegrityError("V3.0 preregistration drifted")
    if _sha256_file(root / DEFAULT_V300_REQUIREMENTS_PATH.relative_to(root)) != (
        V300_REQUIREMENTS_BYTE_SHA256
    ):
        raise V300IntegrityError("V3.0 requirements lock drifted")
    if _git(root, "status", "--porcelain"):
        raise V300IntegrityError("V3.0 frozen checkout is dirty")
    return head, implementation


def _locked_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise V300IntegrityError("Cannot read V3.0 requirements lock") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, version = stripped.partition("==")
        if not separator or not name or not version or name.lower() in requirements:
            raise V300IntegrityError("V3.0 requirements lock is not exact")
        requirements[name.lower()] = version
    return requirements


def _verify_packages(requirements_path: Path) -> None:
    for package, expected in _locked_requirements(requirements_path).items():
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise V300IntegrityError("A frozen package is absent") from exc
        if observed != expected:
            raise V300IntegrityError("A frozen package version drifted")


def _available_physical_memory_bytes() -> int:
    if sys.platform != "win32":
        raise V300IntegrityError("V3.0 formal execution requires Windows")

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise V300IntegrityError("Cannot attest available physical memory")
    return int(status.ullAvailPhys)


def _parse_utc(value: object, *, context: str) -> datetime:
    if not isinstance(value, str):
        raise V300IntegrityError(f"{context} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise V300IntegrityError(f"{context} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise V300IntegrityError(f"{context} timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _validate_authorization(
    path: Path,
    *,
    freeze_commit: str,
    freeze_created_utc: datetime,
) -> dict[str, Any]:
    record = _read_json_object(path, context="V3.0 authorization record")
    if (
        set(record)
        != {
            "schema_version",
            "protocol_id",
            "attempt_id",
            "freeze_commit",
            "authorization_status",
            "authorization_id",
            "authorized_by",
            "authorized_utc",
            "statement",
        }
        or record.get("schema_version") != "lifetwin_v300_authorization/1.0.0"
        or record.get("protocol_id") != V300_PROTOCOL_ID
        or record.get("attempt_id") != V300_ONLY_ATTEMPT_ID
        or record.get("freeze_commit") != freeze_commit
        or record.get("authorization_status") != "authorized_post_freeze"
        or record.get("statement")
        != "Authorize the sole frozen V3.0 formal attempt; any launch consumes it."
        or not isinstance(record.get("authorization_id"), str)
        or _AUTHORIZATION_TOKEN.fullmatch(record["authorization_id"]) is None
        or not isinstance(record.get("authorized_by"), str)
        or not record["authorized_by"].strip()
        or len(record["authorized_by"]) > 256
    ):
        raise V300IntegrityError("V3.0 authorization identity changed")
    if _parse_utc(record.get("authorized_utc"), context="authorization") <= (
        freeze_created_utc
    ):
        raise V300IntegrityError("V3.0 authorization does not postdate freeze")
    return record


def preflight_v300(
    repo_root: Path,
    *,
    require_authorization: bool = False,
) -> V300PreflightReport:
    """Verify frozen execution readiness without creating rows or consuming a seed."""

    root = repo_root.resolve()
    design = load_v300_design()
    freeze_commit, implementation = verify_v300_freeze(root)
    freeze_record = _read_json_object(
        root / "reports" / "runtime_reliability_v3_0_freeze_record.json",
        context="V3.0 freeze record",
    )
    freeze_created = max(
        _parse_utc(freeze_record.get("created_utc"), context="freeze record"),
        _parse_utc(
            _git(root, "show", "-s", "--format=%cI", freeze_commit),
            context="freeze commit",
        ),
    )
    if (
        sys.platform != "win32"
        or platform.python_implementation() != "CPython"
        or sys.version_info[:3] != _EXPECTED_PYTHON
        or platform.architecture()[0] != "64bit"
    ):
        raise V300IntegrityError("V3.0 runtime identity does not match the lock")
    prereg_path = root / "reports" / "runtime_reliability_v3_0_preregistration.md"
    requirements_path = root / "requirements" / "v300-formal.txt"
    if (
        _sha256_file(prereg_path) != V300_PREREG_BYTE_SHA256
        or _sha256_file(requirements_path) != V300_REQUIREMENTS_BYTE_SHA256
    ):
        raise V300IntegrityError("V3.0 protocol support bytes drifted")
    _verify_packages(requirements_path)
    powershell = shutil.which("pwsh")
    if powershell is None:
        raise V300IntegrityError("PowerShell 7 is unavailable")
    if design.attempt_root.exists():
        raise V300IntegrityError("V3.0 formal attempt root already exists")
    available_memory = _available_physical_memory_bytes()
    minimum_memory = design.raw["environment"][
        "minimum_prelaunch_available_physical_memory_bytes"
    ]
    if available_memory < minimum_memory:
        raise V300IntegrityError("Prelaunch physical memory gate failed")
    disk_free = shutil.disk_usage(root.anchor or root).free
    minimum_disk = design.raw["environment"]["minimum_prelaunch_disk_free_bytes"]
    if disk_free < minimum_disk:
        raise V300IntegrityError("Prelaunch disk gate failed")
    if design.authorization_record.exists():
        _validate_authorization(
            design.authorization_record,
            freeze_commit=freeze_commit,
            freeze_created_utc=freeze_created,
        )
        authorization_status = "authorized_post_freeze"
    else:
        authorization_status = "pending_post_freeze_authorization"
    if require_authorization and authorization_status != "authorized_post_freeze":
        raise V300IntegrityError("Post-freeze authorization is absent")
    return V300PreflightReport(
        status=(
            "ready"
            if authorization_status == "authorized_post_freeze"
            else "ready_pending_authorization"
        ),
        protocol_id=V300_PROTOCOL_ID,
        attempt_id=V300_ONLY_ATTEMPT_ID,
        freeze_commit=freeze_commit,
        implementation_commit=implementation,
        git_clean=True,
        environment_verified=True,
        requirements_verified=True,
        formal_root_absent=True,
        authorization_status=authorization_status,
        available_physical_memory_bytes=available_memory,
        disk_free_bytes=disk_free,
        powershell_executable=powershell,
    )


def create_v300_authorization(
    repo_root: Path,
    *,
    authorization_id: str,
    authorized_by: str,
    now: datetime | None = None,
) -> Path:
    """Create the fixed authorization record after an explicit external decision."""

    if _AUTHORIZATION_TOKEN.fullmatch(authorization_id) is None:
        raise V300IntegrityError("Authorization ID is invalid")
    if not authorized_by.strip() or len(authorized_by) > 256:
        raise V300IntegrityError("Authorizer identity is invalid")
    report = preflight_v300(repo_root, require_authorization=False)
    design = load_v300_design()
    if design.authorization_record.exists():
        raise V300IntegrityError("V3.0 authorization record already exists")
    authorized_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    freeze_record = _read_json_object(
        repo_root.resolve() / "reports" / "runtime_reliability_v3_0_freeze_record.json",
        context="V3.0 freeze record",
    )
    freeze_boundary = max(
        _parse_utc(freeze_record.get("created_utc"), context="freeze record"),
        _parse_utc(
            _git(
                repo_root.resolve(), "show", "-s", "--format=%cI", report.freeze_commit
            ),
            context="freeze commit",
        ),
    )
    if authorized_utc <= freeze_boundary:
        raise V300IntegrityError("Authorization time must postdate freeze")
    payload = {
        "schema_version": "lifetwin_v300_authorization/1.0.0",
        "protocol_id": V300_PROTOCOL_ID,
        "attempt_id": V300_ONLY_ATTEMPT_ID,
        "freeze_commit": report.freeze_commit,
        "authorization_status": "authorized_post_freeze",
        "authorization_id": authorization_id,
        "authorized_by": authorized_by.strip(),
        "authorized_utc": authorized_utc.isoformat().replace("+00:00", "Z"),
        "statement": (
            "Authorize the sole frozen V3.0 formal attempt; any launch consumes it."
        ),
    }
    _exclusive_json(design.authorization_record, payload)
    return design.authorization_record


def _keys_are_result_blind(value: object) -> bool:
    if isinstance(value, Mapping):
        if any(key in _FORBIDDEN_RESULT_KEYS for key in value):
            return False
        return all(_keys_are_result_blind(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_keys_are_result_blind(item) for item in value)
    return True


def _normal_record_is_valid(job: V300Job, record: object) -> bool:
    if not isinstance(record, Mapping) or set(record) != _NORMAL_RECORD_KEYS:
        return False
    payload = record.get("payload")
    progress = record.get("progress")
    exit_manifest = record.get("exit_manifest")
    if not all(
        isinstance(value, Mapping) for value in (payload, progress, exit_manifest)
    ):
        return False
    hashes = payload.get("hashes")
    resources = payload.get("resource_telemetry")
    if not isinstance(hashes, Mapping) or set(hashes) != _NORMAL_HASH_KEYS:
        return False
    if any(
        not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None
        for value in hashes.values()
    ):
        return False
    if not isinstance(resources, Mapping):
        return False
    elapsed = payload.get("elapsed_seconds")
    return (
        payload.get("status") == "passed"
        and set(payload) == _NORMAL_PAYLOAD_KEYS
        and set(progress) == _NORMAL_PAYLOAD_KEYS
        and set(exit_manifest) == _EXIT_MANIFEST_KEYS
        and set(resources) == _RESOURCE_KEYS
        and payload.get("phase") == "completed"
        and payload.get("clusters") == job.clusters
        and payload.get("workers") == job.workers
        and payload.get("suite") == "mixed"
        and payload.get("repeat") == 1
        and payload.get("diagnostic_rows") == job.clusters * 86
        and payload.get("forecast_rows") == job.clusters * 86 * 8
        and payload.get("worker_exit_codes") == []
        and payload.get("runtime_failure_telemetry") is None
        and payload.get("formal_inputs_used") is True
        and payload.get("formal_rows_opened") is False
        and payload.get("formal_seeds_used") is True
        and payload.get("sealed_truth_opened") is False
        and type(elapsed) in (int, float)
        and math.isfinite(elapsed)
        and elapsed >= 0.0
        and dict(progress) == dict(payload)
        and exit_manifest.get("wrapper_status") == "completed"
        and exit_manifest.get("process_exit_code") == 0
        and exit_manifest.get("timed_out") is False
        and record.get("wrapper_exit_code") == 0
        and record.get("stderr_byte_count") == 0
        and _keys_are_result_blind(record)
    )


def _resource_gate(
    normal_records: Mapping[str, object],
    jobs: tuple[V300Job, ...],
    key: str,
    predicate: Any,
) -> bool:
    values: list[object] = []
    for job in jobs:
        record = normal_records.get(job.job_id)
        if not isinstance(record, Mapping):
            return False
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            return False
        resources = payload.get("resource_telemetry")
        if not isinstance(resources, Mapping):
            return False
        values.append(resources.get(key))
    return all(type(value) is int and predicate(value) for value in values)


def evaluate_v300_attempt(
    design: ValidatedV300Design,
    normal_records: Mapping[str, object],
    failure_matrix: object,
) -> V300Evaluation:
    """Apply only the preregistered conjunctive gates to sanitized records."""

    jobs = design.jobs
    job_ids = {job.job_id for job in jobs}
    records_exact = set(normal_records) == job_ids
    normal_valid = records_exact and all(
        _normal_record_is_valid(job, normal_records.get(job.job_id)) for job in jobs
    )
    hashes_equal = normal_valid
    if hashes_equal:
        for group in design.hash_equivalence_groups:
            anchors = [normal_records[job_id]["payload"]["hashes"] for job_id in group]
            if any(item != anchors[0] for item in anchors[1:]):
                hashes_equal = False
                break
    full_scale_elapsed = normal_valid and all(
        normal_records[job.job_id]["payload"]["elapsed_seconds"]
        <= design.primary_gates.full_scale_elapsed_seconds_maximum
        for job in jobs
        if job.clusters == 5950
    )
    working_set = _resource_gate(
        normal_records,
        jobs,
        "peak_process_tree_working_set_bytes",
        lambda value: 0
        <= value
        <= design.primary_gates.peak_process_tree_working_set_bytes_maximum,
    )
    private_bytes = _resource_gate(
        normal_records,
        jobs,
        "peak_process_tree_private_bytes",
        lambda value: 0
        <= value
        <= design.primary_gates.peak_process_tree_private_bytes_maximum,
    )
    available_memory = _resource_gate(
        normal_records,
        jobs,
        "minimum_available_physical_memory_bytes",
        lambda value: 0
        < value
        >= design.primary_gates.minimum_available_physical_memory_bytes_minimum,
    )
    sampling = _resource_gate(
        normal_records,
        jobs,
        "sampling_error_count",
        lambda value: 0
        <= value
        <= design.primary_gates.resource_sampling_error_count_maximum,
    )
    workers = _resource_gate(
        normal_records,
        jobs,
        "peak_worker_process_count",
        lambda value: 0
        <= value
        <= design.primary_gates.peak_worker_process_count_maximum,
    )

    failure_matrix_valid = False
    failure_blind = False
    if isinstance(failure_matrix, Mapping):
        cases_raw = failure_matrix.get("cases")
        if isinstance(cases_raw, list):
            cases = {
                case.get("case"): case
                for case in cases_raw
                if isinstance(case, Mapping) and isinstance(case.get("case"), str)
            }
            expected = dict(V300_EXPECTED_FAILURE_PHASES)
            failure_matrix_valid = (
                failure_matrix.get("status") == "passed"
                and len(cases_raw) == len(expected)
                and len(cases) == len(expected)
                and set(cases) == set(expected)
                and all(
                    case.get("status") == "expected_failure_observed"
                    and (
                        case.get("runtime_failure_telemetry") is None
                        if expected[name] is None
                        else isinstance(case.get("runtime_failure_telemetry"), Mapping)
                        and case["runtime_failure_telemetry"].get("phase")
                        == expected[name]
                    )
                    for name, case in cases.items()
                )
            )
        failure_blind = (
            failure_matrix.get("result_blind") is True
            and failure_matrix.get("formal_inputs_used") is False
            and failure_matrix.get("formal_rows_opened") is False
            and failure_matrix.get("formal_seeds_used") is False
            and failure_matrix.get("sealed_truth_opened") is False
            and _keys_are_result_blind(failure_matrix)
        )

    gates = {
        "normal_record_contract": normal_valid,
        "hash_equivalence": hashes_equal,
        "full_scale_elapsed": full_scale_elapsed,
        "working_set_ceiling": working_set,
        "private_bytes_ceiling": private_bytes,
        "available_memory_floor": available_memory,
        "resource_sampling": sampling,
        "worker_count_ceiling": workers,
        "failure_matrix_exact_phases": failure_matrix_valid,
        "failure_matrix_result_blind": failure_blind,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    return V300Evaluation(
        disposition="success" if not failed else "operational_failure",
        passed=not failed,
        gates=gates,
        failed_gates=failed,
    )


def _verify_exit_manifest(
    manifest: Mapping[str, object],
    *,
    script_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    if (
        manifest.get("script_sha256") != _sha256_file(script_path)
        or manifest.get("stdout_byte_count") != stdout_path.stat().st_size
        or manifest.get("stderr_byte_count") != stderr_path.stat().st_size
        or manifest.get("stdout_sha256") != _sha256_file(stdout_path)
        or manifest.get("stderr_sha256") != _sha256_file(stderr_path)
    ):
        raise V300IntegrityError("Result-blind wrapper manifest drifted")


def _run_wrapped(
    *,
    preflight: V300PreflightReport,
    repo_root: Path,
    script_path: Path,
    output_root: Path,
    script_arguments: tuple[str, ...],
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    output_root.mkdir(parents=True, exist_ok=False)
    stdout_path = output_root / "stdout.json"
    stderr_path = output_root / "stderr.txt"
    exit_path = output_root / "exit_manifest.json"
    wrapper = repo_root / "scripts" / "run_result_blind_python.ps1"
    command = (
        preflight.powershell_executable,
        "-NoProfile",
        "-File",
        str(wrapper),
        "-Python",
        sys.executable,
        "-Script",
        str(script_path),
        "-WorkingDirectory",
        str(repo_root),
        "-StdoutPath",
        str(stdout_path),
        "-StderrPath",
        str(stderr_path),
        "-ExitManifestPath",
        str(exit_path),
        "-TimeoutSeconds",
        str(timeout_seconds),
        *script_arguments,
    )
    child_environment = os.environ.copy()
    child_environment.update(_DETERMINISTIC_ENVIRONMENT)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child_environment,
    )
    if completed.stdout or completed.stderr or not exit_path.is_file():
        raise V300IntegrityError("Result-blind wrapper emitted an invalid boundary")
    manifest = _read_json_object(exit_path, context="wrapper exit manifest")
    _verify_exit_manifest(
        manifest,
        script_path=script_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    try:
        payload = _read_json_object(stdout_path, context="child stdout")
    except V300IntegrityError as exc:
        raise V300ExecutionError("Child process produced no valid result") from exc
    return payload, manifest, completed.returncode


def _normal_job_record(
    *,
    preflight: V300PreflightReport,
    design: ValidatedV300Design,
    repo_root: Path,
    attempt_root: Path,
    job: V300Job,
) -> dict[str, object]:
    job_root = attempt_root / "jobs" / job.job_id
    progress_path = job_root / "progress.json"
    script_path = repo_root / str(design.raw["output_contract"]["child_probe_script"])
    payload, manifest, wrapper_exit_code = _run_wrapped(
        preflight=preflight,
        repo_root=repo_root,
        script_path=script_path,
        output_root=job_root,
        timeout_seconds=design.primary_gates.full_scale_elapsed_seconds_maximum,
        script_arguments=(
            "--clusters",
            str(job.clusters),
            "--workers",
            str(job.workers),
            "--repeat",
            "1",
            "--suite",
            "mixed",
            "--seed-root",
            str(design.formal_seed_root),
            "--cluster-prefix",
            str(design.raw["workload"]["cluster_id_prefix"]),
            "--execution-profile",
            "v300-formal",
            "--authorization-record",
            str(design.authorization_record),
            "--progress-file",
            str(progress_path),
        ),
    )
    try:
        progress = _read_json_object(progress_path, context="child progress")
    except V300IntegrityError as exc:
        raise V300ExecutionError("Child process produced no valid progress") from exc
    return {
        "payload": payload,
        "progress": progress,
        "exit_manifest": manifest,
        "wrapper_exit_code": wrapper_exit_code,
        "stderr_byte_count": (job_root / "stderr.txt").stat().st_size,
    }


def _failure_matrix_record(
    *,
    preflight: V300PreflightReport,
    design: ValidatedV300Design,
    repo_root: Path,
    attempt_root: Path,
) -> dict[str, object]:
    script_path = repo_root / str(design.raw["output_contract"]["fault_matrix_script"])
    payload, manifest, wrapper_exit_code = _run_wrapped(
        preflight=preflight,
        repo_root=repo_root,
        script_path=script_path,
        output_root=attempt_root / "failure-matrix",
        script_arguments=(),
        timeout_seconds=300,
    )
    if (
        manifest.get("wrapper_status") != "completed"
        or manifest.get("process_exit_code") != 0
        or manifest.get("timed_out") is not False
        or wrapper_exit_code != 0
        or manifest.get("stderr_byte_count") != 0
    ):
        raise V300ExecutionError("The failure matrix child did not complete")
    return payload


def _artifact_manifest(attempt_root: Path) -> dict[str, object]:
    manifest_path = attempt_root / "artifact_manifest.json"
    files = {
        path.relative_to(attempt_root).as_posix(): {
            "byte_count": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(attempt_root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    return {
        "schema_version": "lifetwin_v300_artifact_manifest/1.0.0",
        "protocol_id": V300_PROTOCOL_ID,
        "attempt_id": V300_ONLY_ATTEMPT_ID,
        "files": files,
    }


def execute_v300_formal_attempt(repo_root: Path) -> dict[str, object]:
    """Consume the sole authorized attempt and always try to commit a terminal record."""

    root = repo_root.resolve()
    design = load_v300_design()
    preflight = preflight_v300(root, require_authorization=True)
    attempt_root = design.attempt_root
    attempt_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    normal_records: dict[str, object] = {}
    failure_matrix: object = None
    disposition = "integrity_void"
    evaluation: V300Evaluation | None = None
    failure_identity: dict[str, str] | None = None
    progress_path = attempt_root / "attempt_progress.json"
    _atomic_json(
        progress_path,
        {
            "schema_version": "lifetwin_v300_attempt_progress/1.0.0",
            "protocol_id": V300_PROTOCOL_ID,
            "attempt_id": V300_ONLY_ATTEMPT_ID,
            "status": "in_progress",
            "completed_jobs": [],
            "formal_seed_consumed": False,
        },
    )
    try:
        for job in design.jobs:
            normal_records[job.job_id] = _normal_job_record(
                preflight=preflight,
                design=design,
                repo_root=root,
                attempt_root=attempt_root,
                job=job,
            )
            _atomic_json(
                progress_path,
                {
                    "schema_version": "lifetwin_v300_attempt_progress/1.0.0",
                    "protocol_id": V300_PROTOCOL_ID,
                    "attempt_id": V300_ONLY_ATTEMPT_ID,
                    "status": "in_progress",
                    "completed_jobs": list(normal_records),
                    "formal_seed_consumed": True,
                },
            )
        failure_matrix = _failure_matrix_record(
            preflight=preflight,
            design=design,
            repo_root=root,
            attempt_root=attempt_root,
        )
        evaluation = evaluate_v300_attempt(design, normal_records, failure_matrix)
        disposition = evaluation.disposition
    except KeyboardInterrupt as exc:
        disposition = "interrupted_inconclusive"
        failure_identity = {
            "phase": "formal_orchestration",
            "exception_class": type(exc).__name__,
        }
    except V300ExecutionError as exc:
        disposition = "operational_failure"
        failure_identity = {
            "phase": "formal_orchestration",
            "exception_class": type(exc).__name__,
        }
    except Exception as exc:
        disposition = "integrity_void"
        failure_identity = {
            "phase": "formal_orchestration",
            "exception_class": type(exc).__name__,
        }
    finished = datetime.now(timezone.utc)
    terminal = {
        "schema_version": "lifetwin_v300_terminal_record/1.0.0",
        "protocol_id": V300_PROTOCOL_ID,
        "attempt_id": V300_ONLY_ATTEMPT_ID,
        "freeze_commit": preflight.freeze_commit,
        "authorization_record_sha256": _sha256_file(design.authorization_record),
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_utc": finished.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "disposition": disposition,
        "formal_seed_consumed": bool(normal_records),
        "completed_jobs": list(normal_records),
        "evaluation": asdict(evaluation) if evaluation is not None else None,
        "failure_identity": failure_identity,
        "normal_records": normal_records,
        "failure_matrix": failure_matrix,
    }
    _exclusive_json(attempt_root / "terminal_record.json", terminal)
    _atomic_json(
        progress_path,
        {
            "schema_version": "lifetwin_v300_attempt_progress/1.0.0",
            "protocol_id": V300_PROTOCOL_ID,
            "attempt_id": V300_ONLY_ATTEMPT_ID,
            "status": "terminal",
            "disposition": disposition,
            "completed_jobs": list(normal_records),
            "formal_seed_consumed": bool(normal_records),
        },
    )
    _exclusive_json(
        attempt_root / "artifact_manifest.json",
        _artifact_manifest(attempt_root),
    )
    return terminal


__all__ = [
    "V300Evaluation",
    "V300ExecutionError",
    "V300IntegrityError",
    "V300PreflightReport",
    "V300RuntimeError",
    "V300_FREEZE_METADATA_PATHS",
    "create_v300_authorization",
    "evaluate_v300_attempt",
    "execute_v300_formal_attempt",
    "preflight_v300",
    "source_tree_sha256",
    "verify_v300_freeze",
]
