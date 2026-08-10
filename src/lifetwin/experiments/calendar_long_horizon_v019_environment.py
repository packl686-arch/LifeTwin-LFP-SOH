"""Fail-closed implementation and environment guard for the V2.4 attempt."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from types import MappingProxyType
from typing import Mapping

from threadpoolctl import threadpool_info

from lifetwin.experiments.calendar_long_horizon_v019_protocol import (
    V024_AMENDMENT_SEMANTIC_SHA256,
    V024_PROTOCOL_ID,
    load_v024_design,
)


class V024EnvironmentError(RuntimeError):
    """Raised when a formal process is not the attested V2.4 implementation."""


_EXPECTED_PYTHON = (3, 12, 13)
_EXPECTED_PACKAGES = MappingProxyType(
    {
        "duckdb": "1.5.4",
        "joblib": "1.5.3",
        "jsonschema": "4.25.1",
        "numpy": "2.5.1",
        "pandas": "3.0.3",
        "scikit-learn": "1.9.0",
        "scipy": "1.18.0",
        "threadpoolctl": "3.6.0",
    }
)
_LOCK_RELATIVE_PATH = Path("requirements") / "v024-formal.txt"
_LOCK_BYTE_SHA256 = "15c7e4cbe9ee300a097409553695f9286012a2739ec8280ff3c2bca9570f97b3"
_AMENDMENT_RELATIVE_PATH = (
    Path("configs")
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_4_amendment.json"
)
_PREREG_RELATIVE_PATH = (
    Path("reports") / "synthetic_long_horizon_identifiability_prereg_v2_4.md"
)
_FREEZE_RECORD_RELATIVE_PATH = (
    Path("reports") / "synthetic_long_horizon_identifiability_freeze_record_v2_4.json"
)
_IMPLEMENTATION_AUDIT_RELATIVE_PATH = (
    Path("reports")
    / "synthetic_long_horizon_identifiability_implementation_audit_v2_4.md"
)
_IMPLEMENTATION_METADATA_CHANGES = tuple(
    sorted(
        {
            _IMPLEMENTATION_AUDIT_RELATIVE_PATH.as_posix(),
            _FREEZE_RECORD_RELATIVE_PATH.as_posix(),
        }
    )
)
_PREREG_BYTE_SHA256 = "23ed77ff3facb94e836393e2a01b60917a04fdfa7235fd2cfb69b6b0300181ca"
_IMPLEMENTATION_AUDIT_BYTE_SHA256 = (
    "43263f9fedce7757a156e95d6d37381ff3a7d2681ba0aa65e56637801df47659"
)
_DESIGN_FREEZE_COMMIT = "f70c6dfa3e32d578ff96b73cc7cb7648527033bf"
_EXPECTED_HASH_SENTINEL = -1939000917158325260
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_FULL_GIT_HASH = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


@dataclass(frozen=True, slots=True)
class FormalEnvironmentIdentity:
    protocol_id: str
    git_commit: str
    git_dirty: bool
    config_byte_sha256: str
    config_canonical_sha256: str
    preregistration_byte_sha256: str
    environment_lock_byte_sha256: str
    python_version: str
    python_implementation: str
    platform: str
    machine: str
    processor: str
    package_versions: Mapping[str, str]
    deterministic_environment: Mapping[str, str]
    source_byte_hashes: Mapping[str, str]
    active_threadpools: tuple[Mapping[str, object], ...]

    def as_manifest_record(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_byte_sha256": self.config_byte_sha256,
            "config_canonical_sha256": self.config_canonical_sha256,
            "preregistration_byte_sha256": self.preregistration_byte_sha256,
            "environment_lock_byte_sha256": self.environment_lock_byte_sha256,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "package_versions": dict(self.package_versions),
            "deterministic_environment": dict(self.deterministic_environment),
            "source_byte_hashes": dict(self.source_byte_hashes),
            "active_threadpools": [dict(item) for item in self.active_threadpools],
        }


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise V024EnvironmentError(f"Cannot verify git {' '.join(arguments)}") from exc
    return result.stdout.strip()


def _source_hashes(repo_root: Path) -> Mapping[str, str]:
    experiment_root = repo_root / "src" / "lifetwin" / "experiments"
    inherited = sorted(experiment_root.glob("calendar_long_horizon_v015_*.py"))
    amended = sorted(experiment_root.glob("calendar_long_horizon_v019_*.py"))
    shared = experiment_root / "calendar_long_horizon_synthetic.py"
    scripts = sorted((repo_root / "scripts").glob("*v019*.py"))
    if not inherited or not amended or not shared.is_file():
        raise V024EnvironmentError(
            "The complete V2.4 and inherited V2 implementation source set is absent"
        )
    paths = tuple(sorted({*inherited, *amended, shared, *scripts}))
    if any(not path.is_file() for path in paths):
        raise V024EnvironmentError("A V2.4 implementation source is not a file")
    return MappingProxyType(
        {path.relative_to(repo_root).as_posix(): _sha256_path(path) for path in paths}
    )


def _implementation_source_tree_sha256(source_hashes: Mapping[str, str]) -> str:
    raw = (
        json.dumps(
            dict(source_hashes),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _strict_json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V024EnvironmentError(f"{context} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise V024EnvironmentError(f"{context} must be a JSON object")
    return decoded


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V024EnvironmentError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise V024EnvironmentError(f"Nonfinite JSON constant: {token}")


def _validate_freeze_record(
    path: Path,
    *,
    amendment_byte_sha256: str,
    source_hashes: Mapping[str, str],
) -> str:
    payload = _strict_json_object(path, context="V2.4 freeze record")
    expected = {
        "schema_version": "1.0.0",
        "protocol_id": V024_PROTOCOL_ID,
        "design_freeze_commit": _DESIGN_FREEZE_COMMIT,
        "amendment_path": _AMENDMENT_RELATIVE_PATH.as_posix(),
        "amendment_byte_sha256": amendment_byte_sha256,
        "amendment_semantic_sha256": V024_AMENDMENT_SEMANTIC_SHA256,
        "preregistration_path": _PREREG_RELATIVE_PATH.as_posix(),
        "preregistration_byte_sha256": _PREREG_BYTE_SHA256,
        "environment_lock_path": _LOCK_RELATIVE_PATH.as_posix(),
        "environment_lock_byte_sha256": _LOCK_BYTE_SHA256,
        "implementation_audit_path": _IMPLEMENTATION_AUDIT_RELATIVE_PATH.as_posix(),
        "implementation_audit_byte_sha256": _IMPLEMENTATION_AUDIT_BYTE_SHA256,
        "formal_v2_4_generation_executed_before_implementation_freeze": False,
        "v2_4_outcome_exposure_before_implementation_freeze": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise V024EnvironmentError(
            "V2.4 freeze record does not bind the frozen protocol and environment"
        )
    implementation_commit = payload.get("implementation_source_commit")
    recorded_source_hashes = payload.get("implementation_source_byte_hashes")
    if (
        payload.get("review_status") != "implementation_frozen"
        or not isinstance(implementation_commit, str)
        or _FULL_GIT_HASH.fullmatch(implementation_commit) is None
        or set(implementation_commit) == {"0"}
        or not isinstance(recorded_source_hashes, dict)
        or recorded_source_hashes != dict(source_hashes)
        or payload.get("implementation_source_tree_sha256")
        != _implementation_source_tree_sha256(source_hashes)
        or payload.get("execution_metadata_paths")
        != list(_IMPLEMENTATION_METADATA_CHANGES)
    ):
        raise V024EnvironmentError(
            "V2.4 freeze record does not bind the frozen implementation"
        )
    return implementation_commit


def _runtime_hash_sentinel() -> int:
    return hash("lifetwin-v024-formal")


def _active_threadpool_records() -> tuple[Mapping[str, object], ...]:
    records: list[dict[str, object]] = []
    for pool in threadpool_info():
        threads = pool.get("num_threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads != 1:
            raise V024EnvironmentError(
                "A loaded BLAS/OpenMP thread pool is not locked to one thread"
            )
        records.append(
            {
                "user_api": str(pool.get("user_api", "")),
                "internal_api": str(pool.get("internal_api", "")),
                "prefix": str(pool.get("prefix", "")),
                "version": str(pool.get("version", "")),
                "num_threads": threads,
                "filepath": str(pool.get("filepath", "")),
            }
        )
    if not records:
        raise V024EnvironmentError(
            "No loaded BLAS/OpenMP thread pool could be verified"
        )
    records.sort(
        key=lambda item: (
            str(item["user_api"]),
            str(item["internal_api"]),
            str(item["prefix"]),
            str(item["filepath"]),
        )
    )
    return tuple(MappingProxyType(item) for item in records)


def _verify_paths_are_tracked(repo_root: Path, paths: tuple[Path, ...]) -> None:
    relative = tuple(path.relative_to(repo_root).as_posix() for path in paths)
    observed = set(
        _git(
            repo_root,
            "ls-files",
            "--error-unmatch",
            "--",
            *relative,
        ).splitlines()
    )
    if observed != set(relative):
        raise V024EnvironmentError(
            "A formal freeze, lock, or implementation source is not git-tracked"
        )


def verify_formal_environment(
    repo_root: str | Path,
) -> FormalEnvironmentIdentity:
    """Verify the frozen status, attestation lineage, source bytes, and runtime."""

    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise V024EnvironmentError("Formal repository root has no .git directory")
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise V024EnvironmentError("Formal repository root is not the git top level")
    git_commit = _git(root, "rev-parse", "HEAD")
    if _FULL_GIT_HASH.fullmatch(git_commit) is None:
        raise V024EnvironmentError("Git did not return a full commit hash")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise V024EnvironmentError("Formal implementation worktree is dirty")

    amendment_path = root / _AMENDMENT_RELATIVE_PATH
    prereg_path = root / _PREREG_RELATIVE_PATH
    freeze_record_path = root / _FREEZE_RECORD_RELATIVE_PATH
    implementation_audit_path = root / _IMPLEMENTATION_AUDIT_RELATIVE_PATH
    lock_path = root / _LOCK_RELATIVE_PATH
    for path in (
        amendment_path,
        prereg_path,
        freeze_record_path,
        implementation_audit_path,
        lock_path,
    ):
        if not path.is_file():
            raise V024EnvironmentError(f"Required freeze file is absent: {path}")
    try:
        design = load_v024_design(amendment_path)
    except (OSError, ValueError) as exc:
        raise V024EnvironmentError("V2.4 amendment validation failed") from exc
    if design.status != "implementation_frozen":
        raise V024EnvironmentError(
            "V2.4 amendment has not entered immutable implementation_frozen status"
        )
    if _sha256_path(prereg_path) != _PREREG_BYTE_SHA256:
        raise V024EnvironmentError("V2.4 preregistration hash changed")
    if _sha256_path(lock_path) != _LOCK_BYTE_SHA256:
        raise V024EnvironmentError("V2.4 environment lock hash changed")
    if _sha256_path(implementation_audit_path) != _IMPLEMENTATION_AUDIT_BYTE_SHA256:
        raise V024EnvironmentError("V2.4 implementation audit hash changed")

    source_hashes = _source_hashes(root)
    _verify_paths_are_tracked(
        root,
        (
            amendment_path,
            prereg_path,
            freeze_record_path,
            implementation_audit_path,
            lock_path,
            *(root / relative for relative in source_hashes),
        ),
    )
    implementation_commit = _validate_freeze_record(
        freeze_record_path,
        amendment_byte_sha256=design.config_byte_sha256,
        source_hashes=source_hashes,
    )
    if implementation_commit == git_commit:
        raise V024EnvironmentError(
            "Implementation source commit must precede its attestation commit"
        )
    if _git(root, "rev-parse", "HEAD^") != implementation_commit:
        raise V024EnvironmentError(
            "Execution commit is not the direct V2.4 implementation attestation"
        )
    _git(root, "merge-base", "--is-ancestor", implementation_commit, git_commit)
    metadata_changes = tuple(
        line
        for line in _git(
            root,
            "diff",
            "--name-only",
            implementation_commit,
            git_commit,
        ).splitlines()
        if line
    )
    if (
        len(metadata_changes) != len(set(metadata_changes))
        or tuple(sorted(metadata_changes)) != _IMPLEMENTATION_METADATA_CHANGES
    ):
        raise V024EnvironmentError(
            "Post-implementation changes differ from the attestation allowlist"
        )

    import sys

    if sys.version_info[:3] != _EXPECTED_PYTHON:
        raise V024EnvironmentError("Formal Python runtime differs from CPython 3.12.13")
    if _runtime_hash_sentinel() != _EXPECTED_HASH_SENTINEL:
        raise V024EnvironmentError(
            "PYTHONHASHSEED was not active when this interpreter started"
        )
    observed_packages: dict[str, str] = {}
    for package, expected in _EXPECTED_PACKAGES.items():
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise V024EnvironmentError(
                f"Formal package is not installed: {package}"
            ) from exc
        if observed != expected:
            raise V024EnvironmentError(
                f"Formal package version changed: {package}={observed}, "
                f"expected={expected}"
            )
        observed_packages[package] = observed

    deterministic_environment = {
        name: os.environ.get(name, "") for name in _THREAD_ENVIRONMENT
    }
    deterministic_environment["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "")
    if (
        any(deterministic_environment[name] != "1" for name in _THREAD_ENVIRONMENT)
        or deterministic_environment["PYTHONHASHSEED"] != "0"
    ):
        raise V024EnvironmentError(
            "Formal deterministic thread/PYTHONHASHSEED environment is not locked"
        )
    active_threadpools = _active_threadpool_records()

    return FormalEnvironmentIdentity(
        protocol_id=V024_PROTOCOL_ID,
        git_commit=git_commit,
        git_dirty=False,
        config_byte_sha256=design.config_byte_sha256,
        config_canonical_sha256=V024_AMENDMENT_SEMANTIC_SHA256,
        preregistration_byte_sha256=_PREREG_BYTE_SHA256,
        environment_lock_byte_sha256=_LOCK_BYTE_SHA256,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        package_versions=MappingProxyType(observed_packages),
        deterministic_environment=MappingProxyType(deterministic_environment),
        source_byte_hashes=source_hashes,
        active_threadpools=active_threadpools,
    )


__all__ = [
    "FormalEnvironmentIdentity",
    "V024EnvironmentError",
    "verify_formal_environment",
]
