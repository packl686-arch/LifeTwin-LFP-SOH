"""Fail-closed implementation and environment guard for the V0.15 attempt."""

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

from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256,
    FROZEN_CONFIG_CANONICAL_SHA256,
    load_frozen_protocol_config,
)


class V015EnvironmentError(RuntimeError):
    """Raised when the formal process is not the frozen implementation."""


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
_LOCK_RELATIVE_PATH = Path("requirements") / "v015-formal.txt"
_LOCK_BYTE_SHA256 = "f9546f70f4e9a5134993ead7016e69f6ecea19c295bd20680d42a330be58f5f1"
_CONFIG_RELATIVE_PATH = (
    Path("configs") / "experiments" / "synthetic_long_horizon_identifiability_v2.json"
)
_PREREG_RELATIVE_PATH = (
    Path("reports") / "synthetic_long_horizon_identifiability_prereg_v2.md"
)
_FREEZE_RECORD_RELATIVE_PATH = (
    Path("reports") / "synthetic_long_horizon_identifiability_freeze_record_v2.json"
)
_IMPLEMENTATION_AUDIT_RELATIVE_PATH = (
    Path("reports")
    / "synthetic_long_horizon_identifiability_implementation_audit_v2.md"
)
_IMPLEMENTATION_METADATA_CHANGES = tuple(
    sorted(
        {
            _FREEZE_RECORD_RELATIVE_PATH.as_posix(),
            _IMPLEMENTATION_AUDIT_RELATIVE_PATH.as_posix(),
            "release_manifest.json",
        }
    )
)
_PREREG_BYTE_SHA256 = "c1dee9f9b4ef134b1a52e9a51300c591e790c10a0e97b3fe6c15eb441b2c09f0"
_PROTOCOL_FREEZE_COMMIT = "b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91"
_EXPECTED_HASH_SENTINEL = -3192278995227909433
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_FULL_GIT_HASH = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


@dataclass(frozen=True)
class FormalEnvironmentIdentity:
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
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_byte_sha256": self.config_byte_sha256,
            "config_canonical_sha256": self.config_canonical_sha256,
            "preregistration_byte_sha256": self.preregistration_byte_sha256,
            "environment_lock_byte_sha256": (self.environment_lock_byte_sha256),
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
        raise V015EnvironmentError(f"Cannot verify git {' '.join(arguments)}") from exc
    return result.stdout.strip()


def _source_hashes(repo_root: Path) -> Mapping[str, str]:
    experiment_root = repo_root / "src" / "lifetwin" / "experiments"
    paths = sorted(experiment_root.glob("calendar_long_horizon_v015_*.py"))
    paths.append(experiment_root / "calendar_long_horizon_synthetic.py")
    paths.extend(sorted((repo_root / "scripts").glob("*v015*.py")))
    if any(not path.is_file() for path in paths):
        raise V015EnvironmentError("No V0.15 implementation sources were found")
    records = {
        path.relative_to(repo_root).as_posix(): _sha256_path(path)
        for path in sorted(set(paths))
    }
    return MappingProxyType(records)


def _implementation_source_tree_sha256(
    source_hashes: Mapping[str, str],
) -> str:
    raw = (
        json.dumps(
            dict(source_hashes),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_freeze_record(
    path: Path,
    *,
    source_hashes: Mapping[str, str],
) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V015EnvironmentError("V0.15 freeze record is invalid JSON") from exc
    expected = {
        "schema_version": "1.0.0",
        "protocol_id": "synthetic_long_horizon_identifiability_v2",
        "freeze_commit": _PROTOCOL_FREEZE_COMMIT,
        "config_path": _CONFIG_RELATIVE_PATH.as_posix(),
        "config_byte_sha256": FROZEN_CONFIG_BYTE_SHA256,
        "config_canonical_sha256": FROZEN_CONFIG_CANONICAL_SHA256,
        "preregistration_path": _PREREG_RELATIVE_PATH.as_posix(),
        "preregistration_byte_sha256": _PREREG_BYTE_SHA256,
        "v2_outcome_exposure_before_freeze": False,
        "formal_v2_generation_executed_before_freeze": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise V015EnvironmentError(
            "V0.15 freeze record does not bind the frozen protocol"
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
        or payload.get("formal_v2_generation_executed_before_implementation_freeze")
        is not False
    ):
        raise V015EnvironmentError(
            "V0.15 freeze record does not bind the frozen implementation"
        )
    return implementation_commit


def _runtime_hash_sentinel() -> int:
    return hash("lifetwin-v015-formal")


def _active_threadpool_records() -> tuple[Mapping[str, object], ...]:
    records: list[dict[str, object]] = []
    for pool in threadpool_info():
        threads = pool.get("num_threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads != 1:
            raise V015EnvironmentError(
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
        raise V015EnvironmentError(
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
        raise V015EnvironmentError(
            "A formal freeze or implementation source is not git-tracked"
        )


def verify_formal_environment(
    repo_root: str | Path,
) -> FormalEnvironmentIdentity:
    """Verify clean committed code, immutable protocol bytes, and exact runtime."""

    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise V015EnvironmentError("Formal repository root has no .git directory")
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise V015EnvironmentError("Formal repository root is not the git top level")
    git_commit = _git(root, "rev-parse", "HEAD")
    if _FULL_GIT_HASH.fullmatch(git_commit) is None:
        raise V015EnvironmentError("Git did not return a full commit hash")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise V015EnvironmentError("Formal implementation worktree is dirty")

    config_path = root / _CONFIG_RELATIVE_PATH
    prereg_path = root / _PREREG_RELATIVE_PATH
    freeze_record_path = root / _FREEZE_RECORD_RELATIVE_PATH
    lock_path = root / _LOCK_RELATIVE_PATH
    for path in (config_path, prereg_path, freeze_record_path, lock_path):
        if not path.is_file():
            raise V015EnvironmentError(f"Required freeze file is absent: {path}")
    source_hashes = _source_hashes(root)
    _verify_paths_are_tracked(
        root,
        (
            config_path,
            prereg_path,
            freeze_record_path,
            lock_path,
            *(root / relative for relative in source_hashes),
        ),
    )
    if _sha256_path(config_path) != FROZEN_CONFIG_BYTE_SHA256:
        raise V015EnvironmentError("Frozen V2 config byte hash changed")
    try:
        protocol = load_frozen_protocol_config(config_path)
    except (OSError, ValueError) as exc:
        raise V015EnvironmentError(
            "Frozen V2 config canonical validation failed"
        ) from exc
    if protocol.config_sha256 != FROZEN_CONFIG_CANONICAL_SHA256:
        raise V015EnvironmentError("Frozen V2 config canonical hash changed")
    if _sha256_path(prereg_path) != _PREREG_BYTE_SHA256:
        raise V015EnvironmentError("Frozen V2 preregistration hash changed")
    if _sha256_path(lock_path) != _LOCK_BYTE_SHA256:
        raise V015EnvironmentError("V0.15 environment lock hash changed")
    implementation_commit = _validate_freeze_record(
        freeze_record_path,
        source_hashes=source_hashes,
    )
    if implementation_commit == git_commit:
        raise V015EnvironmentError(
            "Implementation source commit must precede its attestation commit"
        )
    if _git(root, "rev-parse", "HEAD^") != implementation_commit:
        raise V015EnvironmentError(
            "Execution commit is not the direct implementation attestation"
        )
    _git(
        root,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        git_commit,
    )
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
        raise V015EnvironmentError(
            "Post-implementation commit changes differ from the attestation allowlist"
        )

    import sys

    if sys.version_info[:3] != _EXPECTED_PYTHON:
        raise V015EnvironmentError("Formal Python runtime differs from CPython 3.12.13")
    if _runtime_hash_sentinel() != _EXPECTED_HASH_SENTINEL:
        raise V015EnvironmentError(
            "PYTHONHASHSEED was not active when this interpreter started"
        )
    observed_packages: dict[str, str] = {}
    for package, expected in _EXPECTED_PACKAGES.items():
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise V015EnvironmentError(
                f"Formal package is not installed: {package}"
            ) from exc
        if observed != expected:
            raise V015EnvironmentError(
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
        raise V015EnvironmentError(
            "Formal deterministic thread/PYTHONHASHSEED environment is not locked"
        )
    active_threadpools = _active_threadpool_records()

    return FormalEnvironmentIdentity(
        git_commit=git_commit,
        git_dirty=False,
        config_byte_sha256=FROZEN_CONFIG_BYTE_SHA256,
        config_canonical_sha256=FROZEN_CONFIG_CANONICAL_SHA256,
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
    "V015EnvironmentError",
    "verify_formal_environment",
]
