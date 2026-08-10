"""Standalone formal-environment attester for the V2.3 prediction capsule.

This module deliberately does not import the V2.3 protocol, contract, training,
generation, or truth-capable modules.  It validates the immutable amendment and
implementation attestation directly from repository bytes before a prediction
process is allowed to open a label-free bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Mapping

from threadpoolctl import threadpool_info


class V023PredictionEnvironmentError(RuntimeError):
    """Raised when the prediction process is not the attested V2.3 build."""


V023_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2_3"
V023_AMENDMENT_SEMANTIC_SHA256 = (
    "1eb3778d0825e0a06e7c1b63221ba6aa914f9df21b4ef2646156035b4a001bec"
)
V023_AMENDMENT_BYTE_SHA256 = (
    "4aa048e77172c3a2c1069e97f3c1d33dd2a5952b7aee6c5f623a2259b847085f"
)
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
_LOCK_RELATIVE_PATH = Path("requirements") / "v023-formal.txt"
_LOCK_BYTE_SHA256 = "a146229d8bfd75ade103e25bf5e5776c4c951bdd351feaa37dbf6a6b15e4843c"
_AMENDMENT_RELATIVE_PATH = (
    Path("configs")
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_3_amendment.json"
)
_PREREG_RELATIVE_PATH = (
    Path("reports") / "synthetic_long_horizon_identifiability_prereg_v2_3.md"
)
_FREEZE_RECORD_RELATIVE_PATH = (
    Path("reports") / "synthetic_long_horizon_identifiability_freeze_record_v2_3.json"
)
_IMPLEMENTATION_AUDIT_RELATIVE_PATH = (
    Path("reports")
    / "synthetic_long_horizon_identifiability_implementation_audit_v2_3.md"
)
_IMPLEMENTATION_METADATA_CHANGES = tuple(
    sorted(
        {
            _FREEZE_RECORD_RELATIVE_PATH.as_posix(),
        }
    )
)
_PREREG_BYTE_SHA256 = "e301eca9836e4ad9ae816a7dda192350fc3e7aee211c08b007e381def561c970"
_IMPLEMENTATION_AUDIT_BYTE_SHA256 = (
    "48ad060026bf3f2a2165a8e8f1724d2ce65f7ee512e431c132012e9e4d4b5064"
)
_DESIGN_FREEZE_COMMIT = "6e605e79a72c76500a109d3f44b460e394859d82"
_EXPECTED_HASH_SENTINEL = -7323235757074059332
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_FULL_GIT_HASH = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_AMENDMENT_TOP_LEVEL_KEYS = frozenset(
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
        "numeric_output_contract",
        "lifecycle_order",
        "terminal_rules",
        "pre_result_test_gate",
        "freeze_requirements",
        "claim_boundary",
    }
)


@dataclass(frozen=True, slots=True)
class PredictionEnvironmentIdentity:
    """Minimal identity needed to bind a label-free prediction bundle."""

    git_commit: str
    config_byte_sha256: str


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
        raise V023PredictionEnvironmentError(
            f"Cannot verify git {' '.join(arguments)}"
        ) from exc
    return result.stdout.strip()


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V023PredictionEnvironmentError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise V023PredictionEnvironmentError(f"Nonfinite JSON constant: {token}")


def _strict_json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V023PredictionEnvironmentError(f"{context} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise V023PredictionEnvironmentError(f"{context} must be a JSON object")
    return decoded


def _json_object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise V023PredictionEnvironmentError(f"{context} must be a JSON object")
    return value


def _validate_frozen_amendment(path: Path) -> str:
    payload = _strict_json_object(path, context="V2.3 amendment")
    if set(payload) != _AMENDMENT_TOP_LEVEL_KEYS:
        raise V023PredictionEnvironmentError("V2.3 amendment keys changed")
    if (
        payload.get("schema_version") != "lifetwin_synthetic_long_horizon_v2_3/1.0.0"
        or payload.get("protocol_id") != V023_PROTOCOL_ID
    ):
        raise V023PredictionEnvironmentError("V2.3 amendment identity changed")
    if payload.get("status") != "implementation_frozen":
        raise V023PredictionEnvironmentError(
            "V2.3 amendment has not entered immutable implementation_frozen status"
        )
    generation = _json_object(
        payload.get("fresh_generation"),
        context="V2.3 fresh_generation",
    )
    if (
        generation.get("generation_has_started") is not False
        or generation.get("seed_consumed") is not False
        or generation.get("sealed_truth_created_or_opened") is not False
    ):
        raise V023PredictionEnvironmentError(
            "V2.3 frozen amendment unexpectedly records prior generation"
        )
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != V023_AMENDMENT_SEMANTIC_SHA256:
        raise V023PredictionEnvironmentError(
            "V2.3 amendment semantic content changed"
        )
    byte_sha256 = _sha256_path(path)
    if byte_sha256 != V023_AMENDMENT_BYTE_SHA256:
        raise V023PredictionEnvironmentError("V2.3 amendment bytes changed")
    return byte_sha256


def _source_hashes(repo_root: Path) -> Mapping[str, str]:
    experiment_root = repo_root / "src" / "lifetwin" / "experiments"
    inherited = sorted(experiment_root.glob("calendar_long_horizon_v015_*.py"))
    amended = sorted(experiment_root.glob("calendar_long_horizon_v018_*.py"))
    shared = experiment_root / "calendar_long_horizon_synthetic.py"
    scripts = sorted((repo_root / "scripts").glob("*v018*.py"))
    if not inherited or not amended or not shared.is_file():
        raise V023PredictionEnvironmentError(
            "The complete V2.3 and inherited V2 implementation source set is absent"
        )
    paths = tuple(sorted({*inherited, *amended, shared, *scripts}))
    if any(not path.is_file() for path in paths):
        raise V023PredictionEnvironmentError(
            "A V2.3 implementation source is not a file"
        )
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


def _validate_freeze_record(
    path: Path,
    *,
    amendment_byte_sha256: str,
    source_hashes: Mapping[str, str],
) -> str:
    payload = _strict_json_object(path, context="V2.3 freeze record")
    expected = {
        "schema_version": "1.0.0",
        "protocol_id": V023_PROTOCOL_ID,
        "design_freeze_commit": _DESIGN_FREEZE_COMMIT,
        "amendment_path": _AMENDMENT_RELATIVE_PATH.as_posix(),
        "amendment_byte_sha256": amendment_byte_sha256,
        "amendment_semantic_sha256": V023_AMENDMENT_SEMANTIC_SHA256,
        "preregistration_path": _PREREG_RELATIVE_PATH.as_posix(),
        "preregistration_byte_sha256": _PREREG_BYTE_SHA256,
        "environment_lock_path": _LOCK_RELATIVE_PATH.as_posix(),
        "environment_lock_byte_sha256": _LOCK_BYTE_SHA256,
        "implementation_audit_path": _IMPLEMENTATION_AUDIT_RELATIVE_PATH.as_posix(),
        "implementation_audit_byte_sha256": _IMPLEMENTATION_AUDIT_BYTE_SHA256,
        "formal_v2_3_generation_executed_before_implementation_freeze": False,
        "v2_3_outcome_exposure_before_implementation_freeze": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise V023PredictionEnvironmentError(
            "V2.3 freeze record does not bind the frozen protocol and environment"
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
        raise V023PredictionEnvironmentError(
            "V2.3 freeze record does not bind the frozen implementation"
        )
    return implementation_commit


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
        raise V023PredictionEnvironmentError(
            "A formal freeze, lock, or implementation source is not git-tracked"
        )


def _runtime_hash_sentinel() -> int:
    return hash("lifetwin-v023-formal")


def _active_threadpool_records() -> tuple[Mapping[str, object], ...]:
    records: list[dict[str, object]] = []
    for pool in threadpool_info():
        threads = pool.get("num_threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads != 1:
            raise V023PredictionEnvironmentError(
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
        raise V023PredictionEnvironmentError(
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


def _verify_runtime() -> None:
    if sys.version_info[:3] != _EXPECTED_PYTHON:
        raise V023PredictionEnvironmentError(
            "Formal Python runtime differs from CPython 3.12.13"
        )
    if _runtime_hash_sentinel() != _EXPECTED_HASH_SENTINEL:
        raise V023PredictionEnvironmentError(
            "PYTHONHASHSEED was not active when this interpreter started"
        )
    for package, expected in _EXPECTED_PACKAGES.items():
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise V023PredictionEnvironmentError(
                f"Formal package is not installed: {package}"
            ) from exc
        if observed != expected:
            raise V023PredictionEnvironmentError(
                f"Formal package version changed: {package}={observed}, "
                f"expected={expected}"
            )
    deterministic_environment = {
        name: os.environ.get(name, "") for name in _THREAD_ENVIRONMENT
    }
    deterministic_environment["PYTHONHASHSEED"] = os.environ.get(
        "PYTHONHASHSEED",
        "",
    )
    if (
        any(deterministic_environment[name] != "1" for name in _THREAD_ENVIRONMENT)
        or deterministic_environment["PYTHONHASHSEED"] != "0"
    ):
        raise V023PredictionEnvironmentError(
            "Formal deterministic thread/PYTHONHASHSEED environment is not locked"
        )
    _active_threadpool_records()


def verify_prediction_environment(
    repo_root: str | Path,
) -> PredictionEnvironmentIdentity:
    """Attest the frozen direct-child build without importing unsafe modules."""

    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise V023PredictionEnvironmentError(
            "Formal repository root has no .git directory"
        )
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise V023PredictionEnvironmentError(
            "Formal repository root is not the git top level"
        )
    git_commit = _git(root, "rev-parse", "HEAD")
    if _FULL_GIT_HASH.fullmatch(git_commit) is None:
        raise V023PredictionEnvironmentError("Git did not return a full commit hash")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise V023PredictionEnvironmentError("Formal implementation worktree is dirty")

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
            raise V023PredictionEnvironmentError(
                f"Required freeze file is absent: {path}"
            )
    amendment_byte_sha256 = _validate_frozen_amendment(amendment_path)
    if _sha256_path(prereg_path) != _PREREG_BYTE_SHA256:
        raise V023PredictionEnvironmentError("V2.3 preregistration hash changed")
    if _sha256_path(lock_path) != _LOCK_BYTE_SHA256:
        raise V023PredictionEnvironmentError("V2.3 environment lock hash changed")
    if _sha256_path(implementation_audit_path) != _IMPLEMENTATION_AUDIT_BYTE_SHA256:
        raise V023PredictionEnvironmentError("V2.3 implementation audit hash changed")

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
        amendment_byte_sha256=amendment_byte_sha256,
        source_hashes=source_hashes,
    )
    if implementation_commit == git_commit:
        raise V023PredictionEnvironmentError(
            "Implementation source commit must precede its attestation commit"
        )
    if _git(root, "rev-parse", "HEAD^") != implementation_commit:
        raise V023PredictionEnvironmentError(
            "Execution commit is not the direct V2.3 implementation attestation"
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
        raise V023PredictionEnvironmentError(
            "Post-implementation changes differ from the attestation allowlist"
        )

    _verify_runtime()
    return PredictionEnvironmentIdentity(
        git_commit=git_commit,
        config_byte_sha256=amendment_byte_sha256,
    )


__all__ = [
    "PredictionEnvironmentIdentity",
    "V023PredictionEnvironmentError",
    "verify_prediction_environment",
]
