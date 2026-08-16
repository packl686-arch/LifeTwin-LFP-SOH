"""Fail-closed environment attesters for the fixed V2.10 formal profile."""

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
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from threadpoolctl import threadpool_info

from lifetwin.experiments.calendar_long_horizon_v025_protocol import (
    V030_AMENDMENT_BYTE_SHA256,
    V030_AMENDMENT_SEMANTIC_SHA256,
    V030_DESIGN_FREEZE_COMMIT,
    V030_PREREG_BYTE_SHA256,
    V030_PROTOCOL_ID,
    V030_REQUIREMENTS_BYTE_SHA256,
    load_v030_design,
)

if TYPE_CHECKING:
    from lifetwin.experiments.calendar_long_horizon_v019_contract import (
        V024ContractView,
    )


class V030EnvironmentError(RuntimeError):
    """Raised when a process is not the attested V2.10 build."""


@dataclass(frozen=True, slots=True)
class V030EnvironmentIdentity:
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


@dataclass(frozen=True, slots=True)
class V030PredictionEnvironmentIdentity:
    git_commit: str
    protocol_id: str
    config_byte_sha256: str


_AMENDMENT_PATH = Path(
    "configs/experiments/synthetic_long_horizon_identifiability_v2_10_amendment.json"
)
_PREREG_PATH = Path("reports/synthetic_long_horizon_identifiability_prereg_v2_10.md")
_LOCK_PATH = Path("requirements/v030-formal.txt")
_AUDIT_PATH = Path(
    "reports/synthetic_long_horizon_identifiability_implementation_audit_v2_10.md"
)
_FREEZE_PATH = Path(
    "reports/synthetic_long_horizon_identifiability_freeze_record_v2_10.json"
)
_METADATA_PATHS = tuple(sorted((_AUDIT_PATH.as_posix(), _FREEZE_PATH.as_posix())))
_EXPECTED_PYTHON = (3, 12, 13)
_EXPECTED_HASH_SENTINEL = -6098734223404096640
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
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_FULL_GIT_HASH = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise V030EnvironmentError("Git attestation failed") from exc
    return completed.stdout.strip()


def _strict_json(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise V030EnvironmentError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise V030EnvironmentError(f"Nonfinite JSON constant: {token}")

    try:
        payload = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V030EnvironmentError("Freeze record is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise V030EnvironmentError("Freeze record must be a JSON object")
    return payload


def _source_hashes(root: Path) -> Mapping[str, str]:
    experiment_root = root / "src" / "lifetwin" / "experiments"
    paths = {
        experiment_root / "calendar_long_horizon_synthetic.py",
        *experiment_root.glob("calendar_long_horizon_v015_*.py"),
        *experiment_root.glob("calendar_long_horizon_v017_*.py"),
        *experiment_root.glob("calendar_long_horizon_v018_*.py"),
        *experiment_root.glob("calendar_long_horizon_v019_*.py"),
        *experiment_root.glob("calendar_long_horizon_v020_*.py"),
        *experiment_root.glob("calendar_long_horizon_v021_*.py"),
        *experiment_root.glob("calendar_long_horizon_v022_*.py"),
        *experiment_root.glob("calendar_long_horizon_v023_*.py"),
        *experiment_root.glob("calendar_long_horizon_v024_*.py"),
        *experiment_root.glob("calendar_long_horizon_v025_*.py"),
        *(root / "scripts").glob("*v019*.py"),
        *(root / "scripts").glob("*v020*.py"),
        *(root / "scripts").glob("*v021*.py"),
        *(root / "scripts").glob("*v022*.py"),
        *(root / "scripts").glob("*v023*.py"),
        *(root / "scripts").glob("*v024*.py"),
        *(root / "scripts").glob("*v025*.py"),
        root / "tests/test_v016_training.py",
        root / "tests/test_v019_fit_commitment_atomicity.py",
        *(root / "tests").glob("test_v020*.py"),
        *(root / "tests").glob("test_v026*.py"),
        *(root / "tests").glob("test_v027*.py"),
        *(root / "tests").glob("test_v028*.py"),
        *(root / "tests").glob("test_v029*.py"),
        *(root / "tests").glob("test_v030*.py"),
        root / "tests/test_v019_pair_registry_row_contract.py",
    }
    ordered = tuple(sorted(paths))
    if not ordered or any(not path.is_file() for path in ordered):
        raise V030EnvironmentError("A bound implementation source is absent")
    return MappingProxyType(
        {path.relative_to(root).as_posix(): _sha256(path) for path in ordered}
    )


def source_tree_sha256(source_hashes: Mapping[str, str]) -> str:
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


def _require_tracked(root: Path, paths: tuple[Path, ...]) -> None:
    relative = tuple(path.relative_to(root).as_posix() for path in paths)
    observed = set(
        _git(root, "ls-files", "--error-unmatch", "--", *relative).splitlines()
    )
    if observed != set(relative):
        raise V030EnvironmentError("A frozen input is not git-tracked")


def _validate_freeze_record(
    path: Path,
    *,
    root: Path,
    source_hashes: Mapping[str, str],
) -> str:
    payload = _strict_json(path)
    expected = {
        "schema_version": "lifetwin_v030_freeze_record/1.0.0",
        "protocol_id": V030_PROTOCOL_ID,
        "review_status": "implementation_frozen",
        "design_freeze_commit": V030_DESIGN_FREEZE_COMMIT,
        "amendment_path": _AMENDMENT_PATH.as_posix(),
        "amendment_byte_sha256": V030_AMENDMENT_BYTE_SHA256,
        "amendment_semantic_sha256": V030_AMENDMENT_SEMANTIC_SHA256,
        "preregistration_path": _PREREG_PATH.as_posix(),
        "preregistration_byte_sha256": V030_PREREG_BYTE_SHA256,
        "environment_lock_path": _LOCK_PATH.as_posix(),
        "environment_lock_byte_sha256": V030_REQUIREMENTS_BYTE_SHA256,
        "implementation_audit_path": _AUDIT_PATH.as_posix(),
        "execution_metadata_paths": list(_METADATA_PATHS),
        "formal_v2_10_generation_executed_before_implementation_freeze": False,
        "v2_10_outcome_exposure_before_implementation_freeze": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise V030EnvironmentError("Freeze record identity changed")
    implementation_commit = payload.get("implementation_source_commit")
    recorded_hashes = payload.get("implementation_source_byte_hashes")
    audit_sha = payload.get("implementation_audit_byte_sha256")
    if (
        not isinstance(implementation_commit, str)
        or _FULL_GIT_HASH.fullmatch(implementation_commit) is None
        or not isinstance(recorded_hashes, dict)
        or recorded_hashes != dict(source_hashes)
        or payload.get("implementation_source_tree_sha256")
        != source_tree_sha256(source_hashes)
        or not isinstance(audit_sha, str)
        or _SHA256.fullmatch(audit_sha) is None
        or _sha256(root / _AUDIT_PATH) != audit_sha
    ):
        raise V030EnvironmentError("Freeze record source commitment changed")
    return implementation_commit


def _runtime() -> tuple[
    Mapping[str, str], Mapping[str, str], tuple[Mapping[str, object], ...]
]:
    if sys.version_info[:3] != _EXPECTED_PYTHON:
        raise V030EnvironmentError("Formal Python must be CPython 3.12.13")
    if hash("lifetwin-v030-formal") != _EXPECTED_HASH_SENTINEL:
        raise V030EnvironmentError("PYTHONHASHSEED was not active at process start")
    packages: dict[str, str] = {}
    for package, expected in _EXPECTED_PACKAGES.items():
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise V030EnvironmentError(f"Formal package is absent: {package}") from exc
        if observed != expected:
            raise V030EnvironmentError(f"Formal package changed: {package}")
        packages[package] = observed
    environment = {name: os.environ.get(name, "") for name in _THREAD_VARIABLES}
    environment["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "")
    if any(environment[name] != "1" for name in _THREAD_VARIABLES) or (
        environment["PYTHONHASHSEED"] != "0"
    ):
        raise V030EnvironmentError("Formal deterministic environment is not locked")
    pools: list[Mapping[str, object]] = []
    for pool in threadpool_info():
        if pool.get("num_threads") != 1:
            raise V030EnvironmentError("A loaded thread pool is not single-threaded")
        pools.append(MappingProxyType(dict(pool)))
    if not pools:
        raise V030EnvironmentError("No active numeric thread pool was attested")
    return (
        MappingProxyType(packages),
        MappingProxyType(environment),
        tuple(pools),
    )


def _attest(
    root: Path,
) -> tuple[
    str,
    Mapping[str, str],
    tuple[
        Mapping[str, str],
        Mapping[str, str],
        tuple[Mapping[str, object], ...],
    ],
]:
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise V030EnvironmentError("Repository root is not the git top level")
    head = _git(root, "rev-parse", "HEAD")
    if _FULL_GIT_HASH.fullmatch(head) is None:
        raise V030EnvironmentError("Git HEAD is not a full object ID")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise V030EnvironmentError("Formal implementation worktree is dirty")
    design = load_v030_design(root / _AMENDMENT_PATH)
    if (
        design.config_byte_sha256 != V030_AMENDMENT_BYTE_SHA256
        or design.config_semantic_sha256 != V030_AMENDMENT_SEMANTIC_SHA256
    ):
        raise V030EnvironmentError("V2.10 amendment identity changed")
    paths = tuple(
        root / path for path in (_PREREG_PATH, _LOCK_PATH, _AUDIT_PATH, _FREEZE_PATH)
    )
    if any(not path.is_file() for path in paths):
        raise V030EnvironmentError("A required freeze input is absent")
    if _sha256(root / _PREREG_PATH) != V030_PREREG_BYTE_SHA256:
        raise V030EnvironmentError("V2.10 preregistration changed")
    if _sha256(root / _LOCK_PATH) != V030_REQUIREMENTS_BYTE_SHA256:
        raise V030EnvironmentError("V2.10 environment lock changed")
    source_hashes = _source_hashes(root)
    _require_tracked(
        root,
        (root / _AMENDMENT_PATH, *paths, *(root / name for name in source_hashes)),
    )
    implementation_commit = _validate_freeze_record(
        root / _FREEZE_PATH,
        root=root,
        source_hashes=source_hashes,
    )
    if _git(root, "rev-parse", "HEAD^") != implementation_commit:
        raise V030EnvironmentError(
            "Freeze commit is not the direct child of implementation"
        )
    changes = tuple(
        sorted(
            line
            for line in _git(
                root, "diff", "--name-only", implementation_commit, head
            ).splitlines()
            if line
        )
    )
    if changes != _METADATA_PATHS:
        raise V030EnvironmentError("Freeze commit changed files outside the allowlist")
    runtime = _runtime()
    return head, source_hashes, runtime


def verify_formal_environment_v030(
    repo_root: str | Path,
    contract_view: V024ContractView,
) -> V030EnvironmentIdentity:
    """Attest V2.10 and bind the authenticated contract carried by the caller."""

    root = Path(repo_root).resolve()
    if (
        contract_view.protocol.protocol_id != V030_PROTOCOL_ID
        or contract_view.artifacts.config_byte_sha256 != V030_AMENDMENT_BYTE_SHA256
        or contract_view.config_canonical_sha256 != V030_AMENDMENT_SEMANTIC_SHA256
        or contract_view.design_status != "implementation_frozen"
    ):
        raise V030EnvironmentError("Formal contract differs from V2.10")
    head, source_hashes, runtime = _attest(root)
    packages, environment, pools = runtime
    return V030EnvironmentIdentity(
        protocol_id=V030_PROTOCOL_ID,
        git_commit=head,
        git_dirty=False,
        config_byte_sha256=V030_AMENDMENT_BYTE_SHA256,
        config_canonical_sha256=V030_AMENDMENT_SEMANTIC_SHA256,
        preregistration_byte_sha256=V030_PREREG_BYTE_SHA256,
        environment_lock_byte_sha256=V030_REQUIREMENTS_BYTE_SHA256,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        package_versions=packages,
        deterministic_environment=environment,
        source_byte_hashes=source_hashes,
        active_threadpools=pools,
    )


def verify_prediction_environment_v030(
    repo_root: str | Path,
) -> V030PredictionEnvironmentIdentity:
    """Attest the same frozen identity without importing a truth capability."""

    head, _, _ = _attest(Path(repo_root).resolve())
    return V030PredictionEnvironmentIdentity(
        git_commit=head,
        protocol_id=V030_PROTOCOL_ID,
        config_byte_sha256=V030_AMENDMENT_BYTE_SHA256,
    )


__all__ = [
    "V030EnvironmentError",
    "V030EnvironmentIdentity",
    "V030PredictionEnvironmentIdentity",
    "source_tree_sha256",
    "verify_formal_environment_v030",
    "verify_prediction_environment_v030",
]
