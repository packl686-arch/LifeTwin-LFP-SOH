from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v016_prediction_environment as environment,
)


_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_IMPLEMENTATION_COMMIT = "a" * 40
_ATTESTATION_COMMIT = "b" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _frozen_amendment() -> dict[str, object]:
    path = _SOURCE_ROOT / environment._AMENDMENT_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "implementation_frozen"
    generation = payload["fresh_generation"]
    assert isinstance(generation, dict)
    generation["implementation_exists"] = True
    return payload


def _fake_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    for relative in (
        environment._PREREG_RELATIVE_PATH,
        environment._LOCK_RELATIVE_PATH,
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((_SOURCE_ROOT / relative).read_bytes())

    amendment_path = repo / environment._AMENDMENT_RELATIVE_PATH
    _write_json(amendment_path, _frozen_amendment())

    experiment_root = repo / "src" / "lifetwin" / "experiments"
    experiment_root.mkdir(parents=True)
    (experiment_root / "calendar_long_horizon_v015_fixture.py").write_text(
        "VALUE = 15\n",
        encoding="ascii",
    )
    (experiment_root / "calendar_long_horizon_v016_fixture.py").write_text(
        "VALUE = 16\n",
        encoding="ascii",
    )
    (experiment_root / "calendar_long_horizon_synthetic.py").write_text(
        "VALUE = 1\n",
        encoding="ascii",
    )

    source_hashes = environment._source_hashes(repo)
    freeze_record = {
        "schema_version": "1.0.0",
        "protocol_id": environment.V021_PROTOCOL_ID,
        "design_freeze_commit": environment._DESIGN_FREEZE_COMMIT,
        "amendment_path": environment._AMENDMENT_RELATIVE_PATH.as_posix(),
        "amendment_byte_sha256": environment._sha256_path(amendment_path),
        "amendment_semantic_sha256": (environment.V021_AMENDMENT_SEMANTIC_SHA256),
        "preregistration_path": environment._PREREG_RELATIVE_PATH.as_posix(),
        "preregistration_byte_sha256": environment._PREREG_BYTE_SHA256,
        "environment_lock_path": environment._LOCK_RELATIVE_PATH.as_posix(),
        "environment_lock_byte_sha256": environment._LOCK_BYTE_SHA256,
        "review_status": "implementation_frozen",
        "implementation_source_commit": _IMPLEMENTATION_COMMIT,
        "implementation_source_tree_sha256": (
            environment._implementation_source_tree_sha256(source_hashes)
        ),
        "implementation_source_byte_hashes": dict(source_hashes),
        "execution_metadata_paths": list(environment._IMPLEMENTATION_METADATA_CHANGES),
        "formal_v2_1_generation_executed_before_implementation_freeze": False,
        "v2_1_outcome_exposure_before_implementation_freeze": False,
    }
    _write_json(repo / environment._FREEZE_RECORD_RELATIVE_PATH, freeze_record)
    return repo


def _clean_git(repo: Path, *arguments: str) -> str:
    if arguments == ("rev-parse", "--show-toplevel"):
        return str(repo.resolve())
    if arguments == ("rev-parse", "HEAD"):
        return _ATTESTATION_COMMIT
    if arguments == ("rev-parse", "HEAD^"):
        return _IMPLEMENTATION_COMMIT
    if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
        return ""
    if arguments[:3] == ("ls-files", "--error-unmatch", "--"):
        return "\n".join(arguments[3:])
    if arguments == (
        "merge-base",
        "--is-ancestor",
        _IMPLEMENTATION_COMMIT,
        _ATTESTATION_COMMIT,
    ):
        return ""
    if arguments == (
        "diff",
        "--name-only",
        _IMPLEMENTATION_COMMIT,
        _ATTESTATION_COMMIT,
    ):
        return "\n".join(environment._IMPLEMENTATION_METADATA_CHANGES)
    raise AssertionError(arguments)


def _lock_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in environment._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(
        environment,
        "_runtime_hash_sentinel",
        lambda: environment._EXPECTED_HASH_SENTINEL,
    )
    monkeypatch.setattr(
        environment,
        "_active_threadpool_records",
        lambda: ({"internal_api": "fixture", "num_threads": 1},),
    )
    monkeypatch.setattr(
        environment.metadata,
        "version",
        lambda package: environment._EXPECTED_PACKAGES[package],
    )
    monkeypatch.setattr(
        environment.sys,
        "version_info",
        (*environment._EXPECTED_PYTHON, "final", 0),
    )


def test_frozen_amendment_allows_only_the_paired_status_switch(
    tmp_path: Path,
) -> None:
    payload = _frozen_amendment()
    path = tmp_path / "amendment.json"
    _write_json(path, payload)

    assert (
        environment._amendment_semantic_sha256(payload)
        == environment.V021_AMENDMENT_SEMANTIC_SHA256
    )
    assert environment._validate_frozen_amendment(path) == (
        environment._sha256_path(path)
    )

    payload["status"] = "design_candidate_preimplementation"
    generation = payload["fresh_generation"]
    assert isinstance(generation, dict)
    generation["implementation_exists"] = False
    assert (
        environment._amendment_semantic_sha256(payload)
        == environment.V021_AMENDMENT_SEMANTIC_SHA256
    )
    _write_json(path, payload)
    with pytest.raises(
        environment.V021PredictionEnvironmentError,
        match="implementation_frozen",
    ):
        environment._validate_frozen_amendment(path)

    payload["status"] = "implementation_frozen"
    _write_json(path, payload)
    with pytest.raises(
        environment.V021PredictionEnvironmentError,
        match="implementation exists",
    ):
        environment._validate_frozen_amendment(path)


def test_frozen_amendment_rejects_semantic_drift_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    payload = _frozen_amendment()
    payload["title"] = f"{payload['title']} changed"
    path = tmp_path / "amendment.json"
    _write_json(path, payload)
    with pytest.raises(
        environment.V021PredictionEnvironmentError,
        match="semantic content changed",
    ):
        environment._validate_frozen_amendment(path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key": 1, "key": 2}\n', encoding="ascii")
    with pytest.raises(
        environment.V021PredictionEnvironmentError,
        match="Duplicate JSON key",
    ):
        environment._strict_json_object(duplicate, context="fixture")


def test_attester_import_has_no_experimental_capability_dependencies() -> None:
    source_root = _SOURCE_ROOT / "src"
    module_name = (
        "lifetwin.experiments.calendar_long_horizon_v016_prediction_environment"
    )
    prefix = "lifetwin.experiments.calendar_long_horizon_"
    probe = (
        "import json,sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        f"import {module_name};"
        "print(json.dumps(sorted(name for name in sys.modules "
        f"if name.startswith({prefix!r}))))"
    )
    completed = subprocess.run(
        (sys.executable, "-c", probe),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        cwd=_SOURCE_ROOT,
    )
    assert json.loads(completed.stdout) == [module_name]


def test_prediction_environment_accepts_only_direct_attestation_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)
    _lock_runtime(monkeypatch)
    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )

    observed = environment.verify_prediction_environment(repo)

    assert observed == environment.PredictionEnvironmentIdentity(
        git_commit=_ATTESTATION_COMMIT,
        config_byte_sha256=environment._sha256_path(
            repo / environment._AMENDMENT_RELATIVE_PATH
        ),
    )
    assert tuple(environment._IMPLEMENTATION_METADATA_CHANGES) == (
        "release_manifest.json",
        "reports/synthetic_long_horizon_identifiability_freeze_record_v2_1.json",
        "reports/synthetic_long_horizon_identifiability_implementation_audit_v2_1.md",
    )


def test_prediction_environment_rejects_nonchild_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)

    def nonchild_git(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD^"):
            return "c" * 40
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", nonchild_git)
    with pytest.raises(
        environment.V021PredictionEnvironmentError,
        match="direct V2.1 implementation attestation",
    ):
        environment.verify_prediction_environment(repo)


def test_prediction_environment_rejects_metadata_allowlist_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)

    def changed_git(root: Path, *arguments: str) -> str:
        if arguments[:2] == ("diff", "--name-only"):
            return (
                "release_manifest.json\n"
                "src/lifetwin/experiments/calendar_long_horizon_v016_fit.py"
            )
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", changed_git)
    with pytest.raises(
        environment.V021PredictionEnvironmentError,
        match="attestation allowlist",
    ):
        environment.verify_prediction_environment(repo)


def test_prediction_environment_rejects_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)

    def dirty_git(root: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M src/lifetwin/experiments/unsafe_change.py"
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", dirty_git)
    with pytest.raises(
        environment.V021PredictionEnvironmentError,
        match="worktree is dirty",
    ):
        environment.verify_prediction_environment(repo)
