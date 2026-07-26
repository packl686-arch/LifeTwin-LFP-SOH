from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v016_environment as environment,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_AMENDMENT_SEMANTIC_SHA256,
    V021_PROTOCOL_ID,
)


def _fake_repository(tmp_path: Path, *, frozen: bool = True) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    copies = (
        environment._AMENDMENT_RELATIVE_PATH,
        environment._PREREG_RELATIVE_PATH,
        environment._LOCK_RELATIVE_PATH,
    )
    for relative in copies:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())

    amendment_path = repo / environment._AMENDMENT_RELATIVE_PATH
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["status"] = (
        "implementation_frozen" if frozen else "design_candidate_preimplementation"
    )
    amendment["fresh_generation"]["implementation_exists"] = frozen
    amendment_path.write_text(
        json.dumps(amendment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

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
    amendment_hash = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    record = {
        "schema_version": "1.0.0",
        "protocol_id": V021_PROTOCOL_ID,
        "design_freeze_commit": environment._DESIGN_FREEZE_COMMIT,
        "amendment_path": environment._AMENDMENT_RELATIVE_PATH.as_posix(),
        "amendment_byte_sha256": amendment_hash,
        "amendment_semantic_sha256": V021_AMENDMENT_SEMANTIC_SHA256,
        "preregistration_path": environment._PREREG_RELATIVE_PATH.as_posix(),
        "preregistration_byte_sha256": environment._PREREG_BYTE_SHA256,
        "environment_lock_path": environment._LOCK_RELATIVE_PATH.as_posix(),
        "environment_lock_byte_sha256": environment._LOCK_BYTE_SHA256,
        "review_status": "implementation_frozen",
        "implementation_source_commit": "a" * 40,
        "implementation_source_tree_sha256": (
            environment._implementation_source_tree_sha256(source_hashes)
        ),
        "implementation_source_byte_hashes": dict(source_hashes),
        "execution_metadata_paths": list(environment._IMPLEMENTATION_METADATA_CHANGES),
        "formal_v2_1_generation_executed_before_implementation_freeze": False,
        "v2_1_outcome_exposure_before_implementation_freeze": False,
    }
    record_path = repo / environment._FREEZE_RECORD_RELATIVE_PATH
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return repo


def _locked_environment(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _clean_git(repo: Path, *arguments: str) -> str:
    if arguments == ("rev-parse", "--show-toplevel"):
        return str(repo.resolve())
    if arguments == ("rev-parse", "HEAD"):
        return "b" * 40
    if arguments == ("rev-parse", "HEAD^"):
        return "a" * 40
    if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
        return ""
    if arguments[:3] == ("ls-files", "--error-unmatch", "--"):
        return "\n".join(arguments[3:])
    if arguments == ("merge-base", "--is-ancestor", "a" * 40, "b" * 40):
        return ""
    if arguments == ("diff", "--name-only", "a" * 40, "b" * 40):
        return "\n".join(environment._IMPLEMENTATION_METADATA_CHANGES)
    raise AssertionError(arguments)


def test_formal_environment_requires_frozen_status_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)
    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )

    observed = environment.verify_formal_environment(repo)

    assert observed.protocol_id == V021_PROTOCOL_ID
    assert observed.git_commit == "b" * 40
    assert observed.git_dirty is False
    assert observed.config_canonical_sha256 == V021_AMENDMENT_SEMANTIC_SHA256
    assert observed.environment_lock_byte_sha256 == environment._LOCK_BYTE_SHA256
    assert any("v015" in path for path in observed.source_byte_hashes)
    assert any("v016" in path for path in observed.source_byte_hashes)


def test_unfrozen_candidate_never_authorizes_formal_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path, frozen=False)
    _locked_environment(monkeypatch)
    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )
    with pytest.raises(environment.V021EnvironmentError, match="implementation_frozen"):
        environment.verify_formal_environment(repo)


def test_formal_environment_rejects_source_bytes_outside_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)
    (
        repo
        / "src"
        / "lifetwin"
        / "experiments"
        / "calendar_long_horizon_v016_fixture.py"
    ).write_text("VALUE = 17\n", encoding="ascii")
    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )
    with pytest.raises(environment.V021EnvironmentError, match="implementation"):
        environment.verify_formal_environment(repo)


def test_formal_environment_rejects_nonmetadata_attestation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)

    def changed_git(root: Path, *arguments: str) -> str:
        if arguments[:2] == ("diff", "--name-only"):
            return "src/lifetwin/experiments/calendar_long_horizon_v016_fixture.py"
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", changed_git)
    with pytest.raises(environment.V021EnvironmentError, match="allowlist"):
        environment.verify_formal_environment(repo)


def test_environment_lock_and_direct_attestation_child_are_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)
    (repo / environment._LOCK_RELATIVE_PATH).write_text(
        "changed\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )
    with pytest.raises(environment.V021EnvironmentError, match="lock hash"):
        environment.verify_formal_environment(repo)

    repo = _fake_repository(tmp_path / "second")

    def nonchild_git(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD^"):
            return "c" * 40
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", nonchild_git)
    with pytest.raises(
        environment.V021EnvironmentError,
        match="direct V2.1 implementation attestation",
    ):
        environment.verify_formal_environment(repo)


def test_loaded_native_threadpool_must_be_single_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        environment,
        "threadpool_info",
        lambda: [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "version": "fixture",
                "num_threads": 8,
                "filepath": "fixture.dll",
            }
        ],
    )
    with pytest.raises(environment.V021EnvironmentError, match="thread pool"):
        environment._active_threadpool_records()
