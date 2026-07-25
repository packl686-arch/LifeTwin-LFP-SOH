from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v015_environment as environment,
)


def _fake_repository(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    required = (
        Path("configs/experiments/synthetic_long_horizon_identifiability_v2.json"),
        Path("reports/synthetic_long_horizon_identifiability_prereg_v2.md"),
        Path("reports/synthetic_long_horizon_identifiability_freeze_record_v2.json"),
        Path("requirements/v015-formal.txt"),
    )
    for relative in required:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())
    implementation = (
        repo
        / "src"
        / "lifetwin"
        / "experiments"
        / "calendar_long_horizon_v015_fixture.py"
    )
    implementation.parent.mkdir(parents=True)
    implementation.write_text("VALUE = 1\n", encoding="ascii")
    shared = (
        repo / "src" / "lifetwin" / "experiments" / "calendar_long_horizon_synthetic.py"
    )
    shared.write_bytes(
        (
            source_root
            / "src"
            / "lifetwin"
            / "experiments"
            / "calendar_long_horizon_synthetic.py"
        ).read_bytes()
    )
    source_hashes = environment._source_hashes(repo)
    record_path = (
        repo
        / "reports"
        / "synthetic_long_horizon_identifiability_freeze_record_v2.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.update(
        {
            "review_status": "implementation_frozen",
            "implementation_source_commit": "a" * 40,
            "implementation_source_tree_sha256": (
                environment._implementation_source_tree_sha256(source_hashes)
            ),
            "implementation_source_byte_hashes": dict(source_hashes),
            "execution_metadata_paths": list(
                environment._IMPLEMENTATION_METADATA_CHANGES
            ),
            "formal_v2_generation_executed_before_implementation_freeze": (False),
        }
    )
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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


def _clean_git(repo: Path, *arguments: str) -> str:
    if arguments == ("rev-parse", "--show-toplevel"):
        return str(repo.resolve())
    if arguments == ("rev-parse", "HEAD"):
        return "b" * 40
    if arguments == ("rev-parse", "HEAD^"):
        return "a" * 40
    if arguments == (
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        return ""
    if arguments[:3] == ("ls-files", "--error-unmatch", "--"):
        return "\n".join(arguments[3:])
    if arguments == (
        "merge-base",
        "--is-ancestor",
        "a" * 40,
        "b" * 40,
    ):
        return ""
    if arguments == (
        "diff",
        "--name-only",
        "a" * 40,
        "b" * 40,
    ):
        return "\n".join(environment._IMPLEMENTATION_METADATA_CHANGES)
    raise AssertionError(arguments)


def test_formal_environment_records_clean_committed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)

    def fake_git(root: Path, *arguments: str) -> str:
        assert root == repo.resolve()
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", fake_git)
    observed = environment.verify_formal_environment(repo)

    assert observed.git_commit == "b" * 40
    assert observed.git_dirty is False
    assert observed.environment_lock_byte_sha256 == (environment._LOCK_BYTE_SHA256)
    assert tuple(observed.source_byte_hashes) == (
        "src/lifetwin/experiments/calendar_long_horizon_synthetic.py",
        "src/lifetwin/experiments/calendar_long_horizon_v015_fixture.py",
    )
    assert observed.as_manifest_record()["package_versions"] == dict(
        environment._EXPECTED_PACKAGES
    )


def test_formal_environment_rejects_dirty_worktree_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repository(tmp_path)

    def fake_git(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repo.resolve())
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments[0] == "status":
            return "?? uncommitted.py"
        raise AssertionError(arguments)

    monkeypatch.setattr(environment, "_git", fake_git)
    with pytest.raises(environment.V015EnvironmentError, match="dirty"):
        environment.verify_formal_environment(repo)


def test_formal_environment_rejects_unlocked_thread_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )
    with pytest.raises(environment.V015EnvironmentError, match="not locked"):
        environment.verify_formal_environment(repo)


def test_late_pythonhashseed_mutation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)
    monkeypatch.setattr(environment, "_runtime_hash_sentinel", lambda: 0)

    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )
    with pytest.raises(environment.V015EnvironmentError, match="interpreter"):
        environment.verify_formal_environment(repo)


def test_loaded_native_threadpool_must_really_be_single_threaded(
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
    with pytest.raises(environment.V015EnvironmentError, match="thread pool"):
        environment._active_threadpool_records()


def test_formal_environment_rejects_source_bytes_outside_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)
    (
        repo
        / "src"
        / "lifetwin"
        / "experiments"
        / "calendar_long_horizon_v015_fixture.py"
    ).write_text("VALUE = 2\n", encoding="ascii")
    monkeypatch.setattr(
        environment,
        "_git",
        lambda root, *arguments: _clean_git(repo, *arguments),
    )

    with pytest.raises(
        environment.V015EnvironmentError,
        match="frozen implementation",
    ):
        environment.verify_formal_environment(repo)


def test_formal_environment_rejects_nonmetadata_change_after_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)

    def changed_git(root: Path, *arguments: str) -> str:
        if arguments[:2] == ("diff", "--name-only"):
            return "src/lifetwin/experiments/calendar_long_horizon_v015_fixture.py"
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", changed_git)
    with pytest.raises(
        environment.V015EnvironmentError,
        match="attestation allowlist",
    ):
        environment.verify_formal_environment(repo)


def test_formal_environment_requires_direct_attestation_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repository(tmp_path)
    _locked_environment(monkeypatch)

    def nonchild_git(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD^"):
            return "c" * 40
        return _clean_git(repo, *arguments)

    monkeypatch.setattr(environment, "_git", nonchild_git)
    with pytest.raises(
        environment.V015EnvironmentError,
        match="direct implementation attestation",
    ):
        environment.verify_formal_environment(repo)
