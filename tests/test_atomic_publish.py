from __future__ import annotations

import os
from pathlib import Path

import pytest

from lifetwin import atomic_publish


def _source_and_destination(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "evidence.txt").write_text("complete\n", encoding="utf-8")
    return source, tmp_path / "published"


def _windows_permission_error(winerror: int) -> PermissionError:
    error = PermissionError(13, "access denied")
    error.winerror = winerror
    return error


def test_protocol_constants_are_frozen() -> None:
    assert atomic_publish.MAX_REPLACE_ATTEMPTS == 7
    assert atomic_publish.RETRY_DELAYS_SECONDS == (
        0.05,
        0.10,
        0.20,
        0.40,
        0.80,
        1.60,
    )


def test_first_replace_succeeds_without_sleep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    attempts = atomic_publish.publish_directory(source, destination)

    assert attempts == 1
    assert sleeps == []
    assert not source.exists()
    assert destination.is_dir()
    assert (destination / "evidence.txt").read_text(encoding="utf-8") == (
        "complete\n"
    )


def test_windows_access_denied_retries_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    real_replace = os.replace
    replace_calls = 0
    sleeps: list[float] = []

    def flaky_replace(current: Path, target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls < 3:
            raise _windows_permission_error(5)
        real_replace(current, target)

    monkeypatch.setattr(atomic_publish.sys, "platform", "win32")
    monkeypatch.setattr(atomic_publish.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    attempts = atomic_publish.publish_directory(source, destination)

    assert attempts == 3
    assert replace_calls == 3
    assert sleeps == [0.05, 0.10]
    assert not source.exists()
    assert destination.is_dir()


def test_windows_access_denied_exhaustion_retains_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    replace_calls = 0
    sleeps: list[float] = []

    def denied_replace(_current: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        raise _windows_permission_error(5)

    monkeypatch.setattr(atomic_publish.sys, "platform", "win32")
    monkeypatch.setattr(atomic_publish.os, "replace", denied_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    with pytest.raises(
        atomic_publish.AtomicPublishRetryExhausted,
        match="exhausted after 7 attempts",
    ) as captured:
        atomic_publish.publish_directory(source, destination)

    assert captured.value.source == source
    assert captured.value.destination == destination
    assert captured.value.attempts == 7
    assert replace_calls == 7
    assert sleeps == list(atomic_publish.RETRY_DELAYS_SECONDS)
    assert source.is_dir()
    assert not destination.exists()


def test_other_windows_permission_error_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    sleeps: list[float] = []
    replace_calls = 0

    def denied_replace(_current: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        raise _windows_permission_error(32)

    monkeypatch.setattr(atomic_publish.sys, "platform", "win32")
    monkeypatch.setattr(atomic_publish.os, "replace", denied_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    with pytest.raises(PermissionError):
        atomic_publish.publish_directory(source, destination)

    assert replace_calls == 1
    assert sleeps == []
    assert source.is_dir()


def test_non_windows_access_denied_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    sleeps: list[float] = []
    replace_calls = 0

    def denied_replace(_current: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        raise _windows_permission_error(5)

    monkeypatch.setattr(atomic_publish.sys, "platform", "linux")
    monkeypatch.setattr(atomic_publish.os, "replace", denied_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    with pytest.raises(PermissionError):
        atomic_publish.publish_directory(source, destination)

    assert replace_calls == 1
    assert sleeps == []
    assert source.is_dir()


def test_existing_destination_is_rejected_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    destination.mkdir()
    replace_calls = 0

    def unexpected_replace(_current: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1

    monkeypatch.setattr(atomic_publish.os, "replace", unexpected_replace)

    with pytest.raises(FileExistsError, match="never overwrites"):
        atomic_publish.publish_directory(source, destination)

    assert replace_calls == 0
    assert source.is_dir()
    assert destination.is_dir()


def test_destination_appearing_during_retry_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    replace_calls = 0

    def denied_replace(_current: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        raise _windows_permission_error(5)

    def destination_appears(_delay: float) -> None:
        destination.mkdir()

    monkeypatch.setattr(atomic_publish.sys, "platform", "win32")
    monkeypatch.setattr(atomic_publish.os, "replace", denied_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", destination_appears)

    with pytest.raises(FileExistsError, match="never overwrites"):
        atomic_publish.publish_directory(source, destination)

    assert replace_calls == 1
    assert source.is_dir()
    assert destination.is_dir()


def test_missing_source_is_rejected_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "missing-staging"
    destination = tmp_path / "published"
    replace_calls = 0

    def unexpected_replace(_current: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1

    monkeypatch.setattr(atomic_publish.os, "replace", unexpected_replace)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        atomic_publish.publish_directory(source, destination)

    assert replace_calls == 0
    assert not destination.exists()


def test_replace_noop_fails_postcondition_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination = _source_and_destination(tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(atomic_publish.os, "replace", lambda *_: None)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    with pytest.raises(
        atomic_publish.AtomicPublishPostconditionError,
        match="returned without moving",
    ):
        atomic_publish.publish_directory(source, destination)

    assert sleeps == []
    assert source.is_dir()
    assert not destination.exists()
