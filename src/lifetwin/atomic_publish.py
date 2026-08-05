from __future__ import annotations

import os
from pathlib import Path
import sys
import time


MAX_REPLACE_ATTEMPTS = 7
RETRY_DELAYS_SECONDS = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
WINDOWS_ACCESS_DENIED = 5


class AtomicPublishRetryExhausted(RuntimeError):
    """A retryable Windows directory publication failed on every attempt."""

    def __init__(self, source: Path, destination: Path, attempts: int) -> None:
        self.source = source
        self.destination = destination
        self.attempts = attempts
        super().__init__(
            "Atomic directory publish exhausted after "
            f"{attempts} attempts; source staging retained: "
            f"source={source}, destination={destination}"
        )


class AtomicPublishPostconditionError(RuntimeError):
    """The directory replacement returned without satisfying publication."""


def _validate_attempt(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(
            f"Atomic publish source staging directory does not exist: {source}"
        )
    if os.path.lexists(destination):
        raise FileExistsError(
            "Atomic publish never overwrites an existing destination: "
            f"{destination}"
        )


def publish_directory(source: Path, destination: Path) -> int:
    """Publish one staging directory without overwriting an existing target.

    Only Windows ``PermissionError`` instances carrying ``winerror == 5`` are
    retried. The fixed retry loop surrounds only ``os.replace``; callers must
    perform all scientific computation before invoking this function.
    """

    source = Path(source)
    destination = Path(destination)
    for attempt in range(1, MAX_REPLACE_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(RETRY_DELAYS_SECONDS[attempt - 2])
        _validate_attempt(source, destination)
        try:
            os.replace(source, destination)
        except PermissionError as error:
            retryable = (
                sys.platform == "win32"
                and getattr(error, "winerror", None) == WINDOWS_ACCESS_DENIED
            )
            if not retryable:
                raise
            if attempt == MAX_REPLACE_ATTEMPTS:
                raise AtomicPublishRetryExhausted(
                    source,
                    destination,
                    attempt,
                ) from error
            continue

        if os.path.lexists(source) or not destination.is_dir():
            raise AtomicPublishPostconditionError(
                "Atomic directory publish returned without moving the source: "
                f"source={source}, destination={destination}, attempt={attempt}"
            )
        return attempt

    raise AssertionError("unreachable atomic publish retry state")
