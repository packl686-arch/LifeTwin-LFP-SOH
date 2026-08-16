"""Verify the V3.0 frozen checkout without consuming its formal seed."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifetwin.experiments.runtime_reliability_v300_runner import (  # noqa: E402
    preflight_v300,
)


def main() -> int:
    try:
        report = asdict(preflight_v300(ROOT, require_authorization=False))
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": "lifetwin_v300_freeze_verifier/1.0.0",
                    "status": "failed",
                    "exception_class": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "schema_version": "lifetwin_v300_freeze_verifier/1.0.0",
                "status": "passed",
                "preflight": report,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
