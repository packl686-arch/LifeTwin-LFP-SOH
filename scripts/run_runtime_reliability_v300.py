"""Capability-minimal CLI for the frozen V3.0 runtime-reliability study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifetwin.experiments.runtime_reliability_v300_runner import (  # noqa: E402
    execute_v300_formal_attempt,
    preflight_v300,
)


def _emit(payload: object) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight or execute the sole frozen V3.0 formal attempt."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            _emit(asdict(preflight_v300(ROOT, require_authorization=False)))
            return 0
        result = execute_v300_formal_attempt(ROOT)
    except Exception as error:
        _emit(
            {
                "schema_version": "lifetwin_v300_cli_error/1.0.0",
                "status": "failed_closed",
                "exception_class": type(error).__name__,
            }
        )
        return 2
    _emit(
        {
            "schema_version": "lifetwin_v300_cli_terminal/1.0.0",
            "attempt_id": result["attempt_id"],
            "disposition": result["disposition"],
        }
    )
    return 0 if result["disposition"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
