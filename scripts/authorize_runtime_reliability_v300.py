"""Create the fixed V3.0 authorization record after an explicit user decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifetwin.experiments.runtime_reliability_v300_runner import (  # noqa: E402
    create_v300_authorization,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Authorize the sole frozen V3.0 attempt."
    )
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--authorized-by", required=True)
    args = parser.parse_args()
    try:
        path = create_v300_authorization(
            ROOT,
            authorization_id=args.authorization_id,
            authorized_by=args.authorized_by,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": "lifetwin_v300_authorization_error/1.0.0",
                    "status": "failed_closed",
                    "exception_class": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema_version": "lifetwin_v300_authorization_cli/1.0.0",
                "status": "authorized_post_freeze",
                "record": path.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
