from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifetwin.validation.long_term_protocol import (
    IndependentLongTermProtocolValidationError,
    validate_independent_long_term_protocol,
)


DEFAULT_SCHEMA = Path(
    "configs/validation/independent_long_term_lfp_protocol.schema.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a dataset-specific LifeTwin long-term LFP protocol using "
            "JSON Schema and the cross-field semantic gates."
        )
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    try:
        payload = json.loads(args.protocol.read_text(encoding="utf-8"))
        validated = validate_independent_long_term_protocol(
            payload,
            schema_path=args.schema,
        )
    except (
        OSError,
        json.JSONDecodeError,
        IndependentLongTermProtocolValidationError,
    ) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "protocol_id": validated["protocol_id"],
                "protocol_status": validated["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
