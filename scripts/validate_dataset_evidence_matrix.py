from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifetwin.experiments.nasa_prefix_loco import canonical_json_sha256
from lifetwin.validation.dataset_evidence_matrix import (
    validate_dataset_evidence_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen LifeTwin dataset evidence-role matrix."
    )
    parser.add_argument("matrix")
    args = parser.parse_args()
    matrix = validate_dataset_evidence_matrix(
        json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "matrix_id": matrix["matrix_id"],
                "dataset_count": len(matrix["datasets"]),
                "matrix_content_sha256": canonical_json_sha256(matrix),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
