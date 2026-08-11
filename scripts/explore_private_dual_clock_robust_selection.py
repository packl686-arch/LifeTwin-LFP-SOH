from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from lifetwin.experiments.private_dual_clock_prior_v3 import (
    predict_private_dual_clock_prior_v3,
    score_private_dual_clock_prior_v3,
    validate_private_dual_clock_prior_v3_config,
)
from lifetwin.private_artifacts import atomic_write_json


def run_robust_selection_grid(
    input_directory: Path,
    base_config: dict[str, object],
    penalties: tuple[float, ...],
) -> dict[str, object]:
    references = pd.read_parquet(input_directory / "outer_fold_references.parquet")
    prefixes = pd.read_parquet(input_directory / "target_prefixes.parquet")
    results = []
    for penalty in penalties:
        candidate = deepcopy(base_config)
        candidate["selection"]["worst_condition_penalty"] = float(penalty)
        config = validate_private_dual_clock_prior_v3_config(candidate)
        predictions, decisions, manifest = predict_private_dual_clock_prior_v3(
            references, prefixes, config
        )

        # Held-condition capacity suffixes are opened only after prediction freeze.
        truth = pd.read_parquet(input_directory / "target_truth.parquet")
        _, summary = score_private_dual_clock_prior_v3(
            truth, predictions, decisions, manifest, config
        )
        results.append(
            {
                "worst_condition_penalty": float(penalty),
                "prediction_manifest_content_sha256": manifest[
                    "manifest_content_sha256"
                ],
                "score_summary_content_sha256": summary[
                    "summary_content_sha256"
                ],
                "model_summary_by_landmark": summary[
                    "model_summary_by_landmark"
                ],
                "comparison_vs_v1_condition_prior": summary[
                    "comparison_vs_v1_condition_prior"
                ],
            }
        )
    return {
        "schema_version": "lifetwin.private_dual_clock_robust_selection.exploration.v1",
        "private_only": True,
        "evidence_role": "outcome_exposed_robust_selection_exploration",
        "candidate_penalties": list(penalties),
        "results": results,
        "claim_boundary": (
            "Retrospective private development on an outcome-exposed SNL cohort; "
            "any selected penalty requires new external confirmation."
        ),
        "public_release_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory", default="artifacts/snl-lfp-rpt-loco-v1"
    )
    parser.add_argument(
        "--base-config",
        default="artifacts/private-dual-clock-prior-v3/private_config.json",
    )
    parser.add_argument(
        "--penalties", nargs="+", type=float, default=[0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/private-dual-clock-prior-v3-post-outcome-audit/"
            "robust_selection_exploration.json"
        ),
    )
    args = parser.parse_args()
    base_config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    result = run_robust_selection_grid(
        Path(args.input_directory),
        base_config,
        tuple(float(value) for value in args.penalties),
    )
    atomic_write_json(result, Path(args.output))
    compact = {
        str(item["worst_condition_penalty"]): item[
            "comparison_vs_v1_condition_prior"
        ]
        for item in result["results"]
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
