from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.experiments.nasa_dynamic_gate_v2 import load_nasa_dynamic_gate_config
from lifetwin.experiments.nasa_evidence_weighted_moe_v3 import (
    load_nasa_evidence_weighted_moe_config,
)
from lifetwin.experiments.nasa_v3_post_outcome_audit import (
    audit_nasa_v3_result,
    run_nasa_v3_attacks,
)


DEFAULT_INPUT_DIRECTORY = Path("artifacts/nasa-evidence-weighted-moe-v3")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/nasa-v3-post-outcome-audit")
DEFAULT_CYCLES = Path("artifacts/nasa-dynamic-gate-v2/cycles.csv")
DEFAULT_V2_CONFIG = Path("configs/experiments/nasa_dynamic_gate_v2.json")
DEFAULT_V3_CONFIG = Path("configs/experiments/nasa_evidence_weighted_moe_v3.json")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _output_paths(output_directory: Path) -> dict[str, Path]:
    return {
        "ablation_scores": output_directory / "ablation_fold_scores.csv",
        "prefix_summary": output_directory / "ablation_prefix_summary.csv",
        "evidence": output_directory / "evidence_diagnostics.csv",
        "attacks": output_directory / "attack_results.csv",
        "summary": output_directory / "audit_summary.json",
    }


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite NASA V3 audit artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional rerun."
        )


def run(args: argparse.Namespace) -> int:
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)
    paths = _output_paths(output_directory)
    _ensure_available(list(paths.values()), overwrite=args.overwrite)
    cycles = pd.read_csv(args.cycles, float_precision="round_trip")
    predictions = pd.read_csv(
        input_directory / "predictions.csv",
        float_precision="round_trip",
    )
    scores = pd.read_csv(
        input_directory / "scores.csv",
        float_precision="round_trip",
    )
    prediction_manifest = json.loads(
        (input_directory / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    score_summary = json.loads(
        (input_directory / "score_summary.json").read_text(encoding="utf-8")
    )
    v2_config = load_nasa_dynamic_gate_config(args.v2_config)
    v3_config = load_nasa_evidence_weighted_moe_config(args.v3_config)
    ablation, prefix_summary, evidence, summary = audit_nasa_v3_result(
        cycles,
        predictions,
        prediction_manifest,
        scores,
        score_summary,
        v2_config,
        v3_config,
    )
    attacks, attack_summary = run_nasa_v3_attacks(
        cycles,
        predictions,
        v2_config,
        v3_config,
    )
    summary["attack_audit"] = attack_summary
    _write_csv(ablation, paths["ablation_scores"])
    _write_csv(prefix_summary, paths["prefix_summary"])
    _write_csv(evidence, paths["evidence"])
    _write_csv(attacks, paths["attacks"])
    _write_json(summary, paths["summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed post-outcome diagnostics for the frozen NASA V3 result"
    )
    parser.add_argument("--input-directory", default=str(DEFAULT_INPUT_DIRECTORY))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--cycles", default=str(DEFAULT_CYCLES))
    parser.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    parser.add_argument("--v3-config", default=str(DEFAULT_V3_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
