from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.data.nasa_pcoe import prepare_nasa_pcoe_frames
from lifetwin.experiments.nasa_dynamic_gate_v2 import (
    build_nasa_dynamic_gate_fold_table,
    load_nasa_dynamic_gate_config,
    predict_nasa_dynamic_gate,
    score_nasa_dynamic_gate,
)


DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/nasa-dynamic-gate-v2")
DEFAULT_CONFIG = Path("configs/experiments/nasa_dynamic_gate_v2.json")


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


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite V2 artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional local rerun."
        )


def _artifact_paths(output_directory: Path) -> dict[str, Path]:
    return {
        "cycles": output_directory / "cycles.csv",
        "labels": output_directory / "labels.csv",
        "inventory": output_directory / "inventory.csv",
        "ingest_audit": output_directory / "ingest_audit.json",
        "fold_table": output_directory / "fold_table.csv",
        "predictions": output_directory / "predictions.csv",
        "prediction_manifest": output_directory / "prediction_manifest.json",
        "scores": output_directory / "scores.csv",
        "score_summary": output_directory / "score_summary.json",
    }


def _run_cycles(
    cycles: pd.DataFrame,
    *,
    config_path: str | Path,
    output_directory: Path,
    overwrite: bool,
) -> dict[str, object]:
    paths = _artifact_paths(output_directory)
    generated = [
        paths["fold_table"],
        paths["predictions"],
        paths["prediction_manifest"],
        paths["scores"],
        paths["score_summary"],
    ]
    _ensure_available(generated, overwrite=overwrite)
    config = load_nasa_dynamic_gate_config(config_path)
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, config)
    predictions, prediction_manifest = predict_nasa_dynamic_gate(
        fold_table,
        config,
    )
    _write_csv(fold_table, paths["fold_table"])
    _write_csv(predictions, paths["predictions"])
    _write_json(prediction_manifest, paths["prediction_manifest"])
    scores, score_summary = score_nasa_dynamic_gate(
        cycles,
        predictions,
        prediction_manifest,
        config,
    )
    _write_csv(scores, paths["scores"])
    _write_json(score_summary, paths["score_summary"])
    return score_summary


def _run_from_cycles(args: argparse.Namespace) -> int:
    cycles = pd.read_csv(args.cycles, float_precision="round_trip")
    summary = _run_cycles(
        cycles,
        config_path=args.config,
        output_directory=Path(args.output_directory),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _run_from_source(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _artifact_paths(output_directory)
    source_paths = [
        paths["cycles"],
        paths["labels"],
        paths["inventory"],
        paths["ingest_audit"],
    ]
    _ensure_available(source_paths, overwrite=args.overwrite)
    cycles, labels, inventory, audit = prepare_nasa_pcoe_frames(args.source_directory)
    _write_csv(cycles, paths["cycles"])
    _write_csv(labels, paths["labels"])
    _write_csv(inventory, paths["inventory"])
    _write_json(audit, paths["ingest_audit"])
    summary = _run_cycles(
        cycles,
        config_path=args.config,
        output_directory=output_directory,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _predict(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _artifact_paths(output_directory)
    generated = [
        paths["fold_table"],
        paths["predictions"],
        paths["prediction_manifest"],
    ]
    _ensure_available(generated, overwrite=args.overwrite)
    cycles = pd.read_csv(args.cycles, float_precision="round_trip")
    config = load_nasa_dynamic_gate_config(args.config)
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, config)
    predictions, manifest = predict_nasa_dynamic_gate(fold_table, config)
    _write_csv(fold_table, paths["fold_table"])
    _write_csv(predictions, paths["predictions"])
    _write_json(manifest, paths["prediction_manifest"])
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _score(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _artifact_paths(output_directory)
    generated = [paths["scores"], paths["score_summary"]]
    _ensure_available(generated, overwrite=args.overwrite)
    cycles = pd.read_csv(args.cycles, float_precision="round_trip")
    predictions = pd.read_csv(args.predictions, float_precision="round_trip")
    manifest = json.loads(Path(args.prediction_manifest).read_text(encoding="utf-8"))
    config = load_nasa_dynamic_gate_config(args.config)
    scores, summary = score_nasa_dynamic_gate(
        cycles,
        predictions,
        manifest,
        config,
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["score_summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the nested-LOCO NASA dynamic-gate V2 development benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_source = subparsers.add_parser("run-source")
    run_source.add_argument("source_directory")
    run_source.add_argument("--config", default=str(DEFAULT_CONFIG))
    run_source.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    run_source.add_argument("--overwrite", action="store_true")
    run_source.set_defaults(handler=_run_from_source)

    run_cycles = subparsers.add_parser("run-cycles")
    run_cycles.add_argument("cycles")
    run_cycles.add_argument("--config", default=str(DEFAULT_CONFIG))
    run_cycles.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    run_cycles.add_argument("--overwrite", action="store_true")
    run_cycles.set_defaults(handler=_run_from_cycles)

    predict = subparsers.add_parser("predict")
    predict.add_argument("cycles")
    predict.add_argument("--config", default=str(DEFAULT_CONFIG))
    predict.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    predict.add_argument("--overwrite", action="store_true")
    predict.set_defaults(handler=_predict)

    score = subparsers.add_parser("score")
    score.add_argument("cycles")
    score.add_argument("predictions")
    score.add_argument("prediction_manifest")
    score.add_argument("--config", default=str(DEFAULT_CONFIG))
    score.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
