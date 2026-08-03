from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.data.nasa_pcoe import prepare_nasa_pcoe_frames
from lifetwin.experiments.nasa_dynamic_gate_v2 import (
    build_nasa_dynamic_gate_fold_table,
    load_nasa_dynamic_gate_config,
)
from lifetwin.experiments.nasa_evidence_weighted_moe_v3 import (
    load_nasa_evidence_weighted_moe_config,
    predict_nasa_evidence_weighted_moe,
    score_nasa_evidence_weighted_moe,
)


DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/nasa-evidence-weighted-moe-v3")
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


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite V3 artifacts: "
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


def _load_configs(
    v2_config_path: str | Path,
    v3_config_path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        load_nasa_dynamic_gate_config(v2_config_path),
        load_nasa_evidence_weighted_moe_config(v3_config_path),
    )


def _run_cycles(
    cycles: pd.DataFrame,
    *,
    v2_config_path: str | Path,
    v3_config_path: str | Path,
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
    v2_config, v3_config = _load_configs(v2_config_path, v3_config_path)
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, v2_config)
    predictions, prediction_manifest = predict_nasa_evidence_weighted_moe(
        fold_table,
        v2_config,
        v3_config,
    )
    _write_csv(fold_table, paths["fold_table"])
    _write_csv(predictions, paths["predictions"])
    _write_json(prediction_manifest, paths["prediction_manifest"])
    scores, score_summary = score_nasa_evidence_weighted_moe(
        cycles,
        predictions,
        prediction_manifest,
        v2_config,
        v3_config,
    )
    _write_csv(scores, paths["scores"])
    _write_json(score_summary, paths["score_summary"])
    return score_summary


def _run_from_cycles(args: argparse.Namespace) -> int:
    cycles = pd.read_csv(args.cycles, float_precision="round_trip")
    summary = _run_cycles(
        cycles,
        v2_config_path=args.v2_config,
        v3_config_path=args.v3_config,
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
        v2_config_path=args.v2_config,
        v3_config_path=args.v3_config,
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
    v2_config, v3_config = _load_configs(args.v2_config, args.v3_config)
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, v2_config)
    predictions, manifest = predict_nasa_evidence_weighted_moe(
        fold_table,
        v2_config,
        v3_config,
    )
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
    fold_table = pd.read_csv(args.fold_table, float_precision="round_trip")
    predictions = pd.read_csv(args.predictions, float_precision="round_trip")
    manifest = json.loads(Path(args.prediction_manifest).read_text(encoding="utf-8"))
    v2_config, v3_config = _load_configs(args.v2_config, args.v3_config)
    rebuilt = build_nasa_dynamic_gate_fold_table(cycles, v2_config)
    if not rebuilt.equals(fold_table):
        raise ValueError("Persisted fold table differs from the rebuilt input")
    scores, summary = score_nasa_evidence_weighted_moe(
        cycles,
        predictions,
        manifest,
        v2_config,
        v3_config,
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["score_summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    parser.add_argument("--v3-config", default=str(DEFAULT_V3_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen NASA evidence-weighted mixture V3 benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_source = subparsers.add_parser("run-source")
    run_source.add_argument("source_directory")
    _add_common_config_arguments(run_source)
    run_source.set_defaults(handler=_run_from_source)

    run_cycles = subparsers.add_parser("run-cycles")
    run_cycles.add_argument("cycles")
    _add_common_config_arguments(run_cycles)
    run_cycles.set_defaults(handler=_run_from_cycles)

    predict = subparsers.add_parser("predict")
    predict.add_argument("cycles")
    _add_common_config_arguments(predict)
    predict.set_defaults(handler=_predict)

    score = subparsers.add_parser("score")
    score.add_argument("cycles")
    score.add_argument("fold_table")
    score.add_argument("predictions")
    score.add_argument("prediction_manifest")
    _add_common_config_arguments(score)
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
