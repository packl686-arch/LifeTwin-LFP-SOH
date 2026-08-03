from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from lifetwin.data.nasa_pcoe import prepare_nasa_pcoe_frames
from lifetwin.experiments.nasa_prefix_loco import (
    build_nasa_prefix_table,
    load_nasa_prefix_loco_config,
    predict_nasa_prefix_loco,
    score_nasa_prefix_loco,
)


DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/nasa-prefix-loco-v1")
DEFAULT_CONFIG = Path("configs/experiments/nasa_prefix_loco_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite NASA benchmark artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional local rerun."
        )


def _write_canonical_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def _prepare(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    cycles_path = output_directory / "cycles.csv"
    labels_path = output_directory / "labels.csv"
    inventory_path = output_directory / "inventory.csv"
    audit_path = output_directory / "ingest_audit.json"
    _ensure_available(
        [cycles_path, labels_path, inventory_path, audit_path],
        overwrite=args.overwrite,
    )
    cycles, labels, inventory, audit = prepare_nasa_pcoe_frames(
        args.source_directory,
    )
    _write_canonical_csv(cycles, cycles_path)
    _write_canonical_csv(labels, labels_path)
    _write_canonical_csv(inventory, inventory_path)
    audit["artifacts"] = {
        "cycle_summary": {
            "path": cycles_path.as_posix(),
            "row_count": len(cycles),
            "sha256": _sha256(cycles_path),
        },
        "cell_labels": {
            "path": labels_path.as_posix(),
            "row_count": len(labels),
            "sha256": _sha256(labels_path),
        },
        "source_inventory": {
            "path": inventory_path.as_posix(),
            "row_count": len(inventory),
            "sha256": _sha256(inventory_path),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def _predict(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    predictions_path = output_directory / "predictions.csv"
    manifest_path = output_directory / "prediction_manifest.json"
    _ensure_available(
        [predictions_path, manifest_path],
        overwrite=args.overwrite,
    )
    cycles = pd.read_csv(args.cycles, float_precision="round_trip")
    config = load_nasa_prefix_loco_config(args.config)
    prefix_table = build_nasa_prefix_table(cycles, config)
    predictions, manifest = predict_nasa_prefix_loco(prefix_table, config)
    _write_canonical_csv(predictions, predictions_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _score(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    scores_path = output_directory / "scores.csv"
    summary_path = output_directory / "score_summary.json"
    _ensure_available(
        [scores_path, summary_path],
        overwrite=args.overwrite,
    )
    cycles = pd.read_csv(args.cycles, float_precision="round_trip")
    predictions = pd.read_csv(args.predictions, float_precision="round_trip")
    manifest = json.loads(Path(args.prediction_manifest).read_text(encoding="utf-8"))
    config = load_nasa_prefix_loco_config(args.config)
    scores, summary = score_nasa_prefix_loco(
        cycles,
        predictions,
        manifest,
        config,
    )
    _write_canonical_csv(scores, scores_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the claim-bounded NASA PCoE auxiliary benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("source_directory")
    prepare.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=_prepare)

    predict = subparsers.add_parser("predict")
    predict.add_argument("cycles")
    predict.add_argument("--config", default=str(DEFAULT_CONFIG))
    predict.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    predict.add_argument("--overwrite", action="store_true")
    predict.set_defaults(handler=_predict)

    score = subparsers.add_parser("score")
    score.add_argument("cycles")
    score.add_argument("predictions")
    score.add_argument("prediction_manifest")
    score.add_argument("--config", default=str(DEFAULT_CONFIG))
    score.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
