from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from lifetwin.data.snl import (
    audit_snl_archive_structure,
    extract_snl_rpt_trajectories,
    load_snl_metadata,
)
from lifetwin.experiments.snl_rpt_loco import (
    build_snl_rpt_loco_inputs,
    load_snl_rpt_loco_config,
    predict_snl_rpt_loco,
    score_snl_rpt_loco,
)


DEFAULT_CONFIG = Path("configs/experiments/snl_lfp_rpt_loco_v1.json")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/snl-lfp-rpt-loco-v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _paths(output_directory: Path) -> dict[str, Path]:
    return {
        "metadata_audit": output_directory / "metadata_audit.json",
        "archive_audit": output_directory / "archive_structure_audit.json",
        "rpt": output_directory / "rpt_trajectories.parquet",
        "rpt_audit": output_directory / "rpt_extraction_audit.json",
        "references": output_directory / "outer_fold_references.parquet",
        "prefixes": output_directory / "target_prefixes.parquet",
        "truth": output_directory / "target_truth.parquet",
        "input_audit": output_directory / "prediction_input_audit.json",
        "predictions": output_directory / "predictions.parquet",
        "decisions": output_directory / "selector_decisions.parquet",
        "decisions_csv": output_directory / "selector_decisions.csv",
        "manifest": output_directory / "prediction_manifest.json",
        "scores": output_directory / "scores.csv",
        "summary": output_directory / "score_summary.json",
    }


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite SNL RPT LOCO artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional replay."
        )


def _prepare(args: argparse.Namespace) -> int:
    config = load_snl_rpt_loco_config(args.config)
    paths = _paths(Path(args.output_directory))
    generated = [
        paths["metadata_audit"],
        paths["archive_audit"],
        paths["rpt"],
        paths["rpt_audit"],
        paths["references"],
        paths["prefixes"],
        paths["truth"],
        paths["input_audit"],
    ]
    _ensure_available(generated, overwrite=args.overwrite)
    zip_path = Path(args.raw_zip)
    metadata_path = Path(args.metadata_xlsx)
    if _sha256(zip_path) != config["dataset"]["raw_zip_sha256"]:
        raise ValueError("SNL raw ZIP SHA-256 changed")
    if _sha256(metadata_path) != config["dataset"]["metadata_xlsx_sha256"]:
        raise ValueError("SNL metadata workbook SHA-256 changed")
    metadata, metadata_audit = load_snl_metadata(
        metadata_path,
        expected_lfp_rows_sha256=config["dataset"][
            "metadata_lfp_rows_canonical_sha256"
        ],
        expected_cell_count=int(config["dataset"]["physical_cell_count"]),
        expected_condition_count=int(config["dataset"]["condition_cluster_count"]),
    )
    archive_audit = audit_snl_archive_structure(zip_path, metadata)
    trajectories, rpt_audit = extract_snl_rpt_trajectories(
        zip_path,
        metadata,
        rest_gap_hours=float(config["rpt_adapter"]["long_rest_gap_hours"]),
        duplicate_visit_efc=float(
            config["rpt_adapter"]["adjacent_check_collapse_maximum_efc"]
        ),
    )
    references, prefixes, truth, input_audit = build_snl_rpt_loco_inputs(
        trajectories, config
    )
    _write_json(metadata_audit, paths["metadata_audit"])
    _write_json(archive_audit, paths["archive_audit"])
    _write_parquet(trajectories, paths["rpt"])
    _write_json(rpt_audit, paths["rpt_audit"])
    _write_parquet(references, paths["references"])
    _write_parquet(prefixes, paths["prefixes"])
    _write_parquet(truth, paths["truth"])
    _write_json(input_audit, paths["input_audit"])
    print(
        json.dumps(
            {
                "rpt_extraction_audit": rpt_audit,
                "prediction_input_audit": input_audit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _predict(args: argparse.Namespace) -> int:
    config = load_snl_rpt_loco_config(args.config)
    paths = _paths(Path(args.output_directory))
    generated = [
        paths["predictions"],
        paths["decisions"],
        paths["decisions_csv"],
        paths["manifest"],
    ]
    _ensure_available(generated, overwrite=args.overwrite)
    references = pd.read_parquet(args.references)
    prefixes = pd.read_parquet(args.target_prefixes)
    predictions, decisions, manifest = predict_snl_rpt_loco(
        references, prefixes, config
    )
    _write_parquet(predictions, paths["predictions"])
    _write_parquet(decisions, paths["decisions"])
    _write_csv(decisions, paths["decisions_csv"])
    _write_json(manifest, paths["manifest"])
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _score(args: argparse.Namespace) -> int:
    config = load_snl_rpt_loco_config(args.config)
    paths = _paths(Path(args.output_directory))
    generated = [paths["scores"], paths["summary"]]
    _ensure_available(generated, overwrite=args.overwrite)
    truth = pd.read_parquet(args.target_truth)
    predictions = pd.read_parquet(args.predictions)
    decisions = pd.read_parquet(args.selector_decisions)
    manifest = json.loads(Path(args.prediction_manifest).read_text(encoding="utf-8"))
    scores, summary = score_snl_rpt_loco(
        truth, predictions, decisions, manifest, config
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen retrospective SNL LFP RPT leave-one-condition-out "
            "development benchmark in explicit prepare, predict, and score phases"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("raw_zip")
    prepare.add_argument("metadata_xlsx")
    _common(prepare)
    prepare.set_defaults(handler=_prepare)

    predict = subparsers.add_parser("predict")
    predict.add_argument("references")
    predict.add_argument("target_prefixes")
    _common(predict)
    predict.set_defaults(handler=_predict)

    score = subparsers.add_parser("score")
    score.add_argument("target_truth")
    score.add_argument("predictions")
    score.add_argument("selector_decisions")
    score.add_argument("prediction_manifest")
    _common(score)
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
