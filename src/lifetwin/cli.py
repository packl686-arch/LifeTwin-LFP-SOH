from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys

import pandas as pd

from lifetwin.data.beep import prepare_fastcharge_frames
from lifetwin.data.attia import (
    ATTIA_AUTHOR_CODE_COMMIT,
    ATTIA_AUTHORITY_STATUS,
    ATTIA_CELLJAR_COMMIT,
    ATTIA_CROSSWALK_METHOD,
    ATTIA_DATASET_ID,
    ATTIA_DATASET_SNAPSHOT_ID,
    ATTIA_FINAL_RESULTS_SHA256,
    ATTIA_FINAL_RESULTS_URL,
    ATTIA_LABEL_STRATEGY,
    ATTIA_LABEL_VERSION,
    ATTIA_OUTCOME_SCHEMA_VERSION,
    ATTIA_TARGET_TIMESERIES_SHA256,
    attia_outcome_artifact_sha256,
    build_attia_outcome_pack,
    load_attia_label_free_metadata,
    validate_attia_outcome_pack,
)
from lifetwin.data.extract import extract_celljar_curve_subset
from lifetwin.data.schema import validate_cycle_summary
from lifetwin.data.split import stable_group_split
from lifetwin.data.synthetic import make_synthetic_cycle_data
from lifetwin.data.celljar import load_matr_metadata, matr_metadata_audit
from lifetwin.data.naumann import load_naumann_calendar_observations
from lifetwin.experiments.calendar_aging import run_calendar_aging_backtest
from lifetwin.experiments.external_validation import (
    EXTERNAL_PREDICTION_SCHEMA_VERSION,
    build_source_only_external_predictions,
    external_prediction_artifact_sha256,
    external_target_feature_identity_sha256,
    score_source_only_external_predictions,
)
from lifetwin.experiments.probabilistic import (
    run_probabilistic_lifetime_experiment,
    run_tuned_probabilistic_lifetime_experiment,
)
from lifetwin.experiments.ipcw_validation import run_synthetic_ipcw_validation
from lifetwin.experiments.reference_proxy import (
    run_batch_reference_prior_scale_sensitivity,
    run_batch_reference_proxy_experiment,
)
from lifetwin.evaluation.reference_cells import reference_cell_support, support_gate
from lifetwin.features.early import extract_early_cycle_features
from lifetwin.features.curves import extract_delta_q_features
from lifetwin.features.external_curves import extract_external_delta_q_features
from lifetwin.models.baselines import run_baselines


MATR_CURVE_FEATURE_COLUMNS = (
    "delta_q_min_ah",
    "delta_q_max_ah",
    "delta_q_mean_ah",
    "log10_delta_q_variance",
    "delta_q_skewness",
    "delta_q_kurtosis",
    "delta_q_abs_area_ah_v",
    "q_early_mean_ah",
    "q_late_mean_ah",
    "temperature_mean_c",
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _write_frozen_prediction_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise ValueError("Frozen prediction artifacts must use canonical CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_hashes(project_root: Path) -> dict[str, str]:
    paths = [project_root / "pyproject.toml"]
    paths.extend(sorted((project_root / "src" / "lifetwin").rglob("*.py")))
    return {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in paths
    }


def _source_tree_sha256(source_hashes: dict[str, str]) -> str:
    payload = json.dumps(
        source_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_outputs_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing frozen artifacts: "
            + ", ".join(existing)
            + ". Use new output paths or pass --overwrite explicitly."
        )


def _split_map_sha256(frame: pd.DataFrame) -> str:
    candidates = (
        (
            "dataset_id",
            "condition_id",
            "cell_id",
            "test_id",
            "temperature_c",
            "storage_soc_fraction",
        )
        if "condition_id" in frame
        else (
            "dataset_id",
            "cell_id",
            "batch_id",
            "protocol_id",
            "split_cell",
            "split_protocol",
            "paper_split",
        )
    )
    columns = [
        column
        for column in candidates
        if column in frame
    ]
    normalized = (
        frame[columns]
        .fillna("<NA>")
        .astype(str)
        .sort_values(columns, kind="stable")
    )
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _experiment_provenance(
    *,
    input_path: Path,
    features: pd.DataFrame,
    config_path: Path,
) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
    project_root = Path(__file__).resolve().parents[2]
    source_hashes = _source_tree_hashes(project_root)
    source_tree_hash = _source_tree_sha256(source_hashes)
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        packages[package] = importlib_metadata.version(package)
    input_hash = _sha256_file(input_path)
    config_hash = _sha256_file(config_path)
    split_hash = _split_map_sha256(features)
    run_payload = json.dumps(
        {
            "input": input_hash,
            "config": config_hash,
            "split_map": split_hash,
            "source": source_hashes,
            "source_tree": source_tree_hash,
            "python": sys.version,
            "packages": packages,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "run_id": hashlib.sha256(run_payload).hexdigest()[:20],
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "input_sha256": input_hash,
        "split_map_sha256": split_hash,
        "source_sha256": source_hashes,
        "source_tree_sha256": source_tree_hash,
        "python": sys.version,
        "packages": packages,
    }


def _synthetic_ipcw_provenance(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
    project_root = Path(__file__).resolve().parents[2]
    source_hashes = _source_tree_hashes(project_root)
    source_tree_hash = _source_tree_sha256(source_hashes)
    packages = {
        package: importlib_metadata.version(package)
        for package in ("numpy", "pandas", "scipy", "scikit-learn")
    }
    config_hash = _sha256_file(config_path)
    run_payload = json.dumps(
        {
            "config": config_hash,
            "source": source_hashes,
            "source_tree": source_tree_hash,
            "python": sys.version,
            "packages": packages,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "run_id": hashlib.sha256(run_payload).hexdigest()[:20],
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "source_sha256": source_hashes,
        "source_tree_sha256": source_tree_hash,
        "python": sys.version,
        "packages": packages,
    }


def _load_experiment_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal SHA-256") from exc
    return text.lower()


def _validate_attia_external_config(config: dict[str, object]) -> None:
    required = {
        "experiment_id",
        "source_dataset_id",
        "source_dataset_snapshot_id",
        "source_feature_pack_sha256",
        "source_label_table_sha256",
        "target_dataset_id",
        "target_dataset_snapshot_id",
        "target_timeseries_sha256",
        "target_feature_pack_sha256",
        "target_label_version",
        "target_label_sha256",
        "target_author_code_commit",
        "target_label_source_url",
        "target_label_authority_status",
        "target_label_strategy",
        "target_crosswalk_method",
        "target_outcome_schema_version",
        "target_outcome_pack_sha256",
        "celljar_commit",
        "feature_column",
        "source_cell_count",
        "target_cell_count",
        "target_protocol_count",
        "cells_per_target_protocol",
        "evidence_role",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing frozen Attia config fields: {missing}")
    expected = {
        "target_dataset_id": ATTIA_DATASET_ID,
        "target_dataset_snapshot_id": ATTIA_DATASET_SNAPSHOT_ID,
        "target_timeseries_sha256": ATTIA_TARGET_TIMESERIES_SHA256,
        "target_label_version": ATTIA_LABEL_VERSION,
        "target_label_sha256": ATTIA_FINAL_RESULTS_SHA256,
        "target_author_code_commit": ATTIA_AUTHOR_CODE_COMMIT,
        "target_label_source_url": ATTIA_FINAL_RESULTS_URL,
        "target_label_authority_status": ATTIA_AUTHORITY_STATUS,
        "target_label_strategy": ATTIA_LABEL_STRATEGY,
        "target_crosswalk_method": ATTIA_CROSSWALK_METHOD,
        "target_outcome_schema_version": ATTIA_OUTCOME_SCHEMA_VERSION,
        "celljar_commit": ATTIA_CELLJAR_COMMIT,
        "target_cell_count": 45,
        "target_protocol_count": 9,
        "cells_per_target_protocol": 5,
    }
    for field, expected_value in expected.items():
        if config[field] != expected_value:
            raise ValueError(f"Frozen Attia config identity mismatch for {field}")
    for field in (
        "source_label_table_sha256",
        "source_feature_pack_sha256",
        "target_timeseries_sha256",
        "target_feature_pack_sha256",
        "target_label_sha256",
        "target_outcome_pack_sha256",
    ):
        _require_sha256(config[field], field=f"config.{field}")


def _attia_prediction_experiment_identity(
    config: dict[str, object],
) -> dict[str, object]:
    keys = (
        "experiment_id",
        "source_dataset_id",
        "source_dataset_snapshot_id",
        "source_feature_pack_sha256",
        "source_label_table_sha256",
        "target_dataset_id",
        "target_dataset_snapshot_id",
        "target_timeseries_sha256",
        "target_label_version",
        "target_label_sha256",
        "feature_column",
        "evidence_role",
    )
    return {key: config[key] for key in keys}


def _attia_outcome_identity(config: dict[str, object]) -> dict[str, object]:
    return {
        "dataset_id": config["target_dataset_id"],
        "dataset_snapshot_id": config["target_dataset_snapshot_id"],
        "label_version": config["target_label_version"],
        "author_code_commit": config["target_author_code_commit"],
        "author_final_results_sha256": config["target_label_sha256"],
        "label_source_url": config["target_label_source_url"],
        "authority_status": config["target_label_authority_status"],
        "label_strategy": config["target_label_strategy"],
        "crosswalk_method": config["target_crosswalk_method"],
        "outcome_schema_version": config["target_outcome_schema_version"],
        "celljar_commit": config["celljar_commit"],
    }


def _validate_attia_prediction_manifest(
    manifest: dict[str, object],
    *,
    config: dict[str, object],
    config_sha256: str,
    predictions: pd.DataFrame,
    predictions_sha256: str,
) -> str:
    """Bind score inputs to the exact frozen prediction context."""
    if not isinstance(manifest, dict):
        raise ValueError("Prediction manifest must be a JSON object")
    if manifest.get("status") != "predictions_frozen_before_target_outcome_access":
        raise ValueError("Prediction manifest does not have frozen status")
    if (
        manifest.get("prediction_schema_version")
        != EXTERNAL_PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("Prediction manifest schema version is not frozen")
    frozen_hash = _require_sha256(
        manifest.get("frozen_prediction_sha256"),
        field="prediction_manifest.frozen_prediction_sha256",
    )
    if external_prediction_artifact_sha256(predictions) != frozen_hash:
        raise ValueError("Prediction manifest hash does not match prediction content")
    if manifest.get("experiment") != _attia_prediction_experiment_identity(config):
        raise ValueError("Prediction manifest experiment identity does not match config")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Prediction manifest provenance is missing")
    if provenance.get("config_sha256") != config_sha256:
        raise ValueError("Prediction manifest config SHA does not match score config")
    source_feature_sha = _require_sha256(
        provenance.get("source_features_sha256"),
        field="prediction_manifest.provenance.source_features_sha256",
    )
    target_feature_sha = _require_sha256(
        provenance.get("target_features_sha256"),
        field="prediction_manifest.provenance.target_features_sha256",
    )
    if source_feature_sha != str(config["source_feature_pack_sha256"]):
        raise ValueError("Prediction manifest source feature SHA does not match config")
    if target_feature_sha != str(config["target_feature_pack_sha256"]):
        raise ValueError("Prediction manifest target feature SHA does not match config")

    source_authority = manifest.get("source_label_authority")
    expected_source_authority = {
        "column": "authoritative_crosswalk_sha256",
        "sha256": str(config["source_label_table_sha256"]),
        "verified_before_fit": True,
    }
    if source_authority != expected_source_authority:
        raise ValueError("Prediction manifest source label authority is not frozen")

    target_feature_pack = manifest.get("target_feature_pack")
    if not isinstance(target_feature_pack, dict):
        raise ValueError("Prediction manifest target feature pack is missing")
    target_feature_identity = external_target_feature_identity_sha256(
        predictions,
        feature_column=str(config["feature_column"]),
    )
    expected_target_feature_pack = {
        "dataset_id": config["target_dataset_id"],
        "dataset_snapshot_id": config["target_dataset_snapshot_id"],
        "feature_column": config["feature_column"],
        "row_count": int(config["target_cell_count"]),
        "sha256": target_feature_sha,
        "identity_sha256": target_feature_identity,
    }
    if target_feature_pack != expected_target_feature_pack:
        raise ValueError("Prediction manifest target feature identity is not frozen")

    source_feature_pack = manifest.get("source_feature_pack")
    expected_source_feature_pack = {
        "dataset_id": config["source_dataset_id"],
        "dataset_snapshot_id": config["source_dataset_snapshot_id"],
        "row_count": int(config["source_cell_count"]),
        "sha256": source_feature_sha,
        "authoritative_crosswalk_sha256": config["source_label_table_sha256"],
    }
    if source_feature_pack != expected_source_feature_pack:
        raise ValueError("Prediction manifest source feature identity is not frozen")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Prediction manifest artifact identity is missing")
    if artifact.get("row_count") != int(config["target_cell_count"]):
        raise ValueError("Prediction manifest artifact row count does not match config")
    if artifact.get("sha256") != predictions_sha256:
        raise ValueError("Prediction file SHA does not match prediction manifest")
    if artifact.get("canonical_prediction_sha256") != frozen_hash:
        raise ValueError("Prediction artifact canonical hash is inconsistent")
    return frozen_hash


def _experiment_identity(config: dict[str, object]) -> dict[str, object]:
    keys = (
        "experiment_id",
        "dataset_id",
        "dataset_snapshot_id",
        "label_version",
        "label_table_sha256",
        "evidence_role",
    )
    return {key: config[key] for key in keys if key in config}


def _require_frozen_value(name: str, cli_value: object, config_value: object) -> None:
    if cli_value != config_value:
        raise ValueError(
            f"CLI {name}={cli_value!r} does not match frozen config value "
            f"{config_value!r}. Edit and version a new config instead of overriding it."
        )


def _validate_cycles(args: argparse.Namespace) -> int:
    report = validate_cycle_summary(_read_table(Path(args.input)))
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0


def _extract_features(args: argparse.Namespace) -> int:
    cycles = _read_table(Path(args.cycles))
    labels = _read_table(Path(args.labels))
    features = extract_early_cycle_features(
        cycles,
        labels,
        observation_cycle=args.observation_cycle,
        minimum_observed_cycles=args.minimum_observed_cycles,
    )
    _write_table(features, Path(args.output))
    print(f"Wrote {len(features)} cell-level rows to {args.output}")
    return 0


def _synthetic_smoke(args: argparse.Namespace) -> int:
    cycles, labels = make_synthetic_cycle_data()
    features = extract_early_cycle_features(cycles, labels)
    features["split"] = stable_group_split(
        features,
        ["cell_id"],
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )
    result = {
        "warning": "Synthetic data validate software only; these are not research results.",
        "cell_count": len(features),
        "metrics": run_baselines(features),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _prepare_matr_metadata(args: argparse.Namespace) -> int:
    metadata = load_matr_metadata(
        args.celljar_repository,
        args.authoritative_crosswalk,
        official_cohort_only=True,
    )
    metadata["split_cell"] = stable_group_split(metadata, ["cell_id"], seed=42)
    metadata["split_protocol"] = stable_group_split(metadata, ["protocol_id"], seed=42)
    output = Path(args.output)
    _write_table(metadata, output)
    audit = matr_metadata_audit(metadata)
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def _prepare_naumann_calendar(args: argparse.Namespace) -> int:
    output = Path(args.output)
    audit_output = Path(args.audit_output)
    _ensure_outputs_available([output, audit_output], overwrite=args.overwrite)
    observations, audit = load_naumann_calendar_observations(
        args.celljar_repository,
        args.cycle_summary,
    )
    _write_frozen_prediction_csv(observations, output)
    project_root = Path(__file__).resolve().parents[2]
    source_hashes = _source_tree_hashes(project_root)
    audit["provenance"] = {
        "source_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
    }
    audit["artifact"] = {
        "path": str(output),
        "row_count": len(observations),
        "sha256": _sha256_file(output),
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def _calendar_aging_backtest(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output = Path(args.output)
    predictions_output = Path(args.predictions_output)
    condition_metrics_output = Path(args.condition_metrics_output)
    diagnostics_output = Path(args.diagnostics_output)
    parameters_output = Path(args.parameters_output)
    splits_output = Path(args.splits_output)
    output_paths = [
        output,
        predictions_output,
        condition_metrics_output,
        diagnostics_output,
        parameters_output,
        splits_output,
    ]
    _ensure_outputs_available(output_paths, overwrite=args.overwrite)
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    observations = _read_table(input_path)
    dataset_ids = observations["dataset_id"].astype(str).unique().tolist()
    if len(dataset_ids) != 1:
        raise ValueError(f"Expected one calendar dataset id, found {dataset_ids}")
    _require_frozen_value("dataset_id", dataset_ids[0], config["dataset_id"])
    _require_frozen_value(
        "effective_independent_condition_count",
        int(observations["condition_id"].nunique()),
        int(config["effective_independent_condition_count"]),
    )
    if bool(config["allow_projection_beyond_observed_horizon"]):
        raise ValueError("Calendar backtest config cannot enable horizon extrapolation")
    maximum_horizon = float(observations["elapsed_days"].max())
    configured_horizon = float(config["maximum_supported_horizon_days"])
    if abs(maximum_horizon - configured_horizon) > 1e-6:
        raise ValueError(
            "Observed calendar horizon does not match the frozen config: "
            f"observed={maximum_horizon}, configured={configured_horizon}"
        )

    (
        result,
        predictions,
        condition_metrics,
        diagnostics,
        parameters,
        splits,
    ) = run_calendar_aging_backtest(
        observations,
        scenarios=config["scenarios"],
        prefix_checkups=config["prefix_checkups"],
        primary_prefix_checkups=int(config["primary_prefix_checkups"]),
        model_parameters=config["model"],
        adaptation_parameters=config["adaptation"],
        validation_thresholds=config["validation_thresholds"],
        gate_scenarios=config["gate_scenarios"],
    )
    _write_frozen_prediction_csv(predictions, predictions_output)
    _write_frozen_prediction_csv(condition_metrics, condition_metrics_output)
    _write_frozen_prediction_csv(diagnostics, diagnostics_output)
    _write_frozen_prediction_csv(parameters, parameters_output)
    _write_frozen_prediction_csv(splits, splits_output)
    result["experiment"] = _experiment_identity(config)
    result["provenance"] = _experiment_provenance(
        input_path=input_path,
        features=observations,
        config_path=config_path,
    )
    result["artifacts"] = {
        "label_free_predictions": {
            "path": str(predictions_output),
            "row_count": len(predictions),
            "sha256": _sha256_file(predictions_output),
        },
        "condition_metrics": {
            "path": str(condition_metrics_output),
            "row_count": len(condition_metrics),
            "sha256": _sha256_file(condition_metrics_output),
        },
        "fold_diagnostics": {
            "path": str(diagnostics_output),
            "row_count": len(diagnostics),
            "sha256": _sha256_file(diagnostics_output),
        },
        "fold_parameters": {
            "path": str(parameters_output),
            "row_count": len(parameters),
            "sha256": _sha256_file(parameters_output),
        },
        "condition_splits": {
            "path": str(splits_output),
            "row_count": len(splits),
            "sha256": _sha256_file(splits_output),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _attia_external_predict(args: argparse.Namespace) -> int:
    source_path = Path(args.source_features)
    target_path = Path(args.target_features)
    output = Path(args.output)
    predictions_output = Path(args.predictions_output)
    _ensure_outputs_available(
        [output, predictions_output],
        overwrite=args.overwrite,
    )
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    _validate_attia_external_config(config)
    source_feature_sha = _sha256_file(source_path)
    target_feature_sha = _sha256_file(target_path)
    if source_feature_sha != str(config["source_feature_pack_sha256"]):
        raise ValueError("Source feature-pack SHA does not match the frozen config")
    if target_feature_sha != str(config["target_feature_pack_sha256"]):
        raise ValueError("Target feature-pack SHA does not match the frozen config")
    source = _read_table(source_path)
    target = _read_table(target_path)
    if set(source["dataset_id"].astype(str)) != {config["source_dataset_id"]}:
        raise ValueError("Source feature dataset id does not match the frozen config")
    if set(target["dataset_id"].astype(str)) != {config["target_dataset_id"]}:
        raise ValueError("Target feature dataset id does not match the frozen config")
    predictions, result = build_source_only_external_predictions(
        source,
        target,
        feature_column=str(config["feature_column"]),
        label_column="cycle_life",
        l2_penalty=float(config["model"]["l2_penalty"]),
        expected_source_crosswalk_sha256=str(
            config["source_label_table_sha256"]
        ),
        expected_source_count=int(config["source_cell_count"]),
        expected_target_count=int(config["target_cell_count"]),
    )
    _write_frozen_prediction_csv(predictions, predictions_output)
    result["experiment"] = _attia_prediction_experiment_identity(config)
    result["provenance"] = _experiment_provenance(
        input_path=source_path,
        features=source,
        config_path=config_path,
    )
    result["provenance"]["source_features_sha256"] = source_feature_sha
    result["provenance"]["target_features_sha256"] = target_feature_sha
    result["artifact"] = {
        "path": str(predictions_output),
        "row_count": len(predictions),
        "sha256": _sha256_file(predictions_output),
        "canonical_prediction_sha256": external_prediction_artifact_sha256(
            _read_table(predictions_output)
        ),
    }
    if (
        result["artifact"]["canonical_prediction_sha256"]
        != result["frozen_prediction_sha256"]
    ):
        raise RuntimeError("Written external predictions do not match the frozen hash")
    written_predictions = _read_table(predictions_output)
    target_feature_identity = external_target_feature_identity_sha256(
        written_predictions,
        feature_column=str(config["feature_column"]),
    )
    if target_feature_identity != result["target_feature_identity_sha256"]:
        raise RuntimeError("Written target feature identity is not stable")
    result["source_feature_pack"] = {
        "dataset_id": config["source_dataset_id"],
        "dataset_snapshot_id": config["source_dataset_snapshot_id"],
        "row_count": len(source),
        "sha256": source_feature_sha,
        "authoritative_crosswalk_sha256": config["source_label_table_sha256"],
    }
    result["target_feature_pack"] = {
        "dataset_id": config["target_dataset_id"],
        "dataset_snapshot_id": config["target_dataset_snapshot_id"],
        "feature_column": config["feature_column"],
        "row_count": len(target),
        "sha256": target_feature_sha,
        "identity_sha256": target_feature_identity,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _prepare_attia_target_features(args: argparse.Namespace) -> int:
    timeseries_path = Path(args.timeseries)
    metadata_output = Path(args.metadata_output)
    curves_output = Path(args.curves_output)
    features_output = Path(args.features_output)
    audit_output = Path(args.audit_output)
    _ensure_outputs_available(
        [metadata_output, curves_output, features_output, audit_output],
        overwrite=args.overwrite,
    )
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    _validate_attia_external_config(config)
    timeseries_hash = _sha256_file(timeseries_path)
    if timeseries_hash != str(config["target_timeseries_sha256"]):
        raise ValueError(
            "CellJAR time-series SHA does not match the frozen external config"
        )
    metadata, metadata_audit = load_attia_label_free_metadata(
        args.celljar_repository
    )
    _write_frozen_prediction_csv(metadata, metadata_output)
    curve_audit = extract_celljar_curve_subset(
        timeseries_path,
        metadata,
        curves_output,
        cycles=[10, 100],
        overwrite=args.overwrite,
    )
    if not curve_audit["complete"]:
        raise ValueError("Attia target curve extraction is incomplete")
    curves = _read_table(curves_output)
    features = extract_external_delta_q_features(
        curves,
        metadata,
        dataset_id=ATTIA_DATASET_ID,
        minimum_samples_per_curve=int(
            config["voltage_grid"]["minimum_curve_samples"]
        ),
    )
    if len(features) != int(config["target_cell_count"]):
        raise ValueError("Unexpected Attia target feature count")
    _write_frozen_prediction_csv(features, features_output)
    feature_pack_sha = _sha256_file(features_output)
    if feature_pack_sha != str(config["target_feature_pack_sha256"]):
        raise ValueError(
            "Prepared Attia target feature pack does not match frozen config SHA"
        )
    project_root = Path(__file__).resolve().parents[2]
    source_hashes = _source_tree_hashes(project_root)
    audit: dict[str, object] = {
        "status": "passed",
        "dataset_id": ATTIA_DATASET_ID,
        "dataset_snapshot_id": config["target_dataset_snapshot_id"],
        "evidence_role": "label_free_external_target_feature_preparation",
        "metadata": metadata_audit,
        "curve_extraction": curve_audit,
        "feature_gate": {
            "cell_count": len(features),
            "protocol_count": int(features["protocol_id"].nunique()),
            "expected_curve_pairs": 90,
            "observed_curve_pairs": int(
                curve_audit["observed_cell_cycle_pairs"]
            ),
            "early_cycle": 10,
            "late_cycle": 100,
            "voltage_min_v": 2.0,
            "voltage_max_v": 3.5,
            "voltage_points": 1000,
            "outcome_columns": [],
        },
        "source": {
            "timeseries_path": str(timeseries_path.resolve()),
            "timeseries_sha256": timeseries_hash,
            "celljar_commit": config["celljar_commit"],
        },
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "source_sha256": source_hashes,
            "source_tree_sha256": _source_tree_sha256(source_hashes),
        },
        "artifacts": {
            "label_free_metadata": {
                "path": str(metadata_output),
                "row_count": len(metadata),
                "sha256": _sha256_file(metadata_output),
            },
            "curve_subset": {
                "path": str(curves_output),
                "row_count": int(curve_audit["row_count"]),
                "sha256": _sha256_file(curves_output),
            },
            "label_free_features": {
                "path": str(features_output),
                "row_count": len(features),
                "sha256": feature_pack_sha,
            },
        },
        "warning": (
            "Target outcomes were not emitted or joined. Authoritative Attia labels "
            "belong to the separate post-prediction outcome command."
        ),
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def _prepare_attia_outcomes(args: argparse.Namespace) -> int:
    output = Path(args.output)
    audit_output = Path(args.audit_output)
    _ensure_outputs_available([output, audit_output], overwrite=args.overwrite)
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    _validate_attia_external_config(config)
    outcomes, audit = build_attia_outcome_pack(
        args.celljar_repository,
        args.author_final_results,
    )
    for column, expected in _attia_outcome_identity(config).items():
        if set(outcomes[column].tolist()) != {expected}:
            raise ValueError(f"Attia outcome identity mismatch for {column}")
    canonical_outcome_sha = attia_outcome_artifact_sha256(outcomes)
    if canonical_outcome_sha != str(config["target_outcome_pack_sha256"]):
        raise ValueError("Attia outcome content does not match frozen config SHA")
    _write_frozen_prediction_csv(outcomes, output)
    written_canonical_outcome_sha = attia_outcome_artifact_sha256(
        _read_table(output)
    )
    if written_canonical_outcome_sha != canonical_outcome_sha:
        raise RuntimeError("Written Attia outcome content hash is not stable")
    project_root = Path(__file__).resolve().parents[2]
    source_hashes = _source_tree_hashes(project_root)
    audit["provenance"] = {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "source_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
    }
    audit["artifact"] = {
        "path": str(output),
        "row_count": len(outcomes),
        "sha256": _sha256_file(output),
        "canonical_outcome_sha256": canonical_outcome_sha,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def _attia_external_score(args: argparse.Namespace) -> int:
    predictions_path = Path(args.predictions)
    outcomes_path = Path(args.outcomes)
    prediction_manifest_path = Path(args.prediction_manifest)
    output = Path(args.output)
    cell_metrics_output = Path(args.cell_metrics_output)
    protocol_metrics_output = Path(args.protocol_metrics_output)
    _ensure_outputs_available(
        [output, cell_metrics_output, protocol_metrics_output],
        overwrite=args.overwrite,
    )
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    _validate_attia_external_config(config)
    config_sha = _sha256_file(config_path)
    predictions = _read_table(predictions_path)
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    predictions_sha = _sha256_file(predictions_path)
    frozen_hash = _validate_attia_prediction_manifest(
        prediction_manifest,
        config=config,
        config_sha256=config_sha,
        predictions=predictions,
        predictions_sha256=predictions_sha,
    )
    outcomes = _read_table(outcomes_path)
    validate_attia_outcome_pack(outcomes)
    canonical_outcome_sha = attia_outcome_artifact_sha256(outcomes)
    if canonical_outcome_sha != str(config["target_outcome_pack_sha256"]):
        raise ValueError("Attia outcome content does not match frozen config SHA")
    outcome_identity = _attia_outcome_identity(config)
    result, cell_metrics, protocol_metrics = (
        score_source_only_external_predictions(
            predictions,
            outcomes,
            frozen_prediction_sha256=frozen_hash,
            expected_outcome_sha256=canonical_outcome_sha,
            expected_outcome_identity=outcome_identity,
            validation_thresholds=config["validation_thresholds"],
            bootstrap_resamples=int(config["bootstrap"]["resamples"]),
            bootstrap_seed=int(config["bootstrap"]["seed"]),
        )
    )
    _write_frozen_prediction_csv(cell_metrics, cell_metrics_output)
    _write_frozen_prediction_csv(protocol_metrics, protocol_metrics_output)
    project_root = Path(__file__).resolve().parents[2]
    source_hashes = _source_tree_hashes(project_root)
    result["experiment"] = {
        key: config[key]
        for key in (
            "experiment_id",
            "source_dataset_id",
            "source_dataset_snapshot_id",
            "target_dataset_id",
            "target_dataset_snapshot_id",
            "target_label_version",
            "target_label_sha256",
            "evidence_role",
        )
    }
    result["provenance"] = {
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "predictions_sha256": predictions_sha,
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "outcomes_sha256": _sha256_file(outcomes_path),
        "canonical_outcome_sha256": canonical_outcome_sha,
        "source_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
    }
    result["artifacts"] = {
        "cell_metric_rows": {
            "path": str(cell_metrics_output),
            "row_count": len(cell_metrics),
            "sha256": _sha256_file(cell_metrics_output),
        },
        "protocol_metrics": {
            "path": str(protocol_metrics_output),
            "row_count": len(protocol_metrics),
            "sha256": _sha256_file(protocol_metrics_output),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _prepare_beep_fastcharge(args: argparse.Namespace) -> int:
    cycle_output = Path(args.cycle_output)
    inventory_output = Path(args.inventory_output)
    audit_output = Path(args.audit_output)
    _ensure_outputs_available(
        [cycle_output, inventory_output, audit_output],
        overwrite=args.overwrite,
    )
    cycles, inventory, audit = prepare_fastcharge_frames(
        args.source_directory,
        hash_sources=not args.skip_source_hash,
        observation_cycle=args.observation_cycle,
    )
    _write_table(cycles, cycle_output)
    _write_table(inventory, inventory_output)
    audit["artifacts"] = {
        "cycle_summary": {
            "path": str(cycle_output),
            "row_count": len(cycles),
            "sha256": _sha256_file(cycle_output),
        },
        "source_inventory": {
            "path": str(inventory_output),
            "row_count": len(inventory),
            "sha256": _sha256_file(inventory_output),
        },
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def _metadata_probe(args: argparse.Namespace) -> int:
    metadata = _read_table(Path(args.input))
    numeric = [
        "nominal_capacity_ah",
        "c_rate_charge",
        "c_rate_discharge",
        "temperature_c",
    ]
    experiments: dict[str, object] = {
        "warning": (
            "Protocol-only leakage probe; this is not an early-health prediction result."
        )
    }
    for split_column in ("split_cell", "split_protocol"):
        if split_column not in metadata:
            raise ValueError(f"Missing split column: {split_column}")
        experiment_frame = metadata.assign(split=metadata[split_column])
        experiments[split_column] = {
            "protocol_blind": run_baselines(
                experiment_frame,
                feature_columns=numeric,
                categorical_columns=["batch_id"],
            ),
            "protocol_visible": run_baselines(
                experiment_frame,
                feature_columns=numeric,
                categorical_columns=["batch_id", "protocol_id"],
            ),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(experiments, indent=2), encoding="utf-8")
    print(json.dumps(experiments, ensure_ascii=False, indent=2))
    return 0


def _extract_curve_features(args: argparse.Namespace) -> int:
    curves = _read_table(Path(args.curves))
    metadata = _read_table(Path(args.metadata))
    features = extract_delta_q_features(
        curves,
        metadata,
        early_cycle=args.early_cycle,
        late_cycle=args.late_cycle,
    )
    output = Path(args.output)
    _write_table(features, output)
    report = {
        "cell_count": len(features),
        "early_cycle": args.early_cycle,
        "late_cycle": args.late_cycle,
        "failed_curve_pairs": features.attrs.get("failed_curve_pairs", []),
        "output_sha256": _sha256_file(output),
        "warning": "Feature extraction result only; model metrics require frozen splits.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _extract_celljar_curves(args: argparse.Namespace) -> int:
    metadata = _read_table(Path(args.metadata))
    report = extract_celljar_curve_subset(
        args.source,
        metadata,
        args.output,
        cycles=args.cycles,
        overwrite=args.overwrite,
    )
    report["output_sha256"] = _sha256_file(Path(args.output))
    audit_output = Path(args.audit_output) if args.audit_output else Path(
        args.output
    ).with_suffix(".audit.json")
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 2


def _curve_baseline(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    features = _read_table(input_path)
    feature_columns = list(MATR_CURVE_FEATURE_COLUMNS)
    cohort_complete = len(features) == 124
    result: dict[str, object] = {
        "warning": (
            "Public MATR benchmark only; it does not establish accuracy on Hithium "
            "large-format storage cells."
            if cohort_complete
            else "Preliminary early-curve baseline; interpret only when the full frozen cohort is present."
        ),
        "cell_count": len(features),
        "cohort_complete": cohort_complete,
        "input_sha256": _sha256_file(input_path),
        "model_specification": {
            "variance_log_linear": {
                "feature": "log10_delta_q_variance",
                "target": "log10_cycle_life",
                "estimator": "ordinary_least_squares",
                "variance_ddof": 0,
            },
            "multifeature_elastic_net": {
                "features": feature_columns,
                "target": "log10_cycle_life",
                "alpha": 0.02,
                "l1_ratio": 0.5,
            },
            "bootstrap": {
                "method": "frozen-model grouped percentile",
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
            },
        },
    }

    def evaluate_models(
        experiment: pd.DataFrame,
        *,
        bootstrap_group_column: str,
    ) -> dict[str, object]:
        variance = run_baselines(
            experiment,
            feature_columns=["log10_delta_q_variance"],
            target_transform="log10",
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_group_column=bootstrap_group_column,
            bootstrap_random_state=args.bootstrap_seed,
        )
        multifeature = run_baselines(
            experiment,
            feature_columns=feature_columns,
            target_transform="log10",
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_group_column=bootstrap_group_column,
            bootstrap_random_state=args.bootstrap_seed,
        )
        return {
            "median": multifeature["median"],
            "variance_log_linear": variance["linear_regression"],
            "multifeature_elastic_net": multifeature["elastic_net"],
        }

    for split_column in ("split_cell", "split_protocol"):
        if split_column not in features:
            raise ValueError(f"Missing split column: {split_column}")
        group_column = "cell_id" if split_column == "split_cell" else "protocol_id"
        result[split_column] = evaluate_models(
            features.assign(split=features[split_column]),
            bootstrap_group_column=group_column,
        )
    result["split_summary"] = {
        "split_cell": {
            "rows": features["split_cell"].value_counts().sort_index().to_dict(),
            "groups": features.groupby("split_cell")["cell_id"].nunique().to_dict(),
        },
        "split_protocol": {
            "rows": features["split_protocol"].value_counts().sort_index().to_dict(),
            "groups": features.groupby("split_protocol")["protocol_id"].nunique().to_dict(),
        },
    }
    if "paper_split" in features:
        result["split_summary"]["paper_split"] = {
            "rows": features["paper_split"].value_counts().sort_index().to_dict(),
            "groups": features.groupby("paper_split")["cell_id"].nunique().to_dict(),
        }
        for name, held_out in (
            ("paper_primary_test", "primary_test"),
            ("paper_secondary_test", "secondary_test"),
        ):
            paper = features.loc[
                features["paper_split"].isin(["train", held_out])
            ].copy()
            paper["split"] = paper["paper_split"].replace({held_out: "test"})
            if set(paper["split"]) == {"train", "test"}:
                result[name] = evaluate_models(
                    paper,
                    bootstrap_group_column="cell_id",
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _probabilistic_experiment_frames(
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    experiment_frames: dict[str, pd.DataFrame] = {}
    for split_column in ("split_cell", "split_protocol"):
        if split_column not in features:
            raise ValueError(f"Missing split column: {split_column}")
        experiment_frames[split_column] = features.assign(split=features[split_column])
    if "paper_split" in features:
        for name, held_out in (
            ("paper_primary_test", "primary_test"),
            ("paper_secondary_test", "secondary_test"),
        ):
            paper = features.loc[
                features["paper_split"].isin(["train", held_out])
            ].copy()
            paper["split"] = paper["paper_split"].replace({held_out: "test"})
            experiment_frames[name] = paper
    return experiment_frames


def _prepare_censoring_column(
    features: pd.DataFrame,
    *,
    assume_all_observed: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "is_censored" in features:
        return features, {
            "column_status": "present_in_input",
            "assumption_applied": False,
        }
    if not assume_all_observed:
        raise ValueError(
            "Missing required is_censored column. Pass --assume-all-observed only "
            "when a dataset audit confirms every row reached its EOL event."
        )
    result = features.copy()
    result["is_censored"] = False
    return result, {
        "column_status": "missing_in_input",
        "assumption_applied": True,
        "assumption": "all rows are observed EOL events",
        "activation": "explicit --assume-all-observed CLI flag",
    }


def _probabilistic_outer_policy(
    experiment_name: str,
) -> tuple[tuple[str, ...], str | None]:
    if experiment_name == "split_protocol":
        return ("cell_id", "protocol_id"), "protocol_id"
    return ("cell_id",), None


def _calibration_gate(
    result: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    gate = config["calibration_gate"]
    primary_split = str(config["primary_split"])
    primary_model = str(config["primary_model"])
    minimum_groups = int(gate["minimum_test_groups"])
    maximum_error = float(gate["maximum_absolute_group_coverage_error"])
    experiment = result["experiments"][primary_split][primary_model]
    conformal = experiment["conformal"]
    reasons: list[str] = []
    group_evaluation = conformal.get("test_simultaneous_group_coverage")
    if conformal.get("status") != "available" or group_evaluation is None:
        reasons.append("primary group-level conformal evaluation is unavailable")
        group_count = 0
        coverage_error = None
    else:
        group_count = int(group_evaluation["group_count"])
        coverage_error = float(group_evaluation["coverage_error"])
        if group_count < minimum_groups:
            reasons.append(
                f"only {group_count} test groups; at least {minimum_groups} required"
            )
        if abs(coverage_error) > maximum_error:
            reasons.append(
                f"absolute coverage error {abs(coverage_error):.4f} exceeds "
                f"{maximum_error:.4f}"
            )
    return {
        "name": gate["name"],
        "status": "passed" if not reasons else "failed",
        "primary_split": primary_split,
        "primary_model": primary_model,
        "estimand": "simultaneous observed-cell coverage for a new protocol",
        "minimum_test_groups": minimum_groups,
        "maximum_absolute_group_coverage_error": maximum_error,
        "observed_test_group_count": group_count,
        "observed_group_coverage_error": coverage_error,
        "reasons": reasons,
        "note": (
            "This audit safety gate was frozen after the initial Phase 1 diagnostic; "
            "it is not a prospective claim for the previously viewed MATR test set."
        ),
    }


def _probabilistic_baseline(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    configured_coverage = float(config["conformal"]["target_coverage"])
    configured_l2 = float(config["model"]["l2_penalty"])
    _require_frozen_value("--conformal-coverage", args.conformal_coverage, configured_coverage)
    _require_frozen_value("--l2-penalty", args.l2_penalty, configured_l2)
    output = Path(args.output)
    predictions_output = Path(args.predictions_output)
    _ensure_outputs_available(
        [output, predictions_output],
        overwrite=args.overwrite,
    )
    features = _read_table(input_path)
    features, censoring_source = _prepare_censoring_column(
        features,
        assume_all_observed=args.assume_all_observed,
    )
    feature_sets = {
        name: list(columns) for name, columns in config["feature_sets"].items()
    }
    experiment_frames = _probabilistic_experiment_frames(features)

    result: dict[str, object] = {
        "experiment": _experiment_identity(config),
        "warning": (
            "Public MATR cycle-life probability baseline only. It does not validate "
            "15-25 year calendar aging or Hithium product accuracy."
        ),
        "input_sha256": _sha256_file(input_path),
        "provenance": _experiment_provenance(
            input_path=input_path,
            features=features,
            config_path=config_path,
        ),
        "cell_count": len(features),
        "right_censored_count": int(features["is_censored"].sum()),
        "censoring_source": censoring_source,
        "quantiles": [0.1, 0.5, 0.9],
        "conformal_coverage": configured_coverage,
        "model_specification": {
            "distribution": "log_normal_accelerated_failure_time",
            "right_censoring": "included_in_training_and_test_likelihood",
            "uncertainty_scope": (
                "log-normal residual dispersion only; coefficient, reference-cell, "
                "and domain-shift uncertainty are not yet included"
            ),
            "l2_penalty": configured_l2,
            "feature_sets": feature_sets,
        },
        "experiments": {},
    }
    prediction_frames: list[pd.DataFrame] = []
    for experiment_name, experiment_frame in experiment_frames.items():
        experiment_result: dict[str, object] = {}
        for model_name, model_features in feature_sets.items():
            isolation_columns, conformal_group_column = _probabilistic_outer_policy(
                experiment_name
            )
            evaluation, predictions = run_probabilistic_lifetime_experiment(
                experiment_frame,
                feature_columns=model_features,
                conformal_coverage=configured_coverage,
                l2_penalty=configured_l2,
                group_isolation_columns=isolation_columns,
                conformal_group_column=conformal_group_column,
            )
            experiment_result[model_name] = evaluation
            predictions.insert(0, "model", model_name)
            predictions.insert(0, "experiment", experiment_name)
            prediction_frames.append(predictions)
        result["experiments"][experiment_name] = experiment_result

    result["calibration_gate"] = _calibration_gate(result, config)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    _write_table(predictions, predictions_output)
    result["predictions"] = {
        "path": str(predictions_output),
        "row_count": len(predictions),
        "sha256": _sha256_file(predictions_output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _probabilistic_tuned_baseline(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    selection_config = config["model_selection"]
    configured_coverage = float(config["conformal"]["target_coverage"])
    configured_l2_grid = [
        float(value) for value in selection_config["candidate_l2_penalties"]
    ]
    configured_group = str(selection_config["group_column"])
    configured_folds = int(selection_config["fold_count"])
    configured_seed = int(selection_config["seed"])
    _require_frozen_value("--conformal-coverage", args.conformal_coverage, configured_coverage)
    _require_frozen_value("--l2-grid", args.l2_grid, configured_l2_grid)
    _require_frozen_value("--inner-group-column", args.inner_group_column, configured_group)
    _require_frozen_value("--inner-cv-folds", args.inner_cv_folds, configured_folds)
    _require_frozen_value("--inner-cv-seed", args.inner_cv_seed, configured_seed)
    output = Path(args.output)
    predictions_output = Path(args.predictions_output)
    _ensure_outputs_available(
        [output, predictions_output],
        overwrite=args.overwrite,
    )
    features = _read_table(input_path)
    features, censoring_source = _prepare_censoring_column(
        features,
        assume_all_observed=args.assume_all_observed,
    )
    feature_sets = {
        name: list(columns) for name, columns in config["feature_sets"].items()
    }
    experiment_frames = _probabilistic_experiment_frames(features)
    result: dict[str, object] = {
        "experiment": _experiment_identity(config),
        "warning": (
            "Public MATR nested-CV probability baseline only. It does not validate "
            "15-25 year calendar aging or Hithium product accuracy."
        ),
        "input_sha256": _sha256_file(input_path),
        "provenance": _experiment_provenance(
            input_path=input_path,
            features=features,
            config_path=config_path,
        ),
        "cell_count": len(features),
        "right_censored_count": int(features["is_censored"].sum()),
        "censoring_source": censoring_source,
        "quantiles": [0.1, 0.5, 0.9],
        "conformal_coverage": configured_coverage,
        "model_specification": {
            "distribution": "log_normal_accelerated_failure_time",
            "right_censoring": "included_in_training_and_test_likelihood",
            "uncertainty_scope": (
                "log-normal residual dispersion plus validation-set conformal radius; "
                "coefficient, reference-cell, and domain-shift uncertainty remain excluded"
            ),
            "model_selection": {
                "method": "nested_group_cross_validation",
                "group_column": configured_group,
                "fold_count": configured_folds,
                "seed": configured_seed,
                "candidate_l2_penalties": configured_l2_grid,
                "score": "protocol-balanced mean validation negative log-likelihood",
                "outer_validation_use": "conformal calibration only",
                "outer_test_use": "final evaluation only",
            },
            "feature_sets": feature_sets,
        },
        "experiments": {},
    }
    prediction_frames: list[pd.DataFrame] = []
    for experiment_name, experiment_frame in experiment_frames.items():
        experiment_result: dict[str, object] = {}
        for model_name, model_features in feature_sets.items():
            isolation_columns, conformal_group_column = _probabilistic_outer_policy(
                experiment_name
            )
            evaluation, predictions = run_tuned_probabilistic_lifetime_experiment(
                experiment_frame,
                feature_columns=model_features,
                l2_candidates=configured_l2_grid,
                group_column=configured_group,
                conformal_coverage=configured_coverage,
                inner_cv_folds=configured_folds,
                inner_cv_seed=configured_seed,
                group_isolation_columns=isolation_columns,
                conformal_group_column=conformal_group_column,
            )
            experiment_result[model_name] = evaluation
            predictions.insert(
                0,
                "selected_l2_penalty",
                evaluation["model_selection"]["selected_l2_penalty"],
            )
            predictions.insert(0, "model", model_name)
            predictions.insert(0, "experiment", experiment_name)
            prediction_frames.append(predictions)
        result["experiments"][experiment_name] = experiment_result

    result["calibration_gate"] = _calibration_gate(result, config)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    _write_table(predictions, predictions_output)
    result["predictions"] = {
        "path": str(predictions_output),
        "row_count": len(predictions),
        "sha256": _sha256_file(predictions_output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _reference_cell_feasibility(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output = Path(args.output)
    _ensure_outputs_available([output], overwrite=args.overwrite)
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    features = _read_table(input_path)
    k_values = [int(value) for value in config["reference_cell_counts"]]
    minimum_queries = int(config["minimum_query_cells_per_domain"])

    strict_config = config["strict_protocol_scope"]
    split_column = str(strict_config["split_column"])
    target_split = str(strict_config["target_split"])
    protocol_column = str(strict_config["domain_column"])
    required = {"cell_id", "batch_id", protocol_column, split_column, "paper_split"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Missing reference-cell feasibility columns: {missing}")
    strict_target = features.loc[features[split_column] == target_split]
    strict_validation = features.loc[features[split_column] == "validation"]

    strict_audit = reference_cell_support(
        strict_target,
        domain_column=protocol_column,
        k_values=k_values,
        minimum_query_cells_per_domain=minimum_queries,
    )
    validation_audit = reference_cell_support(
        strict_validation,
        domain_column=protocol_column,
        k_values=k_values,
        minimum_query_cells_per_domain=minimum_queries,
    )
    all_protocol_audit = reference_cell_support(
        features,
        domain_column=protocol_column,
        k_values=k_values,
        minimum_query_cells_per_domain=minimum_queries,
    )
    batch_config = config["batch_proxy_scope"]
    batch_column = str(batch_config["domain_column"])
    batch_audit = reference_cell_support(
        features,
        domain_column=batch_column,
        k_values=k_values,
        minimum_query_cells_per_domain=minimum_queries,
    )
    paper_secondary = features.loc[features["paper_split"] == "secondary_test"]
    secondary_audit = reference_cell_support(
        paper_secondary,
        domain_column=batch_column,
        k_values=k_values,
        minimum_query_cells_per_domain=minimum_queries,
    )

    result: dict[str, object] = {
        "experiment": _experiment_identity(config),
        "warning": (
            "Feasibility audit only. No reference-cell model is fitted and no "
            "storage-product accuracy claim is supported."
        ),
        "provenance": _experiment_provenance(
            input_path=input_path,
            features=features,
            config_path=config_path,
        ),
        "reference_cell_counts": k_values,
        "minimum_query_cells_per_domain": minimum_queries,
        "scopes": {
            "strict_protocol_test": strict_audit,
            "strict_protocol_validation": validation_audit,
            "all_protocols_exploratory": all_protocol_audit,
            "leave_one_batch_out_proxy": batch_audit,
            "paper_secondary_batch_proxy": secondary_audit,
        },
        "gates": {
            "strict_protocol": support_gate(
                strict_audit,
                minimum_eligible_domains=int(
                    strict_config["minimum_eligible_domains"]
                ),
            ),
            "batch_proxy": support_gate(
                batch_audit,
                minimum_eligible_domains=int(
                    batch_config["minimum_eligible_domains"]
                ),
            ),
        },
        "decision": (
            "Do not run a k=1/3/5/10 same-protocol benchmark on the frozen MATR "
            "test set unless every requested k has enough independent target "
            "protocols and at least one untouched query cell per protocol."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _batch_reference_proxy(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output = Path(args.output)
    predictions_output = Path(args.predictions_output)
    selections_output = Path(args.selections_output)
    metrics_output = Path(args.metrics_output)
    _ensure_outputs_available(
        [output, predictions_output, selections_output, metrics_output],
        overwrite=args.overwrite,
    )
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    features = _read_table(input_path)
    features, censoring_source = _prepare_censoring_column(
        features,
        assume_all_observed=args.assume_all_observed,
    )
    result, predictions, selections, metrics = run_batch_reference_proxy_experiment(
        features,
        feature_columns=list(config["feature_columns"]),
        k_values=[int(value) for value in config["reference_cell_counts"]],
        repeats=int(config["repeats"]),
        seed=int(config["seed"]),
        l2_penalty=float(config["l2_penalty"]),
        prior_scale_multiplier=float(config["prior"]["scale_multiplier"]),
        survival_times=[float(value) for value in config["survival_times"]],
        batch_column=str(config["target_domain"]),
    )
    result["experiment"] = _experiment_identity(config)
    result["warning"] = (
        "MATR batch-proxy software experiment only. Three target batches cannot "
        "establish reference-cell efficacy or storage-product accuracy."
    )
    result["censoring_source"] = censoring_source
    result["provenance"] = _experiment_provenance(
        input_path=input_path,
        features=features,
        config_path=config_path,
    )
    _write_frozen_prediction_csv(predictions, predictions_output)
    prediction_file_hash = _sha256_file(predictions_output)
    if prediction_file_hash != result["prediction_freeze"]["sha256"]:
        raise RuntimeError(
            "Written prediction artifact does not match the pre-score freeze hash"
        )
    _write_table(selections, selections_output)
    _write_table(metrics, metrics_output)
    result["artifacts"] = {
        "predictions": {
            "path": str(predictions_output),
            "row_count": len(predictions),
            "sha256": prediction_file_hash,
        },
        "reference_selections": {
            "path": str(selections_output),
            "row_count": len(selections),
            "sha256": _sha256_file(selections_output),
        },
        "repeat_metrics": {
            "path": str(metrics_output),
            "row_count": len(metrics),
            "sha256": _sha256_file(metrics_output),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _synthetic_ipcw_validation_command(args: argparse.Namespace) -> int:
    output = Path(args.output)
    details_output = Path(args.details_output)
    _ensure_outputs_available([output, details_output], overwrite=args.overwrite)
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    result, details = run_synthetic_ipcw_validation(
        scenarios=config["scenarios"],
        repetitions=int(config["repetitions"]),
        seed=int(config["seed"]),
        counts=config["counts"],
        distribution=config["distribution"],
        evaluation_times=[float(value) for value in config["evaluation_times"]],
        l2_penalty=float(config["l2_penalty"]),
        policy_parameters=config["policy"],
        validation_thresholds=config["validation_thresholds"],
    )
    result["experiment"] = _experiment_identity(config)
    result["warning"] = (
        "Synthetic validation checks IPCW software behavior only. It provides no "
        "evidence for storage-cell lifetime accuracy or 15-25 year extrapolation."
    )
    result["provenance"] = _synthetic_ipcw_provenance(config_path)
    _write_table(details, details_output)
    result["artifacts"] = {
        "time_point_details": {
            "path": str(details_output),
            "row_count": len(details),
            "sha256": _sha256_file(details_output),
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _batch_reference_sensitivity(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output = Path(args.output)
    details_output = Path(args.details_output)
    _ensure_outputs_available([output, details_output], overwrite=args.overwrite)
    config_path = Path(args.config)
    config = _load_experiment_config(config_path)
    features = _read_table(input_path)
    features, censoring_source = _prepare_censoring_column(
        features,
        assume_all_observed=args.assume_all_observed,
    )
    result, details = run_batch_reference_prior_scale_sensitivity(
        features,
        feature_columns=list(config["feature_columns"]),
        prior_scale_multipliers=[
            float(value) for value in config["prior_scale_multipliers"]
        ],
        primary_prior_scale_multiplier=float(
            config["primary_prior_scale_multiplier"]
        ),
        k_values=[int(value) for value in config["reference_cell_counts"]],
        repeats=int(config["repeats"]),
        seed=int(config["seed"]),
        l2_penalty=float(config["l2_penalty"]),
        survival_times=[float(value) for value in config["survival_times"]],
        batch_column=str(config["target_domain"]),
    )
    result["experiment"] = _experiment_identity(config)
    result["censoring_source"] = censoring_source
    result["provenance"] = _experiment_provenance(
        input_path=input_path,
        features=features,
        config_path=config_path,
    )
    _write_table(details, details_output)
    result["artifacts"] = {
        "sensitivity_details": {
            "path": str(details_output),
            "row_count": len(details),
            "sha256": _sha256_file(details_output),
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lifetwin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-cycles")
    validate_parser.add_argument("input")
    validate_parser.set_defaults(handler=_validate_cycles)

    feature_parser = subparsers.add_parser("extract-features")
    feature_parser.add_argument("cycles")
    feature_parser.add_argument("labels")
    feature_parser.add_argument("output")
    feature_parser.add_argument("--observation-cycle", type=int, default=100)
    feature_parser.add_argument("--minimum-observed-cycles", type=int, default=20)
    feature_parser.set_defaults(handler=_extract_features)

    smoke_parser = subparsers.add_parser("synthetic-smoke")
    smoke_parser.add_argument("--output", default="artifacts/synthetic-smoke.json")
    smoke_parser.set_defaults(handler=_synthetic_smoke)

    metadata_parser = subparsers.add_parser("prepare-matr-metadata")
    metadata_parser.add_argument("celljar_repository")
    metadata_parser.add_argument("output")
    metadata_parser.add_argument(
        "--authoritative-crosswalk",
        default="data/reference/severson_table9_cells.csv",
    )
    metadata_parser.add_argument(
        "--audit-output", default="artifacts/matr-metadata-audit.json"
    )
    metadata_parser.set_defaults(handler=_prepare_matr_metadata)

    naumann_parser = subparsers.add_parser("prepare-naumann-calendar")
    naumann_parser.add_argument("celljar_repository")
    naumann_parser.add_argument("cycle_summary")
    naumann_parser.add_argument(
        "output",
        nargs="?",
        default="data/interim/naumann_calendar_observations.csv",
    )
    naumann_parser.add_argument(
        "--audit-output",
        default="artifacts/naumann-calendar-data-audit.json",
    )
    naumann_parser.add_argument("--overwrite", action="store_true")
    naumann_parser.set_defaults(handler=_prepare_naumann_calendar)

    calendar_parser = subparsers.add_parser("calendar-aging-backtest")
    calendar_parser.add_argument(
        "input",
        nargs="?",
        default="data/interim/naumann_calendar_observations.csv",
    )
    calendar_parser.add_argument(
        "--config",
        default="configs/experiments/naumann_calendar_prefix_backtest.json",
    )
    calendar_parser.add_argument(
        "--output",
        default="artifacts/naumann-calendar-prefix-backtest.json",
    )
    calendar_parser.add_argument(
        "--predictions-output",
        default="artifacts/naumann-calendar-label-free-predictions.csv",
    )
    calendar_parser.add_argument(
        "--condition-metrics-output",
        default="artifacts/naumann-calendar-condition-metrics.csv",
    )
    calendar_parser.add_argument(
        "--diagnostics-output",
        default="artifacts/naumann-calendar-fold-diagnostics.csv",
    )
    calendar_parser.add_argument(
        "--parameters-output",
        default="artifacts/naumann-calendar-fold-parameters.csv",
    )
    calendar_parser.add_argument(
        "--splits-output",
        default="artifacts/naumann-calendar-condition-splits.csv",
    )
    calendar_parser.add_argument("--overwrite", action="store_true")
    calendar_parser.set_defaults(handler=_calendar_aging_backtest)

    attia_features_parser = subparsers.add_parser("prepare-attia-target-features")
    attia_features_parser.add_argument("celljar_repository")
    attia_features_parser.add_argument("timeseries")
    attia_features_parser.add_argument(
        "--config",
        default="configs/experiments/attia_clo_external_validation.json",
    )
    attia_features_parser.add_argument(
        "--metadata-output",
        default="data/interim/attia_target_label_free_metadata.csv",
    )
    attia_features_parser.add_argument(
        "--curves-output",
        default="data/raw/celljar/attia_early_discharge_curves.parquet",
    )
    attia_features_parser.add_argument(
        "--features-output",
        default="data/processed/attia_delta_q_features_label_free.csv",
    )
    attia_features_parser.add_argument(
        "--audit-output",
        default="artifacts/attia-target-feature-audit.json",
    )
    attia_features_parser.add_argument("--overwrite", action="store_true")
    attia_features_parser.set_defaults(handler=_prepare_attia_target_features)

    attia_outcome_parser = subparsers.add_parser("prepare-attia-outcomes")
    attia_outcome_parser.add_argument("celljar_repository")
    attia_outcome_parser.add_argument(
        "author_final_results",
        nargs="?",
        default="data/reference/attia_final_results.csv",
    )
    attia_outcome_parser.add_argument(
        "--config",
        default="configs/experiments/attia_clo_external_validation.json",
    )
    attia_outcome_parser.add_argument(
        "--output",
        default="data/reference/attia_validation45_outcomes.csv",
    )
    attia_outcome_parser.add_argument(
        "--audit-output",
        default="artifacts/attia-outcome-crosswalk-audit.json",
    )
    attia_outcome_parser.add_argument("--overwrite", action="store_true")
    attia_outcome_parser.set_defaults(handler=_prepare_attia_outcomes)

    attia_predict_parser = subparsers.add_parser("attia-external-predict")
    attia_predict_parser.add_argument("source_features")
    attia_predict_parser.add_argument("target_features")
    attia_predict_parser.add_argument(
        "--config",
        default="configs/experiments/attia_clo_external_validation.json",
    )
    attia_predict_parser.add_argument(
        "--output",
        default="artifacts/attia-external-prediction-manifest.json",
    )
    attia_predict_parser.add_argument(
        "--predictions-output",
        default="artifacts/attia-external-label-free-predictions.csv",
    )
    attia_predict_parser.add_argument("--overwrite", action="store_true")
    attia_predict_parser.set_defaults(handler=_attia_external_predict)

    attia_score_parser = subparsers.add_parser("attia-external-score")
    attia_score_parser.add_argument("predictions")
    attia_score_parser.add_argument("outcomes")
    attia_score_parser.add_argument(
        "--prediction-manifest",
        default="artifacts/attia-external-prediction-manifest.json",
    )
    attia_score_parser.add_argument(
        "--config",
        default="configs/experiments/attia_clo_external_validation.json",
    )
    attia_score_parser.add_argument(
        "--output",
        default="artifacts/attia-external-validation.json",
    )
    attia_score_parser.add_argument(
        "--cell-metrics-output",
        default="artifacts/attia-external-cell-metrics.csv",
    )
    attia_score_parser.add_argument(
        "--protocol-metrics-output",
        default="artifacts/attia-external-protocol-metrics.csv",
    )
    attia_score_parser.add_argument("--overwrite", action="store_true")
    attia_score_parser.set_defaults(handler=_attia_external_score)

    beep_parser = subparsers.add_parser("prepare-beep-fastcharge")
    beep_parser.add_argument("source_directory")
    beep_parser.add_argument(
        "--cycle-output",
        default="data/processed/fastcharge_cycle_summary.parquet",
    )
    beep_parser.add_argument(
        "--inventory-output",
        default="data/interim/fastcharge_source_inventory.csv",
    )
    beep_parser.add_argument(
        "--audit-output",
        default="artifacts/fastcharge-ingest-audit.json",
    )
    beep_parser.add_argument("--observation-cycle", type=int, default=100)
    beep_parser.add_argument("--skip-source-hash", action="store_true")
    beep_parser.add_argument("--overwrite", action="store_true")
    beep_parser.set_defaults(handler=_prepare_beep_fastcharge)

    probe_parser = subparsers.add_parser("metadata-probe")
    probe_parser.add_argument("input")
    probe_parser.add_argument("--output", default="artifacts/matr-metadata-probe.json")
    probe_parser.set_defaults(handler=_metadata_probe)

    curve_feature_parser = subparsers.add_parser("extract-curve-features")
    curve_feature_parser.add_argument("curves")
    curve_feature_parser.add_argument("metadata")
    curve_feature_parser.add_argument("output")
    curve_feature_parser.add_argument("--early-cycle", type=int, default=10)
    curve_feature_parser.add_argument("--late-cycle", type=int, default=100)
    curve_feature_parser.set_defaults(handler=_extract_curve_features)

    local_curve_parser = subparsers.add_parser("extract-celljar-curves")
    local_curve_parser.add_argument("source")
    local_curve_parser.add_argument("metadata")
    local_curve_parser.add_argument("output")
    local_curve_parser.add_argument("--cycles", type=int, nargs="+", default=[10, 100])
    local_curve_parser.add_argument("--audit-output")
    local_curve_parser.add_argument("--overwrite", action="store_true")
    local_curve_parser.set_defaults(handler=_extract_celljar_curves)

    curve_baseline_parser = subparsers.add_parser("curve-baseline")
    curve_baseline_parser.add_argument("input")
    curve_baseline_parser.add_argument(
        "--output", default="artifacts/matr-curve-baseline.json"
    )
    curve_baseline_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    curve_baseline_parser.add_argument("--bootstrap-seed", type=int, default=42)
    curve_baseline_parser.set_defaults(handler=_curve_baseline)

    probabilistic_parser = subparsers.add_parser("probabilistic-baseline")
    probabilistic_parser.add_argument("input")
    probabilistic_parser.add_argument(
        "--output", default="artifacts/matr-probabilistic-baseline.json"
    )
    probabilistic_parser.add_argument(
        "--predictions-output",
        default="artifacts/matr-probabilistic-predictions.csv",
    )
    probabilistic_parser.add_argument("--conformal-coverage", type=float, default=0.8)
    probabilistic_parser.add_argument("--l2-penalty", type=float, default=1e-4)
    probabilistic_parser.add_argument("--assume-all-observed", action="store_true")
    probabilistic_parser.add_argument(
        "--config", default="configs/experiments/matr_probabilistic_baseline.json"
    )
    probabilistic_parser.add_argument("--overwrite", action="store_true")
    probabilistic_parser.set_defaults(handler=_probabilistic_baseline)

    tuned_parser = subparsers.add_parser("probabilistic-tuned-baseline")
    tuned_parser.add_argument("input")
    tuned_parser.add_argument(
        "--output", default="artifacts/matr-probabilistic-tuned-baseline.json"
    )
    tuned_parser.add_argument(
        "--predictions-output",
        default="artifacts/matr-probabilistic-tuned-predictions.csv",
    )
    tuned_parser.add_argument("--conformal-coverage", type=float, default=0.8)
    tuned_parser.add_argument(
        "--l2-grid",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    )
    tuned_parser.add_argument("--inner-group-column", default="protocol_id")
    tuned_parser.add_argument("--inner-cv-folds", type=int, default=5)
    tuned_parser.add_argument("--inner-cv-seed", type=int, default=42)
    tuned_parser.add_argument("--assume-all-observed", action="store_true")
    tuned_parser.add_argument(
        "--config",
        default="configs/experiments/matr_probabilistic_tuned_baseline.json",
    )
    tuned_parser.add_argument("--overwrite", action="store_true")
    tuned_parser.set_defaults(handler=_probabilistic_tuned_baseline)

    reference_parser = subparsers.add_parser("reference-cell-feasibility")
    reference_parser.add_argument("input")
    reference_parser.add_argument(
        "--config",
        default="configs/experiments/matr_reference_cell_feasibility.json",
    )
    reference_parser.add_argument(
        "--output", default="artifacts/matr-reference-cell-feasibility.json"
    )
    reference_parser.add_argument("--overwrite", action="store_true")
    reference_parser.set_defaults(handler=_reference_cell_feasibility)

    proxy_parser = subparsers.add_parser("batch-reference-proxy")
    proxy_parser.add_argument("input")
    proxy_parser.add_argument(
        "--config", default="configs/experiments/matr_batch_reference_proxy.json"
    )
    proxy_parser.add_argument(
        "--output", default="artifacts/matr-batch-reference-proxy.json"
    )
    proxy_parser.add_argument(
        "--predictions-output",
        default="artifacts/matr-batch-reference-proxy-predictions.csv",
    )
    proxy_parser.add_argument(
        "--selections-output",
        default="artifacts/matr-batch-reference-proxy-selections.csv",
    )
    proxy_parser.add_argument(
        "--metrics-output",
        default="artifacts/matr-batch-reference-proxy-repeat-metrics.csv",
    )
    proxy_parser.add_argument("--assume-all-observed", action="store_true")
    proxy_parser.add_argument("--overwrite", action="store_true")
    proxy_parser.set_defaults(handler=_batch_reference_proxy)

    sensitivity_parser = subparsers.add_parser("batch-reference-proxy-sensitivity")
    sensitivity_parser.add_argument("input")
    sensitivity_parser.add_argument(
        "--config",
        default=(
            "configs/experiments/matr_batch_reference_proxy_sensitivity.json"
        ),
    )
    sensitivity_parser.add_argument(
        "--output",
        default="artifacts/matr-batch-reference-proxy-sensitivity.json",
    )
    sensitivity_parser.add_argument(
        "--details-output",
        default="artifacts/matr-batch-reference-proxy-sensitivity-details.csv",
    )
    sensitivity_parser.add_argument("--assume-all-observed", action="store_true")
    sensitivity_parser.add_argument("--overwrite", action="store_true")
    sensitivity_parser.set_defaults(handler=_batch_reference_sensitivity)

    ipcw_parser = subparsers.add_parser("synthetic-ipcw-validation")
    ipcw_parser.add_argument(
        "--config",
        default="configs/experiments/synthetic_ipcw_validation.json",
    )
    ipcw_parser.add_argument(
        "--output",
        default="artifacts/synthetic-ipcw-validation.json",
    )
    ipcw_parser.add_argument(
        "--details-output",
        default="artifacts/synthetic-ipcw-validation-timepoints.csv",
    )
    ipcw_parser.add_argument("--overwrite", action="store_true")
    ipcw_parser.set_defaults(handler=_synthetic_ipcw_validation_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
