from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Sequence
import uuid

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_analysis import (
    SyntheticAnalysisResult,
    analyze_synthetic_identifiability,
)
from lifetwin.experiments.calendar_long_horizon_synthetic import (
    DECISION_COLUMNS,
    FORECAST_COORDINATE_COLUMNS,
    MATCHED_PAIR_COLUMNS,
    MEMBER_DIAGNOSTIC_COLUMNS,
    PREDICTION_COLUMNS,
    PREFIX_COLUMNS,
    TRUTH_FAMILY_IDS,
    TRUTH_PACK_COLUMNS,
    ValidatedSyntheticProtocol,
    build_disagreement_decisions,
    build_label_free_predictions,
    canonical_csv_bytes,
    evaluate_matched_pair_rejection,
    generate_all_matched_pair_packs,
    generate_cluster_packs,
    load_frozen_protocol_config,
    sample_truth_spec,
    score_frozen_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(
    "configs/experiments/synthetic_long_horizon_identifiability_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/synthetic_long_horizon_identifiability_v1"
)
TRUTH_COMMITMENT_KEYS = {
    "protocol_id",
    "config_sha256",
    "truth_pack_byte_sha256",
    "truth_pack_row_count",
    "created_utc",
    "truth_values_withheld_until_prediction_commitment",
}
PREDICTION_COMMITMENT_KEYS = {
    "protocol_id",
    "config_sha256",
    "prefix_pack_byte_sha256",
    "forecast_coordinates_byte_sha256",
    "prediction_bundle_byte_sha256",
    "decision_bundle_byte_sha256",
    "member_fit_diagnostics_byte_sha256",
    "row_counts",
    "created_utc",
    "truth_pack_opened_before_commitment",
}
FROZEN_SOURCE_PATHS = (
    Path("configs/experiments/synthetic_long_horizon_identifiability_v1.json"),
    Path("src/lifetwin/experiments/calendar_long_horizon_synthetic.py"),
    Path("src/lifetwin/experiments/calendar_long_horizon_analysis.py"),
    Path("scripts/run_synthetic_long_horizon_identifiability.py"),
    Path("pyproject.toml"),
    Path("requirements/reproduction.txt"),
)
THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
}


_WORKER_PROTOCOL: ValidatedSyntheticProtocol | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Evidence JSON cannot contain NaN or infinity")
        return result
    if value is pd.NA:
        return None
    return value


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Evidence artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            _json_ready(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            _json_ready(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(
    path: Path,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, Any]:
    raw = canonical_csv_bytes(frame.loc[:, list(columns)], columns=columns)
    _atomic_write_bytes(path, raw)
    return {
        "path": path.name,
        "row_count": int(len(frame)),
        "byte_sha256": _sha256_bytes(raw),
    }


def _read_canonical_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    raw = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(f"Unexpected columns in {path.name}: {tuple(frame.columns)}")
    if canonical_csv_bytes(frame, columns=columns) != raw:
        raise ValueError(f"{path.name} is not the frozen canonical CSV serialization")
    return frame


def _read_commitment(path: Path, expected_keys: set[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"{path.name} does not have its exact frozen key allowlist")
    return payload


def _ordinary_cluster_id(
    protocol: ValidatedSyntheticProtocol,
    partition: str,
    family_id: str,
    cluster_index: int,
) -> str:
    root = dict(protocol.partition_seed_roots)[partition]
    material = (
        f"{protocol.protocol_id}|{root}|opaque_ordinary_cluster|{partition}|"
        f"{family_id}|{cluster_index}"
    ).encode("ascii")
    return "c_" + hashlib.sha256(material).hexdigest()[:32]


def _sort_bundle(frame: pd.DataFrame, keys: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    return (
        frame.sort_values(list(keys), kind="stable")
        .reset_index(drop=True)
        .loc[:, list(columns)]
    )


def generate_phase(
    config_path: Path,
    label_free_dir: Path,
    sealed_dir: Path,
) -> dict[str, Any]:
    """Generate truth in an isolated process and expose only its commitment."""
    protocol = load_frozen_protocol_config(config_path)
    label_free_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=False)
    protected_outputs = (
        label_free_dir / "prefix_pack.csv",
        label_free_dir / "forecast_coordinates.csv",
        label_free_dir / "truth_commitment.json",
        sealed_dir / "truth_pack.csv",
        sealed_dir / "matched_prefix_pairs.csv",
    )
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError("Generation phase never overwrites evidence artifacts")

    prefix_frames: list[pd.DataFrame] = []
    coordinate_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    seen_cluster_ids: set[str] = set()
    seen_stream_seeds: set[int] = set()
    ordinary_count = 0
    for partition, per_family in protocol.cluster_counts_per_truth_family:
        for family_id in TRUTH_FAMILY_IDS:
            for index in range(per_family):
                cluster_id = _ordinary_cluster_id(
                    protocol, partition, family_id, index
                )
                if cluster_id in seen_cluster_ids:
                    raise RuntimeError("Opaque ordinary cluster ID collision")
                seen_cluster_ids.add(cluster_id)
                truth_spec = sample_truth_spec(
                    protocol,
                    partition=partition,
                    family_id=family_id,
                    zero_based_family_cluster_index=index,
                    opaque_cluster_id=cluster_id,
                )
                for seed in (truth_spec.truth_seed, truth_spec.measurement_seed):
                    if seed in seen_stream_seeds:
                        raise RuntimeError("Ordinary truth/noise seed collision")
                    seen_stream_seeds.add(seed)
                packs = generate_cluster_packs(protocol, truth_spec)
                prefix_frames.append(packs.prefix_pack)
                coordinate_frames.append(packs.forecast_coordinates)
                truth_frames.append(packs.truth_pack)
                ordinary_count += 1

    matched = generate_all_matched_pair_packs(protocol)
    matched_ids = set(matched.prefix_pack["cluster_id"].astype(str))
    if seen_cluster_ids & matched_ids:
        raise RuntimeError("Ordinary and matched opaque cluster IDs collided")
    prefix_frames.append(matched.prefix_pack)
    coordinate_frames.append(matched.forecast_coordinates)
    truth_frames.append(matched.truth_pack)
    prefix = _sort_bundle(
        pd.concat(prefix_frames, ignore_index=True),
        ("partition", "cluster_id", "prefix_day"),
        PREFIX_COLUMNS,
    )
    coordinates = _sort_bundle(
        pd.concat(coordinate_frames, ignore_index=True),
        ("partition", "cluster_id", "forecast_day"),
        FORECAST_COORDINATE_COLUMNS,
    )
    truth = _sort_bundle(
        pd.concat(truth_frames, ignore_index=True),
        ("partition", "cluster_id", "forecast_day"),
        TRUTH_PACK_COLUMNS,
    )
    pair_mapping = _sort_bundle(
        matched.matched_prefix_pairs,
        ("pair_id",),
        MATCHED_PAIR_COLUMNS,
    )
    prefix_meta = _write_csv(label_free_dir / "prefix_pack.csv", prefix, PREFIX_COLUMNS)
    coordinate_meta = _write_csv(
        label_free_dir / "forecast_coordinates.csv",
        coordinates,
        FORECAST_COORDINATE_COLUMNS,
    )
    truth_meta = _write_csv(sealed_dir / "truth_pack.csv", truth, TRUTH_PACK_COLUMNS)
    pair_meta = _write_csv(
        sealed_dir / "matched_prefix_pairs.csv", pair_mapping, MATCHED_PAIR_COLUMNS
    )
    commitment = {
        "protocol_id": protocol.protocol_id,
        "config_sha256": protocol.config_sha256,
        "truth_pack_byte_sha256": truth_meta["byte_sha256"],
        "truth_pack_row_count": truth_meta["row_count"],
        "created_utc": _utc_now(),
        "truth_values_withheld_until_prediction_commitment": True,
    }
    _write_json(label_free_dir / "truth_commitment.json", commitment)
    return {
        "ordinary_cluster_count": ordinary_count,
        "matched_cluster_count": int(matched.prefix_pack["cluster_id"].nunique()),
        "unique_ordinary_stream_seed_count": len(seen_stream_seeds),
        "prefix": prefix_meta,
        "coordinates": coordinate_meta,
        "truth": truth_meta,
        "matched_pairs": pair_meta,
    }


def _initialize_prediction_worker(config_path: str) -> None:
    global _WORKER_PROTOCOL
    _WORKER_PROTOCOL = load_frozen_protocol_config(Path(config_path))


def _prediction_worker(
    task: tuple[str, str, tuple[float, ...], tuple[float, ...], tuple[float, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if _WORKER_PROTOCOL is None:
        raise RuntimeError("Prediction worker protocol was not initialized")
    partition, cluster_id, prefix_days, observed, forecast_days = task
    prefix = pd.DataFrame(
        {
            "protocol_id": _WORKER_PROTOCOL.protocol_id,
            "partition": partition,
            "cluster_id": cluster_id,
            "prefix_day": prefix_days,
            "observed_retention_pct": observed,
        },
        columns=PREFIX_COLUMNS,
    )
    coordinates = pd.DataFrame(
        {
            "protocol_id": _WORKER_PROTOCOL.protocol_id,
            "partition": partition,
            "cluster_id": cluster_id,
            "forecast_day": forecast_days,
        },
        columns=FORECAST_COORDINATE_COLUMNS,
    )
    result = build_label_free_predictions(prefix, coordinates, _WORKER_PROTOCOL)
    return result.prediction_bundle, result.member_diagnostics


def _prediction_tasks(
    prefix: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> list[tuple[str, str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    tasks = []
    coordinate_groups = {
        (str(partition), str(cluster_id)): tuple(
            group.sort_values("forecast_day", kind="stable")["forecast_day"].astype(float)
        )
        for (partition, cluster_id), group in coordinates.groupby(
            ["partition", "cluster_id"], sort=False
        )
    }
    for (partition, cluster_id), group in prefix.groupby(
        ["partition", "cluster_id"], sort=True
    ):
        key = (str(partition), str(cluster_id))
        ordered = group.sort_values("prefix_day", kind="stable")
        tasks.append(
            (
                key[0],
                key[1],
                tuple(ordered["prefix_day"].astype(float)),
                tuple(ordered["observed_retention_pct"].astype(float)),
                coordinate_groups[key],
            )
        )
    if len(tasks) != len(coordinate_groups):
        raise ValueError("Prefix and forecast-coordinate cluster sets differ")
    return tasks


def _iter_prediction_results(
    tasks: list[
        tuple[str, str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    ],
    config_path: Path,
    workers: int,
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
    if workers == 1:
        _initialize_prediction_worker(str(config_path))
        for task in tasks:
            yield _prediction_worker(task)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_prediction_worker,
        initargs=(str(config_path),),
    ) as executor:
        yield from executor.map(_prediction_worker, tasks, chunksize=1)


def prediction_phase(
    config_path: Path,
    label_free_dir: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Fit from the two allowlisted label-free CSVs; no truth path is accepted."""
    if workers < 1:
        raise ValueError("workers must be positive")
    protocol = load_frozen_protocol_config(config_path)
    truth_commitment = _read_commitment(
        label_free_dir / "truth_commitment.json", TRUTH_COMMITMENT_KEYS
    )
    if (
        truth_commitment["protocol_id"] != protocol.protocol_id
        or truth_commitment["config_sha256"] != protocol.config_sha256
        or truth_commitment["truth_values_withheld_until_prediction_commitment"]
        is not True
    ):
        raise ValueError("Truth commitment does not match the frozen protocol")
    prefix_path = label_free_dir / "prefix_pack.csv"
    coordinates_path = label_free_dir / "forecast_coordinates.csv"
    prefix = _read_canonical_csv(prefix_path, PREFIX_COLUMNS)
    coordinates = _read_canonical_csv(
        coordinates_path, FORECAST_COORDINATE_COLUMNS
    )
    tasks = _prediction_tasks(prefix, coordinates)
    predictions: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for completed, (cluster_predictions, cluster_diagnostics) in enumerate(
        _iter_prediction_results(tasks, config_path, workers), start=1
    ):
        predictions.append(cluster_predictions)
        diagnostics.append(cluster_diagnostics)
        if completed == 1 or completed % 25 == 0 or completed == len(tasks):
            print(
                f"prediction progress: {completed}/{len(tasks)} clusters",
                flush=True,
            )
    prediction_bundle = _sort_bundle(
        pd.concat(predictions, ignore_index=True),
        ("partition", "cluster_id", "forecast_day"),
        PREDICTION_COLUMNS,
    )
    member_diagnostics = _sort_bundle(
        pd.concat(diagnostics, ignore_index=True),
        ("partition", "cluster_id", "model_id", "variant_id"),
        MEMBER_DIAGNOSTIC_COLUMNS,
    )
    decision = build_disagreement_decisions(
        prediction_bundle, member_diagnostics, protocol
    ).decision_bundle
    prediction_meta = _write_csv(
        label_free_dir / "prediction_bundle.csv",
        prediction_bundle,
        PREDICTION_COLUMNS,
    )
    decision_meta = _write_csv(
        label_free_dir / "decision_bundle.csv", decision, DECISION_COLUMNS
    )
    diagnostics_meta = _write_csv(
        label_free_dir / "member_fit_diagnostics.csv",
        member_diagnostics,
        MEMBER_DIAGNOSTIC_COLUMNS,
    )
    commitment = {
        "protocol_id": protocol.protocol_id,
        "config_sha256": protocol.config_sha256,
        "prefix_pack_byte_sha256": _sha256_path(prefix_path),
        "forecast_coordinates_byte_sha256": _sha256_path(coordinates_path),
        "prediction_bundle_byte_sha256": prediction_meta["byte_sha256"],
        "decision_bundle_byte_sha256": decision_meta["byte_sha256"],
        "member_fit_diagnostics_byte_sha256": diagnostics_meta["byte_sha256"],
        "row_counts": {
            "prefix_pack": len(prefix),
            "forecast_coordinates": len(coordinates),
            "prediction_bundle": len(prediction_bundle),
            "decision_bundle": len(decision),
            "member_fit_diagnostics": len(member_diagnostics),
        },
        "created_utc": _utc_now(),
        "truth_pack_opened_before_commitment": False,
    }
    _write_json(label_free_dir / "prediction_commitment.json", commitment)
    return {
        "cluster_count": len(tasks),
        "prediction": prediction_meta,
        "decision": decision_meta,
        "member_diagnostics": diagnostics_meta,
    }


def _write_analysis_tables(
    output_dir: Path,
    analysis: SyntheticAnalysisResult,
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, value in vars(analysis).items():
        if name == "report" or not isinstance(value, pd.DataFrame):
            continue
        path = output_dir / f"{name}.csv"
        columns = tuple(str(column) for column in value.columns)
        artifacts[name] = _write_csv(path, value, columns)
    return artifacts


def score_phase(config_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = load_frozen_protocol_config(config_path)
    truth_commitment = _read_commitment(
        output_dir / "truth_commitment.json", TRUTH_COMMITMENT_KEYS
    )
    prediction_commitment = _read_commitment(
        output_dir / "prediction_commitment.json", PREDICTION_COMMITMENT_KEYS
    )
    if prediction_commitment["truth_pack_opened_before_commitment"] is not False:
        raise ValueError("Prediction commitment reports premature truth access")
    if (
        truth_commitment["protocol_id"] != protocol.protocol_id
        or prediction_commitment["protocol_id"] != protocol.protocol_id
        or truth_commitment["config_sha256"] != protocol.config_sha256
        or prediction_commitment["config_sha256"] != protocol.config_sha256
    ):
        raise ValueError("Commitment protocol/config identity mismatch")

    # The strict scorer verifies every label-free byte commitment before this
    # call can open and parse truth_pack.csv.
    score = score_frozen_predictions(
        output_dir / "prefix_pack.csv",
        output_dir / "prediction_bundle.csv",
        output_dir / "decision_bundle.csv",
        output_dir / "forecast_coordinates.csv",
        output_dir / "member_fit_diagnostics.csv",
        output_dir / "truth_pack.csv",
        protocol,
        expected_prefix_sha256=prediction_commitment[
            "prefix_pack_byte_sha256"
        ],
        expected_prediction_sha256=prediction_commitment[
            "prediction_bundle_byte_sha256"
        ],
        expected_decision_sha256=prediction_commitment[
            "decision_bundle_byte_sha256"
        ],
        expected_forecast_coordinates_sha256=prediction_commitment[
            "forecast_coordinates_byte_sha256"
        ],
        expected_member_diagnostics_sha256=prediction_commitment[
            "member_fit_diagnostics_byte_sha256"
        ],
        expected_truth_sha256=truth_commitment["truth_pack_byte_sha256"],
    )
    matched_mapping = _read_canonical_csv(
        output_dir / "matched_prefix_pairs.csv", MATCHED_PAIR_COLUMNS
    )
    matched_audit = evaluate_matched_pair_rejection(score, matched_mapping, protocol)
    analysis = analyze_synthetic_identifiability(score, matched_audit, protocol)
    table_artifacts = _write_analysis_tables(output_dir, analysis)
    table_artifacts["point_scores"] = _write_csv(
        output_dir / "point_scores.csv",
        score.point_scores,
        tuple(str(column) for column in score.point_scores.columns),
    )
    table_artifacts["trajectory_scores"] = _write_csv(
        output_dir / "trajectory_scores.csv",
        score.trajectory_scores,
        tuple(str(column) for column in score.trajectory_scores.columns),
    )
    table_artifacts["matched_pair_scores"] = _write_csv(
        output_dir / "matched_pair_scores.csv",
        matched_audit.pair_scores,
        tuple(str(column) for column in matched_audit.pair_scores.columns),
    )
    input_artifacts = {}
    for filename in (
        "prefix_pack.csv",
        "forecast_coordinates.csv",
        "truth_commitment.json",
        "prediction_bundle.csv",
        "decision_bundle.csv",
        "prediction_commitment.json",
        "member_fit_diagnostics.csv",
        "truth_pack.csv",
        "matched_prefix_pairs.csv",
        "environment.json",
    ):
        path = output_dir / filename
        input_artifacts[filename] = {
            "byte_count": path.stat().st_size,
            "byte_sha256": _sha256_path(path),
        }
    report = dict(analysis.report)
    report.update(
        {
            "scored_utc": _utc_now(),
            "protocol_deviations": [],
            "verified_commitments": {
                "prefix_pack_byte_sha256": score.prefix_sha256,
                "forecast_coordinates_byte_sha256": (
                    score.forecast_coordinates_sha256
                ),
                "prediction_bundle_byte_sha256": score.prediction_sha256,
                "decision_bundle_byte_sha256": score.decision_sha256,
                "member_fit_diagnostics_byte_sha256": (
                    score.member_diagnostics_sha256
                ),
                "truth_pack_byte_sha256": score.truth_sha256,
            },
            "input_artifacts": input_artifacts,
            "analysis_artifacts": table_artifacts,
        }
    )
    _write_json(output_dir / "score_report.json", report)
    return report


def _git_output(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _current_source_hashes() -> dict[str, str]:
    source_hashes: dict[str, str] = {}
    for path in FROZEN_SOURCE_PATHS:
        source_path = PROJECT_ROOT / path
        if not source_path.is_file():
            raise RuntimeError(f"Frozen source file is missing: {path.as_posix()}")
        source_hashes[path.as_posix()] = _sha256_path(source_path)
    return source_hashes


def _source_tree_sha256(source_hashes: dict[str, str]) -> str:
    source_tree = json.dumps(
        source_hashes, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return _sha256_bytes(source_tree)


def _verify_environment_freeze(environment: dict[str, Any]) -> None:
    expected_status = environment.get("git_status_porcelain")
    if expected_status != "":
        raise RuntimeError("Frozen environment did not record a clean worktree")
    current_status = _git_output(
        ("status", "--porcelain=v1", "--untracked-files=all")
    )
    if current_status != expected_status:
        raise RuntimeError(
            "Frozen evidence environment drift: git worktree is no longer clean"
        )

    expected_commit = environment.get("git_commit")
    current_commit = _git_output(("rev-parse", "HEAD"))
    if current_commit != expected_commit:
        raise RuntimeError(
            "Frozen evidence environment drift: git HEAD changed during execution"
        )

    expected_hashes = environment.get("source_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("Frozen environment source hash record is invalid")
    current_hashes = _current_source_hashes()
    if current_hashes != expected_hashes:
        changed_paths = sorted(
            path
            for path in set(current_hashes) | set(expected_hashes)
            if current_hashes.get(path) != expected_hashes.get(path)
        )
        raise RuntimeError(
            "Frozen evidence environment drift: source files changed: "
            + ", ".join(changed_paths)
        )

    expected_tree_hash = environment.get("source_tree_sha256")
    current_tree_hash = _source_tree_sha256(current_hashes)
    if current_tree_hash != expected_tree_hash:
        raise RuntimeError(
            "Frozen evidence environment drift: source tree hash changed"
        )


def _environment_record(
    protocol: ValidatedSyntheticProtocol,
    *,
    workers: int,
) -> dict[str, Any]:
    status = _git_output(("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise RuntimeError(
            "The primary evidence run requires a clean committed worktree; "
            "commit implementation changes before simulation"
        )
    source_hashes = _current_source_hashes()
    environment = {
        "protocol_id": protocol.protocol_id,
        "config_canonical_sha256": protocol.config_sha256,
        "config_byte_sha256": source_hashes[DEFAULT_CONFIG.as_posix()],
        "git_commit": _git_output(("rev-parse", "HEAD")),
        "git_status_porcelain": "",
        "source_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
        "started_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "prediction_worker_count": workers,
        "thread_environment": THREAD_ENVIRONMENT,
        "packages": {
            package: importlib_metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn")
        },
    }
    _verify_environment_freeze(environment)
    return environment


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    source_path = str(PROJECT_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not existing else source_path + os.pathsep + existing
    )
    return environment


def _run_child(arguments: Sequence[str]) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        cwd=PROJECT_ROOT,
        env=_child_environment(),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Isolated phase failed with exit code {completed.returncode}: "
            f"{' '.join(arguments)}"
        )


def _run_child_with_freeze_guard(
    arguments: Sequence[str],
    environment: dict[str, Any],
) -> None:
    _verify_environment_freeze(environment)
    try:
        _run_child(arguments)
    finally:
        _verify_environment_freeze(environment)


def _append_exposure_event(
    exposure_path: Path,
    event: str,
    **details: Any,
) -> None:
    if exposure_path.exists():
        payload = json.loads(exposure_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "protocol_id": None,
            "truth_pack_opened_before_prediction_commitment": False,
            "events": [],
            "protocol_deviations": [],
        }
    payload["events"].append(
        {
            "sequence": len(payload["events"]) + 1,
            "event": event,
            "created_utc": _utc_now(),
            **details,
        }
    )
    _replace_json(exposure_path, payload)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Execute generation, label-free prediction, then one-time scoring."""
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Evidence runner never overwrites {output_dir}")
    protocol = load_frozen_protocol_config(config_path)
    environment = _environment_record(protocol, workers=workers)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
    sealed = parent / f".{output_dir.name}.sealed.{uuid.uuid4().hex}"
    staging.mkdir()
    exposure_path = staging / "exposure_log.json"
    try:
        _write_json(staging / "environment.json", environment)
        _append_exposure_event(
            exposure_path,
            "environment_and_code_frozen",
            git_commit=environment["git_commit"],
            source_tree_sha256=environment["source_tree_sha256"],
        )
        exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
        exposure["protocol_id"] = protocol.protocol_id
        _replace_json(exposure_path, exposure)
        _run_child_with_freeze_guard(
            (
                "--phase",
                "generate",
                "--config",
                str(config_path),
                "--work-dir",
                str(staging),
                "--sealed-dir",
                str(sealed),
            ),
            environment,
        )
        truth_commitment = _read_commitment(
            staging / "truth_commitment.json", TRUTH_COMMITMENT_KEYS
        )
        _append_exposure_event(
            exposure_path,
            "sealed_truth_generated_and_committed",
            truth_pack_byte_sha256=truth_commitment[
                "truth_pack_byte_sha256"
            ],
            predictor_received_truth_path=False,
        )
        _run_child_with_freeze_guard(
            (
                "--phase",
                "predict",
                "--config",
                str(config_path),
                "--work-dir",
                str(staging),
                "--workers",
                str(workers),
            ),
            environment,
        )
        prediction_commitment = _read_commitment(
            staging / "prediction_commitment.json",
            PREDICTION_COMMITMENT_KEYS,
        )
        if prediction_commitment["truth_pack_opened_before_commitment"] is not False:
            raise RuntimeError("Prediction phase reported premature truth access")
        _append_exposure_event(
            exposure_path,
            "label_free_predictions_and_decisions_committed",
            prediction_bundle_byte_sha256=prediction_commitment[
                "prediction_bundle_byte_sha256"
            ],
            decision_bundle_byte_sha256=prediction_commitment[
                "decision_bundle_byte_sha256"
            ],
            truth_pack_opened_before_commitment=False,
        )
        for filename in ("truth_pack.csv", "matched_prefix_pairs.csv"):
            os.replace(sealed / filename, staging / filename)
        sealed.rmdir()
        _append_exposure_event(
            exposure_path,
            "sealed_truth_released_to_scorer_after_prediction_commitment",
            truth_pack_opened_before_prediction_commitment=False,
        )
        _run_child_with_freeze_guard(
            (
                "--phase",
                "score",
                "--config",
                str(config_path),
                "--work-dir",
                str(staging),
            ),
            environment,
        )
        report = json.loads(
            (staging / "score_report.json").read_text(encoding="utf-8")
        )
        _append_exposure_event(
            exposure_path,
            "strict_scoring_completed",
            result_status=report["status"],
            truth_pack_opened_before_prediction_commitment=False,
        )
        _verify_environment_freeze(environment)
        os.replace(staging, output_dir)
        return report
    except BaseException as exc:
        try:
            _append_exposure_event(
                exposure_path,
                "execution_void_due_to_exception",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
        except Exception:
            pass
        if sealed.exists() and staging.exists():
            sealed_destination = staging / "sealed_unopened_after_void"
            if not sealed_destination.exists():
                os.replace(sealed, sealed_destination)
        if staging.exists():
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            void_path = parent / f"{output_dir.name}.void.{suffix}"
            if not void_path.exists():
                os.replace(staging, void_path)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen synthetic 25-year identifiability experiment"
    )
    parser.add_argument("--phase", choices=("run", "generate", "predict", "score"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--sealed-dir", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if args.phase == "run":
        output = (
            args.output_dir
            if args.output_dir.is_absolute()
            else PROJECT_ROOT / args.output_dir
        )
        result = run(config, output, workers=args.workers)
    else:
        if args.work_dir is None:
            raise ValueError("Internal phases require --work-dir")
        work = args.work_dir.resolve()
        if args.phase == "generate":
            if args.sealed_dir is None:
                raise ValueError("Generation phase requires --sealed-dir")
            result = generate_phase(config, work, args.sealed_dir.resolve())
        elif args.phase == "predict":
            result = prediction_phase(config, work, workers=args.workers)
        else:
            result = score_phase(config, work)
    print(json.dumps(_json_ready(result), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
