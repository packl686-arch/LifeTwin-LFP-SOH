"""Run truth-free Windows spawn stress probes for the V2.10 fit path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:  # noqa: E402
    from v210_diagnostic_resources import ResourceSampler
except ModuleNotFoundError:  # pragma: no cover - import-mode compatibility
    from scripts.v210_diagnostic_resources import ResourceSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifetwin.experiments import (  # noqa: E402
    calendar_long_horizon_v015_prediction as prediction,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (  # noqa: E402
    canonical_csv_bytes,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (  # noqa: E402
    PREFIX_DAYS,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (  # noqa: E402
    FORECAST_COORDINATE_COLUMNS,
    FROZEN_PROTOCOL_ID,
    PREFIX_COLUMNS,
)


FORECAST_DAYS = (
    1095.75,
    1461.0,
    1826.25,
    2556.75,
    3652.5,
    5478.75,
    7305.0,
    9131.25,
)


def _stable_softplus(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def _structured_curve(index: int) -> np.ndarray:
    days = np.asarray(PREFIX_DAYS, dtype=np.float64)
    years = days / 365.25
    kind = index % 12
    scale = 0.35 + 0.07 * (index % 9)
    if kind == 0:
        values = 100.0 - scale * np.sqrt(years)
    elif kind == 1:
        values = 100.0 - scale * np.power(years, 0.2 + 0.12 * (index % 7))
    elif kind == 2:
        values = 100.0 - 1.8 * (1.0 - np.exp(-days / 90.0)) - scale * years**0.7
    elif kind == 3:
        values = 100.0 - 0.4 * years**0.2 - scale * years**1.2
    elif kind == 4:
        activation = 1.6 * (1.0 - np.exp(-days / 18.0)) * np.exp(-days / 240.0)
        values = 100.0 - scale * years**0.65 + activation
    elif kind == 5:
        knee = (
            0.002
            * 180.0
            * (
                _stable_softplus((days - 540.0) / 180.0)
                - _stable_softplus(np.full_like(days, -3.0))
            )
        )
        values = 100.0 - scale * years**0.55 - knee
    elif kind == 6:
        values = 100.0 - (0.001 + index * 0.00001) * days
    elif kind == 7:
        values = np.full_like(days, 100.0 - 0.02 * (index % 5))
    elif kind == 8:
        values = 100.0 - scale * np.sqrt(years) + 0.3 * np.sin(days / 45.0)
    elif kind == 9:
        values = 100.0 - 4.5 * (1.0 - np.exp(-days / 12.0)) - 0.2 * years
    elif kind == 10:
        values = 100.0 + 2.5 * (1.0 - np.exp(-days / 8.0)) * np.exp(-days / 300.0)
        values -= scale * years**0.5
    else:
        values = 100.0 - 8.0 * years**1.45
    perturbation = 0.04 * np.sin((index + 1) * days / 137.0)
    values = np.clip(values + perturbation - perturbation[0], 40.0, 105.0)
    values[0] = 100.0
    return values


def _randomized_curve(index: int, seed_root: int = 31_000_000) -> np.ndarray:
    days = np.asarray(PREFIX_DAYS, dtype=np.float64)
    years = days / 365.25
    rng = np.random.Generator(np.random.PCG64DXSM(seed_root + index))
    loss = rng.uniform(0.0, 10.0) * np.power(years, rng.uniform(0.05, 2.2))
    loss += rng.uniform(0.0, 6.0) * np.power(years, rng.uniform(0.05, 2.2))
    loss += rng.uniform(0.0, 8.0) * (
        1.0 - np.exp(-np.power(days / rng.uniform(3.0, 1800.0), rng.uniform(0.15, 2.5)))
    )
    knee_width = rng.uniform(5.0, 420.0)
    knee_day = rng.uniform(20.0, 900.0)
    loss += (
        rng.uniform(0.0, 0.015)
        * knee_width
        * (
            _stable_softplus((days - knee_day) / knee_width)
            - _stable_softplus(np.full_like(days, -knee_day / knee_width))
        )
    )
    activation = (
        rng.uniform(-3.0, 6.0)
        * (1.0 - np.exp(-days / rng.uniform(2.0, 240.0)))
        * np.exp(-days / rng.uniform(30.0, 1500.0))
    )
    innovations = rng.standard_normal(len(days))
    rho = rng.uniform(0.0, 0.97)
    sigma = rng.uniform(0.01, 2.5)
    noise = np.empty_like(days)
    noise[0] = sigma * innovations[0]
    innovation_scale = sigma * math.sqrt(1.0 - rho**2)
    for position in range(1, len(days)):
        noise[position] = (
            rho * noise[position - 1] + innovation_scale * innovations[position]
        )
    values = 100.0 - loss + activation + noise - noise[0]
    values[0] = 100.0
    return values


def _fixture_tables(
    cluster_count: int,
    suite: str,
    *,
    seed_root: int = 31_000_000,
    cluster_prefix: str = "v031-diagnostic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_records: list[dict[str, object]] = []
    coordinate_records: list[dict[str, object]] = []
    for index in range(cluster_count):
        cluster_id = f"{cluster_prefix}-{index:05d}"
        if suite == "structured" or (suite == "mixed" and index % 2 == 0):
            curve = _structured_curve(index)
        else:
            curve = _randomized_curve(index, seed_root)
        if not np.isfinite(curve).all():
            raise RuntimeError("Diagnostic curve is nonfinite")
        for day, observed in zip(PREFIX_DAYS, curve, strict=True):
            prefix_records.append(
                {
                    "protocol_id": FROZEN_PROTOCOL_ID,
                    "partition": "calibration",
                    "cluster_id": cluster_id,
                    "prefix_day": day,
                    "observed_retention_pct": float(observed),
                }
            )
        for day in FORECAST_DAYS:
            coordinate_records.append(
                {
                    "protocol_id": FROZEN_PROTOCOL_ID,
                    "partition": "calibration",
                    "cluster_id": cluster_id,
                    "forecast_day": day,
                }
            )
    return (
        pd.DataFrame(prefix_records, columns=PREFIX_COLUMNS),
        pd.DataFrame(coordinate_records, columns=FORECAST_COORDINATE_COLUMNS),
    )


def _result_hashes(result: prediction.V015FitResult) -> dict[str, str]:
    contract = load_artifact_contract()
    frames = {
        "member_fit_diagnostics.csv": result.member_fit_diagnostics,
        "member_forecast_bundle.csv": result.member_forecast_bundle,
    }
    return {
        name: hashlib.sha256(
            canonical_csv_bytes(
                frame,
                contract.csv_schema(name),
                contract,
                formal=False,
            )
        ).hexdigest()
        for name, frame in frames.items()
    }


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    relationship = "outer"
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "exception_class": type(current).__name__,
                "relationship": relationship,
            }
        )
        if current.__cause__ is not None:
            current = current.__cause__
            relationship = "cause"
        elif not current.__suppress_context__:
            current = current.__context__
            relationship = "context"
        else:
            current = None
    return chain


def _commit_progress(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=96)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--suite",
        choices=("structured", "randomized", "mixed"),
        default="structured",
    )
    parser.add_argument("--seed-root", type=int, default=31_000_000)
    parser.add_argument("--cluster-prefix", default="v031-diagnostic")
    parser.add_argument(
        "--execution-profile",
        choices=("nonformal", "v300-formal"),
        default="nonformal",
    )
    parser.add_argument("--authorization-record", type=Path)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    if args.clusters < 1 or args.repeat < 1:
        raise SystemExit("clusters and repeat must be positive")
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    if args.seed_root < 0 or args.seed_root > 2**63 - args.clusters:
        raise SystemExit("seed-root is outside the supported deterministic range")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}", args.cluster_prefix) is None:
        raise SystemExit("cluster-prefix is invalid")
    formal_execution = args.execution_profile == "v300-formal"
    if formal_execution:
        from lifetwin.experiments.runtime_reliability_v300_protocol import (
            V300_EXPECTED_JOBS,
            V300_FORMAL_SEED_ROOT,
            V300_ONLY_ATTEMPT_ID,
            V300_PROTOCOL_ID,
        )

        expected_authorization = (
            ROOT / "artifacts" / "v300-formal-20260815-authorization.json"
        ).resolve()
        if args.authorization_record is None:
            raise SystemExit("v300-formal requires its fixed authorization record")
        authorization_path = args.authorization_record.resolve()
        try:
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SystemExit("v300-formal authorization is unreadable") from exc
        if (
            authorization_path != expected_authorization
            or not isinstance(authorization, dict)
            or authorization.get("protocol_id") != V300_PROTOCOL_ID
            or authorization.get("attempt_id") != V300_ONLY_ATTEMPT_ID
            or authorization.get("authorization_status") != "authorized_post_freeze"
            or args.seed_root != V300_FORMAL_SEED_ROOT
            or args.suite != "mixed"
            or args.cluster_prefix != "v300-formal-runtime"
            or args.repeat != 1
            or (args.clusters, args.workers)
            not in {(clusters, workers) for _, clusters, workers in V300_EXPECTED_JOBS}
        ):
            raise SystemExit("v300-formal profile identity is invalid")
    elif args.authorization_record is not None:
        raise SystemExit("nonformal probes cannot receive a formal authorization")
    boundary = {
        "formal_inputs_used": formal_execution,
        "formal_rows_opened": False,
        "formal_seeds_used": formal_execution,
        "sealed_truth_opened": False,
    }
    prefix, coordinates = _fixture_tables(
        args.clusters,
        args.suite,
        seed_root=args.seed_root,
        cluster_prefix=args.cluster_prefix,
    )
    observed_hashes: list[dict[str, str]] = []
    repeat_elapsed_seconds: list[float] = []
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        for _ in range(args.repeat):
            repeat_started = time.perf_counter()
            result = prediction.fit_structure_library_parallel(
                prefix_pack=prefix,
                forecast_coordinates=coordinates,
                worker_count=args.workers,
            )
            observed_hashes.append(_result_hashes(result))
            del result
            gc.collect()
            repeat_elapsed_seconds.append(time.perf_counter() - repeat_started)
            deterministic_so_far = all(
                item == observed_hashes[0] for item in observed_hashes
            )
            _commit_progress(
                args.progress_file,
                {
                    "schema_version": "1.0.0",
                    "status": "in_progress",
                    "completed_repeats": len(observed_hashes),
                    "requested_repeats": args.repeat,
                    "clusters": args.clusters,
                    "workers": args.workers,
                    "suite": args.suite,
                    "elapsed_seconds": time.perf_counter() - started,
                    "hashes": observed_hashes[0],
                    "deterministic_so_far": deterministic_so_far,
                    **boundary,
                },
            )
            if not deterministic_so_far:
                break
    except BaseException as error:
        resource_telemetry = sampler.stop()
        runtime_telemetry = prediction.result_blind_worker_failure_telemetry(error)
        payload = {
            "schema_version": "1.0.0",
            "status": "failed",
            "phase": (
                runtime_telemetry["phase"]
                if runtime_telemetry is not None
                else "fit_structure_library_parallel"
            ),
            "clusters": args.clusters,
            "workers": args.workers,
            "suite": args.suite,
            "elapsed_seconds": time.perf_counter() - started,
            "exception_chain": _exception_chain(error),
            "runtime_failure_telemetry": runtime_telemetry,
            "worker_exit_codes": (
                runtime_telemetry["worker_exit_codes"]
                if runtime_telemetry is not None
                else []
            ),
            "resource_telemetry": resource_telemetry,
            **boundary,
        }
        _commit_progress(args.progress_file, payload)
        print(json.dumps(payload, sort_keys=True))
        return 1
    resource_telemetry = sampler.stop()
    if not all(item == observed_hashes[0] for item in observed_hashes):
        payload = {
            "schema_version": "1.0.0",
            "status": "failed",
            "phase": "repeat_hash_comparison",
            "clusters": args.clusters,
            "workers": args.workers,
            "suite": args.suite,
            "elapsed_seconds": time.perf_counter() - started,
            "exception_chain": [
                {
                    "exception_class": "RepeatHashMismatch",
                    "relationship": "outer",
                }
            ],
            "runtime_failure_telemetry": None,
            "worker_exit_codes": [],
            "resource_telemetry": resource_telemetry,
            **boundary,
        }
        _commit_progress(args.progress_file, payload)
        print(json.dumps(payload, sort_keys=True))
        return 1
    payload = {
        "schema_version": "1.0.0",
        "status": "passed",
        "phase": "completed",
        "clusters": args.clusters,
        "workers": args.workers,
        "repeat": args.repeat,
        "suite": args.suite,
        "elapsed_seconds": time.perf_counter() - started,
        "repeat_elapsed_seconds": repeat_elapsed_seconds,
        "diagnostic_rows": args.clusters * 86,
        "forecast_rows": args.clusters * 86 * 8,
        "hashes": observed_hashes[0],
        "runtime_failure_telemetry": None,
        "worker_exit_codes": [],
        "resource_telemetry": resource_telemetry,
        **boundary,
    }
    _commit_progress(args.progress_file, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
