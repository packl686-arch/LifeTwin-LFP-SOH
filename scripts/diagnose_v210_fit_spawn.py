"""Run truth-free Windows spawn stress probes for the V2.10 fit path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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

from v210_diagnostic_resources import ResourceSampler  # noqa: E402


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
        knee = 0.002 * 180.0 * (
            _stable_softplus((days - 540.0) / 180.0)
            - _stable_softplus(np.full_like(days, -3.0))
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
        values = 100.0 + 2.5 * (1.0 - np.exp(-days / 8.0)) * np.exp(
            -days / 300.0
        )
        values -= scale * years**0.5
    else:
        values = 100.0 - 8.0 * years**1.45
    perturbation = 0.04 * np.sin((index + 1) * days / 137.0)
    values = np.clip(values + perturbation - perturbation[0], 40.0, 105.0)
    values[0] = 100.0
    return values


def _randomized_curve(index: int) -> np.ndarray:
    days = np.asarray(PREFIX_DAYS, dtype=np.float64)
    years = days / 365.25
    rng = np.random.Generator(np.random.PCG64DXSM(31_000_000 + index))
    loss = rng.uniform(0.0, 10.0) * np.power(years, rng.uniform(0.05, 2.2))
    loss += rng.uniform(0.0, 6.0) * np.power(years, rng.uniform(0.05, 2.2))
    loss += rng.uniform(0.0, 8.0) * (
        1.0
        - np.exp(
            -np.power(days / rng.uniform(3.0, 1800.0), rng.uniform(0.15, 2.5))
        )
    )
    knee_width = rng.uniform(5.0, 420.0)
    knee_day = rng.uniform(20.0, 900.0)
    loss += rng.uniform(0.0, 0.015) * knee_width * (
        _stable_softplus((days - knee_day) / knee_width)
        - _stable_softplus(np.full_like(days, -knee_day / knee_width))
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
            rho * noise[position - 1]
            + innovation_scale * innovations[position]
        )
    values = 100.0 - loss + activation + noise - noise[0]
    values[0] = 100.0
    return values


def _fixture_tables(
    cluster_count: int,
    suite: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_records: list[dict[str, object]] = []
    coordinate_records: list[dict[str, object]] = []
    curve_factory = _structured_curve if suite == "structured" else _randomized_curve
    for index in range(cluster_count):
        cluster_id = f"v031-diagnostic-{index:05d}"
        curve = curve_factory(index)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=96)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--suite",
        choices=("structured", "randomized"),
        default="structured",
    )
    args = parser.parse_args()
    if args.clusters < 1 or args.repeat < 1:
        raise SystemExit("clusters and repeat must be positive")
    prefix, coordinates = _fixture_tables(args.clusters, args.suite)
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
            repeat_elapsed_seconds.append(time.perf_counter() - repeat_started)
    except BaseException as error:
        resource_telemetry = sampler.stop()
        print(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "status": "failed",
                    "phase": "fit_structure_library_parallel",
                    "clusters": args.clusters,
                    "workers": args.workers,
                    "suite": args.suite,
                    "elapsed_seconds": time.perf_counter() - started,
                    "exception_chain": _exception_chain(error),
                    "worker_exit_codes": [],
                    "resource_telemetry": resource_telemetry,
                },
                sort_keys=True,
            )
        )
        return 1
    resource_telemetry = sampler.stop()
    if not all(item == observed_hashes[0] for item in observed_hashes):
        print(
            json.dumps(
                {
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
                    "worker_exit_codes": [],
                    "resource_telemetry": resource_telemetry,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
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
                "worker_exit_codes": [],
                "resource_telemetry": resource_telemetry,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
