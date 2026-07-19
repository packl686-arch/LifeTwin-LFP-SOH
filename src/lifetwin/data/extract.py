from __future__ import annotations

from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd


TIMESERIES_COLUMNS = (
    "timestamp_s",
    "current_A",
    "voltage_V",
    "test_id",
    "cycle_number",
    "temperature_C",
    "coulomb_count_Ah",
    "step_type",
)


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def extract_celljar_curve_subset(
    source: str | Path,
    metadata: pd.DataFrame,
    output: str | Path,
    *,
    cycles: Sequence[int] = (10, 100),
    step_type: str = "discharge",
    overwrite: bool = False,
) -> dict[str, object]:
    """Scan a local celljar Parquet once and materialize selected curve cycles."""
    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source Parquet does not exist: {source_path}")
    if source_path.stat().st_size < 1024:
        raise ValueError(
            "Source is too small to be the celljar time-series Parquet; "
            "it may still be a Git LFS pointer"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Source and output paths must be different")
    if "cell_id" not in metadata:
        raise ValueError("Metadata must contain cell_id")
    if metadata["cell_id"].isna().any() or metadata["cell_id"].duplicated().any():
        raise ValueError("Metadata cell_id values must be non-null and unique")

    requested_cycles = sorted(set(int(cycle) for cycle in cycles))
    if not requested_cycles or requested_cycles[0] < 1:
        raise ValueError("At least one positive cycle number is required")
    if not step_type or "'" in step_type:
        raise ValueError("Invalid step_type")

    if "test_id" in metadata:
        if metadata["test_id"].isna().any() or metadata["test_id"].duplicated().any():
            raise ValueError("Metadata test_id values must be non-null and unique")
        selected_tests = metadata[["test_id"]].astype(str).copy()
    else:
        selected_tests = pd.DataFrame(
            {"test_id": metadata["cell_id"].astype(str) + "_CYCLING"}
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.name}.partial")
    temporary_output.unlink(missing_ok=True)
    cycle_sql = ",".join(str(cycle) for cycle in requested_cycles)
    column_sql = ", ".join(f"source.{column}" for column in TIMESERIES_COLUMNS)
    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order = false")
        connection.register("selected_tests", selected_tests)
        connection.execute(
            f"""
            COPY (
                SELECT {column_sql}
                FROM read_parquet('{_sql_path(source_path)}') AS source
                SEMI JOIN selected_tests USING (test_id)
                WHERE source.cycle_number IN ({cycle_sql})
                  AND source.step_type = '{step_type}'
                ORDER BY source.test_id, source.cycle_number, source.timestamp_s
            ) TO '{_sql_path(temporary_output)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        observed = connection.execute(
            f"""
            SELECT test_id,
                   CAST(cycle_number AS BIGINT) AS cycle_number,
                   COUNT(*) AS sample_count
            FROM read_parquet('{_sql_path(temporary_output)}')
            GROUP BY test_id, cycle_number
            ORDER BY test_id, cycle_number
            """
        ).fetchdf()
    finally:
        connection.close()

    expected_pairs = {
        (test_id, cycle)
        for test_id in selected_tests["test_id"]
        for cycle in requested_cycles
    }
    observed_pairs = set(
        observed[["test_id", "cycle_number"]].itertuples(index=False, name=None)
    )
    missing_pairs = sorted(expected_pairs - observed_pairs)
    unexpected_pairs = sorted(observed_pairs - expected_pairs)
    temporary_output.replace(output_path)
    return {
        "source": str(source_path.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "output": str(output_path.resolve()),
        "output_size_bytes": output_path.stat().st_size,
        "cell_count_requested": len(selected_tests),
        "cycles_requested": requested_cycles,
        "expected_cell_cycle_pairs": len(expected_pairs),
        "observed_cell_cycle_pairs": len(observed_pairs),
        "row_count": int(observed["sample_count"].sum()),
        "minimum_samples_per_curve": int(observed["sample_count"].min())
        if not observed.empty
        else 0,
        "missing_cell_cycle_pairs": [list(pair) for pair in missing_pairs],
        "unexpected_cell_cycle_pairs": [list(pair) for pair in unexpected_pairs],
        "complete": not missing_pairs and not unexpected_pairs,
    }
