from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
from typing import Any
import uuid

from lifetwin.atomic_publish import (
    AtomicPublishRetryExhausted,
    publish_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE8_RUNNER = Path("scripts/run_calendar_v3_activation_development.py")
PHASE8_ANALYZER = Path("showcase/analyze_phase8_results.py")
V012_ANALYZER = Path("showcase/analyze_v012_robustness.py")
PHASE1_AUDIT_RUNNER = Path("scripts/run_phase1_adversarial_audit.py")
PHASE8_INPUT = Path("data/interim/naumann_calendar_observations.csv")
REPRODUCTION_CONSTRAINTS = Path("requirements/reproduction.txt")
PHASE8_CONFIG = Path(
    "configs/experiments/naumann_calendar_v3_activation_development.json"
)
LANDMARK_RUNNER = Path("scripts/run_calendar_landmark_readiness.py")
LANDMARK_CONFIG = Path("configs/experiments/naumann_calendar_landmark_readiness.json")
V4_RUNNER = Path("scripts/run_calendar_v4_hybrid_development.py")
V4_CONFIG = Path("configs/experiments/naumann_calendar_v4_hybrid_development.json")
GEISBAUER_RUNNER = Path("scripts/run_geisbauer_external_stress.py")
GEISBAUER_CONFIG = Path(
    "configs/experiments/geisbauer_lfp_calendar_external_stress.json"
)
GEISBAUER_INPUT = Path("data/external/geisbauer_2022/LFP_Data.csv")
V4_CALIBRATION_ROBUSTNESS_RUNNER = Path(
    "scripts/run_calendar_v4_calibration_robustness.py"
)
V4_CALIBRATION_ROBUSTNESS_CONFIG = Path(
    "configs/experiments/naumann_calendar_v4_calibration_robustness.json"
)
GEISBAUER_ROBUSTNESS_RUNNER = Path("scripts/run_geisbauer_robustness_audit.py")
GEISBAUER_ROBUSTNESS_CONFIG = Path(
    "configs/experiments/geisbauer_lfp_calendar_robustness_audit.json"
)
V011_PUBLISHED_ROOT = Path("showcase/evidence_v011")
V011_RESULT_FILES = {
    "landmark": "decision.json",
    "v4": "result.json",
    "geisbauer": "result.json",
}
V011_CORE_CSVS = {
    "landmark": (
        "common_support_metrics.csv",
        "landmark_summary.csv",
    ),
    "v4": (
        "calibration_condition_scores.csv",
        "calibration_quantiles.csv",
        "condition_metrics.csv",
        "condition_splits.csv",
    ),
    "geisbauer": (
        "condition_summary.csv",
        "comparison_summary.csv",
    ),
}
V011_VOLATILE_SHA256_COLUMNS = {
    ("landmark", "common_support_metrics.csv"): frozenset(
        {"training_state_sha256", "prediction_state_sha256"}
    ),
    ("v4", "calibration_condition_scores.csv"): frozenset(
        {"training_state_sha256", "calibration_prediction_state_sha256"}
    ),
}
V012_PUBLISHED_ROOT = Path("showcase/evidence_v012")
V012_RESULT_FILES = {
    "v4_calibration_robustness": "result.json",
    "geisbauer_robustness": "result.json",
}
V012_CORE_CSVS = {
    "v4_calibration_robustness": (
        "candidate_label_free_predictions.csv",
        "candidate_condition_scores.csv",
        "baseline_route_metrics.csv",
        "baseline_condition_metrics.csv",
        "loco_route_metrics.csv",
        "loco_condition_metrics.csv",
        "partition_catalog.csv",
        "partition_route_metrics.csv",
        "partition_condition_metrics.csv",
        "sensitivity_summary.csv",
    ),
    "geisbauer_robustness": (
        "cell_paired_deltas.csv",
        "cell_day_paired_deltas.csv",
        "stratum_diagnostics.csv",
        "leave_one_cell_out.csv",
    ),
}
V012_VOLATILE_SHA256_COLUMNS = {
    (
        "v4_calibration_robustness",
        "candidate_label_free_predictions.csv",
    ): frozenset({"training_state_sha256", "condition_prediction_state_sha256"}),
    (
        "v4_calibration_robustness",
        "candidate_condition_scores.csv",
    ): frozenset({"training_state_sha256", "condition_prediction_state_sha256"}),
}
CORE_PHASE8_CSVS = (
    "comparison_summary.csv",
    "tau_sensitivity_summary.csv",
    "target_diagnostics.csv",
)
PHASE1_AUDIT_FILES = (
    "phase1_adversarial_audit.json",
    "data_condition_audit.csv",
    "future_label_attack_cases.csv",
    "independent_metric_audit.csv",
    "ablation_audit.csv",
    "gate_boundary_cases.csv",
    "failure_condition_table.csv",
)
PACKAGE_IMPORTS = {
    "duckdb": "duckdb",
    "jsonschema": "jsonschema",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}
NUMERIC_RELATIVE_TOLERANCE = 1e-8
NUMERIC_ABSOLUTE_TOLERANCE = 2e-4
PHASE1_SOLVER_RELATIVE_TOLERANCE = 1e-8
PHASE1_SOLVER_ABSOLUTE_TOLERANCE = 5e-3
# Maximum drift found while forcing alternate local OpenBLAS kernels.
PHASE1_SOLVER_MAX_OBSERVED_ABSOLUTE_DELTA_PP = 3.8927328e-3
PHASE1_FRACTION_RELATIVE_TOLERANCE = 1e-8
PHASE1_FRACTION_ABSOLUTE_TOLERANCE = 1e-4
PHASE1_AUDIT_RESIDUAL_RELATIVE_TOLERANCE = 0.0
PHASE1_AUDIT_RESIDUAL_ABSOLUTE_TOLERANCE = 1e-10
PHASE1_FAILURE_CLASSIFICATION_TOLERANCE = 1e-12
STATE_HASH_COLUMNS = frozenset({"training_state_sha256", "prediction_state_sha256"})
FUTURE_ATTACK_HASH_PAIRS = (
    ("prediction_sha256_baseline", "prediction_sha256_attacked"),
    ("sensitivity_sha256_baseline", "sensitivity_sha256_attacked"),
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REPRODUCTION_PYTHON = (3, 12)
STAGING_PREFIX = ".lt-stage-"
STAGING_GLOB = f"{STAGING_PREFIX}*"
AUXILIARY_TOKEN_LENGTH = 12
RUNTIME_TEMP_PREFIX = ".lt-tmp-"
TEST_SCRATCH_PREFIX = ".lt-test-"


@dataclass(frozen=True)
class _Phase1CsvSchema:
    key_columns: tuple[str, ...]
    tolerant_columns: tuple[tuple[str, float, float], ...] = ()
    volatile_sha256_columns: frozenset[str] = frozenset()
    sha256_pairs: tuple[tuple[str, str], ...] = ()


def _column_tolerances(
    columns: tuple[str, ...],
    *,
    absolute: float,
    relative: float,
) -> tuple[tuple[str, float, float], ...]:
    return tuple((column, absolute, relative) for column in columns)


_INDEPENDENT_SOLVER_COLUMNS = (
    "trajectory_iae_pp",
    "future_point_mae_pp",
    "final_predicted_retention_pct",
    "final_error_pp",
    "trajectory_iae_pp_recomputed",
    "future_point_mae_pp_recomputed",
    "final_predicted_retention_pct_recomputed",
    "final_error_pp_recomputed",
)
_INDEPENDENT_AUDIT_RESIDUAL_COLUMNS = (
    "trajectory_iae_pp_audit_difference",
    "future_point_mae_pp_audit_difference",
    "final_true_retention_pct_audit_difference",
    "final_predicted_retention_pct_audit_difference",
    "final_error_pp_audit_difference",
)
_INDEPENDENT_METRIC_RELATIONSHIPS = (
    (
        "future_checkup_count",
        "future_checkup_count_recomputed",
        "future_checkup_count_audit_difference",
    ),
    (
        "trajectory_iae_pp",
        "trajectory_iae_pp_recomputed",
        "trajectory_iae_pp_audit_difference",
    ),
    (
        "future_point_mae_pp",
        "future_point_mae_pp_recomputed",
        "future_point_mae_pp_audit_difference",
    ),
    (
        "final_true_retention_pct",
        "final_true_retention_pct_recomputed",
        "final_true_retention_pct_audit_difference",
    ),
    (
        "final_predicted_retention_pct",
        "final_predicted_retention_pct_recomputed",
        "final_predicted_retention_pct_audit_difference",
    ),
    (
        "final_error_pp",
        "final_error_pp_recomputed",
        "final_error_pp_audit_difference",
    ),
)
_FAILURE_SOLVER_COLUMNS = (
    "candidate_trajectory_iae_pp",
    "candidate_future_point_mae_pp",
    "candidate_final_predicted_retention_pct",
    "candidate_final_error_pp",
    "comparator_trajectory_iae_pp",
    "comparator_future_point_mae_pp",
    "comparator_final_predicted_retention_pct",
    "comparator_final_error_pp",
    "ungated_target_trajectory_iae_pp",
    "gated_hierarchical_trajectory_iae_pp",
    "primary_vs_v2_delta_iae_pp",
    "ungated_target_vs_v2_delta_iae_pp",
    "gated_hierarchical_vs_v2_delta_iae_pp",
    "candidate_final_absolute_error_pp",
    "comparator_final_absolute_error_pp",
)
_FUTURE_ATTACK_HASH_COLUMNS = frozenset(
    column for pair in FUTURE_ATTACK_HASH_PAIRS for column in pair
)
PHASE1_CSV_SCHEMAS = {
    "data_condition_audit.csv": _Phase1CsvSchema(("condition_id",)),
    "future_label_attack_cases.csv": _Phase1CsvSchema(
        ("prefix_checkups",),
        _column_tolerances(
            ("maximum_absolute_score_change_pp",),
            absolute=PHASE1_AUDIT_RESIDUAL_ABSOLUTE_TOLERANCE,
            relative=PHASE1_AUDIT_RESIDUAL_RELATIVE_TOLERANCE,
        ),
        _FUTURE_ATTACK_HASH_COLUMNS,
        FUTURE_ATTACK_HASH_PAIRS,
    ),
    "independent_metric_audit.csv": _Phase1CsvSchema(
        (
            "scenario",
            "fold_id",
            "target_condition_id",
            "prefix_checkups",
            "method",
        ),
        _column_tolerances(
            _INDEPENDENT_SOLVER_COLUMNS,
            absolute=PHASE1_SOLVER_ABSOLUTE_TOLERANCE,
            relative=PHASE1_SOLVER_RELATIVE_TOLERANCE,
        )
        + _column_tolerances(
            _INDEPENDENT_AUDIT_RESIDUAL_COLUMNS,
            absolute=PHASE1_AUDIT_RESIDUAL_ABSOLUTE_TOLERANCE,
            relative=PHASE1_AUDIT_RESIDUAL_RELATIVE_TOLERANCE,
        ),
    ),
    "ablation_audit.csv": _Phase1CsvSchema(
        ("ablation", "scenario", "prefix_checkups"),
        _column_tolerances(
            (
                "candidate_iae_pp_mean",
                "comparator_iae_pp_mean",
                "mean_delta_iae_pp",
            ),
            absolute=PHASE1_SOLVER_ABSOLUTE_TOLERANCE,
            relative=PHASE1_SOLVER_RELATIVE_TOLERANCE,
        )
        + _column_tolerances(
            ("relative_improvement_fraction",),
            absolute=PHASE1_FRACTION_ABSOLUTE_TOLERANCE,
            relative=PHASE1_FRACTION_RELATIVE_TOLERANCE,
        ),
    ),
    "gate_boundary_cases.csv": _Phase1CsvSchema(("case",)),
    "failure_condition_table.csv": _Phase1CsvSchema(
        (
            "scenario",
            "fold_id",
            "target_condition_id",
            "prefix_checkups",
        ),
        _column_tolerances(
            _FAILURE_SOLVER_COLUMNS,
            absolute=PHASE1_SOLVER_ABSOLUTE_TOLERANCE,
            relative=PHASE1_SOLVER_RELATIVE_TOLERANCE,
        )
        + _column_tolerances(
            ("primary_vs_v2_relative_improvement_fraction",),
            absolute=PHASE1_FRACTION_ABSOLUTE_TOLERANCE,
            relative=PHASE1_FRACTION_RELATIVE_TOLERANCE,
        ),
    ),
}


def _requires_exact_numeric_value(column: str) -> bool:
    return column == "prefix_checkups" or column.endswith(("_count", "_index"))


class ReproductionError(RuntimeError):
    """A reproducibility check failed without publishing partial output."""


def _rmtree(path: Path) -> None:
    """Remove one tree, retrying Windows read-only entries within that tree."""
    if not path.exists():
        return
    root = path.resolve()

    def retry_readonly(function: Any, failing_path: str, exc_info: Any) -> None:
        candidate = Path(failing_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise OSError(
                f"Refusing to change permissions outside cleanup root: {candidate}"
            ) from exc_info[1]
        os.chmod(failing_path, stat.S_IWRITE)
        function(failing_path)

    shutil.rmtree(path, onerror=retry_readonly)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locked_versions(path: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split("==")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ReproductionError(
                f"Invalid exact constraint at {path}:{line_number}: {raw_line}"
            )
        distribution, version = (part.strip() for part in parts)
        if distribution in locked:
            raise ReproductionError(
                f"Duplicate reproduction constraint: {distribution}"
            )
        locked[distribution] = version
    if not locked:
        raise ReproductionError(f"No reproduction constraints found: {path}")
    return locked


def _tracked_files(project_root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReproductionError(f"Git tracked-file preflight failed: {detail}")
    return {
        value
        for value in completed.stdout.decode("utf-8", errors="surrogateescape").split(
            "\0"
        )
        if value
    }


def _clean_git_head(project_root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if head.returncode != 0:
        raise ReproductionError("Cannot resolve the Git commit for reproduction")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if status.returncode != 0:
        raise ReproductionError("Cannot inspect the Git work tree for reproduction")
    if status.stdout.strip():
        raise ReproductionError(
            "Tracked or untracked files are modified; reproduce from a clean "
            "committed checkout"
        )
    return head.stdout.strip()


def _release_verification(project_root: Path) -> dict[str, object]:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.verify_public_release import verify

    result = verify(project_root)
    if result["status"] != "passed":
        raise ReproductionError(
            "Public release verification failed: "
            + json.dumps(result, ensure_ascii=False, sort_keys=True)
        )
    return result


def _preflight(project_root: Path, mode: str) -> dict[str, object]:
    if sys.version_info[:2] != REPRODUCTION_PYTHON:
        raise ReproductionError(
            "Canonical reproduction requires Python "
            f"{REPRODUCTION_PYTHON[0]}.{REPRODUCTION_PYTHON[1]}.x; "
            f"found {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    manifest_path = project_root / "release_manifest.json"
    if not manifest_path.is_file():
        raise ReproductionError(f"Missing release manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_paths = (
        PHASE8_RUNNER,
        PHASE8_ANALYZER,
        V012_ANALYZER,
        PHASE1_AUDIT_RUNNER,
        LANDMARK_RUNNER,
        V4_RUNNER,
        GEISBAUER_RUNNER,
        V4_CALIBRATION_ROBUSTNESS_RUNNER,
        GEISBAUER_ROBUSTNESS_RUNNER,
        PHASE8_INPUT,
        PHASE8_CONFIG,
        LANDMARK_CONFIG,
        V4_CONFIG,
        GEISBAUER_CONFIG,
        V4_CALIBRATION_ROBUSTNESS_CONFIG,
        GEISBAUER_ROBUSTNESS_CONFIG,
        GEISBAUER_INPUT,
        REPRODUCTION_CONSTRAINTS,
    )
    missing_paths = [
        relative.as_posix()
        for relative in required_paths
        if not (project_root / relative).is_file()
    ]
    if missing_paths:
        raise ReproductionError(
            "Missing reproduction inputs: " + ", ".join(missing_paths)
        )
    tracked = _tracked_files(project_root)
    git_head = _clean_git_head(project_root)
    required_tracked = {"release_manifest.json", *manifest["frozen_files_sha256"]}
    missing_from_git = sorted(required_tracked - tracked)
    if missing_from_git:
        raise ReproductionError(
            "Release manifest or frozen files are not Git tracked: "
            + ", ".join(missing_from_git)
        )
    constraints_relative = REPRODUCTION_CONSTRAINTS.as_posix()
    if constraints_relative not in manifest["frozen_files_sha256"]:
        raise ReproductionError(
            "Reproduction constraints are not frozen in release_manifest.json"
        )

    required_packages = {
        "duckdb",
        "jsonschema",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
    }
    if mode in {"experiment", "full"}:
        required_packages.add("matplotlib")
    if mode in {"tests", "full"}:
        required_packages.add("pytest")
    missing_packages: list[str] = []
    package_versions: dict[str, str] = {}
    locked_versions = _locked_versions(project_root / REPRODUCTION_CONSTRAINTS)
    for distribution in sorted(required_packages):
        if find_spec(PACKAGE_IMPORTS[distribution]) is None:
            missing_packages.append(distribution)
            continue
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            missing_packages.append(distribution)
    if missing_packages:
        raise ReproductionError(
            "Missing dependencies: "
            + ", ".join(missing_packages)
            + ". Install them with: python -m pip install -c "
            + 'requirements/reproduction.txt -e ".[dev,showcase]"'
        )
    unlocked = sorted(required_packages - set(locked_versions))
    if unlocked:
        raise ReproductionError(
            "Required packages missing from reproduction constraints: "
            + ", ".join(unlocked)
        )
    version_mismatches = {
        distribution: {
            "required": locked_versions[distribution],
            "installed": package_versions[distribution],
        }
        for distribution in sorted(required_packages)
        if package_versions[distribution] != locked_versions[distribution]
    }
    if version_mismatches:
        raise ReproductionError(
            "Installed packages do not match the frozen reproduction constraints: "
            + json.dumps(version_mismatches, sort_keys=True)
        )

    return {
        "python": sys.version,
        "platform": sys.platform,
        "mode": mode,
        "git_tracked_file_count": len(tracked),
        "git_head": git_head,
        "git_worktree_clean": True,
        "release_manifest_tracked": True,
        "frozen_files_tracked": len(manifest["frozen_files_sha256"]),
        "packages": package_versions,
        "locked_package_versions": {
            distribution: locked_versions[distribution]
            for distribution in sorted(required_packages)
        },
        "reproduction_constraints_sha256": _sha256(
            project_root / REPRODUCTION_CONSTRAINTS
        ),
    }


def _command_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = str(project_root / "src")
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        environment[variable] = "1"
    return environment


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    result: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:].strip()
        raise ReproductionError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{detail}"
        )
    return result


def _write_command_log(staging: Path, name: str, result: dict[str, Any]) -> None:
    log_dir = staging / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for stream in ("stdout", "stderr"):
        (log_dir / f"{name}.{stream}.txt").write_text(
            str(result[stream]), encoding="utf-8"
        )


def _csv_content(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def _csv_semantically_equal(
    published_rows: list[list[str]],
    generated_rows: list[list[str]],
    *,
    volatile_sha256_columns: frozenset[str] = frozenset(),
) -> bool:
    if len(published_rows) != len(generated_rows):
        return False
    if not published_rows:
        return True
    if published_rows[0] != generated_rows[0]:
        return False
    header = published_rows[0]
    if any(
        len(row) != len(header) for row in (*published_rows[1:], *generated_rows[1:])
    ):
        return False
    volatile_indexes = {
        index
        for index, column in enumerate(header)
        if column in volatile_sha256_columns
    }
    if volatile_sha256_columns - set(header):
        return False
    for index in volatile_indexes:
        published_topology = _sha256_equivalence_topology(published_rows[1:], index)
        generated_topology = _sha256_equivalence_topology(generated_rows[1:], index)
        if (
            published_topology is None
            or generated_topology is None
            or published_topology != generated_topology
        ):
            return False
    for published_row, generated_row in zip(
        published_rows[1:], generated_rows[1:], strict=True
    ):
        if len(published_row) != len(generated_row):
            return False
        for index, (published_value, generated_value) in enumerate(
            zip(published_row, generated_row, strict=True)
        ):
            if index in volatile_indexes:
                continue
            if published_value == generated_value:
                continue
            if _requires_exact_numeric_value(header[index]):
                return False
            try:
                published_number = float(published_value)
                generated_number = float(generated_value)
            except ValueError:
                return False
            if math.isnan(published_number) and math.isnan(generated_number):
                continue
            if not math.isclose(
                published_number,
                generated_number,
                rel_tol=NUMERIC_RELATIVE_TOLERANCE,
                abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
            ):
                return False
    return True


def _sha256_equivalence_topology(
    rows: list[list[str]],
    column_index: int,
) -> list[int | None] | None:
    classes: dict[str, int] = {}
    topology: list[int | None] = []
    for row in rows:
        value = row[column_index]
        if value == "":
            topology.append(None)
            continue
        if not SHA256_PATTERN.fullmatch(value):
            return None
        if value not in classes:
            classes[value] = len(classes)
        topology.append(classes[value])
    return topology


def _paired_sha256_columns_valid(
    rows: list[list[str]],
    pairs: tuple[tuple[str, str], ...],
) -> bool:
    if not rows:
        return False
    header = rows[0]
    try:
        indexes = [(header.index(left), header.index(right)) for left, right in pairs]
    except ValueError:
        return False
    for row in rows[1:]:
        for left_index, right_index in indexes:
            left = row[left_index]
            right = row[right_index]
            if left == right == "":
                continue
            if left != right or not SHA256_PATTERN.fullmatch(left):
                return False
    return True


def _phase1_csv_mismatch(
    reason: str,
    *,
    key: dict[str, str] | None,
    column: str,
    published_value: object,
    generated_value: object,
    delta: float | None = None,
    tolerance: object = None,
) -> dict[str, object]:
    return {
        "reason": reason,
        "key": key,
        "column": column,
        "values": {
            "published": published_value,
            "generated": generated_value,
        },
        "delta": delta,
        "tolerance": tolerance or {"mode": "exact"},
    }


def _phase1_row_invariant_mismatch(
    filename: str,
    indexed_rows: dict[tuple[str, ...], list[str]],
    indexes: dict[str, int],
    key_columns: tuple[str, ...],
    *,
    side: str,
) -> dict[str, object] | None:
    """Validate arithmetic and routing witnesses within one audit table."""

    def side_payload(payload: object) -> tuple[object, object]:
        return (payload, None) if side == "published" else (None, payload)

    failure_rank: dict[tuple[str, ...], float] = {}
    failure_group_size: dict[tuple[str, ...], int] = {}
    failure_occurrences: dict[tuple[str, str], int] = {}
    failure_minimum_temperature = 0.0
    failure_maximum_temperature = 0.0
    if filename == "failure_condition_table.csv":
        grouped_keys: dict[tuple[str, str], list[tuple[str, ...]]] = {}
        occurrence_scenarios: dict[tuple[str, str], set[str]] = {}
        temperatures: list[float] = []
        for key, row in indexed_rows.items():
            group = (
                row[indexes["scenario"]],
                row[indexes["prefix_checkups"]],
            )
            grouped_keys.setdefault(group, []).append(key)
            occurrence = (
                row[indexes["prefix_checkups"]],
                row[indexes["target_condition_id"]],
            )
            occurrence_scenarios.setdefault(occurrence, set()).add(
                row[indexes["scenario"]]
            )
            temperatures.append(float(row[indexes["temperature_c"]]))
        for keys in grouped_keys.values():
            values = {
                key: float(indexed_rows[key][indexes["candidate_trajectory_iae_pp"]])
                for key in keys
            }
            for key, value in values.items():
                failure_rank[key] = 1.0 + sum(
                    other > value for other in values.values()
                )
                failure_group_size[key] = len(keys)
        failure_occurrences = {
            key: len(scenarios) for key, scenarios in occurrence_scenarios.items()
        }
        failure_minimum_temperature = min(temperatures)
        failure_maximum_temperature = max(temperatures)

    for key in sorted(indexed_rows):
        row = indexed_rows[key]
        key_report = dict(zip(key_columns, key, strict=True))

        def number(column: str) -> float:
            return float(row[indexes[column]])

        def numeric_check(
            column: str,
            expected: float,
            *,
            absolute: float = PHASE1_AUDIT_RESIDUAL_ABSOLUTE_TOLERANCE,
        ) -> dict[str, object] | None:
            actual = number(column)
            if math.isclose(actual, expected, rel_tol=0.0, abs_tol=absolute):
                return None
            published_value, generated_value = side_payload(
                {"actual": row[indexes[column]], "expected": expected}
            )
            return _phase1_csv_mismatch(
                "row_invariant_mismatch",
                key=key_report,
                column=column,
                published_value=published_value,
                generated_value=generated_value,
                delta=actual - expected,
                tolerance={"absolute": absolute, "relative": 0.0},
            )

        def exact_check(
            column: str,
            expected: str,
        ) -> dict[str, object] | None:
            actual = row[indexes[column]]
            if actual == expected:
                return None
            published_value, generated_value = side_payload(
                {"actual": actual, "expected": expected}
            )
            return _phase1_csv_mismatch(
                "row_invariant_mismatch",
                key=key_report,
                column=column,
                published_value=published_value,
                generated_value=generated_value,
                tolerance={"mode": "derived_exact"},
            )

        checks: list[dict[str, object] | None] = []
        if filename == "independent_metric_audit.csv":
            for official, recomputed, difference in _INDEPENDENT_METRIC_RELATIONSHIPS:
                checks.append(
                    numeric_check(
                        difference,
                        number(official) - number(recomputed),
                    )
                )
        elif filename == "ablation_audit.csv":
            candidate = number("candidate_iae_pp_mean")
            comparator = number("comparator_iae_pp_mean")
            if comparator == 0.0:
                published_value, generated_value = side_payload(
                    {"actual": row[indexes["comparator_iae_pp_mean"]]}
                )
                return _phase1_csv_mismatch(
                    "zero_invariant_denominator",
                    key=key_report,
                    column="comparator_iae_pp_mean",
                    published_value=published_value,
                    generated_value=generated_value,
                    tolerance={"mode": "nonzero_denominator"},
                )
            checks.extend(
                [
                    numeric_check("mean_delta_iae_pp", candidate - comparator),
                    numeric_check(
                        "relative_improvement_fraction",
                        (comparator - candidate) / comparator,
                    ),
                    numeric_check(
                        "condition_count",
                        number("candidate_better_condition_count")
                        + number("candidate_worse_condition_count")
                        + number("candidate_equal_condition_count"),
                        absolute=0.0,
                    ),
                ]
            )
        elif filename == "failure_condition_table.csv":
            candidate_iae = number("candidate_trajectory_iae_pp")
            comparator_iae = number("comparator_trajectory_iae_pp")
            prefix_end_days = number("prefix_end_days")
            for denominator_column, denominator in (
                ("comparator_trajectory_iae_pp", comparator_iae),
                ("prefix_end_days", prefix_end_days),
            ):
                if denominator == 0.0:
                    published_value, generated_value = side_payload(
                        {"actual": row[indexes[denominator_column]]}
                    )
                    return _phase1_csv_mismatch(
                        "zero_invariant_denominator",
                        key=key_report,
                        column=denominator_column,
                        published_value=published_value,
                        generated_value=generated_value,
                        tolerance={"mode": "nonzero_denominator"},
                    )
            primary_delta = candidate_iae - comparator_iae
            gate_ready = row[indexes["activation_gate_ready"]] == "True"
            component_selected = row[indexes["activation_component_selected"]] == "True"
            negative_loss_evidence = row[indexes["negative_loss_evidence"]] == "True"
            positive_count = number("positive_time_observation_count")
            occurrence_key = (
                row[indexes["prefix_checkups"]],
                row[indexes["target_condition_id"]],
            )
            expected_occurrences = failure_occurrences[occurrence_key]
            expected_rank = failure_rank[key]
            expected_top_quartile = expected_rank <= math.ceil(
                failure_group_size[key] * 0.25
            )
            temperature = number("temperature_c")
            expected_outside_hull = "temperature" in row[
                indexes["scenario"]
            ] and temperature in (
                failure_minimum_temperature,
                failure_maximum_temperature,
            )
            ungated_delta = number("ungated_target_trajectory_iae_pp") - comparator_iae
            gated_hierarchical_delta = (
                number("gated_hierarchical_trajectory_iae_pp") - comparator_iae
            )
            checks.extend(
                [
                    numeric_check(
                        "candidate_final_error_pp",
                        number("candidate_final_predicted_retention_pct")
                        - number("final_true_retention_pct"),
                    ),
                    numeric_check(
                        "comparator_final_error_pp",
                        number("comparator_final_predicted_retention_pct")
                        - number("final_true_retention_pct"),
                    ),
                    numeric_check("primary_vs_v2_delta_iae_pp", primary_delta),
                    numeric_check(
                        "primary_vs_v2_relative_improvement_fraction",
                        -primary_delta / comparator_iae,
                    ),
                    numeric_check(
                        "ungated_target_vs_v2_delta_iae_pp",
                        ungated_delta,
                    ),
                    numeric_check(
                        "gated_hierarchical_vs_v2_delta_iae_pp",
                        gated_hierarchical_delta,
                    ),
                    numeric_check(
                        "candidate_final_absolute_error_pp",
                        abs(number("candidate_final_error_pp")),
                    ),
                    numeric_check(
                        "comparator_final_absolute_error_pp",
                        abs(number("comparator_final_error_pp")),
                    ),
                    numeric_check(
                        "horizon_to_prefix_time_ratio",
                        number("validation_horizon_days") / prefix_end_days,
                    ),
                    numeric_check(
                        "scenario_occurrence_count",
                        float(expected_occurrences),
                        absolute=0.0,
                    ),
                    numeric_check(
                        "candidate_error_rank_desc",
                        expected_rank,
                        absolute=0.0,
                    ),
                    exact_check(
                        "is_primary_prefix",
                        "True" if number("prefix_checkups") == 10 else "False",
                    ),
                    exact_check(
                        "selected_branch",
                        (
                            "target_activation_specialist"
                            if component_selected
                            else "hierarchical_v2_fallback"
                        ),
                    ),
                    exact_check(
                        "activation_component_selected",
                        "True" if gate_ready else "False",
                    ),
                    exact_check(
                        "activation_gate_ready",
                        (
                            "True"
                            if positive_count >= 7 and negative_loss_evidence
                            else "False"
                        ),
                    ),
                    exact_check(
                        "gate_evidence_gap",
                        (
                            "none"
                            if gate_ready
                            else (
                                "insufficient_positive_time_observations"
                                if positive_count < 7
                                else "negative_loss_evidence_absent"
                            )
                        ),
                    ),
                    exact_check(
                        "independent_evidence_key",
                        row[indexes["target_condition_id"]],
                    ),
                    exact_check(
                        "duplicated_across_scenarios",
                        "True" if expected_occurrences > 1 else "False",
                    ),
                    exact_check(
                        "candidate_error_top_quartile",
                        "True" if expected_top_quartile else "False",
                    ),
                    exact_check(
                        "temperature_outside_training_hull",
                        "True" if expected_outside_hull else "False",
                    ),
                ]
            )
            if primary_delta > PHASE1_FAILURE_CLASSIFICATION_TOLERANCE:
                outcome_class = "relative_regression"
            elif abs(primary_delta) <= PHASE1_FAILURE_CLASSIFICATION_TOLERANCE:
                outcome_class = "exact_v2_fallback"
            else:
                outcome_class = "retrospective_improvement"
            checks.append(exact_check("outcome_class", outcome_class))
            observed_failure = primary_delta > (
                PHASE1_FAILURE_CLASSIFICATION_TOLERANCE
            ) or (
                gate_ready and primary_delta >= -PHASE1_FAILURE_CLASSIFICATION_TOLERANCE
            )
            if observed_failure:
                trust_status = "observed_relative_failure"
            elif not gate_ready:
                trust_status = "fallback_only_no_v3_evidence"
            else:
                trust_status = "development_signal_requires_external_validation"
            checks.append(exact_check("trust_status", trust_status))

            risk_flags = [
                "retrospective_post_hoc",
                "condition_mean_not_cell_level",
            ]
            if not gate_ready:
                risk_flags.append("specialist_not_activated")
            if outcome_class == "relative_regression":
                risk_flags.append("primary_regression_vs_v2")
            elif outcome_class == "exact_v2_fallback":
                risk_flags.append("fallback_same_as_v2")
            else:
                risk_flags.append("retrospective_improvement_signal")
            if gate_ready and primary_delta >= -PHASE1_FAILURE_CLASSIFICATION_TOLERANCE:
                risk_flags.append("gate_triggered_without_trajectory_gain")
            if (
                number("candidate_final_absolute_error_pp")
                > number("comparator_final_absolute_error_pp")
                + PHASE1_FAILURE_CLASSIFICATION_TOLERANCE
            ):
                risk_flags.append("candidate_worse_at_final_checkup")
            if number("horizon_to_prefix_time_ratio") > 5.0:
                risk_flags.append("long_horizon_from_short_prefix")
            if expected_top_quartile:
                risk_flags.append(
                    "top_quartile_absolute_error_within_scenario_landmark"
                )
            if expected_outside_hull:
                risk_flags.append("temperature_outside_training_convex_hull")
            if expected_occurrences > 1:
                risk_flags.append("scenario_duplicate_not_independent_evidence")
            if ungated_delta > PHASE1_FAILURE_CLASSIFICATION_TOLERANCE:
                risk_flags.append("ungated_specialist_would_regress")
            if gated_hierarchical_delta > PHASE1_FAILURE_CLASSIFICATION_TOLERANCE:
                risk_flags.append("alternative_hierarchical_gate_regresses")
            checks.append(exact_check("risk_flags", ";".join(risk_flags)))

            if trust_status == "observed_relative_failure":
                recommended_action = (
                    "Do not use the candidate; diagnose the specialist and retain V2."
                )
            elif not gate_ready:
                recommended_action = (
                    "Retain the V2 fallback and collect denser early "
                    "condition-level data."
                )
            else:
                recommended_action = (
                    "Freeze the rule and test an independent cell-level cohort "
                    "before use."
                )
            checks.append(exact_check("recommended_action", recommended_action))

            if not gate_ready:
                for candidate_column, comparator_column in (
                    (
                        "candidate_trajectory_iae_pp",
                        "comparator_trajectory_iae_pp",
                    ),
                    (
                        "candidate_future_point_mae_pp",
                        "comparator_future_point_mae_pp",
                    ),
                    (
                        "candidate_final_predicted_retention_pct",
                        "comparator_final_predicted_retention_pct",
                    ),
                    ("candidate_final_error_pp", "comparator_final_error_pp"),
                ):
                    checks.append(
                        numeric_check(candidate_column, number(comparator_column))
                    )

        for mismatch in checks:
            if mismatch is not None:
                return mismatch
    return None


def _phase1_csv_semantic_comparison(
    filename: str,
    published_rows: list[list[str]],
    generated_rows: list[list[str]],
) -> dict[str, object]:
    """Compare a Phase 1 CSV by its stable row identity and column policy."""
    try:
        schema = PHASE1_CSV_SCHEMAS[filename]
    except KeyError as exc:
        raise ReproductionError(
            f"No Phase 1 CSV schema declared for {filename}"
        ) from exc

    tolerances = {
        column: (absolute, relative)
        for column, absolute, relative in schema.tolerant_columns
    }
    tolerance_report = {
        column: {"absolute": absolute, "relative": relative}
        for column, (absolute, relative) in tolerances.items()
    }

    def result(
        mismatch: dict[str, object] | None,
    ) -> dict[str, object]:
        return {
            "semantic_content_equal": mismatch is None,
            "exact_by_default": True,
            "key_columns": list(schema.key_columns),
            "numeric_tolerances": tolerance_report,
            "solver_calibration_max_observed_absolute_delta_pp": (
                PHASE1_SOLVER_MAX_OBSERVED_ABSOLUTE_DELTA_PP
                if any(
                    absolute == PHASE1_SOLVER_ABSOLUTE_TOLERANCE
                    for absolute, _relative in tolerances.values()
                )
                else None
            ),
            "cross_platform_volatile_sha256_columns": sorted(
                schema.volatile_sha256_columns
            ),
            "mismatch": mismatch,
        }

    if not published_rows or not generated_rows:
        return result(
            _phase1_csv_mismatch(
                "missing_header",
                key=None,
                column="__header__",
                published_value=published_rows[0] if published_rows else None,
                generated_value=generated_rows[0] if generated_rows else None,
            )
        )

    published_header = published_rows[0]
    generated_header = generated_rows[0]
    if published_header != generated_header:
        return result(
            _phase1_csv_mismatch(
                "header_drift",
                key=None,
                column="__header__",
                published_value=published_header,
                generated_value=generated_header,
            )
        )
    header = published_header
    if len(header) != len(set(header)) or "" in header:
        return result(
            _phase1_csv_mismatch(
                "invalid_header",
                key=None,
                column="__header__",
                published_value=header,
                generated_value=generated_header,
            )
        )

    required_columns = (
        set(schema.key_columns)
        | set(tolerances)
        | set(schema.volatile_sha256_columns)
        | {column for pair in schema.sha256_pairs for column in pair}
    )
    missing_columns = sorted(required_columns - set(header))
    if missing_columns:
        return result(
            _phase1_csv_mismatch(
                "missing_schema_column",
                key=None,
                column=missing_columns[0],
                published_value="missing",
                generated_value="missing",
            )
        )

    indexes = {column: index for index, column in enumerate(header)}

    for side, rows in (
        ("published", published_rows[1:]),
        ("generated", generated_rows[1:]),
    ):
        for row_number, row in enumerate(rows, start=2):
            if len(row) != len(header):
                return result(
                    _phase1_csv_mismatch(
                        "row_width_mismatch",
                        key=None,
                        column="__row__",
                        published_value=(
                            {"row_number": row_number, "width": len(row)}
                            if side == "published"
                            else None
                        ),
                        generated_value=(
                            {"row_number": row_number, "width": len(row)}
                            if side == "generated"
                            else None
                        ),
                    )
                )

    def index_rows(
        rows: list[list[str]], side: str
    ) -> tuple[dict[tuple[str, ...], list[str]], dict[str, object] | None]:
        indexed: dict[tuple[str, ...], list[str]] = {}
        for row_number, row in enumerate(rows[1:], start=2):
            key = tuple(row[indexes[column]] for column in schema.key_columns)
            key_report = dict(zip(schema.key_columns, key, strict=True))
            for column, value in zip(schema.key_columns, key, strict=True):
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    return indexed, _phase1_csv_mismatch(
                        "non_finite_numeric_value",
                        key=key_report,
                        column=column,
                        published_value=value if side == "published" else None,
                        generated_value=value if side == "generated" else None,
                        tolerance={"mode": "finite_values_only"},
                    )
            if any(value == "" for value in key):
                missing_column = schema.key_columns[key.index("")]
                return indexed, _phase1_csv_mismatch(
                    "missing_key_value",
                    key=key_report,
                    column=missing_column,
                    published_value=(
                        {"row_number": row_number, "value": ""}
                        if side == "published"
                        else None
                    ),
                    generated_value=(
                        {"row_number": row_number, "value": ""}
                        if side == "generated"
                        else None
                    ),
                )
            if key in indexed:
                return indexed, _phase1_csv_mismatch(
                    "duplicate_key",
                    key=key_report,
                    column="__key__",
                    published_value=("duplicate" if side == "published" else None),
                    generated_value=("duplicate" if side == "generated" else None),
                )
            indexed[key] = row
        return indexed, None

    published_index, mismatch = index_rows(published_rows, "published")
    if mismatch is not None:
        return result(mismatch)
    generated_index, mismatch = index_rows(generated_rows, "generated")
    if mismatch is not None:
        return result(mismatch)

    published_keys = set(published_index)
    generated_keys = set(generated_index)
    if published_keys != generated_keys:
        missing_generated = sorted(published_keys - generated_keys)
        unexpected_generated = sorted(generated_keys - published_keys)
        if missing_generated:
            key = missing_generated[0]
            published_value, generated_value = "present", "missing"
        else:
            key = unexpected_generated[0]
            published_value, generated_value = "missing", "present"
        return result(
            _phase1_csv_mismatch(
                "key_set_mismatch",
                key=dict(zip(schema.key_columns, key, strict=True)),
                column="__key__",
                published_value=published_value,
                generated_value=generated_value,
            )
        )

    ordered_keys = sorted(published_keys)
    for key in ordered_keys:
        key_report = dict(zip(schema.key_columns, key, strict=True))
        published_row = published_index[key]
        generated_row = generated_index[key]
        for column, published_value, generated_value in zip(
            header, published_row, generated_row, strict=True
        ):
            for value in (published_value, generated_value):
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    return result(
                        _phase1_csv_mismatch(
                            "non_finite_numeric_value",
                            key=key_report,
                            column=column,
                            published_value=published_value,
                            generated_value=generated_value,
                            tolerance={"mode": "finite_values_only"},
                        )
                    )

    for side, indexed_rows in (
        ("published", published_index),
        ("generated", generated_index),
    ):
        mismatch = _phase1_row_invariant_mismatch(
            filename,
            indexed_rows,
            indexes,
            schema.key_columns,
            side=side,
        )
        if mismatch is not None:
            return result(mismatch)

    for key in ordered_keys:
        key_report = dict(zip(schema.key_columns, key, strict=True))
        published_row = published_index[key]
        generated_row = generated_index[key]
        for left_column, right_column in schema.sha256_pairs:
            left_index = indexes[left_column]
            right_index = indexes[right_column]
            published_pair = (
                published_row[left_index],
                published_row[right_index],
            )
            generated_pair = (
                generated_row[left_index],
                generated_row[right_index],
            )
            for side, pair in (
                ("published", published_pair),
                ("generated", generated_pair),
            ):
                valid = pair == ("", "") or (
                    pair[0] == pair[1] and SHA256_PATTERN.fullmatch(pair[0]) is not None
                )
                if not valid:
                    return result(
                        _phase1_csv_mismatch(
                            "invalid_sha256_pair",
                            key=key_report,
                            column=f"{left_column},{right_column}",
                            published_value=(
                                published_pair if side == "published" else None
                            ),
                            generated_value=(
                                generated_pair if side == "generated" else None
                            ),
                            tolerance={"mode": "equal_sha256_or_both_blank"},
                        )
                    )

    volatile_columns = tuple(
        column for column in header if column in schema.volatile_sha256_columns
    )

    def sha256_topology(
        indexed_rows: dict[tuple[str, ...], list[str]],
    ) -> list[int | None]:
        classes: dict[str, int] = {}
        topology: list[int | None] = []
        for key in ordered_keys:
            for column in volatile_columns:
                value = indexed_rows[key][indexes[column]]
                if value == "":
                    topology.append(None)
                    continue
                if value not in classes:
                    classes[value] = len(classes)
                topology.append(classes[value])
        return topology

    published_topology = sha256_topology(published_index)
    generated_topology = sha256_topology(generated_index)
    if published_topology != generated_topology:
        mismatch_index = next(
            index
            for index, values in enumerate(
                zip(published_topology, generated_topology, strict=True)
            )
            if values[0] != values[1]
        )
        key_index, column_index = divmod(mismatch_index, len(volatile_columns))
        key = ordered_keys[key_index]
        column = volatile_columns[column_index]
        value_index = indexes[column]
        return result(
            _phase1_csv_mismatch(
                "sha256_topology_mismatch",
                key=dict(zip(schema.key_columns, key, strict=True)),
                column=column,
                published_value=published_index[key][value_index],
                generated_value=generated_index[key][value_index],
                tolerance={"mode": "global_equivalent_sha256_topology"},
            )
        )

    for key in ordered_keys:
        key_report = dict(zip(schema.key_columns, key, strict=True))
        published_row = published_index[key]
        generated_row = generated_index[key]
        for column, published_value, generated_value in zip(
            header, published_row, generated_row, strict=True
        ):
            if column in schema.volatile_sha256_columns:
                continue
            if column not in tolerances:
                if published_value != generated_value:
                    return result(
                        _phase1_csv_mismatch(
                            "exact_value_mismatch",
                            key=key_report,
                            column=column,
                            published_value=published_value,
                            generated_value=generated_value,
                        )
                    )
                continue

            absolute, relative = tolerances[column]
            tolerance = {"absolute": absolute, "relative": relative}
            try:
                published_number = float(published_value)
                generated_number = float(generated_value)
            except ValueError:
                return result(
                    _phase1_csv_mismatch(
                        "invalid_numeric_value",
                        key=key_report,
                        column=column,
                        published_value=published_value,
                        generated_value=generated_value,
                        tolerance=tolerance,
                    )
                )
            delta = generated_number - published_number
            allowed_delta = max(
                absolute,
                relative * max(abs(published_number), abs(generated_number)),
            )
            if not math.isclose(
                published_number,
                generated_number,
                rel_tol=relative,
                abs_tol=absolute,
            ):
                return result(
                    _phase1_csv_mismatch(
                        "numeric_value_mismatch",
                        key=key_report,
                        column=column,
                        published_value=published_value,
                        generated_value=generated_value,
                        delta=delta,
                        tolerance={**tolerance, "allowed_delta": allowed_delta},
                    )
                )

    return result(None)


def _json_semantically_equal(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _json_semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_semantically_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if isinstance(left, int) or isinstance(right, int):
            return type(left) is type(right) and left == right
        return math.isclose(
            float(left),
            float(right),
            rel_tol=NUMERIC_RELATIVE_TOLERANCE,
            abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
        )
    return left == right


def _compare_core_csvs(
    project_root: Path,
    generated_root: Path,
) -> list[dict[str, object]]:
    manifest = json.loads(
        (project_root / "release_manifest.json").read_text(encoding="utf-8")
    )
    frozen = manifest["frozen_files_sha256"]
    comparisons: list[dict[str, object]] = []
    for filename in CORE_PHASE8_CSVS:
        published_relative = f"showcase/results/{filename}"
        published = project_root / published_relative
        generated = generated_root / filename
        if published_relative not in frozen:
            raise ReproductionError(
                f"Core Phase 8 CSV is not frozen in release_manifest.json: "
                f"{published_relative}"
            )
        if not published.is_file() or not generated.is_file():
            raise ReproductionError(
                f"Missing published or generated Phase 8 CSV: {filename}"
            )
        expected_sha256 = str(frozen[published_relative])
        published_sha256 = _sha256(published)
        generated_sha256 = _sha256(generated)
        byte_equal = published.read_bytes() == generated.read_bytes()
        published_rows = _csv_content(published)
        generated_rows = _csv_content(generated)
        volatile_sha256_columns = (
            STATE_HASH_COLUMNS if filename == "target_diagnostics.csv" else frozenset()
        )
        csv_content_equal = _csv_semantically_equal(
            published_rows,
            generated_rows,
            volatile_sha256_columns=volatile_sha256_columns,
        )
        generated_sha_matches_release = generated_sha256 == expected_sha256
        passed = published_sha256 == expected_sha256 and csv_content_equal
        row_count = max(len(generated_rows) - 1, 0)
        comparison = {
            "path": published_relative,
            "generated_path": f"phase8/{filename}",
            "expected_release_sha256": expected_sha256,
            "published_sha256": published_sha256,
            "generated_sha256": generated_sha256,
            "generated_sha_matches_release": generated_sha_matches_release,
            "byte_equal": byte_equal,
            "csv_content_equal": csv_content_equal,
            "numeric_tolerance": {
                "relative": NUMERIC_RELATIVE_TOLERANCE,
                "absolute": NUMERIC_ABSOLUTE_TOLERANCE,
            },
            "cross_platform_volatile_sha256_columns": sorted(volatile_sha256_columns),
            "row_count": row_count,
            "status": "passed" if passed else "failed",
        }
        comparisons.append(comparison)
        if not passed:
            raise ReproductionError(
                "Phase 8 CSV reproduction mismatch: "
                + json.dumps(comparison, ensure_ascii=False, sort_keys=True)
            )
    return comparisons


def _normalized_v011_result(
    payload: dict[str, object], group: str
) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload, allow_nan=False))
    normalized.pop("provenance", None)
    normalized.pop("artifacts", None)
    if group == "landmark":
        normalized.pop("regenerated_v3_prediction_sha256", None)
    elif group == "v4":
        normalized.pop("prediction_pack_sha256", None)
    elif group == "geisbauer":
        firewall = normalized.get("future_label_firewall")
        if isinstance(firewall, dict):
            firewall.pop("label_free_prediction_sha256", None)
    else:
        raise ReproductionError(f"Unknown V0.11 evidence group: {group}")
    return normalized


def _compare_v011_core_csvs(
    project_root: Path,
    generated_root: Path,
    group: str,
) -> list[dict[str, object]]:
    manifest = json.loads(
        (project_root / "release_manifest.json").read_text(encoding="utf-8")
    )
    frozen = manifest["frozen_files_sha256"]
    comparisons: list[dict[str, object]] = []
    for filename in V011_CORE_CSVS[group]:
        published_relative = (V011_PUBLISHED_ROOT / group / filename).as_posix()
        published = project_root / published_relative
        generated = generated_root / filename
        if published_relative not in frozen:
            raise ReproductionError(
                f"V0.11 evidence CSV is not frozen: {published_relative}"
            )
        if not published.is_file() or not generated.is_file():
            raise ReproductionError(
                f"Missing published or generated V0.11 CSV: {published_relative}"
            )
        expected_sha256 = str(frozen[published_relative])
        published_sha256 = _sha256(published)
        generated_sha256 = _sha256(generated)
        volatile_sha256_columns = V011_VOLATILE_SHA256_COLUMNS.get(
            (group, filename), frozenset()
        )
        semantically_equal = _csv_semantically_equal(
            _csv_content(published),
            _csv_content(generated),
            volatile_sha256_columns=volatile_sha256_columns,
        )
        passed = published_sha256 == expected_sha256 and semantically_equal
        comparison = {
            "path": published_relative,
            "generated_path": (Path("evidence_v011") / group / filename).as_posix(),
            "expected_release_sha256": expected_sha256,
            "published_sha256": published_sha256,
            "generated_sha256": generated_sha256,
            "generated_sha_matches_release": generated_sha256 == expected_sha256,
            "semantic_content_equal": semantically_equal,
            "byte_equal": generated.read_bytes() == published.read_bytes(),
            "numeric_tolerance": {
                "relative": NUMERIC_RELATIVE_TOLERANCE,
                "absolute": NUMERIC_ABSOLUTE_TOLERANCE,
            },
            "cross_platform_volatile_sha256_columns": sorted(volatile_sha256_columns),
            "status": "passed" if passed else "failed",
        }
        comparisons.append(comparison)
        if not passed:
            raise ReproductionError(
                "V0.11 CSV reproduction mismatch: "
                + json.dumps(comparison, ensure_ascii=False, sort_keys=True)
            )
    return comparisons


def _inspect_v011_group(
    project_root: Path,
    generated_root: Path,
    group: str,
) -> dict[str, object]:
    manifest = json.loads(
        (project_root / "release_manifest.json").read_text(encoding="utf-8")
    )
    try:
        result_filename = V011_RESULT_FILES[group]
    except KeyError as exc:
        raise ReproductionError(f"Unknown V0.11 evidence group: {group}") from exc
    published_relative = (V011_PUBLISHED_ROOT / group / result_filename).as_posix()
    published_path = project_root / published_relative
    generated_path = generated_root / result_filename
    if published_relative not in manifest["frozen_files_sha256"]:
        raise ReproductionError(
            f"V0.11 result JSON is not frozen: {published_relative}"
        )
    if not published_path.is_file() or not generated_path.is_file():
        raise ReproductionError(f"Missing published or generated V0.11 result: {group}")
    expected_sha256 = str(manifest["frozen_files_sha256"][published_relative])
    published_sha256 = _sha256(published_path)
    if published_sha256 != expected_sha256:
        raise ReproductionError(
            f"Published V0.11 result hash mismatch: {published_relative}"
        )
    published = json.loads(published_path.read_text(encoding="utf-8"))
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    semantic_content_equal = _json_semantically_equal(
        _normalized_v011_result(published, group),
        _normalized_v011_result(generated, group),
    )
    if not semantic_content_equal:
        raise ReproductionError(
            f"V0.11 result semantics changed for evidence group: {group}"
        )

    if group == "landmark":
        guarded = (
            generated.get("status") == "retrospective_signal_only_confirmation_blocked"
            and generated.get("retrospective_signal_landmark") == 10
            and generated.get("confirmed_earliest_landmark") is None
            and generated.get("model_validation_status") == "not_confirmed"
        )
    elif group == "v4":
        confirmation = generated.get("confirmation", {})
        calibration = generated.get("calibration", {})
        guarded = (
            generated.get("status")
            == "retrospective_hybrid_diagnostic_complete_not_confirmed"
            and confirmation.get("status") == "not_confirmed"
            and confirmation.get("15_to_25_year_claim_allowed") is False
            and calibration.get("operational_issued_trajectory_count") == 0
        )
    else:
        decision = generated.get("decision", {})
        gate = generated.get("mechanism_gate", {})
        comparison = generated.get("primary_comparison", {})
        guarded = (
            generated.get("model_validation_status") == "not_confirmed"
            and generated.get("descriptive_signal_status")
            == "primary_candidate_did_not_outperform_comparator"
            and gate.get("gate_ready_physical_cell_count") == 0
            and gate.get("fallback_physical_cell_count") == 15
            and float(comparison.get("mean_paired_delta_iae_pp", -1.0)) > 0.0
            and decision.get("independent_long_term_validation_claim_allowed") is False
        )
    if not guarded:
        raise ReproductionError(f"V0.11 claim guard failed for evidence group: {group}")
    return {
        "status": "passed",
        "group": group,
        "published_result": published_relative,
        "expected_release_sha256": expected_sha256,
        "published_sha256": published_sha256,
        "generated_sha256": _sha256(generated_path),
        "semantic_content_equal": semantic_content_equal,
        "claim_guard_passed": True,
        "core_csv_comparisons": _compare_v011_core_csvs(
            project_root, generated_root, group
        ),
    }


def _normalized_v012_result(
    payload: dict[str, object], group: str
) -> dict[str, object]:
    if group not in V012_RESULT_FILES:
        raise ReproductionError(f"Unknown V0.12 evidence group: {group}")
    normalized = json.loads(json.dumps(payload, allow_nan=False))
    normalized.pop("provenance", None)
    normalized.pop("artifacts", None)
    if group == "v4_calibration_robustness":
        normalized.pop("candidate_prediction_pack_sha256", None)
    else:
        base_evidence = normalized.get("base_evidence")
        if isinstance(base_evidence, dict):
            base_evidence.pop("label_free_prediction_sha256", None)
    return normalized


def _compare_v012_core_csvs(
    project_root: Path,
    generated_root: Path,
    group: str,
) -> list[dict[str, object]]:
    manifest = json.loads(
        (project_root / "release_manifest.json").read_text(encoding="utf-8")
    )
    frozen = manifest["frozen_files_sha256"]
    try:
        filenames = V012_CORE_CSVS[group]
    except KeyError as exc:
        raise ReproductionError(f"Unknown V0.12 evidence group: {group}") from exc
    comparisons: list[dict[str, object]] = []
    for filename in filenames:
        published_relative = (V012_PUBLISHED_ROOT / group / filename).as_posix()
        published = project_root / published_relative
        generated = generated_root / filename
        if published_relative not in frozen:
            raise ReproductionError(
                f"V0.12 evidence CSV is not frozen: {published_relative}"
            )
        if not published.is_file() or not generated.is_file():
            raise ReproductionError(
                f"Missing published or generated V0.12 CSV: {published_relative}"
            )
        expected_sha256 = str(frozen[published_relative])
        published_sha256 = _sha256(published)
        generated_sha256 = _sha256(generated)
        volatile_sha256_columns = V012_VOLATILE_SHA256_COLUMNS.get(
            (group, filename), frozenset()
        )
        semantically_equal = _csv_semantically_equal(
            _csv_content(published),
            _csv_content(generated),
            volatile_sha256_columns=volatile_sha256_columns,
        )
        passed = published_sha256 == expected_sha256 and semantically_equal
        comparison = {
            "path": published_relative,
            "generated_path": (Path("evidence_v012") / group / filename).as_posix(),
            "expected_release_sha256": expected_sha256,
            "published_sha256": published_sha256,
            "generated_sha256": generated_sha256,
            "generated_sha_matches_release": generated_sha256 == expected_sha256,
            "semantic_content_equal": semantically_equal,
            "byte_equal": generated.read_bytes() == published.read_bytes(),
            "numeric_tolerance": {
                "relative": NUMERIC_RELATIVE_TOLERANCE,
                "absolute": NUMERIC_ABSOLUTE_TOLERANCE,
            },
            "cross_platform_volatile_sha256_columns": sorted(volatile_sha256_columns),
            "status": "passed" if passed else "failed",
        }
        comparisons.append(comparison)
        if not passed:
            raise ReproductionError(
                "V0.12 CSV reproduction mismatch: "
                + json.dumps(comparison, ensure_ascii=False, sort_keys=True)
            )
    return comparisons


def _finite_json_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _has_prohibited_claims(
    payload: dict[str, object], required: frozenset[str]
) -> bool:
    claims = payload.get("prohibited_claims")
    if not isinstance(claims, list):
        return False
    observed = {claim for claim in claims if isinstance(claim, str)}
    return required.issubset(observed)


def _v012_claim_guard(payload: dict[str, object], group: str) -> bool:
    if group == "v4_calibration_robustness":
        design = payload.get("design")
        interpretation = payload.get("interpretation")
        confirmation = payload.get("confirmation")
        if not all(
            isinstance(value, dict) for value in (design, interpretation, confirmation)
        ):
            return False
        return bool(
            payload.get("status")
            == "retrospective_calibration_robustness_complete_not_confirmed"
            and payload.get("execution_status") == "completed"
            and design.get("design_status")
            == "retrospective_protocol_locked_after_v011_result_inspection"
            and design.get("selection_status") == "post_hoc_robustness_audit"
            and design.get("exhaustive_partition_count") == 210
            and design.get("partition_outcomes_are_overlapping") is True
            and design.get("partition_results_are_independent_replications") is False
            and interpretation.get("operational_interval_issued") is False
            and interpretation.get("formal_coverage_claim_allowed") is False
            and interpretation.get("appropriate_use")
            == "retrospective_route_support_and_partition_sensitivity_diagnostic_only"
            and interpretation.get("coverage_fraction_denominator")
            == "overlapping_condition_partition_evaluation_instances"
            and interpretation.get(
                "coverage_fraction_is_effective_independent_sample_estimate"
            )
            is False
            and confirmation.get("status") == "not_confirmed"
            and confirmation.get("independent_long_term_dataset_available") is False
            and confirmation.get("15_to_25_year_claim_allowed") is False
            and _has_prohibited_claims(
                payload,
                frozenset(
                    {
                        "preregistered_or_outcome_blind_analysis",
                        "formal_finite_sample_coverage_on_reused_naumann_data",
                        "independent_partition_replications",
                        "15_to_25_year_extrapolation",
                    }
                ),
            )
        )
    if group == "geisbauer_robustness":
        scope = payload.get("scope")
        route_reality = payload.get("route_reality")
        overall = payload.get("overall_paired_diagnostic")
        leave_one_out = payload.get("leave_one_cell_out")
        diagnosis = payload.get("negative_transfer_diagnosis")
        boundary = payload.get("claim_boundary")
        if not all(
            isinstance(value, dict)
            for value in (
                scope,
                route_reality,
                overall,
                leave_one_out,
                diagnosis,
                boundary,
            )
        ):
            return False
        overall_delta = _finite_json_number(overall.get("mean_paired_delta_pp"))
        minimum_loco = _finite_json_number(
            leave_one_out.get("minimum_mean_paired_delta_pp")
        )
        maximum_loco = _finite_json_number(
            leave_one_out.get("maximum_mean_paired_delta_pp")
        )
        return bool(
            payload.get("status") == "retrospective_external_robustness_audit_complete"
            and payload.get("execution_status") == "completed"
            and payload.get("design_status")
            == "retrospective_audit_designed_after_v011_outcome_review"
            and payload.get("inference_status")
            == "exploratory_nominal_diagnostics_not_confirmatory_inference"
            and scope.get("outcomes_reviewed_before_audit_design") is True
            and scope.get("physical_cell_count") == 15
            and scope.get("storage_temperature_c") == 60.0
            and scope.get("maximum_observed_days") == 120.0
            and route_reality.get(
                "candidate_exactly_equals_hierarchical_power_fallback"
            )
            is True
            and route_reality.get("activation_gate_ready_physical_cell_count") == 0
            and route_reality.get("activation_specialist_tested") is False
            and overall_delta is not None
            and overall_delta > 0.0
            and overall.get("nominal_diagnostics_are_confirmatory") is False
            and diagnosis.get("aggregate_mean_negative_transfer_observed") is True
            and _finite_json_number(
                diagnosis.get("physical_cells_with_negative_transfer")
            )
            is not None
            and int(diagnosis["physical_cells_with_negative_transfer"]) > 0
            and leave_one_out.get("scenario_count") == 15
            and leave_one_out.get("remaining_physical_cell_count_per_scenario") == 14
            and leave_one_out.get("scenarios_are_highly_overlapping") is True
            and leave_one_out.get("scenarios_are_independent_replications") is False
            and minimum_loco is not None
            and minimum_loco < 0.0
            and maximum_loco is not None
            and maximum_loco > 0.0
            and _finite_json_number(
                leave_one_out.get("candidate_better_direction_count")
            )
            is not None
            and int(leave_one_out["candidate_better_direction_count"]) > 0
            and _finite_json_number(
                leave_one_out.get("candidate_worse_direction_count")
            )
            is not None
            and int(leave_one_out["candidate_worse_direction_count"]) > 0
            and boundary.get("model_validation_status") == "not_confirmed"
            and boundary.get("confirmatory_inference_allowed") is False
            and boundary.get("independent_long_term_validation_claim_allowed") is False
            and _has_prohibited_claims(
                payload,
                frozenset(
                    {
                        "outcome_blind_external_validation",
                        "independent_long_term_validation",
                        "confirmatory_p_value",
                        "15_to_25_year_extrapolation",
                    }
                ),
            )
        )
    raise ReproductionError(f"Unknown V0.12 evidence group: {group}")


def _inspect_v012_group(
    project_root: Path,
    generated_root: Path,
    group: str,
) -> dict[str, object]:
    manifest = json.loads(
        (project_root / "release_manifest.json").read_text(encoding="utf-8")
    )
    try:
        result_filename = V012_RESULT_FILES[group]
    except KeyError as exc:
        raise ReproductionError(f"Unknown V0.12 evidence group: {group}") from exc
    published_relative = (V012_PUBLISHED_ROOT / group / result_filename).as_posix()
    published_path = project_root / published_relative
    generated_path = generated_root / result_filename
    if published_relative not in manifest["frozen_files_sha256"]:
        raise ReproductionError(
            f"V0.12 result JSON is not frozen: {published_relative}"
        )
    if not published_path.is_file() or not generated_path.is_file():
        raise ReproductionError(f"Missing published or generated V0.12 result: {group}")
    expected_sha256 = str(manifest["frozen_files_sha256"][published_relative])
    published_sha256 = _sha256(published_path)
    if published_sha256 != expected_sha256:
        raise ReproductionError(
            f"Published V0.12 result hash mismatch: {published_relative}"
        )
    published = json.loads(published_path.read_text(encoding="utf-8"))
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    if not _v012_claim_guard(generated, group):
        raise ReproductionError(f"V0.12 claim guard failed for evidence group: {group}")
    semantic_content_equal = _json_semantically_equal(
        _normalized_v012_result(published, group),
        _normalized_v012_result(generated, group),
    )
    if not semantic_content_equal:
        raise ReproductionError(
            f"V0.12 result semantics changed for evidence group: {group}"
        )
    return {
        "status": "passed",
        "group": group,
        "published_result": published_relative,
        "expected_release_sha256": expected_sha256,
        "published_sha256": published_sha256,
        "generated_sha256": _sha256(generated_path),
        "semantic_content_equal": semantic_content_equal,
        "claim_guard_passed": True,
        "core_csv_comparisons": _compare_v012_core_csvs(
            project_root, generated_root, group
        ),
    }


def _inspect_png(
    path: Path,
    *,
    reported_path: str | Path | None = None,
) -> dict[str, object]:
    payload = path.read_bytes() if path.is_file() else b""
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ReproductionError(f"Headless figure is not a valid PNG: {path}")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ReproductionError(f"Headless figure has invalid dimensions: {path}")
    artifact_path = path if reported_path is None else Path(reported_path)
    return {
        "path": artifact_path.as_posix(),
        "sha256": _sha256(path),
        "byte_count": len(payload),
        "width_px": width,
        "height_px": height,
        "backend": "Agg",
        "status": "passed",
    }


def _inspect_phase1_audit(
    output_dir: Path,
    project_root: Path,
) -> dict[str, object]:
    summary_path = output_dir / "phase1_adversarial_audit.json"
    if not summary_path.is_file():
        raise ReproductionError("Phase 1 audit did not write its summary JSON")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("audit_execution_status") != "passed":
        raise ReproductionError(
            "Phase 1 adversarial audit did not pass: "
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
    if payload.get("model_validation_status") != "not_confirmed":
        raise ReproductionError(
            "Phase 1 audit must retain model_validation_status=not_confirmed"
        )
    missing = [
        filename
        for filename in PHASE1_AUDIT_FILES
        if not (output_dir / filename).is_file()
    ]
    if missing:
        raise ReproductionError(
            "Phase 1 audit output is incomplete: " + ", ".join(missing)
        )
    manifest = json.loads(
        (project_root / "release_manifest.json").read_text(encoding="utf-8")
    )
    frozen = manifest["frozen_files_sha256"]
    files: list[dict[str, object]] = []
    for filename in PHASE1_AUDIT_FILES:
        generated = output_dir / filename
        published_relative = f"showcase/audit_results/{filename}"
        published = project_root / published_relative
        if published_relative not in frozen:
            raise ReproductionError(
                f"Phase 1 audit file is not frozen: {published_relative}"
            )
        if not published.is_file():
            raise ReproductionError(
                f"Published Phase 1 audit file is missing: {published_relative}"
            )
        expected_sha256 = str(frozen[published_relative])
        published_sha256 = _sha256(published)
        generated_sha256 = _sha256(generated)
        volatile_sha256_columns: frozenset[str] = frozenset()
        key_columns: list[str] = []
        semantic_mismatch: dict[str, object] | None = None
        numeric_tolerance: dict[str, object] = {
            "relative": NUMERIC_RELATIVE_TOLERANCE,
            "absolute": NUMERIC_ABSOLUTE_TOLERANCE,
        }
        if filename.endswith(".json"):
            semantically_equal = _json_semantically_equal(
                json.loads(published.read_text(encoding="utf-8")),
                json.loads(generated.read_text(encoding="utf-8")),
            )
        else:
            published_rows = _csv_content(published)
            generated_rows = _csv_content(generated)
            volatile_sha256_columns = (
                frozenset(
                    column for pair in FUTURE_ATTACK_HASH_PAIRS for column in pair
                )
                if filename == "future_label_attack_cases.csv"
                else frozenset()
            )
            csv_comparison = _phase1_csv_semantic_comparison(
                filename, published_rows, generated_rows
            )
            semantically_equal = bool(csv_comparison["semantic_content_equal"])
            key_columns = list(csv_comparison["key_columns"])
            semantic_mismatch = csv_comparison["mismatch"]
            numeric_tolerance = {
                "exact_by_default": True,
                "columns": csv_comparison["numeric_tolerances"],
                "solver_calibration_max_observed_absolute_delta_pp": (
                    csv_comparison["solver_calibration_max_observed_absolute_delta_pp"]
                ),
            }
        passed = published_sha256 == expected_sha256 and semantically_equal
        comparison = {
            "path": published_relative,
            "generated_path": f"phase1_audit/{filename}",
            "expected_release_sha256": expected_sha256,
            "published_sha256": published_sha256,
            "generated_sha256": generated_sha256,
            "generated_sha_matches_release": generated_sha256 == expected_sha256,
            "semantic_content_equal": semantically_equal,
            "numeric_tolerance": numeric_tolerance,
            "key_columns": key_columns,
            "mismatch": semantic_mismatch,
            "cross_platform_volatile_sha256_columns": sorted(volatile_sha256_columns),
            "byte_equal": generated.read_bytes() == published.read_bytes(),
            "byte_count": generated.stat().st_size,
            "status": "passed" if passed else "failed",
        }
        files.append(comparison)
        if not passed:
            raise ReproductionError(
                "Phase 1 audit reproduction mismatch: "
                + json.dumps(comparison, ensure_ascii=False, sort_keys=True)
            )
    return {
        "status": "passed",
        "audit_execution_status": payload["audit_execution_status"],
        "model_validation_status": payload.get("model_validation_status"),
        "file_count": len(files),
        "files": files,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def _staging_path(output: Path, run_id: str | None = None) -> Path:
    """Return a short, output-name-independent sibling used for atomic publishing."""
    identifier = uuid.uuid4().hex if run_id is None else run_id
    return output.parent / f"{STAGING_PREFIX}{identifier}"


def _auxiliary_path(
    output: Path,
    prefix: str,
    run_id: str | None = None,
) -> Path:
    """Return a short sibling for subprocess scratch data on Windows."""
    identifier = uuid.uuid4().hex if run_id is None else run_id
    return output.parent / f"{prefix}{identifier[:AUXILIARY_TOKEN_LENGTH]}"


def reproduce(
    project_root: Path,
    output: Path,
    mode: str,
    *,
    retain_failed_staging: bool = False,
) -> dict[str, object]:
    project_root = project_root.resolve()
    output = output.resolve()
    if output.exists():
        raise ReproductionError(
            f"Output already exists; refusing to overwrite an evidence bundle: {output}"
        )
    started = time.perf_counter()
    preflight = _preflight(project_root, mode)
    release_verification = _release_verification(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(output)
    staging.mkdir()
    runtime_temp = _auxiliary_path(output, RUNTIME_TEMP_PREFIX)
    test_scratch = _auxiliary_path(output, TEST_SCRATCH_PREFIX)
    try:
        environment = _command_environment(project_root)
        runtime_temp.mkdir()
        test_scratch.mkdir()
        for variable in ("TMP", "TEMP", "TMPDIR"):
            environment[variable] = str(runtime_temp)
        environment["LIFETWIN_TEST_SCRATCH"] = str(test_scratch)
        command_summaries: dict[str, object] = {}
        phase8_comparisons: list[dict[str, object]] = []
        figure: dict[str, object] | None = None
        v012_figure: dict[str, object] | None = None
        phase1_audit: dict[str, object] = {
            "status": "skipped_by_mode",
            "mode": mode,
        }
        v011_evidence: dict[str, object] = {
            "status": "skipped_by_mode",
            "mode": mode,
        }
        v012_evidence: dict[str, object] = {
            "status": "skipped_by_mode",
            "mode": mode,
        }

        if mode in {"experiment", "full"}:
            phase8 = _run_command(
                [
                    sys.executable,
                    str(project_root / PHASE8_RUNNER),
                    "--input",
                    str(project_root / PHASE8_INPUT),
                    "--config",
                    str(project_root / PHASE8_CONFIG),
                    "--artifact-dir",
                    "phase8",
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(staging, "phase8", phase8)
            command_summaries["phase8"] = {
                key: phase8[key] for key in ("command", "returncode", "elapsed_seconds")
            }
            phase8_comparisons = _compare_core_csvs(project_root, staging / "phase8")

            analysis = _run_command(
                [
                    sys.executable,
                    str(project_root / PHASE8_ANALYZER),
                    "--results-root",
                    "phase8",
                    "--output",
                    "showcase/phase8_results.png",
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(staging, "showcase", analysis)
            command_summaries["showcase"] = {
                key: analysis[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }
            figure = _inspect_png(
                staging / "showcase/phase8_results.png",
                reported_path="showcase/phase8_results.png",
            )

            evidence_root = staging / "evidence_v011"
            landmark = _run_command(
                [
                    sys.executable,
                    str(project_root / LANDMARK_RUNNER),
                    "--observations",
                    str(project_root / PHASE8_INPUT),
                    "--v3-config",
                    str(project_root / PHASE8_CONFIG),
                    "--protocol",
                    str(project_root / LANDMARK_CONFIG),
                    "--output-dir",
                    str(evidence_root / "landmark"),
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(staging, "landmark", landmark)
            command_summaries["landmark"] = {
                key: landmark[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }

            v4 = _run_command(
                [
                    sys.executable,
                    str(project_root / V4_RUNNER),
                    "--input",
                    str(project_root / PHASE8_INPUT),
                    "--config",
                    str(project_root / V4_CONFIG),
                    "--output-dir",
                    str(evidence_root / "v4"),
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(staging, "v4", v4)
            command_summaries["v4"] = {
                key: v4[key] for key in ("command", "returncode", "elapsed_seconds")
            }

            geisbauer = _run_command(
                [
                    sys.executable,
                    str(project_root / GEISBAUER_RUNNER),
                    "--source",
                    str(project_root / PHASE8_INPUT),
                    "--target",
                    str(project_root / GEISBAUER_INPUT),
                    "--protocol",
                    str(project_root / GEISBAUER_CONFIG),
                    "--output-dir",
                    str(evidence_root / "geisbauer"),
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(staging, "geisbauer", geisbauer)
            command_summaries["geisbauer"] = {
                key: geisbauer[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }
            v011_evidence = {
                "status": "passed",
                "landmark": _inspect_v011_group(
                    project_root, evidence_root / "landmark", "landmark"
                ),
                "v4": _inspect_v011_group(project_root, evidence_root / "v4", "v4"),
                "geisbauer": _inspect_v011_group(
                    project_root, evidence_root / "geisbauer", "geisbauer"
                ),
            }

            evidence_v012_root = staging / "evidence_v012"
            v4_calibration_robustness = _run_command(
                [
                    sys.executable,
                    str(project_root / V4_CALIBRATION_ROBUSTNESS_RUNNER),
                    "--input",
                    str(project_root / PHASE8_INPUT),
                    "--upstream-config",
                    str(project_root / V4_CONFIG),
                    "--audit-config",
                    str(project_root / V4_CALIBRATION_ROBUSTNESS_CONFIG),
                    "--output-dir",
                    str(evidence_v012_root / "v4_calibration_robustness"),
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(
                staging,
                "v4_calibration_robustness",
                v4_calibration_robustness,
            )
            command_summaries["v4_calibration_robustness"] = {
                key: v4_calibration_robustness[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }

            geisbauer_robustness = _run_command(
                [
                    sys.executable,
                    str(project_root / GEISBAUER_ROBUSTNESS_RUNNER),
                    "--source",
                    str(project_root / PHASE8_INPUT),
                    "--target",
                    str(project_root / GEISBAUER_INPUT),
                    "--external-protocol",
                    str(project_root / GEISBAUER_CONFIG),
                    "--audit-protocol",
                    str(project_root / GEISBAUER_ROBUSTNESS_CONFIG),
                    "--output-dir",
                    str(evidence_v012_root / "geisbauer_robustness"),
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(
                staging,
                "geisbauer_robustness",
                geisbauer_robustness,
            )
            command_summaries["geisbauer_robustness"] = {
                key: geisbauer_robustness[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }
            v012_evidence = {
                "status": "passed",
                "v4_calibration_robustness": _inspect_v012_group(
                    project_root,
                    evidence_v012_root / "v4_calibration_robustness",
                    "v4_calibration_robustness",
                ),
                "geisbauer_robustness": _inspect_v012_group(
                    project_root,
                    evidence_v012_root / "geisbauer_robustness",
                    "geisbauer_robustness",
                ),
            }

            v012_analysis = _run_command(
                [
                    sys.executable,
                    str(project_root / V012_ANALYZER),
                    "--evidence-root",
                    str(evidence_v012_root),
                    "--output",
                    "showcase/v012_robustness.png",
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(staging, "v012_showcase", v012_analysis)
            command_summaries["v012_showcase"] = {
                key: v012_analysis[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }
            v012_figure = _inspect_png(
                staging / "showcase/v012_robustness.png",
                reported_path="showcase/v012_robustness.png",
            )

        if mode == "full":
            audit = _run_command(
                [
                    sys.executable,
                    str(project_root / PHASE1_AUDIT_RUNNER),
                    "--data",
                    str(project_root / PHASE8_INPUT),
                    "--config",
                    str(project_root / PHASE8_CONFIG),
                    "--output-dir",
                    "phase1_audit",
                ],
                cwd=staging,
                environment=environment,
            )
            _write_command_log(staging, "phase1_audit", audit)
            command_summaries["phase1_audit"] = {
                key: audit[key] for key in ("command", "returncode", "elapsed_seconds")
            }
            phase1_audit = _inspect_phase1_audit(
                staging / "phase1_audit",
                project_root,
            )

        pytest_result: dict[str, object]
        if mode in {"tests", "full"}:
            pytest_command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--basetemp",
                str(test_scratch),
            ]
            pytest_run = _run_command(
                pytest_command,
                cwd=project_root,
                environment=environment,
            )
            _write_command_log(staging, "pytest", pytest_run)
            command_summaries["pytest"] = {
                key: pytest_run[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }
            pytest_result = {
                "status": "passed",
                "command": pytest_command,
            }
        else:
            pytest_result = {
                "status": "skipped_by_mode",
                "mode": mode,
            }

        experiment_status = (
            "passed" if mode in {"experiment", "full"} else "skipped_by_mode"
        )
        for scratch_path in (runtime_temp, test_scratch):
            if scratch_path.exists():
                _rmtree(scratch_path)
        summary: dict[str, object] = {
            "schema_version": 4,
            "status": "passed",
            "mode": mode,
            "atomic_publish": True,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "preflight": preflight,
            "release_verification": release_verification,
            "phase8": {
                "status": experiment_status,
                "core_csv_comparisons": phase8_comparisons,
            },
            "headless_figure": figure
            if figure is not None
            else {"status": "skipped_by_mode", "mode": mode},
            "v012_headless_figure": v012_figure
            if v012_figure is not None
            else {"status": "skipped_by_mode", "mode": mode},
            "phase1_adversarial_audit": phase1_audit,
            "evidence_v011": v011_evidence,
            "evidence_v012": v012_evidence,
            "pytest": pytest_result,
            "commands": command_summaries,
        }
        _write_json(staging / "reproduction_summary.json", summary)
        publish_directory(staging, output)
        return summary
    except BaseException as error:
        auxiliary_cleanup_errors: list[OSError] = []
        for scratch_path in (runtime_temp, test_scratch):
            try:
                if scratch_path.exists():
                    _rmtree(scratch_path)
            except OSError as cleanup_error:
                auxiliary_cleanup_errors.append(cleanup_error)
        publish_retry_exhausted = isinstance(
            error,
            AtomicPublishRetryExhausted,
        )
        if retain_failed_staging or publish_retry_exhausted:
            detail = f"{error}\nFailed diagnostic staging retained at: {staging}"
            if auxiliary_cleanup_errors:
                detail += (
                    "\nAuxiliary scratch cleanup also failed: "
                    + "; ".join(str(item) for item in auxiliary_cleanup_errors)
                )
            raise ReproductionError(detail) from error
        try:
            if staging.exists():
                _rmtree(staging)
        except OSError as cleanup_error:
            raise ReproductionError(
                "Reproduction failed and staging cleanup also failed: "
                f"{staging} ({cleanup_error})"
            ) from error
        if auxiliary_cleanup_errors:
            raise ReproductionError(
                "Reproduction failed and auxiliary scratch cleanup also failed: "
                + "; ".join(str(item) for item in auxiliary_cleanup_errors)
            ) from error
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Atomically reproduce and verify the LifeTwin public release."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "experiment", "tests"),
        default="full",
        help=(
            "full runs Phase 8, the V0.11 landmark/V4/external evidence, the "
            "V0.12 calibration/external robustness audits, the Phase 1 "
            "adversarial audit, both headless figures, and pytest including "
            "the future-label-free prefix demo and Judge Console checks"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reproduction"),
    )
    parser.add_argument(
        "--retain-failed-staging",
        action="store_true",
        help=(
            "retain the hidden, unpublished staging directory after failure "
            "for CI diagnostics"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        summary = reproduce(
            args.project_root,
            args.output,
            args.mode,
            retain_failed_staging=args.retain_failed_staging,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "mode": args.mode,
                    "output_published": False,
                    "error": str(error),
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": summary["status"],
                "mode": summary["mode"],
                "output": args.output.resolve().as_posix(),
                "summary": (
                    args.output.resolve() / "reproduction_summary.json"
                ).as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
