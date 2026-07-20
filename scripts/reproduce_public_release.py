from __future__ import annotations

import argparse
import csv
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE8_RUNNER = Path("scripts/run_calendar_v3_activation_development.py")
PHASE8_ANALYZER = Path("showcase/analyze_phase8_results.py")
PHASE1_AUDIT_RUNNER = Path("scripts/run_phase1_adversarial_audit.py")
PHASE8_INPUT = Path("data/interim/naumann_calendar_observations.csv")
REPRODUCTION_CONSTRAINTS = Path("requirements/reproduction.txt")
PHASE8_CONFIG = Path(
    "configs/experiments/naumann_calendar_v3_activation_development.json"
)
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
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}
NUMERIC_RELATIVE_TOLERANCE = 1e-8
NUMERIC_ABSOLUTE_TOLERANCE = 1e-8
STATE_HASH_COLUMNS = frozenset(
    {"training_state_sha256", "prediction_state_sha256"}
)
FUTURE_ATTACK_HASH_PAIRS = (
    ("prediction_sha256_baseline", "prediction_sha256_attacked"),
    ("sensitivity_sha256_baseline", "sensitivity_sha256_attacked"),
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REPRODUCTION_PYTHON = (3, 12)


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
        for value in completed.stdout.decode(
            "utf-8", errors="surrogateescape"
        ).split("\0")
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
        PHASE1_AUDIT_RUNNER,
        PHASE8_INPUT,
        PHASE8_CONFIG,
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

    required_packages = {"numpy", "pandas", "scipy", "scikit-learn"}
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
            package_versions[distribution] = importlib_metadata.version(
                distribution
            )
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
        len(row) != len(header)
        for row in (*published_rows[1:], *generated_rows[1:])
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
        published_topology = _sha256_equivalence_topology(
            published_rows[1:], index
        )
        generated_topology = _sha256_equivalence_topology(
            generated_rows[1:], index
        )
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


def _json_semantically_equal(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _json_semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_semantically_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
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
            "cross_platform_volatile_sha256_columns": sorted(
                volatile_sha256_columns
            ),
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


def _inspect_png(path: Path) -> dict[str, object]:
    payload = path.read_bytes() if path.is_file() else b""
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ReproductionError(f"Headless figure is not a valid PNG: {path}")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ReproductionError(f"Headless figure has invalid dimensions: {path}")
    return {
        "path": "showcase/phase8_results.png",
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
        if filename.endswith(".json"):
            semantically_equal = _json_semantically_equal(
                json.loads(published.read_text(encoding="utf-8")),
                json.loads(generated.read_text(encoding="utf-8")),
            )
        else:
            published_rows = _csv_content(published)
            generated_rows = _csv_content(generated)
            volatile_sha256_columns = (
                frozenset(column for pair in FUTURE_ATTACK_HASH_PAIRS for column in pair)
                if filename == "future_label_attack_cases.csv"
                else frozenset()
            )
            semantically_equal = _csv_semantically_equal(
                published_rows,
                generated_rows,
                volatile_sha256_columns=volatile_sha256_columns,
            )
            if filename == "future_label_attack_cases.csv":
                semantically_equal = bool(
                    semantically_equal
                    and _paired_sha256_columns_valid(
                        published_rows, FUTURE_ATTACK_HASH_PAIRS
                    )
                    and _paired_sha256_columns_valid(
                        generated_rows, FUTURE_ATTACK_HASH_PAIRS
                    )
                )
        passed = published_sha256 == expected_sha256 and semantically_equal
        comparison = {
            "path": published_relative,
            "generated_path": f"phase1_audit/{filename}",
            "expected_release_sha256": expected_sha256,
            "published_sha256": published_sha256,
            "generated_sha256": generated_sha256,
            "generated_sha_matches_release": generated_sha256 == expected_sha256,
            "semantic_content_equal": semantically_equal,
            "numeric_tolerance": {
                "relative": NUMERIC_RELATIVE_TOLERANCE,
                "absolute": NUMERIC_ABSOLUTE_TOLERANCE,
            },
            "cross_platform_volatile_sha256_columns": sorted(
                volatile_sha256_columns
            ),
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
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def reproduce(project_root: Path, output: Path, mode: str) -> dict[str, object]:
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
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        environment = _command_environment(project_root)
        runtime_temp = staging / "tmp"
        runtime_temp.mkdir()
        for variable in ("TMP", "TEMP", "TMPDIR"):
            environment[variable] = str(runtime_temp)
        environment["LIFETWIN_TEST_SCRATCH"] = str(staging / "test-scratch")
        command_summaries: dict[str, object] = {}
        phase8_comparisons: list[dict[str, object]] = []
        figure: dict[str, object] | None = None
        phase1_audit: dict[str, object] = {
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
            phase8_comparisons = _compare_core_csvs(
                project_root, staging / "phase8"
            )

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
            figure = _inspect_png(staging / "showcase/phase8_results.png")

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
                key: audit[key]
                for key in ("command", "returncode", "elapsed_seconds")
            }
            phase1_audit = _inspect_phase1_audit(
                staging / "phase1_audit",
                project_root,
            )

        pytest_result: dict[str, object]
        if mode in {"tests", "full"}:
            pytest_run = _run_command(
                [sys.executable, "-m", "pytest", "-q"],
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
                "command": [sys.executable, "-m", "pytest", "-q"],
            }
        else:
            pytest_result = {
                "status": "skipped_by_mode",
                "mode": mode,
            }

        experiment_status = (
            "passed" if mode in {"experiment", "full"} else "skipped_by_mode"
        )
        for scratch_path in (runtime_temp, staging / "test-scratch"):
            if scratch_path.exists():
                _rmtree(scratch_path)
        summary: dict[str, object] = {
            "schema_version": 1,
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
            "phase1_adversarial_audit": phase1_audit,
            "pytest": pytest_result,
            "commands": command_summaries,
        }
        _write_json(staging / "reproduction_summary.json", summary)
        os.replace(staging, output)
        return summary
    except BaseException as error:
        try:
            if staging.exists():
                _rmtree(staging)
        except OSError as cleanup_error:
            raise ReproductionError(
                "Reproduction failed and staging cleanup also failed: "
                f"{staging} ({cleanup_error})"
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
            "full runs Phase 8, the Phase 1 adversarial audit, the headless "
            "figure, and pytest"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reproduction"),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        summary = reproduce(args.project_root, args.output, args.mode)
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
                "summary": (args.output.resolve() / "reproduction_summary.json").as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
