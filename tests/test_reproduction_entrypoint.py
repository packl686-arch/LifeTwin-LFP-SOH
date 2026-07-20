from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import uuid

import pytest

from scripts.reproduce_public_release import (
    CORE_PHASE8_CSVS,
    FUTURE_ATTACK_HASH_PAIRS,
    PHASE1_AUDIT_FILES,
    PHASE1_SOLVER_ABSOLUTE_TOLERANCE,
    ReproductionError,
    STATE_HASH_COLUMNS,
    _clean_git_head,
    _compare_core_csvs,
    _csv_content,
    _command_environment,
    _csv_semantically_equal,
    _inspect_phase1_audit,
    _locked_versions,
    _paired_sha256_columns_valid,
    _phase1_csv_semantic_comparison,
    _rmtree,
    reproduce,
)
from scripts.verify_public_release import _version_consistency, verify


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def writable_root() -> Path:
    scratch_root = Path(
        os.environ.get(
            "LIFETWIN_TEST_SCRATCH",
            str(PROJECT_ROOT / "artifacts/test-scratch"),
        )
    )
    root = scratch_root / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        _rmtree(root)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase1_csv_rows(filename: str) -> list[list[str]]:
    return _csv_content(PROJECT_ROOT / "showcase/audit_results" / filename)


def _init_release_repository(root: Path) -> None:
    payload = root / "payload.txt"
    payload.write_text("release payload\n", encoding="utf-8")
    manifest = {
        "release_id": "test-release",
        "maximum_file_size_bytes": 1_000_000,
        "frozen_files_sha256": {"payload.txt": _sha256(payload)},
        "forbidden_release_globs": ["artifacts/**"],
    }
    (root / "release_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "add", "payload.txt", "release_manifest.json"],
        cwd=root,
        check=True,
    )


def _commit_repository(root: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=LifeTwin Tests", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=root,
        check=True,
    )


def _version_fixture(root: Path) -> dict[str, object]:
    (root / "src/lifetwin").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    (root / "CITATION.cff").write_text(
        "version: 1.2.3\ndate-released: 2026-07-20\n", encoding="utf-8"
    )
    (root / "src/lifetwin/__init__.py").write_text(
        '__version__ = "1.2.3"\n', encoding="utf-8"
    )
    return {
        "schema_version": 2,
        "release_id": "fixture_v1.2.3",
        "release_date": "2026-07-20",
    }


def test_clean_head_rejects_tracked_and_untracked_changes(
    writable_root: Path,
) -> None:
    _init_release_repository(writable_root)
    _commit_repository(writable_root)

    assert len(_clean_git_head(writable_root)) == 40
    (writable_root / "payload.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ReproductionError, match="modified"):
        _clean_git_head(writable_root)
    subprocess.run(
        ["git", "restore", "payload.txt"], cwd=writable_root, check=True
    )
    (writable_root / "sitecustomize.py").write_text(
        "raise RuntimeError('polluted')\n", encoding="utf-8"
    )
    with pytest.raises(ReproductionError, match="modified"):
        _clean_git_head(writable_root)


def test_subprocess_environment_isolated_from_external_pythonpath(
    writable_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(writable_root / "external"))

    environment = _command_environment(writable_root)

    assert environment["PYTHONPATH"] == str(writable_root / "src")
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHONUTF8"] == "1"
    assert environment["MPLBACKEND"] == "Agg"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["NUMEXPR_NUM_THREADS"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_semantic_csv_comparison_allows_numeric_drift_but_validates_hashes() -> None:
    hash_header = [
        "value",
        "training_state_sha256",
        "prediction_state_sha256",
    ]
    published = [
        hash_header,
        ["1.0", "a" * 64, "b" * 64],
    ]
    generated = [
        published[0],
        ["1.00019", "c" * 64, "d" * 64],
    ]

    assert _csv_semantically_equal(
        published,
        generated,
        volatile_sha256_columns=STATE_HASH_COLUMNS,
    )
    generated[1][0] = "1.001"
    assert not _csv_semantically_equal(
        published,
        generated,
        volatile_sha256_columns=STATE_HASH_COLUMNS,
    )
    invalid_hash = [hash_header, ["1.0", "not-a-hash", "d" * 64]]
    assert not _csv_semantically_equal(
        published,
        invalid_hash,
        volatile_sha256_columns=STATE_HASH_COLUMNS,
    )

    published_topology = [
        hash_header,
        ["1.0", "a" * 64, "b" * 64],
        ["2.0", "e" * 64, "f" * 64],
    ]
    collapsed_topology = [
        hash_header,
        ["1.0", "c" * 64, "d" * 64],
        ["2.0", "c" * 64, "d" * 64],
    ]
    assert not _csv_semantically_equal(
        published_topology,
        collapsed_topology,
        volatile_sha256_columns=STATE_HASH_COLUMNS,
    )

    published_count = [["prefix_checkups", "value"], ["10", "1.0"]]
    drifted_count = [["prefix_checkups", "value"], ["10.0001", "1.0"]]
    assert not _csv_semantically_equal(published_count, drifted_count)


def test_future_attack_hash_pairs_require_within_run_equality() -> None:
    header = [column for pair in FUTURE_ATTACK_HASH_PAIRS for column in pair]
    rows = [header, ["a" * 64, "a" * 64, "", ""]]

    assert _paired_sha256_columns_valid(rows, FUTURE_ATTACK_HASH_PAIRS)
    rows[1][1] = "b" * 64
    assert not _paired_sha256_columns_valid(rows, FUTURE_ATTACK_HASH_PAIRS)


def test_phase1_csv_comparison_aligns_rows_by_declared_key() -> None:
    published = _phase1_csv_rows("ablation_audit.csv")
    generated = [published[0].copy(), *reversed(published[1:])]

    comparison = _phase1_csv_semantic_comparison(
        "ablation_audit.csv", published, generated
    )

    assert comparison["semantic_content_equal"] is True
    assert comparison["mismatch"] is None


def test_phase1_csv_comparison_rejects_duplicate_keys_and_header_drift() -> None:
    published = _phase1_csv_rows("data_condition_audit.csv")
    duplicate = [row.copy() for row in published]
    duplicate.append(duplicate[1].copy())

    comparison = _phase1_csv_semantic_comparison(
        "data_condition_audit.csv", published, duplicate
    )

    assert comparison["semantic_content_equal"] is False
    assert comparison["mismatch"]["reason"] == "duplicate_key"
    assert comparison["mismatch"]["key"] == {
        "condition_id": duplicate[1][0]
    }

    drifted_header = [row.copy() for row in published]
    drifted_header[0][0] = "condition"
    comparison = _phase1_csv_semantic_comparison(
        "data_condition_audit.csv", published, drifted_header
    )
    assert comparison["mismatch"]["reason"] == "header_drift"


def test_phase1_csv_comparison_keeps_protocol_columns_exact() -> None:
    published = _phase1_csv_rows("failure_condition_table.csv")
    generated = [row.copy() for row in published]
    column = published[0].index("temperature_c")
    generated[1][column] = str(float(published[1][column]) + 0.0001)

    comparison = _phase1_csv_semantic_comparison(
        "failure_condition_table.csv", published, generated
    )
    mismatch = comparison["mismatch"]

    assert comparison["semantic_content_equal"] is False
    assert mismatch["reason"] == "exact_value_mismatch"
    assert mismatch["column"] == "temperature_c"
    assert mismatch["values"] == {
        "published": published[1][column],
        "generated": generated[1][column],
    }
    assert mismatch["tolerance"] == {"mode": "exact"}


def test_phase1_csv_comparison_enforces_solver_tolerance_boundary() -> None:
    published = _phase1_csv_rows("independent_metric_audit.csv")
    columns = [
        published[0].index("trajectory_iae_pp"),
        published[0].index("trajectory_iae_pp_recomputed"),
    ]

    within = [row.copy() for row in published]
    for column in columns:
        within[1][column] = str(
            float(published[1][column])
            + 0.9 * PHASE1_SOLVER_ABSOLUTE_TOLERANCE
        )
    comparison = _phase1_csv_semantic_comparison(
        "independent_metric_audit.csv", published, within
    )
    assert comparison["semantic_content_equal"] is True

    beyond = [row.copy() for row in published]
    for column in columns:
        beyond[1][column] = str(
            float(published[1][column])
            + 1.1 * PHASE1_SOLVER_ABSOLUTE_TOLERANCE
        )
    comparison = _phase1_csv_semantic_comparison(
        "independent_metric_audit.csv", published, beyond
    )
    mismatch = comparison["mismatch"]

    assert comparison["semantic_content_equal"] is False
    assert mismatch["reason"] == "numeric_value_mismatch"
    assert mismatch["column"] == "trajectory_iae_pp"
    assert mismatch["key"] == {
        key: published[1][published[0].index(key)]
        for key in comparison["key_columns"]
    }
    assert mismatch["delta"] == pytest.approx(
        1.1 * PHASE1_SOLVER_ABSOLUTE_TOLERANCE
    )
    assert mismatch["tolerance"]["absolute"] == (
        PHASE1_SOLVER_ABSOLUTE_TOLERANCE
    )


def test_phase1_csv_comparison_rejects_inconsistent_metric_witnesses() -> None:
    published = _phase1_csv_rows("independent_metric_audit.csv")
    generated = [row.copy() for row in published]
    official = published[0].index("trajectory_iae_pp")
    recomputed = published[0].index("trajectory_iae_pp_recomputed")
    shift = 0.8 * PHASE1_SOLVER_ABSOLUTE_TOLERANCE
    generated[1][official] = str(float(published[1][official]) + shift)
    generated[1][recomputed] = str(float(published[1][recomputed]) - shift)

    comparison = _phase1_csv_semantic_comparison(
        "independent_metric_audit.csv", published, generated
    )
    mismatch = comparison["mismatch"]

    assert comparison["semantic_content_equal"] is False
    assert mismatch["reason"] == "row_invariant_mismatch"
    assert mismatch["column"] == "trajectory_iae_pp_audit_difference"


def test_phase1_csv_comparison_recomputes_failure_risk_flags() -> None:
    published = _phase1_csv_rows("failure_condition_table.csv")
    generated = [row.copy() for row in published]
    gated = published[0].index("gated_hierarchical_trajectory_iae_pp")
    delta = published[0].index("gated_hierarchical_vs_v2_delta_iae_pp")
    row_index = next(
        index
        for index, row in enumerate(published[1:], start=1)
        if abs(float(row[delta])) <= 1e-12
    )
    generated[row_index][gated] = str(
        float(published[row_index][gated]) + 0.004
    )
    generated[row_index][delta] = str(
        float(published[row_index][delta]) + 0.004
    )

    comparison = _phase1_csv_semantic_comparison(
        "failure_condition_table.csv", published, generated
    )

    assert comparison["semantic_content_equal"] is False
    assert comparison["mismatch"]["reason"] == "row_invariant_mismatch"
    assert comparison["mismatch"]["column"] == "risk_flags"


@pytest.mark.parametrize(
    ("filename", "column"),
    [
        ("ablation_audit.csv", "comparator_iae_pp_mean"),
        ("failure_condition_table.csv", "comparator_trajectory_iae_pp"),
        ("failure_condition_table.csv", "prefix_end_days"),
    ],
)
def test_phase1_csv_comparison_rejects_zero_invariant_denominators(
    filename: str,
    column: str,
) -> None:
    published = _phase1_csv_rows(filename)
    generated = [row.copy() for row in published]
    generated[1][published[0].index(column)] = "0"

    comparison = _phase1_csv_semantic_comparison(
        filename, published, generated
    )

    assert comparison["semantic_content_equal"] is False
    assert comparison["mismatch"]["reason"] == "zero_invariant_denominator"
    assert comparison["mismatch"]["column"] == column


def test_phase1_csv_comparison_preserves_global_hash_aliases() -> None:
    published = _phase1_csv_rows("future_label_attack_cases.csv")
    header = published[0]
    hash_columns = [
        header.index(column)
        for pair in FUTURE_ATTACK_HASH_PAIRS
        for column in pair
    ]
    renamed = [row.copy() for row in published]
    mapping: dict[str, str] = {}
    for row in renamed[1:]:
        for column in hash_columns:
            value = row[column]
            if value:
                mapping.setdefault(value, f"{len(mapping) + 1:064x}")
                row[column] = mapping[value]

    comparison = _phase1_csv_semantic_comparison(
        "future_label_attack_cases.csv", published, renamed
    )
    assert comparison["semantic_content_equal"] is True

    aliased = [row.copy() for row in renamed]
    prefix_index = header.index("prefix_checkups")
    prefix_10 = next(row for row in aliased[1:] if row[prefix_index] == "10")
    prediction_hash = prefix_10[header.index("prediction_sha256_baseline")]
    prefix_10[header.index("sensitivity_sha256_baseline")] = prediction_hash
    prefix_10[header.index("sensitivity_sha256_attacked")] = prediction_hash

    comparison = _phase1_csv_semantic_comparison(
        "future_label_attack_cases.csv", published, aliased
    )
    assert comparison["semantic_content_equal"] is False
    assert comparison["mismatch"]["reason"] == "sha256_topology_mismatch"


def test_phase1_csv_comparison_rejects_nonfinite_values() -> None:
    published = _phase1_csv_rows("independent_metric_audit.csv")
    generated = [row.copy() for row in published]
    column = published[0].index("trajectory_iae_pp")
    generated[1][column] = "nan"

    comparison = _phase1_csv_semantic_comparison(
        "independent_metric_audit.csv", published, generated
    )
    mismatch = comparison["mismatch"]

    assert comparison["semantic_content_equal"] is False
    assert mismatch["reason"] == "non_finite_numeric_value"
    assert mismatch["column"] == "trajectory_iae_pp"
    assert mismatch["values"]["generated"] == "nan"


def test_reproduction_constraints_must_be_exact_and_unique(
    writable_root: Path,
) -> None:
    constraints = writable_root / "constraints.txt"
    constraints.write_text("numpy==2.5.1\npytest==9.1.1\n", encoding="utf-8")
    assert _locked_versions(constraints) == {
        "numpy": "2.5.1",
        "pytest": "9.1.1",
    }
    constraints.write_text("numpy>=2.5\n", encoding="utf-8")
    with pytest.raises(ReproductionError, match="Invalid exact constraint"):
        _locked_versions(constraints)
    constraints.write_text("numpy==2.5.1\nnumpy==2.5.1\n", encoding="utf-8")
    with pytest.raises(ReproductionError, match="Duplicate"):
        _locked_versions(constraints)


def test_rmtree_removes_readonly_files_within_requested_tree(
    writable_root: Path,
) -> None:
    cleanup_root = writable_root / "readonly-tree"
    nested = cleanup_root / ".git" / "objects"
    nested.mkdir(parents=True)
    readonly = nested / "object"
    readonly.write_bytes(b"git object")
    readonly.chmod(stat.S_IREAD)

    _rmtree(cleanup_root)

    assert not cleanup_root.exists()


def test_version_consistency_rejects_drift_and_invalid_dates(
    writable_root: Path,
) -> None:
    manifest = _version_fixture(writable_root)

    assert _version_consistency(writable_root, manifest)["status"] == "passed"
    (writable_root / "CITATION.cff").write_text(
        "version: 9.9.9\ndate-released: 2026-07-20\n", encoding="utf-8"
    )
    assert _version_consistency(writable_root, manifest)["status"] == "failed"
    (writable_root / "CITATION.cff").write_text(
        "version: 1.2.3\ndate-released: not-a-date\n", encoding="utf-8"
    )
    result = _version_consistency(
        writable_root,
        {**manifest, "release_date": "not-a-date"},
    )
    assert result["status"] == "failed"
    assert "error" in result


def test_version_consistency_reports_missing_metadata(writable_root: Path) -> None:
    manifest = _version_fixture(writable_root)
    (writable_root / "CITATION.cff").unlink()

    result = _version_consistency(writable_root, manifest)

    assert result["status"] == "failed"
    assert "FileNotFoundError" in result["error"]


def test_version_consistency_rejects_invalid_schema(writable_root: Path) -> None:
    result = _version_consistency(writable_root, {"schema_version": None})

    assert result["status"] == "failed"
    assert "TypeError" in result["error"]


def test_verifier_scans_git_tracked_files_only(writable_root: Path) -> None:
    _init_release_repository(writable_root)
    ignored = writable_root / "artifacts"
    ignored.mkdir()
    (ignored / "large.bin").write_bytes(b"x" * 1_000_001)
    (writable_root / "untracked_broken.md").write_text(
        "[broken](missing.txt)\n", encoding="utf-8"
    )

    result = verify(writable_root)

    assert result["status"] == "passed"
    assert result["scan_mode"] == "git_tracked_files"
    assert result["manifest_tracked"] is True
    assert result["scanned_file_count"] == 2
    assert result["forbidden_files"] == []
    assert result["oversized_files"] == []
    assert result["broken_markdown_links"] == []


def test_verifier_rejects_untracked_frozen_file(writable_root: Path) -> None:
    _init_release_repository(writable_root)
    local_only = writable_root / "local-only.txt"
    local_only.write_text("not in git\n", encoding="utf-8")
    manifest_path = writable_root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_files_sha256"]["local-only.txt"] = _sha256(local_only)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = verify(writable_root)

    assert result["status"] == "failed"
    assert result["untracked_frozen_files"] == ["local-only.txt"]


def test_core_csv_comparison_checks_manifest_hash_and_content(
    writable_root: Path,
) -> None:
    generated = writable_root / "generated"
    published = writable_root / "showcase/results"
    generated.mkdir()
    published.mkdir(parents=True)
    frozen: dict[str, str] = {}
    for filename in CORE_PHASE8_CSVS:
        if filename == "target_diagnostics.csv":
            content = (
                "condition,value,training_state_sha256,prediction_state_sha256\n"
                f"A,1.25,{'a' * 64},{'b' * 64}\n"
            )
        else:
            content = "condition,value\nA,1.25\n"
        generated_path = generated / filename
        published_path = published / filename
        generated_path.write_text(content, encoding="utf-8", newline="")
        published_path.write_text(content, encoding="utf-8", newline="")
        frozen[f"showcase/results/{filename}"] = _sha256(published_path)
    (writable_root / "release_manifest.json").write_text(
        json.dumps({"frozen_files_sha256": frozen}), encoding="utf-8"
    )

    comparisons = _compare_core_csvs(writable_root, generated)

    assert len(comparisons) == 3
    assert all(row["status"] == "passed" for row in comparisons)
    (generated / CORE_PHASE8_CSVS[0]).write_text(
        "condition,value\nA,9\n", encoding="utf-8", newline=""
    )
    with pytest.raises(ReproductionError, match="reproduction mismatch"):
        _compare_core_csvs(writable_root, generated)


def test_failed_reproduction_leaves_no_partial_output(
    writable_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = writable_root / "artifacts/reproduction"
    monkeypatch.setattr(
        "scripts.reproduce_public_release._preflight",
        lambda *_: {"status": "passed"},
    )
    monkeypatch.setattr(
        "scripts.reproduce_public_release._release_verification",
        lambda *_: {"status": "passed"},
    )

    def fail_command(*_args: object, **_kwargs: object) -> dict[str, object]:
        environment = _kwargs["environment"]
        scratch = Path(environment["LIFETWIN_TEST_SCRATCH"])
        scratch.mkdir(parents=True)
        readonly = scratch / "readonly"
        readonly.write_bytes(b"git object")
        readonly.chmod(stat.S_IREAD)
        raise ReproductionError("intentional failure")

    monkeypatch.setattr(
        "scripts.reproduce_public_release._run_command", fail_command
    )
    with pytest.raises(ReproductionError, match="intentional failure"):
        reproduce(writable_root, output, "tests")

    assert not output.exists()
    assert not list(output.parent.glob(".reproduction.staging-*"))


def test_failed_reproduction_can_retain_unpublished_diagnostics(
    writable_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = writable_root / "artifacts/reproduction"
    monkeypatch.setattr(
        "scripts.reproduce_public_release._preflight",
        lambda *_: {"status": "passed"},
    )
    monkeypatch.setattr(
        "scripts.reproduce_public_release._release_verification",
        lambda *_: {"status": "passed"},
    )

    def fail_command(*_args: object, **_kwargs: object) -> dict[str, object]:
        staging = Path(_kwargs["environment"]["LIFETWIN_TEST_SCRATCH"]).parent
        (staging / "diagnostic.txt").write_text(
            "failure evidence\n", encoding="utf-8"
        )
        raise ReproductionError("intentional failure")

    monkeypatch.setattr(
        "scripts.reproduce_public_release._run_command", fail_command
    )
    with pytest.raises(ReproductionError, match="diagnostic staging retained"):
        reproduce(
            writable_root,
            output,
            "tests",
            retain_failed_staging=True,
        )

    retained = list(output.parent.glob(".reproduction.staging-*"))
    assert not output.exists()
    assert len(retained) == 1
    assert (retained[0] / "diagnostic.txt").read_text(encoding="utf-8") == (
        "failure evidence\n"
    )


def test_phase1_audit_summary_and_file_inventory(writable_root: Path) -> None:
    audit_root = writable_root / "phase1_audit"
    audit_root.mkdir()
    for filename in PHASE1_AUDIT_FILES:
        if filename.endswith(".csv"):
            shutil.copy2(
                PROJECT_ROOT / "showcase/audit_results" / filename,
                audit_root / filename,
            )
    (audit_root / "phase1_adversarial_audit.json").write_text(
        json.dumps(
            {
                "audit_execution_status": "passed",
                "model_validation_status": "not_confirmed",
            }
        ),
        encoding="utf-8",
    )
    published_root = writable_root / "showcase/audit_results"
    published_root.mkdir(parents=True)
    frozen: dict[str, str] = {}
    for filename in PHASE1_AUDIT_FILES:
        published = published_root / filename
        shutil.copy2(audit_root / filename, published)
        frozen[f"showcase/audit_results/{filename}"] = _sha256(published)
    (writable_root / "release_manifest.json").write_text(
        json.dumps({"frozen_files_sha256": frozen}),
        encoding="utf-8",
    )

    inventory = _inspect_phase1_audit(audit_root, writable_root)

    assert inventory["status"] == "passed"
    assert inventory["model_validation_status"] == "not_confirmed"
    assert inventory["file_count"] == len(PHASE1_AUDIT_FILES)
    assert all(row["semantic_content_equal"] for row in inventory["files"])

    generated_summary = audit_root / "phase1_adversarial_audit.json"
    summary = json.loads(generated_summary.read_text(encoding="utf-8"))
    summary["model_validation_status"] = "confirmed"
    generated_summary.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ReproductionError, match="not_confirmed"):
        _inspect_phase1_audit(audit_root, writable_root)
    shutil.copy2(published_root / generated_summary.name, generated_summary)

    manifest_path = writable_root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_key = "showcase/audit_results/phase1_adversarial_audit.json"
    expected_summary_hash = manifest["frozen_files_sha256"][summary_key]
    manifest["frozen_files_sha256"][summary_key] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReproductionError, match="reproduction mismatch"):
        _inspect_phase1_audit(audit_root, writable_root)
    manifest["frozen_files_sha256"][summary_key] = expected_summary_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    generated_failure_table = audit_root / "failure_condition_table.csv"
    generated_failure_table.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ReproductionError, match="reproduction mismatch"):
        _inspect_phase1_audit(audit_root, writable_root)
