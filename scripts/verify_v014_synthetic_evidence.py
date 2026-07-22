from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_RELATIVE = (
    Path("showcase") / "evidence_v014" / "synthetic_long_horizon_identifiability_v1"
)
PROTOCOL_ID = "synthetic_long_horizon_identifiability_v1"
CONFIG_PATH = "configs/experiments/synthetic_long_horizon_identifiability_v1.json"
EXECUTION_SOURCE_PATHS = {
    CONFIG_PATH,
    "pyproject.toml",
    "requirements/reproduction.txt",
    "scripts/run_synthetic_long_horizon_identifiability.py",
    "src/lifetwin/experiments/calendar_long_horizon_analysis.py",
    "src/lifetwin/experiments/calendar_long_horizon_synthetic.py",
}
EXPECTED_CLAIM_BOUNDARY = (
    "Synthetic mechanism stress test only; no real-LFP, Hithium-product, "
    "storage-station, operational-coverage, or 15-25-year accuracy claim."
)
EXPECTED_REPORT_CLAIM_BOUNDARY = (
    "This result is a frozen synthetic mechanism stress test. It does not "
    "validate real LFP, Hithium product, individual-cell, storage-station, "
    "operational interval, or 15-25 year accuracy claims."
)
EXPECTED_EVIDENCE_MANIFEST_SHA256 = (
    "d50e78eccecd5f297b7dd7174e71ee0745fa6156faaefdf6d5dc807f7aaea103"
)
EXPECTED_FULL_BUNDLE_MANIFEST_SHA256 = (
    "25bd94923c14e5217670de971a5479950519a027e0f099ba4ea3e8be60f13645"
)
EXPECTED_EXPOSURE_EVENTS = (
    "environment_and_code_frozen",
    "sealed_truth_generated_and_committed",
    "label_free_predictions_and_decisions_committed",
    "sealed_truth_released_to_scorer_after_prediction_commitment",
    "strict_scoring_completed",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

DECISION_COLUMNS = (
    "protocol_id",
    "partition",
    "cluster_id",
    "canonical_prefix_content_sha256",
    "credible_structure_family_count",
    "fit_failure_count",
    "best_prefix_rmse_pp",
    "disagreement_score_pp",
    "hard_eligible",
    "primary_issuance_rank",
    "primary_issued",
    "abstention_reasons",
)
TRAJECTORY_COLUMNS = (
    "partition",
    "cluster_id",
    "truth_family",
    "hard_eligible",
    "primary_issued",
    "credible_structure_family_count",
    "disagreement_score_pp",
    "candidate_endpoint_absolute_error_pp",
    "candidate_trajectory_iae_pp",
    "catastrophic_error",
)
RANDOM_COLUMNS = (
    "ranking_index",
    "status",
    "issued_count",
    "catastrophic_count",
    "catastrophic_rate",
)
BOOTSTRAP_COLUMNS = (
    "replicate",
    "status",
    "hard_eligible_count",
    "random_expected_catastrophic_rate",
    "issued_catastrophic_rate",
    "risk_reduction_fraction",
)
MATCHED_PAIR_COLUMNS = (
    "protocol_id",
    "pair_id",
    "left_cluster_id",
    "right_cluster_id",
    "left_family",
    "right_family",
    "latent_prefix_rmse_pp",
    "latent_prefix_max_abs_difference_pp",
    "truth_separation_25y_pp",
    "max_forecast_truth_separation_pp",
)
MATCHED_SCORE_COLUMNS = (
    "pair_id",
    "left_disagreement_score_pp",
    "right_disagreement_score_pp",
    "left_exceeds_threshold",
    "right_exceeds_threshold",
    "both_members_rejected",
)


class EvidenceVerificationError(ValueError):
    """Published evidence is incomplete, inconsistent, or tampered."""


def _fail(message: str) -> None:
    raise EvidenceVerificationError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail(f"JSON contains nonfinite value: {value}")


def _require_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{context} contains a nonfinite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, context=f"{context}[{index}]")


def _loads_json_strict(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError(f"{context} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _fail(f"{context} must contain one JSON object")
    _require_finite_json(value, context=context)
    return value


def _load_json_strict(path: Path) -> dict[str, Any]:
    try:
        return _loads_json_strict(path.read_bytes(), context=path.name)
    except OSError as exc:
        raise EvidenceVerificationError(f"Cannot read {path}: {exc}") from exc


def _require_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        _fail(
            f"{context} keys changed; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _canonical_relative_path(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        _fail(f"{context} is not a canonical relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.drive
        or value != posix.as_posix()
        or any(part in {".", ".."} for part in posix.parts)
    ):
        _fail(f"{context} is not a canonical relative path")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceVerificationError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{context} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def _entry_map(
    values: Any,
    *,
    context: str,
    extra_keys: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        _fail(f"{context} must be an array")
    expected_keys = {"path", "byte_count", "sha256", *extra_keys}
    result: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            _fail(f"{context}[{index}] must be an object")
        _require_keys(item, expected_keys, context=f"{context}[{index}]")
        path = _canonical_relative_path(
            item["path"], context=f"{context}[{index}].path"
        )
        if path in result:
            _fail(f"{context} contains duplicate path: {path}")
        _integer(item["byte_count"], context=f"{context}[{index}].byte_count")
        _sha256(item["sha256"], context=f"{context}[{index}].sha256")
        for key in extra_keys:
            if not isinstance(item[key], str) or not item[key]:
                _fail(f"{context}[{index}].{key} must be a non-empty string")
        result[path] = dict(item)
        order.append(path)
    if order != sorted(order):
        _fail(f"{context} must be sorted by path")
    return result


def _verify_file(path: Path, entry: Mapping[str, Any], *, context: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{context} is missing or is not a regular file")
    if path.stat().st_size != entry["byte_count"]:
        _fail(f"{context} byte count differs from its manifest")
    if _sha256_path(path) != entry["sha256"]:
        _fail(f"{context} SHA-256 differs from its manifest")


def _core_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: entry[key] for key in ("path", "byte_count", "sha256")}


def _verify_manifests(evidence_dir: Path) -> dict[str, Any]:
    evidence_path = evidence_dir / "evidence_manifest.json"
    full_path = evidence_dir / "full_bundle_manifest.json"
    if _sha256_path(evidence_path) != EXPECTED_EVIDENCE_MANIFEST_SHA256:
        _fail("Evidence manifest differs from the verifier-pinned digest")
    if _sha256_path(full_path) != EXPECTED_FULL_BUNDLE_MANIFEST_SHA256:
        _fail("Full-bundle manifest differs from the verifier-pinned digest")
    evidence = _load_json_strict(evidence_path)
    full = _load_json_strict(full_path)
    _require_keys(
        evidence,
        {
            "schema_version",
            "protocol_id",
            "result_status",
            "claim_boundary",
            "execution_git_commit",
            "tracked_original_artifacts",
            "omitted_from_git_but_listed_in_full_bundle_manifest",
            "full_bundle",
            "preoutcome_void_attempt",
        },
        context="evidence_manifest",
    )
    _require_keys(
        full,
        {
            "schema_version",
            "protocol_id",
            "artifact_set",
            "file_count",
            "byte_count",
            "canonical_entries_byte_count",
            "canonical_entries_encoding",
            "canonical_entries_sha256",
            "entries",
        },
        context="full_bundle_manifest",
    )
    if evidence["schema_version"] != "1.0.0" or full["schema_version"] != "1.0.0":
        _fail("Published evidence manifest schema version changed")
    if evidence["protocol_id"] != PROTOCOL_ID or full["protocol_id"] != PROTOCOL_ID:
        _fail("Published evidence protocol ID changed")
    if evidence["claim_boundary"] != EXPECTED_CLAIM_BOUNDARY:
        _fail("Published evidence claim boundary was weakened or changed")
    if full["artifact_set"] != "canonical_successful_formal_run":
        _fail("Full bundle no longer identifies the canonical formal execution")

    tracked = _entry_map(
        evidence["tracked_original_artifacts"],
        context="tracked_original_artifacts",
    )
    omitted = _entry_map(
        evidence["omitted_from_git_but_listed_in_full_bundle_manifest"],
        context="omitted_from_git_but_listed_in_full_bundle_manifest",
        extra_keys=("reason",),
    )
    full_entries = _entry_map(full["entries"], context="full_bundle.entries")
    if set(full_entries) != set(tracked) | set(omitted):
        _fail("Full bundle entries differ from tracked plus explicitly omitted files")
    for path, entry in tracked.items():
        if _core_entry(entry) != _core_entry(full_entries[path]):
            _fail(f"Tracked and full-bundle metadata differ for {path}")
    for path, entry in omitted.items():
        if _core_entry(entry) != _core_entry(full_entries[path]):
            _fail(f"Omitted and full-bundle metadata differ for {path}")
        if (evidence_dir / path).exists():
            _fail(
                f"Omitted full-bundle file unexpectedly appears in Git evidence: {path}"
            )

    canonical_entries = json.dumps(
        full["entries"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if full["canonical_entries_encoding"] != (
        "json.dumps(entries,sort_keys=True,separators=(comma,colon)).encode(utf-8)"
    ):
        _fail("Full-bundle canonical entry encoding changed")
    if full["canonical_entries_byte_count"] != len(canonical_entries):
        _fail("Full-bundle canonical entry byte count changed")
    if full["canonical_entries_sha256"] != _sha256_bytes(canonical_entries):
        _fail("Full-bundle canonical entry digest changed")
    if full["file_count"] != len(full_entries):
        _fail("Full-bundle file count changed")
    if full["byte_count"] != sum(
        entry["byte_count"] for entry in full_entries.values()
    ):
        _fail("Full-bundle aggregate byte count changed")

    full_summary = evidence["full_bundle"]
    if not isinstance(full_summary, dict):
        _fail("evidence_manifest.full_bundle must be an object")
    _require_keys(
        full_summary,
        {
            "release_asset_name",
            "file_count",
            "byte_count",
            "canonical_entries_sha256",
            "score_report_sha256",
            "exposure_log_sha256",
            "release_asset_byte_count",
            "release_asset_sha256",
        },
        context="evidence_manifest.full_bundle",
    )
    if (
        not isinstance(full_summary["release_asset_name"], str)
        or not full_summary["release_asset_name"]
    ):
        _fail("Full release-asset name is missing")
    _integer(
        full_summary["release_asset_byte_count"],
        context="evidence_manifest.full_bundle.release_asset_byte_count",
        minimum=1,
    )
    _sha256(
        full_summary["release_asset_sha256"],
        context="evidence_manifest.full_bundle.release_asset_sha256",
    )
    if (
        full_summary["file_count"] != full["file_count"]
        or full_summary["byte_count"] != full["byte_count"]
        or full_summary["canonical_entries_sha256"] != full["canonical_entries_sha256"]
        or full_summary["score_report_sha256"]
        != full_entries["score_report.json"]["sha256"]
        or full_summary["exposure_log_sha256"]
        != full_entries["exposure_log.json"]["sha256"]
    ):
        _fail("Evidence and full-bundle manifest summaries disagree")

    void_summary = evidence["preoutcome_void_attempt"]
    if not isinstance(void_summary, dict):
        _fail("preoutcome_void_attempt must be an object")
    _require_keys(
        void_summary,
        {
            "status",
            "reason",
            "included_in_evidence",
            "prediction_commitment_created",
            "truth_opened_before_void",
            "same_execution_git_commit",
            "successful_exact_rerun_truth_commitment_match",
            "successful_exact_rerun_generated_bundle_byte_matches",
            "files",
        },
        context="preoutcome_void_attempt",
    )
    void_files = _entry_map(
        void_summary["files"], context="preoutcome_void_attempt.files"
    )
    if set(void_files) != {
        "preoutcome_void_attempt/environment.json",
        "preoutcome_void_attempt/exposure_log.json",
        "preoutcome_void_attempt/truth_commitment.json",
    }:
        _fail("Pre-outcome void evidence file set changed")

    expected_physical = {
        "evidence_manifest.json",
        "full_bundle_manifest.json",
        *tracked,
        *void_files,
    }
    observed_physical = {
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_file()
    }
    if observed_physical != expected_physical:
        _fail(
            "Published evidence file set changed; "
            f"missing={sorted(expected_physical - observed_physical)}, "
            f"extra={sorted(observed_physical - expected_physical)}"
        )
    for path, entry in tracked.items():
        _verify_file(evidence_dir / path, entry, context=path)
    for path, entry in void_files.items():
        _verify_file(evidence_dir / path, entry, context=path)

    return {
        "evidence": evidence,
        "full": full,
        "tracked": tracked,
        "omitted": omitted,
        "full_entries": full_entries,
        "void_files": void_files,
        "manifest_sha256": {
            "evidence_manifest.json": EXPECTED_EVIDENCE_MANIFEST_SHA256,
            "full_bundle_manifest.json": EXPECTED_FULL_BUNDLE_MANIFEST_SHA256,
        },
    }


def _git_blob(project_root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "show", f"{commit}:{path}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"Cannot read frozen Git blob {commit}:{path}: {message}")
    return completed.stdout


def _verify_execution_sources(
    project_root: Path,
    environment: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    commit = environment.get("git_commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or evidence.get("execution_git_commit") != commit
    ):
        _fail("Execution Git commit is missing, malformed, or inconsistent")
    if environment.get("git_status_porcelain") != "":
        _fail("Formal execution did not record a clean Git worktree")
    source_hashes = environment.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != EXECUTION_SOURCE_PATHS
    ):
        _fail("Frozen execution source path set changed")
    observed: dict[str, str] = {}
    config_blob: bytes | None = None
    for path in sorted(EXECUTION_SOURCE_PATHS):
        expected = _sha256(source_hashes[path], context=f"source_sha256.{path}")
        blob = _git_blob(project_root, commit, path)
        digest = _sha256_bytes(blob)
        if digest != expected:
            _fail(f"Git blob differs from formal execution source hash: {path}")
        observed[path] = digest
        if path == CONFIG_PATH:
            config_blob = blob
    tree = _sha256_bytes(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    if environment.get("source_tree_sha256") != tree:
        _fail("Frozen execution source-tree digest changed")
    assert config_blob is not None
    if environment.get("config_byte_sha256") != _sha256_bytes(config_blob):
        _fail("Frozen config byte digest changed")
    config = _loads_json_strict(config_blob, context="frozen config Git blob")
    canonical = _sha256_bytes(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    if environment.get("config_canonical_sha256") != canonical:
        _fail("Frozen config canonical digest changed")
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or config.get("status") != "frozen_before_first_simulation_execution"
    ):
        _fail("Git-frozen v0.14 protocol identity or freeze status changed")
    return {"commit": commit, "source_tree_sha256": tree, "config": config}


def _parse_utc(value: Any, *, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(f"{context} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EvidenceVerificationError(f"{context} is not an ISO timestamp") from exc
    return parsed


def _verify_exposure(
    exposure: Mapping[str, Any],
    environment: Mapping[str, Any],
    truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    _require_keys(
        exposure,
        {
            "protocol_id",
            "events",
            "protocol_deviations",
            "truth_pack_opened_before_prediction_commitment",
        },
        context="exposure_log",
    )
    if (
        exposure["protocol_id"] != PROTOCOL_ID
        or exposure["protocol_deviations"] != []
        or exposure["truth_pack_opened_before_prediction_commitment"] is not False
    ):
        _fail("Exposure log reports a protocol deviation or premature truth access")
    events = exposure["events"]
    if not isinstance(events, list) or len(events) != len(EXPECTED_EXPOSURE_EVENTS):
        _fail("Exposure event count changed")
    event_key_sets = (
        {"sequence", "event", "created_utc", "git_commit", "source_tree_sha256"},
        {
            "sequence",
            "event",
            "created_utc",
            "truth_pack_byte_sha256",
            "predictor_received_truth_path",
        },
        {
            "sequence",
            "event",
            "created_utc",
            "prediction_bundle_byte_sha256",
            "decision_bundle_byte_sha256",
            "truth_pack_opened_before_commitment",
        },
        {
            "sequence",
            "event",
            "created_utc",
            "truth_pack_opened_before_prediction_commitment",
        },
        {
            "sequence",
            "event",
            "created_utc",
            "result_status",
            "truth_pack_opened_before_prediction_commitment",
        },
    )
    times: list[datetime] = []
    for index, (event, expected_name, expected_keys) in enumerate(
        zip(events, EXPECTED_EXPOSURE_EVENTS, event_key_sets, strict=True), start=1
    ):
        if not isinstance(event, dict):
            _fail(f"Exposure event {index} must be an object")
        _require_keys(event, expected_keys, context=f"exposure event {index}")
        if event["sequence"] != index or event["event"] != expected_name:
            _fail("Exposure sequence or event name changed")
        times.append(_parse_utc(event["created_utc"], context=f"event {index}"))
    if any(right <= left for left, right in zip(times, times[1:])):
        _fail("Exposure timestamps are not strictly increasing")
    if (
        events[0]["git_commit"] != environment["git_commit"]
        or events[0]["source_tree_sha256"] != environment["source_tree_sha256"]
        or events[1]["truth_pack_byte_sha256"] != truth["truth_pack_byte_sha256"]
        or events[1]["predictor_received_truth_path"] is not False
        or events[2]["prediction_bundle_byte_sha256"]
        != prediction["prediction_bundle_byte_sha256"]
        or events[2]["decision_bundle_byte_sha256"]
        != prediction["decision_bundle_byte_sha256"]
        or events[2]["truth_pack_opened_before_commitment"] is not False
        or events[3]["truth_pack_opened_before_prediction_commitment"] is not False
        or events[4]["truth_pack_opened_before_prediction_commitment"] is not False
        or events[4]["result_status"] != report["status"]
    ):
        _fail("Exposure events disagree with the frozen commitments")
    ordered_times = (
        _parse_utc(environment["started_utc"], context="environment.started_utc"),
        times[0],
        _parse_utc(truth["created_utc"], context="truth_commitment.created_utc"),
        times[1],
        _parse_utc(
            prediction["created_utc"], context="prediction_commitment.created_utc"
        ),
        times[2],
        times[3],
        _parse_utc(report["scored_utc"], context="score_report.scored_utc"),
        times[4],
    )
    if any(right <= left for left, right in zip(ordered_times, ordered_times[1:])):
        _fail(
            "Environment, commitment, truth-release, and score times are out of order"
        )


def _verify_void_attempt(
    evidence_dir: Path,
    summary: Mapping[str, Any],
    formal_environment: Mapping[str, Any],
    formal_truth: Mapping[str, Any],
) -> None:
    if (
        summary["status"] != "void_before_prediction_commitment"
        or summary["included_in_evidence"] is not False
        or summary["prediction_commitment_created"] is not False
        or summary["truth_opened_before_void"] is not False
        or summary["same_execution_git_commit"] is not True
        or summary["successful_exact_rerun_truth_commitment_match"] is not True
    ):
        _fail("Pre-outcome void attempt summary changed")
    bundle_matches = summary["successful_exact_rerun_generated_bundle_byte_matches"]
    if not isinstance(bundle_matches, dict) or bundle_matches != {
        "forecast_coordinates.csv": True,
        "matched_prefix_pairs.csv": True,
        "prefix_pack.csv": True,
        "truth_pack.csv": True,
    }:
        _fail("Exact-rerun generation match record changed")
    void_dir = evidence_dir / "preoutcome_void_attempt"
    void_environment = _load_json_strict(void_dir / "environment.json")
    void_truth = _load_json_strict(void_dir / "truth_commitment.json")
    void_exposure = _load_json_strict(void_dir / "exposure_log.json")
    if {
        key: value for key, value in void_environment.items() if key != "started_utc"
    } != {
        key: value for key, value in formal_environment.items() if key != "started_utc"
    }:
        _fail("Void and successful-run frozen environments differ")
    if {key: value for key, value in void_truth.items() if key != "created_utc"} != {
        key: value for key, value in formal_truth.items() if key != "created_utc"
    }:
        _fail("Void and successful-run truth commitments differ")
    _require_keys(
        void_exposure,
        {
            "protocol_id",
            "events",
            "protocol_deviations",
            "truth_pack_opened_before_prediction_commitment",
        },
        context="void exposure_log",
    )
    events = void_exposure["events"]
    if (
        void_exposure["protocol_id"] != PROTOCOL_ID
        or void_exposure["protocol_deviations"] != []
        or void_exposure["truth_pack_opened_before_prediction_commitment"] is not False
        or not isinstance(events, list)
        or [event.get("sequence") for event in events] != [1, 2, 3]
        or [event.get("event") for event in events]
        != [
            "environment_and_code_frozen",
            "sealed_truth_generated_and_committed",
            "execution_void_due_to_exception",
        ]
        or events[0].get("git_commit") != formal_environment["git_commit"]
        or events[1].get("truth_pack_byte_sha256")
        != formal_truth["truth_pack_byte_sha256"]
        or events[1].get("predictor_received_truth_path") is not False
    ):
        _fail("Pre-outcome void exposure record is inconsistent")


def _read_csv(path: Path, columns: tuple[str, ...], *, rows: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:
        raise EvidenceVerificationError(f"Cannot parse {path.name} as CSV") from exc
    if tuple(frame.columns) != columns:
        _fail(f"{path.name} columns changed")
    if len(frame) != rows:
        _fail(f"{path.name} row count changed: expected {rows}, observed {len(frame)}")
    return frame


def _numeric(
    values: pd.Series,
    *,
    context: str,
    finite: bool = True,
) -> np.ndarray:
    try:
        result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise EvidenceVerificationError(f"{context} must be numeric") from exc
    if np.isnan(result).any() or (finite and not np.isfinite(result).all()):
        _fail(f"{context} contains a prohibited nonfinite value")
    return result


def _booleans(values: pd.Series, *, context: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    mapped = values.map({"True": True, "False": False})
    if mapped.isna().any():
        _fail(f"{context} must contain only True or False")
    return mapped.to_numpy(dtype=bool)


def _assert_close(observed: Any, expected: float, *, context: str) -> None:
    if isinstance(observed, bool):
        _fail(f"{context} must be numeric")
    try:
        numeric = float(observed)
    except (TypeError, ValueError) as exc:
        raise EvidenceVerificationError(f"{context} must be numeric") from exc
    if not math.isfinite(numeric) or not math.isclose(
        numeric, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        _fail(f"{context} differs from independent recomputation")


def _same_float_arrays(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.all(
            (np.isfinite(left) & np.isfinite(right) & np.isclose(left, right, 0.0, 0.0))
            | (np.isposinf(left) & np.isposinf(right))
            | (np.isneginf(left) & np.isneginf(right))
        )
    )


def _tie_digest(protocol_id: str, prefix_hash: str) -> str:
    return hashlib.sha256(f"{protocol_id}|{prefix_hash}".encode("utf-8")).hexdigest()


def _validate_decisions(
    decision: pd.DataFrame,
    trajectory: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    keys = ["partition", "cluster_id"]
    if decision.duplicated(keys).any() or trajectory.duplicated(keys).any():
        _fail("Decision or trajectory cluster keys are duplicated")
    if set(decision.loc[:, keys].itertuples(index=False, name=None)) != set(
        trajectory.loc[:, keys].itertuples(index=False, name=None)
    ):
        _fail("Decision and trajectory cluster keys differ")
    if not decision["protocol_id"].astype(str).eq(PROTOCOL_ID).all():
        _fail("Decision bundle protocol ID changed")
    prefix_hashes = decision["canonical_prefix_content_sha256"].astype(str)
    if not prefix_hashes.map(
        lambda value: SHA256_PATTERN.fullmatch(value) is not None
    ).all():
        _fail("Decision bundle contains a malformed prefix-content digest")

    merged = trajectory.merge(
        decision,
        on=keys,
        how="inner",
        suffixes=("_trajectory", "_decision"),
        validate="one_to_one",
    )
    for name in ("hard_eligible", "primary_issued"):
        left = _booleans(merged[f"{name}_trajectory"], context=f"trajectory {name}")
        right = _booleans(merged[f"{name}_decision"], context=f"decision {name}")
        if not np.array_equal(left, right):
            _fail(f"Trajectory and decision {name} values differ")
    credible_left = _numeric(
        merged["credible_structure_family_count_trajectory"],
        context="trajectory credible family count",
    )
    credible_right = _numeric(
        merged["credible_structure_family_count_decision"],
        context="decision credible family count",
    )
    if not np.array_equal(credible_left, credible_right):
        _fail("Trajectory and decision credible-family counts differ")
    disagreement_left = _numeric(
        merged["disagreement_score_pp_trajectory"],
        context="trajectory disagreement",
        finite=False,
    )
    disagreement_right = _numeric(
        merged["disagreement_score_pp_decision"],
        context="decision disagreement",
        finite=False,
    )
    if not _same_float_arrays(disagreement_left, disagreement_right):
        _fail("Trajectory and decision disagreement scores differ")
    endpoint = _numeric(
        merged["candidate_endpoint_absolute_error_pp"],
        context="candidate endpoint error",
    )
    _numeric(merged["candidate_trajectory_iae_pp"], context="candidate trajectory IAE")
    catastrophic = _booleans(
        merged["catastrophic_error"], context="trajectory catastrophic_error"
    )
    if not np.array_equal(catastrophic, endpoint >= 5.0):
        _fail("Catastrophic-error labels differ from the frozen >=5 pp definition")
    hard = _booleans(merged["hard_eligible_decision"], context="decision hard_eligible")
    if not np.array_equal(
        hard, (credible_right >= 2) & np.isfinite(disagreement_right)
    ):
        _fail("Hard eligibility differs from the frozen finite-disagreement rule")

    partitions = config["partitions"]
    family_ids = [
        item["family_id"] for item in config["truth_generation"]["truth_families"]
    ]
    expected_partition_counts = dict(partitions["total_cluster_counts"])
    pair_count = int(
        config["matched_prefix_counterexample_audit"]["required_total_pair_clusters"]
    )
    expected_partition_counts["matched_prefix_counterexamples"] = 2 * pair_count
    observed_counts = decision.groupby("partition").size().to_dict()
    if observed_counts != expected_partition_counts:
        _fail("Published decision partition counts differ from the frozen protocol")
    truth_counts = trajectory.groupby(["partition", "truth_family"]).size().to_dict()
    for partition, per_family in partitions["cluster_counts_per_truth_family"].items():
        for family in family_ids:
            if truth_counts.get((partition, family)) != int(per_family):
                _fail(f"Truth-family count changed for {partition}:{family}")
    matched_expected = {
        ("matched_prefix_counterexamples", "single_power"): pair_count,
        ("matched_prefix_counterexamples", "late_knee"): pair_count,
    }
    matched_observed = {
        key: value
        for key, value in truth_counts.items()
        if key[0] == "matched_prefix_counterexamples"
    }
    if matched_observed != matched_expected:
        _fail("Matched-prefix truth-family counts changed")

    decision = decision.copy()
    decision["_hard"] = _booleans(
        decision["hard_eligible"], context="decision hard_eligible"
    )
    decision["_issued"] = _booleans(
        decision["primary_issued"], context="decision primary_issued"
    )
    for partition in ("calibration", "test", "audit"):
        rows = decision.loc[decision["partition"].eq(partition)].copy()
        eligible = rows.loc[rows["_hard"]].copy()
        eligible["_tie"] = [
            _tie_digest(PROTOCOL_ID, value)
            for value in eligible["canonical_prefix_content_sha256"].astype(str)
        ]
        if eligible["_tie"].duplicated().any():
            _fail(f"{partition} prefix tie-break digest collision")
        eligible = eligible.sort_values(
            ["disagreement_score_pp", "_tie"], kind="stable"
        )
        expected_ranks = np.arange(1, len(eligible) + 1, dtype=float)
        observed_ranks = _numeric(
            eligible["primary_issuance_rank"], context=f"{partition} issuance rank"
        )
        if not np.array_equal(observed_ranks, expected_ranks):
            _fail(f"{partition} issuance ranks differ from the frozen ranking")
        ineligible_ranks = rows.loc[~rows["_hard"], "primary_issuance_rank"]
        if ineligible_ranks.notna().any():
            _fail(f"{partition} ineligible clusters unexpectedly have ranks")
        target = 0
        if partition in {"test", "audit"}:
            target = int(
                round(
                    int(expected_partition_counts[partition])
                    * float(
                        config["candidate"]["primary_issuance_policy"][
                            "target_issuance_fraction"
                        ]
                    )
                )
            )
        expected_issued = set(eligible.iloc[:target]["cluster_id"].astype(str))
        observed_issued = set(rows.loc[rows["_issued"], "cluster_id"].astype(str))
        if observed_issued != expected_issued:
            _fail(f"{partition} issued set differs from the frozen ranking")
    for partition in ("development", "matched_prefix_counterexamples"):
        rows = decision.loc[decision["partition"].eq(partition)]
        if rows["primary_issuance_rank"].notna().any() or rows["_issued"].any():
            _fail(f"{partition} must not contain batch-issued clusters")
    return merged


def _policy_summary(merged: pd.DataFrame, partition: str) -> dict[str, Any]:
    rows = merged.loc[merged["partition"].eq(partition)]
    hard = _booleans(
        rows["hard_eligible_decision"], context=f"{partition} hard eligibility"
    )
    issued_mask = _booleans(
        rows["primary_issued_decision"], context=f"{partition} issuance"
    )
    catastrophic = _booleans(
        rows["catastrophic_error"], context=f"{partition} catastrophic error"
    )
    eligible = catastrophic[hard]
    issued = catastrophic[issued_mask]
    if not len(eligible) or not len(issued):
        _fail(f"{partition} policy summary is not evaluable")
    random_expected = float(eligible.mean())
    issued_rate = float(issued.mean())
    if random_expected <= 0.0:
        _fail(f"{partition} analytic random risk is not positive")
    return {
        "cluster_count": int(len(rows)),
        "hard_eligible_count": int(len(eligible)),
        "hard_eligible_catastrophic_count": int(eligible.sum()),
        "issued_count": int(len(issued)),
        "issued_catastrophic_count": int(issued.sum()),
        "issued_catastrophic_rate": issued_rate,
        "analytic_random_expected_catastrophic_rate": random_expected,
        "analytic_risk_reduction_fraction": 1.0 - issued_rate / random_expected,
    }


def _verify_policy_report(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    for key in (
        "cluster_count",
        "hard_eligible_count",
        "hard_eligible_catastrophic_count",
        "issued_count",
        "issued_catastrophic_count",
    ):
        if observed.get(key) != expected[key]:
            _fail(f"{context}.{key} differs from independent recomputation")
    for key in (
        "issued_catastrophic_rate",
        "analytic_random_expected_catastrophic_rate",
        "analytic_risk_reduction_fraction",
    ):
        _assert_close(
            observed.get(key), float(expected[key]), context=f"{context}.{key}"
        )


def _family_reversals(
    merged: pd.DataFrame, partition: str, family_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    partition_rows = merged.loc[merged["partition"].eq(partition)]
    all_records: list[dict[str, Any]] = []
    reversals: list[dict[str, Any]] = []
    for family in family_ids:
        rows = partition_rows.loc[partition_rows["truth_family"].eq(family)]
        hard = _booleans(rows["hard_eligible_decision"], context="family eligibility")
        issued = _booleans(rows["primary_issued_decision"], context="family issuance")
        catastrophic = _booleans(
            rows["catastrophic_error"], context="family catastrophic error"
        )
        eligible_catastrophic = catastrophic[hard]
        issued_catastrophic = catastrophic[issued]
        if not len(eligible_catastrophic) or not len(issued_catastrophic):
            _fail(f"Family reversal is not evaluable for {partition}:{family}")
        random_expected = float(eligible_catastrophic.mean())
        issued_rate = float(issued_catastrophic.mean())
        reduction = 1.0 - issued_rate / random_expected
        record = {
            "partition": partition,
            "truth_family": family,
            "hard_eligible_count": int(len(eligible_catastrophic)),
            "issued_count": int(len(issued_catastrophic)),
            "issued_catastrophic_count": int(issued_catastrophic.sum()),
            "issued_catastrophic_rate": issued_rate,
            "analytic_random_expected_catastrophic_rate": random_expected,
            "issued_vs_analytic_random_risk_reduction_fraction": reduction,
            "family_specific_reversal": bool(issued_rate > random_expected),
        }
        all_records.append(record)
        if record["family_specific_reversal"]:
            reversals.append(record)
    return all_records, reversals


def _compare_reversal_records(
    observed: Any, expected: list[dict[str, Any]], *, context: str
) -> None:
    if not isinstance(observed, list) or len(observed) != len(expected):
        _fail(f"{context} count differs from independent recomputation")
    observed_by_key = {
        (item.get("partition"), item.get("truth_family")): item
        for item in observed
        if isinstance(item, dict)
    }
    expected_by_key = {
        (item["partition"], item["truth_family"]): item for item in expected
    }
    if set(observed_by_key) != set(expected_by_key):
        _fail(f"{context} identities differ from independent recomputation")
    for key, record in expected_by_key.items():
        item = observed_by_key[key]
        for count_key in (
            "hard_eligible_count",
            "issued_count",
            "issued_catastrophic_count",
            "family_specific_reversal",
        ):
            if item.get(count_key) != record[count_key]:
                _fail(f"{context} field changed for {key}: {count_key}")
        for numeric_key in (
            "issued_catastrophic_rate",
            "analytic_random_expected_catastrophic_rate",
            "issued_vs_analytic_random_risk_reduction_fraction",
        ):
            _assert_close(
                item.get(numeric_key),
                float(record[numeric_key]),
                context=f"{context}.{key}.{numeric_key}",
            )


def _verify_metrics(
    evidence_dir: Path,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ordinary_total = sum(
        int(value) for value in config["partitions"]["total_cluster_counts"].values()
    )
    pair_count = int(
        config["matched_prefix_counterexample_audit"]["required_total_pair_clusters"]
    )
    total_clusters = ordinary_total + 2 * pair_count
    decision = _read_csv(
        evidence_dir / "decision_bundle.csv", DECISION_COLUMNS, rows=total_clusters
    )
    trajectory = _read_csv(
        evidence_dir / "trajectory_scores.csv", TRAJECTORY_COLUMNS, rows=total_clusters
    )
    merged = _validate_decisions(decision, trajectory, config)
    decision = decision.copy()
    decision["_hard"] = _booleans(
        decision["hard_eligible"], context="decision hard_eligible"
    )
    decision["_issued"] = _booleans(
        decision["primary_issued"], context="decision primary_issued"
    )
    test = _policy_summary(merged, "test")
    audit = _policy_summary(merged, "audit")
    _verify_policy_report(report["test_policy"], test, context="test_policy")
    _verify_policy_report(report["audit_policy"], audit, context="audit_policy")

    random_rows = 10_000
    random_frame = _read_csv(
        evidence_dir / "random_rejection.csv", RANDOM_COLUMNS, rows=random_rows
    )
    ranking_index = _numeric(
        random_frame["ranking_index"], context="random ranking index"
    )
    if not np.array_equal(ranking_index, np.arange(random_rows, dtype=float)):
        _fail("Random ranking indices changed")
    if not random_frame["status"].astype(str).eq("defined").all():
        _fail("Published random rankings are not all defined")
    issue_count = test["issued_count"]
    issued_count = _numeric(random_frame["issued_count"], context="random issued count")
    catastrophic_count = _numeric(
        random_frame["catastrophic_count"], context="random catastrophic count"
    )
    rates = _numeric(random_frame["catastrophic_rate"], context="random risk")
    if (
        not np.all(issued_count == issue_count)
        or np.any(catastrophic_count != np.floor(catastrophic_count))
        or np.any((catastrophic_count < 0) | (catastrophic_count > issue_count))
        or not np.allclose(
            rates, catastrophic_count / issue_count, rtol=0.0, atol=1e-15
        )
    ):
        _fail("Random ranking counts or rates are internally inconsistent")
    random_mean = float(rates.mean())
    published_random_reduction = 1.0 - test["issued_catastrophic_rate"] / random_mean
    test_report = report["test_policy"]
    if (
        test_report.get("published_random_ranking_count") != random_rows
        or test_report.get("random_rankings_fully_defined") is not True
    ):
        _fail("Score report random-ranking status changed")
    _assert_close(
        test_report.get("published_random_mean_catastrophic_rate"),
        random_mean,
        context="test_policy.published_random_mean_catastrophic_rate",
    )
    _assert_close(
        test_report.get("risk_reduction_fraction"),
        published_random_reduction,
        context="test_policy.risk_reduction_fraction",
    )

    bootstrap_rows = int(config["endpoints"]["bootstrap"]["resamples"])
    bootstrap = _read_csv(
        evidence_dir / "bootstrap.csv", BOOTSTRAP_COLUMNS, rows=bootstrap_rows
    )
    replicates = _numeric(bootstrap["replicate"], context="bootstrap replicate")
    if not np.array_equal(replicates, np.arange(bootstrap_rows, dtype=float)):
        _fail("Bootstrap replicate indices changed")
    if not bootstrap["status"].astype(str).eq("defined").all():
        _fail("Published bootstrap is not fully defined")
    eligible_count = _numeric(
        bootstrap["hard_eligible_count"], context="bootstrap eligible count"
    )
    random_expected = _numeric(
        bootstrap["random_expected_catastrophic_rate"],
        context="bootstrap random risk",
    )
    issued_risk = _numeric(
        bootstrap["issued_catastrophic_rate"], context="bootstrap issued risk"
    )
    reductions = _numeric(
        bootstrap["risk_reduction_fraction"], context="bootstrap risk reduction"
    )
    if (
        np.any(eligible_count < issue_count)
        or np.any(random_expected <= 0.0)
        or np.any((issued_risk < 0.0) | (issued_risk > 1.0))
        or not np.allclose(
            reductions,
            1.0 - issued_risk / random_expected,
            rtol=1e-13,
            atol=1e-13,
        )
    ):
        _fail("Bootstrap rows are internally inconsistent")
    lower_bound = float(np.quantile(reductions, 0.05, method="linear"))
    if (
        test_report.get("bootstrap_resamples") != bootstrap_rows
        or test_report.get("bootstrap_defined_resample_count") != bootstrap_rows
    ):
        _fail("Score report bootstrap counts changed")
    _assert_close(
        test_report.get("bootstrap_one_sided_95pct_lower_bound"),
        lower_bound,
        context="test_policy.bootstrap_one_sided_95pct_lower_bound",
    )

    mapping = _read_csv(
        evidence_dir / "matched_prefix_pairs.csv", MATCHED_PAIR_COLUMNS, rows=pair_count
    )
    pair_scores = _read_csv(
        evidence_dir / "matched_pair_scores.csv", MATCHED_SCORE_COLUMNS, rows=pair_count
    )
    if (
        mapping["pair_id"].duplicated().any()
        or pair_scores["pair_id"].duplicated().any()
        or set(mapping["pair_id"].astype(str))
        != set(pair_scores["pair_id"].astype(str))
        or not mapping["protocol_id"].astype(str).eq(PROTOCOL_ID).all()
        or not mapping["left_family"].astype(str).eq("single_power").all()
        or not mapping["right_family"].astype(str).eq("late_knee").all()
    ):
        _fail("Matched-prefix pair identities or family mapping changed")
    left_ids = mapping["left_cluster_id"].astype(str)
    right_ids = mapping["right_cluster_id"].astype(str)
    mapped_ids = [*left_ids, *right_ids]
    if (
        left_ids.duplicated().any()
        or right_ids.duplicated().any()
        or len(set(mapped_ids)) != 2 * pair_count
    ):
        _fail("Every matched-prefix member must occur in exactly one pair")
    matched_truth = trajectory.loc[
        trajectory["partition"].eq("matched_prefix_counterexamples")
    ].set_index("cluster_id")["truth_family"]
    if set(mapped_ids) != set(matched_truth.index.astype(str)):
        _fail("Matched mapping does not cover exactly the 400 matched clusters")
    if (
        not matched_truth.loc[left_ids].astype(str).eq("single_power").all()
        or not matched_truth.loc[right_ids].astype(str).eq("late_knee").all()
    ):
        _fail("Matched mapping side labels disagree with scored truth families")
    match_config = config["matched_prefix_counterexample_audit"]
    prefix_rmse = _numeric(
        mapping["latent_prefix_rmse_pp"], context="matched latent prefix RMSE"
    )
    prefix_max = _numeric(
        mapping["latent_prefix_max_abs_difference_pp"],
        context="matched latent prefix maximum difference",
    )
    separation = _numeric(
        mapping["truth_separation_25y_pp"], context="matched 25-year separation"
    )
    max_separation = _numeric(
        mapping["max_forecast_truth_separation_pp"],
        context="matched maximum separation",
    )
    qualified = (
        (prefix_rmse <= float(match_config["latent_prefix_rmse_max_pp"]))
        & (
            prefix_max
            <= float(match_config["latent_prefix_max_absolute_difference_pp"])
        )
        & (separation >= float(match_config["minimum_25_year_truth_separation_pp"]))
        & (
            max_separation
            >= float(match_config["minimum_maximum_forecast_grid_truth_separation_pp"])
        )
    )
    qualified_count = int(qualified.sum())

    calibration = decision.loc[
        decision["partition"].eq("calibration") & decision["_hard"]
    ].copy()
    calibration_rank = _numeric(
        calibration["primary_issuance_rank"], context="calibration rank"
    )
    calibration = calibration.assign(_rank=calibration_rank)
    threshold_row = calibration.loc[calibration["_rank"].eq(250.0)]
    if len(threshold_row) != 1:
        _fail("Frozen rank-250 calibration threshold is unavailable")
    threshold = float(threshold_row.iloc[0]["disagreement_score_pp"])

    matched_decision = decision.loc[
        decision["partition"].eq("matched_prefix_counterexamples")
    ].set_index("cluster_id")
    ordered_scores = pair_scores.set_index("pair_id").loc[
        mapping["pair_id"].astype(str)
    ]
    try:
        left_decision = matched_decision.loc[
            mapping["left_cluster_id"].astype(str), "disagreement_score_pp"
        ].to_numpy(dtype=float)
        right_decision = matched_decision.loc[
            mapping["right_cluster_id"].astype(str), "disagreement_score_pp"
        ].to_numpy(dtype=float)
    except KeyError as exc:
        raise EvidenceVerificationError(
            "Matched mapping refers to an unknown decision cluster"
        ) from exc
    left_score = _numeric(
        ordered_scores["left_disagreement_score_pp"],
        context="matched left disagreement",
        finite=False,
    )
    right_score = _numeric(
        ordered_scores["right_disagreement_score_pp"],
        context="matched right disagreement",
        finite=False,
    )
    if not _same_float_arrays(left_decision, left_score) or not _same_float_arrays(
        right_decision, right_score
    ):
        _fail("Matched-pair scores differ from the published decisions")
    left_exceeds = np.isfinite(left_score) & (left_score > threshold)
    right_exceeds = np.isfinite(right_score) & (right_score > threshold)
    both = left_exceeds & right_exceeds
    if (
        not np.array_equal(
            left_exceeds,
            _booleans(
                ordered_scores["left_exceeds_threshold"],
                context="matched left threshold result",
            ),
        )
        or not np.array_equal(
            right_exceeds,
            _booleans(
                ordered_scores["right_exceeds_threshold"],
                context="matched right threshold result",
            ),
        )
        or not np.array_equal(
            both,
            _booleans(
                ordered_scores["both_members_rejected"],
                context="matched joint rejection",
            ),
        )
    ):
        _fail("Matched-pair rejection flags differ from strict thresholding")
    both_count = int(both.sum())
    both_fraction = float(both.mean())
    nonfinite_member_count = int(
        (~np.isfinite(np.column_stack((left_score, right_score)))).sum()
    )
    nonfinite_pair_count = int(
        np.any(~np.isfinite(np.column_stack((left_score, right_score))), axis=1).sum()
    )
    matched_report = report["matched_prefix_audit"]
    matched_expected = {
        "qualified_pair_count": qualified_count,
        "both_rejected_pair_count": both_count,
        "evaluated_pair_row_count": pair_count,
        "evaluated_member_count": 2 * pair_count,
        "nonfinite_disagreement_member_count": nonfinite_member_count,
        "model_failure_member_count": nonfinite_member_count,
        "pair_with_any_nonfinite_disagreement_count": nonfinite_pair_count,
        "pair_with_any_model_failure_count": nonfinite_pair_count,
    }
    for key, value in matched_expected.items():
        if matched_report.get(key) != value:
            _fail(f"matched_prefix_audit.{key} differs from recomputation")
    if matched_report.get("endpoint_available") is not True:
        _fail("Matched-prefix endpoint unexpectedly became unavailable")
    _assert_close(
        matched_report.get("calibration_disagreement_threshold_pp"),
        threshold,
        context="matched_prefix_audit.calibration_disagreement_threshold_pp",
    )
    _assert_close(
        matched_report.get("both_rejected_fraction"),
        both_fraction,
        context="matched_prefix_audit.both_rejected_fraction",
    )

    family_ids = [
        item["family_id"] for item in config["truth_generation"]["truth_families"]
    ]
    test_family, test_reversals = _family_reversals(merged, "test", family_ids)
    audit_family, audit_reversals = _family_reversals(merged, "audit", family_ids)
    reversals = [*test_reversals, *audit_reversals]
    secondary = report["secondary"]
    if secondary.get("family_specific_reversal_count") != len(reversals):
        _fail("Family-specific reversal count differs from recomputation")
    _compare_reversal_records(
        secondary.get("family_specific_reversals"),
        reversals,
        context="family_specific_reversals",
    )
    for report_key, records in (
        ("test_family_metrics", test_family),
        ("audit_family_metrics", audit_family),
    ):
        observed = {
            item.get("truth_family"): item
            for item in secondary.get(report_key, [])
            if isinstance(item, dict) and item.get("truth_family") != "__all__"
        }
        if set(observed) != set(family_ids):
            _fail(f"{report_key} family identities changed")
        for record in records:
            item = observed[record["truth_family"]]
            for key in (
                "hard_eligible_count",
                "issued_count",
                "issued_catastrophic_count",
                "family_specific_reversal",
            ):
                if item.get(key) != record[key]:
                    _fail(f"{report_key}.{record['truth_family']}.{key} changed")
            for key in (
                "issued_catastrophic_rate",
                "analytic_random_expected_catastrophic_rate",
                "issued_vs_analytic_random_risk_reduction_fraction",
            ):
                _assert_close(
                    item.get(key),
                    float(record[key]),
                    context=f"{report_key}.{record['truth_family']}.{key}",
                )

    mean_comparison = report["mean_forecast_comparison"]
    if mean_comparison.get("evaluable") is not True:
        _fail("Published IAE noninferiority endpoint is not evaluable")
    issued_test_rows = merged.loc[
        merged["partition"].eq("test") & merged["primary_issued_decision"].eq(True)  # noqa: E712
    ]
    recomputed_candidate_iae = float(
        _numeric(
            issued_test_rows["candidate_trajectory_iae_pp"],
            context="issued candidate trajectory IAE",
        ).mean()
    )
    candidate_iae = float(mean_comparison["candidate_issued_mean_trajectory_iae_pp"])
    baseline_iae = float(
        mean_comparison["baseline_on_same_issued_clusters_mean_trajectory_iae_pp"]
    )
    iae_delta = float(mean_comparison["candidate_minus_baseline_iae_pp"])
    if not all(
        math.isfinite(value) for value in (candidate_iae, baseline_iae, iae_delta)
    ):
        _fail("Published IAE endpoint contains a nonfinite value")
    _assert_close(
        candidate_iae,
        recomputed_candidate_iae,
        context="mean_forecast_comparison.candidate_issued_mean_trajectory_iae_pp",
    )
    baseline_ids = {
        item["model_id"] for item in config["comparators"]["mean_forecast_baselines"]
    }
    baseline_selection = report["calibration_baseline_selection"]
    if not isinstance(baseline_selection, list) or len(baseline_selection) != len(
        baseline_ids
    ):
        _fail("Calibration baseline-selection table changed")
    calibration_means: dict[str, float] = {}
    for item in baseline_selection:
        if not isinstance(item, dict) or item.get("model_id") not in baseline_ids:
            _fail("Calibration baseline identity changed")
        if (
            item.get("expected_calibration_cluster_count") != 500
            or item.get("observed_calibration_cluster_count") != 500
            or item.get("finite_trajectory_count") != 500
            or item.get("unavailable_trajectory_iae_count") != 0
            or item.get("selection_eligible") is not True
        ):
            _fail("Calibration baseline completeness changed")
        mean_value = float(item["mean_trajectory_iae_pp"])
        if not math.isfinite(mean_value):
            _fail("Calibration baseline mean contains a nonfinite value")
        calibration_means[str(item["model_id"])] = mean_value
    if set(calibration_means) != baseline_ids:
        _fail("Calibration baseline-selection identities are duplicated or missing")
    strongest_baseline = min(
        calibration_means,
        key=lambda model_id: (calibration_means[model_id], model_id),
    )
    if mean_comparison.get("strongest_calibration_baseline") != strongest_baseline:
        _fail("Strongest calibration baseline differs from the frozen selection rule")
    _assert_close(
        iae_delta,
        candidate_iae - baseline_iae,
        context="mean_forecast_comparison.candidate_minus_baseline_iae_pp",
    )
    endpoint_specs = {
        item["endpoint_id"]: item for item in config["endpoints"]["primary"]
    }
    risk_gate = bool(
        published_random_reduction
        >= float(
            endpoint_specs["catastrophic_risk_reduction_at_50_percent_issuance"][
                "threshold"
            ]
        )
        and lower_bound > 0.0
    )
    matched_gate = bool(
        qualified_count >= pair_count
        and both_fraction
        >= float(endpoint_specs["matched_prefix_both_members_rejected"]["threshold"])
    )
    iae_gate = bool(
        iae_delta
        <= float(endpoint_specs["issued_trajectory_iae_noninferiority"]["threshold_pp"])
    )
    primary_gates = {
        "catastrophic_risk_reduction_at_50_percent_issuance": risk_gate,
        "matched_prefix_both_members_rejected": matched_gate,
        "issued_trajectory_iae_noninferiority": iae_gate,
    }
    if report["primary_gates"] != primary_gates:
        _fail("Primary gate decisions differ from independent recomputation")

    rules = config["decision_rules"]
    finite_test_fraction = float(
        np.isfinite(
            merged.loc[
                merged["partition"].eq("test"), "candidate_trajectory_iae_pp"
            ].to_numpy(dtype=float)
        ).mean()
    )
    minimum_counts = bool(
        test["hard_eligible_catastrophic_count"]
        >= int(
            rules[
                "minimum_hard_eligible_test_catastrophic_cluster_count_before_rejection"
            ]
        )
        and test["hard_eligible_count"]
        >= int(rules["minimum_eligible_test_cluster_count"])
        and len(calibration) >= int(rules["minimum_eligible_calibration_cluster_count"])
        and audit["hard_eligible_count"]
        >= int(rules["minimum_eligible_audit_cluster_count"])
        and qualified_count >= int(rules["minimum_qualified_counterexample_pair_count"])
        and finite_test_fraction
        >= float(rules["minimum_finite_point_forecast_fraction"])
    )
    safety_gates = {
        "minimum_counts_and_finite_forecasts": minimum_counts,
        "audit_directional_consistency": bool(
            audit["analytic_risk_reduction_fraction"] > 0.0
        ),
        "random_rankings_fully_defined": True,
        "bootstrap_fully_defined": True,
    }
    if report["required_safety_gates"] != safety_gates:
        _fail("Required safety gates differ from independent recomputation")
    if report.get("inconclusive_reasons") != []:
        _fail("Published evaluable run unexpectedly reports inconclusive reasons")
    expected_status = (
        "success"
        if all(primary_gates.values()) and all(safety_gates.values())
        else "failure"
    )
    if report.get("status") != expected_status:
        _fail("Result status differs from the frozen decision rules")

    return {
        "test": test,
        "audit": audit,
        "random_mean_catastrophic_rate": random_mean,
        "random_ranking_risk_reduction_fraction": published_random_reduction,
        "bootstrap_one_sided_95pct_lower_bound": lower_bound,
        "matched": {
            "qualified_pair_count": qualified_count,
            "both_rejected_pair_count": both_count,
            "both_rejected_fraction": both_fraction,
            "model_failure_member_count": nonfinite_member_count,
            "pair_with_any_model_failure_count": nonfinite_pair_count,
        },
        "iae_noninferiority_delta_pp": iae_delta,
        "candidate_issued_mean_trajectory_iae_pp": recomputed_candidate_iae,
        "family_reversals": [
            {
                "partition": item["partition"],
                "truth_family": item["truth_family"],
                "risk_reduction_fraction": item[
                    "issued_vs_analytic_random_risk_reduction_fraction"
                ],
            }
            for item in reversals
        ],
        "primary_gates": primary_gates,
        "safety_gates": safety_gates,
        "result_status": expected_status,
    }


def _verify_report_contracts(
    evidence_dir: Path,
    manifest_data: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _load_json_strict(evidence_dir / "score_report.json")
    environment = _load_json_strict(evidence_dir / "environment.json")
    exposure = _load_json_strict(evidence_dir / "exposure_log.json")
    truth = _load_json_strict(evidence_dir / "truth_commitment.json")
    prediction = _load_json_strict(evidence_dir / "prediction_commitment.json")
    _require_keys(
        truth,
        {
            "protocol_id",
            "config_sha256",
            "truth_pack_byte_sha256",
            "truth_pack_row_count",
            "created_utc",
            "truth_values_withheld_until_prediction_commitment",
        },
        context="truth_commitment",
    )
    _require_keys(
        prediction,
        {
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
        },
        context="prediction_commitment",
    )
    if (
        truth["protocol_id"] != PROTOCOL_ID
        or prediction["protocol_id"] != PROTOCOL_ID
        or truth["truth_values_withheld_until_prediction_commitment"] is not True
        or prediction["truth_pack_opened_before_commitment"] is not False
    ):
        _fail("Truth or prediction commitment identity/firewall flag changed")

    source = _verify_execution_sources(
        project_root, environment, manifest_data["evidence"]
    )
    config = source["config"]
    canonical = environment["config_canonical_sha256"]
    if (
        truth["config_sha256"] != canonical
        or prediction["config_sha256"] != canonical
        or report.get("config_canonical_sha256") != canonical
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("evidence_role") != config["evidence_role"]
        or report.get("claim_boundary") != EXPECTED_REPORT_CLAIM_BOUNDARY
        or report.get("protocol_deviations") != []
    ):
        _fail("Report, commitments, and Git-frozen protocol identity disagree")

    full_entries = manifest_data["full_entries"]
    score_entries: dict[str, str] = {}
    input_artifacts = report.get("input_artifacts")
    analysis_artifacts = report.get("analysis_artifacts")
    if not isinstance(input_artifacts, dict) or not isinstance(
        analysis_artifacts, dict
    ):
        _fail("Score report artifact maps are missing")
    for path, metadata in input_artifacts.items():
        _canonical_relative_path(path, context="score input artifact path")
        if not isinstance(metadata, dict):
            _fail(f"Score input metadata must be an object: {path}")
        _require_keys(
            metadata, {"byte_count", "byte_sha256"}, context=f"score input {path}"
        )
        if path in score_entries:
            _fail(f"Duplicate score artifact path: {path}")
        score_entries[path] = _sha256(
            metadata["byte_sha256"], context=f"score input {path} hash"
        )
        if full_entries.get(path, {}).get("byte_count") != metadata["byte_count"]:
            _fail(f"Score and full manifest byte counts differ for {path}")
    for artifact_id, metadata in analysis_artifacts.items():
        if not isinstance(metadata, dict):
            _fail(f"Score analysis metadata must be an object: {artifact_id}")
        _require_keys(
            metadata,
            {"path", "row_count", "byte_sha256"},
            context=f"score analysis {artifact_id}",
        )
        path = _canonical_relative_path(
            metadata["path"], context=f"score analysis {artifact_id} path"
        )
        _integer(metadata["row_count"], context=f"score analysis {artifact_id} rows")
        if path in score_entries:
            _fail(f"Duplicate score artifact path: {path}")
        score_entries[path] = _sha256(
            metadata["byte_sha256"], context=f"score analysis {artifact_id} hash"
        )
    expected_score_paths = set(full_entries) - {
        "score_report.json",
        "exposure_log.json",
    }
    if set(score_entries) != expected_score_paths:
        _fail("Score report artifact map differs from the full bundle")
    for path, digest in score_entries.items():
        if full_entries[path]["sha256"] != digest:
            _fail(f"Score and full-bundle hashes differ for {path}")

    verified = report.get("verified_commitments")
    if not isinstance(verified, dict):
        _fail("Score report verified commitments are missing")
    commitment_hashes = {
        "prefix_pack_byte_sha256": prediction["prefix_pack_byte_sha256"],
        "forecast_coordinates_byte_sha256": prediction[
            "forecast_coordinates_byte_sha256"
        ],
        "prediction_bundle_byte_sha256": prediction["prediction_bundle_byte_sha256"],
        "decision_bundle_byte_sha256": prediction["decision_bundle_byte_sha256"],
        "member_fit_diagnostics_byte_sha256": prediction[
            "member_fit_diagnostics_byte_sha256"
        ],
        "truth_pack_byte_sha256": truth["truth_pack_byte_sha256"],
    }
    if verified != commitment_hashes:
        _fail("Score report verified commitments differ from commitment files")
    commitment_paths = {
        "prefix_pack.csv": prediction["prefix_pack_byte_sha256"],
        "forecast_coordinates.csv": prediction["forecast_coordinates_byte_sha256"],
        "prediction_bundle.csv": prediction["prediction_bundle_byte_sha256"],
        "decision_bundle.csv": prediction["decision_bundle_byte_sha256"],
        "member_fit_diagnostics.csv": prediction["member_fit_diagnostics_byte_sha256"],
        "truth_pack.csv": truth["truth_pack_byte_sha256"],
    }
    for path, digest in commitment_paths.items():
        if full_entries[path]["sha256"] != digest:
            _fail(f"Commitment and full-bundle hashes differ for {path}")

    expected_rows = {
        "prefix_pack": 34_800,
        "forecast_coordinates": 23_200,
        "prediction_bundle": 23_200,
        "decision_bundle": 2_900,
        "member_fit_diagnostics": 246_500,
    }
    if (
        prediction["row_counts"] != expected_rows
        or truth["truth_pack_row_count"] != 23_200
    ):
        _fail("Prediction or truth commitment row counts changed")
    for metadata in analysis_artifacts.values():
        path = evidence_dir / metadata["path"]
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8", newline="") as stream:
                    observed_rows = sum(1 for _ in stream) - 1
            except OSError as exc:
                raise EvidenceVerificationError(f"Cannot count rows in {path}") from exc
            if observed_rows != metadata["row_count"]:
                _fail(f"Published analysis row count changed: {metadata['path']}")

    _verify_exposure(exposure, environment, truth, prediction, report)
    _verify_void_attempt(
        evidence_dir,
        manifest_data["evidence"]["preoutcome_void_attempt"],
        environment,
        truth,
    )
    metrics = _verify_metrics(evidence_dir, report, config)
    if (
        manifest_data["evidence"]["result_status"] != report["status"]
        or report["status"] != metrics["result_status"]
    ):
        _fail("Evidence manifest and independently derived result status disagree")
    return metrics, source


def verify(
    project_root: Path = PROJECT_ROOT,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify the compact, tracked v0.14 evidence without reading artifacts/."""
    project_root = project_root.resolve()
    selected = (
        project_root / DEFAULT_EVIDENCE_RELATIVE
        if evidence_dir is None
        else evidence_dir
    )
    if not selected.is_absolute():
        selected = project_root / selected
    selected = selected.resolve()
    if not selected.is_dir():
        _fail(f"Published evidence directory is missing: {selected}")
    manifests = _verify_manifests(selected)
    metrics, source = _verify_report_contracts(selected, manifests, project_root)
    return {
        "verification_status": "passed",
        "protocol_id": PROTOCOL_ID,
        "result_status": metrics["result_status"],
        "negative_result_accepted": metrics["result_status"] == "failure",
        "execution_git_commit": source["commit"],
        "source_tree_sha256": source["source_tree_sha256"],
        "full_bundle": {
            "file_count": manifests["full"]["file_count"],
            "byte_count": manifests["full"]["byte_count"],
            "canonical_entries_sha256": manifests["full"]["canonical_entries_sha256"],
            "evidence_manifest_sha256": EXPECTED_EVIDENCE_MANIFEST_SHA256,
            "full_bundle_manifest_sha256": EXPECTED_FULL_BUNDLE_MANIFEST_SHA256,
        },
        "headline": {
            "test_random_ranking_risk_reduction_fraction": metrics[
                "random_ranking_risk_reduction_fraction"
            ],
            "test_analytic_risk_reduction_fraction": metrics["test"][
                "analytic_risk_reduction_fraction"
            ],
            "random_mean_catastrophic_rate": metrics["random_mean_catastrophic_rate"],
            "bootstrap_one_sided_95pct_lower_bound": metrics[
                "bootstrap_one_sided_95pct_lower_bound"
            ],
            "matched_both_rejected_fraction": metrics["matched"][
                "both_rejected_fraction"
            ],
            "matched_model_failure_member_count": metrics["matched"][
                "model_failure_member_count"
            ],
            "issued_iae_noninferiority_delta_pp": metrics[
                "iae_noninferiority_delta_pp"
            ],
            "audit_analytic_risk_reduction_fraction": metrics["audit"][
                "analytic_risk_reduction_fraction"
            ],
        },
        "primary_gates": metrics["primary_gates"],
        "safety_gates": metrics["safety_gates"],
        "family_reversals": metrics["family_reversals"],
        "verification_scope": {
            "candidate_issued_iae": "independently_recomputed_from_trajectory_scores",
            "baseline_issued_iae": (
                "commitment_and_hash_bound_only_not_independently_recomputed_"
                "from_compact_evidence"
            ),
            "baseline_model_metrics_sha256": manifests["full_entries"][
                "model_metrics.csv"
            ]["sha256"],
        },
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the compact tracked evidence for the frozen v0.14 synthetic "
            "long-horizon experiment. A scientifically valid failure result exits 0."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.project_root, args.evidence_dir)
    except (EvidenceVerificationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "verification_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
