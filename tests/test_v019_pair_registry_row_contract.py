from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifetwin.experiments import calendar_long_horizon_v019_prediction as prediction
from lifetwin.experiments import (
    calendar_long_horizon_v019_prediction_capsule as capsule,
)


_ORDINARY_TRUTH_ROWS = {
    "center_development_truth.csv": 4_800,
    "risk_development_truth.csv": 4_800,
    "calibration_truth.csv": 7_200,
    "test_truth.csv": 15_200,
    "audit_truth.csv": 7_600,
    "intrinsic_matched_truth.csv": 4_000,
    "stress_plan_matched_truth.csv": 4_000,
}
_PAIR_FILENAMES = (
    "intrinsic_matched_pairs.csv",
    "stress_plan_matched_pairs.csv",
)


def _truth_commitment_raw(
    *,
    row_overrides: dict[str, int] | None = None,
    remove: str | None = None,
    rename: tuple[str, str] | None = None,
    extra: str | None = None,
) -> bytes:
    pair_rows = {
        filename: capsule.PARTITION_MEMBER_COUNTS[filename.removesuffix(".csv")] // 2
        for filename in _PAIR_FILENAMES
    }
    rows = {**_ORDINARY_TRUTH_ROWS, **pair_rows, **(row_overrides or {})}
    entries = [
        {
            "path": filename,
            "row_count": rows[filename],
            "byte_count": 1,
            "byte_sha256": hashlib.sha256(filename.encode("ascii")).hexdigest(),
        }
        for filename in capsule._SEALED_FILENAMES
        if filename != remove
    ]
    if rename is not None:
        old, new = rename
        next(entry for entry in entries if entry["path"] == old)["path"] = new
    if extra is not None:
        entries.append(
            {
                "path": extra,
                "row_count": 1,
                "byte_count": 1,
                "byte_sha256": hashlib.sha256(extra.encode("ascii")).hexdigest(),
            }
        )
    return capsule.canonical_json_bytes(
        {
            "protocol_id": "fixture_protocol",
            "config_sha256": "a" * 64,
            "files": entries,
            "created_utc": "2026-08-14T00:00:00Z",
            "truth_values_withheld_by_physical_path": True,
        }
    )


def _verify(raw: bytes) -> dict[str, str]:
    return capsule._verify_truth_commitment(
        raw,
        progress=SimpleNamespace(
            truth_commitments_byte_sha256=hashlib.sha256(raw).hexdigest()
        ),
        config_sha256="a" * 64,
        protocol_id="fixture_protocol",
    )


def test_pair_registries_use_pair_rows_while_other_truth_counts_stay_frozen() -> None:
    assert {
        filename: capsule._TRUTH_REQUIRED_ROWS[filename]
        for filename in _ORDINARY_TRUTH_ROWS
    } == _ORDINARY_TRUTH_ROWS
    for filename in _PAIR_FILENAMES:
        partition = filename.removesuffix(".csv")
        assert capsule._TRUTH_REQUIRED_ROWS[filename] == 250
        assert (
            capsule._TRUTH_REQUIRED_ROWS[filename] * 2
            == capsule.PARTITION_MEMBER_COUNTS[partition]
            == 500
        )
    assert set(_verify(_truth_commitment_raw())) == set(capsule._SEALED_FILENAMES)


@pytest.mark.parametrize("filename", _PAIR_FILENAMES)
@pytest.mark.parametrize("wrong_rows", (500, 249, 251))
def test_pair_registry_wrong_row_counts_fail_closed(
    filename: str, wrong_rows: int
) -> None:
    with pytest.raises(capsule.V024PredictionCapsuleError):
        _verify(_truth_commitment_raw(row_overrides={filename: wrong_rows}))


@pytest.mark.parametrize(
    "raw",
    (
        _truth_commitment_raw(remove="intrinsic_matched_pairs.csv"),
        _truth_commitment_raw(extra="unknown_pairs.csv"),
        _truth_commitment_raw(
            rename=("stress_plan_matched_pairs.csv", "renamed_pairs.csv")
        ),
    ),
)
def test_pair_registry_missing_extra_or_renamed_entries_fail_closed(raw: bytes) -> None:
    with pytest.raises(capsule.V024PredictionCapsuleError):
        _verify(raw)


def test_current_prediction_entry_imports_the_fixed_v019_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def reject_at_loader(**kwargs):
        calls.append(kwargs)
        raise capsule.V024PredictionCapsuleError("loader reached")

    monkeypatch.setattr(capsule, "load_prediction_bundle", reject_at_loader)
    environment = SimpleNamespace(
        protocol_id="fixture_protocol",
        config_byte_sha256="a" * 64,
        git_commit="b" * 40,
    )
    root = Path("fixture-label-free-root")
    with pytest.raises(prediction.V024PredictionError, match="rejected"):
        prediction.run_isolated_prediction_process_v024(
            label_free_root=root,
            attempt_id="fixture-attempt",
            repo_root=root,
            _environment_verifier=lambda _root: environment,
        )
    assert calls == [
        {
            "label_free_root": root,
            "attempt_id": "fixture-attempt",
            "expected_protocol_id": "fixture_protocol",
            "expected_config_sha256": "a" * 64,
            "expected_git_commit": "b" * 40,
            "_input_filenames_by_stage": None,
        }
    ]
