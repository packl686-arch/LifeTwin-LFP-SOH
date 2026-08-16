from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_historical_freezes import verify_freeze_record


ROOT = Path(__file__).resolve().parents[1]
FREEZE_RECORDS = tuple(
    ROOT
    / f"reports/synthetic_long_horizon_identifiability_freeze_record_v2_{version}.json"
    for version in range(5, 11)
)


@pytest.mark.parametrize("record_path", FREEZE_RECORDS)
def test_historical_freeze_is_attested_from_git_objects(record_path: Path) -> None:
    result = verify_freeze_record(ROOT, record_path)

    assert result["status"] == "passed"
    assert result["source_file_count"] > 0
