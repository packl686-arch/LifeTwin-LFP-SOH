from __future__ import annotations

import hashlib
from pathlib import Path

from showcase.build_judge_console import _self_check, build_payload, render_html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = PROJECT_ROOT / "docs/judge-console/index.html"
EXPECTED_SHA256 = (
    "9552ce42da2e9e843b1ccf758f692f0df1dee6008c325c750f759a6a7179fb7a"
)


def test_judge_console_is_deterministic_self_contained_evidence_replay() -> None:
    first_payload = build_payload()
    first = render_html(first_payload)
    second = render_html(build_payload())
    _self_check(first, first_payload)
    assert first == second
    assert PUBLISHED.read_text(encoding="utf-8") == first
    assert hashlib.sha256(PUBLISHED.read_bytes()).hexdigest() == EXPECTED_SHA256
    assert len(first_payload["cases"]) == 3
    assert len(first_payload["sources"]) == 8
    assert first_payload["mode"] == "retrospective_evidence_replay_not_live_inference"
    assert "<script src=" not in first
    assert "15–25 年" in first
    assert "10.17632/kxh42bfgtj.1" in first
    assert "10.5281/zenodo.6685365" in first
    assert "CC BY 4.0" in first


def test_every_console_case_abstains_and_carries_source_hashes() -> None:
    payload = build_payload()
    assert all(case["decision"]["code"] == "abstained" for case in payload["cases"])
    assert all(
        len(source["sha256"]) == 64 and int(source["sha256"], 16) >= 0
        for source in payload["sources"]
    )
    geisbauer = next(
        case for case in payload["cases"] if case["id"] == "geisbauer-negative-transfer"
    )
    assert geisbauer["metrics"][0]["value"].startswith("+0.088 pp")
