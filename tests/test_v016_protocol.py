from __future__ import annotations

import json

import pytest

from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    DEFAULT_V021_AMENDMENT_PATH,
    TERMINATED_ATTEMPT_MANIFEST_SHA256,
    V021_EXPECTED_SEED_ROOTS,
    V021_MINIMUM_RISK_ISOTONIC_ELIGIBLE_COUNT,
    V021_PROTOCOL_ID,
    V021ProtocolError,
    _validate_v021_payload,
    load_v021_design,
)


def _mutated_design(
    mutate: object,
) -> object:
    payload = json.loads(DEFAULT_V021_AMENDMENT_PATH.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    return _validate_v021_payload(
        payload,
        config_path=DEFAULT_V021_AMENDMENT_PATH,
        raw_bytes=json.dumps(payload, sort_keys=True).encode(),
    )


def test_v021_design_loads_without_executing_generation() -> None:
    design = load_v021_design()
    assert design.protocol_id == V021_PROTOCOL_ID
    assert design.status == "implementation_frozen"
    assert design.raw["fresh_generation"]["implementation_exists"] is True
    assert design.raw["fresh_generation"]["generation_has_started"] is False
    assert design.seed_roots == V021_EXPECTED_SEED_ROOTS
    assert len(set(design.seed_roots.values())) == 13
    assert design.config_byte_sha256


def test_v021_binds_the_published_termination_manifest() -> None:
    design = load_v021_design()
    base = design.raw["base_protocol"]
    assert base["terminated_attempt_manifest_sha256"] == (
        TERMINATED_ATTEMPT_MANIFEST_SHA256
    )
    with pytest.raises(TypeError):
        base["terminated_attempt_manifest_sha256"] = "0" * 64
    opened = design.raw["exposure_disclosure"]["opened_development_truth"]
    assert isinstance(opened, tuple)
    with pytest.raises(TypeError):
        opened[0] = "changed"


def test_v021_rejects_eligibility_threshold_drift() -> None:
    with pytest.raises(V021ProtocolError, match="minimum_eligible_count"):
        _mutated_design(
            lambda payload: payload["calibration_population_split"][
                "risk_isotonic"
            ].update({"minimum_eligible_count": 899})
        )
    assert V021_MINIMUM_RISK_ISOTONIC_ELIGIBLE_COUNT == 855


def test_v021_rejects_seed_reuse() -> None:
    with pytest.raises(V021ProtocolError, match="seed roots"):
        _mutated_design(
            lambda payload: payload["fresh_generation"]["seed_roots"].update(
                {"calibration": 202607230103}
            )
        )


def test_v021_rejects_overlapping_artifact_registries() -> None:
    def mutate(payload: dict[str, object]) -> None:
        registries = payload["artifact_registries"]
        assert isinstance(registries, dict)
        terminal = registries["terminal_pre_prediction"]
        assert isinstance(terminal, dict)
        filenames = terminal["filenames"]
        assert isinstance(filenames, list)
        filenames[0] = "score_report.json"

    with pytest.raises(V021ProtocolError, match="Terminal artifact registry"):
        _mutated_design(mutate)


def test_v021_rejects_predecessor_hash_drift() -> None:
    with pytest.raises(V021ProtocolError, match="manifest_sha256"):
        _mutated_design(
            lambda payload: payload["base_protocol"].update(
                {"terminated_attempt_manifest_sha256": "0" * 64}
            )
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["calibration_population_split"]["risk_isotonic"][
            "exact_mask"
        ].__setitem__(0, "twelve observations, perhaps"),
        lambda payload: payload["terminal_reason_codes"].update(
            {"classification_precedence": "free choice"}
        ),
        lambda payload: payload["evidence_reuse_policy"]["forbidden_v2_reuse"].pop(),
    ],
)
def test_v021_rejects_unvalidated_semantic_text_drift(mutate: object) -> None:
    with pytest.raises(V021ProtocolError, match="semantic content changed"):
        _mutated_design(mutate)


def test_v021_allows_only_the_paired_implementation_status_switch() -> None:
    def mark_implementation_present(payload: dict[str, object]) -> None:
        payload["status"] = "implementation_candidate_unfrozen"
        generation = payload["fresh_generation"]
        assert isinstance(generation, dict)
        generation["implementation_exists"] = True

    design = _mutated_design(mark_implementation_present)
    assert design.status == "implementation_candidate_unfrozen"

    def mark_implementation_frozen(payload: dict[str, object]) -> None:
        payload["status"] = "implementation_frozen"
        generation = payload["fresh_generation"]
        assert isinstance(generation, dict)
        generation["implementation_exists"] = True

    frozen = _mutated_design(mark_implementation_frozen)
    assert frozen.status == "implementation_frozen"
