from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lifetwin.experiments.calendar_long_horizon_v018_protocol import (
    V021_SEED_ROOTS,
    V022_SEED_ROOTS,
    V023_EXPECTED_SEED_ROOTS,
    V2_SEED_ROOTS,
)


ROOT = Path(__file__).resolve().parents[1]
V023_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_3_amendment.json"
)
V024_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2_4_amendment.json"
)
V024_REQUIREMENTS = ROOT / "requirements" / "v024-formal.txt"
V023_REQUIREMENTS = ROOT / "requirements" / "v023-formal.txt"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rng_state_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_v024_prereg_identity_and_seed_registry_without_rng_consumption() -> None:
    before = np.random.get_state()
    design = _load(V024_PATH)
    after = np.random.get_state()
    assert _rng_state_equal(before, after)
    assert design["schema_version"] == "lifetwin_synthetic_long_horizon_v2_4/1.0.0"
    assert design["protocol_id"] == "synthetic_long_horizon_identifiability_v2_4"
    assert design["status"] == "preregistered_post_root_cause_pre_formalization"
    assert design["attempt_registry"]["only_attempt_id"] == ("v024-formal-20260810-a1")
    expected = list(range(202608100401, 202608100414))
    observed = list(design["fresh_generation"]["seed_roots"].values())
    assert observed == expected
    predecessors = {
        *V2_SEED_ROOTS,
        *V021_SEED_ROOTS,
        *V022_SEED_ROOTS,
        *V023_EXPECTED_SEED_ROOTS.values(),
    }
    assert not predecessors.intersection(observed)
    assert design["fresh_generation"]["generation_has_started"] is False
    assert design["fresh_generation"]["seed_consumed"] is False
    assert design["fresh_generation"]["sealed_truth_created_or_opened"] is False


def test_v024_scientific_fields_are_inherited_from_v023() -> None:
    v023 = _load(V023_PATH)
    v024 = _load(V024_PATH)
    assert (
        v024["scientific_inheritance"]["unchanged"]
        == v023["scientific_inheritance"]["unchanged"]
    )
    assert v024["whole_bundle_contract"] == v023["whole_bundle_contract"]
    assert v024["partition_contract"] == v023["partition_contract"]
    assert v024["lifecycle_order"] == v023["lifecycle_order"]
    for key in (
        "scored_success",
        "scored_failure",
        "terminal_inconclusive_not_success",
        "integrity_void",
        "unclassified_terminal_not_success",
        "known_partition_error_may_use_unknown_default",
        "prediction_and_terminal_registries_mutually_exclusive",
        "terminal_registry_exact_files",
    ):
        assert v024["terminal_rules"][key] == v023["terminal_rules"][key]


def test_v024_only_adds_exact_member_fit_and_atomic_commitment_semantics() -> None:
    design = _load(V024_PATH)
    numeric = design["numeric_output_contract"]
    assert (
        numeric["risk_bundle"]
        == _load(V023_PATH)["numeric_output_contract"]["risk_bundle"]
    )
    assert set(numeric["member_fit"]) == {
        "status_registry",
        "succeeded_mask",
        "failed_mask",
        "forbidden",
    }
    assert set(numeric["fit_commitment_atomicity"]) == {
        "write_order",
        "ledger_order",
        "failure_rule",
        "terminal_manifest_rule",
    }
    assert (
        "INTEGRITY_MEMBER_FIT_NUMERIC_CONTRACT_MISMATCH"
        in design["terminal_rules"]["registered_v024_integrity_codes"]
    )


def test_v024_environment_package_pins_are_identical_to_v023() -> None:
    def pins(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]

    assert pins(V024_REQUIREMENTS) == pins(V023_REQUIREMENTS)


def test_v024_prereg_does_not_create_formal_roots() -> None:
    design = _load(V024_PATH)
    isolation = design["path_isolation"]
    for key in (
        "label_free_root",
        "sealed_truth_root",
        "score_root",
        "termination_root",
    ):
        assert not (ROOT / isolation[key]).exists()
