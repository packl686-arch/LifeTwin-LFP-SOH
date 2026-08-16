from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
from pathlib import Path

import numpy as np

from lifetwin.experiments import calendar_long_horizon_v017_collision as collision
from lifetwin.experiments import calendar_long_horizon_v017_contract as contract
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    DEFAULT_V2_CONFIG_PATH,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    load_frozen_protocol_config,
)
from lifetwin.experiments.calendar_long_horizon_v017_protocol import (
    V021_SEED_ROOTS,
    V022_AMENDMENT_BYTE_SHA256,
    V022_EXPECTED_SEED_ROOTS,
    V022_ONLY_ATTEMPT_ID,
    V022_PROTOCOL_ID,
    V2_SEED_ROOTS,
    load_v022_design,
)


def _rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_frozen_amendment_identity_and_seed_isolation_without_rng_use() -> None:
    before = np.random.get_state()
    design = load_v022_design()
    view = contract.load_v022_contract_view()
    after = np.random.get_state()
    assert _rng_equal(before, after)
    assert design.status == "implementation_frozen"
    assert design.protocol_id == V022_PROTOCOL_ID
    assert V022_ONLY_ATTEMPT_ID == "v022-formal-20260809-a1"
    assert hashlib.sha256(design.config_path.read_bytes()).hexdigest() == (
        V022_AMENDMENT_BYTE_SHA256
    )
    assert tuple(design.seed_roots.items()) == tuple(V022_EXPECTED_SEED_ROOTS.items())
    assert len(set(design.seed_roots.values())) == 13
    assert not set(design.seed_roots.values()).intersection(V2_SEED_ROOTS)
    assert not set(design.seed_roots.values()).intersection(V021_SEED_ROOTS)
    assert view.protocol.seed_roots == tuple(V022_EXPECTED_SEED_ROOTS.items())


def test_contract_adapter_changes_only_identity_hash_seeds_and_config() -> None:
    base_protocol = load_frozen_protocol_config(DEFAULT_V2_CONFIG_PATH)
    base_artifacts = load_artifact_contract(DEFAULT_V2_CONFIG_PATH)
    view = contract.load_v022_contract_view()
    protocol_differences = {
        field.name
        for field in fields(base_protocol)
        if getattr(base_protocol, field.name) != getattr(view.protocol, field.name)
    }
    artifact_differences = {
        field.name
        for field in fields(base_artifacts)
        if getattr(base_artifacts, field.name) != getattr(view.artifacts, field.name)
    }
    assert protocol_differences == {
        "protocol_id",
        "config_sha256",
        "seed_roots",
        "config_json",
    }
    assert artifact_differences == {
        "protocol_id",
        "config_path",
        "config_byte_sha256",
    }


def test_rng_free_formal_collision_plan_is_exact_and_fresh() -> None:
    before = np.random.get_state()
    commitment = collision.audit_formal_v022_generation_plan()
    after = np.random.get_state()
    assert _rng_equal(before, after)
    assert commitment.byte_sha256 == (
        "82881e764abc4fbef352328dcaee311fb85c1a2f8000490af18940324ad514b5"
    )
    assert commitment.payload["current"]["counts"] == {
        "analysis_tie_formula_witnesses": 9,
        "bootstrap_coordinates": 40_000,
        "generated_members": 5_950,
        "identity_and_formula_witness_digest_count": 96_959,
        "identifier_count": 6_450,
        "matched_pairs": 500,
        "namespace_ledger_records": 171_159,
        "ordinary_clusters": 4_950,
        "random_ranking_coordinates": 10_000,
        "seed_count": 67_750,
        "stress_permutation_coordinates": 80_000,
    }


def test_v017_formal_sources_expose_no_formal_false_literal() -> None:
    source_root = Path(inspect.getsourcefile(contract) or "").parent
    sources = sorted(source_root.glob("calendar_long_horizon_v017_*.py"))
    sources.append(source_root.parents[2] / "scripts" / "run_calendar_long_horizon_v017.py")
    assert sources
    for path in sources:
        assert "formal=False" not in path.read_text(encoding="utf-8")
