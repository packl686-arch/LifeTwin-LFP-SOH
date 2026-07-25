from __future__ import annotations

import hashlib

import pytest

from lifetwin.experiments.calendar_long_horizon_v015_collision import (
    CollisionLedger,
    V015CollisionError,
    derive_bootstrap_analysis_seed,
    validate_predictor_hash_ledger,
)


def test_seed_and_identifier_namespaces_are_globally_unique() -> None:
    ledger = CollisionLedger()
    ledger.register_seed(label="ordinary/a/operating", seed=11)
    ledger.register_seed(label="matched/b/truth", seed=12)
    ledger.register_identifier(label="ordinary/a/id", identifier="c_alpha")
    ledger.register_identifier(label="matched/b/id", identifier="c_beta")
    assert ledger.seed_count == 2
    assert ledger.identifier_count == 2

    with pytest.raises(V015CollisionError, match="Seed collision"):
        ledger.register_seed(label="matched/c/noise", seed=11)
    with pytest.raises(V015CollisionError, match="Identifier collision"):
        ledger.register_identifier(label="matched/c/id", identifier="c_alpha")


def test_content_hash_must_match_canonical_bytes() -> None:
    ledger = CollisionLedger()
    with pytest.raises(V015CollisionError, match="does not match"):
        ledger.register_content_hash(
            namespace="arm_a",
            digest="0" * 64,
            canonical_content=b"predictor-content",
            unique_content_required=False,
        )


def test_identical_content_can_repeat_only_outside_unique_pools() -> None:
    content = b"same predictor"
    digest = hashlib.sha256(content).hexdigest()
    ledger = CollisionLedger()
    ledger.register_content_hash(
        namespace="development",
        digest=digest,
        canonical_content=content,
        unique_content_required=False,
    )
    ledger.register_content_hash(
        namespace="development",
        digest=digest,
        canonical_content=content,
        unique_content_required=False,
    )
    assert ledger.content_hash_count == 1

    ledger.register_content_hash(
        namespace="test",
        digest=digest,
        canonical_content=content,
        unique_content_required=True,
    )
    with pytest.raises(V015CollisionError, match="Duplicate predictor content"):
        ledger.register_content_hash(
            namespace="test",
            digest=digest,
            canonical_content=content,
            unique_content_required=True,
        )


def test_distinct_content_same_digest_is_a_protocol_failure() -> None:
    content = b"first"
    digest = hashlib.sha256(content).hexdigest()
    ledger = CollisionLedger()
    ledger.register_content_hash(
        namespace="arm_b",
        digest=digest,
        canonical_content=content,
        unique_content_required=False,
    )
    with pytest.raises(V015CollisionError, match="does not match"):
        ledger.register_content_hash(
            namespace="arm_b",
            digest=digest,
            canonical_content=b"second",
            unique_content_required=False,
        )


def test_named_predictor_hash_ledger_keeps_namespaces_separate() -> None:
    content = b"shared allowed content"
    digest = hashlib.sha256(content).hexdigest()
    observed = validate_predictor_hash_ledger(
        {
            "arm_a/member-1": (digest, content, False),
            "arm_b/member-1": (digest, content, False),
        }
    )
    assert observed == 2


def test_bootstrap_seed_helper_matches_frozen_specific_formula() -> None:
    protocol_id = "synthetic_long_horizon_identifiability_v2"
    root = 202607230111
    replicate = 37
    family = "late_knee"
    material = f"{protocol_id}|{root}|bootstrap|{replicate}|{family}".encode("ascii")
    expected = int(hashlib.sha256(material).hexdigest()[:16], 16) % (2**63 - 1)
    assert (
        derive_bootstrap_analysis_seed(
            protocol_id,
            replicate_index=replicate,
            family_id=family,
            seed_root=root,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("seed", "identifier"),
    [
        (True, "c_ok"),
        (-1, "c_ok"),
        (1, ""),
    ],
)
def test_invalid_ledger_values_are_rejected(seed: int, identifier: str) -> None:
    ledger = CollisionLedger()
    if seed is True or seed < 0:
        with pytest.raises(V015CollisionError):
            ledger.register_seed(label="bad", seed=seed)
    if not identifier:
        with pytest.raises(V015CollisionError):
            ledger.register_identifier(label="bad", identifier=identifier)
