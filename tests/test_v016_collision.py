from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path

import pytest

from lifetwin.experiments import calendar_long_horizon_v016_collision as collision
from lifetwin.experiments.calendar_long_horizon_v016_collision import (
    ActualAnalysisContentRecord,
    CoordinateCollisionLedger,
    GenerationPlanSpec,
    MatchedPlanGroup,
    OrdinaryPlanGroup,
    V021CollisionError,
    audit_generation_coordinate_plans,
    bind_actual_analysis_hash_ledger,
    derive_bootstrap_seed,
    derive_analysis_tie_digest,
    derive_random_ranking_digest,
    derive_stream_seed,
    derive_stress_permutation_digest,
    verify_actual_analysis_hash_ledger_commitment,
    verify_generation_plan_commitment,
)


def _small_plan(
    *,
    protocol_id: str,
    hash_digit: str,
    root_offset: int,
) -> GenerationPlanSpec:
    roots = (
        ("test", root_offset + 1),
        ("audit", root_offset + 2),
        ("fixture_matched", root_offset + 3),
        ("placebo_covariate", root_offset + 4),
        ("bootstrap", root_offset + 5),
        ("random_rankings", root_offset + 6),
        ("stress_permutations", root_offset + 7),
    )
    return GenerationPlanSpec(
        protocol_id=protocol_id,
        protocol_byte_sha256=hash_digit * 64,
        protocol_semantic_sha256=hash_digit.upper().lower() * 63 + "0",
        seed_roots=roots,
        ordinary_groups=(
            OrdinaryPlanGroup("test", "family_a", 2, "test"),
            OrdinaryPlanGroup("audit", "family_a", 1, "audit"),
        ),
        matched_groups=(MatchedPlanGroup("fixture_matched", 1, "fixture_matched"),),
        bootstrap_partition="test",
        bootstrap_families=("family_a",),
        bootstrap_resamples=2,
        ranking_partitions=("test", "audit"),
        random_ranking_count=2,
        stress_partition="test",
        stress_families=("family_a",),
        stress_permutation_count=2,
    )


def _small_pair() -> tuple[GenerationPlanSpec, GenerationPlanSpec]:
    return (
        _small_plan(
            protocol_id="fixture_protocol_v2_1",
            hash_digit="a",
            root_offset=10_000,
        ),
        _small_plan(
            protocol_id="fixture_protocol_v2",
            hash_digit="b",
            root_offset=20_000,
        ),
    )


def test_small_cross_generation_audit_is_exact_and_deterministic() -> None:
    current, predecessor = _small_pair()
    first = audit_generation_coordinate_plans(
        current=current,
        predecessor=predecessor,
    )
    second = audit_generation_coordinate_plans(
        current=current,
        predecessor=predecessor,
    )

    assert first == second
    assert first.canonical_bytes == second.canonical_bytes
    assert first.byte_sha256 == second.byte_sha256
    assert hash(first) == hash(second)
    assert first.current.counts.ordinary_clusters == 3
    assert first.current.counts.matched_pairs == 1
    assert first.current.counts.generated_members == 5
    assert first.current.counts.bootstrap_coordinates == 2
    assert first.current.counts.random_ranking_coordinates == 2
    assert first.current.counts.stress_permutation_coordinates == 2
    assert first.current.counts.analysis_tie_coordinates == 9
    assert first.current.counts.seed_count == 23
    assert first.current.counts.identifier_count == 6
    assert first.current.counts.digest_count == 20
    assert first.complete_ordered_ledger_records == 98
    cross = first.payload["cross_protocol_coordinate_namespace"]
    assert (
        cross["content_dependent_hash_comparison"] == "deferred_until_post_generation"
    )
    assert cross["formula_witness_content_sha256"] == "0" * 64
    assert "exact_value_comparison_performed" not in cross
    assert not any(key.endswith("_overlap_count") for key in cross)


def test_seed_helpers_match_inherited_formulas_without_rng() -> None:
    ordinary_material = (
        "fixture_protocol_v2_1|10001|test|family_a|1|measurement_noise"
    ).encode("ascii")
    ordinary_expected = int(hashlib.sha256(ordinary_material).hexdigest()[:16], 16) % (
        2**63 - 1
    )
    assert (
        derive_stream_seed(
            "fixture_protocol_v2_1",
            seed_root=10001,
            partition="test",
            family_id="family_a",
            zero_based_index=1,
            stream_name="measurement_noise",
        )
        == ordinary_expected
    )

    bootstrap_material = ("fixture_protocol_v2_1|10005|bootstrap|1|family_a").encode(
        "ascii"
    )
    bootstrap_expected = int(
        hashlib.sha256(bootstrap_material).hexdigest()[:16], 16
    ) % (2**63 - 1)
    assert (
        derive_bootstrap_seed(
            "fixture_protocol_v2_1",
            seed_root=10005,
            replicate_index=1,
            family_id="family_a",
        )
        == bootstrap_expected
    )

    content_hash = "c" * 64
    assert (
        derive_random_ranking_digest(
            seed_root=10006,
            ranking_index=7,
            content_hash=content_hash,
        )
        == hashlib.sha256(f"10006|7|{content_hash}".encode("ascii")).hexdigest()
    )
    assert (
        derive_analysis_tie_digest(
            "fixture_protocol_v2_1",
            arm="visible_stress",
            content_hash=content_hash,
        )
        == hashlib.sha256(
            f"fixture_protocol_v2_1|visible_stress|{content_hash}".encode("ascii")
        ).hexdigest()
    )
    assert (
        derive_stress_permutation_digest(
            seed_root=10007,
            permutation_index=3,
            family_id="family_a",
            random_policy_content_sha256=content_hash,
        )
        == hashlib.sha256(
            f"10007|3|family_a|{content_hash}".encode("ascii")
        ).hexdigest()
    )


def _actual_hash_records() -> tuple[ActualAnalysisContentRecord, ...]:
    arms = ("prefix_only", "visible_stress")
    return (
        ActualAnalysisContentRecord(
            partition="test",
            family_id="family_a",
            member_id="member_0",
            random_policy_content_sha256="1" * 64,
            predictor_content_hashes=tuple(
                (arm, f"{index + 2:064x}") for index, arm in enumerate(arms)
            ),
        ),
        ActualAnalysisContentRecord(
            partition="audit",
            family_id="family_a",
            member_id="member_1",
            # The inherited formula intentionally permits the same content key
            # in two independently ranked partition pools.
            random_policy_content_sha256="1" * 64,
            predictor_content_hashes=tuple(
                (arm, f"{index + 12:064x}") for index, arm in enumerate(arms)
            ),
        ),
    )


def _bind_small_actual_hash_ledger(
    records: tuple[ActualAnalysisContentRecord, ...],
) -> collision.ActualAnalysisHashLedgerCommitment:
    return bind_actual_analysis_hash_ledger(
        protocol_id="fixture_protocol_v2_1",
        random_ranking_root=10006,
        stress_permutation_root=10007,
        records=records,
        ranking_partitions=("test", "audit"),
        random_ranking_count=3,
        stress_partition="test",
        stress_families=("family_a",),
        stress_permutation_count=2,
        tie_arms=("prefix_only", "visible_stress"),
        expected_group_counts=(
            ("test", "family_a", 1),
            ("audit", "family_a", 1),
        ),
    )


def test_actual_hash_ledger_binds_predictor_tie_ranking_and_permutation_values() -> (
    None
):
    records = _actual_hash_records()
    first = _bind_small_actual_hash_ledger(records)
    reordered = _bind_small_actual_hash_ledger(tuple(reversed(records)))

    assert first == reordered
    counts = first.payload["ledger_counts"]
    assert counts == {
        "predictor_content_hash": 4,
        "analysis_tie_hash": 4,
        "random_ranking_hash": 6,
        "stress_permutation_hash": 2,
        "complete_ordered_ledger_records": 16,
        "source_content_records": 2,
    }
    assert (
        verify_actual_analysis_hash_ledger_commitment(
            first.payload,
            expected_byte_sha256=first.byte_sha256,
            expected_protocol_id="fixture_protocol_v2_1",
            expected_random_ranking_root=10006,
            expected_stress_permutation_root=10007,
        )
        == first.byte_sha256
    )

    changed = replace(
        records[0],
        predictor_content_hashes=(
            ("prefix_only", "e" * 64),
            records[0].predictor_content_hashes[1],
        ),
    )
    changed_commitment = _bind_small_actual_hash_ledger((changed, records[1]))
    assert changed_commitment.byte_sha256 != first.byte_sha256
    assert (
        changed_commitment.payload["domains"][0]["ordered_ledger_sha256"]
        != first.payload["domains"][0]["ordered_ledger_sha256"]
    )
    assert (
        changed_commitment.payload["domains"][1]["ordered_ledger_sha256"]
        != first.payload["domains"][1]["ordered_ledger_sha256"]
    )


def test_actual_hash_ledger_rejects_missing_groups_and_in_partition_duplicates() -> (
    None
):
    records = _actual_hash_records()
    with pytest.raises(V021CollisionError, match="declared generation plan"):
        _bind_small_actual_hash_ledger((records[0],))

    duplicate = replace(
        records[1],
        partition="test",
        member_id="member_2",
        predictor_content_hashes=(
            ("prefix_only", "d" * 64),
            ("visible_stress", "e" * 64),
        ),
    )
    with pytest.raises(V021CollisionError, match="Random-policy content is duplicated"):
        bind_actual_analysis_hash_ledger(
            protocol_id="fixture_protocol_v2_1",
            random_ranking_root=10006,
            stress_permutation_root=10007,
            records=(records[0], duplicate),
            ranking_partitions=("test",),
            random_ranking_count=1,
            stress_partition="test",
            stress_families=("family_a",),
            stress_permutation_count=1,
            tie_arms=("prefix_only", "visible_stress"),
        )


def test_ledger_fails_closed_on_duplicate_coordinates_and_values() -> None:
    ledger = CoordinateCollisionLedger()
    ledger.register_seed(label="ordinary/a/seed", seed=11)
    with pytest.raises(V021CollisionError, match="Duplicate coordinate"):
        ledger.register_seed(label="ordinary/a/seed", seed=12)
    with pytest.raises(V021CollisionError, match="Seed collision"):
        ledger.register_seed(label="ordinary/b/seed", seed=11)

    first_id = "c_" + "1" * 32
    ledger.register_identifier(label="ordinary/a/id", identifier=first_id)
    with pytest.raises(V021CollisionError, match="Identifier collision"):
        ledger.register_identifier(label="ordinary/b/id", identifier=first_id)

    digest = "2" * 64
    ledger.register_digest(label="ordinary/a/digest", digest=digest)
    with pytest.raises(V021CollisionError, match="Digest collision"):
        ledger.register_digest(label="ordinary/b/digest", digest=digest)


def test_cross_generation_formula_witness_conflict_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, predecessor = _small_pair()
    current = replace(current, random_ranking_count=1)
    predecessor = replace(predecessor, random_ranking_count=1)
    monkeypatch.setattr(
        collision,
        "derive_random_ranking_digest",
        lambda **_: "f" * 64,
    )

    with pytest.raises(V021CollisionError, match="Digest collision"):
        audit_generation_coordinate_plans(
            current=current,
            predecessor=predecessor,
        )


def test_bad_roots_coordinates_and_protocol_identity_fail_closed() -> None:
    current, predecessor = _small_pair()
    duplicate_roots = list(current.seed_roots)
    duplicate_roots[-1] = (duplicate_roots[-1][0], duplicate_roots[0][1])
    with pytest.raises(V021CollisionError, match="Seed-root values collide"):
        replace(current, seed_roots=tuple(duplicate_roots))

    with pytest.raises(V021CollisionError, match="duplicate groups"):
        replace(
            current,
            ordinary_groups=(
                current.ordinary_groups[0],
                current.ordinary_groups[0],
            ),
        )

    overlapping = tuple(
        (
            name,
            predecessor.seed_roots[0][1] if name == current.seed_roots[0][0] else value,
        )
        for name, value in current.seed_roots
    )
    with pytest.raises(V021CollisionError, match="seed roots overlap"):
        audit_generation_coordinate_plans(
            current=replace(current, seed_roots=overlapping),
            predecessor=predecessor,
        )

    with pytest.raises(V021CollisionError, match="protocol IDs must differ"):
        audit_generation_coordinate_plans(
            current=replace(current, protocol_id=predecessor.protocol_id),
            predecessor=predecessor,
        )


def test_commitment_binds_identities_full_ledgers_and_rejects_tampering() -> None:
    current, predecessor = _small_pair()
    commitment = audit_generation_coordinate_plans(
        current=current,
        predecessor=predecessor,
    )
    assert (
        verify_generation_plan_commitment(
            commitment.payload,
            expected_byte_sha256=commitment.byte_sha256,
            expected_current_protocol_id=current.protocol_id,
            expected_current_protocol_byte_sha256=current.protocol_byte_sha256,
            expected_predecessor_protocol_id=predecessor.protocol_id,
            expected_predecessor_protocol_byte_sha256=(
                predecessor.protocol_byte_sha256
            ),
        )
        == commitment.byte_sha256
    )

    tampered = commitment.payload
    tampered["current"]["counts"]["seed_count"] += 1
    with pytest.raises(V021CollisionError, match="commitment bytes changed"):
        verify_generation_plan_commitment(
            tampered,
            expected_byte_sha256=commitment.byte_sha256,
        )

    identity_tampered = commitment.payload
    identity_tampered["current"]["protocol_id"] = "fixture_protocol_wrong"
    with pytest.raises(V021CollisionError, match="protocol identity changed"):
        verify_generation_plan_commitment(
            identity_tampered,
            expected_byte_sha256=commitment.byte_sha256,
            expected_current_protocol_id=current.protocol_id,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: derive_stream_seed(
            "bad|protocol",
            seed_root=1,
            partition="test",
            family_id="family_a",
            zero_based_index=0,
            stream_name="opaque_id",
        ),
        lambda: derive_stream_seed(
            "fixture_protocol",
            seed_root=True,
            partition="test",
            family_id="family_a",
            zero_based_index=0,
            stream_name="opaque_id",
        ),
        lambda: derive_bootstrap_seed(
            "fixture_protocol",
            seed_root=1,
            replicate_index=-1,
            family_id="family_a",
        ),
    ],
)
def test_public_derivation_inputs_are_strict(call: object) -> None:
    assert callable(call)
    with pytest.raises(V021CollisionError):
        call()


def test_module_has_no_rng_data_generation_or_outcome_capability() -> None:
    source_path = Path(inspect.getsourcefile(collision) or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden_import_fragments = {
        "analysis",
        "generation",
        "numpy",
        "pandas",
        "pipeline",
        "random",
        "scoring",
    }
    assert not any(
        fragment in module
        for module in imported_modules
        for fragment in forbidden_import_fragments
    )

    forbidden_calls = {
        "Generator",
        "PCG64",
        "PCG64DXSM",
        "default_rng",
        "generate_cluster_packs",
        "sample_truth_spec",
    }
    observed_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    observed_calls.update(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    assert observed_calls.isdisjoint(forbidden_calls)

    formal_source = inspect.getsource(collision.audit_formal_v021_generation_plan)
    assert "commit_generation_coordinate_namespaces" in formal_source
    assert "build_formal_plan_specs" in formal_source
