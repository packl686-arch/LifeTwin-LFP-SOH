from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from types import MappingProxyType
from pathlib import Path

import pytest

from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v015_generation import (
    LABEL_FREE_CSV_FILENAMES,
    PreparedGenerationArtifacts,
    TRUTH_COMMITMENT_FILENAME,
)
from lifetwin.experiments.calendar_long_horizon_v016_firewall import (
    AttemptProgress,
    FormalAttemptIdentity,
)
from lifetwin.experiments import calendar_long_horizon_v016_generation as generation


GIT_COMMIT = "1" * 40


class _Environment:
    def __init__(self, *, git_commit: str, config_hash: str) -> None:
        self.git_commit = git_commit
        self.config_byte_sha256 = config_hash


def _progress(
    *,
    completed_phase: str,
    pending_phase: str | None = None,
    terminal_failed: bool = False,
    plan_hash: str | None = None,
    truth_hash: str | None = None,
) -> AttemptProgress:
    config_hash = load_v021_contract_view().artifacts.config_byte_sha256
    return AttemptProgress(
        identity=FormalAttemptIdentity("fixture", GIT_COMMIT, config_hash),
        completed_phase=completed_phase,
        pending_phase=pending_phase,
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(),
        terminal_failed=terminal_failed,
        generation_plan_commitment_byte_sha256=plan_hash,
    )


def test_public_formal_surface_has_no_seed_or_design_override() -> None:
    assert set(
        inspect.signature(generation.commit_frozen_v021_generation_plan).parameters
    ) == {"label_free_root"}
    assert set(
        inspect.signature(generation.generate_frozen_v021_artifacts).parameters
    ) == {"label_free_root", "sealed_truth_root"}
    forbidden = {
        "seed",
        "root_value",
        "protocol",
        "config",
        "partition",
        "family",
        "count",
        "mode",
        "truth_frame",
    }
    for function in (
        generation.commit_frozen_v021_generation_plan,
        generation.generate_frozen_v021_artifacts,
    ):
        assert set(inspect.signature(function).parameters).isdisjoint(forbidden)


def test_module_import_does_not_instantiate_rng_or_run_formal_audit() -> None:
    tree = ast.parse(inspect.getsource(generation))
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert not top_level_calls
    assert "numpy" not in {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def test_plan_commit_candidate_allows_only_exact_preseed_state() -> None:
    config_hash = load_v021_contract_view().artifacts.config_byte_sha256
    environment = _Environment(git_commit=GIT_COMMIT, config_hash=config_hash)
    initial = _progress(completed_phase="before_generation")
    selected = generation._select_plan_commit_candidate(
        {"fixture": initial},
        environment=environment,
    )
    assert selected is initial

    interrupted = _progress(
        completed_phase="before_generation",
        pending_phase="generation_plan_committed",
    )
    assert (
        generation._select_plan_commit_candidate(
            {"fixture": interrupted},
            environment=environment,
        )
        is interrupted
    )

    for invalid in (
        _progress(completed_phase="truth_committed", truth_hash="2" * 64),
        _progress(completed_phase="before_generation", terminal_failed=True),
        _progress(
            completed_phase="before_generation",
            pending_phase="truth_committed",
        ),
    ):
        with pytest.raises(generation.V021GenerationError, match="Exactly one"):
            generation._select_plan_commit_candidate(
                {"fixture": invalid},
                environment=environment,
            )


def test_generation_candidate_requires_plan_hash_and_no_truth() -> None:
    config_hash = load_v021_contract_view().artifacts.config_byte_sha256
    environment = _Environment(git_commit=GIT_COMMIT, config_hash=config_hash)
    ready = _progress(
        completed_phase="generation_plan_committed",
        plan_hash="9" * 64,
    )
    assert (
        generation._select_generation_candidate(
            {"fixture": ready},
            environment=environment,
        )
        is ready
    )
    interrupted = _progress(
        completed_phase="generation_plan_committed",
        pending_phase="truth_committed",
        plan_hash="9" * 64,
    )
    assert (
        generation._select_generation_candidate(
            {"fixture": interrupted},
            environment=environment,
        )
        is interrupted
    )
    for invalid in (
        _progress(completed_phase="generation_plan_committed"),
        _progress(
            completed_phase="generation_plan_committed",
            plan_hash="9" * 64,
            truth_hash="2" * 64,
        ),
        _progress(
            completed_phase="generation_plan_committed",
            pending_phase="prediction_started",
            plan_hash="9" * 64,
        ),
    ):
        with pytest.raises(generation.V021GenerationError, match="Exactly one"):
            generation._select_generation_candidate(
                {"fixture": invalid},
                environment=environment,
            )


def test_attempt_identity_mismatch_fails_closed() -> None:
    config_hash = load_v021_contract_view().artifacts.config_byte_sha256
    environment = _Environment(git_commit="2" * 40, config_hash=config_hash)
    with pytest.raises(generation.V021GenerationError, match="different"):
        generation._select_plan_commit_candidate(
            {"fixture": _progress(completed_phase="before_generation")},
            environment=environment,
        )


def test_canonical_plan_decoder_rejects_duplicate_and_noncanonical_json() -> None:
    raw = b'{"a":1,"b":{"c":2}}\n'
    assert generation._decode_canonical_plan_commitment(raw) == {
        "a": 1,
        "b": {"c": 2},
    }
    with pytest.raises(generation.V021GenerationError, match="duplicate"):
        generation._decode_canonical_plan_commitment(b'{"a":1,"a":2}\n')
    with pytest.raises(generation.V021GenerationError, match="not canonical"):
        generation._decode_canonical_plan_commitment(b'{ "a": 1 }\n')
    with pytest.raises(generation.V021GenerationError, match="nonfinite"):
        generation._decode_canonical_plan_commitment(b'{"a":NaN}\n')


def test_contract_view_requires_immutable_v021_freeze_status() -> None:
    view = load_v021_contract_view()
    if view.design_status == "implementation_frozen":
        generation._validate_contract_view(view)
    for forbidden_status in (
        "design_candidate_preimplementation",
        "implementation_candidate_unfrozen",
    ):
        candidate = replace(view, design_status=forbidden_status)
        with pytest.raises(generation.V021GenerationError, match="frozen"):
            generation._validate_contract_view(candidate)


def test_generation_source_uses_fresh_view_not_v2_formal_entrypoint() -> None:
    source = inspect.getsource(generation.generate_frozen_v021_artifacts)
    assert "load_v021_contract_view" in source
    assert "generate_frozen_v015_artifacts" not in source
    assert "audit_formal_v021_generation_plan" not in source
    assert "_verify_committed_generation_plan" in source


def test_plan_commitment_filename_is_fixed_and_direct() -> None:
    assert (
        generation.GENERATION_PLAN_COMMITMENT_FILENAME
        == "generation_plan_commitment.json"
    )
    payload = json.loads(
        generation._decode_canonical_plan_commitment(b'{"value":1}\n') and '{"value":1}'
    )
    assert payload == {"value": 1}


def _tiny_prepared() -> PreparedGenerationArtifacts:
    view = load_v021_contract_view()
    label_names = (*LABEL_FREE_CSV_FILENAMES, TRUTH_COMMITMENT_FILENAME)
    sealed_names = view.artifacts.sealed_filenames
    label_bytes = MappingProxyType(
        {name: f"label:{name}\n".encode("ascii") for name in label_names}
    )
    sealed_bytes = MappingProxyType(
        {name: f"sealed:{name}\n".encode("ascii") for name in sealed_names}
    )
    row_counts = MappingProxyType({name: 1 for name in (*label_names, *sealed_names)})
    return PreparedGenerationArtifacts(
        label_free_bytes=label_bytes,
        sealed_bytes=sealed_bytes,
        row_counts=row_counts,
        truth_commitment_payload=MappingProxyType({}),
    )


def test_resumable_writer_accepts_only_byte_identical_partial_files(
    tmp_path: Path,
) -> None:
    label = tmp_path / "label"
    sealed = tmp_path / "sealed"
    label.mkdir()
    sealed.mkdir()
    (label / "exposure_log.jsonl").write_bytes(b"ledger\n")
    (label / generation.GENERATION_PLAN_COMMITMENT_FILENAME).write_bytes(b"plan\n")
    prepared = _tiny_prepared()
    first_sealed = load_v021_contract_view().artifacts.sealed_filenames[0]
    (sealed / first_sealed).write_bytes(prepared.sealed_bytes[first_sealed])
    label_identity, sealed_identity = generation._bind_generation_roots(
        label_free_root=label,
        sealed_truth_root=sealed,
    )

    label_meta, sealed_meta, truth_hash = (
        generation._write_prepared_generation_artifacts_resumable(
            prepared,
            label_identity=label_identity,
            sealed_identity=sealed_identity,
            view=load_v021_contract_view(),
        )
    )

    assert len(label_meta) == len(prepared.label_free_bytes)
    assert len(sealed_meta) == len(prepared.sealed_bytes)
    assert truth_hash
    for filename, expected in prepared.label_free_bytes.items():
        assert (label / filename).read_bytes() == expected
    for filename, expected in prepared.sealed_bytes.items():
        assert (sealed / filename).read_bytes() == expected

    (sealed / first_sealed).write_bytes(b"tampered\n")
    with pytest.raises(generation.V021GenerationError, match="conflicts"):
        generation._write_prepared_generation_artifacts_resumable(
            prepared,
            label_identity=label_identity,
            sealed_identity=sealed_identity,
            view=load_v021_contract_view(),
        )


def test_physical_root_identity_detects_directory_replacement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    identity = generation._physical_root_identity(root, context="fixture")
    moved = tmp_path / "moved"
    root.rename(moved)
    root.mkdir()
    with pytest.raises(generation.V021GenerationError, match="identity changed"):
        generation._verify_physical_root_identity(identity, context="fixture")
