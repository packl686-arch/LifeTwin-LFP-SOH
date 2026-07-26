from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lifetwin.experiments import calendar_long_horizon_v016_analysis as analysis
from lifetwin.experiments import calendar_long_horizon_v016_collision as collision
from lifetwin.experiments import calendar_long_horizon_v016_io as v021_io
from lifetwin.experiments import calendar_long_horizon_v016_prediction as prediction
from lifetwin.experiments import calendar_long_horizon_v016_provenance as provenance
from lifetwin.experiments import calendar_long_horizon_v016_scoring as scoring


_EXPERIMENT_PREFIX = "lifetwin.experiments.calendar_long_horizon_"
_PREDICTION_MODULE = "lifetwin.experiments.calendar_long_horizon_v016_prediction"


def _actual_records() -> tuple[collision.ActualAnalysisContentRecord, ...]:
    arms = ("prefix_only", "visible_stress")
    return tuple(
        collision.ActualAnalysisContentRecord(
            partition="test",
            family_id="family_a",
            member_id=f"member_{index}",
            random_policy_content_sha256=f"{index + 1:064x}",
            predictor_content_hashes=tuple(
                (arm, f"{10 * (index + 1) + arm_index:064x}")
                for arm_index, arm in enumerate(arms)
            ),
        )
        for index in range(2)
    )


def _bind_actual_fixture() -> collision.ActualAnalysisHashLedgerCommitment:
    return collision.bind_actual_analysis_hash_ledger(
        protocol_id="fixture_protocol_v2_1",
        random_ranking_root=10_006,
        stress_permutation_root=10_007,
        records=_actual_records(),
        ranking_partitions=("test",),
        random_ranking_count=2,
        stress_partition="test",
        stress_families=("family_a",),
        stress_permutation_count=2,
        tie_arms=("prefix_only", "visible_stress"),
    )


def _evidence_entries() -> list[dict[str, object]]:
    return [
        {
            "path": filename,
            "row_count": index + 1,
            "byte_count": index + 101,
            "byte_sha256": hashlib.sha256(filename.encode("ascii")).hexdigest(),
        }
        for index, filename in enumerate(v021_io._COMMITMENT_FILE_REGISTRY)
    ]


def _sealed_prediction_evidence() -> v021_io.V021PredictionCommitmentEvidence:
    entries = _evidence_entries()
    actual_hash = next(
        str(entry["byte_sha256"])
        for entry in entries
        if entry["path"] == "actual_analysis_hash_ledger_commitment.json"
    )
    return v021_io._issue_prediction_commitment_evidence(
        attempt_id="v021-security-fixture",
        byte_sha256="1" * 64,
        artifact_set_sha256="2" * 64,
        actual_analysis_hash_ledger_commitment_byte_sha256=actual_hash,
        file_entries=entries,
        ledger_committed=True,
    )


def _module_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _top_level_local_imports(module_name: str) -> set[str]:
    path = _module_path(module_name)
    if path is None:
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            result.update(
                alias.name
                for alias in node.names
                if alias.name.startswith(_EXPERIMENT_PREFIX)
            )
        elif isinstance(node, ast.ImportFrom):
            imported = node.module or ""
            if imported == "lifetwin.experiments":
                result.update(
                    f"{imported}.{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("calendar_long_horizon_")
                )
            elif imported.startswith(_EXPERIMENT_PREFIX):
                result.add(imported)
    return result


def _top_level_import_closure(root: str) -> set[str]:
    pending = [root]
    visited: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in visited:
            continue
        visited.add(module_name)
        pending.extend(_top_level_local_imports(module_name) - visited)
    return visited


@pytest.mark.parametrize(
    ("derivation_name", "message"),
    [
        ("derive_analysis_tie_digest", "tie-hash collision"),
        ("derive_random_ranking_digest", "Random-ranking hash collision"),
        ("derive_stress_permutation_digest", "Stress-permutation hash collision"),
    ],
)
def test_actual_hash_ledger_fails_closed_on_derived_digest_collisions(
    monkeypatch: pytest.MonkeyPatch,
    derivation_name: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        collision,
        derivation_name,
        lambda *_, **__: "f" * 64,
    )
    with pytest.raises(collision.V021CollisionError, match=message):
        _bind_actual_fixture()


def test_actual_hash_collision_scopes_do_not_cross_independent_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arms = ("prefix_only", "visible_stress")

    one_test = (_actual_records()[0],)
    monkeypatch.setattr(
        collision,
        "derive_analysis_tie_digest",
        lambda *_, **__: "a" * 64,
    )
    collision.bind_actual_analysis_hash_ledger(
        protocol_id="fixture_protocol_v2_1",
        random_ranking_root=10_006,
        stress_permutation_root=10_007,
        records=one_test,
        ranking_partitions=("test",),
        random_ranking_count=1,
        stress_partition="test",
        stress_families=("family_a",),
        stress_permutation_count=1,
        tie_arms=arms,
    )
    monkeypatch.undo()

    across_partitions = (
        _actual_records()[0],
        collision.ActualAnalysisContentRecord(
            partition="audit",
            family_id="family_a",
            member_id="member_audit",
            random_policy_content_sha256="9" * 64,
            predictor_content_hashes=(
                ("prefix_only", "8" * 64),
                ("visible_stress", "7" * 64),
            ),
        ),
    )
    monkeypatch.setattr(
        collision,
        "derive_random_ranking_digest",
        lambda *_, **__: "b" * 64,
    )
    collision.bind_actual_analysis_hash_ledger(
        protocol_id="fixture_protocol_v2_1",
        random_ranking_root=10_006,
        stress_permutation_root=10_007,
        records=across_partitions,
        ranking_partitions=("test", "audit"),
        random_ranking_count=1,
        stress_partition="test",
        stress_families=("family_a",),
        stress_permutation_count=1,
        tie_arms=arms,
    )
    monkeypatch.undo()

    across_families = (
        _actual_records()[0],
        collision.ActualAnalysisContentRecord(
            partition="test",
            family_id="family_b",
            member_id="member_family_b",
            random_policy_content_sha256="9" * 64,
            predictor_content_hashes=(
                ("prefix_only", "8" * 64),
                ("visible_stress", "7" * 64),
            ),
        ),
    )
    monkeypatch.setattr(
        collision,
        "derive_stress_permutation_digest",
        lambda *_, **__: "c" * 64,
    )
    collision.bind_actual_analysis_hash_ledger(
        protocol_id="fixture_protocol_v2_1",
        random_ranking_root=10_006,
        stress_permutation_root=10_007,
        records=across_families,
        ranking_partitions=("test",),
        random_ranking_count=1,
        stress_partition="test",
        stress_families=("family_a", "family_b"),
        stress_permutation_count=1,
        tie_arms=arms,
    )


def test_formal_prediction_and_scoring_capabilities_are_not_constructible() -> None:
    with pytest.raises(TypeError, match="issued only"):
        provenance.V021CommittedModelStateEnvelope(
            _issuer_key=object(),
            validated_model_state=object(),  # type: ignore[arg-type]
            model_state_commitment_artifact_byte_sha256="1" * 64,
            ledger_model_state_commitment_byte_sha256="1" * 64,
            provenance_sha256="2" * 64,
        )
    with pytest.raises(TypeError, match="issued only"):
        v021_io.V021CommittedLabelFreeBundle(
            _seal=object(),
            root=Path.cwd(),
            artifact_contract=object(),  # type: ignore[arg-type]
            design_status="implementation_frozen",
            config_sha256="1" * 64,
            identity=object(),
            frames={},
            file_hashes={},
            ledger_prefix=b"fixture",
            ledger_event_count=1,
            model_state=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="issued only"):
        v021_io.V021PredictionCommitmentEvidence(
            _seal=object(),
            attempt_id="v021-forged",
            byte_sha256="1" * 64,
            artifact_set_sha256="2" * 64,
            actual_analysis_hash_ledger_commitment_byte_sha256="3" * 64,
            file_entries=(),
            ledger_committed=True,
            provenance_sha256="4" * 64,
        )
    with pytest.raises(TypeError, match="issued only"):
        scoring.V021PredictionCommitmentEnvelope(
            _issuer_key=object(),
            protocol_id="synthetic_long_horizon_identifiability_v2_1",
            config_sha256="0" * 64,
            attempt_id="v021-forged",
            evidence=object(),  # type: ignore[arg-type]
            prediction_commitment_byte_sha256="1" * 64,
            artifact_set_sha256="2" * 64,
            artifact_metadata=(),
            provenance_sha256="3" * 64,
        )

    with pytest.raises(prediction.V021PredictionError, match="rejected"):
        prediction.run_formal_prediction_v021(
            label_free_bundle=object(),  # type: ignore[arg-type]
            model_state_envelope=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(scoring.V021ScoringError, match="IO-issued"):
        scoring.score_committed_artifacts(
            prediction_frames={},
            truth_frames={},
            model_state_envelope=object(),  # type: ignore[arg-type]
            prediction_commitment_envelope=object(),  # type: ignore[arg-type]
        )


def test_prediction_metadata_is_exactly_bound_to_sealed_io_evidence() -> None:
    evidence = _sealed_prediction_evidence()
    metadata = scoring._prediction_metadata_from_evidence(evidence)
    committed = {entry["path"]: entry for entry in evidence.file_entries}
    assert metadata == tuple(
        (
            filename,
            committed[filename]["row_count"],
            committed[filename]["byte_count"],
            committed[filename]["byte_sha256"],
        )
        for filename in scoring._PREDICTION_FRAME_FILENAMES
    )

    issuer_source = inspect.getsource(
        scoring._issue_prediction_commitment_envelope_v021
    )
    assert "_require_prediction_commitment_evidence_v021" in issuer_source
    assert "require_ledger_committed=True" in issuer_source
    assert "_canonical_prediction_metadata" in issuer_source
    assert "_prediction_metadata_from_evidence" in issuer_source
    assert "metadata !=" in issuer_source

    entries = list(evidence._file_entries)
    target = next(
        index
        for index, entry in enumerate(entries)
        if entry[0] == "prediction_bundle.csv"
    )
    path, row_count, byte_count, _ = entries[target]
    entries[target] = (path, row_count, byte_count, "0" * 64)
    object.__setattr__(evidence, "_file_entries", tuple(entries))
    with pytest.raises(v021_io.V021IOError, match="digest changed"):
        v021_io._require_prediction_commitment_evidence_v021(
            evidence,
            require_ledger_committed=True,
        )


def test_v016_analysis_uses_only_shared_collision_derivations() -> None:
    source_path = Path(inspect.getsourcefile(analysis) or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    collision_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "lifetwin.experiments.calendar_long_horizon_v016_collision"
        for alias in node.names
    }
    expected = {
        "derive_analysis_tie_digest",
        "derive_bootstrap_seed",
        "derive_random_ranking_digest",
        "derive_stress_permutation_digest",
    }
    assert expected.issubset(collision_imports)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert expected.issubset(called_names)

    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "hashlib" not in imports
    assert "random" not in imports
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"setattr", "delattr"}
        for node in ast.walk(tree)
    )
    assert analysis.derive_analysis_tie_digest is collision.derive_analysis_tie_digest
    assert analysis.derive_bootstrap_seed is collision.derive_bootstrap_seed
    assert (
        analysis.derive_random_ranking_digest is collision.derive_random_ranking_digest
    )
    assert (
        analysis.derive_stress_permutation_digest
        is collision.derive_stress_permutation_digest
    )


def test_public_formal_surfaces_expose_no_scientific_overrides() -> None:
    formal_functions = (
        prediction.fit_verified_generation_bundle_v021,
        prediction.run_formal_prediction_v021,
        scoring.score_committed_artifacts,
        collision.audit_formal_v021_generation_plan,
        collision.bind_formal_v021_actual_analysis_hash_ledger,
    )
    forbidden_fragments = {
        "bootstrap",
        "coverage",
        "issue_count",
        "permutation",
        "random",
        "seed",
        "threshold",
        "worker",
    }
    for function in formal_functions:
        parameters = inspect.signature(function).parameters
        assert not any(
            fragment in parameter
            for parameter in parameters
            for fragment in forbidden_fragments
        ), function.__qualname__
    assert tuple(inspect.signature(scoring.score_committed_artifacts).parameters) == (
        "prediction_frames",
        "truth_frames",
        "model_state_envelope",
        "prediction_commitment_envelope",
    )


def test_prediction_runtime_import_closure_is_truth_incapable() -> None:
    closure = _top_level_import_closure(_PREDICTION_MODULE)
    forbidden_fragments = (
        "_actual_ledger_io",
        "_analysis",
        "_collision",
        "_firewall",
        "_generation",
        "_scoring",
        "_terminal",
    )
    assert not any(
        fragment in module for module in closure for fragment in forbidden_fragments
    )

    source_root = Path(__file__).resolve().parents[1] / "src"
    probe = (
        "import json,sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        f"import {_PREDICTION_MODULE};"
        "print(json.dumps(sorted(name for name in sys.modules "
        f"if name.startswith('{_EXPERIMENT_PREFIX}'))))"
    )
    completed = subprocess.run(
        (sys.executable, "-c", probe),
        check=True,
        capture_output=True,
        text=True,
    )
    imported = json.loads(completed.stdout)
    assert not any(
        fragment in module for module in imported for fragment in forbidden_fragments
    )


def test_prediction_cannot_receive_predecessor_paths_or_family_crosswalks() -> None:
    prediction_parameters = inspect.signature(
        prediction.run_formal_prediction_v021
    ).parameters
    forbidden_parameter_fragments = {
        "decoded",
        "family",
        "mapping",
        "path",
        "predecessor",
        "root",
        "truth",
        "v2_",
    }
    assert not any(
        fragment in parameter
        for parameter in prediction_parameters
        for fragment in forbidden_parameter_fragments
    )
    source = inspect.getsource(prediction)
    forbidden_names = {
        "actual_analysis_content_records_v021",
        "ordinary_family_lookup_v021",
        "sealed_truth_root",
        "score_root",
        "truth_family",
    }
    assert forbidden_names.isdisjoint(source.split())
    assert "calendar_long_horizon_v016_actual_ledger_io" not in source
    assert "calendar_long_horizon_v016_generation" not in source
    assert "calendar_long_horizon_v015_generation" not in source
    assert "model_state_bytes" not in prediction_parameters
    assert "decoded_model_state" not in prediction_parameters

    extract_parameters = inspect.signature(
        v021_io._extract_prediction_inputs_v021
    ).parameters
    assert tuple(extract_parameters) == ("value", "model_state")
    extract_source = inspect.getsource(v021_io._extract_prediction_inputs_v021)
    assert "_require_committed_bundle" in extract_source
    assert "model_state.provenance_sha256" in extract_source
    assert "family" not in extract_source
