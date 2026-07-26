from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v016_actual_ledger_io as actual_io,
)
from lifetwin.experiments import calendar_long_horizon_v016_io as io
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    canonical_json_bytes,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    derive_ordinary_cluster_id,
)
from lifetwin.experiments.calendar_long_horizon_v016_collision import (
    ANALYSIS_TIE_ARMS,
    build_formal_plan_specs,
)
from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    V021ContractView,
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_ledger import (
    AttemptProgress,
    FormalAttemptIdentity,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_PROTOCOL_ID,
)


def _frozen_view() -> V021ContractView:
    return replace(
        load_v021_contract_view(),
        design_status="implementation_frozen",
    )


def _progress(
    *,
    completed_phase: str = "truth_committed",
    pending_phase: str | None = "actual_analysis_hash_ledger_committed",
    generation_hash: str | None = None,
    truth_hash: str | None = None,
    actual_hash: str | None = None,
    model_hash: str | None = None,
) -> AttemptProgress:
    view = _frozen_view()
    return AttemptProgress(
        identity=FormalAttemptIdentity(
            attempt_id="v021-io-fixture",
            git_commit="a" * 40,
            config_byte_sha256=view.artifacts.config_byte_sha256,
        ),
        completed_phase=completed_phase,
        pending_phase=pending_phase,
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(),
        terminal_failed=False,
        generation_plan_commitment_byte_sha256=generation_hash,
        actual_analysis_hash_ledger_commitment_byte_sha256=actual_hash,
        model_state_commitment_byte_sha256=model_hash,
    )


def _write_members(root: Path, filenames: frozenset[str]) -> None:
    for filename in filenames:
        (root / filename).write_bytes(b"{}\n" if filename.endswith(".json") else b"x")


def _evidence_entries() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "path": filename,
            "row_count": index + 1,
            "byte_count": index + 11,
            "byte_sha256": hashlib.sha256(filename.encode("ascii")).hexdigest(),
        }
        for index, filename in enumerate(io._COMMITMENT_FILE_REGISTRY)
    )


def test_formal_io_requires_explicit_implementation_freeze() -> None:
    candidate = replace(
        load_v021_contract_view(),
        design_status="design_candidate_preimplementation",
    )
    with pytest.raises(io.V021IOError, match="implementation-frozen"):
        io._require_contract(candidate)
    assert io._require_contract(_frozen_view()).design_status == (
        "implementation_frozen"
    )


def test_fresh_loader_rejects_partial_and_fake_ledger_roots(
    tmp_path: Path,
) -> None:
    with pytest.raises(io.V021IOError, match="membership changed"):
        io.load_fresh_generation_bundle_v021(
            label_free_root=tmp_path,
            attempt_id="v021-io-fixture",
            contract_view=_frozen_view(),
        )

    _write_members(tmp_path, io._GENERATION_FILES)
    (tmp_path / "exposure_log.jsonl").write_bytes(b"{}\n")
    with pytest.raises(io.V021IOError):
        io.load_fresh_generation_bundle_v021(
            label_free_root=tmp_path,
            attempt_id="v021-io-fixture",
            contract_view=_frozen_view(),
        )


def test_membership_rejects_nonphysical_entry(tmp_path: Path) -> None:
    (tmp_path / "artifact.json").mkdir()
    with pytest.raises(io.V021IOError, match="nonphysical"):
        io._require_membership(
            tmp_path,
            frozenset({"artifact.json"}),
            context="Fixture",
        )


def test_actual_creator_rejects_wrong_ledger_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_members(tmp_path, io._POST_TRUTH_FILES)
    wrong = _progress(
        completed_phase="generation_plan_committed",
        pending_phase="truth_committed",
    )
    monkeypatch.setattr(
        io,
        "_load_ledger",
        lambda *args, **kwargs: (wrong, b"ledger\n", 1),
    )
    with pytest.raises(io.V021IOError, match="pending actual-analysis"):
        io.create_actual_analysis_hash_ledger_commitment_v021(
            label_free_root=tmp_path,
            attempt_id=wrong.identity.attempt_id,
            contract_view=_frozen_view(),
        )


def test_strict_json_rejects_duplicates_and_noncanonical_bytes() -> None:
    with pytest.raises(io.V021IOError, match="Duplicate JSON key"):
        io._strict_json(b'{"a":1,"a":2}\n', filename="duplicate.json")
    with pytest.raises(io.V021IOError, match="not canonical"):
        io._strict_json(b'{"b":2, "a":1}\n', filename="loose.json")


def test_ledger_bound_fake_generation_plan_is_still_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"fake":true}\n'
    (tmp_path / "generation_plan_commitment.json").write_bytes(raw)
    progress = _progress(generation_hash=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        actual_io,
        "recompute_generation_plan_commitment_bytes_v021",
        lambda view: b'{"frozen":true}\n',
    )
    with pytest.raises(io.V021IOError, match="not the frozen formal plan"):
        io._verify_generation_and_truth(
            tmp_path,
            view=_frozen_view(),
            progress=progress,
            require_semantic_plan_recompute=True,
        )


def test_truth_commitment_requires_exact_nine_file_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _frozen_view()
    plan_raw = b'{"plan":"fixture"}\n'
    (tmp_path / "generation_plan_commitment.json").write_bytes(plan_raw)
    entries = [
        {
            "path": filename,
            "row_count": view.artifacts.csv_schema(filename).required_rows,
            "byte_count": 10,
            "byte_sha256": hashlib.sha256(filename.encode("ascii")).hexdigest(),
        }
        for filename in view.artifacts.sealed_filenames[:-1]
    ]
    truth_raw = canonical_json_bytes(
        {
            "protocol_id": V021_PROTOCOL_ID,
            "config_sha256": view.artifacts.config_byte_sha256,
            "files": entries,
            "created_utc": "2026-07-26T00:00:00Z",
            "truth_values_withheld_by_physical_path": True,
        }
    )
    (tmp_path / "truth_commitments.json").write_bytes(truth_raw)
    monkeypatch.setattr(
        actual_io,
        "recompute_generation_plan_commitment_bytes_v021",
        lambda candidate: plan_raw,
    )
    progress = _progress(
        generation_hash=hashlib.sha256(plan_raw).hexdigest(),
        truth_hash=hashlib.sha256(truth_raw).hexdigest(),
    )
    with pytest.raises(io.V021IOError, match="file registry changed"):
        io._verify_generation_and_truth(
            tmp_path,
            view=view,
            progress=progress,
            require_semantic_plan_recompute=True,
        )


def test_actual_ledger_requires_ledger_hash_and_exact_input_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"fixture":"actual"}\n'
    (tmp_path / io._ACTUAL_ANALYSIS_HASH_FILENAME).write_bytes(raw)
    wrong_hash = _progress(actual_hash="0" * 64)
    with pytest.raises(io.V021IOError, match="ledger commitment"):
        io._verify_actual_analysis_hash_ledger(
            tmp_path,
            view=_frozen_view(),
            progress=wrong_hash,
            frames={},
            require_semantic_recompute=True,
        )

    matching = _progress(actual_hash=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        actual_io,
        "verify_actual_analysis_hash_ledger_payload_v021",
        lambda payload, expected_byte_sha256, view: None,
    )
    monkeypatch.setattr(
        io,
        "_recompute_actual_analysis_hash_ledger_bytes",
        lambda frames, view: b'{"fixture":"different-inputs"}\n',
    )
    with pytest.raises(io.V021IOError, match="canonical inputs"):
        io._verify_actual_analysis_hash_ledger(
            tmp_path,
            view=_frozen_view(),
            progress=matching,
            frames={},
            require_semantic_recompute=True,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_model_commitment_registry_is_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    view = _frozen_view()
    entries: list[dict[str, object]] = []
    for filename in io._MODEL_STATE_COMMITMENT_FILES:
        raw = filename.encode("ascii")
        (tmp_path / filename).write_bytes(raw)
        entries.append(
            {
                "path": filename,
                "row_count": 1,
                "byte_count": len(raw),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if mutation == "missing":
        entries.pop()
    elif mutation == "extra":
        entries.append(
            {
                "path": "extra.json",
                "row_count": 1,
                "byte_count": 1,
                "byte_sha256": "0" * 64,
            }
        )
    else:
        entries[0], entries[1] = entries[1], entries[0]
    commitment_raw = canonical_json_bytes(
        {
            "protocol_id": V021_PROTOCOL_ID,
            "config_sha256": view.artifacts.config_byte_sha256,
            "git_commit": "a" * 40,
            "files": entries,
            "created_utc": "2026-07-26T00:00:00Z",
        }
    )
    (tmp_path / "model_state_commitment.json").write_bytes(commitment_raw)
    progress = _progress(
        model_hash=hashlib.sha256(commitment_raw).hexdigest(),
    )
    with pytest.raises(io.V021IOError, match="registry|does not bind"):
        io._verify_model_commitment(
            tmp_path,
            view=view,
            progress=progress,
        )


def test_prediction_evidence_is_sealed_immutable_and_metadata_bound() -> None:
    entries = _evidence_entries()
    actual_hash = next(
        str(entry["byte_sha256"])
        for entry in entries
        if entry["path"] == io._ACTUAL_ANALYSIS_HASH_FILENAME
    )
    with pytest.raises(TypeError, match="issued only"):
        io.V021PredictionCommitmentEvidence(
            _seal=object(),
            attempt_id="v021-forged",
            byte_sha256="1" * 64,
            artifact_set_sha256="2" * 64,
            actual_analysis_hash_ledger_commitment_byte_sha256=actual_hash,
            file_entries=(),
            ledger_committed=True,
            provenance_sha256="3" * 64,
        )
    evidence = io._issue_prediction_commitment_evidence(
        attempt_id="v021-io-fixture",
        byte_sha256="1" * 64,
        artifact_set_sha256="2" * 64,
        actual_analysis_hash_ledger_commitment_byte_sha256=actual_hash,
        file_entries=entries,
        ledger_committed=True,
    )
    copied = evidence.file_entries
    copied[0]["path"] = "tampered.json"
    assert evidence.file_entries[0]["path"] == io._COMMITMENT_FILE_REGISTRY[0]
    assert (
        io._require_prediction_commitment_evidence_v021(
            evidence,
            require_ledger_committed=True,
        )
        is evidence
    )
    object.__setattr__(evidence, "_byte_sha256", "3" * 64)
    with pytest.raises(io.V021IOError, match="digest changed"):
        io._require_prediction_commitment_evidence_v021(
            evidence,
            require_ledger_committed=True,
        )


def test_exclusive_create_never_overwrites_partial_artifact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "prediction_bundle.csv"
    io._exclusive_create(target, b"first")
    with pytest.raises(io.V021IOError, match="already exists"):
        io._exclusive_create(target, b"second")
    assert target.read_bytes() == b"first"


def test_issued_fresh_bundle_detects_post_issuance_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        io, "_verify_stored_bundle_frames", lambda *args, **kwargs: None
    )
    _write_members(tmp_path, io._GENERATION_FILES)
    hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in io._GENERATION_FILES
    }
    ledger_raw = (tmp_path / "exposure_log.jsonl").read_bytes()
    bundle = io.V021FreshGenerationBundle(
        _seal=io._SEAL,
        root=tmp_path,
        contract_view=_frozen_view(),
        identity=_progress().identity,
        frames={},
        file_hashes=hashes,
        ledger_prefix=ledger_raw,
    )
    assert io._require_fresh_bundle(bundle) is bundle
    (tmp_path / "prefix_pack.csv").write_bytes(b"tampered")
    with pytest.raises(io.V021IOError, match="changed after issuance"):
        io._require_fresh_bundle(bundle)


def test_fit_writer_detects_input_race_after_first_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        io, "_verify_stored_bundle_frames", lambda *args, **kwargs: None
    )
    _write_members(tmp_path, io._GENERATION_FILES)
    hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in io._GENERATION_FILES
    }
    bundle = io.V021FreshGenerationBundle(
        _seal=io._SEAL,
        root=tmp_path,
        contract_view=_frozen_view(),
        identity=_progress().identity,
        frames={filename: pd.DataFrame({"row": [1]}) for filename in io._LABEL_INPUTS},
        file_hashes=hashes,
        ledger_prefix=(tmp_path / "exposure_log.jsonl").read_bytes(),
    )
    monkeypatch.setattr(io, "canonical_csv_bytes", lambda *args, **kwargs: b"fit\n")
    exclusive_create = io._exclusive_create

    def racing_create(path: Path, raw: bytes) -> None:
        exclusive_create(path, raw)
        if path.name == io._FIT_OUTPUTS[0]:
            (tmp_path / "prefix_pack.csv").write_bytes(b"raced")

    monkeypatch.setattr(io, "_exclusive_create", racing_create)
    with pytest.raises(io.V021IOError, match="changed after issuance"):
        io._write_verified_fit_outputs_and_commitment_v021(
            bundle,
            frames={
                filename: pd.DataFrame({"row": [1]}) for filename in io._FIT_OUTPUTS
            },
            created_utc="2026-07-26T00:00:00Z",
        )
    assert not (tmp_path / "fit_commitment.json").exists()


def test_sealed_bundle_detects_private_in_memory_frame_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_members(tmp_path, io._GENERATION_FILES)
    hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in io._GENERATION_FILES
    }
    frames = {filename: pd.DataFrame({"value": [1]}) for filename in io._LABEL_INPUTS}
    bundle = io.V021FreshGenerationBundle(
        _seal=io._SEAL,
        root=tmp_path,
        contract_view=_frozen_view(),
        identity=_progress().identity,
        frames=frames,
        file_hashes=hashes,
        ledger_prefix=(tmp_path / "exposure_log.jsonl").read_bytes(),
    )
    monkeypatch.setattr(
        io,
        "canonical_csv_bytes",
        lambda frame, *args, **kwargs: (
            b"x" if int(frame.iloc[0, 0]) == 1 else b"mutated"
        ),
    )
    assert io._require_fresh_bundle(bundle) is bundle
    dict(bundle._frames)["prefix_pack.csv"].iloc[0, 0] = 2
    with pytest.raises(io.V021IOError, match="in-memory frame changed"):
        io._require_fresh_bundle(bundle)


def test_prediction_writer_detects_input_race_after_first_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        io, "_verify_stored_bundle_frames", lambda *args, **kwargs: None
    )
    _write_members(tmp_path, io._PRE_PREDICTION_FILES)
    hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in io._PRE_PREDICTION_FILES
    }
    view = _frozen_view()
    bundle = io.V021CommittedLabelFreeBundle(
        _seal=io._SEAL,
        root=tmp_path,
        artifact_contract=io._prediction_artifact_contract_snapshot(view.artifacts),
        design_status=view.design_status,
        config_sha256=view.artifacts.config_byte_sha256,
        identity=_progress().identity,
        frames={},
        file_hashes=hashes,
        ledger_prefix=(tmp_path / "exposure_log.jsonl").read_bytes(),
        ledger_event_count=1,
        model_state=object(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(io, "canonical_csv_bytes", lambda *args, **kwargs: b"pred\n")
    exclusive_create = io._exclusive_create

    def racing_create(path: Path, raw: bytes) -> None:
        exclusive_create(path, raw)
        if path.name == io._PREDICTION_OUTPUTS[0]:
            (tmp_path / "prefix_pack.csv").write_bytes(b"raced")

    monkeypatch.setattr(io, "_exclusive_create", racing_create)
    with pytest.raises(io.V021IOError, match="artifact changed"):
        io._write_prediction_outputs_v021(
            bundle,
            frames={
                filename: pd.DataFrame({"row": [1]})
                for filename in io._PREDICTION_OUTPUTS
            },
        )


def test_fit_commitment_registry_persists_semantic_generation_proof() -> None:
    assert io._FIT_COMMITMENT_FILES == (
        "generation_plan_commitment.json",
        "prefix_pack.csv",
        "forecast_coordinates.csv",
        "operating_pack.csv",
        "truth_commitments.json",
        "actual_analysis_hash_ledger_commitment.json",
        "member_fit_diagnostics.csv",
        "member_forecast_bundle.csv",
    )


def test_actual_tie_arm_crosswalk_uses_declared_content_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_id = "c_fixture"
    monkeypatch.setattr(
        actual_io,
        "ordinary_family_lookup_v021",
        lambda view: {("test", member_id): "linear"},
    )
    prefix = pd.DataFrame(
        {
            "protocol_id": [V021_PROTOCOL_ID] * 12,
            "partition": ["test"] * 12,
            "cluster_id": [member_id] * 12,
            "prefix_day": list(range(12)),
            "observed_retention_pct": [100.0 - index for index in range(12)],
        }
    )
    forecast = pd.DataFrame(
        {
            "protocol_id": [V021_PROTOCOL_ID] * 8,
            "partition": ["test"] * 8,
            "cluster_id": [member_id] * 8,
            "forecast_day": list(range(12, 20)),
        }
    )
    operating_row = {
        "protocol_id": V021_PROTOCOL_ID,
        "partition": "test",
        "cluster_id": member_id,
        "past_mean_temperature_c": 25.0,
        "past_mean_soc_fraction": 0.5,
        "past_mean_dod_fraction": 0.4,
        "past_efc_per_year": 200.0,
        "planned_mean_temperature_c": 30.0,
        "planned_mean_soc_fraction": 0.6,
        "planned_mean_dod_fraction": 0.5,
        "planned_efc_per_year": 250.0,
        **{f"placebo_control_{index}": float(index) for index in range(1, 9)},
    }
    operating = pd.DataFrame([operating_row])
    records = actual_io.actual_analysis_content_records_v021(
        {
            "prefix_pack.csv": prefix,
            "forecast_coordinates.csv": forecast,
            "operating_pack.csv": operating,
        },
        view=_frozen_view(),
    )
    assert len(records) == 1
    record = records[0]
    by_arm = dict(record.predictor_content_hashes)
    assert tuple(by_arm) == ANALYSIS_TIE_ARMS
    assert by_arm["prefix_only"] == by_arm["strongest_single_feature"]
    assert by_arm["visible_stress"] == by_arm["planned_stress_only"]
    assert by_arm["placebo_8"] not in {
        by_arm["prefix_only"],
        by_arm["visible_stress"],
    }


def test_ordinary_family_crosswalk_matches_shared_generator() -> None:
    view = load_v021_contract_view()
    lookup = io._ordinary_family_lookup(view)
    assert len(lookup) == 2_850
    assert len(set(lookup)) == 2_850
    current, _ = build_formal_plan_specs(view)
    checked = 0
    for group in current.ordinary_groups:
        if group.partition not in {"test", "audit"}:
            continue
        for index in {0, group.count - 1}:
            member_id = derive_ordinary_cluster_id(
                view.protocol,
                partition=group.partition,
                family_id=group.family_id,
                zero_based_index=index,
            )
            assert lookup[(group.partition, member_id)] == group.family_id
            checked += 1
    assert checked > 0
