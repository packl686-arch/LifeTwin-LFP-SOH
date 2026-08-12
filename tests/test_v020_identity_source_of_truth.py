from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v019_analysis as analysis
from lifetwin.experiments import calendar_long_horizon_v019_collision as collision
from lifetwin.experiments import calendar_long_horizon_v019_contract as contract
from lifetwin.experiments import calendar_long_horizon_v019_pipeline as pipeline
from lifetwin.experiments import (
    calendar_long_horizon_v019_prediction_capsule as capsule,
)
from lifetwin.experiments import calendar_long_horizon_v019_state as state
from lifetwin.experiments import calendar_long_horizon_v019_scoring as scoring
from lifetwin.experiments import calendar_long_horizon_v019_terminal as terminal


def _alternate_view(tmp_path) -> contract.V024ContractView:
    base = contract.resolve_contract_view(None)
    protocol_id = "synthetic_identity_fixture"
    roots = tuple(
        (name, value + 1_000_000_000) for name, value in base.protocol.seed_roots
    )
    config_payload = base.protocol.config()
    config_payload["protocol_id"] = protocol_id
    config_payload["design_partitions"]["seed_roots"] = dict(roots)
    config_json = json.dumps(
        config_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    amendment_payload = json.loads(base.artifacts.config_path.read_text("utf-8"))
    amendment_payload["protocol_id"] = protocol_id
    amendment_payload["status"] = "fixture_authenticated"
    amendment_payload["fresh_generation"]["seed_roots"] = dict(roots)
    amendment_json = json.dumps(
        amendment_payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
    )
    config_path = tmp_path / "alternate_contract.json"
    config_path.write_text(amendment_json + "\n", encoding="ascii")
    byte_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    canonical_hash = hashlib.sha256(
        json.dumps(
            amendment_payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return contract.V024ContractView(
        protocol=replace(
            base.protocol,
            protocol_id=protocol_id,
            config_sha256=byte_hash,
            seed_roots=roots,
            config_json=config_json,
        ),
        artifacts=replace(
            base.artifacts,
            protocol_id=protocol_id,
            config_path=config_path.resolve(),
            config_byte_sha256=byte_hash,
        ),
        design_status="fixture_authenticated",
        config_canonical_sha256=canonical_hash,
        whole_rows=base.whole_rows,
        partition_rows=base.partition_rows,
        base_config_canonical_sha256=base.base_config_canonical_sha256,
        base_config_byte_sha256=base.base_config_byte_sha256,
    )


def test_alternate_authenticated_identity_reaches_generic_paths(tmp_path) -> None:
    view = _alternate_view(tmp_path)
    protocol_id = view.protocol.protocol_id
    current, _ = collision.build_formal_plan_specs(view)
    assert current.protocol_id == protocol_id
    assert current.seed_roots == view.protocol.seed_roots
    assert state._require_contract_identity(view) == (
        protocol_id,
        view.artifacts.config_byte_sha256,
    )

    trajectories = pd.DataFrame(
        {
            "canonical_prefix_content_sha256": ("1" * 64, "2" * 64),
            "hard_eligible_visible_stress": (True, True),
            "catastrophic": (False, True),
        }
    )
    rankings = analysis.deterministic_random_rankings(
        trajectories,
        issue_count=1,
        rankings=2,
        contract_view=view,
    )
    assert len(rankings) == 2
    assert scoring._contract(view) is view

    content_hash = "3" * 64
    assert pipeline._tie_hash_v024(protocol_id, "prefix_only", content_hash) != (
        pipeline._tie_hash_v024(
            contract.resolve_contract_view(None).protocol.protocol_id,
            "prefix_only",
            content_hash,
        )
    )

    frame = pd.DataFrame(
        {
            "protocol_id": [protocol_id],
            "partition": ["test"],
            "cluster_id": ["fixture-cluster"],
            "prefix_day": [0.0],
            "observed_retention_pct": [100.0],
        }
    )
    capsule.canonicalize_frame(
        frame,
        "prefix_pack.csv",
        formal=False,
        protocol_id=protocol_id,
    )
    with pytest.raises(capsule.V024PredictionCapsuleError):
        capsule.canonicalize_frame(
            frame,
            "prefix_pack.csv",
            formal=False,
            protocol_id="different_identity_fixture",
        )

    context = terminal.TerminalContext(
        protocol_id=protocol_id,
        attempt_id="fixture-attempt",
        git_commit="0" * 40,
        git_dirty=False,
        config_byte_sha256=view.artifacts.config_byte_sha256,
        created_utc="2026-08-12T00:00:00Z",
        terminated_utc="2026-08-12T00:00:01Z",
        attempted_phase="center_state_committed",
        last_completed_phase="center_truth_opened",
        truth_commitments_byte_sha256=None,
    )
    assert context.protocol_id == protocol_id


def test_default_identity_keeps_canonical_behavior() -> None:
    view = contract.resolve_contract_view(None)
    trajectories = pd.DataFrame(
        {
            "canonical_prefix_content_sha256": ("1" * 64, "2" * 64),
            "hard_eligible_visible_stress": (True, True),
            "catastrophic": (False, True),
        }
    )
    implicit = analysis.deterministic_random_rankings(
        trajectories,
        issue_count=1,
        rankings=2,
    )
    explicit = analysis.deterministic_random_rankings(
        trajectories,
        issue_count=1,
        rankings=2,
        protocol_id=view.protocol.protocol_id,
        contract_view=view,
    )
    pd.testing.assert_frame_equal(implicit, explicit)

    frame = pd.DataFrame(
        {
            "protocol_id": [view.protocol.protocol_id],
            "partition": ["test"],
            "cluster_id": ["fixture-cluster"],
            "prefix_day": [0.0],
            "observed_retention_pct": [100.0],
        }
    )
    assert capsule.canonical_csv_bytes(
        frame,
        "prefix_pack.csv",
        formal=False,
    ) == capsule.canonical_csv_bytes(
        frame,
        "prefix_pack.csv",
        formal=False,
        protocol_id=view.protocol.protocol_id,
    )


def test_contract_view_rejects_cross_object_identity_drift(tmp_path) -> None:
    view = _alternate_view(tmp_path)
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            artifacts=replace(view.artifacts, protocol_id="different_identity_fixture"),
        )
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            protocol=replace(
                view.protocol,
                seed_roots=tuple(
                    (name, root + (1 if index == 0 else 0))
                    for index, (name, root) in enumerate(view.protocol.seed_roots)
                ),
            ),
        )


def test_contract_view_rejects_unbound_scientific_and_artifact_drift(tmp_path) -> None:
    view = _alternate_view(tmp_path)
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            protocol=replace(
                view.protocol,
                time_scale_days=view.protocol.time_scale_days + 1.0,
            ),
        )

    config_payload = view.protocol.config()
    config_payload["time_grid"]["days_per_year_for_labels"] += 1.0
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            protocol=replace(
                view.protocol,
                config_json=json.dumps(
                    config_payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
            ),
        )

    reduced_schemas = dict(view.artifacts.csv_schemas)
    reduced_schemas.pop(next(iter(reduced_schemas)))
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            artifacts=replace(
                view.artifacts,
                csv_schemas=MappingProxyType(reduced_schemas),
            ),
        )
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            artifacts=replace(
                view.artifacts,
                truth_filenames=view.artifacts.truth_filenames[:-1],
            ),
        )
    reduced_allowlists = dict(view.artifacts.json_key_allowlists)
    reduced_allowlists.pop(next(iter(reduced_allowlists)))
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            artifacts=replace(
                view.artifacts,
                json_key_allowlists=MappingProxyType(reduced_allowlists),
            ),
        )
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            artifacts=replace(
                view.artifacts,
                exposure_keys=frozenset((*view.artifacts.exposure_keys, "extra")),
            ),
        )
    changed_member_counts = dict(view.artifacts.partition_member_counts)
    partition = next(iter(changed_member_counts))
    changed_member_counts[partition] += 1
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            artifacts=replace(
                view.artifacts,
                partition_member_counts=MappingProxyType(changed_member_counts),
            ),
        )


def test_contract_view_rejects_mutable_or_unbound_row_registries(tmp_path) -> None:
    view = _alternate_view(tmp_path)
    with pytest.raises(contract.V024ContractError):
        replace(view, whole_rows=dict(view.whole_rows))

    changed_whole = dict(view.whole_rows)
    changed_whole["prefix_pack.csv"] += 1
    with pytest.raises(contract.V024ContractError):
        replace(view, whole_rows=MappingProxyType(changed_whole))

    changed_partitions = dict(view.partition_rows)
    counts = changed_partitions["center_development"]
    changed_partitions["center_development"] = (counts[0] + 1, *counts[1:])
    with pytest.raises(contract.V024ContractError):
        replace(view, partition_rows=MappingProxyType(changed_partitions))


def test_contract_view_snapshots_mapping_proxy_backing_dicts(tmp_path) -> None:
    view = _alternate_view(tmp_path)
    whole_backing = dict(view.whole_rows)
    schema_backing = dict(view.artifacts.csv_schemas)
    snapshotted = replace(
        view,
        whole_rows=MappingProxyType(whole_backing),
        artifacts=replace(
            view.artifacts,
            csv_schemas=MappingProxyType(schema_backing),
        ),
    )

    whole_backing["prefix_pack.csv"] += 1
    schema_backing.pop(next(iter(schema_backing)))

    assert snapshotted.whole_rows == view.whole_rows
    assert snapshotted.artifacts.csv_schemas == view.artifacts.csv_schemas
    assert contract.require_contract_view(snapshotted) is snapshotted


def test_contract_view_rejects_amendment_commitment_drift(tmp_path) -> None:
    view = _alternate_view(tmp_path)
    with pytest.raises(contract.V024ContractError):
        replace(view, config_canonical_sha256="0" * 64)

    view.artifacts.config_path.write_text(
        view.artifacts.config_path.read_text("ascii") + " ",
        encoding="ascii",
    )
    with pytest.raises(contract.V024ContractError):
        contract.require_contract_view(view)


def test_contract_view_rejects_self_committed_amendment_science_drift(
    tmp_path,
) -> None:
    view = _alternate_view(tmp_path)
    payload = json.loads(view.artifacts.config_path.read_text("ascii"))
    payload["numeric_output_contract"]["infinity_forbidden"] = False
    amended = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
    )
    view.artifacts.config_path.write_text(amended + "\n", encoding="ascii")
    byte_hash = hashlib.sha256(view.artifacts.config_path.read_bytes()).hexdigest()
    canonical_hash = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(contract.V024ContractError):
        replace(
            view,
            protocol=replace(view.protocol, config_sha256=byte_hash),
            artifacts=replace(view.artifacts, config_byte_sha256=byte_hash),
            config_canonical_sha256=canonical_hash,
        )


def test_legacy_identity_names_remain_only_at_boundary_adapters() -> None:
    root = Path(__file__).parents[1] / "src" / "lifetwin" / "experiments"
    allow = {
        "V024_PROTOCOL_ID": {
            "calendar_long_horizon_v019_contract.py",
            "calendar_long_horizon_v019_prediction_environment.py",
            "calendar_long_horizon_v019_protocol.py",
        },
        "V024_EXPECTED_SEED_ROOTS": {
            "calendar_long_horizon_v019_contract.py",
            "calendar_long_horizon_v019_protocol.py",
        },
        "load_v024_design": {
            "calendar_long_horizon_v019_contract.py",
        },
        "load_v024_contract_view": {
            "calendar_long_horizon_v019_contract.py",
        },
    }
    observed = {name: set() for name in allow}
    for path in root.glob("calendar_long_horizon_v019_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for name in observed:
            if name in names:
                observed[name].add(path.name)
    assert observed == allow
