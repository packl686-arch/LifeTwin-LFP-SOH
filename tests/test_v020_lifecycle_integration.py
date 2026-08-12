from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v019_firewall as firewall
from lifetwin.experiments import calendar_long_horizon_v019_io as io
from lifetwin.experiments import calendar_long_horizon_v019_runner as runner
from lifetwin.experiments.calendar_long_horizon_v015_io import canonical_json_bytes
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    load_v024_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v019_ledger import (
    AttemptProgress,
    FormalAttemptIdentity,
)
from lifetwin.experiments.calendar_long_horizon_v019_protocol import V024_PROTOCOL_ID
from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
    INPUT_FILENAMES_BY_STAGE,
)


_NEXT_TRUTH = {
    "center_development": "risk_development_truth.csv",
    "risk_development": "calibration_truth.csv",
}


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    label = tmp_path / "label-free"
    sealed = tmp_path / "sealed-truth"
    label.mkdir()
    sealed.mkdir()
    filenames = set().union(*map(set, INPUT_FILENAMES_BY_STAGE.values()))
    filenames.update(_NEXT_TRUTH.values())
    for index, filename in enumerate(sorted(filenames)):
        root = sealed if filename.endswith("_truth.csv") else label
        (root / filename).write_bytes(f"fixture/{index}/{filename}\n".encode())
    return label, sealed


def _producer_hashes(
    stage: str,
    *,
    label: Path,
    sealed: Path,
    registry: object = INPUT_FILENAMES_BY_STAGE,
) -> dict[str, str]:
    truth_filename = next(
        name for name in INPUT_FILENAMES_BY_STAGE[stage] if name.endswith("_truth.csv")
    )
    values = runner._phase_input_bytes(
        paths=SimpleNamespace(
            label_free_root=label,
            sealed_truth_root=sealed,
        ),
        label_filenames=(),
        truth_filename=truth_filename,
        _input_filenames_by_stage=registry,
        _stage=stage,
    )
    assert tuple(values) == INPUT_FILENAMES_BY_STAGE[stage]
    return runner._input_hashes(values)


def _truth_hashes(sealed: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sealed.iterdir()
    }


def _center_checkpoint(
    *,
    label: Path,
    sealed: Path,
    config_sha256: str,
) -> tuple[dict[str, object], bytes]:
    payload: dict[str, object] = {
        "protocol_id": V024_PROTOCOL_ID,
        "config_sha256": config_sha256,
        "state_kind": "center_development",
        "center_state_sha256": "1" * 64,
        "center_beta": 1.0,
        "development_cluster_count": 1,
        "forecast_horizon_count": 1,
        "ridge_penalty": 0.0,
        "completeness_rule": "deterministic fixture",
        "input_byte_hashes": _producer_hashes(
            "center_development",
            label=label,
            sealed=sealed,
        ),
        "created_utc": "2026-08-12T00:00:00+00:00",
    }
    raw = canonical_json_bytes(payload)
    (label / "center_state_checkpoint.json").write_bytes(raw)
    return payload, raw


def _risk_checkpoint(
    *,
    label: Path,
    sealed: Path,
    config_sha256: str,
    center: dict[str, object],
    center_raw: bytes,
) -> tuple[dict[str, object], bytes]:
    risk_inputs = _producer_hashes("risk_development", label=label, sealed=sealed)
    training = {
        "protocol_id": V024_PROTOCOL_ID,
        "config_sha256": config_sha256,
        "center_state_sha256": center["center_state_sha256"],
        "risk_state_sha256": "3" * 64,
        "center_development_input_hashes": center["input_byte_hashes"],
        "risk_development_input_hashes": risk_inputs,
        "opened_truth_files": [
            "center_development_truth.csv",
            "risk_development_truth.csv",
        ],
        "forbidden_v1_evidence_matches": [],
        "created_utc": "2026-08-12T00:00:01+00:00",
    }
    training_raw = canonical_json_bytes(training)
    (label / "training_manifest.json").write_bytes(training_raw)
    payload: dict[str, object] = {
        "protocol_id": V024_PROTOCOL_ID,
        "config_sha256": config_sha256,
        "state_kind": "risk_development",
        "center_checkpoint_byte_sha256": hashlib.sha256(center_raw).hexdigest(),
        "training_manifest_byte_sha256": hashlib.sha256(training_raw).hexdigest(),
        "risk_state_sha256": training["risk_state_sha256"],
        "development_cluster_count": 2,
        "eligible_cluster_count": 2,
        "positive_label_count": 1,
        "negative_label_count": 1,
        "input_byte_hashes": risk_inputs,
        "created_utc": "2026-08-12T00:00:02+00:00",
    }
    raw = canonical_json_bytes(payload)
    (label / "risk_state_checkpoint.json").write_bytes(raw)
    return payload, raw


def _write_truth_commitment(
    *,
    label: Path,
    sealed: Path,
    contract: object,
) -> str:
    entries = []
    for index, filename in enumerate(contract.sealed_filenames):
        path = sealed / filename
        if not path.exists():
            path.write_bytes(f"sealed fixture/{index}/{filename}\n".encode())
        raw = path.read_bytes()
        entries.append(
            {
                "path": filename,
                "row_count": contract.csv_schema(filename).required_rows,
                "byte_count": len(raw),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    raw = canonical_json_bytes(
        {
            "protocol_id": V024_PROTOCOL_ID,
            "config_sha256": contract.config_byte_sha256,
            "files": entries,
            "created_utc": "2026-08-12T00:00:00+00:00",
            "truth_values_withheld_by_physical_path": True,
        }
    )
    (label / "truth_commitments.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _append_completed_phase(
    *,
    label: Path,
    identity: FormalAttemptIdentity,
    contract: object,
    phase: str,
    index: int,
    truth_hash: str | None,
    artifact_hash: str | None = None,
) -> None:
    message = (
        firewall.phase_commitment_message(phase, artifact_hash)
        if artifact_hash is not None
        else "deterministic fixture transition"
    )
    firewall.append_formal_exposure_event(
        path=label / "exposure_log.jsonl",
        identity=identity,
        contract=contract,
        created_utc=f"2026-08-12T00:{index:02d}:00+00:00",
        phase=phase,
        exit_status="completed",
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=None,
        message=message,
    )


def _ledger_through_center(
    *,
    label: Path,
    identity: FormalAttemptIdentity,
    contract: object,
    truth_hash: str,
    center_hash: str,
) -> None:
    commitments = {
        "generation_plan_committed": hashlib.sha256(
            (label / "generation_plan_commitment.json").read_bytes()
        ).hexdigest(),
        "actual_analysis_hash_ledger_committed": hashlib.sha256(
            (label / "actual_analysis_hash_ledger_commitment.json").read_bytes()
        ).hexdigest(),
        "label_free_fit_committed": hashlib.sha256(
            (label / "fit_commitment.json").read_bytes()
        ).hexdigest(),
        "center_state_committed": center_hash,
    }
    phases = (
        "before_generation",
        "generation_plan_committed",
        "truth_committed",
        "actual_analysis_hash_ledger_committed",
        "label_free_fit_committed",
        "center_truth_opened",
        "center_state_committed",
    )
    for index, phase in enumerate(phases):
        _append_completed_phase(
            label=label,
            identity=identity,
            contract=contract,
            phase=phase,
            index=index,
            truth_hash=(None if index < 2 else truth_hash),
            artifact_hash=commitments.get(phase),
        )


def _patch_independent_reveal_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    opened: list[str],
) -> None:
    for name in (
        "_verify_generation_plan",
        "_verify_actual_analysis_hash_ledger",
        "_verify_fit_commitment",
        "_verify_mask_commitment",
    ):
        monkeypatch.setattr(firewall, name, lambda **_: None)

    def read_spy(path: Path, contract: object, *, formal: bool) -> pd.DataFrame:
        del formal
        opened.append(path.name)
        return pd.DataFrame(index=range(contract.csv_schema(path.name).required_rows))

    monkeypatch.setattr(firewall, "read_canonical_csv", read_spy)


def _mutate_registry_or_bytes(
    mutation: str,
    *,
    stage: str,
    hashes: dict[str, str],
    label: Path,
    sealed: Path,
) -> dict[str, str]:
    mutated = dict(hashes)
    filename = INPUT_FILENAMES_BY_STAGE[stage][0]
    if mutation == "missing":
        mutated.pop(filename)
    elif mutation == "extra":
        mutated["unexpected.json"] = "0" * 64
    elif mutation == "renamed":
        mutated[f"renamed-{filename}"] = mutated.pop(filename)
    else:
        root = sealed if filename.endswith("_truth.csv") else label
        (root / filename).write_bytes(b"changed\n")
    return mutated


@pytest.mark.parametrize(
    ("stage", "expected_truth"),
    tuple(_NEXT_TRUTH.items()),
)
def test_shared_registry_helper_opens_once_after_validation(
    tmp_path: Path,
    stage: str,
    expected_truth: str,
) -> None:
    label, sealed = _roots(tmp_path)
    hashes = _producer_hashes(stage, label=label, sealed=sealed)
    opened: list[str] = []

    def open_truth() -> bytes:
        opened.append(expected_truth)
        return (sealed / expected_truth).read_bytes()

    result = firewall._open_after_checkpoint_registry_v020(
        stage=stage,
        input_byte_hashes=hashes,
        label_root=label,
        sealed_root=sealed,
        _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
        opener=open_truth,
    )
    assert result
    assert opened == [expected_truth]


@pytest.mark.parametrize("stage", tuple(INPUT_FILENAMES_BY_STAGE))
@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed", "bytes"])
def test_shared_registry_helper_rejects_before_open(
    tmp_path: Path,
    stage: str,
    mutation: str,
) -> None:
    label, sealed = _roots(tmp_path)
    hashes = _producer_hashes(stage, label=label, sealed=sealed)
    mutated = _mutate_registry_or_bytes(
        mutation,
        stage=stage,
        hashes=hashes,
        label=label,
        sealed=sealed,
    )
    opened: list[bool] = []
    with pytest.raises(firewall.V024FirewallError, match="input integrity"):
        firewall._open_after_checkpoint_registry_v020(
            stage=stage,
            input_byte_hashes=mutated,
            label_root=label,
            sealed_root=sealed,
            _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
            opener=lambda: opened.append(True),
        )
    assert opened == []


def test_open_truth_for_phase_validates_real_checkpoints_before_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label, sealed = _roots(tmp_path)
    view = load_v024_contract_view()
    truth_hash = _write_truth_commitment(
        label=label,
        sealed=sealed,
        contract=view.artifacts,
    )
    center, center_raw = _center_checkpoint(
        label=label,
        sealed=sealed,
        config_sha256=view.artifacts.config_byte_sha256,
    )
    _, risk_raw = _risk_checkpoint(
        label=label,
        sealed=sealed,
        config_sha256=view.artifacts.config_byte_sha256,
        center=center,
        center_raw=center_raw,
    )
    identity = FormalAttemptIdentity(
        "development-fixture",
        "2" * 40,
        view.artifacts.config_byte_sha256,
    )
    center_hash = hashlib.sha256(center_raw).hexdigest()
    _ledger_through_center(
        label=label,
        identity=identity,
        contract=view.artifacts,
        truth_hash=truth_hash,
        center_hash=center_hash,
    )
    opened: list[str] = []
    _patch_independent_reveal_prerequisites(monkeypatch, opened)

    risk_truth = firewall.open_truth_for_phase(
        ledger_path=label / "exposure_log.jsonl",
        identity=identity,
        contract=view.artifacts,
        commitment_path=label / "truth_commitments.json",
        sealed_truth_root=sealed,
        label_free_root=label,
        phase="risk_truth_opened",
        created_utc="2026-08-12T00:07:00+00:00",
        formal=False,
        _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
    )
    assert tuple(risk_truth) == ("risk_development_truth.csv",)
    assert opened == ["risk_development_truth.csv"]

    risk_hash = hashlib.sha256(risk_raw).hexdigest()
    _append_completed_phase(
        label=label,
        identity=identity,
        contract=view.artifacts,
        phase="risk_state_committed",
        index=8,
        truth_hash=truth_hash,
        artifact_hash=risk_hash,
    )
    mask_raw = (label / "calibration_mask_commitment.json").read_bytes()
    _append_completed_phase(
        label=label,
        identity=identity,
        contract=view.artifacts,
        phase="calibration_mask_committed",
        index=9,
        truth_hash=truth_hash,
        artifact_hash=hashlib.sha256(mask_raw).hexdigest(),
    )
    calibration_truth = firewall.open_truth_for_phase(
        ledger_path=label / "exposure_log.jsonl",
        identity=identity,
        contract=view.artifacts,
        commitment_path=label / "truth_commitments.json",
        sealed_truth_root=sealed,
        label_free_root=label,
        phase="calibration_truth_opened",
        created_utc="2026-08-12T00:10:00+00:00",
        calibration_mask_commitment_path=(label / "calibration_mask_commitment.json"),
        formal=False,
        _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
    )
    assert tuple(calibration_truth) == ("calibration_truth.csv",)
    assert opened == ["risk_development_truth.csv", "calibration_truth.csv"]


@pytest.mark.parametrize("stage", ["center_development", "risk_development"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed", "bytes"])
def test_open_truth_for_phase_rejects_registry_before_sealed_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    mutation: str,
) -> None:
    label, sealed = _roots(tmp_path)
    view = load_v024_contract_view()
    truth_hash = _write_truth_commitment(
        label=label,
        sealed=sealed,
        contract=view.artifacts,
    )
    center, center_raw = _center_checkpoint(
        label=label,
        sealed=sealed,
        config_sha256=view.artifacts.config_byte_sha256,
    )
    risk, risk_raw = _risk_checkpoint(
        label=label,
        sealed=sealed,
        config_sha256=view.artifacts.config_byte_sha256,
        center=center,
        center_raw=center_raw,
    )
    payload, path = (
        (center, label / "center_state_checkpoint.json")
        if stage == "center_development"
        else (risk, label / "risk_state_checkpoint.json")
    )
    payload["input_byte_hashes"] = _mutate_registry_or_bytes(
        mutation,
        stage=stage,
        hashes=dict(payload["input_byte_hashes"]),
        label=label,
        sealed=sealed,
    )
    if mutation != "bytes":
        path.write_bytes(canonical_json_bytes(payload))
    center_raw = (label / "center_state_checkpoint.json").read_bytes()
    risk_raw = (label / "risk_state_checkpoint.json").read_bytes()
    identity = FormalAttemptIdentity(
        "development-fixture",
        "2" * 40,
        view.artifacts.config_byte_sha256,
    )
    _ledger_through_center(
        label=label,
        identity=identity,
        contract=view.artifacts,
        truth_hash=truth_hash,
        center_hash=hashlib.sha256(center_raw).hexdigest(),
    )
    phase = "risk_truth_opened"
    mask_path: Path | None = None
    if stage == "risk_development":
        _append_completed_phase(
            label=label,
            identity=identity,
            contract=view.artifacts,
            phase="risk_truth_opened",
            index=7,
            truth_hash=truth_hash,
        )
        _append_completed_phase(
            label=label,
            identity=identity,
            contract=view.artifacts,
            phase="risk_state_committed",
            index=8,
            truth_hash=truth_hash,
            artifact_hash=hashlib.sha256(risk_raw).hexdigest(),
        )
        mask_path = label / "calibration_mask_commitment.json"
        _append_completed_phase(
            label=label,
            identity=identity,
            contract=view.artifacts,
            phase="calibration_mask_committed",
            index=9,
            truth_hash=truth_hash,
            artifact_hash=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        )
        phase = "calibration_truth_opened"

    opened: list[str] = []
    _patch_independent_reveal_prerequisites(monkeypatch, opened)
    with pytest.raises(firewall.V024FirewallError):
        firewall.open_truth_for_phase(
            ledger_path=label / "exposure_log.jsonl",
            identity=identity,
            contract=view.artifacts,
            commitment_path=label / "truth_commitments.json",
            sealed_truth_root=sealed,
            label_free_root=label,
            phase=phase,
            created_utc="2026-08-12T00:10:00+00:00",
            calibration_mask_commitment_path=mask_path,
            formal=False,
            _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
        )
    assert opened == []


@pytest.mark.parametrize("stage", tuple(INPUT_FILENAMES_BY_STAGE))
def test_io_training_chain_resolves_exact_stage_registry(
    tmp_path: Path,
    stage: str,
) -> None:
    label, sealed = _roots(tmp_path)
    hashes = _producer_hashes(stage, label=label, sealed=sealed)
    assert (
        io._resolve_stage_input_hashes(
            hashes,
            root=label,
            truth_hashes=_truth_hashes(sealed),
            context=stage,
            stage=stage,
            _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
        )
        == hashes
    )


@pytest.mark.parametrize("stage", tuple(INPUT_FILENAMES_BY_STAGE))
@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed", "bytes"])
def test_io_training_chain_rejects_every_registry_mutation(
    tmp_path: Path,
    stage: str,
    mutation: str,
) -> None:
    label, sealed = _roots(tmp_path)
    hashes = _producer_hashes(stage, label=label, sealed=sealed)
    truth_hashes = _truth_hashes(sealed)
    mutated = _mutate_registry_or_bytes(
        mutation,
        stage=stage,
        hashes=hashes,
        label=label,
        sealed=sealed,
    )
    with pytest.raises(io.V024IOError, match="registry changed|input changed"):
        io._resolve_stage_input_hashes(
            mutated,
            root=label,
            truth_hashes=truth_hashes,
            context=stage,
            stage=stage,
            _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
        )


def test_model_state_chain_is_exact_at_stage_and_filename_levels(
    tmp_path: Path,
) -> None:
    label, sealed = _roots(tmp_path)
    expected = {
        stage: io._resolve_stage_input_hashes(
            _producer_hashes(stage, label=label, sealed=sealed),
            root=label,
            truth_hashes=_truth_hashes(sealed),
            context=stage,
            stage=stage,
            _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
        )
        for stage in INPUT_FILENAMES_BY_STAGE
    }
    io._require_model_state_input_hashes(expected, expected)

    for mutation in ("missing", "extra", "renamed", "nested"):
        value = deepcopy(expected)
        if mutation == "missing":
            value.pop("center_development")
        elif mutation == "extra":
            value["unexpected"] = {}
        elif mutation == "renamed":
            value["center"] = value.pop("center_development")
        else:
            value["calibration"].pop(next(iter(value["calibration"])))
        with pytest.raises(io.V024IOError, match="Model-state input hashes"):
            io._require_model_state_input_hashes(value, expected)


def test_equal_but_noncanonical_registry_cannot_be_an_override(tmp_path: Path) -> None:
    label, sealed = _roots(tmp_path)
    copied = dict(INPUT_FILENAMES_BY_STAGE)
    with pytest.raises(runner.V024RunnerError, match="registry is invalid"):
        _producer_hashes(
            "center_development",
            label=label,
            sealed=sealed,
            registry=copied,
        )
    opened: list[bool] = []
    with pytest.raises(firewall.V024FirewallError, match="input integrity"):
        firewall._open_after_checkpoint_registry_v020(
            stage="center_development",
            input_byte_hashes={},
            label_root=label,
            sealed_root=sealed,
            _input_filenames_by_stage=copied,
            opener=lambda: opened.append(True),
        )
    assert opened == []
    with pytest.raises(io.V024IOError, match="registry is invalid"):
        io._resolve_stage_input_hashes(
            {},
            root=label,
            truth_hashes=_truth_hashes(sealed),
            context="center",
            stage="center_development",
            _input_filenames_by_stage=copied,
        )


def test_internal_registry_seam_is_not_a_formal_runner_override() -> None:
    assert (
        "_input_filenames_by_stage"
        in inspect.signature(firewall.open_truth_for_phase).parameters
    )
    assert (
        "_input_filenames_by_stage"
        in inspect.signature(io.load_committed_label_free_bundle_v024).parameters
    )
    assert (
        "_input_filenames_by_stage"
        not in inspect.signature(runner.run_formal_attempt).parameters
    )


def test_v019_default_producer_and_io_paths_remain_permissive(tmp_path: Path) -> None:
    label, sealed = _roots(tmp_path)
    values = runner._phase_input_bytes(
        paths=SimpleNamespace(
            label_free_root=label,
            sealed_truth_root=sealed,
        ),
        label_filenames=("prefix_pack.csv",),
        truth_filename="center_development_truth.csv",
    )
    assert tuple(values) == ("prefix_pack.csv", "center_development_truth.csv")
    hashes = runner._input_hashes({"prefix_pack.csv": values["prefix_pack.csv"]})
    assert (
        io._resolve_stage_input_hashes(
            hashes,
            root=label,
            truth_hashes={},
            context="legacy",
            stage="unused",
            _input_filenames_by_stage=None,
        )
        == hashes
    )


def test_center_checkpoint_real_firewall_uses_the_shared_ten_key_registry(
    tmp_path: Path,
) -> None:
    label, sealed = _roots(tmp_path)
    view = load_v024_contract_view()
    payload, raw = _center_checkpoint(
        label=label,
        sealed=sealed,
        config_sha256=view.artifacts.config_byte_sha256,
    )
    identity = FormalAttemptIdentity(
        "development-fixture",
        "2" * 40,
        view.artifacts.config_byte_sha256,
    )
    progress = AttemptProgress(
        identity=identity,
        completed_phase="center_state_committed",
        pending_phase=None,
        truth_commitments_byte_sha256=None,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=("center_development_truth.csv",),
        terminal_failed=False,
        center_state_checkpoint_byte_sha256=hashlib.sha256(raw).hexdigest(),
    )
    observed = firewall._verify_center_checkpoint(
        label_root=label,
        sealed_root=sealed,
        progress=progress,
        contract=view.artifacts,
        _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
    )
    assert observed["input_byte_hashes"] == payload["input_byte_hashes"]


def test_risk_checkpoint_real_firewall_uses_the_shared_eleven_key_registry(
    tmp_path: Path,
) -> None:
    label, sealed = _roots(tmp_path)
    view = load_v024_contract_view()
    center, center_raw = _center_checkpoint(
        label=label,
        sealed=sealed,
        config_sha256=view.artifacts.config_byte_sha256,
    )
    risk, risk_raw = _risk_checkpoint(
        label=label,
        sealed=sealed,
        config_sha256=view.artifacts.config_byte_sha256,
        center=center,
        center_raw=center_raw,
    )
    identity = FormalAttemptIdentity(
        "development-fixture",
        "2" * 40,
        view.artifacts.config_byte_sha256,
    )
    progress = AttemptProgress(
        identity=identity,
        completed_phase="risk_state_committed",
        pending_phase=None,
        truth_commitments_byte_sha256=None,
        prediction_commitment_byte_sha256=None,
        opened_truth_files=(
            "center_development_truth.csv",
            "risk_development_truth.csv",
        ),
        terminal_failed=False,
        center_state_checkpoint_byte_sha256=hashlib.sha256(center_raw).hexdigest(),
        risk_state_checkpoint_byte_sha256=hashlib.sha256(risk_raw).hexdigest(),
    )
    observed = firewall._verify_risk_checkpoint(
        label_root=label,
        sealed_root=sealed,
        progress=progress,
        contract=view.artifacts,
        _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
    )
    assert observed["input_byte_hashes"] == risk["input_byte_hashes"]


def test_ci_runs_reproduction_only_on_reviewed_release_refs() -> None:
    workflow = (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    quality, reproduce = workflow.split("\n  reproduce:\n", maxsplit=1)
    assert "\n    if:" not in quality
    assert reproduce.startswith(
        "    name: reproduce (${{ matrix.os }})\n"
        "    if: >-\n"
        "      github.event_name == 'pull_request' ||\n"
        "      github.ref == 'refs/heads/main' ||\n"
        "      startsWith(github.ref, 'refs/tags/v')\n"
    )
    upload = reproduce.split("      - uses: actions/upload-artifact@v4\n", maxsplit=1)[
        1
    ]
    assert "          if-no-files-found: warn\n" in upload
    assert "if-no-files-found: error" not in workflow
