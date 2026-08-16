"""One-shot, post-freeze generation for the V2.3 synthetic protocol.

The public lifecycle has two explicit capabilities.  The first performs and
commits the complete RNG-free V2.3/V2 coordinate audit.  The second verifies
that exact commitment before it can consume the first V2.3 seed.  Neither
entry point accepts a seed, protocol override, family, count, or reduced mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from lifetwin.experiments.calendar_long_horizon_v015_generation import (
    ArtifactMetadata,
    LABEL_FREE_CSV_FILENAMES,
    OrdinaryPackRecord,
    PreparedGenerationArtifacts,
    TRUTH_COMMITMENT_FILENAME,
    assemble_generated_artifact_frames,
    assert_generation_destinations_available,
    prepare_generated_artifacts,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    generate_cluster_packs,
    generate_intrinsic_matched_pair,
    generate_operating_covariates,
    generate_stress_plan_matched_pair,
    sample_truth_spec,
    validate_unique_stream_seeds,
)
from lifetwin.experiments.calendar_long_horizon_v018_collision import (
    GenerationPlanCommitment,
    V023CollisionError,
    audit_formal_v023_generation_plan,
    verify_generation_plan_commitment,
)
from lifetwin.experiments.calendar_long_horizon_v018_contract import (
    V023ContractView,
    load_v023_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v018_firewall import (
    AttemptProgress,
    FormalAttemptIdentity,
    V023FirewallError,
    append_formal_exposure_event,
    append_phase_error_without_masking,
    phase_commitment_message,
    validate_formal_exposure_log,
)
from lifetwin.experiments.calendar_long_horizon_v018_protocol import (
    V023_PROTOCOL_ID,
)


GENERATION_PLAN_COMMITMENT_FILENAME = "generation_plan_commitment.json"
_LEDGER_FILENAME = "exposure_log.jsonl"
_IMPLEMENTABLE_STATUS = "implementation_frozen"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class V023GenerationError(ValueError):
    """Raised when the V2.3 generation lifecycle cannot advance exactly."""


@dataclass(frozen=True)
class WrittenV023GenerationArtifacts:
    """Metadata for one committed V2.3 generation bundle."""

    generation_plan_commitment: GenerationPlanCommitment
    label_free_metadata: tuple[ArtifactMetadata, ...]
    sealed_metadata: tuple[ArtifactMetadata, ...]
    truth_commitment_byte_sha256: str


@dataclass(frozen=True, slots=True)
class _PhysicalRootIdentity:
    path: Path
    device: int
    inode: int


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _is_reparse_entry(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise V023GenerationError(f"Cannot inspect formal path: {path}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _physical_root_identity(
    raw_path: str | Path,
    *,
    context: str,
) -> _PhysicalRootIdentity:
    path = Path(os.path.abspath(os.fspath(raw_path)))
    existing_chain = tuple(reversed(path.parents)) + (path,)
    for component in existing_chain:
        if os.path.lexists(component) and _is_reparse_entry(component):
            raise V023GenerationError(
                f"{context} root has a reparse-point ancestor: {component}"
            )
    if not path.is_dir():
        raise V023GenerationError(
            f"{context} root must be a pre-created physical directory"
        )
    try:
        resolved = path.resolve(strict=True)
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise V023GenerationError(f"Cannot bind {context} root identity") from exc
    if resolved != path:
        raise V023GenerationError(f"{context} root changed during resolution")
    return _PhysicalRootIdentity(
        path=path,
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
    )


def _verify_physical_root_identity(
    identity: _PhysicalRootIdentity,
    *,
    context: str,
) -> None:
    observed = _physical_root_identity(identity.path, context=context)
    if observed != identity:
        raise V023GenerationError(f"{context} directory identity changed")


def _bind_generation_roots(
    *,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
) -> tuple[_PhysicalRootIdentity, _PhysicalRootIdentity]:
    label = _physical_root_identity(label_free_root, context="label-free")
    sealed = _physical_root_identity(sealed_truth_root, context="sealed-truth")
    try:
        common = Path(os.path.commonpath((label.path, sealed.path)))
    except ValueError:
        return label, sealed
    if common in {label.path, sealed.path}:
        raise V023GenerationError(
            "Label-free and sealed-truth roots must be disjoint trees"
        )
    return label, sealed


def _require_exact_direct_files(
    root: Path,
    *,
    allowed_sets: tuple[frozenset[str], ...],
    context: str,
) -> frozenset[str]:
    if not root.is_dir() or _is_reparse_entry(root):
        raise V023GenerationError(f"{context} root is not a physical directory")
    entries = tuple(root.iterdir())
    names = frozenset(entry.name for entry in entries)
    if names not in allowed_sets:
        expected = " or ".join(str(sorted(item)) for item in allowed_sets)
        raise V023GenerationError(
            f"{context} root membership is {sorted(names)}, expected {expected}"
        )
    for entry in entries:
        if _is_reparse_entry(entry) or not entry.is_file():
            raise V023GenerationError(
                f"{context} artifact is not a direct regular file: {entry}"
            )
    return names


def _exclusive_create_or_verify(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if _is_reparse_entry(path) or not path.is_file() or path.read_bytes() != raw:
            raise V023GenerationError(
                f"Existing formal commitment conflicts: {path}"
            ) from None
    except OSError as exc:
        raise V023GenerationError(
            f"Could not create formal commitment: {path}"
        ) from exc
    if path.read_bytes() != raw:
        raise V023GenerationError(
            f"Formal commitment bytes changed after creation: {path}"
        )


def _artifact_metadata(
    *,
    filename: str,
    raw: bytes,
    row_count: int,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        path=filename,
        row_count=row_count,
        byte_count=len(raw),
        byte_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _write_prepared_generation_artifacts_resumable(
    prepared: PreparedGenerationArtifacts,
    *,
    label_identity: _PhysicalRootIdentity,
    sealed_identity: _PhysicalRootIdentity,
    view: V023ContractView,
) -> tuple[tuple[ArtifactMetadata, ...], tuple[ArtifactMetadata, ...], str]:
    """Create missing files and require any interrupted files to be byte-identical."""

    label_names = (*LABEL_FREE_CSV_FILENAMES, TRUTH_COMMITMENT_FILENAME)
    sealed_names = view.artifacts.sealed_filenames
    if (
        tuple(prepared.label_free_bytes) != label_names
        or tuple(prepared.sealed_bytes) != sealed_names
    ):
        raise V023GenerationError("Prepared artifact membership or order changed")

    allowed_label = frozenset(
        {_LEDGER_FILENAME, GENERATION_PLAN_COMMITMENT_FILENAME, *label_names}
    )
    allowed_sealed = frozenset(sealed_names)
    for identity, allowed, context in (
        (label_identity, allowed_label, "label-free"),
        (sealed_identity, allowed_sealed, "sealed-truth"),
    ):
        _verify_physical_root_identity(identity, context=context)
        observed = frozenset(entry.name for entry in identity.path.iterdir())
        if not observed.issubset(allowed):
            raise V023GenerationError(
                f"{context} generation root contains unexpected artifacts"
            )

    for identity, names, content, context in (
        (
            sealed_identity,
            sealed_names,
            prepared.sealed_bytes,
            "sealed-truth",
        ),
        (
            label_identity,
            label_names,
            prepared.label_free_bytes,
            "label-free",
        ),
    ):
        for filename in names:
            _verify_physical_root_identity(identity, context=context)
            _exclusive_create_or_verify(identity.path / filename, content[filename])
        _verify_physical_root_identity(identity, context=context)

    sealed_metadata = tuple(
        _artifact_metadata(
            filename=filename,
            raw=prepared.sealed_bytes[filename],
            row_count=prepared.row_counts[filename],
        )
        for filename in sealed_names
    )
    label_metadata = tuple(
        _artifact_metadata(
            filename=filename,
            raw=prepared.label_free_bytes[filename],
            row_count=(
                prepared.row_counts[filename]
                if filename in LABEL_FREE_CSV_FILENAMES
                else 1
            ),
        )
        for filename in label_names
    )
    truth_hash = hashlib.sha256(
        prepared.label_free_bytes[TRUTH_COMMITMENT_FILENAME]
    ).hexdigest()
    return label_metadata, sealed_metadata, truth_hash


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V023GenerationError(
                "Generation-plan commitment contains duplicate JSON keys"
            )
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise V023GenerationError(
        f"Generation-plan commitment contains nonfinite token {token}"
    )


def _decode_canonical_plan_commitment(raw: bytes) -> Mapping[str, object]:
    if not isinstance(raw, bytes):
        raise TypeError("Generation-plan commitment must be bytes")
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V023GenerationError(
            "Generation-plan commitment is not strict ASCII JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise V023GenerationError("Generation-plan commitment must be an object")
    canonical = (
        json.dumps(
            decoded,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    if raw != canonical:
        raise V023GenerationError("Generation-plan commitment bytes are not canonical")
    return decoded


def _formal_environment() -> object:
    from lifetwin.experiments.calendar_long_horizon_v018_environment import (
        verify_formal_environment,
    )

    environment = verify_formal_environment(_PROJECT_ROOT)
    if (
        getattr(environment, "protocol_id", None) != V023_PROTOCOL_ID
        or getattr(environment, "git_dirty", None) is not False
    ):
        raise V023GenerationError("Formal V2.3 environment identity is invalid")
    return environment


def _validate_contract_view(view: V023ContractView) -> None:
    if type(view) is not V023ContractView:
        raise V023GenerationError("V2.3 contract view has an invalid type")
    if (
        view.protocol.protocol_id != V023_PROTOCOL_ID
        or view.artifacts.protocol_id != V023_PROTOCOL_ID
        or view.protocol.config_sha256 != view.artifacts.config_byte_sha256
    ):
        raise V023GenerationError("V2.3 protocol and artifact contracts disagree")
    if view.design_status != _IMPLEMENTABLE_STATUS:
        raise V023GenerationError("V2.3 implementation has not been immutably frozen")


def _validate_attempt_identities(
    states: Mapping[str, AttemptProgress],
    *,
    environment: object,
) -> None:
    git_commit = getattr(environment, "git_commit", None)
    config_hash = getattr(environment, "config_byte_sha256", None)
    if not states:
        raise V023GenerationError("Formal exposure ledger contains no attempt")
    for state in states.values():
        if (
            state.identity.git_commit != git_commit
            or state.identity.config_byte_sha256 != config_hash
        ):
            raise V023GenerationError(
                "A formal attempt used a different implementation identity"
            )


def _select_plan_commit_candidate(
    states: Mapping[str, AttemptProgress],
    *,
    environment: object,
) -> AttemptProgress:
    _validate_attempt_identities(states, environment=environment)
    candidates = [
        state
        for state in states.values()
        if (
            state.completed_phase == "before_generation"
            and state.pending_phase in {None, "generation_plan_committed"}
            and not state.terminal_failed
            and state.truth_commitments_byte_sha256 is None
            and state.prediction_commitment_byte_sha256 is None
            and not state.opened_truth_files
        )
    ]
    if len(candidates) != 1:
        raise V023GenerationError(
            "Exactly one attempt may commit the pre-generation coordinate plan"
        )
    if any(
        state.identity.attempt_id != candidates[0].identity.attempt_id
        and not state.terminal_failed
        and state.completed_phase != "scoring_completed"
        for state in states.values()
    ):
        raise V023GenerationError(
            "Another formal attempt remains unfinished in this ledger"
        )
    return candidates[0]


def _select_generation_candidate(
    states: Mapping[str, AttemptProgress],
    *,
    environment: object,
) -> AttemptProgress:
    _validate_attempt_identities(states, environment=environment)
    candidates = [
        state
        for state in states.values()
        if (
            state.completed_phase == "generation_plan_committed"
            and state.pending_phase in {None, "truth_committed"}
            and not state.terminal_failed
            and state.generation_plan_commitment_byte_sha256 is not None
            and state.truth_commitments_byte_sha256 is None
            and state.prediction_commitment_byte_sha256 is None
            and not state.opened_truth_files
        )
    ]
    if len(candidates) != 1:
        raise V023GenerationError(
            "Exactly one plan-committed attempt may begin V2.3 generation"
        )
    if any(
        state.identity.attempt_id != candidates[0].identity.attempt_id
        and not state.terminal_failed
        and state.completed_phase != "scoring_completed"
        for state in states.values()
    ):
        raise V023GenerationError(
            "Another formal attempt remains unfinished in this ledger"
        )
    return candidates[0]


def _identity(progress: AttemptProgress) -> FormalAttemptIdentity:
    return FormalAttemptIdentity(
        attempt_id=progress.identity.attempt_id,
        git_commit=progress.identity.git_commit,
        config_byte_sha256=progress.identity.config_byte_sha256,
    )


def _append_failure(
    *,
    error: BaseException,
    ledger_path: Path,
    identity: FormalAttemptIdentity,
    view: V023ContractView,
    phase: str,
    message: str,
) -> None:
    append_phase_error_without_masking(
        error=error,
        ledger_path=ledger_path,
        identity=identity,
        contract=view.artifacts,
        created_utc=_utc_now(),
        phase=phase,
        exit_status=(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        ),
        truth_commitments_byte_sha256=None,
        prediction_commitment_byte_sha256=None,
        message=message,
    )


def commit_frozen_v023_generation_plan(
    *,
    label_free_root: str | Path,
) -> str:
    """Audit and byte-commit the full coordinate plan without consuming a seed."""

    environment = _formal_environment()
    view = load_v023_contract_view()
    _validate_contract_view(view)
    if getattr(environment, "config_byte_sha256", None) != (
        view.artifacts.config_byte_sha256
    ):
        raise V023GenerationError(
            "Formal environment and V2.3 amendment commitments disagree"
        )

    root = _physical_root_identity(
        label_free_root,
        context="label-free",
    ).path
    _require_exact_direct_files(
        root,
        allowed_sets=(
            frozenset({_LEDGER_FILENAME}),
            frozenset({_LEDGER_FILENAME, GENERATION_PLAN_COMMITMENT_FILENAME}),
        ),
        context="pre-generation label-free",
    )
    ledger_path = root / _LEDGER_FILENAME
    try:
        states = validate_formal_exposure_log(ledger_path, view.artifacts)
        progress = _select_plan_commit_candidate(states, environment=environment)
    except (OSError, V023FirewallError) as exc:
        raise V023GenerationError(
            "Formal exposure ledger failed before the coordinate audit"
        ) from exc
    identity = _identity(progress)
    append_formal_exposure_event(
        path=ledger_path,
        identity=identity,
        contract=view.artifacts,
        created_utc=_utc_now(),
        phase="generation_plan_committed",
        exit_status="started",
        truth_commitments_byte_sha256=None,
        prediction_commitment_byte_sha256=None,
        message="Started the exhaustive RNG-free V2.3/V2 coordinate audit.",
    )
    try:
        commitment = audit_formal_v023_generation_plan(view)
        target = root / GENERATION_PLAN_COMMITMENT_FILENAME
        _exclusive_create_or_verify(target, commitment.canonical_bytes)
        payload = _decode_canonical_plan_commitment(target.read_bytes())
        verify_generation_plan_commitment(
            payload,
            expected_byte_sha256=commitment.byte_sha256,
            expected_current_protocol_id=V023_PROTOCOL_ID,
            expected_current_protocol_byte_sha256=(view.artifacts.config_byte_sha256),
            expected_predecessor_protocol_id="synthetic_long_horizon_identifiability_v2",
            expected_predecessor_protocol_byte_sha256=(view.base_config_byte_sha256),
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            ledger_path=ledger_path,
            identity=identity,
            view=view,
            phase="generation_plan_committed",
            message="The pre-generation coordinate audit did not complete.",
        )
        raise
    append_formal_exposure_event(
        path=ledger_path,
        identity=identity,
        contract=view.artifacts,
        created_utc=_utc_now(),
        phase="generation_plan_committed",
        exit_status="completed",
        truth_commitments_byte_sha256=None,
        prediction_commitment_byte_sha256=None,
        message=phase_commitment_message(
            "generation_plan_committed", commitment.byte_sha256
        ),
    )
    return commitment.byte_sha256


def _verify_committed_generation_plan(
    *,
    root: Path,
    view: V023ContractView,
    expected_byte_sha256: str,
) -> GenerationPlanCommitment:
    target = root / GENERATION_PLAN_COMMITMENT_FILENAME
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise V023GenerationError(
            "Committed generation plan is absent or unreadable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != expected_byte_sha256:
        raise V023GenerationError(
            "Generation-plan bytes differ from the exposure ledger"
        )
    payload = _decode_canonical_plan_commitment(raw)
    try:
        verify_generation_plan_commitment(
            payload,
            expected_byte_sha256=expected_byte_sha256,
            expected_current_protocol_id=V023_PROTOCOL_ID,
            expected_current_protocol_byte_sha256=(view.artifacts.config_byte_sha256),
            expected_predecessor_protocol_id="synthetic_long_horizon_identifiability_v2",
            expected_predecessor_protocol_byte_sha256=(view.base_config_byte_sha256),
        )
        recomputed = audit_formal_v023_generation_plan(view)
    except V023CollisionError as exc:
        raise V023GenerationError(
            "Generation-plan collision commitment failed verification"
        ) from exc
    if (
        recomputed.byte_sha256 != expected_byte_sha256
        or recomputed.canonical_bytes != raw
        or recomputed.payload != dict(payload)
    ):
        raise V023GenerationError(
            "Generation plan differs from its independent full recomputation"
        )
    return recomputed


def generate_frozen_v023_artifacts(
    *,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
) -> WrittenV023GenerationArtifacts:
    """Generate fresh V2.3 rows only after the committed collision audit."""

    environment = _formal_environment()
    view = load_v023_contract_view()
    _validate_contract_view(view)
    if getattr(environment, "config_byte_sha256", None) != (
        view.artifacts.config_byte_sha256
    ):
        raise V023GenerationError(
            "Formal environment and V2.3 amendment commitments disagree"
        )
    label_identity, sealed_identity = _bind_generation_roots(
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
    )
    label_root = label_identity.path
    _require_exact_direct_files(
        label_root,
        allowed_sets=(
            frozenset({_LEDGER_FILENAME, GENERATION_PLAN_COMMITMENT_FILENAME}),
        ),
        context="plan-committed label-free",
    )
    ledger_path = label_root / _LEDGER_FILENAME
    try:
        states = validate_formal_exposure_log(ledger_path, view.artifacts)
        progress = _select_generation_candidate(states, environment=environment)
    except (OSError, V023FirewallError) as exc:
        raise V023GenerationError(
            "Formal exposure ledger failed before V2.3 generation"
        ) from exc
    identity = _identity(progress)
    expected_plan_hash = progress.generation_plan_commitment_byte_sha256
    assert expected_plan_hash is not None

    try:
        generation_plan = _verify_committed_generation_plan(
            root=label_root,
            view=view,
            expected_byte_sha256=expected_plan_hash,
        )
        if progress.pending_phase is None:
            assert_generation_destinations_available(
                label_free_root=label_root,
                sealed_truth_root=sealed_identity.path,
                contract=view.artifacts,
            )
        if progress.pending_phase is None:
            append_formal_exposure_event(
                path=ledger_path,
                identity=identity,
                contract=view.artifacts,
                created_utc=_utc_now(),
                phase="truth_committed",
                exit_status="started",
                truth_commitments_byte_sha256=None,
                prediction_commitment_byte_sha256=None,
                message="Verified the coordinate commitment before first seed use.",
            )

        protocol = view.protocol
        ordinary: list[OrdinaryPackRecord] = []
        truth_specs = []
        for partition, family_counts in protocol.partition_family_counts:
            for family_id, count in family_counts:
                for index in range(count):
                    operating = generate_operating_covariates(
                        protocol,
                        partition=partition,
                        family_id=family_id,
                        zero_based_index=index,
                    )
                    truth_spec, fixed_operating = sample_truth_spec(
                        protocol,
                        partition=partition,
                        family_id=family_id,
                        zero_based_index=index,
                        operating=operating,
                    )
                    generated = generate_cluster_packs(
                        protocol, truth_spec, fixed_operating
                    )
                    ordinary.append(
                        OrdinaryPackRecord(
                            partition=partition,
                            family_id=family_id,
                            zero_based_index=index,
                            packs=generated.packs,
                        )
                    )
                    truth_specs.append(truth_spec)
        validate_unique_stream_seeds(truth_specs)
        intrinsic = tuple(
            generate_intrinsic_matched_pair(protocol, zero_based_pair_index=pair_index)
            for pair_index in range(250)
        )
        stress = tuple(
            generate_stress_plan_matched_pair(
                protocol, zero_based_pair_index=pair_index
            )
            for pair_index in range(250)
        )
        frames = assemble_generated_artifact_frames(
            ordinary_records=ordinary,
            intrinsic_pairs=intrinsic,
            stress_plan_pairs=stress,
            protocol=protocol,
            formal=True,
        )
        prepared = prepare_generated_artifacts(
            frames,
            protocol=protocol,
            contract=view.artifacts,
            created_utc=_utc_now(),
            formal=True,
        )
        _verify_physical_root_identity(
            label_identity,
            context="label-free",
        )
        _verify_physical_root_identity(
            sealed_identity,
            context="sealed-truth",
        )
        label_metadata, sealed_metadata, truth_hash = (
            _write_prepared_generation_artifacts_resumable(
                prepared,
                label_identity=label_identity,
                sealed_identity=sealed_identity,
                view=view,
            )
        )
    except BaseException as exc:
        _append_failure(
            error=exc,
            ledger_path=ledger_path,
            identity=identity,
            view=view,
            phase="truth_committed",
            message="V2.3 generation did not reach its truth commitment.",
        )
        raise

    append_formal_exposure_event(
        path=ledger_path,
        identity=identity,
        contract=view.artifacts,
        created_utc=_utc_now(),
        phase="truth_committed",
        exit_status="completed",
        truth_commitments_byte_sha256=truth_hash,
        prediction_commitment_byte_sha256=None,
        message="All nine fresh V2.3 sealed files were committed before truth access.",
    )
    return WrittenV023GenerationArtifacts(
        generation_plan_commitment=generation_plan,
        label_free_metadata=label_metadata,
        sealed_metadata=sealed_metadata,
        truth_commitment_byte_sha256=truth_hash,
    )


__all__ = [
    "GENERATION_PLAN_COMMITMENT_FILENAME",
    "V023GenerationError",
    "WrittenV023GenerationArtifacts",
    "commit_frozen_v023_generation_plan",
    "generate_frozen_v023_artifacts",
]
