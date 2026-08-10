"""Minimal V0.19 fit-stage wiring for result-blind contract development.

This is not a formal runner and defines no attempt, seed or protocol identity.
It preserves the inherited fit/commit lifecycle and changes only the final
whole-bundle validator called before the fit stage can complete.
"""

from __future__ import annotations

from lifetwin.experiments import calendar_long_horizon_v018_runner as _v018
from lifetwin.experiments.calendar_long_horizon_v018_contract import V023ContractView
from lifetwin.experiments.calendar_long_horizon_v018_ledger import (
    FormalAttemptIdentity,
)
from lifetwin.experiments.calendar_long_horizon_v018_partition import (
    WholeBundleValidated,
)
from lifetwin.experiments.calendar_long_horizon_v018_runner import V023RunPaths
from lifetwin.experiments.calendar_long_horizon_v019_partition import (
    validate_whole_bundle_from_root,
)


def _fit_structure_stage(
    *,
    paths: V023RunPaths,
    identity: FormalAttemptIdentity,
    view: V023ContractView,
    truth_hash: str,
) -> tuple[WholeBundleValidated, str]:
    """Run the inherited fit-once stage, then apply the V0.19 whole contract."""

    phase = "label_free_fit_committed"
    _v018._append_phase(
        paths=paths,
        identity=identity,
        contract=view.artifacts,
        phase=phase,
        exit_status="started",
        truth_hash=truth_hash,
        prediction_hash=None,
        message="Six-worker fit-once stage started from verified label-free bytes.",
    )
    try:
        bundle = _v018.load_fresh_generation_bundle_v023(
            label_free_root=paths.label_free_root,
            attempt_id=identity.attempt_id,
            contract_view=view,
        )
        fitted = _v018.fit_verified_generation_bundle_v023(bundle)
        digest = _v018.commit_verified_fit_result_v023(
            bundle=bundle,
            fit_result=fitted,
            created_utc=_v018._utc_now(),
        )
        del fitted, bundle
        whole = validate_whole_bundle_from_root(
            paths.label_free_root,
            view.artifacts,
        )
        _v018._append_phase(
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            exit_status="completed",
            truth_hash=truth_hash,
            prediction_hash=None,
            message=_v018.phase_commitment_message(phase, digest),
        )
        progress = _v018.validate_formal_exposure_log(
            paths.ledger_path,
            view.artifacts,
        )[identity.attempt_id]
        _v018.verify_phase_artifact_commitment(
            progress,
            phase=phase,
            artifact_path=paths.label_free_root / "fit_commitment.json",
        )
    except BaseException as exc:
        _v018._append_failure(
            error=exc,
            paths=paths,
            identity=identity,
            contract=view.artifacts,
            phase=phase,
            truth_hash=truth_hash,
            prediction_hash=None,
        )
        raise
    return whole, digest


__all__ = ["_fit_structure_stage"]
