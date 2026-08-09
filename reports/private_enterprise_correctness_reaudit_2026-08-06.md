# Private enterprise correctness re-audit

Date: 2026-08-06
Author: Jincheng Liu

## Scope

This audit covers the private enterprise data adapter, prediction/scoring
firewall, dual-clock time coordinates, schedule sealing, metric computation,
promotion gates, and public-release boundary. It contains no Hithium
measurement or private model score.

## Findings and actions

1. The prediction API accepts development trajectories and target prefixes but
   has no truth-vault or future-capacity argument. Scoring is a separate API and
   validates the sealed prediction manifest before linking truth.
2. Metadata-only partitioning keeps each batch in exactly one partition. The
   blind bundle separates calibration and locked-test prefixes from their truth
   vaults and binds every member by canonical hashes.
3. Forecast EFC and elapsed-day coordinates must be finite, strictly beyond the
   prefix, and monotone. A declared schedule must exactly cover every sealed
   cell, landmark, and frozen EFC grid point.
4. V4.1 uses only forecast elapsed days and EFC for numeric prediction. Planned
   temperature, SOC bounds, discharge rate, and segment duty are support-domain
   diagnostics. Planned charge rate is provenance only.
5. The trajectory IAE implementation was independently checked against a hand
   trapezoid calculation that includes zero error at the prediction landmark.
6. The promotion gate was hardened during this audit. It now verifies score
   summary hashes, score-row hashes, experiment/adapter/dataset identities,
   unique cell-landmark keys, finite metrics, metric bounds, and exact candidate
   mode binding before evaluating performance thresholds.
7. Selective abstention cannot improve the gate: missing conditions or cells,
   duplicated keys, modified score rows, oracle schedules, and re-labeled model
   modes all fail closed.
8. The public release verifier scans Git-tracked files and continues to exclude
   raw private archives, private scores, model capsules, and ignored artifacts.

## Executed verification

- `tests/test_private_cycle_adapter.py`: batch partition and truth-vault split.
- `tests/test_private_enterprise_cycle.py`: suffix independence, schedule
  perturbation, OOD refusal, tamper rejection, CLI sealing, and score replay.
- `tests/test_private_schedule_v4_gates.py`: population completeness, mode
  binding, score/summary hash checks, duplicate-key rejection, and oracle ban.
- `tests/test_private_enterprise_correctness_reaudit.py`: API boundary, metric
  hand calculation, and V4.1 amendment identity.
- `tests/test_dataset_evidence_matrix.py`: evidence-role and rights firewall.

## Residual risks

- Deterministic file splitting is not an operating-system access-control
  boundary. A real blind run still needs separate credentials or processes for
  the truth vault.
- Diagnostic intervals are not a formal conditional-coverage guarantee.
- Synthetic and outcome-exposed public data can test software behavior but
  cannot validate Hithium accuracy.
- No 15-25 year claim is permitted without a separately controlled long-duration
  cohort and one once-opened locked test.

## Decision

The software firewall is suitable for an enterprise dry run. V3 remains the
primary model. V4.1 is the default schedule-aware challenger; the separately
preregistered, explicitly selected V4.2 is the final bounded challenger. Full
V4 remains a negative control. The next evidence-raising action is new,
rights-cleared, outcome-unexposed data rather than more same-cohort tuning.
