# Private schedule V4.1 amendment

Date: 2026-08-06
Author: Jincheng Liu

## Decision

The primary private-data model remains the frozen V3 dual-clock model. The
full schedule-aware V4 condition-delta algorithm is retained as a negative
control and is not promoted. Numeric development diagnostics, private score
tables, source rows, and model capsules remain in ignored private artifact
storage and are not reproduced here.

## Revised candidate

V4.1, `v4_1_explicit_elapsed_dual_clock`, uses the declared future elapsed-day
and EFC coordinates in the existing dual-clock predictor. Planned temperature,
SOC bounds, discharge rate, and segment EFC/day are sealed and checked for
training-support violations, but they do not alter the predicted capacity
curve. Planned charge rate remains provenance only.

This isolates a lower-assumption component from the unsupported direct
condition-prior correction. It is an amendment informed by outcome-exposed
development diagnostics, not independent confirmation.

## Frozen evaluation

The machine-readable amendment is
`configs/experiments/private_enterprise_schedule_v4_1_amendment.json`. V4.1
must be compared with V3 on the complete batch-disjoint Hithium calibration
population and pass every frozen error, condition-coverage, issuance, and
interval-coverage gate. Missing difficult cells or conditions is a failure.

Only a schedule declared at or before the prediction landmark is eligible.
A realized future schedule is an oracle upper bound and cannot promote the
candidate. If V4.1 fails any calibration gate, V3 remains primary without
same-cohort threshold retuning. The locked test is opened once, only after the
calibration decision is frozen.

## Claim boundary

No Hithium measurement has been accessed in this repository. This amendment
does not establish Hithium accuracy, field-station validity, or 15-25 year
forecast accuracy. Those claims require the separately controlled enterprise
calibration and locked-test workflow.
