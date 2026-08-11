# Private schedule V4.2 preregistration

Date: 2026-08-06
Author: Jincheng Liu

## Motivation

Full V4 directly applied the complete condition-prior coefficient difference
to every future segment and remains a negative control. V4.1 removed that
correction and retained only explicit future elapsed-day and EFC coordinates.
V4.2 tests one final, deliberately bounded hypothesis: a small condition delta
may help when the planned condition is well supported, but it must never be
allowed to dominate the training-residual uncertainty scale.

## Frozen algorithm

The candidate ID is `v4_2_support_gated_bounded_delta`.

1. Generate the V4.1 explicit-time prediction.
2. Generate the frozen full-V4 condition-delta prediction without retuning it.
3. Compute schedule support as
   `max(0, min(1, 1 - distance / training_threshold))`.
4. Multiply the raw delta by at most `0.25 * support`.
5. Clip the applied delta to at most 25% of the V3 training-inner-LOCO
   diagnostic half-width at that horizon.
6. Preserve the V4.1 interval width and shift its center and bounds together.
7. Reject schedules outside the frozen support threshold.

Planned charge rate is not a model feature. The maximum weight, residual bound,
support function, and promotion gates may not be changed after Hithium
calibration outcomes are opened.

## Evaluation order

V3, V4.1, and V4.2 must be predicted and sealed separately on the complete
batch-disjoint calibration population. Each schedule candidate is compared
with V3 using the same noninferiority, improvement, worst-condition, issuance,
and interval-coverage gates. Missing cells or conditions fail the gate.

If both candidates pass, the lower condition-equal trajectory IAE wins. A
difference no larger than 0.02 percentage points selects simpler V4.1. At most
one model enters the once-opened locked test. Failure retains V3 without
same-cohort retuning.

## Evidence boundary

This implementation was specified after public development outcomes were
exposed and before any Hithium measurement was accessed. Unit tests and a
synthetic dry run establish only deterministic software behavior. They do not
show that V4.2 improves a real battery forecast, validates a Hithium product,
or supports a 15-25 year accuracy claim.
