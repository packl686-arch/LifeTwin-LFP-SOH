# Private V3 freeze and schedule-aware V4 preregistration

Date: 2026-08-06
Author: Jincheng Liu

## Frozen development baseline

The current primary private-development candidate remains
`v3_dual_clock_kernel_shrinkage`. Numeric private-development metrics, source
rows, score tables, and model capsules are retained only in ignored private
artifact storage and are deliberately omitted from this public freeze record.

These results are outcome-exposed private method development. They are not
Hithium validation, independent confirmation, field-station validation, or a
15-25 year accuracy result. Private measurements and result artifacts remain
outside the public repository.

## Negative results retained

- The nested local-risk V3.1 selector did not replace V3.
- Worst-condition penalties 0.25, 0.5, and 1.0 did not improve the worst
  condition without sacrificing average performance.
- Further gate tuning on the same exposed cohort is prohibited as a route to
  promoting the primary model.

## Interval and abstention audit

- Pooled intervals did not meet the target at every landmark.
- Condition-balanced conservative intervals improved coverage at the cost of
  materially wider intervals.
- Prefix-quality abstention remains diagnostic and is not production-qualified.

## V4 hypothesis

V3 extrapolates future elapsed time using the prefix-average EFC/day duty.
V4 accepts an outcome-free operating plan and preserves the fitted prefix
posterior. For each future segment it interpolates a condition prior from
planned temperature, DOD, discharge rate, and segment EFC/day, then applies the
condition-prior difference to the segment's calendar/cycle basis increment.

When the declared plan equals the prefix condition and duty, V4 must reproduce
V3 within numerical tolerance. `planned_charge_c_rate` is sealed for provenance
but is not used by the current model and must not be described as an active
feature.

## Promotion rule

The machine-readable gates are frozen in
`configs/experiments/private_enterprise_schedule_v4_preregistered.json`.
Only a schedule declared no later than its prediction landmark is eligible for
primary evidence. A realized future schedule is an oracle upper bound and
cannot promote V4. Failure of any noninferiority, worst-condition, issue-rate,
or coverage gate leaves V3 as the primary candidate.

## Freeze limitations

The repository currently contains no Hithium measurements. The Git worktree
also contains a large staged research update that has not yet been committed,
so this document records scientific decisions but is not a Git tag. A commit
and tag should be made only after the complete regression and public-release
checks pass and the project owner approves publication.
