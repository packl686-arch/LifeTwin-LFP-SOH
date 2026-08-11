# V7 reissue-aware innovation evidence

This directory contains compact, derived evidence from the 41-cell MATR
FastCharge training cohort. It does not contain raw cycling measurements or the
81 exposed public-evaluation outcomes.

## What changed

V6 projected the residual slope observed since the previous landmark. V7 first
subtracts the slope of the forecast change that the newly issued V5 trajectory
has already absorbed:

```text
unassimilated_slope = previous_residual_slope - V5_reissue_shift_slope
```

The correction is bounded and applied only when the historical and remaining
slopes are directionally stable. An ineligible cell falls back exactly to the
frozen V5 center.

## Files

- `reissue_innovation_decision.json`: machine-readable decision and claim limits.
- `gate_candidate_summary.csv`: fit-cohort summaries for the 19 frozen gates.
- `nested_leave_one_cell_out_gate_scores.csv`: outer held-out-cell audit.
- `leave_one_batch_out_gate_scores.csv`: two-direction MATR batch-shift stress test.

## Result boundary

Only the P60-to-P100 transition passed both the leave-one-cell-out nomination
gate and the leave-one-batch-out stress gate. In the cell audit, 9 of 41 cells
were activated and all 9 improved; all-cell P100 trajectory MAE changed from
`0.2435991 pp` to `0.2062829 pp`. This is outcome-informed development on the
same 41 training cells, not independent confirmation. V5 remains active and the
V7 P100 rule is frozen only for a future outcome-blind test.

On those same 9 activated cells and at the same projection scale, the naive V6
history-slope correction improved 8/9 and had a worst delta of `+0.00391 pp`;
subtracting the reissue shift improved 9/9 and changed the worst delta to
`-0.02410 pp`. This matched ablation supports the mechanism but does not replace
the required blind test.
