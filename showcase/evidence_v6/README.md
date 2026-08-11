# LifeTwin V6 training-only challenger evidence

This directory publishes compact derived evidence for the V6 bounded residual
state and V6.1 abstaining-gate development study.

## Evidence role

- Outcome-informed method development on 41 V5 cross-fit training cells.
- The 81 exposed public evaluation cells were not used.
- Nested leave-one-physical-cell-out audit, not independent confirmation.
- V5 remains the active champion; V6.1 is not activated.
- No Hithium, calendar-aging, 15-25 year, or production claim.

## Files

- `ungated_state_decision.json`: V6 promotion failure and per-transition metrics.
- `ungated_candidate_summary.csv`: all fixed bounded-state candidates.
- `ungated_nested_selector_scores.csv`: held-out-cell selector outcomes.
- `gated_state_decision.json`: V6.1 nomination decision.
- `gated_candidate_summary.csv`: activation candidate screen.
- `gated_nested_selector_scores.csv`: outer held-out-cell activation outcomes.

The V6.1 P100 gate activated on 10 of 41 held-out cells, improved 8 of those
10, and reduced all-cell P100 mean trajectory MAE by 0.01963 pp. Because the
gate was developed after inspecting the same 41-cell training cohort, it is
frozen only as a candidate for a new outcome-blind batch.

See `reports/fastcharge_v6_bounded_state_development_2026-08-10.md` for the
full interpretation and
`configs/experiments/v6_1_p100_gated_state_blind_candidate.json` for the frozen
next-batch rule.
