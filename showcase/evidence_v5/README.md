# LifeTwin V5 public development evidence

This directory publishes compact, derived evidence for the outcome-exposed MATR
FastCharge development study. It intentionally excludes raw upstream measurements
and large prediction tables.

## Evidence role

- Public, retrospective, outcome-exposed method development.
- Not independent confirmation.
- Not Hithium product validation.
- Not calendar-aging or 15-25 year accuracy evidence.
- No formal cross-domain interval-coverage guarantee.

## Pairwise point model

- `pairwise_training_selection.json`: training-only five-fold selection record.
- `pairwise_training_cv_summary.csv`: all screened candidate summaries.
- `pairwise_firewall_audit.json`: physical-cell target/reference exclusion audit.
- `pairwise_prediction_manifest.json`: suffix-blind prediction commitment and hashes.
- `pairwise_evaluation_summary.json`: aggregate result and paired-cell bootstrap.
- `pairwise_cell_prefix_scores.csv`: one score row per evaluation cell and prefix.

## Support gate and uncertainty

- `support_uncertainty_development.json`: training-only gate and interval choice.
- `support_gate_screen.csv`: every screened gate, including negative choices.
- `interval_method_screen.csv`: cross-conformal interval comparison.
- `calibration_quantiles.csv`: frozen per-prefix/per-horizon calibration values.
- `support_uncertainty_prediction_manifest.json`: prediction commitment and hashes.
- `support_uncertainty_score_summary.json`: aggregate coverage, width, WIS, and gate status.
- `support_uncertainty_cell_prefix_scores.csv`: cell-prefix interval results.

## Dynamic landmark and online residual audit

- `dynamic_landmark_decision.json`: frozen selection, reissue results, H2 status,
  and claim boundary.
- `dynamic_landmark_training_candidate_summary.csv`: all 20 training-only
  residual candidates and activation gates.
- `dynamic_landmark_training_nested_selector_audit.csv`: secondary
  leave-one-physical-cell-out selector audit.
- `dynamic_landmark_training_base_reissue_scores.csv`: previous-versus-current
  prefix reissue scores on the cross-fit training cells.
- `dynamic_landmark_evaluation_base_reissue_scores.csv`: descriptive reissue
  scores on the 81 outcome-exposed evaluation cells.
- `dynamic_landmark_evaluation_selected_update_scores.csv`: only the
  training-selected residual rules evaluated on public cells; no evaluation
  candidate screen is published or permitted.

The dynamic audit found a useful mean reissue signal but no qualifying GP
online residual branch. The frozen V5 center remains active and full H2 remains
failed.

The technical interpretations are in
`reports/fastcharge_v5_pairwise_development_2026-08-09.md` and
`reports/fastcharge_v5_dynamic_landmark_audit_2026-08-09.md`.
