# V9 end-to-end correlated-stability software dry run

This directory contains deterministic synthetic software evidence for the V9
qualification protocol. It is not a battery-accuracy result.

- `decision.json` records the frozen-source hashes, complete-chain stability
  metrics, exact fallback negative control, runtime, and claim boundary.
- `draw_metrics.csv` contains 24 outcome-free refit/reselection diagnostics.
- `stable_correction.csv` is the synthetic qualified correction over cycles
  101-300.

For each draw the runner perturbed all completed synthetic reference histories
and the target prefix with IID error, common bias, AR(1) error, linear drift,
and rare spikes. It then rebuilt the V5 pairwise matrices, refit the frozen
48-tree model at P60 and P100, reselected 12 neighbours, regenerated both
centres, and reran the unchanged V7 gate. The target fixture ends at cycle 100;
no target suffix was generated or read.

The low-noise software path retained 100% activation/sign stability. Its P95
final issued-trajectory deviation was 0.01261 pp and P05 reference-set Jaccard
was 0.84615. A labelled software stress control crossed the trajectory and
neighbour-stability limits and returned an exact zero correction. These values
only prove that both qualification and fail-closed paths execute as registered.
They do not estimate real tester noise, improve model accuracy, validate a
Hithium cell, or change the V5 champion.
