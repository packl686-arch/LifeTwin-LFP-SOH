# V11 Delta-Q multimodal challenger evidence

This directory contains aggregate, training-only development evidence for two
fixed P100 challengers that add cycle-10-to-100 `Delta Q(V)` features to the
frozen V5 pairwise model. It contains no raw voltage curves, no cell-level
scores, and no target suffix data.

The experiment used only the same 41 public MATR training cells. Every held-out
physical cell was excluded from both pair roles and from robust feature scaling.
The 81 exposed evaluation cells were not used. A second stress test held out
MATR manufacturing batches in both directions.

| Candidate | Mean MAE | Relative change vs V5 | Improved cells | Bootstrap delta interval | Result |
|---|---:|---:|---:|---:|---|
| Delta-Q residual only | 0.24385 pp | -0.10% | 46.34% | [-0.01094, 0.00915] pp | failed |
| Delta-Q residual + geometry | 0.24183 pp | +0.73% | 46.34% | [-0.01845, 0.01062] pp | failed |

The better challenger still missed the frozen 5% mean-improvement gate, its
bootstrap interval crossed zero, its P90 cell regression exceeded the limit,
and one batch-holdout direction regressed by 0.00659 pp. No challenger advances
to a blind queue and V5 remains the champion.

`decision.json` is the machine-readable aggregate decision. This is
outcome-exposed method development, not independent confirmation, calendar
aging validation, Hithium product evidence, or a 15-to-25-year accuracy claim.
