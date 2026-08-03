# NASA evidence-weighted mixture V3 development report

Date: 2026-08-03

Author: Jincheng Liu

## Evidence boundary

V3 was frozen after NASA V1/V2 suffix outcomes and an earlier 45-cell Attia
cycle-life result had already been inspected. It is therefore a retrospective
method-development result, not outcome-blind confirmation. NASA contains four
public accelerated cycle-aging cells and is neither long-duration LFP calendar
aging nor stationary-storage field evidence. No Hithium data were used.

The frozen V3 semantic configuration SHA-256 is
`77b2e0465c04cc9f97499d55c12cdf3cbe177b6e4b423fff11ef5da503f9872a`.
The V2 prediction and score commitments were recorded in the V3 exposure
registry before this run.

## Frozen method

For each held-out cell and prefix, V3 retains the four target-prefix experts
from V2. It estimates each expert's risk using the two curve-nearest outer
training cells, inverse-distance weighting, and a dispersion penalty. Inverse
risk weights are then shrunk continuously toward equal weights when either the
risk margin or similarity support is weak. The held-out cell's future suffix is
never available to prediction.

The evidence band combines the V2 expert bands, inter-model disagreement, and a
low-evidence widening factor. A mean neighbor distance of at least six produces
`refuse_recommended`; a numeric trajectory remains in the research artifact so
the refusal policy can be audited independently.

## Primary result

Values are equal-cell mean suffix trajectory MAE in percentage points through
cycle 132. The development gate was written into the frozen configuration
before this V3 run.

| Method | P20 | P40 | P60 | P100 | Mean over 16 folds |
|---|---:|---:|---:|---:|---:|
| V3 evidence-weighted mixture | **4.311** | **5.375** | **2.307** | 2.005 | **3.499** |
| V2 curve-aware gate | 4.771 | 6.960 | 2.651 | 2.499 | 4.220 |
| Equal-weight expert mixture | 4.381 | **5.375** | 4.176 | **1.909** | 3.960 |
| Pure inverse-risk mixture | 5.097 | 6.020 | 2.770 | 1.965 | 3.963 |
| Hard lowest-risk expert | 5.735 | 6.960 | 2.651 | 2.954 | 4.575 |
| Hindsight-best single base expert | 3.646 | 2.966 | **2.006** | 1.054 | 2.418 |

V3 reduced mean MAE by 17.1% relative to the frozen V2 curve-aware gate and
improved all four prefixes. It reduced mean MAE by 11.6% relative to the simple
equal-weight mixture. Both preregistered NASA development checks passed: overall
MAE was below the V2 threshold and no prefix degraded by more than 0.5 pp.

The prediction SHA-256 is
`5bd96924e239b774f3e1f7ee3b985a5880b74cca2422b192353be12055703cce`;
the score-table SHA-256 is
`f98d7924740a1d7d39dd4f6a43cf5a8bf5c1231898ee1dcedbeb3eaf6cbd061d`.

## What the ablations show

The result supports conservative blending, not confident expert identification.
The nominally dominant expert was the hindsight-best base expert in only 4 of
16 folds. Pure inverse-risk weighting and hard expert selection were both worse
than equal weighting. V3's advantage therefore comes from allowing supported
weight shifts while shrinking ambiguous or distant cases toward an ensemble.

The benefit is not uniform. At P40 every fold fell back to equal weights; at
P100 the equal mixture was slightly better than V3. Most incremental value came
at P60. The hindsight row selects the realized best single base expert in each
fold and is not deployable. Its 1.08 pp aggregate difference from V3 describes
remaining single-expert selection opportunity, but it is not a mathematical
lower bound because a convex mixture can outperform every individual expert.

## Evidence and refusal audit

The V3 bands covered 96.5% of observed suffix points but had a mean width of
38.17 pp. This is too broad for a useful calibrated-coverage claim. Across only
16 reused folds, the Spearman correlations of neighbor distance, selection
strength, and band width with V3 MAE were -0.421, -0.116, and 0.265. These are
descriptive and do not establish error ranking.

One real fold, B0006 at P60, crossed the covariate-distance refusal threshold.
Its MAE was 1.759 pp, below the all-fold median of 2.484 pp. Refusal must
therefore be described only as a distribution-support warning, not as a
validated detector of high forecast error.

## Adversarial checks

- All 16 held-out suffix mutation attacks produced byte-equivalent predictions
  for the attacked fold, confirming the future-label firewall.
- Missing required curve data failed closed.
- All 16 synthetic severe curve-shift cases triggered `refuse_recommended`.
- Under a small endpoint-linear sensor drift (0.2% capacity, 0.5% voltage-window
  duration, 2 mV voltage, and 0.1 C temperature), 15 of 16 folds stayed below a
  post-outcome 2 pp maximum-change diagnostic. B0018 at P20 changed by as much as
  7.009 pp and its action changed from warning to predict. This is a negative
  robustness result, not a passed gate.

The post-outcome ablation and evidence hashes are
`343507d4e9db6dd65c691a31dfde80eb7370f261c0a17c528608b5953052d512`
and `c85e885d1d2316dc4232b51e23e0c587561f522ad317675fc274445e5a78d131`.
The attack-table hash is
`7b66cfecf83e5635c2f5f64ad3cfe27ca40daf313579ee0ff7e97a6848529718`.

## Decision

Retain V3 as the best NASA development candidate and retain equal weighting as
the mandatory low-evidence fallback. Do not promote the current interval or
refusal policy as calibrated. A later method version should address early-prefix
weight instability using stability regularization or stronger shrinkage, but it
must not be selected on these four exposed cells.

The next validation step is a protocol-frozen portability audit on the local
FastCharge/CellJAR LFP cohort. Because Attia/Severson outcomes have already been
used elsewhere in this project, that cohort can provide scale and engineering
stress evidence only; it cannot serve as new independent confirmation. Genuine
confirmation still requires a newly licensed, long-running LFP cohort or
Hithium operational data frozen before scoring.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/run_nasa_evidence_weighted_moe_v3.py run-cycles `
  artifacts/nasa-dynamic-gate-v2/cycles.csv
python scripts/run_nasa_v3_post_outcome_audit.py
```

Raw NASA conversions and generated `artifacts/` remain excluded from the public
repository.
