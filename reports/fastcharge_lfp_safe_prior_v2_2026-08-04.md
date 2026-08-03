# FastCharge LFP safe-prior V2 report

Date: 2026-08-04

Author: Jincheng Liu

## Evidence boundary

V2 is a post-V1 development experiment on the public MATR fast-charge
LFP/graphite cohort. The cycle-200 V1 outcomes were already visible, but the V2
configuration and cycle-300 predictions were frozen before scoring target
cycles 201-300. This makes the extension useful development evidence, not an
independent outcome-blind confirmation.

The experiment is accelerated cycle aging. It is not calendar-aging evidence,
stationary-storage field validation, Hithium product validation, or evidence of
15-25 year forecast accuracy. No Hithium data were used.

## Frozen protocol

The semantic configuration SHA-256 was frozen as
`fddf507e6aa812d9360b4765f5b45f3771a20838ea49b5b94246be9c6312263f`.
The cohort contains 41 author-training cells, 41 primary-test cells, and 40
secondary-test cells, each with contiguous support through cycle 300. Test-cell
prediction receives only P20/P40/P60/P100 prefixes; complete histories are
available only for the 41 training cells.

Before prediction, a schema-only audit found one missing `charge_time_s` value
at cycle 251 of training cell `MATR_B2C0`. The protocol froze a within-cell,
past-only forward fill for this allowed covariate. Capacity imputation remains
forbidden. The canonical cycle-table SHA-256 is
`521b0922bb352873b2f01790cce4c13fd8fb3630261a8c0c2cb59e8c3923a54f`.

V1 failed because low-evidence fallback assigned equal mass to every expert,
including a catastrophic unconstrained linear extrapolator. V2 instead learns a
safe pool from strict leave-one-training-cell-out suffix errors at each prefix.
An expert must stay within both 1.25 times persistence MAE and 0.1 percentage
points above persistence MAE. Unsafe experts receive exactly zero weight. Local
similarity evidence may move weight only inside that safe pool; weak evidence
returns to the training-derived safe prior.

The full-linear expert failed qualification at all four prefixes. Persistence,
robust recent linear, constrained sqrt-linear, and nearest-neighbor delta
transfer formed the frozen safe pool. The target-prefix SHA-256 is
`ca041d4306cc909e3649f8a031f02c7b451275e9985937ce2b6bea406df800af`.
The 635,040 prediction rows were committed with SHA-256
`21257adc281205a998184328f7836fc7a87b07323eaea7f093c580b2ea9a5da9`
before target suffix scoring. The final score-table SHA-256 is
`a0b1a33fe02eb513267944d4acfc29cd2ec7816ae77adad3f9dc8a0d405e4711`.

## Frozen result

Values are equal-cell, equal-prefix trajectory MAE in percentage points from
the registered prefix through cycle 300.

| Method | Primary test | Secondary test | Overall |
|---|---:|---:|---:|
| Safe hard local-risk selector | **0.341** | 0.231 | **0.286** |
| Nearest-neighbor delta transfer | 0.361 | 0.231 | 0.297 |
| Safe-prior local-evidence MoE | 0.416 | **0.201** | 0.310 |
| Safe global-prior mixture | 0.465 | 0.210 | 0.339 |
| Robust recent linear | 0.669 | 0.259 | 0.467 |
| Constrained sqrt-linear | 0.800 | 0.328 | 0.567 |
| Persistence | 0.939 | 0.506 | 0.725 |
| Full linear, excluded from safe pool | 33.744 | 27.556 | 30.688 |

The preregistered V2 development gate passed all five checks. The primary
safe-prior MoE reduced MAE by 57.2% relative to persistence. Its 0.310 pp MAE
was 0.013 pp worse than nearest-neighbor transfer, inside the frozen 0.05 pp
overall non-inferiority margin. Both author test splits stayed inside their
0.1 pp margins. The primary model's empirical interval coverage was 95.7% with
3.12 pp mean width, passing the descriptive coverage and width gates.

The primary MoE was not the best observed model. The safe hard selector reached
0.286 pp MAE, a 60.5% reduction from persistence and a 3.5% improvement over
fixed nearest-neighbor transfer. It selected nearest-neighbor transfer for 309
of 324 cell-prefix decisions, robust recent linear for 14, and constrained
sqrt-linear once. Most incremental benefit appeared at P100, where the selector
reached 0.192 pp versus 0.230 pp for fixed delta transfer.

| Prefix | Delta transfer | Safe hard selector | Safe-prior MoE | Persistence |
|---:|---:|---:|---:|---:|
| P20 | 0.386 | **0.383** | 0.402 | 0.701 |
| P40 | **0.301** | **0.301** | 0.325 | 0.747 |
| P60 | 0.272 | **0.270** | 0.281 | 0.747 |
| P100 | 0.230 | **0.192** | 0.232 | 0.704 |

## Interpretation

The strongest reusable mechanism is now clearer: transfer the future retention
change of early-trajectory neighbors, and use training-only local risk to switch
to robust recent linear in a small number of supported cases. The safe prior
solved V1's catastrophic fallback defect, but continuous blending slightly
diluted the best expert on average. A later model should treat the hard selector
as the primary candidate and calibrate a minimum risk margin before switching;
it must not tune that margin on these now-exposed target outcomes.

The nominal 90% intervals are strict training-LOO residual diagnostics, not
formal coverage guarantees. Their 95.7% empirical coverage may reflect
conservatism or train/test differences. Status strings inherited from the V1
diagnostic core that end in `equal_blend` mean safe-prior fallback in V2; the
full-linear expert still has exactly zero V2 weight.

## Limitations and decision

Cycle 300 remains early relative to many cells' end of life and says nothing
directly about decade-scale calendar degradation. The crosswalk is
authoritative-source-derived rather than a direct author assertion, one official
raw cell is missing, source file content hashes are incomplete, and all test
outcomes are now exposed. Do not tune another method on this same cycle-300
score and call it confirmation.

Retain V2 as a successful engineering correction and retain the safe hard
selector as the next candidate. The highest-value next validation is a frozen
test on a newly licensed long-running LFP cohort or Hithium data. Until then,
the project demonstrates a leakage-resistant method and credible public-data
stress results, not deployed long-horizon product accuracy.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/run_fastcharge_safe_prior_v2.py prepare `
  artifacts/fastcharge-portability/fastcharge_cycle_summary.parquet `
  D:\path\to\severson_table9_cells.csv
python scripts/run_fastcharge_safe_prior_v2.py predict `
  artifacts/fastcharge-safe-prior-v2/training_cycles.parquet `
  artifacts/fastcharge-safe-prior-v2/target_prefixes.parquet
python scripts/run_fastcharge_safe_prior_v2.py score `
  artifacts/fastcharge-safe-prior-v2/canonical_cycles.parquet `
  artifacts/fastcharge-safe-prior-v2/predictions.parquet `
  artifacts/fastcharge-safe-prior-v2/prediction_manifest.json
```

Raw data, the local crosswalk, and generated artifacts remain outside the public
release.
