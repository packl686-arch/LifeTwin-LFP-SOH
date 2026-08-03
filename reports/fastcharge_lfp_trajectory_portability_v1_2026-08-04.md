# FastCharge LFP trajectory portability V1 report

Date: 2026-08-04

Author: Jincheng Liu

## Scope and evidence boundary

This experiment is a protocol-frozen held-out trajectory replay on the public
MATR fast-charge LFP/graphite cohort. The project had already inspected related
Attia/Severson cycle-life outcomes, so this is a large-cohort portability stress,
not independent outcome-blind confirmation. It is accelerated cycle aging, not
calendar aging, stationary-storage field evidence, Hithium validation, or a
15-25 year forecast test.

The local transport contains 140 BEEP JSON files totaling 27,946,112,408 bytes.
The streaming adapter read only their summary sections and produced 114,314
cycle rows for 135 barcodes. Source content hashes were skipped for this first
run, so the provenance manifest is explicitly incomplete. The authoritative
Table 9 crosswalk is source-derived rather than a direct author assertion.

Of 124 official cells, the local raw transport lacked `MATR_B1C18`; `MATR_B2C1`
did not have contiguous support through the frozen cycle-200 endpoint. The fixed
cohort therefore contained 41 author-training cells, 41 primary-test cells, and
40 secondary-test cells.

## Firewall and freeze

The configuration semantic SHA-256 was frozen as
`c8665bd57fccc9938207c0d4c16ae650825b16cbffc2fbd23903b07cb10f5bdb`
before target suffix trajectory scoring. Prediction received complete histories
for the 41 training cells and only P20/P40/P60/P100 prefixes for the 81 test
cells. Cycle-life labels and target suffix capacities were absent from the model
input schema.

The target-prefix input SHA-256 is
`ca041d4306cc909e3649f8a031f02c7b451275e9985937ce2b6bea406df800af`.
Prediction produced 375,840 rows with SHA-256
`6b7585add00991893fe996f4fcc95787b710a34b49706f8b487b47a393c86391`
before the scorer received complete target trajectories. The score-table
SHA-256 is
`61421e5b864336831320914675c5d12887ab6c74830c47b6dddc702fe56a6aec`.

## Frozen result

Values are equal-cell, equal-prefix trajectory MAE in percentage points from the
registered prefix through cycle 200.

| Method | Primary test | Secondary test | Overall |
|---|---:|---:|---:|
| Training-risk hard expert | **0.177** | **0.131** | **0.154** |
| Nearest-cell delta transfer | 0.190 | 0.138 | 0.164 |
| Robust recent linear | 0.282 | 0.137 | 0.211 |
| Constrained sqrt-linear | 0.406 | 0.186 | 0.297 |
| Persistence | 0.429 | 0.274 | 0.352 |
| Frozen stability-shrunk MoE | 2.130 | 2.030 | 2.081 |
| Equal-weight mixture | 5.214 | 4.078 | 4.653 |
| Full linear | 26.862 | 20.892 | 23.914 |

The preregistered MoE gate failed. It beat the equal-weight mixture in both test
splits, and its training-calibrated interval diagnostics passed, but it was
491% worse than persistence rather than at least 10% better. The failed status
is retained as the primary V1 decision.

The fixed nearest-cell delta-transfer expert reduced MAE by 53.4% relative to
persistence. The training-risk hard selector reduced it by 56.1%. The selector
chose delta transfer for 285 of 324 cell-prefix decisions, robust recent linear
for 38, and persistence once. These are held-out results for this trajectory
metric, but they remain on an outcome-exposed public cohort.

## Why the MoE failed

The failure is structural rather than random. During the nearly flat early LFP
regime, a small noisy slope can make unconstrained full-linear extrapolation
catastrophic. Training evidence usually identified that expert as poor, but the
V1 low-evidence rule shrank all experts toward equal weights. It therefore
reintroduced an unsafe expert exactly when confidence was low.

This falsifies the assumption that equal weighting over the entire expert
registry is always a safe fallback. A valid fallback must be restricted to a
training-qualified safe pool or use a conservative prior learned without target
outcomes.

## Interval diagnostics

The nominal pointwise target was 90%, calibrated using strict leave-one-training-
cell-out residuals. On the two author test splits, the hard selector achieved
94.6% empirical coverage with 1.45 pp mean width; delta transfer achieved 94.6%
with 1.70 pp width. The failed MoE achieved 96.6% with 10.62 pp width. Because
the train/test protocols are not assumed exchangeable, none of these are formal
coverage guarantees.

## Limits and next decision

All targets are scored only through cycle 200, when many FastCharge cells remain
close to their initial capacity. The very small errors do not imply accurate EOL
or long-horizon prediction. Crosswalk authority is derived, one official raw
cell is missing, source file content hashes are not yet complete, and no
industrial data are present.

V1 must not be promoted as a successful MoE. Retain nearest-cell delta transfer
and the hard selector as promising mechanisms. The next post-outcome development
version should extend the common endpoint to cycle 300 and replace all-expert
equal fallback with a training-qualified safe prior. That version will still be
retrospective development; independent confirmation requires a new licensed LFP
cohort or Hithium data frozen before scoring.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/run_fastcharge_trajectory_portability_v1.py prepare `
  artifacts/fastcharge-portability/fastcharge_cycle_summary.parquet `
  D:\path\to\severson_table9_cells.csv
python scripts/run_fastcharge_trajectory_portability_v1.py predict `
  artifacts/fastcharge-trajectory-portability-v1/training_cycles.parquet `
  artifacts/fastcharge-trajectory-portability-v1/target_prefixes.parquet
python scripts/run_fastcharge_trajectory_portability_v1.py score `
  artifacts/fastcharge-trajectory-portability-v1/canonical_cycles.parquet `
  artifacts/fastcharge-trajectory-portability-v1/predictions.parquet `
  artifacts/fastcharge-trajectory-portability-v1/prediction_manifest.json
```

Raw data, the local crosswalk, and generated artifacts remain outside the public
release.
