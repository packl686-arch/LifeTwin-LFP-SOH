# NASA dynamic-gate V2 development report

Date: 2026-08-03

Author: Jincheng Liu

## Scope and evidence boundary

This is a post-V1 development experiment on four public NASA PCoE cells. It is
an accelerated cycle-aging software stress test, not independent confirmation,
not LFP calendar-aging evidence, not Hithium product validation, and not evidence
for 15-25 year stationary-storage accuracy. The supplied files are third-party
CSV conversions whose raw-data license remains unspecified; they are not
redistributed by this repository.

The adapter verified the four pinned source byte counts and SHA-256 values and
parsed 636 discharge records. All 636 curves supported the preregistered common
3.8-3.4 V and 0.5-1.0 Ah windows. The canonical cycle-table SHA-256 is
`d362dfd38aed433eb48885dbe11b3db63c429b61a74028fd5143425a13016388`.

## What changed

1. The adapter now derives current-cycle-only curve features: integrated charge
   and energy, integration consistency, three voltage-crossing times, common
   voltage-window duration, two common-capacity voltages, dV/dQ, and temperature
   rise. Empty impedance payloads remain excluded.
2. V2 adds four target-prefix base models: persistence, full linear, Theil-Sen
   recent linear, and a non-negative robust `sqrt(cycle) + cycle` loss model.
3. A genuine nested leave-one-cell-out gate uses only the held-out cell prefix
   plus complete histories from the other three cells. It selects models using
   two nearest training cells. Capacity-only and curve-aware signatures are both
   run so their incremental value is directly measurable.
4. Prediction and scoring are separated by a hashed fold table, prediction
   manifest, and independent scorer. Future-label mutation tests verify that a
   held-out cell's unseen suffix cannot change its predictions.
5. Descriptive evidence bands widen for shorter prefixes and for gate
   disagreement. They are diagnostic bands, not calibrated confidence intervals.

## Point-prediction results

Values are equal-cell mean suffix trajectory MAE in percentage points. Each
prefix is evaluated through cycle 132.

| Model | P20 | P40 | P60 | P100 | Mean over 16 folds |
|---|---:|---:|---:|---:|---:|
| Curve-aware mean gate | **4.771** | 6.960 | **2.651** | 2.499 | **4.220** |
| Capacity-only mean gate | 5.437 | 6.960 | **2.651** | 2.499 | 4.387 |
| Robust recent linear | 6.402 | 7.506 | **2.651** | **1.593** | 4.538 |
| Full linear | 5.675 | 6.721 | 4.434 | 2.716 | 4.887 |
| Constrained sqrt-linear | **4.771** | **6.358** | 5.288 | 3.588 | 5.001 |
| Persistence | 14.210 | 11.222 | 8.726 | 2.436 | 9.148 |

Across the 16 outer folds, the curve-aware mean gate reduced mean MAE by 13.6%
relative to full linear, 7.0% relative to the best fixed model by this aggregate
(robust recent linear), and 53.9% relative to persistence. Curve features reduced
the mean-gate MAE by 3.8% relative to the capacity-only gate. These are
descriptive development deltas on four reused cells, not estimates of deployment
generalization.

The prediction artifact SHA-256 is
`805c7c61c0bec3658f67240ea0da62e4b089b5d99141ae7917c8f24d8df2fa86`;
the score-table SHA-256 is
`bd701ccbbad6612eafe97a27299d271a401ac3da8c76d6f06de336c9ab851ee3`.

## Negative results and limits

- The strict consensus rule disagreed in all 16 fold-prefix decisions and always
  fell back to persistence. Its mean MAE was 9.148 pp, so it is retained as a
  failure ablation rather than presented as the recommended predictor.
- Curve-aware mean selection did not beat the best base model at every prefix.
  In particular, the constrained sqrt-linear base model was better at P40 and
  robust recent linear was better at P100.
- The curve-aware gate's descriptive band coverage fell to 44.5% at P100. This
  falsifies any calibrated-coverage interpretation of the current band rule.
- Three cells share a synchronized campaign and different cutoff voltages are
  present. Four cells cannot support inferential significance or stable estimates
  of cross-campaign generalization.
- The NASA chemistry is not authoritatively documented as LFP in this conversion,
  so these results cannot close the project's long-term LFP evidence gap.

## Decision

Keep the constrained sqrt-linear and robust recent experts, and keep the
curve-aware mean gate as the strongest frozen V2 candidate. Do not promote the
strict consensus fallback or claim formal uncertainty coverage. The next method
change should replace binary fallback with an evidence-weighted ensemble and
must be frozen before evaluation on an additional independent dataset. The
highest-value external action remains obtaining a licensed long-running LFP
calendar-aging cohort or industrial Hithium data.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/run_nasa_dynamic_gate_v2.py run-source `
  "D:\path\to\the\four\NASA\CSV\files" `
  --output-directory artifacts/nasa-dynamic-gate-v2
```

The runner writes the normalized cycles, fold table, predictions, prediction
manifest, per-fold scores, and machine-readable score summary. Use `--overwrite`
only for an intentional rerun.
