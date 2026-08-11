# FastCharge V8 measurement-stability software dry run

## Decision

The V8 implementation passed its deterministic synthetic software dry run. This is **not** a model-performance result, does not rehabilitate V7, and does not change the V5 champion.

## Motivation

The frozen V7-P100 gate failed its prefix-noise audit: small perturbations caused material decision disagreement and false activation. V8 therefore adds an outcome-free qualification layer instead of tuning another accuracy threshold on the same 41 development outcomes.

## Implemented controls

1. Exact ten-column repeatability contract; unregistered and future-outcome fields are rejected.
2. Physical-cell leave-one-out selection among zero-mean Gaussian and fixed-df Student-t noise candidates.
3. Correction for the `1 - 1/n` variance shrinkage caused by centering repeated measurements on their own group mean.
4. Repeat-order, daily-reference, tester-bridge, and per-group support gates.
5. Hash-derived Monte Carlo draws and a frozen three-part stability gate.
6. Exact V5 fallback for inactive V7, failed measurement quality, missing noise mapping, or failed stability.
7. Strict single-cell requests, prediction commitments, artifact manifests, and a cohort readiness gate before any future outcome may be opened.

## Synthetic fixture result

| Item | Result |
|---|---:|
| Generated physical-cell identities | 24 |
| Measurement rows | 192 |
| Future outcome columns | 0 |
| Selected noise family | zero-mean Gaussian |
| Tester A / chamber 1 scale | 0.002874 pp |
| Tester B / chamber 2 scale | 0.003081 pp |
| Stable-path activation probability | 1.000 |
| Stable-path correction-sign probability | 1.000 |
| Stable-path endpoint deviation P95 | 0.00794 pp |
| Missing-mapping fallback | exact zero correction |

The dry-run configuration uses 128 draws only to exercise the software. The registered real execution uses 1024 draws.

## Remaining block

Stage C remains blocked until the project has real repeatability data and a fully committed new cohort with at least 60 physical cells from at least three manufacturing batches. At least six cells and ten percent of the cohort must pass the stability gate, with activations spanning at least two batches, before future cycles 101-300 may be opened once for scoring.

The current registered gate holds the already issued P60/P100 V5 centers fixed and propagates measurement noise through the residual activation rule. It does not yet perturb and refit the upstream V5 model or estimate serially correlated measurement error. Those are separate prerequisites for production qualification even if the single-open V8 cohort passes.

## Evidence

- [Protocol template](../configs/experiments/v8_measurement_stability_blind_protocol.template.json)
- [Execution template](../configs/experiments/v8_measurement_stability_execution.template.json)
- [Execution manual](../docs/v8_measurement_stability_execution_cn.md)
- [Synthetic evidence directory](../showcase/evidence_v8_dry_run/README.md)
