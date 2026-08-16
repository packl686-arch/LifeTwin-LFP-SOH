# LifeTwin V0.23 / V2.8 risk-score consistency protocol preregistration

Design date: 2026-08-14

Protocol: `synthetic_long_horizon_identifiability_v2_8`

Reserved one-shot attempt: `v028-formal-20260814-a1`

Status: `preregistered_post_fix_pre_formalization`

Machine-readable amendment: [`synthetic_long_horizon_identifiability_v2_8_amendment.json`](../configs/experiments/synthetic_long_horizon_identifiability_v2_8_amendment.json)

## Result-blind scope

This preregistration is written before any V2.8 seed is consumed, generator or formal runner is called, and before any V2.8 formal root or sealed truth exists. All predecessor attempts and negative terminal records remain immutable and cannot be retried, resumed, reclassified or reused.

V2.7 attempt `v027-formal-20260813-a1` ended before prediction as `terminal_pre_prediction / unclassified_terminal_not_success / UNKNOWN_PRE_PREDICTION_EXCEPTION`. Its fit commitment was registered; only `center_development_truth.csv` and `risk_development_truth.csv` were opened; no prediction commitment and no score were produced. Result-blind analysis traced the failure to a raw-prefix risk-score consistency check: equivalent row and batch dot-product paths can differ at the floating-point bit level. Commit `411a6676e4f40defd16ea0403712c957833887a7` makes the existing logistic linear score use explicit elementwise multiplication followed by one fixed `axis=1` reduction. It does not change the formula, coefficient dtype, threshold, tolerance or fail-closed consistency gate.

A canonical-first development rehearsal, identified only by closeout SHA-256 `b48433fcd0573f91fc8b01ded115398566a838a37aa7eeca11805579d511a28d`, passed whole validation, calibration partition derivation and consumption, production recomputation and two byte-identical calibration-mask commitments at frozen cardinality. Model-state and prediction-capsule boundaries remained unverified because their authenticated ledger and truth-derived prerequisites were deliberately not fabricated. The large fixture and operator paths are not tracked, and this rehearsal created no formal attempt or outcome.

## Frozen inheritance

The committed V2.7 amendment remains the scientific base. V2.8 does not change its six generating families, equations, distributions, data partitions, time grids, features, 86 model variants, optimizers, parameter bounds, fit credibility, calibration, routing, fallback, rejection, uncertainty, endpoints, denominators, thresholds, family gates, safety gates, negative controls, bootstrap, Clopper-Pearson, permutation rules, success conditions or terminal classifications.

The only permitted differences are:

1. a new protocol, attempt, seed, generated-data and path identity;
2. the logistic risk-state decision score uses one fixed explicit elementwise-product and `axis=1` reduction path for both row and batch verification;
3. parent, generation and truth-incapable prediction processes attest the same immutable V2.8 identity.

Truth firewall entry points continue to receive the authenticated view. No runtime protocol, seed, threshold, partition, registry, resolver or success-condition override is permitted.

## Fresh identity and one-shot rule

The thirteen roots are `202608140801` through `202608140813` in the stream order recorded in the amendment. They are mutually distinct and disjoint from every registered V2 through V2.7 root. Recording them here does not consume them.

Only `v028-formal-20260814-a1` may ever be assigned. There is no a2, replacement, retry or result-improving resume. A formal launch requires a verified P-to-I-to-F topology, a clean frozen worktree, a matching runtime and package lock, absent and pairwise-disjoint formal roots, no attempt/process collision, and separate post-freeze authorization. Any valid scored or terminal outcome is final and must be preserved without result-dependent repair.

## Commitment and truth order

The inherited order remains mandatory: fresh generation and whole-bundle validation, fit commitment, center/risk/calibration checkpoint chain, prediction commitment, heldout truth opening, deterministic scoring, and exactly one scored or terminal registry. Fit completion follows canonical write, fresh read-back, whole-bundle validation through the authenticated view and commitment. Truth may not be opened merely to diagnose a label-free or contract error.

## Claim boundary

Any future result can only describe the frozen six-core-family synthetic 25-year protocol. It cannot establish performance on real LFP cells, Hithium products, storage stations or 15-25 year real-world prediction. Stronger baselines, worst-case conditions, domain rejection, interval calibration, physical constraints, dynamic updating, business decisions, a single champion and same-protocol Model B comparison remain post-result review items and are not retroactive V2.8 gates.
