# LifeTwin V0.24 / V2.9 pair-registry row-contract preregistration

Design date: 2026-08-14

Protocol: `synthetic_long_horizon_identifiability_v2_9`

Reserved one-shot attempt: `v029-formal-20260814-a1`

Status: `preregistered_post_fix_pre_formalization`

Machine-readable amendment: [`synthetic_long_horizon_identifiability_v2_9_amendment.json`](../configs/experiments/synthetic_long_horizon_identifiability_v2_9_amendment.json)

## Result-blind scope

This preregistration is written before any V2.9 seed is consumed, generator or formal runner is called, and before any V2.9 formal root or sealed truth exists. All predecessor attempts and negative terminal records remain immutable and cannot be retried, resumed, reclassified or reused.

V2.8 attempt `v028-formal-20260814-a1` ended before prediction commitment and produced no score. Its terminal classification remains immutable. Result-blind analysis traced the direct failure to a metadata contract mismatch: each matched partition has 500 members, but each pair registry has one row for each two-member pair and therefore has 250 rows. The current prediction capsule incorrectly required 500 pair-registry rows.

Commit `6955b6c42773d5c1e2948c2ba0228cf910b7ff02` derives each pair-registry row count from the existing member count divided by the frozen two-members-per-pair relationship. It does not change pair membership, truth data, schemas, ordinary truth row counts, thresholds, tolerances or fail-closed verification. Result-blind tests accepted both 250-row pair registries and rejected 500, 249, 251, missing, extra and renamed registries through the current prediction loader and truth-commitment verifier.

## Frozen inheritance

The committed V2.8 amendment remains the scientific base. V2.9 does not change its six generating families, equations, distributions, data partitions, time grids, features, 86 model variants, optimizers, parameter bounds, fit credibility, calibration, routing, fallback, rejection, uncertainty, endpoints, denominators, thresholds, family gates, safety gates, negative controls, bootstrap, Clopper-Pearson, permutation rules, success conditions or terminal classifications.

The only permitted differences are:

1. a new protocol, attempt, seed, generated-data and path identity;
2. each matched-pair registry requires 250 mapping rows derived from 500 members and two members per pair;
3. parent, generation and truth-incapable prediction processes attest the same immutable V2.9 identity.

Truth firewall entry points continue to receive the authenticated view. No runtime protocol, seed, threshold, partition, registry, resolver or success-condition override is permitted.

## Fresh identity and one-shot rule

The thirteen roots are `202608140901` through `202608140913` in the stream order recorded in the amendment. They are mutually distinct and disjoint from every registered V2 through V2.8 root. Recording them here does not consume them.

Only `v029-formal-20260814-a1` may ever be assigned. There is no a2, replacement, retry or result-improving resume. A formal launch requires a verified P-to-I-to-F topology, a clean frozen worktree, a matching runtime and package lock, absent and pairwise-disjoint formal roots, no attempt/process collision, and separate post-freeze authorization. Any valid scored or terminal outcome is final and must be preserved without result-dependent repair.

## Commitment and truth order

The inherited order remains mandatory: fresh generation and whole-bundle validation, fit commitment, center/risk/calibration checkpoint chain, prediction commitment, heldout truth opening, deterministic scoring, and exactly one scored or terminal registry. Fit completion follows canonical write, fresh read-back, whole-bundle validation through the authenticated view and commitment. Truth may not be opened merely to diagnose a label-free or contract error.

## Claim boundary

Any future result can only describe the frozen six-core-family synthetic 25-year protocol. It cannot establish performance on real LFP cells, Hithium products, storage stations or 15-25 year real-world prediction. Stronger baselines, worst-case conditions, domain rejection, interval calibration, physical constraints, dynamic updating, business decisions, a single champion and same-protocol Model B comparison remain post-result review items and are not retroactive V2.9 gates.
