# Synthetic Long-Horizon Identifiability V2.5 Implementation Audit

Status: `result_blind_implementation_audit_passed`

Protocol: `synthetic_long_horizon_identifiability_v2_5`

Implementation profile: `V0.20`
Reserved sole attempt: `v025-formal-20260812-a1`

## Frozen lineage

- Generic identity core: `1c0b1187e5a44cad3b5aecc1bbd54f88d2fc1d30`.
- Design/preregistration commit P: `6332f36df96f12fb80cf02a6d241c49c18fde805`.
- Implementation commit I: `a9fd9362ae72b8489576ce0730393b97e83cacb7`.
- I is the direct child of P. The implementation attestation commit must be the direct child of I and may add only this audit and the V2.5 freeze record.
- The P-to-I amendment change is exactly the status transition from `preregistered_post_generic_core_pre_formalization` to `implementation_frozen`; all scientific fields remain byte-identical.

## Result-blind implementation scope

V0.20 adds only the fixed V2.5 boundary required to run the generic lifecycle safely:

1. A strict byte- and semantic-hash validator for the V2.5 amendment and its one attempt/seed registry.
2. An authenticated V2.5 contract view that adapts only protocol identity, config commitment, config path, and the 13 seed roots. All inherited scientific and artifact fields are restored and compared with V2.4.
3. A fixed formal and prediction environment attester bound to the direct-child freeze topology, exact source bytes, CPython/package versions, deterministic thread variables, and a runtime hash sentinel.
4. A thin V2.5 runner/CLI profile. No CLI option can override protocol, config, seed, threshold, partition, checkpoint registry, endpoint, gate, or success condition.
5. Explicit internal propagation of the single immutable V0.20 checkpoint registry through center/risk/calibration production, truth firewalls, committed model-state loading, prediction capsule loading, scoring reveal, and terminal publication.
6. Checkpoint registry drift is classified as a proven integrity void through typed cause-chain recognition; it cannot fall through to the unknown terminal default.
7. The generic partition pipeline passes the already authenticated artifact contract to all three output schema validators. No output row, value, model rule, or numeric threshold changes.

The V2.4 default entry points retain their existing defaults and behavior. No lifecycle implementation was copied, and no runtime global, context variable, environment identity switch, import-order switch, or scientific override was introduced.

## Verification evidence

All tests used deterministic hand-written or existing synthetic fixtures. No formal generator, formal seed, sealed truth, prior formal output, or external/raw dataset was used.

- V0.20 registry, generic identity, real reveal entry, and V2.5 fixed-profile tests: `95 passed` before the freeze record; the freeze-only attester test was correctly skipped until F exists.
- Inherited V0.17-V0.19 pre-result, terminal, output numeric, member-fit numeric, fit-commitment atomicity, and V2.4 preregistration tests: `74 passed`.
- V0.17/V0.18 partition contract tests: `11 passed`.
- V0.19 exact-cardinality partition contract tests after final carrier correction: `6 passed`.
- Final V0.20/V2.5 plus V0.19 output numeric gate: `100 passed`, with only the pre-F attester test skipped.
- Ruff check, Ruff format check, compile/import, CLI help, invalid-attempt rejection, and `git diff --check`: passed.
- Invalid attempt `v025-formal-20260812-a2` was rejected without creating any output root.
- The four reserved V2.5 formal roots were absent throughout P and I.

The freeze record independently enumerates every bound implementation/test file and its SHA-256, the aggregate source-tree SHA-256, protocol/config/preregistration/environment hashes, and the P-to-I-to-F topology.

## Inherited scientific rules

V2.5 inherits V2.4 without changing the scientific model families, features, optimizer, numeric rules, partitions, thresholds, five primary endpoints, family gates, negative controls, confidence intervals, success/failure rules, or truth-access order. The sole protocol change is the exact-set checkpoint registry contract and its fail-closed integrity classification.

## Exposure and claim boundary

Before this audit and freeze, V2.5 generation had not started, no V2.5 seed had been consumed, no V2.5 formal root had been created, and no V2.5 truth or outcome had been exposed. The reserved attempt remains unstarted pending separate authorization.

Any future V2.5 result is limited to the frozen six-core-family synthetic 25-year protocol. It is not evidence of real LFP-cell, HiTHIUM-product, storage-station, or 15-25-year real-world prediction accuracy.
