# LifeTwin V0.22 / V2.7 implementation audit

Audit date: 2026-08-13

Review status: `implementation_frozen`

Protocol: `synthetic_long_horizon_identifiability_v2_7`

Reserved attempt: `v027-formal-20260813-a1`

## Linear result-blind history

- Fixed-core parent: `43ca947a02b15ab373422a7b066a361ed711cd1b`.
- Design commit P: `906877fbf627bb32ac9a22f12caea1a63125b3f7`.
- Implementation commit I: `95ee2163287c56296165eaf2a8efc36dc8cab148`.
- P is the direct child of the fixed core. I is the direct child of P. The freeze attestation commit must be the direct child of I and may add only this audit and the machine-readable freeze record.

No V2.7 generator, formal runner, formal seed, sealed truth or formal root was created or consumed before implementation freeze. The V2.6 terminal record remains immutable and was not reused.

## P-to-I consistency

The amendment differs between P and I only in `status`, from `preregistered_post_fix_pre_formalization` to `implementation_frozen`. Generating families, equations, distributions, data scale, partitions, features, models, optimizers, parameter bounds, thresholds, endpoints, denominators, gates, uncertainty rules, negative controls, confidence rules, success conditions and terminal classifications are inherited without change.

V0.22 adds only the fixed V2.7 identity boundary and the result-blind exposure-event contract handoff reviewed in `43ca947a02b15ab373422a7b066a361ed711cd1b`. The shared lifecycle is not copied. Phase and failure appends receive the frozen artifact contract, while truth-firewall entry points retain the authenticated contract view.

## Identity and process-boundary evidence

- Protocol: `synthetic_long_horizon_identifiability_v2_7`.
- Sole attempt: `v027-formal-20260813-a1`; a2 is rejected before any root is created.
- Thirteen seed roots: `202608130701` through `202608130713`, exact named order, unique and disjoint from V2 through V2.6.
- Parent, isolated generation and truth-incapable prediction profiles bind the same amendment byte hash, semantic hash, seed registry and frozen environment attesters.
- The CLI exposes no protocol, config, seed, threshold, partition, registry, resolver or success-condition override.
- The authenticated view reaches whole-bundle validation; naked V2.7 artifacts remain rejected.
- All phase and failure append callers use the artifact contract, including the calibration-mask started event fixed before preregistration.

## Verification evidence

- V2.7 pre-F profile: `12 passed, 1 skipped`; the sole skip is the post-F attester test.
- Result-blind inherited matrix: `125 passed, 1 skipped`, covering V0.20 identity, checkpoint registry and lifecycle entry tests, V2.6 profile behavior, fit-commitment atomicity and terminal contracts.
- V2.6 profile compatibility excluding its version-specific freeze attester: `12 passed`.
- The exact-cardinality heavy fixtures were deliberately excluded because this stage forbids generating a 71,400-row bundle. Their frozen implementation was not changed.
- Ruff check and format check passed for every changed Python file.
- Compile/import, CLI help and `git diff --check` passed.
- Amendment byte SHA-256: `5669638e854d15dd0873ee863c93635f3f287753fa0b823c708f7e12a2c3d6b2`.
- Amendment semantic SHA-256: `d9e25ea634ff5bae3c03c6dbb0a329e994e480db80ef0621479a19023747c9cf`.
- Preregistration byte SHA-256: `ffd99eb9019e8cb86d148a94d697d6f3241701f179d0511ae848cf06d9ad63f8`.
- Environment lock byte SHA-256: `9261baf3be841996c357aa44ef815de2eaebb3a051f97de433ce67ce49047c6a`.
- Bound implementation source and test files: `110`.
- Implementation source-tree SHA-256: `1e376d9a2979f925ec2e90f7b7fbf4b6e4e8d87a105fd743386d3870398557f3`.

## Environment and execution boundary

The frozen runtime is CPython 3.12.13 on 64-bit Windows with the exact direct package versions in `requirements/v027-formal.txt`, `PYTHONHASHSEED=0`, and all recorded numerical thread controls fixed to one. Formal and prediction attesters must independently verify the freeze commit, P-to-I-to-F topology, source hashes, package lock, deterministic environment and clean worktree.

This audit does not authorize the reserved attempt. The frozen state is `frozen_no_formal_attempt_started` pending independent review.

## Claim boundary

Any later result is limited to the frozen six-core-family synthetic 25-year protocol. It is not evidence of real LFP cells, Hithium products, storage stations or 15-25 year real-world predictive accuracy.
