# LifeTwin V0.23 / V2.8 implementation audit

Audit date: 2026-08-14

Review status: `implementation_frozen`

Protocol: `synthetic_long_horizon_identifiability_v2_8`

Reserved attempt: `v028-formal-20260814-a1`

## Linear result-blind history

- Fixed-core parent: `411a6676e4f40defd16ea0403712c957833887a7`.
- Design commit P: `7c0018346831ef76d5079e98bc8db4b884a8e83b`.
- Implementation commit I: `f05a0a88f92217afc725f594f07f46d79a624efe`.
- P is the direct child of the fixed core. I is the direct child of P. The freeze attestation commit must be the direct child of I and may add only this audit and the machine-readable freeze record.

No V2.8 generator, formal runner, formal seed, sealed truth or formal root was created or consumed before implementation freeze. The V2.7 terminal record remains immutable and was not reused.

## P-to-I consistency

The amendment differs between P and I only in `status`, from `preregistered_post_fix_pre_formalization` to `implementation_frozen`. Generating families, equations, distributions, data scale, partitions, features, models, optimizers, parameter bounds, thresholds, endpoints, denominators, gates, uncertainty rules, negative controls, confidence rules, success conditions and terminal classifications are inherited without change.

V0.23 adds only the fixed V2.8 identity boundary over the shared lifecycle and the deterministic risk-score reduction reviewed in `411a6676e4f40defd16ea0403712c957833887a7`. The logistic formula, coefficient dtype, threshold, tolerance and fail-closed primary-score check are unchanged. The shared lifecycle is not copied.

## Identity and process-boundary evidence

- Protocol: `synthetic_long_horizon_identifiability_v2_8`.
- Sole attempt: `v028-formal-20260814-a1`; a2 is rejected before any root is created.
- Thirteen seed roots: `202608140801` through `202608140813`, exact named order, unique and disjoint from V2 through V2.7.
- Parent, isolated generation and truth-incapable prediction profiles bind the same amendment byte hash, semantic hash, seed registry and frozen environment attesters.
- The CLI exposes no protocol, config, seed, threshold, partition, registry, resolver or success-condition override.
- The authenticated V2.8 view reaches whole-bundle validation; naked artifacts remain rejected.
- The V2.7 and earlier thin profiles and terminal records remain byte-addressable and unchanged.

## Result-blind development evidence

The canonical-first rehearsal closeout SHA-256 is `b48433fcd0573f91fc8b01ded115398566a838a37aa7eeca11805579d511a28d`; its fixture inventory SHA-256 is `68cdd48590d05352e77559ecc679cb3db107a3fe2e6549f6176895d5a4e67c10`. At frozen cardinality, whole validation, calibration partition derivation and consumption, production recomputation and two byte-identical calibration-mask commitments passed. The mask bytes have SHA-256 `427944ae027d6928fcc2b56dcc8c29a768fcd5c4fc894192bac5a09746091ac0`.

This evidence is development-only. Model-state and prediction-capsule boundaries remain unverified because their authenticated ledger and truth-derived prerequisites were not fabricated. The large fixture is not tracked and the long rehearsal was not repeated during formalization.

## Verification evidence

- V2.8 profile and risk-score matrix: `39 passed, 1 skipped`; the sole skip is the post-F attester test.
- Numeric, fit-commitment atomicity, preregistration and terminal matrix: `49 passed`.
- Checkpoint registry, lifecycle and V2.6-V2.8 profile compatibility excluding version-specific clean-freeze attesters: `121 passed, 3 deselected`.
- Ruff check and format check passed for every changed Python file.
- Compile/import, CLI help and `git diff --check` passed.
- Amendment byte SHA-256: `b5e93bac3e744cd6bfff09edf437f6a17d37c20fa0382f9546b8459dce740a1a`.
- Amendment semantic SHA-256: `a27124fbb86307b8c02f5e7a011e7941a7a114e6acec5df041a7f0cbed6e99f1`.
- Preregistration byte SHA-256: `a487b4ac6544bbb80d5ddaf71a3e86a85548cf957865e35ad6c9d0734ba22d3a`.
- Environment lock byte SHA-256: `0619ac43d21e48d3f78554b9e3d25ec270974f1fa987653951242748491534f5`.
- Risk-score implementation byte SHA-256: `97ff1b8dd26277b80056f58dacca0900bd6c7280acda86016428936d5dd22b79`.
- Risk-score regression byte SHA-256: `4137a05f998544af34865837ea5786de7bf6b1be95e92c3ef6f3a43ae650f2b9`.
- Bound implementation source and test files: `117`.
- Implementation source-tree SHA-256: `f85524790e27afc4629d1643bebd1e037dd6f930c0056f2e9fe0acb3aa6491d9`.

## Environment and execution boundary

The frozen runtime is CPython 3.12.13 on 64-bit Windows with the exact direct package versions in `requirements/v028-formal.txt`, `PYTHONHASHSEED=0`, and all recorded numerical thread controls fixed to one. Formal and prediction attesters must independently verify the freeze commit, P-to-I-to-F topology, source hashes, package lock, deterministic environment and clean worktree.

This audit does not authorize the reserved attempt. The frozen state is `frozen_no_formal_attempt_started` pending independent review.

## Claim boundary

Any later result is limited to the frozen six-core-family synthetic 25-year protocol. It is not evidence of real LFP cells, Hithium products, storage stations or 15-25 year real-world predictive accuracy.
