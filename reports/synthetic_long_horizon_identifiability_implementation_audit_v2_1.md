# V0.16 Synthetic Long-Horizon V2.1 Implementation Freeze Audit

Date: 2026-07-26

Author: Jincheng Liu

Protocol: `synthetic_long_horizon_identifiability_v2_1`

Review status: implementation frozen; formal V2.1 generation not executed

## Attested identities

The design and implementation are frozen in separate commits, followed by one
metadata-only attestation commit so the source record does not refer to itself.

- Design-freeze commit:
  `01c75354b22c51716bba16b3e4f11e2ca30c6aad`
- Amendment byte SHA-256 after the paired implementation-status switch:
  `3c1348be3dd6e0f86df84283b7a27e57f4e5747c1356497f77bb39c695f06a4e`
- Amendment semantic SHA-256:
  `2dd77ef9f9393cc982fb370f3ed8e5f7d1753a0f7f311d5b2cc24d01e2acbbde`
- Preregistration byte SHA-256:
  `ae346fe8aa5699a7b3ad0d124e8ec29851a16fcbf485309e8c3c441749540efc`
- Environment-lock byte SHA-256:
  `c9f906589332ca4afbdb87ae9ff11e5876409137d1c3e1e186bede873180e6f1`
- Implementation-source commit:
  `d676d7c3a7fffb7806e92822c35d4a027d8677bf`
- Implementation source-tree SHA-256:
  `e4b8a7bbf4612212b7aa38d04d0e520faac6e5d064f3a47dd97eacd75ef71fe0`

The machine-readable freeze record binds the shared synthetic implementation,
all inherited V0.15 modules, every V0.16 module, and the V0.16 formal runner by
byte hash. The execution commit must be the direct child of the implementation
commit, and its exact diff is restricted to this audit, the freeze record, and
`release_manifest.json`.

## Amendment implementation

V2.1 preserves the V2 scientific endpoints, thresholds, model library, feature
definitions, common-pool rules, matched-pair requirements, negative controls,
and reporting obligations. It implements only the preregistered repair and
associated lifecycle rules:

1. The risk and isotonic heads use the precommitted
   `risk_isotonic_eligible` population.
2. Split-conformal expansion retains all 900 calibration clusters.
3. The eligibility mask is computed and committed before calibration truth is
   opened, with exact row-order, count, family-support, and digest checks.
4. A calibration population audit binds the mask, training inputs, fitted
   substates, isotonic maps, conformal state, and final model state.
5. A valid pre-prediction failure produces only the frozen terminal registry;
   it cannot fabricate model, prediction, or score artifacts.
6. V2 and V2.1 seed, opaque-ID, and available hash coordinates are checked for
   collision before any formal generation is permitted.

The amendment file changed only its paired status fields:
`status=implementation_frozen` and
`fresh_generation.implementation_exists=true`.
`generation_has_started` remains `false`, and the normalized semantic digest is
unchanged from the preregistered design commit.

## Prediction capability audit

The isolated prediction child no longer imports the generator, truth readers,
training routines, protocol loader, scorer, or the former truth-capable V0.16
I/O and pipeline modules. It imports only:

- a standalone environment attester;
- the frozen statistical primitives in the inherited V0.15 model module;
- the append-only V0.16 ledger codec; and
- a standalone label-free prediction capsule.

The environment attester independently verifies the exact amendment,
preregistration, lock file, source tree, direct-child Git topology, clean
worktree, CPython and package versions, hash seed, and single-thread native
runtime. The capsule validates canonical bytes, physical path membership,
commitment chains, the allowed truth-hash inventory without opening truth,
training and calibration semantic links, and the immutable in-memory state.

Every formal cluster must contain all 86 frozen model variants and all eight
forecast days. The five label-free input frames must share the exact cluster
universe, diagnostic and member-forecast variant coordinates must agree, and
member forecasts must match the declared target coordinates.

The capsule independently recomputes successful member forecasts, prefix RMSE,
maximum residual, and parameter-boundary fractions from the frozen formulas.
Adversarial changes as small as 0.1 percentage point in a member forecast or
fit diagnostic are rejected.

Prediction outputs are issued as an identity-bound, one-use capability.
Authorization is held outside the public result dataclass and binds the exact
result object, exact input bundle, and canonical output hashes. Public
construction, object replacement, cross-bundle transfer, in-place DataFrame
mutation, and replay after a successful write are rejected. Inputs and every
already-written output are rehashed after each exclusive file creation.

## Verification evidence

Across the final implementation-source state and the exact three-file
attestation state:

- V2.1-focused suite in `implementation_frozen` state:
  275 passed, 0 failed.
- Full repository suite:
  810 passed, 0 failed.
- Public-release manifest verifier with the attestation metadata staged:
  passed with no unfrozen tracked files, hash mismatches, forbidden files, or
  broken links.
- Focused prediction capsule red-team suite:
  23 passed, 0 failed.
- Ruff lint:
  passed.
- Ruff format check for every new V0.16 source, test, and runner:
  passed.

The adversarial coverage includes duplicate JSON keys, noncanonical encodings,
path substitution and reparse points, commitment and ledger drift, training
substate hash breaks, calibration-mask row/count/digest changes, incomplete
per-cluster candidate registries, coordinate drift, numerical formula
tampering, output replacement, cross-bundle reuse, and post-computation
mutation.

No formal V2.1 generator, seed stream, generated-row optimizer, prediction on
generated data, truth opener, or outcome scorer was executed during design or
implementation review. No V2.1 result exists at this audit point.

## Operational boundary

The frozen design expects 511,700 member-fit diagnostic rows and 4,093,600
member-forecast rows. Formal execution therefore remains a research workload,
not lightweight production inference. It requires the exact locked environment
and adequate memory and disk capacity.

This audit establishes implementation integrity, not model validity. It uses no
Hithium internal data, product measurements, or station telemetry. A future
synthetic V2.1 result cannot establish 15-25 year accuracy for real LFP cells
or storage stations, and independent long-duration validation remains
necessary.
