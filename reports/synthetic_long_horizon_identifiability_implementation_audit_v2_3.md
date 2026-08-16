# Synthetic Long-Horizon Identifiability V2.3 Implementation Audit

Status: `implementation_frozen`

Protocol: `synthetic_long_horizon_identifiability_v2_3`

Only formal attempt: `v023-formal-20260810-a1`

Audit date: 2026-08-10

## Result-free boundary

This audit was completed before V2.3 generation. No V2.3 seed was consumed,
no V2.3 sealed-truth file was created or opened, and all four formal roots were
absent. The V2.1 and V2.2 worktrees, attempts, formal roots, operator evidence,
negative outcomes and Git history remained immutable. V2.3 does not reuse any
predecessor generated row, truth value, identifier, content hash, fitted state,
prediction, score, temporary file or cache.

Direct user authorization on 2026-08-10 superseded the earlier project-level
No-Go default only for this isolated forward-only V2.3 lifecycle. It did not
authorize changing predecessor outcomes, frozen success rules or claim
boundaries.

## Frozen identities

- Root-cause evidence commit: `01448adf1e2d6596d82477ac1361407a8cc8ca40`
- Result-free preregistration commit: `6e605e79a72c76500a109d3f44b460e394859d82`
- Amendment byte SHA-256: `4aa048e77172c3a2c1069e97f3c1d33dd2a5952b7aee6c5f623a2259b847085f`
- Amendment semantic SHA-256: `1eb3778d0825e0a06e7c1b63221ba6aa914f9df21b4ef2646156035b4a001bec`
- Preregistration byte SHA-256: `e301eca9836e4ad9ae816a7dda192350fc3e7aee211c08b007e381def561c970`
- Environment lock byte SHA-256: `a146229d8bfd75ade103e25bf5e5776c4c951bdd351feaa37dbf6a6b15e4843c`
- RNG-free collision plan SHA-256: `c0b343db22fe605a73077ca33cc6a3136c88638b7c81f95f651dc44620b0f7e0`

The direct parent of the freeze-attestation commit is the implementation source
commit. Its exact commit ID, complete source byte registry and source-tree hash
are bound by
`reports/synthetic_long_horizon_identifiability_freeze_record_v2_3.json`.

## Closed root cause

The inherited label-free pipeline intentionally emits structural `NaN` values
for unavailable raw risk scores, for all seven non-primary calibration rows,
and for primary calibration probabilities that are not eligible for issuance.
V2.2 applied a blanket `isfinite` check to every numeric output cell, so a
schema-valid `risk_bundle.csv` was necessarily rejected. A 600-cluster,
5,400-risk-row synthetic fixture reproduced that conflict without a generator,
seed, truth capability or formal artifact.

V2.3 replaces only that output check. Whole-bundle and capability-derived input
tables still require their exact schemas, cardinalities, key order, units,
commitment hashes and finite required numeric values. Numerical outputs are
canonicalized first and then checked against exact schema- and state-dependent
finite/structural-NaN masks.

The following rules are now enforced together:

- feature values contain no infinity, and `all_features_finite` equals the
  observed finite mask;
- every prediction trajectory value is finite and interval bounds are ordered;
- raw risk is finite exactly when all features are finite;
- calibrated risk is finite in `[0, 1]` exactly for hard-eligible primary rows;
- decision raw risk equals the risk table, ranks exist exactly for eligible
  rows in issuance partitions, and issued flags equal the frozen rank cutoff;
- wrong-position `NaN`, wrong-position finite values, infinity, cross-table mask
  drift, fill-zero workarounds and silent coercion are rejected;
- `V023NumericContractError` maps to the registered
  `INTEGRITY_NUMERIC_OUTPUT_CONTRACT_MISMATCH` terminal reason, never
  `unknown_default`.

No generating equation, model variant, feature, optimizer, fit threshold,
partition, endpoint, family gate, safety gate, bootstrap rule,
Clopper-Pearson rule, permutation rule, issuance count, success condition or
claim boundary changed. The exact 86 variants and all inherited selection and
calibration rules remain binding.

## Tests and adversarial gates

- Numeric and terminal contract suite: 16 passed. It covers valid structural
  missingness, infinity, wrong-position missingness, cross-table drift,
  rank/issuance corruption and typed terminal classification.
- Full V0.18 partition/numeric/terminal suite: 22 passed. The real
  `_apply_partition` path consumed an exact 600-cluster center capability,
  recomputed 86 variants per cluster, emitted 5,400 risk rows and accepted only
  the registered structural missingness.
- Frozen protocol/seed/collision suite: 4 passed. It confirms no NumPy RNG-state
  consumption, all 13 V2.3 roots are unique and disjoint from V2, V2.1 and
  V2.2, and the complete coordinate commitment is stable.
- Inherited V0.15/V0.16/V0.17 plus V0.18 regression: 539 passed, 1 skipped,
  413 deselected in 250.59 seconds. The single skip is the existing Windows
  symlink-creation limitation.
- The first full regression invocation recorded 432 passes and 108 setup
  errors because pytest could not create its shared Windows system temporary
  directory. It had no assertion failure and no formal side effect. The exact
  selection was rerun with a new ignored D-drive `--basetemp` and passed as
  reported above; the initial infrastructure event is preserved here.
- Ruff, byte compilation, import checks and `git diff --check` are required
  immediately before the implementation source commit.

All V2.3 tests use handwritten or deterministic synthetic fixtures. They do not
call the V2.3 generator, consume a V2.3 seed or read any sealed truth.

## Execution boundary

Formal execution is authorized only from a clean direct-child freeze
attestation commit, with CPython 3.12.13, the exact V2.3 lock, single-thread
environment variables, `PYTHONHASHSEED=0`, four fresh pairwise-disjoint roots
under the V2.3 worktree, and `v023-formal-20260810-a1` as the only attempt.

Any preflight mismatch stops before generation. Any scored or terminal outcome
must be reported unchanged. Even a success is evidence only for the frozen
six-core-family synthetic 25-year protocol; it is not validation of real LFP
cells, Hithium products, storage stations or 15–25-year real-world accuracy.
