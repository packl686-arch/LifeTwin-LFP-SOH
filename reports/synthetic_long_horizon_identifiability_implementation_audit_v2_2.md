# Synthetic Long-Horizon Identifiability V2.2 Implementation Audit

Status: `implementation_frozen`  
Protocol: `synthetic_long_horizon_identifiability_v2_2`  
Only formal attempt: `v022-formal-20260809-a1`  
Audit date: 2026-08-09

## Result-free boundary

This audit was completed before V2.2 generation. No V2.2 seed had been consumed,
no V2.2 sealed-truth file existed or had been opened, and all four formal roots
were absent. V2.1 attempt `v021-formal-20260808-a1` and all predecessor Git and
artifact history remained unchanged. The prior 2026-08-16 No-Go default is
retained as history and was explicitly superseded by direct user authorization
on 2026-08-09 for this isolated, one-shot V2.2 lifecycle.

## Frozen identities

- Design/preregistration commit: `2fb17af12a742dca51107a26300b199e13f27fe7`
- Implementation source commit: `f92280584c4f6aacc3b52b9398ce38a17dded38c`
- Amendment byte SHA-256: `aaadd5b9d5436d6ccfa08806250f0a48bef93e04446d0c089cb2eb5cf8ce0f29`
- Amendment semantic SHA-256: `88f6e32067b9637a931b149e9950f2a690c3e4558effbde6a5002cba8bd5b6a2`
- Preregistration byte SHA-256: `6a720047bbca0671a86ac7ebabaaa15693e1b69b9ff660f7c6cea0ea5b26893a`
- Environment lock byte SHA-256: `ae29836c3fa7130b2b12af0c804349b9ebe2bf3628804b49e67320041208b02b`
- Implementation source-tree SHA-256: `e4517316ffabf8848503e6d7cebb0df368816f359d42042db8b5e591b4e4ef46`
- RNG-free collision plan SHA-256: `82881e764abc4fbef352328dcaee311fb85c1a2f8000490af18940324ad514b5`

The complete 39-file source byte registry is in
`reports/synthetic_long_horizon_identifiability_freeze_record_v2_2.json`.

## Scope of the repair

The V2.2 runner validates all five complete label-free tables against the
whole-bundle formal contract before it may derive any partition. A successful
validation issues an exact-type `WholeBundleValidated` capability. A partition
can then be sliced only through `derive_partition_view`, which applies the
preregistered partition name and exact row counts, validates schema, key order,
units, finite numeric values and canonical bytes, and issues a
`ValidatedPartitionView`. Consumption repeats the canonical hash checks and
returns isolated copies. The formal partition route exposes no `formal=False`
switch.

Center and risk partition validation now completes before the corresponding
truth capability is opened. Known whole-bundle, partition-contract and
partition-capability errors map to registered integrity reason codes rather
than `UNKNOWN_PRE_PREDICTION_EXCEPTION`.

No generating equation, model variant, feature, optimizer, threshold,
partition, endpoint, family gate, safety gate, bootstrap rule,
Clopper-Pearson rule, permutation rule, success condition or claim boundary was
changed. Seeds and identifiers are new only because V2.2 is a new protocol.

## Tests and adversarial gates

- Targeted partition/terminal suite: 9 passed. It covers a 71,400-row complete
  prefix contract, an exact 7,200-row center prefix, exact five-table center
  cardinalities including 412,800 member forecasts, schema/key/order/unit and
  finite-value checks, wrong count/order/nonfinite rejection, capability
  construction and mutation rejection, real `_apply_partition` integration,
  truth-open ordering, and typed terminal classification.
- Protocol/seed/collision suite: 4 passed. It confirms no NumPy RNG-state
  consumption, all 13 new seed roots are unique and disjoint from V2/V2.1,
  adapter changes are identity-only, and the full collision plan is stable.
- Inherited V0.16 plus V0.17 gate: 287 passed, 1 skipped, 639 deselected. The
  skip is the pre-existing Windows symlink-creation limitation.
- Ruff, `git diff --check`, byte compilation and import of all 23 V0.17 modules
  passed.
- Fifteen unchanged V0.17 modules normalize byte-for-byte to V0.16 after the
  declared namespace, protocol, version and seed substitutions. Prediction and
  scoring differ only by removal of the public formal-validation switch;
  environment, protocol, fit, pipeline, runner and terminal contain the audited
  V2.2 boundary changes.

## Execution boundary

The next commit must be the direct child of implementation commit
`f92280584c4f6aacc3b52b9398ce38a17dded38c` and may contain only this audit and
the machine-readable freeze record. Formal execution is authorized only from
that clean direct-child commit, with CPython 3.12.13, the exact V2.2 lock,
single-thread environment variables, `PYTHONHASHSEED=0`, fresh pairwise-disjoint
roots under the V2.2 worktree, and the sole attempt ID
`v022-formal-20260809-a1`.

Any preflight mismatch stops before generation. Any scored or terminal outcome
must be reported unchanged. Even a success is evidence only for the frozen
six-core-family synthetic 25-year protocol; it is not real LFP, Hithium product,
storage-station, or 15–25-year real-accuracy validation.
