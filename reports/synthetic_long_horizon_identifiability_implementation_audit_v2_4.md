# Synthetic long-horizon identifiability V2.4 implementation audit

Status: result-before implementation audit; formal execution has not started.

## Identity and lineage

- Protocol: `synthetic_long_horizon_identifiability_v2_4`
- Implementation: V0.19
- Preregistered one-shot identity: `v024-formal-20260810-a1`
- Preregistration anchor P: `f70c6dfa3e32d578ff96b73cc7cb7648527033bf`
- Implementation source commit: bound by the machine-readable V2.4 freeze record.
- The attestation commit must be the implementation commit's direct child.

The amendment moved from
`preregistered_post_root_cause_pre_formalization` at P to
`implementation_frozen` during formalization. A semantic comparison against P
confirmed that this status field is the only amendment change. No endpoint,
threshold, partition, seed, model, feature, optimizer, gate, success condition,
bootstrap, Clopper-Pearson, permutation or negative-control rule changed.

## Result-blind root-cause closure

The V2.3 trigger is explained by retained failed structural variants. For a
failed variant, the inherited fitter intentionally stores `{}` parameters,
`credible_variant=false`, three structural-NaN fit diagnostics and eight
structural-NaN raw forecasts. V0.18 then applied a blanket finite-value gate to
the complete member-fit tables. Pure deterministic fixtures reproduced the
conflict without reading or hashing a V2.3 formal member-fit value.

V0.19 validates the exact state mask. It rejects infinity, mask-external NaN,
finite values in required structural-null positions, nonnumeric coercion,
duplicate or missing keys, variant/grid drift, parameters/status/credibility
inconsistency and whole-column degeneration. It does not fill, clip, delete or
drop any value, row or column.

## Minimal implementation difference

The complete V0.19 lifecycle is an independent forward copy with V2.4 protocol,
contract, capability, environment, ledger, terminal, prediction capsule, runner
and CLI identities. Unchanged scientific modules are mechanically equivalent to
V0.18 after identity and fresh-seed substitution. The intentional differences
are limited to:

1. fresh V2.4 protocol, attempt and seed identities;
2. schema-aware member-fit state-mask validation at whole and partition gates;
3. a dedicated member-fit integrity reason code;
4. fit-output write/read-back, whole validation and commitment atomicity; and
5. V2.4 source, environment and direct-child attestation checks.

V0.18 remained byte-identical at the following critical anchors:

- numeric contract: `e91e0682899f5e41827c071e5304356758eef89087f6fff4d952d92f3106e783`
- partition contract: `7213b4877b06bbe5b71ef38306df8e1cc60f678382f04f16c6b564d6a57fb3be`
- runner: `d0c63b43872a923a8dd803c5124e1e0b99f5a7e614ec7f54b0e75a9084afa1b1`

## Fit commitment atomicity

The fit stage now orders operations as follows:

1. validate the sealed in-memory fit result;
2. canonicalize and exclusively write both fit output files;
3. freshly read and re-canonicalize the physical bytes;
4. validate all five complete files for schema, cardinality, key, hash and
   numeric contracts;
5. create the unique `fit_commitment.json`; and
6. append the later completed fit-phase ledger event and verify its commitment.

Injected write/read-back, whole-validation and commitment failures produce no
fit commitment or completed phase. A completed-ledger append failure leaves the
commitment file as a partial artifact, while the registered ledger commitment
remains null; terminal evidence therefore distinguishes partial files from
registered commitments without deleting attempted-phase evidence.

## Verification record

- Member-fit/preregistration matrix: 28 passed; the exact-cardinality case was
  run separately.
- Physical exact-cardinality integration: 1 passed in 1036.51 seconds. It
  canonically wrote, freshly read, whole-validated, derived and consumed all
  seven partitions for exactly 71,400 prefix rows, 47,600 coordinates, 5,950
  clusters, 511,700 diagnostics and 4,093,600 forecasts.
- Final V0.19 matrix: 57 passed and 1 separately-proven case deselected in
  670.33 seconds.
- V2.2/V2.3 inherited partition, terminal, numeric and preregistration gates:
  39 passed in 165.92 seconds.
- Deterministic collision-plan isolated check: 1 passed; commitment
  `ee7e0cc7dac3a3cdfba13a74278b03e136b1d234ff3c29fe035334e1b4322829`.
- All 25 V0.19 modules compiled and imported.
- Ruff check and format, Git diff whitespace, CLI help, JSON parsing and package
  pin comparison passed.
- The largest deterministic test scratch set was approximately 0.924 GiB;
  observed peak private memory was approximately 3.38 GiB.

The first temporary-directory permission failure and every later non-scientific
test-infrastructure correction were preserved outside the tracked source tree;
no failing record was deleted or rewritten.

## Information firewall and claim boundary

No V2.4 formal seed was consumed; no generator, formal attempt, formal root,
truth capability, sealed truth, score, predecessor formal model output, raw
third-party data or predecessor member-fit value was used by this stage.

Any future V2.4 result remains limited to the frozen six-core-family synthetic
25-year protocol. It cannot establish real LFP-cell, product, storage-station or
15-25-year real-world predictive accuracy.
