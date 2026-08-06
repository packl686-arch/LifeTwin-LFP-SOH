# LifeTwin data-asset intake V1.1 correction — 2026-08-06

## Status

V1.1 is a forward-only data-governance integrity correction. It does not alter
the frozen dataset roles, NASA partition, prefixes, metrics, configuration hash,
or any prior research result.

## MATR correction

The V1 inventory counts remain valid as data-presence facts: 140 structured JSON
files, 135 unique barcodes, five additional segments, and zero identity
conflicts. Five root MAT representations still add zero physical cells.

V1 used a parser that materialized the JSON summary while collecting identity
fields. Therefore V1 did **not** prove that its identity layer avoided contact
with future-label-bearing summary values, even though it did not use those
values for mapping or modeling. V1.1 replaces that path with an identity-field
whitelist reader. It stops before the summary object, returns only identity and
source fields, does not call the summary parser, and reports zero materialized
outcome values. This boundary is accepted only after the V1.1 adversarial tests
and directed 140-file identity audit pass.

## NASA correction

The official NASA protocol remains blocked because its rights gate remains
false. V1.1 moves that gate into every public prepare, predict, and score library
entry point, makes the command-line gate return a machine-readable blocked
result before opening inputs, and implements append-only single-attempt scoring
receipts in synthetic tests. No real NASA outcome was opened and no real score
or model-accuracy result was produced.

## Output integrity correction

Audit and score output directories are now new-directory-only. Existing output
directories are rejected, critical child-audit statuses are allowlisted, and
generated files are covered by a byte-count and SHA-256 output manifest. Failed
synthetic locked-test attempts retain an append-only failed receipt and cannot be
automatically retried with the same protocol, prediction, and future-label
identity.

## Historical boundary

Commit `9e2884a82710c2d64ca9b4d412acca5030a21986` included scope and commit behavior
that departed from its original instruction. V1.1 does not rewrite, revert, or
hide that history; it only applies a forward correction. This correction adds no
model-accuracy evidence and does not change V0.14 `failure`, V0.15
`inconclusive_not_success`, or the implementation-only V0.16/V2.1 status.

## V1.2 frozen release boundary

V1.2 restored `src/lifetwin/data/beep.py` exactly to its frozen SHA-256
`555d47dd4c3bc3310667cbdb9ba01922e4b34b52720035b76fb3932bd3049c11`.
The identity-only implementation moved to the unfrozen
`src/lifetwin/data/beep_identity.py` module, and public release verification
passed without changing the frozen hash map, release ID, version, or date.

The synchronized NASA governance evidence remains a separate zero-outcome
metadata object: 38 MAT files, 10 README/TXT files, 34 unique filename-derived
`Bxxxx` identities, and 4 byte-count/SHA-256-identical duplicate groups.
MAT/capacity-value reads, training, prediction, scoring, and SNL-content reads
were all zero, while formal NASA execution remains blocked. This does not create
a new model result, LFP validation, independent test set, or higher evidence
grade.
