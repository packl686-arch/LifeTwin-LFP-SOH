# LifeTwin governed public-data intake — 2026-08-06

## Decision

The data-governance intake passed. The inventory, physical-identity rules,
duplicate controls, access roles, and future-label firewall are frozen. The
official NASA auxiliary baseline is **blocked before future-outcome access**
because the catalog is public but no dataset-specific license identifier, license
URL, or confirmation for publishing aggregate derived results was found. No NASA
future capacity or lifetime value was read in this intake.

This is a governance result, not a model-accuracy improvement. It does not change
any frozen LifeTwin research conclusion.

## Machine-checked inventory

The read-only source contains 261 files and 33,038,472,637 bytes
(30.769475 GiB). Three temporary lock files were excluded, leaving 258 valid
files. The source metadata snapshot before and after the audit is identical.

| Asset | Files observed | Governed statistical unit | Assigned role |
|---|---:|---|---|
| MATR FastCharge | 140 JSON + 5 MAT | physical cell barcode | main-model development training |
| SNL LFP holdout | 60 CSV + 1 catalog workbook | physical cell | external locked holdout |
| NASA ordinary battery | 6 ZIP | physical battery ID | cross-domain cycle/RUL stress |
| NASA randomized usage | 7 ZIP | not yet verified | identity inventory only |
| CALCE A123 | 34 XLSX + 2 XLS | cell, distinct from session/workbook | feature and input validation |
| Oxford Dataset 1 | 3 files | 8 physical cells | cross-chemistry/rejection stress |

The three excluded lock files were never opened, parsed, copied, or deleted.

## Access allocation

1. **Main-model development training:** MATR may be used only with its existing
   exposed-outcome development label. It is not independent confirmation.
2. **Feature and input validation:** CALCE may support field, unit, temperature,
   SOC, OCV, and dynamic-profile checks. It is not a lifetime-training cohort in
   this intake.
3. **Cross-domain stress:** NASA ordinary batteries and Oxford may challenge
   cycle-trajectory or rejection behavior only within their stated chemistry
   boundaries. NASA randomized bundles remain identity-inventory-only.
4. **External locked holdout:** SNL is reserved and unavailable to the main
   model beyond operating-system file metadata.

## SNL hard isolation

The audit paired 30 `cycle_data` filenames with 30 `timeseries` filenames using
only filename suffixes, extensions, and file sizes. It did not open the 60 CSV
files or the catalog workbook and did not compute their content hashes.

The frozen policy is:

- `main_model_access = metadata_inventory_only`
- `outcome_access = forbidden`
- `training_allowed = false`
- `model_selection_allowed = false`
- `reserved_external_holdout = true`
- `physical_cell_count = 30`
- `local_csv_count = 60`
- `metadata_catalog_record_count = 86`
- `locally_available_lfp_record_count = 30`

The catalog count of 86 must never be presented as 86 downloaded cells.

## MATR identity and representation audit

The 140 structured JSON files resolve to 135 unique physical barcodes. Five
barcodes have two JSON segments; these segments are merged within barcode and
must never cross data partitions. The audit found zero parse failures and zero
batch/protocol conflicts. Physical-cell counts by source batch are 45, 43, and
47; there are 70 normalized protocols and 114,314 summary rows.

Only JSON header and summary material was read. The large within-cycle arrays
were skipped and source hashing was disabled. The five root MAT files were
identified as two MATLAB 7.3/HDF5 batch representations and three MATLAB 5
author-result representations. They add zero physical cells. No future
cycle-life or final-capacity value was used to link MAT and JSON. Per-cell MAT
links that cannot be established from authoritative identity metadata remain
ambiguous and excluded.

The authoritative source anchors remain the 124-row Severson crosswalk and the
fixed author-code commits `1ef13d27c66dc3d73affdaa008fbeba5687b2ea4` and
`0068fd0136bcd65884f5cd94b2b967c1ba73a668`.

## NASA identity, duplicate, and split freeze

ZIP central directories contain 34 unique ordinary-battery IDs. B0025, B0026,
B0027, and B0028 each occur in two ZIP bundles with matching uncompressed size
and CRC-32, so the protocol canonicalizes one representation per physical
battery. Any future overlap with conflicting content stops automatic intake.

The physical-battery split is frozen before outcome access: 18 training, 8
validation, and 8 locked-test batteries. The four repeated IDs are all in the
validation group, so no duplicate can cross partitions. Splits use only battery
ID and source-bundle identity, never lifetime, final capacity, or model results.

NASA chemistry is recorded as
`unspecified_li_ion_not_lfp_evidence`; its only allowed quantitative role is
`cross_domain_cycle_trajectory_and_rul_stress`. Seven randomized-usage ZIPs
contain 28 internal candidate identifiers, but representation counts and ZIP
counts are not physical-cell counts. Randomized data cannot be trained or scored
until its physical identity is separately verified.

## Frozen auxiliary protocol and rights stop

The protocol freezes prefixes 20/40/60/100, a cycle-200 maximum score horizon,
at least 20 future observations, and three baselines: target-prefix persistence,
non-positive linear trend, and constrained square-root-loss trend. Prediction
accepts only an exactly truncated prefix table; future labels are written
separately; predictions and their manifest are hashed before scoring.

The intended primary metric is trajectory MAE in percentage points, averaging
physical batteries equally and then valid prefixes equally. All inference is
descriptive. No prediction interval is claimed because none was frozen.

Execution remains blocked: public catalog access and an intended algorithm-
development use do not by themselves settle a dataset-specific license or the
right to publish aggregate derived results. A documented rights review must be
completed in a new protocol version before any future capacity value is read or
the locked test is scored. Raw data redistribution remains forbidden.

## Claim boundary

This intake does not show that new data improved the main model. NASA and Oxford
do not validate LFP performance; CALCE workbooks are not lifetime-training
samples; SNL has not validated the main model; reused MATR is not independent
confirmation. Nothing here validates a Hithium cell, a storage station, or
15–25-year prediction accuracy. V0.14 remains `failure`, V0.15 remains
`inconclusive_not_success`, and V0.16/V2.1 remain implementation-only freezes.
