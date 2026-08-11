# CALCE PLN storage data intake audit

Audit date: 2026-08-04
Prepared for: Jincheng Liu / LifeTwin
Decision: **auxiliary development data only; ineligible for independent long-term LFP validation**

## Scope and exposure boundary

This audit examined the official CALCE landing pages, six downloaded ZIP files,
archive member names and hashes, workbook `Info` sheets, and the condition
crosswalk. It did not inspect follow-up capacity-series values. The first
crosswalk inspection displayed its initialization `Discharge Capacity` column,
so the dataset is conservatively recorded as `development_only` rather than
outcome blind.

The raw archives remain outside the Git repository and must not be redistributed
from this project.

## Qualification decision

The archived CALCE page identifies the PLN sample as a 1500 mAh pouch cell with
**Graphite/LiCoO2** chemistry. It is not LFP. The official source is:

- <https://web.calce.umd.edu/batteries/data/>
- <https://calce.umd.edu/battery-data>

The maximum verified metadata span is **254 days**, from initialization of
PLN 51-54 on 2014-10-28 to a follow-up test on 2015-07-09. This is far below
the frozen 730-day minimum. The 6-month cohorts normally have only one
post-initialization check, so they cannot support a dynamic early-to-future
landmark evaluation.

Accordingly, the dataset fails the core eligibility gates independently on:

1. `cathode_not_lfp`;
2. `duration_below_730_days`;
3. insufficient prefix and future observations for some cohorts;
4. outcome-blindness not preserved during initialization-table inspection;
5. no explicit machine-readable data license or written scope confirmation.

Under the independent-LFP intake compiler it is D0/ineligible. Conceptually it
may still serve as D1 auxiliary evidence for cross-chemistry rejection,
uncertainty widening, ingestion robustness, and data-quality diagnostics.

## Experimental design recovered from the crosswalk

The crosswalk assigns 144 physical cells to the complete factorial design:

- four storage temperatures: -40, -5, 25, and 50 C;
- three storage SOC values: 0%, 50%, and 100%;
- three nominal recheck periods: 3 weeks, 3 months, and 6 months;
- four physical-cell replicates per temperature/SOC/recheck-period condition.

This resolves an ambiguity in the landing-page wording. There are 12 cells per
temperature **within each recheck-period cohort**, or 36 per temperature and
144 overall. Six additional numbered rows, PLN 1, 2, 35, 36, 49, and 50, are
marked unassigned in the crosswalk and are not part of the 144-cell factorial
design.

## Archive and schema audit

The six archives contain 68 non-directory members and 66 unique payloads. The
capacity workbooks named `.xls` are actually OOXML workbooks and must be detected
by file signature rather than extension.

Two byte-identical duplicates are present:

| Duplicate payload | SHA-256 | Consequence |
|---|---|---|
| `Capacity_25C/3M/06_02_2015_25C_3M_Final.xls` and `Capacity_25C/6M/06_02_2015_25C_3M_Final.xls` | `55601b7c2289ab20da72bca431eeb72c9a46b90a73bf962c9e5bd7059983324c` | The apparent 25 C 6-month file is a duplicate of the 3-month file; the intended 25 C 6-month outcomes are not independently present. |
| `Capacity_50C/3M/06_29_2015_50C_3W_IC.xls` and `Capacity_50C/3W/06_29_2015_50C_3W_IC.xls` | `2db58a87e3646a74cefe0ea976b0bc4a8f84af7806ae7ed0ef1ba70f271588a8` | A 3-week workbook is duplicated under the 3-month directory and must not be counted twice. |

The condition-aware `Info`-sheet audit found:

- 144 assigned physical-cell IDs in the crosswalk;
- 133 assigned cells with a parseable initialization mapping;
- 128 assigned cells with at least one parseable follow-up workbook;
- 117 assigned cells with both initialization and follow-up metadata;
- missing parseable initialization mappings for PLN 30, 31, and 98-106;
- no parseable follow-up workbook for PLN 23-26, 41-44, 55-58, and 79-82;
- special `Buldged Samples` files at 50 C whose folder placement mixes nominal
  3-week and 3-month cell assignments and therefore requires an explicit anomaly
  rule before any quantitative use.

These counts describe metadata linkage, not usable outcome trajectories. They
must not be reported as a model sample size without a later measurement-level
parser and QC pass.

## Frozen archive inventory

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `PLN_Number_SOC_Temp_StoragePeriod.zip` | 14,682 | `5efdd7387807b3dae19508a0aae9ec782d1fe614fd2293789afa738f350b03e0` |
| `Capacity Characterization_Initialization.zip` | 57,730,650 | `13fc914990e56ee78a14a7ebde9c303e04fbe7fc85cd13a96404c791fb16154b` |
| `Capacity_-40C.zip` | 44,007,454 | `1ef1fb0c94d9299b218496d019075a835f8289209bf2ed6f3ddac5a68c5cc4d8` |
| `Capacity_-5C.zip` | 45,022,611 | `f939aefc6b88bd6fe3c28a2759df0d92b55c0c97cef1903d6edb9b19e8f7968b` |
| `Capacity_25C.zip` | 58,955,737 | `f77d6c73f2a89034059b381c3e9ae3efa2caff2b6ea3608f68cc6244debfb628` |
| `Capacity_50C.zip` | 58,330,065 | `f9173ce23c3f2e08bdcebf03cf6e61fe53d01f93ef49c90b59d985274bf6f1ae` |

## Permitted project role

Do not use this dataset to fit, select, calibrate, or independently validate the
core long-horizon LFP model. It may be considered later for a separately labeled
cross-chemistry experiment that asks whether LifeTwin rejects an LiCoO2 cohort or
widens uncertainty under chemical-domain mismatch. Any such experiment must
exclude the duplicate/misplaced files, define treatment of bulged samples before
reading outcomes, and retain CALCE attribution.

The official page says the experimental data are open access and requests source
citation for publication use, but it does not identify a standard data license.
Until CALCE confirms the requested scope, do not redistribute raw files, assert
commercial rights, or publish derived cell-level tables.
