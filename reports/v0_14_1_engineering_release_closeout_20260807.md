# LifeTwin v0.14.1 engineering release closeout — 2026-08-07

## Release status

`v0.14.1` is an engineering reproducibility and data-governance patch release
on top of `v0.14.0`. It does not replace, revise, or reinterpret the frozen
`v0.14.0` scientific result. V0.14 remains `failure`; V0.15 remains
`inconclusive_not_success`; V0.16/V2.1 remain implementation freezes without a
formal generation and scoring result.

## Included engineering closeout

- The Windows full-reproduction path now uses the frozen atomic directory
  publication protocol and has completed public cross-platform validation.
- GitHub-hosted quality, Ubuntu reproduction, Windows reproduction, and Pages
  completed successfully after the GitHub Actions/Pages service incident was
  resolved. The outage-era failure and cancellation attempts remain preserved
  in the public record.
- Data-governance changes are forward-only corrections: MATR identity-only
  intake stops before outcome-bearing summary material; NASA formal prepare,
  predict, and score entry points remain rights-gated; generated audit outputs
  are new-directory-only and manifest-covered.
- The extracted NASA ordinary-battery snapshot was admitted only as metadata:
  38 MAT files, 10 README/TXT files, 34 filename-derived `Bxxxx` identities,
  and 4 identical representation groups. MAT/capacity-value reads, training,
  prediction, scoring, and SNL-content reads were all zero.
- The independent-validation candidate and metadata-only intake workflow are
  prepared for a future, licensed, outcome-blind dataset. No eligible public
  independent long-duration LFP confirmation cohort is currently available.

The NASA V3 four-CSV work and FastCharge V1/V2 work remain retrospective
development evidence. NASA chemistry is not authoritatively established as
LFP, and the extracted NASA metadata object is separate from those four CSVs.
None of the governance changes produces a new model-accuracy result.

## Public evidence

- Cross-platform recovery and outage history:
  [`cross_platform_ci_recovery_closeout_20260807.md`](cross_platform_ci_recovery_closeout_20260807.md)
- Forward-only data-governance correction lineage:
  [`data_governance_forward_correction_closeout_20260807.md`](data_governance_forward_correction_closeout_20260807.md)
- Data-asset intake and V1.1 correction:
  [`../docs/data_asset_intake_20260806.md`](../docs/data_asset_intake_20260806.md) and
  [`../docs/data_asset_intake_20260806_v1_1_correction.md`](../docs/data_asset_intake_20260806_v1_1_correction.md)
- NASA provenance and rights boundary:
  [`../docs/nasa_pcoe_battery_data_provenance.md`](../docs/nasa_pcoe_battery_data_provenance.md)
- Independent-validation execution boundary:
  [`../docs/independent_validation_execution_2026_08_cn.md`](../docs/independent_validation_execution_2026_08_cn.md)

The pre-release cross-platform evidence for the same scientific and engineering
tree is retained in
[public-release-ci run 31142437998](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31142437998)
and
[Pages run 31142437189](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31142437189).
The release tag itself must be accepted only after its own GitHub Actions run
completes; a green earlier run is not a substitute for tag validation.

## Local verification

From a clean checkout with Python 3.12 and the frozen reproduction constraints:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements\reproduction.txt -e ".[dev,showcase]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\verify_public_release.py --project-root .
git diff --check
```

The public release manifest freezes every tracked release file except the
manifest itself. The release bundle is derived from the annotated tag and does
not include raw NASA, BEEP, MATR, or SNL data, downloaded CI artifacts, local
competition materials, credentials, or machine-specific paths.

## Evidence boundary

This patch demonstrates engineering reproducibility, release integrity, and
data-governance controls. It adds no independent validation, Hithium product
evidence, storage-station validation, or real 15–25-year accuracy evidence.
The Naumann unit remains a condition-mean trajectory; Geisbauer remains a
60°C/120-day external stress screen; synthetic 25-year evidence remains a
structural stress test rather than real long-duration validation.
