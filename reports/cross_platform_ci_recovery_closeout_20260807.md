# Cross-platform CI recovery closeout — 2026-08-07

## Status

The public-release CI and Pages workflows passed on frozen commit
`b872c33ee6b5b2010e1478a808e48e0c64150928`. This is an engineering
reproducibility result. It does not change any frozen scientific outcome or
increase the evidence grade of a model or dataset.

## Outage and recovery timeline

The first two attempts of
[public-release-ci run 31120438473](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31120438473)
occurred during GitHub incident `qcvjkzcs7j74`:

- Attempt 1: the Windows job `92679779544` failed while GitHub Actions could
  not resolve action download information; quality `92679779540` and Ubuntu
  `92679779599` were cancelled before repository work completed.
- Attempt 2: Windows job `92682619571` crossed setup and entered the full
  reproduction, then received an external cancellation; quality `92682619528`
  and Ubuntu `92682619543` were also cancelled. This was retained as outage
  evidence rather than represented as a repository failure.
- Attempt 3: quality `92735031283`, Ubuntu reproduction `92735031282`, and
  Windows reproduction `92735031392` all completed successfully.

[Pages run 31120438218](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31120438218)
also retained its first failed outage-era attempt. In attempt 2, build
`92735032364`, deploy `92735124405`, and report-build-status `92735124428` all
completed successfully.

GitHub first moved the incident to monitoring at
`2026-08-07T00:06:24.906Z` (`2026-08-07T08:06:24.906+08:00`). The controlled
revalidation was triggered while the incident was still monitoring and both
Actions and Pages were operational. GitHub subsequently marked the incident
resolved at `2026-08-07T02:04:44.460Z`
(`2026-08-07T10:04:44.460+08:00`); a no-cache check at
`2026-08-07T02:43:51.880699Z` confirmed the incident was resolved and both
components remained operational.

## Artifact verification

The attempt-3 reproduction artifacts were downloaded and independently hashed:

| Platform | Artifact ID | Summary status | Pytest | Summary SHA-256 |
|---|---:|---|---:|---|
| Ubuntu | `8978157039` | full reproduction passed | 914 passed, 0 skipped | `cfcf3d5b3ab4de746325d0540c374336abe600afcf6d9317c2206aaf96290c9e` |
| Windows | `8978128115` | full reproduction passed | 914 passed, 0 skipped | `5b01ea991f9c2aecd9fa76af483197fe650d0d9ed543bcbba113b3c0f19318e2` |

Together the downloaded artifacts contained 144 files and 19,549,204 bytes.
All 64 internal generated-file hash references matched their downloaded files,
and all four recorded figure hashes matched. Both summaries recorded full mode,
atomic publication, a clean checkout at `b872c33`, passed release verification,
and zero return codes for every reproduction command.

The complete local audit is retained in the ignored directory
`artifacts/ci-outage-recovery-validation-v1_6_1-20260807`. It is intentionally
not committed. Its `output_manifest.json` covers 154 files and has SHA-256
`79c66bbe867412a88b941944798864d91ab338facdf47c1aa9560c9e952122eb`;
every listed byte count and SHA-256 was independently recomputed.

## Evidence boundary

This closeout proves that the frozen commit passed its public-release checks
and full reproduction workflow on GitHub-hosted Ubuntu and Windows, and that its
Pages workflow completed. It does not add model accuracy, independent
validation, NASA or BEEP validation, real-cell or real-station evidence, or a
validated 15–25 year prediction claim. V0.14 remains `failure`; V0.15 remains
`inconclusive_not_success`; V0.16/V2.1 remain implementation freezes without a
formal generation and scoring result.
