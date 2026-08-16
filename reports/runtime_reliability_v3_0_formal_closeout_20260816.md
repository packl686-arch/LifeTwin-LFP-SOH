# LifeTwin V3.0 formal runtime-reliability closeout

Date: 2026-08-16

Protocol: `lifetwin_structure_fit_runtime_reliability_v3_0`

Attempt: `v300-formal-20260815-a1`

Terminal disposition: **`success`**

## Decision

The sole post-freeze authorized V3.0 attempt completed all seven normal jobs and
the fixed eight-case failure matrix. Independent re-evaluation reproduced the
terminal `success` decision. All ten preregistered conjunctive gates passed and
`failed_gates=[]`.

This is a formal success for the declared truth-free structure-fit runtime
reliability study. It is not a battery-prediction accuracy result, does not use
sealed truth, and does not repair or reclassify V2.10 or the original V1 model
failure.

## Identity and timing

- Freeze F: `e5dda98f0b69f4ccc3ddf7eaf4730660f36c1af0`.
- Implementation I: `cb6e1908f64325178c7b714aa8c1e2fe61de27b0`.
- Protocol P: `942acfa9b221da0d34d4411a76cb37c584293c1f`.
- Authorization record SHA-256:
  `43a696f392d2bf2efb82a858b1efd89aa025c05f02ea30c51bb8b122270ad94f`.
- Started: `2026-08-16T01:03:59.718724Z`.
- Finished: `2026-08-16T04:33:44.929506Z`.
- Total orchestration time: `12585.210782` seconds.

The 7,200-second elapsed gate applies independently to each 5,950-cluster job,
not to total orchestration time. Both full-scale jobs passed it.

## Normal jobs

| Job | Clusters | Workers | Seconds | Peak working set | Peak private | Minimum available memory |
|---|---:|---:|---:|---:|---:|---:|
| `reference-96-serial` | 96 | 1 | 276.510 | 311,865,344 | 276,639,744 | 5,989,068,800 |
| `reference-96-parallel-a` | 96 | 6 | 59.564 | 1,084,133,376 | 1,037,348,864 | 6,531,170,304 |
| `reference-96-parallel-b` | 96 | 6 | 60.621 | 1,083,588,608 | 1,037,312,000 | 6,523,973,632 |
| `scale-1024-a` | 1,024 | 6 | 1083.723 | 1,190,584,320 | 1,144,139,776 | 5,342,855,168 |
| `scale-1024-b` | 1,024 | 6 | 1170.592 | 1,194,176,512 | 1,143,611,392 | 5,136,408,576 |
| `full-5950-a` | 5,950 | 6 | 5872.370 | 1,700,737,024 | 1,733,279,744 | 3,816,800,256 |
| `full-5950-b` | 5,950 | 6 | 4035.117 | 1,700,585,472 | 2,040,004,608 | 2,144,694,272 |

Every job reported `passed`, wrapper and child exit code zero, empty stderr,
empty worker exit-code list, terminal progress, no runtime failure telemetry and
zero resource-sampling errors. The seven jobs produced 1,224,296 diagnostic
rows and 9,794,368 forecast rows in aggregate.

## Determinism

Exact canonical hashes matched within every preregistered group:

| Group | Member-fit diagnostics SHA-256 | Forecast bundle SHA-256 |
|---|---|---|
| 96 serial/parallel A/parallel B | `a80fba631b2e23e6e3c3a6edc0bb8fa096c4d082a41a95fe51c9ebea0848320a` | `aea663636d47ca5fdd24e1b670003ec6f5e5a94589fcd4addf6d021a191e1dd4` |
| 1,024 A/B | `28a7f14fcf03f1895728d812f2aa381fbc30b0d7bb31471a6f1efaf4df752ca2` | `fdd21ef2764a4927ccdd08d536b329b0ae767239a2f798d6e7aa9c2d9906102a` |
| 5,950 A/B | `2e27ecb8c5c6c23e51def9733dfe9eb02ce0ada1c46cd5c7c5177104d52874c4` | `1ad7518be34f436dee2fbc209fc488ff78de0fd1df8f3c6c1f924f9331d1876e` |

## Failure transparency

The eight-case matrix passed and remained result-blind. It observed the exact
declared boundaries for verified-bundle IO, pool construction, submission,
completion wait, worker future, broken pool, output validation and executor
shutdown. The matrix used no formal input, formal seed, formal row or sealed
truth.

## Integrity and sealing

- Terminal record SHA-256:
  `beed353e1644511d36fa01607e4eef9ad13ad6bbd24d1b0e69dbe3c081df9f1f`.
- Artifact manifest SHA-256:
  `716fa08e4fb10113ce0594dd72f7b9d292061371792f1e2644f8f6663a297e64`.
- Attempt progress SHA-256:
  `8af1214167ec4d0e5562dbdf39941b8ea551caad004a9d50a09335d1e43a2f8e`.
- The manifest's 33 entries exactly matched all attempt files other than the
  manifest itself; the attempt root contains 34 files in total.
- Independent endpoint evaluation exactly reproduced all gates and the
  `success` disposition.
- Sealed evidence ZIP: `v300-formal-result-20260816.zip`, 48,087 bytes,
  SHA-256
  `ffef0c67995c3043655bacd15179970b68578823c8ff7c5efb5c1ef175a2c1e5`.

The attempt is terminal and consumed. No resume, retry, `a2` or replacement is
permitted.

## Claim boundary

This result establishes deterministic, bounded and diagnostically transparent
execution only for the frozen mixed synthetic structure-fit workload on the
declared Windows environment. It does not validate 15-25 year battery accuracy,
real LFP cells, products, stations, safety, warranty or business outcomes. The
original V1 scientific model failure and V2.10 terminal outcome remain unchanged.
