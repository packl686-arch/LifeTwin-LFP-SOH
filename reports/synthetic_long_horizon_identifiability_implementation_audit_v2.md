# V0.15 Synthetic Long-Horizon Implementation Freeze Audit

Date: 2026-07-25

Author: Jincheng Liu

Protocol: `synthetic_long_horizon_identifiability_v2`

Review status: implementation frozen; formal V2 generation not yet executed

## Attested identities

The protocol and implementation are frozen in separate commits so the
implementation attestation does not refer to itself.

- Protocol freeze commit:
  `b8340f07e71d05bd1b16e1c5fcc32bfffd3b3d91`
- Frozen config byte SHA-256:
  `27dc7f89178f73779a52068c1878df26c9686faa7433686e60ba6496b6705796`
- Frozen config canonical SHA-256:
  `704fe432c385b7e8223156f12a432afc264b695c45ced019bef55f529e694909`
- Frozen preregistration byte SHA-256:
  `c1dee9f9b4ef134b1a52e9a51300c591e790c10a0e97b3fe6c15eb441b2c09f0`
- Implementation-source commit:
  `945fa1141e87f166ff1e11d566a0beec9f9ed5bd`
- Implementation source-tree SHA-256:
  `e5c8f9538dff703deb86629a569c76692b07ad309c488b66aa52da794bac3dc7`

The machine-readable freeze record binds every executable V0.15 source file,
the shared imported V1 synthetic module, and the formal CLI by byte hash. The
execution commit must be the direct child of the implementation-source commit,
and its only permitted changes are this audit, the freeze record, and
`release_manifest.json`.

## Implemented controls

The formal path is a one-shot, phase-scoped lifecycle:

1. The CLI restarts Python before scientific imports with
   `PYTHONHASHSEED=0` and all declared native thread variables set to one.
2. The environment guard requires exact Python and direct package versions,
   a clean Git tree, exact config/preregistration/lock bytes, the attested
   source tree, and loaded native thread pools with one thread.
3. Generation runs in an isolated process and writes label-free and sealed
   truth roots separately. Only commitments cross into the next phase.
4. Center development, risk development, calibration, prediction, and
   scoring are separate firewall phases with append-only ledger events and
   exact artifact commitments.
5. The prediction child receives no sealed-truth path capability. It
   independently reconstructs the frozen label-free pipeline and writes only
   the declared prediction registry.
6. Scoring validates all committed bytes, independently recomputes predictions,
   opens sealed outcomes once, and emits the exact ten-artifact score registry.

There is no caller-facing seed, count, threshold, feature, worker-count, or
reduced-analysis override on the formal entry point.

## Scientific reaudit

An adversarial metric reaudit found seven material gaps before implementation
freeze. They were corrected before commit
`945fa1141e87f166ff1e11d566a0beec9f9ed5bd`:

- The global common-pool minima are explicit required gates:
  test eligible count at least 1,805 with at least 60 catastrophes, and audit
  eligible count at least 903 with at least 30 catastrophes.
- Zero catastrophic prevalence, zero issuance, insufficient eligible count,
  and other declared count failures resolve to
  `inconclusive_not_success`, not `void`.
- Result precedence is `void`, then observed gate failure, then
  inconclusive, then success. An unavailable metric cannot hide a gate that
  was already shown to fail.
- All 250 intrinsic pairs are reconstructed from committed prefix, operating,
  mapping, and truth artifacts. The audit enforces the 125/125 mechanism
  allocation, exact shared prefixes and operating rows, parameter supports,
  shared measurement noise, curve admissibility, and at least 5 percentage
  points of 25-year latent separation.
- All 250 stress-plan pairs are reconstructed and checked for the frozen family
  allocation, shared past and placebo fields, lower-half versus upper-half
  planned-stress support, shared truth parameters/noise, and ordinary truth
  admissibility.
- Arm-A invariance is byte-level for center, square-root, bounded-power,
  structural and calibrated intervals, every prefix-derived comparator risk
  and content hash, eligibility, decisions, and ranks.
- Required diagnostic sections and all nine frozen policy rows remain present
  in positive, negative, inconclusive, and void reporting paths.

These checks prevent a malformed or easier matched-pair construction from
entering the fixed primary denominator.

## Verification evidence

Before the implementation-source commit:

- V0.15 suite with the default hash seed: 221 passed, 0 failed.
- V0.15 suite with `PYTHONHASHSEED=917`: 221 passed, 0 failed.
- Ruff lint: passed.
- Ruff format check: passed.
- Full repository suite before attestation metadata: 529 passed, 1 expected
  failure. The sole failure was the v0.14 release freeze gate observing the
  newly tracked post-release V2 files before they were added to the explicit
  unfrozen-development allowlist.
- Full repository suite with the three attestation metadata files staged:
  530 passed, 0 failed.

The frozen formal design expects 511,700 member-fit diagnostic rows and
4,093,600 member-forecast rows. I/O hashing is streamed, large CSVs are
canonicalized without retaining a second raw byte copy, and pre-prediction
fit structures are released before the committed prediction bundle is read
back. A full run still requires several gigabytes of free memory and must not
be represented as lightweight production inference.

## Known protocol tension

The frozen V2 text says that center, logistic, isotonic, or conformal fitting
failure is inconclusive. It also unconditionally requires downstream model
state, prediction commitments, and a complete manifest, while missing
commitments are defined as void. Those requirements cannot all be true when
training terminates before a model exists.

The implementation therefore fails closed: it records the actual ledger event,
does not open downstream truth, and does not fabricate model or prediction
artifacts. Such a pre-prediction termination cannot be claimed as a complete
conforming V2 score package. A future `V2.1` protocol would need to preregister
a separate terminal-inconclusive artifact registry before implementing that
path. This limitation does not alter a complete run whose frozen training and
calibration stages succeed.

## Claim boundary

No Hithium internal data, product measurements, station telemetry, or product
performance claims are used here. This is a preregistered synthetic
identifiability experiment and software integrity demonstration. It cannot
establish 15-25 year accuracy for real LFP cells or storage stations, and it
does not replace independent long-duration public or industrial validation.

At this audit point no formal V2 seed stream had been executed and no formal V2
outcome had been observed. Formal execution is authorized only from the clean
direct attestation commit after the machine-readable environment guard passes.
