# V8 measurement-stability software dry run

This directory contains only deterministic synthetic software evidence for the
blocked V8 measurement-stability protocol.

- `decision.json`: contract, hash, quality-gate, stability-gate, fallback, and
  claim-boundary result.
- `noise_candidate_scores.csv`: physical-cell leave-one-out log scores for the
  registered Gaussian and Student-t repeatability candidates.
- `noise_ledger.csv`: two synthetic tester/chamber noise mappings.

The dry run uses 24 generated identities and no future capacity or SOH outcome.
Centered repeat residuals are corrected for their `1 - 1/n` variance shrinkage
before a single-measurement noise scale is fitted. The selected synthetic
Gaussian scales are 0.002874 pp and 0.003081 pp. The stable path achieved 1.0
activation and correction-sign probability with a 0.00794 pp endpoint-deviation
P95 under 128 software-test draws.

It demonstrates that a well-supported low-noise path can issue a correction and
that a missing tester/chamber mapping returns an exact zero correction. It does
not show real measurement repeatability, model accuracy, independent validation,
or production readiness, and it does not change the V5 champion. The registered
real execution uses 1024 draws and still requires a hash-bound cohort commitment
before any future outcome can be opened.
