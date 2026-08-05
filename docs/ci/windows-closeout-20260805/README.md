# Windows atomic directory publishing closeout

This directory records the privacy-safe local evidence for the Windows atomic
directory publishing repair on branch
`codex/windows-ci-atomic-closeout-20260805`.

The code candidate at `18bd294d2abf6945cb54a59d44b936c47210fd61`
passed the targeted tests, four independent 10-run stability gates, the full
test suite, and two consecutive clean full reproductions (`r3` and `r4`). The
machine-readable details and selected output hashes are in `closeout.json`.

The retry protocol was frozen before implementation and before these full
reproductions. Its public copy is `atomic-publish-protocol.json`; it has the
same SHA-256 and semantics as the pre-registered local protocol.

## Evidence boundary

This is an engineering CI and reproducibility repair. It does not create a new
model-validity result and does not change any scientific threshold, seed, data
split, frozen output, or conclusion. In particular:

- V0.14 remains `failure`.
- V0.15 remains `inconclusive_not_success`.
- V0.16/V2.1 remain implementation-frozen without formal generation or
  scoring results.
- Naumann remains condition-mean trajectory evidence, not single-cell
  validation.
- Geisbauer remains a short-term screening dataset, not long-term validation.
- Synthetic 25-year stress tests are not real 25-year lifetime validation.

The post-evidence `r5` reproduction and branch GitHub Actions are later gates.
Their results cannot truthfully be recorded inside the evidence commit that
must exist before those gates run.
