# Windows atomic directory publishing closeout

This directory records the privacy-safe local and remote evidence for the
Windows atomic directory publishing repair. Commit SHAs identify the verified
code states without retaining development-branch names.

The initial code candidate at `18bd294d2abf6945cb54a59d44b936c47210fd61`
passed the targeted tests, four independent 10-run stability gates, the full
test suite, and two consecutive clean full reproductions (`r3` and `r4`). A
post-evidence reproduction (`r5`) then failed at the real source-checkout
calendar-prefix CLI subprocess. Its formal output was not published and its
complete unpublished staging directory remains retained.

The original r5 runner did not persist the failing pytest subprocess stdout or
stderr because it only wrote those command logs after a successful return.
That evidence gap is preserved rather than reconstructed. A pre-registered
follow-up diagnostic reproduced the intermittent failing node after two
passes, then captured the underlying CLI subprocess directly. The captured
trace ends at the final `os.replace(staging, output_dir)` in
`src/lifetwin/cli.py` with Windows `PermissionError` / `WinError 5`.
Deterministic pre-fix fault injection independently demonstrated the same
boundary, with one scientific prediction call, one publication attempt, and
no published output.

The minimal follow-up commit
`169d967f57903a8eada4b996ef9798aff178c5e2` connects only that final CLI
publication point to the already-frozen shared publisher. It does not retry
the scientific calculation. After the change:

- 48 focused atomic-publish and calendar-prefix tests passed;
- the 83-test atomic-publish and related regression group passed;
- the real source-checkout CLI node passed 20 consecutive independent runs
  with no failure, error, or skip;
- the full suite passed with 876 passed and the one pre-existing Windows
  symlink skip; and
- two new clean full reproductions (`r6` and `r7`) passed, including release
  verification and their internal full pytest runs.

The final source state at `a540cfca79f66983a2a242eacc2ca07a8790c731`
then passed the clean full Windows reproduction `r8`. Its summary SHA-256 is
`d7d6fe9bf4fa893d75ba2342a06c84260a44e880749e8668243b3d7da25b0aed`;
the run used CPython 3.12.13, passed release verification, and completed its
internal suite with 876 passed and the one pre-existing Windows symlink skip.

GitHub Actions run
[`31028713233`](https://github.com/packl686-arch/LifeTwin-LFP-SOH/actions/runs/31028713233)
also completed successfully at that same commit. Its `quality`,
`reproduce (ubuntu-latest)`, and `reproduce (windows-latest)` jobs all passed;
the Windows log confirms CPython 3.12.13. The run retained both
`reproduction-ubuntu-latest` and `reproduction-windows-latest` artifacts.

The machine-readable details and selected hashes are in `closeout.json`.

The retry protocol was frozen before implementation. Its public copy is
`atomic-publish-protocol.json`; removal of a non-scientific development-branch
field changed only the public provenance representation, not the frozen retry
semantics. The privacy-safe copy has SHA-256
`993422238f6427a9c6d98b12e2bcbbd60b0c2d1bf6013d74abab8ea2f7b433a9`.

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
