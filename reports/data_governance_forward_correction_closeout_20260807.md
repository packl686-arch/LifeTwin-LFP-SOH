# Data-governance forward-correction audit lineage — 2026-08-07

## Status

Success. This report preserves the public lineage of the V1.2 through V1.4
data-governance audits without publishing their ignored local artifact bundles.
Every output-manifest hash below was recomputed from the actual manifest bytes,
and every listed manifest entry was independently checked against its file byte
count and SHA-256 before this report was written.

| Audit | Status | Output-manifest SHA-256 | Entries | Role |
|---|---|---:|---:|---|
| V1.2 release-boundary closeout | success | `e70092f6d2ab141188eb95424d9de3e4c005839d29b8c6d5274708e323b4ced0` | 6 | Restored the frozen `beep.py` boundary and moved identity-only intake to an unfrozen module. |
| V1.3 evidence synchronization | success | `c4312fe1261ca232400fe610337f82f1b31bdeb1f4bda43f30b93b518de1fb4f` | 7 | Synchronized the already completed MATR, NASA metadata-only, and release-boundary evidence into project documents. |
| V1.3.1 canonical-count correction | success | `0bf5878c5a9659485638450ece52184cf0d3994007e7d2daad93d8718fa9dd50` | 4 | Corrected the normative source-file count without altering the original V1.3 record. |
| V1.4 final precommit review | success | `ad59c6b981651b98f6aa4594f172d174694e1b29de0cfe41fdaf3e61f5bed7e1` | 24 | Reverified Git scope, security boundaries, five historical manifests, the full test suite, release policy, and canonical source hashes. |

## Canonical source-hash correction

V1.3 recorded 301 unchanged paths, but its recursive workspace scan included 88
Python files from temporary `.pytest-tmp` repository copies. The unchanged-hash
finding remains true for those recorded paths, but 301 is not the normative
project-source count.

V1.3.1 defined the canonical collection as the actual output of
`git ls-files -- '*.py'` plus `release_manifest.json`: 212 indexed Python files
and one release manifest, for 213 files total. The V1.3 before and after tables
fully covered those 213 paths, the filtered tables were identical, and an
independent current recomputation also matched. The canonical set contained zero
artifact, pytest-temporary, or other temporary paths. Future source-hash
references must therefore use 213, while the original 301-path record remains
preserved as historical evidence.

## V1.4 verification

V1.4 collected 914 tests. Its accepted full run completed naturally with 913
passed, one existing Windows symlink-capability skip, zero failures, zero errors,
and zero xfails. The first invocation used a 168-character Windows basetemp and
produced 42 path-derived failures; its JUnit showed that every failure referenced
that overlong basetemp. That invocation was retained as invalid rather than
hidden. The single permitted corrected invocation used a 93-character ignored
basetemp and passed.

Full Ruff, public-release verification, Git diff checks, the blocked NASA
execution gate, all five historical output-manifest recomputations, and the
213-file canonical hash recheck passed. The cached diff remained empty and Git
status was byte-for-byte unchanged during the V1.4 review.

## Evidence boundary

This lineage records data identity, metadata-only access, rights gating, release
integrity, and verification status. It does not publish raw NASA or BEEP data,
does not establish NASA chemistry as LFP, and does not create a model result,
accuracy improvement, independent validation, real-station validation, or
higher evidence grade. NASA dataset-specific licensing and public
aggregate-result release rights remain unresolved, so formal NASA prepare,
predict, and score execution remains blocked.
