# LifeTwin V3.0 structure-fit runtime reliability preregistration

Preregistration date: 2026-08-15

Protocol: `lifetwin_structure_fit_runtime_reliability_v3_0`

Reserved attempt: `v300-formal-20260815-a1`

Status: `preregistered_pre_implementation`

Machine-readable protocol: [`runtime_reliability_v3_0.json`](../configs/experiments/runtime_reliability_v3_0.json)

## Result-blind independence

This protocol is committed before the V3.0 implementation, before any V3.0 formal
seed is consumed and before any V3.0 formal or authorization root exists. It is a
new operational methods-validation study, not V2.11 and not a continuation,
replacement or reclassification of V2.10.

The V2.10 attempt `v030-formal-20260814-a1` remains terminal and immutable. No
V2.10 formal row, sealed truth, seed, identifier, fit output, prediction, score,
attempt root or terminal artifact may be opened or reused. The public cardinality
5,950 is used only to declare a workload size. V3.0 has new diagnostic inputs,
identities, seed namespace and roots and contains no truth or accuracy score.

## Fixed hypotheses and workload

The four hypotheses are repeat determinism, one-versus-six-worker output
invariance, full-scale completion under fixed resource ceilings and exact
result-blind lifecycle failure classification. Overall success requires all four.

Seven fresh child-process probes are fixed: a one-worker 96-cluster reference, two
six-worker 96-cluster references, two six-worker 1,024-cluster probes and two
six-worker 5,950-cluster probes. Every probe uses the same mixed truth-free curve
suite, fixed cluster prefix, twelve prefix points, eight forecast coordinates and
the inherited 86-member structure-fit library. Each child runs once; repetition is
provided by fresh process identities rather than an in-process loop.

The single-worker run is the concurrency baseline. The three 96-cluster hashes must
match exactly, as must each 1,024 pair and the 5,950 pair. Canonical SHA-256 equality
is the only equivalence rule.

## Seed and implementation firewall

Formal root `202608153001` is recorded but not consumed. Randomized curve index `i`
uses `PCG64DXSM(formal_seed_root + i)`. Even mixed-suite indices use the public
structured generator and odd indices use the randomized generator. Implementation
tests and nonformal validation must instead use development root `31000000` and a
nonformal cluster prefix.

Static protocol loading, schema checks, mocks and post-freeze preflight must not
instantiate the formal bit generator or create a formal row. The first permissible
formal seed consumption is inside the separately authorized post-freeze attempt.

## Fixed primary gates

Every normal child must exit zero, emit one valid JSON object, leave stderr empty,
report `passed`, commit a terminal atomic progress record and report no worker exit
code. Each declared hash-equivalence group must be exact.

Each 5,950-cluster child must complete within 7,200 seconds. Across normal probes,
peak process-tree working set must not exceed 2,415,919,104 bytes, peak private
bytes must not exceed 2,952,790,016 bytes, available physical memory must never fall
below 1,073,741,824 bytes, sampling errors must equal zero and observed worker count
must not exceed six.

The injected matrix must pass and observe exactly these mappings:

- `pool_startup` -> `process_pool_construction`;
- `worker_submission` -> `worker_submission`;
- `worker_completion_wait` -> `worker_completion_wait`;
- `worker_exception` -> `worker_future_result`;
- `broken_process_pool` -> `broken_process_pool`;
- `invalid_worker_output` -> `worker_output_validation`;
- `executor_shutdown` -> `process_pool_shutdown`; and
- `verified_bundle_io` -> no structure-fit runtime telemetry.

No endpoint is optional. There is no tolerance, multiplicity adjustment,
result-dependent exclusion, timing substitution or post-result threshold change.

## Environment and output contract

The reserved environment is 64-bit CPython 3.12.13 on Windows with the exact direct
versions in `requirements/v300-formal.txt`, `PYTHONHASHSEED=0` and every declared
numeric thread control equal to one. Prelaunch requires at least 4 GiB available
physical memory and 20 GiB free disk.

The fixed attempt root is `artifacts/v300-formal-20260815-a1`. The fixed
authorization record is `artifacts/v300-formal-20260815-authorization.json` and is
outside the attempt root. stdout, stderr, exit manifest and progress paths are
distinct. Existing outputs are never overwritten. Result records may contain only
aggregate counts, hashes, timing, resource aggregates, exit codes and class/phase
identities. Row values, cluster identifiers, exception messages, process IDs,
truth and scores are forbidden.

## Immutable lifecycle

The linear history is design D, this protocol P, implementation I and freeze F. I
must be the direct child of P. Nonformal validation after I uses ignored paths and
does not change source or protocol. F must be the direct child of I and may add only
the implementation audit, nonformal validation attestation, freeze record and
freeze bundle.

Formal execution requires a clean checkout of F, exact source/config/environment
attestation, absent formal root and a new explicit authorization record created
after F. The current instruction to build and freeze the study is not formal-run
authorization.

Only `v300-formal-20260815-a1` may exist. Any success, operational failure,
integrity void, external interruption or partial launch consumes it. There is no
a2, retry, resume or replacement. Partial evidence is retained.

## Claim boundary

Success would support only the declared truth-free structure-fit workload on the
frozen Windows machine and software environment. It cannot reveal V2.10's hidden
cause, repair V2.10, validate long-horizon battery forecasts, establish portability
to other platforms, or support real-cell, product, station, safety, warranty or
business claims.

