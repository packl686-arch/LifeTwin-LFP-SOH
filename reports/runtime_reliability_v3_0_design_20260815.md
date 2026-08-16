# LifeTwin V3.0 structure-fit runtime reliability study design

Design date: 2026-08-15

Proposed protocol: `lifetwin_structure_fit_runtime_reliability_v3_0`

Design status: `independent_design_complete_pre_preregistration`

## Research question

Can the current LifeTwin structure-fit kernel complete a predeclared synthetic
workload ladder on the frozen Windows spawn runtime with byte-stable canonical
outputs, worker-count invariance, bounded resources, atomic progress evidence and
phase-specific result-blind failure telemetry?

This is an operational methods-validation study. It is not a battery-lifetime
prediction study, does not estimate predictive accuracy and does not score against
truth. Its unit of analysis is a complete child-process probe, not a battery cell or
forecast row.

## Independence from V2.10

V2.10 and its only attempt `v030-formal-20260814-a1` remain immutable and terminal.
V3.0 will not read or reuse any V2.10 formal row, sealed truth, seed, generated
identifier, fit table, prediction, score, root or terminal artifact. It will not
reclassify V2.10, reproduce its hidden input or support a claim that V2.10 was
repaired.

The value 5,950 is reused only as a public workload cardinality. The V3.0 inputs are
new truth-free diagnostic curves with a separate protocol identity, cluster prefix,
seed namespace and artifact root. Prior result-blind engineering measurements may
justify resource thresholds, but they are not observations in the V3.0 endpoint
analysis.

## Falsifiable hypotheses

- H1, repeat determinism: independent fresh child processes given the same frozen
  workload produce identical canonical SHA-256 digests for fit diagnostics and
  forecast bundles.
- H2, worker-count invariance: the one-worker reference and six-worker reference
  produce identical canonical output digests.
- H3, full-scale capacity: each predeclared 5,950-cluster child process completes
  within 7,200 seconds without a natural worker failure and within every declared
  resource ceiling.
- H4, failure transparency: every injected lifecycle failure is assigned the exact
  declared phase while emitting no messages, row values, cluster identifiers,
  process identifiers or truth-capable data.

The corresponding null for each hypothesis is any mismatch, incomplete process,
threshold violation or observability violation. Overall success is conjunctive: all
four hypotheses must pass.

## Workload contract

The formal workload will be fixed in the preregistration commit and will contain:

1. a 96-cluster mixed-suite single-worker reference;
2. two independent 96-cluster mixed-suite six-worker references;
3. two independent 1,024-cluster mixed-suite six-worker scale probes;
4. two independent 5,950-cluster mixed-suite six-worker full-scale probes; and
5. one deterministic injected-failure matrix.

The mixed suite alternates between twelve public deterministic curve forms and a
seeded randomized curve generator. It contains no future truth and no scoring
labels. Each normal probe fits the inherited 86-member structure library and emits
only canonical table hashes, aggregate row counts, elapsed time, aggregate
process-tree resources, class-only failure telemetry and atomic progress state.

Formal seed roots will be recorded at preregistration but must not be consumed by
tests, pilots or preflight. Development validation uses a visibly different
nonformal namespace and seed root. A formal seed is first consumed only inside the
one authorized post-freeze attempt.

## Primary gates

All primary gates are exact and predeclared:

- all seven normal probes exit zero, have empty stderr, report `passed`, commit a
  terminal progress file and report no worker exit code;
- hashes match across repetitions at each cardinality;
- the 96-cluster one-worker hashes equal both 96-cluster six-worker hashes;
- each 5,950-cluster run completes in at most 7,200 seconds;
- peak process-tree working set is at most 2,415,919,104 bytes (2.25 GiB);
- peak process-tree private bytes are at most 2,952,790,016 bytes (2.75 GiB);
- minimum available physical memory is at least 1,073,741,824 bytes (1 GiB);
- resource sampling reports no sampling error and never exceeds six workers; and
- the fault matrix observes every predeclared phase and passes its information
  firewall checks.

Timing is a machine-specific capacity gate, not a performance comparison. There is
no optional endpoint, multiplicity adjustment, confidence interval or
result-dependent exclusion because the endpoints are deterministic contracts.

## Baseline and analysis

The single-worker 96-cluster run is the concurrency baseline. The corresponding
six-worker runs differ only in worker count and fresh process identity. Output
equivalence is assessed by exact canonical SHA-256 equality, never by tolerance.

For repeated workloads, the first successfully parsed run is the fixed comparison
anchor and every other run must match both artifact hashes. A missing output,
duplicate output, noncanonical JSON value, partial progress record, nonempty stderr
or wrapper/child exit-code disagreement is a failure rather than a discarded
observation.

## Lifecycle and terminal rules

The intended immutable history is design D, preregistration P, implementation I and
freeze F. P fixes all hypotheses, jobs, thresholds, identities, roots and terminal
rules. I is the direct child of P. Validation after I may create only ignored
nonformal artifacts. F is the direct child of I and may add only audits,
attestations and freeze records.

Exactly one formal attempt will be reserved. It requires a clean checkout of F,
matching source and environment attestations, absent attempt roots and a new
explicit authorization issued after F. This design request is not that
authorization. There is no a2, result-improving retry or partial resume.

Terminal dispositions are:

- `success`: every primary gate passes;
- `operational_failure`: an observed runtime, determinism, resource or telemetry
  gate fails;
- `integrity_void`: source, history, environment, authorization, wrapper or
  manifest integrity is not established before endpoint interpretation; and
- `interrupted_inconclusive`: an external interruption prevents a terminal
  endpoint decision.

Any terminal disposition consumes the only attempt. Partial artifacts are retained
and never used to authorize a replacement attempt.

## Claim boundary

A successful V3.0 result would support only the declared deterministic Windows
spawn workload on the frozen machine and software environment. It would not prove
V2.10's cause, repair V2.10, validate hidden formal inputs, establish numerical
portability to another platform, or demonstrate real-cell, product, station,
long-horizon accuracy, safety, warranty or business performance.

