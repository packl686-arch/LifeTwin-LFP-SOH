# V7 frozen-gate prefix robustness evidence

This directory contains aggregate, redistributable outputs from the frozen V7
P100 prefix-perturbation audit. The full Monte Carlo decision table remains a
local generated artifact and is not required to verify the published decision.

- `decision.json`: machine-readable protocol binding, runtime, result, decision,
  and claim boundaries.
- `scenario_summary.csv`: one aggregate row per registered perturbation scenario.

The audit reuses the same 41 outcome-exposed training cells. It does not use the
81 public evaluation cells, does not activate V7, and is not independent model
confirmation. The failed result withdraws the current V7-P100 candidate from a
new-cell blind test unless a separately preregistered upstream measurement-
quality gate is available.
