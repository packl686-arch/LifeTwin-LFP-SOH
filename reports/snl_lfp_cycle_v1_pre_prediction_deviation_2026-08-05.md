# SNL LFP cycle V1 pre-prediction deviation record

Date: 2026-08-05

## Decision

`snl_lfp_cycle_external_v1` is stopped before prediction and scoring. Its frozen
adapter required one positive full-capacity SOH observation at every raw cycle
from 1 through 300. The SNL cycle-summary files do not satisfy that semantic
assumption.

## Observed deviation

- The official protocol inserts three 0.5C, 0-100% SOC capacity-check cycles at
  round boundaries.
- Routine partial-DOD rows report the programmed throughput (approximately 0.6
  or 0.2 of nominal capacity), not full-cell SOH.
- Transition rows can contain zero throughput or aggregate more than one cycle;
  raw cycle indices can also skip or repeat around long rests.
- The first failed frozen check was cell
  `SNL_18650_LFP_25C_0-100_0.5-0.5C_a`, raw cycle 207, whose summary row had
  zero charge and discharge capacity. Raw cycle 208 was absent and raw cycle
  212 aggregated roughly two cycles.

These are protocol/export semantics, not values that may be silently imputed.
No prediction bundle was generated and no target suffix score was calculated.

## Corrective path

The replacement experiment extracts the periodic full-depth 0.5C reference
performance tests (RPTs), collapses adjacent pre/post-round checks with no
material cycling exposure between them, and uses cumulative discharge
throughput as equivalent full cycles. The replacement is explicitly classified
as retrospective development because individual capacity outcomes were seen
during this deviation audit.

## Claim boundary

This record supports adapter correctness and transparent failure handling only.
It is not a negative model result and cannot be used as independent validation.
