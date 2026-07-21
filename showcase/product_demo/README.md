# Calendar-prefix product demo

This directory contains two future-label-free requests for the frozen Naumann
V4 development model. Each request supplies exactly 10 observed condition-mean
capacity points (`observation_index` 0 through 9) and only the 25 requested
timestamps for the future window (`forecast_index` 10 through 34).

## Run

From the repository root after installing dependencies, or from any directory
when the source checkout is on `PYTHONPATH` / installed editable:

```powershell
python -m lifetwin.cli calendar-prefix-predict `
  --request showcase/product_demo/naumann_t40_soc37_5_request.json `
  --output-dir artifacts/product-demo-fallback

python -m lifetwin.cli calendar-prefix-predict `
  --request showcase/product_demo/naumann_t40_soc12_5_request.json `
  --output-dir artifacts/product-demo-specialist
```

An editable install also exposes the `lifetwin` entry point with the same
arguments. V0.13 intentionally keeps its frozen reference/config/schema assets
in the Git checkout; it is not presented as a standalone wheel inference
service. The command
validates the request against
`configs/inference/calendar_prefix_request.schema.json`, applies the semantic
index/time/support checks, and emits the prediction and decision artifacts into
the requested output directory.

## Expected decisions

| Request | Expected mean route | Diagnostic interval | Operational decision |
|---|---|---|---|
| `naumann_t40_soc37_5_request.json` | `hierarchical_power_fallback` because the specialist gate is not ready | available at requested 80% diagnostic coverage | abstained |
| `naumann_t40_soc12_5_request.json` | `hierarchical_activation_residual` specialist | unavailable because same-route calibration is insufficient | abstained |

The contrast is intentional: it shows both automatic fallback and specialist
routing, while preserving fail-closed uncertainty behavior. A diagnostic
interval is not an operational interval.

## Provenance and boundary

The numeric values were extracted from
`data/interim/naumann_calendar_observations.csv` (SHA-256
`73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c`),
derived from the CC BY 4.0 Naumann LFP calendar-aging dataset
([DOI 10.17632/kxh42bfgtj.1](https://doi.org/10.17632/kxh42bfgtj.1)).
The forecast arrays copy elapsed times only; they contain no capacity, SOH,
loss, resistance, or other future outcome.

These are retrospective, already-inspected public condition-mean examples.
They are not outcome-blind external validation, individual-cell predictions,
Hithium product evidence, utility-scale storage validation, or support for
15-to-25-year extrapolation. Operational issuance remains disabled until an
independent long-term calibration protocol is completed.
