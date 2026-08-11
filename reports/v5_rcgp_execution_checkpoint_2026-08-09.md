# V5-RCGP execution checkpoint (2026-08-09)

## Purpose

This checkpoint freezes the exact public development inputs reused by the V5
reference-conditioned residual experiments. It is not a preregistration and it
does not upgrade the evidence status of the MATR outcomes, which have already
been inspected during earlier development.

## Repository state

- Branch: `main`
- Base commit: `af96cd3edad6abade1ef2425fcfe2544dd5c8d5b`
- Staged-diff Git object hash at checkpoint: `fcb1a2e4c6e9e06a06bfa0d4fc9495f4c3b8c799`
- V5 development-plan SHA-256: `16f43709b1513bc562e73b612198640a344af384259887751857d66551c983ad`

## Reused FastCharge artifacts

| Artifact | SHA-256 |
|---|---|
| `canonical_cycles.parquet` | `322c30de63c7dafb5fde2653c5b45b637dd6c0828ee9294a27d5276c3296e286` |
| `training_cycles.parquet` | `ce43933e2de291f689735b9109499b6c0b2a6a137f123df1a1b4756c9ee75d34` |
| `target_prefixes.parquet` | `8004f2936fd9b9b6c0f4f5746a228cae06ca808fca3a91d3f48a431a33609d27` |
| frozen V2 `predictions.parquet` | `4c331a46cc481084a2b3011f87d6a5e72c8b14033d36d2e36ec9e8f652c007c6` |
| frozen V2 `score_summary.json` | `1508c1895c6458bbf2986c4348a366bb7d9c213def8a7e14f2782f461f18efab` |

The canonical table contains 122 physical cells with contiguous support through
cycle 300: 41 training cells, 41 primary-test cells, and 40 secondary-test
cells. V5 reuses only the 41 complete training histories for fitting and the
registered target prefixes at cycles 20, 40, 60, and 100 for prediction.

## Runtime

- Python environment: project-local Python 3.12 virtual environment
- NumPy 2.5.1
- pandas 3.0.3
- SciPy 1.18.0
- scikit-learn 1.9.0
- PyArrow 25.0.0

## Firewall and claims

- Target suffix rows are unavailable to candidate fitting and reference
  selection.
- Hyperparameter selection is performed using training cells only, with held-out
  physical cells removed from both target and reference roles.
- The two author-defined evaluation splits remain outcome-exposed development
  evidence, not independent confirmation.
- These experiments do not support calendar-aging, Hithium-product, or 15-to-25
  year accuracy claims.
