from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_cycle_data(
    *,
    cell_count: int = 36,
    cycles_per_cell: int = 180,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic fake data for software tests, never scientific evidence."""
    rng = np.random.default_rng(seed)
    cycle_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []

    for cell_number in range(cell_count):
        cell_id = f"synthetic_{cell_number:03d}"
        protocol_number = cell_number % 6
        protocol_id = f"p{protocol_number}"
        batch_id = f"batch_{cell_number % 3}"
        protocol_stress = 0.00035 + 0.00006 * protocol_number
        cell_stress = protocol_stress * rng.lognormal(mean=0.0, sigma=0.12)
        initial_capacity = rng.normal(1.1, 0.015)
        knee = int(rng.normal(145 - 7 * protocol_number, 5))

        for cycle in range(1, cycles_per_cell + 1):
            knee_damage = max(0, cycle - knee) ** 1.35 * 0.00005
            capacity = initial_capacity * (
                1 - cell_stress * cycle - knee_damage
            ) + rng.normal(0, 0.0012)
            cycle_rows.append(
                {
                    "dataset_id": "SYNTHETIC_TEST_ONLY",
                    "cell_id": cell_id,
                    "batch_id": batch_id,
                    "protocol_id": protocol_id,
                    "cycle_index": cycle,
                    "discharge_capacity_ah": max(capacity, 0.05),
                    "charge_capacity_ah": max(capacity + 0.003, 0.05),
                    "internal_resistance_ohm": 0.018 + cell_stress * cycle * 0.2,
                    "temperature_avg_c": 25 + protocol_number + rng.normal(0, 0.25),
                    "temperature_max_c": 28 + protocol_number + rng.normal(0, 0.3),
                    "charge_time_s": 900 - protocol_number * 45 + rng.normal(0, 4),
                }
            )

        cycle_life = (0.2 / cell_stress) * rng.normal(1.0, 0.03)
        label_rows.append(
            {
                "dataset_id": "SYNTHETIC_TEST_ONLY",
                "cell_id": cell_id,
                "cycle_life": cycle_life,
                "eol_threshold": 0.8,
                "is_censored": False,
                "label_source": "synthetic_test_only",
            }
        )

    return pd.DataFrame(cycle_rows), pd.DataFrame(label_rows)

