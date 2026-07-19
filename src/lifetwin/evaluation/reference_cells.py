from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class KShotSupport:
    reference_cell_count: int
    minimum_query_cells_per_domain: int
    eligible_domain_count: int
    eligible_cell_count: int
    available_query_cell_count: int
    excluded_domain_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def reference_cell_support(
    frame: pd.DataFrame,
    *,
    domain_column: str,
    k_values: Sequence[int] = (1, 3, 5, 10),
    minimum_query_cells_per_domain: int = 1,
    identity_column: str = "cell_id",
) -> dict[str, object]:
    """Audit whether target domains can support k reference cells plus queries."""
    required = {domain_column, identity_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing reference-cell audit columns: {missing}")
    if frame.empty:
        raise ValueError("Reference-cell audit frame is empty")
    if frame[[domain_column, identity_column]].isna().any().any():
        raise ValueError("Reference-cell domain and identity cannot contain null values")
    if frame[identity_column].duplicated().any():
        raise ValueError(f"Expected unique {identity_column} rows")
    if minimum_query_cells_per_domain < 1:
        raise ValueError("minimum_query_cells_per_domain must be positive")
    k_shots = tuple(int(value) for value in k_values)
    if not k_shots or any(value < 1 for value in k_shots):
        raise ValueError("k_values must contain positive integers")
    if len(set(k_shots)) != len(k_shots):
        raise ValueError("k_values must be unique")

    sizes = frame.groupby(domain_column, sort=True)[identity_column].nunique()
    support: list[KShotSupport] = []
    for k_shot in sorted(k_shots):
        eligible = sizes >= k_shot + minimum_query_cells_per_domain
        eligible_sizes = sizes[eligible]
        support.append(
            KShotSupport(
                reference_cell_count=k_shot,
                minimum_query_cells_per_domain=minimum_query_cells_per_domain,
                eligible_domain_count=int(eligible.sum()),
                eligible_cell_count=int(eligible_sizes.sum()),
                available_query_cell_count=int((eligible_sizes - k_shot).sum()),
                excluded_domain_count=int((~eligible).sum()),
            )
        )
    size_histogram = {
        str(int(size)): int(count)
        for size, count in sizes.value_counts().sort_index().items()
    }
    return {
        "domain_column": domain_column,
        "domain_count": int(len(sizes)),
        "cell_count": int(sizes.sum()),
        "minimum_domain_size": int(sizes.min()),
        "median_domain_size": float(sizes.median()),
        "maximum_domain_size": int(sizes.max()),
        "domain_size_histogram": size_histogram,
        "k_shot_support": [item.to_dict() for item in support],
    }


def support_gate(
    audit: dict[str, object],
    *,
    minimum_eligible_domains: int,
) -> dict[str, object]:
    if minimum_eligible_domains < 1:
        raise ValueError("minimum_eligible_domains must be positive")
    results = []
    for support in audit["k_shot_support"]:
        eligible = int(support["eligible_domain_count"])
        results.append(
            {
                "reference_cell_count": int(support["reference_cell_count"]),
                "status": "passed" if eligible >= minimum_eligible_domains else "failed",
                "eligible_domain_count": eligible,
                "minimum_eligible_domains": minimum_eligible_domains,
            }
        )
    return {
        "status": (
            "passed" if all(item["status"] == "passed" for item in results) else "failed"
        ),
        "minimum_eligible_domains": minimum_eligible_domains,
        "by_k": results,
    }
