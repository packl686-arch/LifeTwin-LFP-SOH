from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.audits.phase1_adversarial import run_phase1_adversarial_audit
from lifetwin.experiments.calendar_v3_activation_development import (
    run_calendar_v3_activation_development,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv"
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/naumann_calendar_v3_activation_development.json"
)


@pytest.fixture(scope="session")
def observations() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def calendar_v3_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def completed_run(
    observations: pd.DataFrame,
    calendar_v3_config: dict[str, object],
) -> tuple:
    return run_calendar_v3_activation_development(
        observations,
        config=calendar_v3_config,
    )


@pytest.fixture(scope="session")
def phase1_audit(
    observations: pd.DataFrame,
    calendar_v3_config: dict[str, object],
    completed_run: tuple,
):
    return run_phase1_adversarial_audit(
        observations,
        config=calendar_v3_config,
        data_path=DATA_PATH,
        baseline_run=completed_run,
    )
