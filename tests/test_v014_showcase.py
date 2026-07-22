from __future__ import annotations

from pathlib import Path
import uuid

import matplotlib.image as mpimg
import numpy as np

from showcase.analyze_v014_synthetic_identifiability import build_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    PROJECT_ROOT
    / "showcase/evidence_v014/synthetic_long_horizon_identifiability_v1"
)


def test_v014_synthetic_failure_figure_is_deterministic_nonempty_png() -> None:
    scratch = PROJECT_ROOT / "artifacts/test-scratch"
    token = uuid.uuid4().hex
    first = scratch / f"v014-synthetic-{token}-first.png"
    second = scratch / f"v014-synthetic-{token}-second.png"
    try:
        build_figure(EVIDENCE_ROOT, first)
        build_figure(EVIDENCE_ROOT, second)
        payload = first.read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 100_000
        assert payload == second.read_bytes()

        image = mpimg.imread(first)
        assert image.shape[0] >= 1_600
        assert image.shape[1] >= 2_800
        assert float(np.std(image[:, :, :3])) > 0.05
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)
