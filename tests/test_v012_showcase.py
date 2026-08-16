from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pytest

mpimg = pytest.importorskip("matplotlib.image")

from showcase.analyze_v012_robustness import build_figure  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v012_robustness_figure_is_nonempty_png() -> None:
    output_path = (
        PROJECT_ROOT / "artifacts/test-scratch" / f"v012-figure-{uuid.uuid4().hex}.png"
    )
    try:
        output = build_figure(
            PROJECT_ROOT / "showcase/evidence_v012",
            output_path,
        )
        payload = output.read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 50_000
        image = mpimg.imread(output)
        assert image.shape[0] >= 800
        assert image.shape[1] >= 2_000
        assert float(np.std(image[:, :, :3])) > 0.05
    finally:
        output_path.unlink(missing_ok=True)
