"""Shared plotting helpers. Figures saved to results/figures/ at 150dpi
with descriptive filenames (evc.config.FIGURES_DIR).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from evc.config import FIGURES_DIR


def savefig(fig: plt.Figure, name: str, subdir: Path = FIGURES_DIR) -> Path:
    """Save fig to subdir/name.png at 150dpi, creating subdir if needed."""
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out
