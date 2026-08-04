"""Step 2 tests: data loading across all three modes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evc.data import CANONICAL_IV_COLUMNS, load_iv_panel, rescale_to_decimal


def test_rescale_auto_detects_percent_for_atm_like_values() -> None:
    series = pd.Series([8.5, 9.2, 10.75, 12.0])
    out = rescale_to_decimal(series)
    assert out.max() < 1.0


def test_rescale_explicit_flag_required_for_small_skew_quotes() -> None:
    """Auto-detection is unsafe for skew quotes (magnitudes can sit
    below the threshold even though they're still percentage points)
    -- this is exactly why data.py always passes assume_percent=True
    explicitly for rr/bf rather than relying on auto-detect."""
    skew = pd.Series([0.85, -0.6, 0.42])
    auto = rescale_to_decimal(skew)  # auto-detect: median(abs) < threshold -> left alone (unsafe)
    assert auto.iloc[0] == 0.85
    explicit = rescale_to_decimal(skew, assume_percent=True)
    assert explicit.iloc[0] == pytest.approx(0.0085)


def test_csv_mode_loads_real_usdjpy_data_in_plausible_ranges() -> None:
    panel = load_iv_panel("csv")
    assert list(panel.columns) == CANONICAL_IV_COLUMNS + ["rr25_1m", "bf25_1m", "spot"]
    # USDJPY 1M ATM vol has plausibly stayed within roughly 3%-60% over
    # 1995-2026 (even including 1998 LTCM / 2008 GFC / 2024 carry
    # unwind spikes) -- a value outside this is a parsing bug, not a
    # real quote.
    iv1m = panel["iv_1m"].dropna()
    assert iv1m.min() > 0.02
    assert iv1m.max() < 0.6
    # spot USDJPY has traded roughly 75-170 over this window
    spot = panel["spot"].dropna()
    assert spot.min() > 50
    assert spot.max() < 200


def test_proxy_mode_warns_and_produces_plausible_realised_vol(capsys) -> None:
    panel = load_iv_panel("proxy")
    captured = capsys.readouterr()
    assert "backward-looking" in captured.out
    iv1m = panel["iv_1m"].dropna()
    assert iv1m.min() > 0.0
    assert iv1m.max() < 1.0


def test_synthetic_mode_wraps_simulator_into_canonical_schema() -> None:
    panel = load_iv_panel("synthetic")
    assert list(panel.columns) == CANONICAL_IV_COLUMNS + ["rr25_1m", "bf25_1m", "spot"]
    assert np.isfinite(panel["spot"]).all()
    assert (panel["iv_1m"].dropna() > 0).all()
