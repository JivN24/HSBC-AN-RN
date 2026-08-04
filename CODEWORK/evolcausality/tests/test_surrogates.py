"""Step 5 support tests: surrogate generators and BH-FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evc.causality.surrogates import (
    benjamini_hochberg,
    circular_block_bootstrap,
    iaaft_surrogate,
    placebo_event_dates,
)


def test_circular_block_bootstrap_preserves_length_and_values() -> None:
    rng = np.random.default_rng(0)
    series = np.arange(50).astype(float)
    out = circular_block_bootstrap(series, block_len=5, rng=rng)
    assert len(out) == len(series)
    # every value in the resample must have come from the original set
    assert set(out.tolist()) <= set(series.tolist())


def test_placebo_event_dates_shifts_all_dates_by_same_offset() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=10)
    shifted = placebo_event_dates(dates, min_shift=5, max_shift=10, rng=rng)
    diffs = shifted - dates
    assert len(set(diffs)) == 1  # every date moved by the identical offset


def test_iaaft_preserves_amplitude_distribution() -> None:
    rng = np.random.default_rng(0)
    series = rng.standard_normal(200)
    surrogate = iaaft_surrogate(series, rng, n_iter=50)
    assert np.allclose(np.sort(surrogate), np.sort(series), atol=1e-6)


def test_benjamini_hochberg_known_example() -> None:
    # classic textbook example: 5 p-values, alpha=0.05
    pvals = pd.Series([0.01, 0.02, 0.03, 0.04, 0.20])
    out = benjamini_hochberg(pvals, alpha=0.05)
    # BH: q_i = p_i * m / rank_i, then cumulative-min from the top
    # rank1: 0.01*5/1=0.05; rank2: 0.02*5/2=0.05; rank3: 0.03*5/3=0.05
    # rank4: 0.04*5/4=0.05; rank5: 0.20*5/5=0.20
    assert np.allclose(out["qvalue"].to_numpy()[:4], 0.05, atol=1e-9)
    assert out["qvalue"].iloc[4] == 0.20
    assert out["reject"].to_numpy()[:4].all()
    assert not out["reject"].iloc[4]


def test_benjamini_hochberg_preserves_original_index() -> None:
    pvals = pd.Series([0.5, 0.01, 0.3], index=["b", "a", "c"])
    out = benjamini_hochberg(pvals)
    assert list(out.index) == ["b", "a", "c"]
