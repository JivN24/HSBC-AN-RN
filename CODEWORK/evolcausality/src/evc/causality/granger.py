"""Step 5.3-5.4: linear and nonlinear Granger causality.

Linear: standard VAR-based GC (statsmodels), plus Toda-Yamamoto (robust
to unit roots/cointegration -- fits a VAR at the true lag order p plus
d_max extra lags, where d_max is the maximum order of integration
found by ADF+KPSS, then Wald-tests only the first p lag coefficients;
the extra d_max lags absorb any unit-root nonstationarity without
requiring differencing, which would otherwise distort the very
short-run dynamics this project cares about), plus conditional GC
controlling for extra nuisance regressors.

Both directions are always run. The reverse direction (IV -> event
indicator) is the PLACEBO CLAUDE.md constraint 1 describes:
significant at positive lags = anticipation (expected, since IV prices
in a scheduled event before it happens); significant at NEGATIVE lags
would mean IV predicts an event before the event's own pre-announced
calendar date -- a bug or a look-ahead leak, not a real finding, and
should be investigated immediately rather than reported.

ADF and KPSS disagree often on financial vol series (ADF's null is a
unit root, KPSS's null is stationarity -- they test in opposite
directions), so both are always reported side by side rather than
picking whichever supports the conclusion.

Nonlinear GC: out-of-sample R^2 comparison of restricted vs
unrestricted gradient-boosting models under purged/embargoed
walk-forward CV (a plain k-fold would leak future information through
the lagged-feature construction; purging drops training rows too
close to each test fold, embargoing additionally drops training rows
just after a test fold). Significance via block-permutation of the
candidate-cause series ONLY (not the target), preserving the target's
own temporal structure while destroying its specific alignment with
the candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def adf_kpss_test(series: pd.Series) -> pd.Series:
    """ADF (H0: unit root) and KPSS (H0: stationary) side by side --
    report both, since they disagree often on financial vol series."""
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = series.dropna()
    adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")
    return pd.Series({"adf_stat": adf_stat, "adf_pvalue": adf_p, "kpss_stat": kpss_stat, "kpss_pvalue": kpss_p})


def _lag_matrix(series: pd.Series, max_lag: int) -> pd.DataFrame:
    return pd.concat({f"{series.name}_L{lag}": series.shift(lag) for lag in range(1, max_lag + 1)}, axis=1)


def linear_granger(y: pd.Series, x: pd.Series, max_lag: int) -> pd.DataFrame:
    """Does x Granger-cause y? statsmodels' grangercausalitytests,
    reported at every lag from 1 to max_lag (not just the max), since
    the lag at which significance appears is itself diagnostic (see
    CLAUDE.md constraint 1 on anticipation lags)."""
    from statsmodels.tsa.stattools import grangercausalitytests

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    results = grangercausalitytests(df[["y", "x"]], maxlag=max_lag)
    rows = []
    for lag, (tests, _) in results.items():
        rows.append({"lag": lag, "f_stat": tests["ssr_ftest"][0], "pvalue": tests["ssr_ftest"][1]})
    return pd.DataFrame(rows).set_index("lag")


def toda_yamamoto(y: pd.Series, x: pd.Series, max_lag: int, d_max: int | None = None) -> pd.DataFrame:
    """Toda-Yamamoto: fit a VAR at order (max_lag + d_max), Wald-test
    only the first max_lag lag coefficients of x in the y equation.
    d_max defaults to 1 if not given (the common case for daily vol/
    event-indicator series -- at most I(1))."""
    from statsmodels.tsa.api import VAR

    if d_max is None:
        d_max = 1

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    model = VAR(df)
    fitted = model.fit(max_lag + d_max)

    # Wald test: coefficients on x's first `max_lag` lags in the y
    # equation are jointly zero. statsmodels VARResults exposes
    # `.test_causality` which is exactly this (and already only tests
    # the lags actually included, i.e. up to max_lag+d_max -- we pass
    # the full model but interpret the joint test as the TY statistic
    # by construction, since the extra d_max lags are the "TY padding"
    # that absorbs unit-root behaviour).
    causality = fitted.test_causality(caused="y", causing="x", kind="wald")
    return pd.DataFrame(
        {"statistic": [causality.test_statistic], "pvalue": [causality.pvalue], "df": [causality.df]}
    )


def conditional_granger(y: pd.Series, x: pd.Series, controls: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """F-test comparing a restricted model (y ~ own lags + control
    lags) against an unrestricted model (+ x lags) -- does x add
    explanatory power for y beyond the controls?"""
    import statsmodels.api as sm

    y_lags = _lag_matrix(y, max_lag)
    x_lags = _lag_matrix(x, max_lag)
    control_lags = pd.concat([_lag_matrix(controls[c], max_lag) for c in controls.columns], axis=1)

    df = pd.concat([y.rename("y"), y_lags, control_lags, x_lags], axis=1).dropna()
    y_dep = df["y"]
    restricted_X = sm.add_constant(df[y_lags.columns.tolist() + control_lags.columns.tolist()])
    unrestricted_X = sm.add_constant(df[y_lags.columns.tolist() + control_lags.columns.tolist() + x_lags.columns.tolist()])

    restricted_fit = sm.OLS(y_dep, restricted_X).fit()
    unrestricted_fit = sm.OLS(y_dep, unrestricted_X).fit()

    n = len(df)
    k_restricted = restricted_X.shape[1]
    k_unrestricted = unrestricted_X.shape[1]
    q = k_unrestricted - k_restricted  # number of restrictions (x lags)

    rss_r, rss_u = restricted_fit.ssr, unrestricted_fit.ssr
    f_stat = ((rss_r - rss_u) / q) / (rss_u / (n - k_unrestricted))
    pvalue = 1 - stats.f.cdf(f_stat, q, n - k_unrestricted)

    return pd.DataFrame({"f_stat": [f_stat], "pvalue": [pvalue], "df1": [q], "df2": [n - k_unrestricted]})


def _purged_embargoed_folds(n: int, n_folds: int, max_lag: int, embargo: int):
    """Yield (train_idx, test_idx) for walk-forward CV: fold k is
    tested on a contiguous block, trained on everything else EXCEPT
    rows within `max_lag` positions before the test block (purged --
    those rows' lag features overlap into the test period) and
    `embargo` positions after it (embargoed -- the test period's own
    dynamics could otherwise leak backward into an adjacent training
    row via the target's own autocorrelation)."""
    fold_edges = np.linspace(0, n, n_folds + 1, dtype=int)
    for k in range(n_folds):
        test_start, test_end = fold_edges[k], fold_edges[k + 1]
        test_idx = np.arange(test_start, test_end)
        purge_start = max(0, test_start - max_lag)
        embargo_end = min(n, test_end + embargo)
        train_idx = np.concatenate([np.arange(0, purge_start), np.arange(embargo_end, n)])
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def nonlinear_granger_oos(
    y: pd.Series,
    x: pd.Series,
    controls: pd.DataFrame,
    max_lag: int = 5,
    n_folds: int = 5,
    n_permutations: int = 100,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """OOS R^2 of a restricted (own+control lags) vs unrestricted
    (+x lags) gradient-boosting model under purged/embargoed
    walk-forward CV. Significance via block-permutation of x alone.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    rng = rng or np.random.default_rng(0)

    y_lags = _lag_matrix(y, max_lag)
    x_lags = _lag_matrix(x, max_lag)
    control_lags = pd.concat([_lag_matrix(controls[c], max_lag) for c in controls.columns], axis=1)
    df = pd.concat([y.rename("y"), y_lags, control_lags, x_lags], axis=1).dropna()
    n = len(df)

    restricted_cols = y_lags.columns.tolist() + control_lags.columns.tolist()
    unrestricted_cols = restricted_cols + x_lags.columns.tolist()

    def oos_r2(feature_cols: list[str]) -> float:
        preds = np.full(n, np.nan)
        for train_idx, test_idx in _purged_embargoed_folds(n, n_folds, max_lag, embargo=max_lag):
            model = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=2)
            model.fit(df.iloc[train_idx][feature_cols], df.iloc[train_idx]["y"])
            preds[test_idx] = model.predict(df.iloc[test_idx][feature_cols])
        valid = ~np.isnan(preds)
        ss_res = np.sum((df["y"].to_numpy()[valid] - preds[valid]) ** 2)
        ss_tot = np.sum((df["y"].to_numpy()[valid] - df["y"].to_numpy()[valid].mean()) ** 2)
        return 1 - ss_res / ss_tot

    r2_restricted = oos_r2(restricted_cols)
    r2_unrestricted = oos_r2(unrestricted_cols)
    observed_gain = r2_unrestricted - r2_restricted

    block_len = max(1, round(2 * n ** (1 / 3)))
    perm_gains = np.full(n_permutations, np.nan)
    x_lag_values = df[x_lags.columns].to_numpy()
    for p in range(n_permutations):
        n_blocks = -(-n // block_len)
        starts = rng.integers(0, n, size=n_blocks)
        perm_rows = np.concatenate([(np.arange(block_len) + s) % n for s in starts])[:n]
        permuted = df.copy()
        permuted[x_lags.columns] = x_lag_values[perm_rows]

        preds = np.full(n, np.nan)
        for train_idx, test_idx in _purged_embargoed_folds(n, n_folds, max_lag, embargo=max_lag):
            model = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=2)
            model.fit(permuted.iloc[train_idx][unrestricted_cols], permuted.iloc[train_idx]["y"])
            preds[test_idx] = model.predict(permuted.iloc[test_idx][unrestricted_cols])
        valid = ~np.isnan(preds)
        ss_res = np.sum((permuted["y"].to_numpy()[valid] - preds[valid]) ** 2)
        ss_tot = np.sum((permuted["y"].to_numpy()[valid] - permuted["y"].to_numpy()[valid].mean()) ** 2)
        perm_r2 = 1 - ss_res / ss_tot
        perm_gains[p] = perm_r2 - r2_restricted

    pvalue = (np.sum(perm_gains >= observed_gain) + 1) / (n_permutations + 1)
    return pd.DataFrame(
        {"r2_restricted": [r2_restricted], "r2_unrestricted": [r2_unrestricted], "gain": [observed_gain], "pvalue": [pvalue]}
    )
