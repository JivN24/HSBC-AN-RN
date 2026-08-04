"""Step 6: the recovery / power / size validation harness, run against
evc.synthetic ground truth before any real-data result is interpreted.

Recovery: is omega_hat approx omega_true for each estimator? Report
bias and RMSE. (evc.causality.varclock's own module docstring already
documents one recovery finding directly: an upward bias from
same-day multi-event-type co-occurrence in the DENSE 5-event
calendar -- recovery_check exists to quantify exactly that kind of
thing precisely, not just describe it qualitatively.)

Power: sweep N (500..5000) and effect size (omega = 1.1..3.0); find
the N at which each test rejects 80% of the time. Real USDJPY data
gives roughly 80 FOMC and 120 NFP days over 10y -- this harness is
what tells us whether that is enough before interpreting a null
result on real data as "no effect" rather than "no power".

Size: with omega = 1.0, does each test reject at the nominal 5%? Over-
rejection means the surrogate/CI scheme is wrong and must be fixed
before trusting any p-value from it.

The default estimator plugged into power_curve/size_check is the
variance-clock regression (the project's centrepiece, per CLAUDE.md),
tested via an ISOLATED single-event-type synthetic run (an event
generator with only one active type -- mirrors the isolated-FOMC
approach already validated in evc.synthetic's own tests) so that N and
omega can be swept cleanly without the dense-5-event-type confound
documented in evc.causality.varclock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evc.causality.varclock import build_regression_panel, variance_clock_regression
from evc.config import SyntheticConfig, TENOR_CALENDAR_DAYS
from evc.synthetic import simulate_paths, synthesize_iv_surface


def recovery_check(estimator_name: str, omega_hat: dict[str, float], omega_true: dict[str, float]) -> pd.DataFrame:
    """Bias and RMSE of omega_hat against the known omega_true, per
    event type (and pooled)."""
    rows = []
    for event_type, true_val in omega_true.items():
        hat_val = omega_hat.get(event_type, np.nan)
        rows.append({"event_type": event_type, "omega_true": true_val, "omega_hat": hat_val, "bias": hat_val - true_val})
    out = pd.DataFrame(rows).set_index("event_type")
    out["estimator"] = estimator_name
    pooled_rmse = np.sqrt(np.mean(out["bias"] ** 2))
    out.attrs["pooled_rmse"] = pooled_rmse
    return out


def _varclock_rejects_null(n_steps: int, omega: float, n_bootstrap: int, rng: np.random.Generator) -> bool:
    """One trial: simulate an isolated single-event-type synthetic
    series at the given N and omega, fit the variance-clock
    regression, and test whether its bootstrap CI excludes 1 (reject
    H0: omega<=1)."""
    cfg = SyntheticConfig(
        n_steps=n_steps,
        event_omega={"TEST": omega, "NFP": 1.0, "US_CPI": 1.0, "BOJ": 1.0, "JP_CPI": 1.0},
        event_period_days={"TEST": 46, "NFP": 21, "US_CPI": 21, "BOJ": 46, "JP_CPI": 21},
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    sim_rng = np.random.default_rng(cfg.seed)
    paths = simulate_paths(cfg, sim_rng)
    iv = synthesize_iv_surface(paths, cfg, TENOR_CALENDAR_DAYS)
    events = pd.DataFrame({"event_type": "TEST", "snapshot_date": paths.index[paths["is_TEST"]]})

    panel = build_regression_panel(iv, events, TENOR_CALENDAR_DAYS, event_types=["TEST"])
    if panel["n_TEST"].sum() == 0:
        return False
    result = variance_clock_regression(panel, event_types=["TEST"], n_bootstrap=n_bootstrap, rng=rng)
    return bool(result.loc["TEST", "ci_lo"] > 1.0)


def power_curve(
    n_grid: list[int],
    omega_grid: list[float],
    n_trials: int,
    rng: np.random.Generator,
    n_bootstrap: int = 100,
) -> pd.DataFrame:
    """Rejection rate of the variance-clock regression's H0:omega<=1
    test, over a grid of sample sizes N and true effect sizes omega."""
    rows = []
    for n_steps in n_grid:
        for omega in omega_grid:
            rejections = [_varclock_rejects_null(n_steps, omega, n_bootstrap, rng) for _ in range(n_trials)]
            rows.append({"n_steps": n_steps, "omega": omega, "power": np.mean(rejections)})
    return pd.DataFrame(rows)


def size_check(n_trials: int, rng: np.random.Generator, n_steps: int = 3000, n_bootstrap: int = 100) -> float:
    """Rejection rate under omega=1.0 (the null is true) -- should be
    close to the nominal 5% (test uses a 95% CI); a rate far above 5%
    means the bootstrap/CI scheme over-rejects and cannot be trusted."""
    rejections = [_varclock_rejects_null(n_steps, 1.0, n_bootstrap, rng) for _ in range(n_trials)]
    return float(np.mean(rejections))
