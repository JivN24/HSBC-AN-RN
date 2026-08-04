"""The paper's stochastic vol/variance model (docs/SimpleEventVolModel.pdf,
eqs 1-2, n=2), simulated with a known event calendar and known
per-event-type omega, so evc.validate can check whether each causality
estimator recovers the truth before we trust it on real data.

Two deliberate departures from the bare paper (both required, both
explained where they're used below):

1. Mean reversion. The paper's alpha_i is left as "a function of
   t_i, S_i, sigma_i" with no functional form specified. A constant
   alpha gives variance a pure random walk, which over ~3000 steps
   wanders to implausible levels (or to zero) and would confound
   ordinary non-stationarity with the event effect we're trying to
   isolate. We use Heston-style mean reversion instead:
       dv = kappa*(theta^2 - v)*dt + xi*sqrt(v)*dW
   i.e. alpha_i = kappa*(theta^2 - sigma_i^2), nu_i = xi*sigma_i. This
   keeps sigma_i^2 stationary around theta^2 absent events, which is
   what lets an event study or variance-clock regression attribute a
   *level shift* in variance to the event rather than to drift.

2. A synthetic IV surface. The paper only defines the instantaneous
   process; a term-structure of forward-looking implied vol has to be
   built on top of it to reproduce the anticipation/crush pattern
   central to this project (CLAUDE.md constraint 1). See
   synthesize_iv_surface below.

Full-truncation Euler is used for the variance step (v clamped to >=0
before the sqrt and again after the update) because naive Euler on a
square-root/CIR-type process can go negative with discrete time steps.

Dev note on kappa: eq (2) applies omega MULTIPLICATIVELY to the whole
next state, not just that step's innovation, so a run of overlapping
event types compounds geometrically unless mean reversion pulls the
elevated level back down before the next event. With five event
streams (21-46 day periods) an event fires roughly every 5 business
days on average; a textbook equity-Heston kappa (~3, half-life ~2
months) is far too slow against that and blows variance up to ~1e45
over 3000 steps. kappa=50 (half-life ~3.5 business days,
evc.config.SyntheticConfig) keeps the process stationary and matches
observed FX event-vol decay speed.

Dev note on the shape of synthesize_iv_surface's anticipation effect:
because D_T is a fixed rolling (constant-maturity) window, a single
future event's contribution to the window average is constant
(1/D_T) for every t at which it is inside the window -- so a lone,
isolated event produces a *step up* exactly D_T business days before
it (not a gradual ramp), holds flat, then jumps again at realisation
(since sigma_t itself is now inflated) before decaying. This was
verified directly (tests/test_synthetic.py::test_1m_vol_shows_anticipation_and_crush)
against an isolated single-event-type configuration; the shape still
satisfies "elevated before, crushed after" even though it isn't a
smooth ramp. On the full five-event-type calendar this signature is
confounded by other event types landing inside the same window --
disentangling that is what Step 5's event study and variance-clock
regression are for, not something Step 1 needs to resolve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evc.config import SyntheticConfig

_VARIANCE_FLOOR = 1e-8


def _business_day_index(cfg: SyntheticConfig, start: str = "2015-01-01") -> pd.DatetimeIndex:
    """Arbitrary anchor date -- only the business-day spacing matters
    for a synthetic series, not the calendar date itself."""
    return pd.bdate_range(start=start, periods=cfg.n_steps)


def _generate_event_calendar(cfg: SyntheticConfig, rng: np.random.Generator, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build one boolean column per event type with realistic periodic
    spacing plus jitter.

    The jitter is not cosmetic: a perfectly periodic 0/1 indicator is a
    deterministic function of the row index, so a circular-shift of the
    vol series lines up with it in a way that has nothing to do with
    causality (see evc.causality.surrogates docstring) -- it would make
    the circular-shift null degenerate and inflate any test that simply
    picks up periodicity rather than an event effect. Distinct phase
    offsets per event type additionally avoid every event type landing
    on the same days by construction, which would make individual
    omega_e statistically unidentifiable (they'd only ever be observed
    jointly).
    """
    n = cfg.n_steps
    # Distinct phase offsets so event types don't all fire in lockstep.
    phase_offsets = {
        "FOMC": 5,
        "NFP": 3,
        "US_CPI": 10,
        "BOJ": 25,
        "JP_CPI": 15,
    }
    cols: dict[str, np.ndarray] = {}
    for event_type, period in cfg.event_period_days.items():
        flags = np.zeros(n, dtype=bool)
        offset = phase_offsets.get(event_type, 0)
        k = 0
        while True:
            scheduled = offset + k * period
            jitter = int(rng.integers(-cfg.event_jitter_days, cfg.event_jitter_days + 1))
            day = scheduled + jitter
            if day >= n:
                break
            if day >= 0:
                flags[day] = True
            k += 1
        cols[f"is_{event_type}"] = flags
    return pd.DataFrame(cols, index=index)


def _omega_series(events: pd.DataFrame, cfg: SyntheticConfig) -> np.ndarray:
    """omega_{i} = product of event_omega[type] over event types firing
    on day i, or 1.0 on a day with no events (paper's convention)."""
    omega = np.ones(len(events))
    for event_type, weight in cfg.event_omega.items():
        col = f"is_{event_type}"
        if col in events:
            omega = np.where(events[col].to_numpy(), omega * weight, omega)
    return omega


def simulate_paths(cfg: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate S_i, sigma_i under eqs (1)-(2) with n=2 and Heston-style
    mean reversion, driven by a known, jittered-periodic event calendar.

    Convention: omega_{i+1} (the weight realised in the row labelled
    i+1) is determined by which events fire ON day i+1 -- i.e. the
    event's shock lands in the variance step that carries its date,
    matching "sigma_i+1 = omega_i+1 x F_i(...)" in the paper.
    """
    index = _business_day_index(cfg)
    events = _generate_event_calendar(cfg, rng, index)
    omega = _omega_series(events, cfg)

    n = cfg.n_steps
    dt = cfg.dt
    v = np.empty(n)
    S = np.empty(n)
    v[0] = cfg.sigma0**2
    S[0] = cfg.s0

    sqrt_dt = np.sqrt(dt)
    for i in range(n - 1):
        eps = rng.standard_normal() * sqrt_dt
        eta = rng.standard_normal() * sqrt_dt
        v_plus = max(v[i], 0.0)
        sigma_i = np.sqrt(v_plus)

        drift = cfg.kappa * (cfg.theta**2 - v[i]) * dt
        vol_of_vol = cfg.xi * np.sqrt(v_plus) * (cfg.rho * eps + np.sqrt(1 - cfg.rho**2) * eta)
        v_next = omega[i + 1] * (v[i] + drift + vol_of_vol)
        v[i + 1] = max(v_next, _VARIANCE_FLOOR)

        # eps_i is shared between the price and the vol equations (eq 1
        # uses the same epsilon_i as eq 2's correlation term) -- this
        # is exactly the leverage-effect channel (rho < 0: a down move
        # in spot coincides with a vol-of-vol shock in the same
        # direction as a vol increase).
        S[i + 1] = S[i] + sigma_i * S[i] * eps

    out = pd.DataFrame({"S": S, "sigma": np.sqrt(v), "v": v, "omega": omega}, index=index)
    return pd.concat([out, events], axis=1)


def _tenor_business_days(tenor_calendar_days: dict[str, int]) -> dict[str, int]:
    """Convert calendar-day tenor lengths to business-day counts (5/7
    scaling). Approximate -- see evc.config.TENOR_CALENDAR_DAYS
    docstring: VERIFY against real expiry calendars before treating
    this as exact rather than a normalising approximation."""
    return {tenor: max(1, round(days * 5 / 7)) for tenor, days in tenor_calendar_days.items()}


def synthesize_iv_surface(
    paths: pd.DataFrame, cfg: SyntheticConfig, tenors_days: dict[str, int]
) -> pd.DataFrame:
    """Forward-average the mean-reverting variance forecast over each
    tenor window, weighting each future day by its SCHEDULED omega
    (known in advance -- the event calendar is exogenous and
    pre-announced, CLAUDE.md constraint 2), to produce a term
    structure of implied vol with the same anticipation/crush
    structure real USDJPY IV shows:

        IV^2(t,T) = (1/D_T) * sum_{d=t+1}^{t+D_T} omega_d * E_t[sigma^2_{t+h}]
        E_t[sigma^2_{t+h}] = theta^2 + (sigma_t^2 - theta^2) * exp(-kappa*h*dt)

    Using the *scheduled* omega_d (not a realised future shock) is what
    makes this forward-looking without look-ahead bias: at time t we
    already know the calendar of future events, just as a real trader
    knows the FOMC date months in advance, but we do not use any
    stochastic innovation that has not yet occurred.

    Tenors with t + D_T beyond the simulated horizon are left as NaN
    rather than truncated -- a partial-window average would understate
    long-tenor vol near the end of the sample in a way that has nothing
    to do with the model.
    """
    business_days = _tenor_business_days(tenors_days)
    omega = paths["omega"].to_numpy()
    v = paths["v"].to_numpy()
    theta2 = cfg.theta**2
    kappa = cfg.kappa
    dt = cfg.dt
    n = len(paths)

    out = {}
    for tenor, d_t in business_days.items():
        iv2 = np.full(n, np.nan)
        for t in range(n - d_t):
            h = np.arange(1, d_t + 1)
            expected_v = theta2 + (v[t] - theta2) * np.exp(-kappa * h * dt)
            future_omega = omega[t + 1 : t + d_t + 1]
            iv2[t] = np.mean(future_omega * expected_v)
        out[f"iv_{tenor}"] = np.sqrt(iv2)
    return pd.DataFrame(out, index=paths.index)


def plot_diagnostic(paths: pd.DataFrame, iv_surface: pd.DataFrame, cfg: SyntheticConfig, event_type: str = "FOMC"):
    """Three-panel sanity check required before any real-data work
    starts: (i) overnight vol should spike ON event days -- it's driven
    almost entirely by that day's omega multiplier; (ii) 1M vol should
    rise *before* the event and crush after -- if it doesn't, the
    forward-averaging in synthesize_iv_surface is wrong (e.g. omega
    applied at the wrong offset) and must be fixed before Step 2;
    (iii) a zoomed window around one event instance, because the
    anticipation/crush shape lives on a ~1-2 month scale and is
    invisible by eye on a full multi-year time axis (panels i-ii cover
    the whole sample so nothing looks cropped out of context, but the
    per-event shape has to be seen zoomed in to be checked visually).
    """
    import matplotlib.pyplot as plt

    event_days = paths.index[paths[f"is_{event_type}"]]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    axes[0].plot(paths.index, iv_surface["iv_on"], lw=0.8, color="tab:blue")
    for d in event_days:
        axes[0].axvline(d, color="tab:red", alpha=0.25, lw=0.8)
    axes[0].set_title(f"Overnight vol (spikes on {event_type} days, red lines)")
    axes[0].set_ylabel("iv_on")

    axes[1].plot(paths.index, iv_surface["iv_1m"], lw=1.0, color="tab:green")
    for d in event_days:
        axes[1].axvline(d, color="tab:red", alpha=0.25, lw=0.8)
    axes[1].set_title(f"1M vol, full sample (should rise before, crush after {event_type})")
    axes[1].set_ylabel("iv_1m")
    axes[1].set_xlabel("date")

    event_idx = np.where(paths[f"is_{event_type}"].to_numpy())[0]
    event_idx = event_idx[(event_idx > 35) & (event_idx < len(paths) - 35)]
    mid = event_idx[len(event_idx) // 2]  # a representative instance, not cherry-picked
    lo, hi = mid - 35, mid + 35
    axes[2].plot(paths.index[lo:hi], iv_surface["iv_1m"].iloc[lo:hi], lw=1.4, color="tab:green")
    axes[2].axvline(paths.index[mid], color="tab:red", alpha=0.6, lw=1.2)
    axes[2].set_title(f"1M vol zoomed around one {event_type} instance ({paths.index[mid].date()})")
    axes[2].set_ylabel("iv_1m")
    axes[2].set_xlabel("date")

    fig.tight_layout()
    return fig
