# evolcausality

Empirical causality analysis of scheduled macro events (FOMC, NFP, US
CPI, BOJ, JP CPI) on USDJPY implied volatility. Implements an empirical
version of the exercise in `docs/SimpleEventVolModel.pdf` (Too, 2026).

**Phase 1 scope: causality assessment only.** Establish whether there
is a detectable, quantifiable causal link between scheduled events and
IV, and estimate the paper's event weight omega. No forecasting model
until Phase 1's results justify it.

## Three design constraints — read before touching evc.causality

1. **Implied vol is forward-looking, so the causal arrow appears to run
   backwards.** IV prices E^Q[integral of sigma^2] over the tenor. A
   scheduled event is priced in before it happens (IV rises into
   FOMC/NFP/BOJ, crushes after). A naive Granger test of "event -> IV"
   at positive lags looks weak; "IV -> event" looks strong. This is why
   event features always include days-to-*next*-event, not only
   days-since, and why the reverse-direction GC test is run
   deliberately as a placebo: significant at positive lags = expected
   anticipation; significant at negative lags = a bug or look-ahead
   leak, investigate immediately.

2. **Scheduled event timing is exogenous and pre-announced.** FOMC/BOJ
   dates are set years ahead — nothing in the vol surface causes the
   date of the next FOMC. This kills reverse causality, but also means
   "does the event date help forecast IV" is a weak framing (the date
   is a deterministic calendar function). The honest object of study is
   the event-conditional distribution of the vol residual — the
   structural parameter is exactly the paper's omega. The variance
   clock regression (`evc.causality.varclock`) is therefore the
   centrepiece; Granger/TE/TDMI are supporting evidence.

3. **Overnight vol is the cleanest instrument for omega.** sigma^2_ON on
   a day covering an event essentially *is* omega. Always pull
   `USDJPYVON` (the '1D' sheet in the ATM vol workbook) alongside longer
   tenors. Short tenors (ON, 1W, 2W) also break the collinearity in the
   variance-clock regression: a 3M window contains ~3x the events of a
   1M window, so without short tenors the individual omega_e are only
   weakly identified.

## Core math

Take n=2 (stochastic variance) in the paper, log both sides of eq (2):

    log sigma^2_{i+1} = log omega_{i+1} + log F_i(sigma_i, S_i)

The event weight is an **additive shift in log-variance**. Work in
log sigma^2 = 2*log(IV) throughout, never in vol levels. Empirical
target: E[Delta log sigma^2 | event]; omega_hat = exp of that.

Variance clock identity (identifies omega):

    IV^2(t,T) * D_T = sigma_b^2(t) * D_T + sigma_b^2(t) * sum_e (omega_e - 1) * n_e(t,T)

D_T = business days to expiry, n_e(t,T) = count of type-e events
strictly after t within (t, t+D_T]. Linear in event counts, so
omega_hat_e = 1 + beta_hat_e / sigma_hat_b^2. Two identification
problems: (a) sigma_b^2 is latent/time-varying — normalise by
IV_1Y^2*D_T rather than a parameter per day; (b) cross-tenor
collinearity of event counts — fixed by including short tenors
(constraint 3).

## Build order (phase-gated)

Steps 0-6, defined in the original task spec, each ending with a stop
for explicit go-ahead before continuing. Do not run the whole pipeline
end to end without checking in between steps.

Status: **Step 0 complete** (this skeleton). Steps 1-6 not started.

## Engineering rules

- No look-ahead, ever. Any rolling/expanding stat uses only past data.
  `tests/test_no_lookahead.py` (Step 4) plants a future spike in a
  synthetic series and asserts no feature reacts before it.
- Every stochastic function takes an explicit `rng` (`numpy.random.Generator`);
  results must be reproducible from a seed.
- Information-theoretic estimators (KSG MI, transfer entropy) are
  implemented from scratch in `evc.causality.infotheory` — no
  pyinform/IDTxl/JIDT. The bias corrections and neighbour conventions
  matter at our sample sizes and those packages hide them.
- No invented FRED release IDs, no invented Bloomberg field names, no
  ticker not confirmed against the actual data files. Constants that
  are real-world conventions that could be wrong or stale (release
  times, changeover dates) are marked `VERIFY` in `evc/config.py` and
  must be confirmed by the user, not silently trusted.
- Ad-hoc/unscheduled events (MoF FX interventions 2022/2024, emergency
  BOJ meetings) are tagged `scheduled=False` in `evc.events` and
  analysed separately — constraint 2's exogeneity argument does not
  apply to them.

## Data files (real, on disk — see `evc/config.py::DataPaths`)

- `DATA/US_events_merged_1997-2026.xlsx`, `DATA/JN_events_1997-2026_FINAL.xlsx`:
  Bloomberg economic calendar exports, columns
  `[Date Time, Country Code, Event, Period, Survey, Actual, Prior, Revised, Relevance, Ticker]`.
- `DATA/boj_monetary_policy_meetings_2015_2026.csv`: ground truth for
  JST->UTC alignment (`Time (JST)` and `Time (UTC)` columns) — cross-
  check the JN events file's `Date Time` against this rather than
  trusting it blindly.
- `DATA/ATM_Volatility_-_all_currencies_diff_tenors.xlsx`: sheets
  `{1D,1W,2W,1M,2M,3M,6M,1Y}` (1D = overnight/VON), wide layout with
  repeated Date/CCY column pairs, row 0 holding the Bloomberg ticker
  string, values in **vol points (percent)**, not decimals.
  `DATA/25R - all currencies diff tenors.xlsx` (risk reversal, tenors
  {1M,3M,6M,1Y}) and `DATA/25B - all currencies diff tenors.xlsx`
  (butterfly, tenors {1W,1M,3M,6M}) share this layout.

## Environment

`evolcausality/.venv` (Python 3.14), installed via
`pip install -e .[dev,fred,proxy]` against `pyproject.toml`. Run tests
with `pytest` from the `evolcausality/` directory with the venv active.
