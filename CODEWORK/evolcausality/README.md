# evolcausality

Causal analysis of scheduled macro events on USDJPY implied volatility,
implementing an empirical version of the exercise in
`docs/SimpleEventVolModel.pdf`. See `CLAUDE.md` for the modelling
constraints and build order — this file is just setup/usage.

## Setup

```bash
cd evolcausality
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,fred,proxy]"
pytest
```

## Layout

```
docs/                       the paper
src/evc/
  config.py                 all magic numbers / paths / VERIFY constants
  synthetic.py               Step 1: paper's model simulator
  data.py                    Step 2: IV loaders (synthetic / csv / proxy)
  events.py                  Step 2/3: event calendar + timezone alignment
  features.py                Step 4: vol + event feature construction
  causality/
    eventstudy.py             Step 5.1
    varclock.py                Step 5.2 (centrepiece — estimates omega)
    granger.py                  Step 5.3/5.4 (linear + nonlinear GC)
    infotheory.py                Step 5.5/5.6 (TDMI, TE, from scratch)
    surrogates.py                 nulls + BH-FDR (Step 5.7)
  validate.py                Step 6: recovery / power / size harness
  plotting.py                figure-saving helper
tests/
notebooks/01_causality.ipynb  thin: imports + narrates, no logic
results/{figures,tables}/
```

## Status

Step 0 (this skeleton) complete. Steps 1-6 proceed one at a time with
a check-in after each — see `CLAUDE.md`.
