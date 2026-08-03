"""Step 0 smoke test: confirms the package installs and imports, and
that the real data files referenced in evc.config actually exist. Real
functional tests (no-look-ahead, KSG vs analytic Gaussian MI, variance-
clock recovery, BOJ timezone alignment) land in Steps 1-6 alongside the
code they test.
"""

from evc.config import DataPaths, SyntheticConfig


def test_config_imports() -> None:
    cfg = SyntheticConfig()
    assert cfg.event_omega["FOMC"] > 1.0


def test_data_paths_exist() -> None:
    paths = DataPaths()
    for f in (paths.us_events, paths.jn_events, paths.boj_meetings, paths.atm_vol):
        assert f.exists(), f"missing data file: {f}"
