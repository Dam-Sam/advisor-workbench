import numpy as np
import pandas as pd
import pytest

from workbench.metrics import (allocation, annualized_return, annualized_vol, beta, drift_table, holdings,
                               max_drawdown, risk_contributions, sharpe_ratio)


def test_annualized_return_compounds():
    r = pd.Series([0.001] * 252)
    assert annualized_return(r) == pytest.approx(1.001 ** 252 - 1)


def test_annualized_vol_scales_by_sqrt_252():
    r = pd.Series([0.01, -0.01] * 126)
    assert annualized_vol(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_max_drawdown_known_path():
    # 100 -> 120 -> 90 -> 110: worst fall is 120 -> 90 = -25%
    r = pd.Series([0.20, 90 / 120 - 1, 110 / 90 - 1])
    assert max_drawdown(r) == pytest.approx(-0.25)


def test_max_drawdown_counts_fall_from_start():
    assert max_drawdown(pd.Series([-0.10, 0.05])) == pytest.approx(-0.10)


def test_sharpe_uses_excess_return():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 1000))
    expected = (annualized_return(r) - 0.02) / annualized_vol(r)
    assert sharpe_ratio(r, 0.02) == pytest.approx(expected)


def test_beta_of_scaled_series():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 500))
    assert beta(2 * b, b) == pytest.approx(2.0)


def test_risk_contributions_sum_to_one():
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=["A", "B", "C"])
    rc = risk_contributions(pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}), rets)
    assert rc.sum() == pytest.approx(1.0)


def test_allocation_sums_to_one_and_looks_through(ds):
    acct = ds.accounts.iloc[0]
    h = holdings(ds, acct.account_id)
    alloc = allocation(h, acct.cash)
    assert alloc.sum() == pytest.approx(1.0)
    assert set(alloc.index) == {"Equity", "Fixed Income", "Alternative", "Cash"}


def test_drift_flags_only_beyond_band():
    alloc = pd.Series({"Equity": 0.61, "Fixed Income": 0.34, "Alternative": 0.04, "Cash": 0.01})
    d = drift_table(alloc, "Balanced")  # target 55 / 40 / 5
    assert bool(d.at["Equity", "out_of_band"]) is True        # +6 pts
    assert bool(d.at["Fixed Income", "out_of_band"]) is True  # -6 pts
    assert bool(d.at["Alternative", "out_of_band"]) is False  # -1 pt
