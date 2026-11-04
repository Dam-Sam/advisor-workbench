"""Turns computed analytics into the fixed fact sheet the AI is allowed to use."""
from __future__ import annotations

from .data import Dataset, DRIFT_BAND
from .metrics import allocation, drift_table, holdings, portfolio_returns, summary_stats

LOOKBACKS = {"3 months": 63, "1 year": 252, "3 years": 755}


def _p(x: float) -> str:
    return f"{x * 100:.1f}%"


def client_facts(ds: Dataset, account_id: str, period: str = "1 year", risk_free: float = 0.025) -> dict:
    acct = ds.accounts.set_index("account_id").loc[account_id]
    h = holdings(ds, account_id)
    alloc = allocation(h, acct.cash)
    drift = drift_table(alloc, acct.model)
    days = min(LOOKBACKS[period], len(ds.prices) - 1)
    r = portfolio_returns(ds, h, acct.cash, days)
    bench = ds.prices["VBAL.TO"].pct_change().iloc[-days:]
    s = summary_stats(r, bench, risk_free)
    top = h.iloc[0]
    return {
        "account_id": account_id,
        "household": acct.household,
        "model": acct.model,
        "as_of": f"{ds.as_of:%B %d, %Y}",
        "period": period,
        "market_value": f"${h.market_value.sum() + acct.cash:,.0f}",
        "portfolio_return": _p(s["total_return"]),
        "benchmark_return": _p(s["benchmark_total_return"]),
        "annualized_volatility": _p(s["annualized_vol"]),
        "max_drawdown": _p(s["max_drawdown"]),
        "equity_actual": _p(alloc["Equity"]),
        "equity_target": _p(drift.at["Equity", "target"]),
        "fixed_income_actual": _p(alloc["Fixed Income"]),
        "fixed_income_target": _p(drift.at["Fixed Income", "target"]),
        "alternative_actual": _p(alloc["Alternative"]),
        "drift_band": f"plus or minus {DRIFT_BAND * 100:.0f} percentage points",
        "out_of_band_sleeves": [ac.lower() for ac in drift.index[drift.out_of_band]],
        "largest_holding": f"{top['name']} ({_p(top['weight'])} of the portfolio)",
        "holdings_count": int(len(h)),
    }
