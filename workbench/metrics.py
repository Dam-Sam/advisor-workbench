"""Portfolio analytics: holdings, look-through allocation, drift, and risk/return metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ASSET_CLASSES, DRIFT_BAND, MODELS, SECURITIES, Dataset, asset_class_weights, book_positions

TRADING_DAYS = 252


# --------------------------------------------------------------------------- return / risk
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def annualized_return(r: pd.Series) -> float:
    n = len(r)
    return float((1 + r).prod() ** (TRADING_DAYS / n) - 1) if n else 0.0


def annualized_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0


def sharpe_ratio(r: pd.Series, risk_free: float) -> float:
    vol = annualized_vol(r)
    return (annualized_return(r) - risk_free) / vol if vol > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the growth path, as a negative fraction."""
    wealth = (1 + r).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return float((wealth / peak - 1).min()) if len(r) else 0.0


def tracking_error(r: pd.Series, bench: pd.Series) -> float:
    return annualized_vol(r - bench)


def beta(r: pd.Series, bench: pd.Series) -> float:
    v = bench.var(ddof=1)
    return float(r.cov(bench) / v) if v > 0 else float("nan")


def risk_contributions(weights: pd.Series, rets: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance from each holding: w_i (Σw)_i / wᵀΣw. Sums to 1."""
    w = weights.reindex(rets.columns).fillna(0.0).to_numpy()
    cov = rets.cov().to_numpy() * TRADING_DAYS
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=rets.columns)
    return pd.Series(w * (cov @ w) / port_var, index=rets.columns)


# --------------------------------------------------------------------------- holdings
def holdings(ds: Dataset, account_id: str) -> pd.DataFrame:
    pos = book_positions(ds)
    pos = pos[pos.account_id == account_id].copy()
    px = ds.prices.iloc[-1]
    sec = SECURITIES.set_index("ticker")
    pos["name"] = pos.ticker.map(sec["name"])
    pos["vehicle"] = pos.ticker.map(sec["vehicle"])
    pos["asset_class"] = pos.ticker.map(sec["asset_class"])
    pos["price"] = pos.ticker.map(px)
    pos["market_value"] = pos.quantity * pos.price
    cash = float(ds.accounts.set_index("account_id").at[account_id, "cash"])
    total = pos.market_value.sum() + cash
    pos["weight"] = pos.market_value / total
    return pos.sort_values("market_value", ascending=False).reset_index(drop=True)


def allocation(h: pd.DataFrame, cash: float) -> pd.Series:
    """Look-through asset-class weights, including cash."""
    total = h.market_value.sum() + cash
    alloc = dict.fromkeys(ASSET_CLASSES, 0.0)
    for t, mv in zip(h.ticker, h.market_value):
        for ac, w in asset_class_weights(t).items():
            alloc[ac] += mv * w
    alloc["Cash"] += cash
    return pd.Series(alloc) / total


def drift_table(alloc: pd.Series, model: str) -> pd.DataFrame:
    target = pd.Series({**dict.fromkeys(ASSET_CLASSES, 0.0), **MODELS[model]})
    df = pd.DataFrame({"actual": alloc, "target": target})
    df["drift"] = df.actual - df.target
    df["out_of_band"] = (df.drift.abs() > DRIFT_BAND) & (df.index != "Cash")
    return df


def book_drift(ds: Dataset) -> pd.DataFrame:
    """Every account's largest drift: the list an advisor would rebalance from."""
    rows = []
    accts = ds.accounts.set_index("account_id")
    for a in accts.index:
        h = holdings(ds, a)
        d = drift_table(allocation(h, accts.at[a, "cash"]), accts.at[a, "model"]).drop(index="Cash")
        worst = d.drift.abs().idxmax()
        rows.append(dict(account_id=a, household=accts.at[a, "household"], advisor=accts.at[a, "advisor"],
                         model=accts.at[a, "model"], market_value=h.market_value.sum() + accts.at[a, "cash"],
                         equity=d.at["Equity", "actual"], fixed_income=d.at["Fixed Income", "actual"],
                         alternative=d.at["Alternative", "actual"], worst_sleeve=worst,
                         worst_drift=d.at[worst, "drift"], needs_rebalance=bool(d.out_of_band.any())))
    return pd.DataFrame(rows).sort_values("worst_drift", key=lambda s: -s.abs()).reset_index(drop=True)


def portfolio_returns(ds: Dataset, h: pd.DataFrame, cash: float, lookback_days: int) -> pd.Series:
    """Daily returns of today's holdings held at constant weight over the lookback (ex-post risk view)."""
    rets = daily_returns(ds.prices).iloc[-lookback_days:]
    total = h.market_value.sum() + cash
    w = (h.set_index("ticker").market_value / total).reindex(rets.columns).fillna(0.0)
    return rets @ w  # cash earns 0 in this view


def summary_stats(r: pd.Series, bench: pd.Series, risk_free: float) -> dict:
    return {
        "total_return": float((1 + r).prod() - 1),
        "annualized_return": annualized_return(r),
        "annualized_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r, risk_free),
        "max_drawdown": max_drawdown(r),
        "tracking_error": tracking_error(r, bench),
        "beta": beta(r, bench),
        "benchmark_total_return": float((1 + bench).prod() - 1),
        "benchmark_annualized_vol": annualized_vol(bench),
    }
