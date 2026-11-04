"""Reference data, prices and a seeded generator for accounts, trades and custodian files.

Prices come from data/prices.csv (written by scripts/fetch_prices.py) when it exists;
otherwise a seeded synthetic market is generated so the app always runs. Accounts,
trades, dividends and custodian statements are always synthetic: no client data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRICES_CSV = ROOT / "data" / "prices.csv"

BENCHMARK = "VBAL.TO"
MUTUAL_FUND = "DEMO-BAL.F"

# ticker, name, vehicle, asset_class, start_px, beta_eq, beta_rate, idio_vol, drift, div_yield
_SEC_ROWS = [
    ("RY.TO", "Royal Bank of Canada", "Stock", "Equity", 160.0, 0.90, 0.0, 0.16, 0.06, 0.036),
    ("TD.TO", "Toronto-Dominion Bank", "Stock", "Equity", 85.0, 0.95, 0.0, 0.18, 0.04, 0.048),
    ("ENB.TO", "Enbridge", "Stock", "Equity", 58.0, 0.60, 0.3, 0.16, 0.05, 0.062),
    ("CNR.TO", "Canadian National Railway", "Stock", "Equity", 150.0, 0.80, 0.0, 0.17, 0.04, 0.025),
    ("SHOP.TO", "Shopify", "Stock", "Equity", 140.0, 1.60, 0.0, 0.42, 0.10, 0.0),
    ("VCN.TO", "Vanguard FTSE Canada All Cap Index ETF", "ETF", "Equity", 50.0, 1.00, 0.0, 0.02, 0.07, 0.029),
    ("XAW.TO", "iShares Core MSCI All Country World ex Canada Index ETF", "ETF", "Equity", 42.0, 0.85, 0.0, 0.07, 0.08, 0.016),
    ("ZAG.TO", "BMO Aggregate Bond Index ETF", "ETF", "Fixed Income", 13.5, 0.05, 1.0, 0.01, 0.03, 0.034),
    ("XBB.TO", "iShares Core Canadian Universe Bond Index ETF", "ETF", "Fixed Income", 27.0, 0.05, 1.0, 0.01, 0.03, 0.033),
    ("ZRE.TO", "BMO Equal Weight REITs Index ETF", "ETF", "Alternative", 21.0, 0.70, 0.4, 0.12, 0.05, 0.050),
    (MUTUAL_FUND, "Demo Balanced Fund, Series F (fictional)", "Mutual Fund", "Multi-Asset", 12.0, 0, 0, 0, 0, 0.020),
    (BENCHMARK, "Vanguard Balanced ETF Portfolio (benchmark)", "ETF", "Multi-Asset", 31.0, 0, 0, 0, 0, 0.022),
]
SECURITIES = pd.DataFrame(
    _SEC_ROWS,
    columns=["ticker", "name", "vehicle", "asset_class", "start_px", "beta_eq", "beta_rate",
             "idio_vol", "drift", "div_yield"],
)
REAL_TICKERS = [t for t in SECURITIES.ticker if t != MUTUAL_FUND]
HOLDABLE = [t for t in SECURITIES.ticker if t != BENCHMARK]
STOCKS = SECURITIES.loc[SECURITIES.vehicle == "Stock", "ticker"].tolist()

# Multi-asset funds are looked through to their underlying asset classes.
LOOK_THROUGH = {
    MUTUAL_FUND: {"Equity": 0.60, "Fixed Income": 0.40},
    BENCHMARK: {"Equity": 0.60, "Fixed Income": 0.40},
}
ASSET_CLASSES = ["Equity", "Fixed Income", "Alternative", "Cash"]

MODELS = {
    "Conservative": {"Equity": 0.35, "Fixed Income": 0.60, "Alternative": 0.05},
    "Balanced": {"Equity": 0.55, "Fixed Income": 0.40, "Alternative": 0.05},
    "Growth": {"Equity": 0.75, "Fixed Income": 0.20, "Alternative": 0.05},
}
DRIFT_BAND = 0.05  # +/- 5 percentage points from target triggers a rebalance flag

# Trades that have not reached confirmation are still in flight and are not sent to the custodian.
REC_ELIGIBLE_STATUSES = ("CONFIRMED", "SETTLED")
LIFECYCLE = ["EXECUTED", "ALLOCATED", "CONFIRMED", "SETTLED"]


def asset_class_weights(ticker: str) -> dict[str, float]:
    if ticker in LOOK_THROUGH:
        return LOOK_THROUGH[ticker]
    return {SECURITIES.set_index("ticker").at[ticker, "asset_class"]: 1.0}


@dataclass
class Dataset:
    as_of: pd.Timestamp
    prices: pd.DataFrame
    price_source: str
    accounts: pd.DataFrame
    opening_positions: pd.DataFrame
    book_trades: pd.DataFrame
    custodian_trades: pd.DataFrame
    custodian_positions: pd.DataFrame
    book_cash: pd.DataFrame
    custodian_cash: pd.DataFrame
    truth: pd.DataFrame
    securities: pd.DataFrame = field(default_factory=lambda: SECURITIES.copy())


# --------------------------------------------------------------------------- prices
def _derived_returns(rets: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Benchmark and the fictional mutual fund are blends of their sleeves, net of fees."""
    out = rets.copy()
    if BENCHMARK not in out:
        out[BENCHMARK] = (0.6 * (0.5 * rets["VCN.TO"] + 0.5 * rets["XAW.TO"])
                          + 0.4 * rets["ZAG.TO"] - 0.0024 / 252)
    out[MUTUAL_FUND] = (0.6 * (0.4 * rets["VCN.TO"] + 0.6 * rets["XAW.TO"])
                        + 0.4 * rets["XBB.TO"] - 0.009 / 252
                        + rng.normal(0, 0.0008, len(rets)))
    return out


def _synthetic_prices(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    eq_mu, eq_vol = 0.07, 0.15
    f_eq = rng.normal(eq_mu / 252, eq_vol / np.sqrt(252), n)
    f_rate = rng.normal(0.0, 0.05 / np.sqrt(252), n)
    rets = {}
    for s in SECURITIES.itertuples():
        if s.ticker in (BENCHMARK, MUTUAL_FUND):
            continue
        idio = rng.normal(0, s.idio_vol / np.sqrt(252), n)
        alpha = s.drift / 252 - s.beta_eq * eq_mu / 252
        rets[s.ticker] = alpha + s.beta_eq * f_eq + s.beta_rate * f_rate + idio
    rets = pd.DataFrame(rets, index=dates)
    rets.iloc[0] = 0.0
    rets = _derived_returns(rets, rng)
    start = SECURITIES.set_index("ticker")["start_px"]
    return (1 + rets).cumprod() * start[rets.columns]


def load_prices(seed: int = 7, years: int = 3, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, str]:
    """Wide frame of daily prices (index=date, columns=ticker) and a label for its source."""
    rng = np.random.default_rng(seed)
    if PRICES_CSV.exists():
        raw = pd.read_csv(PRICES_CSV, parse_dates=["date"])
        wide = raw.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
        if set(REAL_TICKERS) <= set(wide.columns):
            wide = wide[REAL_TICKERS].ffill().dropna()
            if as_of is not None:
                wide = wide.loc[:as_of]
            wide = wide.iloc[-years * 252:]
            if len(wide) > 300:
                rets = _derived_returns(wide.pct_change().fillna(0.0).drop(columns=[BENCHMARK]), rng)
                rets[BENCHMARK] = wide[BENCHMARK].pct_change().fillna(0.0)
                mf = SECURITIES.set_index("ticker").at[MUTUAL_FUND, "start_px"] * (1 + rets[MUTUAL_FUND]).cumprod()
                wide[MUTUAL_FUND] = mf
                return wide, "Yahoo Finance adjusted closes (mutual fund NAV simulated)"
    end = as_of if as_of is not None else pd.Timestamp.today().normalize() - pd.offsets.BDay(1)
    dates = pd.bdate_range(end=end, periods=years * 252)
    return _synthetic_prices(dates, rng), "Synthetic seeded prices (run scripts/fetch_prices.py for real data)"


# --------------------------------------------------------------------------- generator
def _qty(ticker: str, q: float) -> float:
    return round(q, 3) if ticker == MUTUAL_FUND else float(max(1, round(q)))


def _next_bday(d: pd.Timestamp, n: int = 1) -> pd.Timestamp:
    return (d + pd.offsets.BDay(n)).normalize()


def generate_dataset(seed: int = 42, n_accounts: int = 50, as_of: pd.Timestamp | None = None,
                     inject_breaks: bool = True) -> Dataset:
    rng = np.random.default_rng(seed)
    prices, source = load_prices(seed=seed + 1, as_of=as_of)
    as_of = prices.index[-1]
    inception = prices.index[max(0, len(prices) - 253)]
    px_inc = prices.loc[inception]

    # ---- accounts and opening positions (built a year ago; prices have moved since, so they drift)
    acct_rows, pos_rows = [], []
    for i in range(n_accounts):
        acct = f"ACC-{1001 + i}"
        model = str(rng.choice(list(MODELS), p=[0.30, 0.45, 0.25]))
        tgt = dict(MODELS[model])
        value = float(np.round(np.exp(rng.uniform(np.log(250_000), np.log(5_000_000))), -3))
        w: dict[str, float] = {}
        eq, fi, alt = tgt["Equity"], tgt["Fixed Income"], tgt["Alternative"]
        if rng.random() < 0.3:
            w[MUTUAL_FUND] = 0.20
            eq, fi = eq - 0.12, fi - 0.08
        picks = list(rng.choice(STOCKS, size=int(rng.integers(0, 4)), replace=False))
        stock_share = 0.35 if picks else 0.0
        for s in picks:
            w[s] = eq * stock_share / len(picks)
        w["VCN.TO"] = eq * (1 - stock_share) * 0.45
        w["XAW.TO"] = eq * (1 - stock_share) * 0.55
        r = rng.random()
        bonds = ["ZAG.TO"] if r < 0.4 else ["XBB.TO"] if r < 0.7 else ["ZAG.TO", "XBB.TO"]
        for b in bonds:
            w[b] = fi / len(bonds)
        w["ZRE.TO"] = alt
        w = {k: v * float(np.exp(rng.normal(0, 0.18))) for k, v in w.items()}
        cash_w = float(rng.uniform(0.005, 0.03))
        total = sum(w.values())
        for t, wt in w.items():
            q = _qty(t, wt / total * (1 - cash_w) * value / px_inc[t])
            pos_rows.append((acct, t, q))
        acct_rows.append((acct, f"Household {i + 1:02d}", str(rng.choice(["Advisor A", "Advisor B", "Advisor C"])),
                          model, round(cash_w * value, 2)))
    accounts = pd.DataFrame(acct_rows, columns=["account_id", "household", "advisor", "model", "cash"])
    opening = pd.DataFrame(pos_rows, columns=["account_id", "ticker", "quantity"])

    # ---- trades over the last five business days, each walked through the lifecycle
    window = prices.index[-5:]
    held = opening.groupby("account_id")["ticker"].apply(list).to_dict()
    running = {(a, t): q for a, t, q in opening.itertuples(index=False)}
    value_by_acct = (opening.assign(v=opening.quantity * prices.iloc[-1][opening.ticker].to_numpy())
                     .groupby("account_id")["v"].sum().to_dict())
    trade_rows, seq = [], 0
    trading_accts = rng.choice(accounts.account_id, size=int(n_accounts * 0.45), replace=False)
    for acct in sorted(trading_accts):
        for _ in range(int(rng.integers(1, 5))):
            d = pd.Timestamp(rng.choice(window))
            side = "SELL" if rng.random() < 0.45 else "BUY"
            ticker = str(rng.choice(held[acct]) if side == "SELL" or rng.random() < 0.7 else rng.choice(HOLDABLE))
            px = round(float(prices.at[d, ticker]) * (1 + rng.normal(0, 0.001)), 4)
            q = _qty(ticker, value_by_acct[acct] * rng.uniform(0.005, 0.04) / px)
            if side == "SELL":
                q = min(q, _qty(ticker, running.get((acct, ticker), 0) * 0.5))
                if running.get((acct, ticker), 0) <= 0 or q <= 0:
                    continue
                running[(acct, ticker)] -= q
            else:
                running[(acct, ticker)] = running.get((acct, ticker), 0) + q
            seq += 1
            order_ts = d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=float(rng.uniform(0, 360)))
            exec_ts = order_ts + pd.Timedelta(minutes=float(rng.uniform(1, 30)))
            alloc_ts = exec_ts + pd.Timedelta(minutes=float(rng.uniform(5, 90)))
            conf_ts = alloc_ts + pd.Timedelta(minutes=float(rng.uniform(10, 120)))
            settle = _next_bday(d)  # Canadian equities and ETFs settle T+1
            if d < as_of:
                status = "SETTLED" if settle <= as_of else "CONFIRMED"
            else:
                status = str(rng.choice(["EXECUTED", "ALLOCATED", "CONFIRMED"], p=[0.2, 0.3, 0.5]))
            stage = LIFECYCLE.index(status)
            trade_rows.append(dict(
                trade_id=f"T{d:%y%m%d}-{seq:05d}", account_id=acct, ticker=ticker, side=side,
                quantity=q, price=px, trade_date=d, settle_date=settle, status=status,
                order_ts=order_ts, executed_ts=exec_ts,
                allocated_ts=alloc_ts if stage >= 1 else pd.NaT,
                confirmed_ts=conf_ts if stage >= 2 else pd.NaT,
            ))
    book_trades = pd.DataFrame(trade_rows)

    # ---- custodian copy of everything that reached confirmation
    eligible = book_trades[book_trades.status.isin(REC_ELIGIBLE_STATUSES)].reset_index(drop=True)
    cust = pd.DataFrame(dict(
        cust_trade_id=[f"CST-{80001 + i}" for i in range(len(eligible))],
        client_ref=eligible.trade_id, account_id=eligible.account_id, ticker=eligible.ticker,
        side=eligible.side, quantity=eligible.quantity, price=eligible.price,
        trade_date=eligible.trade_date, settle_date=eligible.settle_date,
    ))
    cust["client_ref"] = cust["client_ref"].astype(object)
    truth = []

    if inject_breaks:
        order = list(rng.permutation(len(cust)))

        def take(n: int) -> list[int]:
            return [order.pop() for _ in range(min(n, len(order)))]

        drop = take(int(rng.integers(1, 3)))
        truth += [("MISSING_AT_CUSTODIAN", cust.at[i, "client_ref"]) for i in drop]
        for i in take(int(rng.integers(1, 4))):
            q = cust.at[i, "quantity"]
            delta = max(1.0, round(q * rng.uniform(0.05, 0.2))) * rng.choice([-1, 1])
            cust.at[i, "quantity"] = _qty(cust.at[i, "ticker"], max(1.0, q + delta)) if q + delta > 0 else q * 10
            truth.append(("QTY_MISMATCH", cust.at[i, "client_ref"]))
        for i in take(int(rng.integers(1, 4))):
            cust.at[i, "price"] = round(cust.at[i, "price"] * (1 + rng.choice([-1, 1]) * rng.uniform(0.006, 0.02)), 4)
            truth.append(("PRICE_MISMATCH", cust.at[i, "client_ref"]))
        for i in take(int(rng.integers(1, 3))):
            cust.at[i, "settle_date"] = _next_bday(cust.at[i, "settle_date"])
            truth.append(("SETTLE_DATE_MISMATCH", cust.at[i, "client_ref"]))
        cust = cust.drop(index=drop).reset_index(drop=True)

        extra = []
        for k in range(int(rng.integers(1, 3))):
            acct = str(rng.choice(accounts.account_id))
            ticker = str(rng.choice(held[acct]))
            d = pd.Timestamp(rng.choice(window[:-1]))
            px = round(float(prices.at[d, ticker]), 4)
            cid = f"CST-{90001 + k}"
            extra.append(dict(cust_trade_id=cid, client_ref=None, account_id=acct, ticker=ticker, side="BUY",
                              quantity=_qty(ticker, 20_000 / px), price=px, trade_date=d,
                              settle_date=_next_bday(d)))
            truth.append(("MISSING_IN_BOOK", cid))
        cust = pd.concat([cust, pd.DataFrame(extra)], ignore_index=True)

    # ---- custodian position statement: opening + its own settled trades
    def settled_positions(trades: pd.DataFrame) -> pd.DataFrame:
        s = trades[trades.settle_date <= as_of]
        signed = s.assign(quantity=np.where(s.side == "BUY", s.quantity, -s.quantity))[
            ["account_id", "ticker", "quantity"]]
        out = pd.concat([opening, signed]).groupby(["account_id", "ticker"], as_index=False)["quantity"].sum()
        out["quantity"] = out["quantity"].round(3)
        return out

    cust_pos = settled_positions(cust)
    if inject_breaks:
        traded = set(zip(book_trades.account_id, book_trades.ticker)) | set(zip(cust.account_id, cust.ticker))
        quiet = cust_pos[[(a, t) not in traded for a, t in zip(cust_pos.account_id, cust_pos.ticker)]]
        i = quiet.index[int(rng.integers(0, len(quiet)))]
        q = cust_pos.at[i, "quantity"]
        cust_pos.at[i, "quantity"] = _qty(cust_pos.at[i, "ticker"], q + max(1.0, q * rng.uniform(0.05, 0.15)))
        truth.append(("UNEXPLAINED_POSITION", f"{cust_pos.at[i, 'account_id']}|{cust_pos.at[i, 'ticker']}"))

    # ---- dividends paid in the window (record-date quantity = opening quantity)
    sec = SECURITIES.set_index("ticker")
    payers = [t for t in HOLDABLE if sec.at[t, "div_yield"] > 0 and t != MUTUAL_FUND]
    div_rows = []
    for t in rng.choice(payers, size=3, replace=False):
        pay = pd.Timestamp(rng.choice(window[:-1]))
        dps = round(float(prices.at[pay, t]) * sec.at[t, "div_yield"] / 4, 4)
        for a, q in opening.loc[opening.ticker == t, ["account_id", "quantity"]].itertuples(index=False):
            div_rows.append(dict(account_id=a, ticker=str(t), pay_date=pay, activity_type="DIVIDEND",
                                 amount=round(q * dps, 2)))
    cust_cash = pd.DataFrame(div_rows)
    book_cash = cust_cash.copy()
    if inject_breaks and len(book_cash) >= 3:
        idx = list(rng.permutation(len(book_cash)))
        unbooked = [idx.pop() for _ in range(int(rng.integers(1, 3)))]
        mism = idx.pop()
        book_cash.at[mism, "amount"] = round(book_cash.at[mism, "amount"] * (1 - rng.uniform(0.03, 0.10)), 2)
        for j in [*unbooked, mism]:
            r_ = book_cash.loc[j]
            kind = "DIVIDEND_AMOUNT_MISMATCH" if j == mism else "UNBOOKED_DIVIDEND"
            truth.append((kind, f"{r_.account_id}|{r_.ticker}|{r_.pay_date:%Y-%m-%d}"))
        book_cash = book_cash.drop(index=unbooked).reset_index(drop=True)

    return Dataset(
        as_of=as_of, prices=prices, price_source=source, accounts=accounts,
        opening_positions=opening, book_trades=book_trades, custodian_trades=cust,
        custodian_positions=cust_pos, book_cash=book_cash, custodian_cash=cust_cash,
        truth=pd.DataFrame(truth, columns=["break_type", "ref"]),
    )


def book_positions(ds: Dataset) -> pd.DataFrame:
    """Current book positions: opening plus settled book trades."""
    s = ds.book_trades[ds.book_trades.settle_date <= ds.as_of]
    signed = s.assign(quantity=np.where(s.side == "BUY", s.quantity, -s.quantity))[["account_id", "ticker", "quantity"]]
    out = pd.concat([ds.opening_positions, signed]).groupby(["account_id", "ticker"], as_index=False)["quantity"].sum()
    return out[out.quantity.abs() > 1e-9]
