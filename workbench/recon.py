"""Daily reconciliation: runs the SQL in sql/ on DuckDB and builds one exception queue."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .data import Dataset

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

TOLERANCES = {
    "qty_tol": 1e-6,        # quantities must match exactly (3-decimal units for funds)
    "price_abs_tol": 0.01,  # a cent...
    "price_rel_tol": 0.005, # ...or 0.5% of price, whichever is larger
    "cash_tol": 0.01,
}

BREAK_LABELS = {
    "MISSING_AT_CUSTODIAN": "Trade in our book, not at custodian",
    "MISSING_IN_BOOK": "Trade at custodian, not in our book",
    "QTY_MISMATCH": "Quantity or side differs",
    "PRICE_MISMATCH": "Price differs beyond tolerance",
    "SETTLE_DATE_MISMATCH": "Settlement date differs",
    "UNEXPLAINED_POSITION": "Position differs with no trade-level cause",
    "UNBOOKED_DIVIDEND": "Dividend received, not booked",
    "DIVIDEND_AMOUNT_MISMATCH": "Dividend amount differs",
    "DIVIDEND_NOT_AT_CUSTODIAN": "Dividend booked, not received",
}


@dataclass
class ReconResult:
    trade_rec: pd.DataFrame
    position_rec: pd.DataFrame
    cash_rec: pd.DataFrame
    queue: pd.DataFrame
    summary: dict


def _sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def _bdays_between(start: pd.Series, end: pd.Timestamp) -> np.ndarray:
    s = pd.to_datetime(start).to_numpy().astype("datetime64[D]")
    return np.busday_count(s, np.datetime64(end.date()))


def _severity(impact: float, age: int) -> str:
    if impact >= 25_000 or age >= 3:
        return "High"
    if impact >= 2_500 or age >= 2:
        return "Medium"
    return "Low"


def run_reconciliation(ds: Dataset, tolerances: dict | None = None) -> ReconResult:
    tol = {**TOLERANCES, **(tolerances or {})}
    con = duckdb.connect()
    for name in ("book_trades", "custodian_trades", "opening_positions", "custodian_positions",
                 "book_cash", "custodian_cash"):
        con.register(name, getattr(ds, name))

    trade_rec = con.execute(_sql("trade_rec.sql"), {k: tol[k] for k in ("qty_tol", "price_abs_tol", "price_rel_tol")}).df()
    con.register("trade_rec", trade_rec)
    position_rec = con.execute(_sql("position_rec.sql"), {"as_of": ds.as_of.date(), "qty_tol": tol["qty_tol"]}).df()
    cash_rec = con.execute(_sql("cash_rec.sql"), {"cash_tol": tol["cash_tol"]}).df()
    con.close()

    # ---- one exception queue across trade, position and cash breaks
    tb = trade_rec[trade_rec.rec_status != "MATCHED"]
    trades_q = pd.DataFrame({
        "category": "Trade", "break_type": tb.rec_status, "account_id": tb.account_id, "ticker": tb.ticker,
        "ref": tb.trade_ref, "event_date": tb.trade_date, "impact_cad": tb.impact_cad,
        "book_value": [_trade_side(r, "book") for r in tb.itertuples()],
        "custodian_value": [_trade_side(r, "cust") for r in tb.itertuples()],
    })
    pb = position_rec[~position_rec.explained]
    last_px = ds.prices.iloc[-1]
    pos_q = pd.DataFrame({
        "category": "Position", "break_type": "UNEXPLAINED_POSITION", "account_id": pb.account_id,
        "ticker": pb.ticker, "ref": pb.account_id + "|" + pb.ticker, "event_date": ds.as_of,
        "impact_cad": (pb.qty_diff.abs() * last_px[pb.ticker].to_numpy()).round(2),
        "book_value": [f"{q:,.3f} units" for q in pb.book_qty],
        "custodian_value": [f"{q:,.3f} units" for q in pb.cust_qty],
    })
    cb = cash_rec[cash_rec.rec_status != "MATCHED"]
    cash_q = pd.DataFrame({
        "category": "Cash", "break_type": cb.rec_status, "account_id": cb.account_id, "ticker": cb.ticker,
        "ref": cb.account_id + "|" + cb.ticker + "|" + pd.to_datetime(cb.pay_date).dt.strftime("%Y-%m-%d"),
        "event_date": cb.pay_date, "impact_cad": cb.impact_cad,
        "book_value": [_money(v) for v in cb.book_amount],
        "custodian_value": [_money(v) for v in cb.cust_amount],
    })
    queue = pd.concat([trades_q, pos_q, cash_q], ignore_index=True)
    queue["event_date"] = pd.to_datetime(queue["event_date"])
    queue["age_bdays"] = _bdays_between(queue.event_date, ds.as_of).astype(int)
    queue["severity"] = [_severity(i, a) for i, a in zip(queue.impact_cad, queue.age_bdays)]
    queue["description"] = queue.break_type.map(BREAK_LABELS)
    queue = queue.sort_values(["severity", "impact_cad"], key=lambda s: s.map({"High": 0, "Medium": 1, "Low": 2})
                              if s.name == "severity" else -s).reset_index(drop=True)
    queue.insert(0, "break_id", [f"BRK-{i + 1:03d}" for i in range(len(queue))])

    summary = {
        "trades_compared": int(len(trade_rec)),
        "trades_matched": int((trade_rec.rec_status == "MATCHED").sum()),
        "match_rate": float((trade_rec.rec_status == "MATCHED").mean()) if len(trade_rec) else 1.0,
        "positions_compared": int(len(ds.custodian_positions)),
        "position_breaks": int(len(position_rec)),
        "position_breaks_explained": int(position_rec.explained.sum()),
        "cash_events_compared": int(len(cash_rec)),
        "open_breaks": int(len(queue)),
        "impact_cad": float(queue.impact_cad.sum()),
        "in_flight_trades": int((~ds.book_trades.status.isin(["CONFIRMED", "SETTLED"])).sum()),
    }
    return ReconResult(trade_rec, position_rec, cash_rec, queue, summary)


def _money(v) -> str:
    return "—" if v is None or pd.isna(v) else f"${v:,.2f}"


def _trade_side(r, side: str) -> str:
    qty, px, sd, sd_ = (getattr(r, f"{side}_qty"), getattr(r, f"{side}_price"),
                        getattr(r, f"{side}_side"), getattr(r, f"{side}_settle_date"))
    if qty is None or pd.isna(qty):
        return "—"
    return f"{sd} {qty:,.3f} @ {px:,.4f}, settles {pd.Timestamp(sd_):%Y-%m-%d}"


# --------------------------------------------------------------------------- evidence for one break
def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = []
    for rec in df[cols].to_dict("records"):
        out.append({k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else
                        (None if (isinstance(v, float) and np.isnan(v)) else v)) for k, v in rec.items()})
    return out


def break_evidence(ds: Dataset, brk: dict) -> dict:
    """Raw records from each side for one break, as plain JSON-able dicts (no labels)."""
    tcols = ["account_id", "ticker", "side", "quantity", "price", "trade_date", "settle_date"]
    if brk["category"] == "Trade":
        ref = brk["ref"]
        b = ds.book_trades[(ds.book_trades.trade_id == ref) & ds.book_trades.status.isin(["CONFIRMED", "SETTLED"])]
        c = ds.custodian_trades[(ds.custodian_trades.client_ref == ref) | (ds.custodian_trades.cust_trade_id == ref)]
        book = _records(b, ["trade_id", *tcols, "status"])
        cust = _records(c, ["cust_trade_id", "client_ref", *tcols])
        return {"record_kind": "trade", "book": book[0] if book else None, "custodian": cust[0] if cust else None}
    if brk["category"] == "Position":
        a, t = brk["account_id"], brk["ticker"]
        trades = ds.book_trades[(ds.book_trades.account_id == a) & (ds.book_trades.ticker == t)]
        return {"record_kind": "position", "account_id": a, "ticker": t, "as_of": f"{ds.as_of:%Y-%m-%d}",
                "book": {"quantity": brk["book_value"]}, "custodian": {"quantity": brk["custodian_value"]},
                "trades_in_window_for_this_position": _records(trades, ["trade_id", *tcols, "status"])}
    a, t, d = brk["ref"].split("|")
    ccols = ["account_id", "ticker", "pay_date", "activity_type", "amount"]
    match = lambda df: df[(df.account_id == a) & (df.ticker == t) & (df.pay_date == pd.Timestamp(d))]
    book, cust = _records(match(ds.book_cash), ccols), _records(match(ds.custodian_cash), ccols)
    return {"record_kind": "cash", "book": book[0] if book else None, "custodian": cust[0] if cust else None}
