"""Download 3+ years of adjusted daily closes for the universe into data/prices.csv.

Run locally (`python scripts/fetch_prices.py`) and commit the CSV, or let the
refresh-prices GitHub Action do it on weekdays. The app reads the CSV, so the
deployed site never calls Yahoo directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.data import PRICES_CSV, REAL_TICKERS  # noqa: E402


def main() -> None:
    import yfinance as yf

    raw = yf.download(REAL_TICKERS, period="4y", interval="1d", auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    missing = [t for t in REAL_TICKERS if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise SystemExit(f"No data for {missing}; keeping the existing CSV.")
    long = (close.ffill().dropna().rename_axis("date").reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="adj_close"))
    long["adj_close"] = long["adj_close"].round(4)
    PRICES_CSV.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(PRICES_CSV, index=False)
    print(f"Wrote {len(long):,} rows, {long.date.min():%Y-%m-%d} to {long.date.max():%Y-%m-%d} -> {PRICES_CSV}")


if __name__ == "__main__":
    main()
