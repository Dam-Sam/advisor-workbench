import pandas as pd

from workbench import data
from workbench.recon import run_reconciliation
from tests.conftest import AS_OF


def test_real_price_csv_path(tmp_path, monkeypatch):
    """The CSV written by scripts/fetch_prices.py is used when present (here: synthetic prices in that format)."""
    wide, _ = data.load_prices(seed=3, as_of=AS_OF)
    long = (wide[data.REAL_TICKERS].rename_axis("date").reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="adj_close"))
    csv = tmp_path / "prices.csv"
    long.to_csv(csv, index=False)
    monkeypatch.setattr(data, "PRICES_CSV", csv)
    prices, source = data.load_prices(seed=3)
    assert source.startswith("Yahoo")
    assert data.MUTUAL_FUND in prices and prices.notna().all().all()
    ds = data.generate_dataset(seed=3)
    assert set(zip(ds.truth.break_type, ds.truth.ref)) == set(
        zip(*[run_reconciliation(ds).queue[c] for c in ("break_type", "ref")]))


def test_product_coverage(ds):
    vehicles = set(ds.securities.vehicle)
    classes = set(ds.securities.asset_class)
    assert {"Stock", "ETF", "Mutual Fund"} <= vehicles
    assert {"Equity", "Fixed Income", "Alternative"} <= classes


def test_lifecycle_timestamps_are_ordered(ds):
    t = ds.book_trades
    assert (t.executed_ts > t.order_ts).all()
    alloc = t.dropna(subset=["allocated_ts"])
    assert (alloc.allocated_ts > alloc.executed_ts).all()
    conf = t.dropna(subset=["confirmed_ts"])
    assert (conf.confirmed_ts > conf.allocated_ts).all()


def test_no_negative_positions(ds):
    assert (data.book_positions(ds).quantity > 0).all()
