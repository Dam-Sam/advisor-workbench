import pandas as pd
import pytest

from workbench.ai import classify_break_rule
from workbench.data import generate_dataset
from workbench.recon import break_evidence, run_reconciliation
from tests.conftest import AS_OF


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 123, 2026])
def test_detects_exactly_the_injected_breaks(seed):
    ds = generate_dataset(seed=seed, as_of=AS_OF)
    queue = run_reconciliation(ds).queue
    found = set(zip(queue.break_type, queue.ref))
    injected = set(zip(ds.truth.break_type, ds.truth.ref))
    assert found == injected, f"missed {injected - found}, false positives {found - injected}"


def test_clean_files_reconcile_to_zero_breaks():
    ds = generate_dataset(seed=5, as_of=AS_OF, inject_breaks=False)
    r = run_reconciliation(ds)
    assert r.queue.empty
    assert r.position_rec.empty
    assert r.summary["match_rate"] == 1.0


def test_in_flight_trades_are_not_breaks(ds):
    r = run_reconciliation(ds)
    in_flight = ds.book_trades[~ds.book_trades.status.isin(["CONFIRMED", "SETTLED"])].trade_id
    assert len(in_flight) > 0
    assert not set(in_flight) & set(r.trade_rec.trade_ref)


def test_only_unexplained_position_breaks_enter_queue(ds):
    r = run_reconciliation(ds)
    queued = r.queue[r.queue.category == "Position"]
    assert len(queued) == int((~r.position_rec.explained).sum())
    assert r.position_rec.explained.sum() >= 1


def test_price_within_tolerance_matches(ds):
    ds2 = generate_dataset(seed=11, as_of=AS_OF, inject_breaks=False)
    ds2.custodian_trades.loc[0, "price"] = ds2.custodian_trades.loc[0, "price"] * 1.004  # inside 0.5%
    assert run_reconciliation(ds2).queue.empty


def test_settlement_is_t_plus_one(ds):
    t = ds.book_trades
    bdays = [len(pd.bdate_range(a, b)) - 1 for a, b in zip(t.trade_date, t.settle_date)]
    assert set(bdays) == {1}


def test_rule_baseline_classifies_every_break_from_raw_evidence(ds):
    r = run_reconciliation(ds)
    for brk in r.queue.to_dict("records"):
        assert classify_break_rule(break_evidence(ds, brk)) == brk["break_type"]
