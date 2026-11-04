"""Shared Streamlit helpers: cached data, formatting, sidebar."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .data import Dataset, generate_dataset
from .recon import ReconResult, run_reconciliation

DEFAULT_SEED = 42


@st.cache_data(show_spinner="Loading prices and generating today's files...")
def get_dataset(seed: int) -> Dataset:
    return generate_dataset(seed=seed)


@st.cache_data(show_spinner="Running reconciliation...")
def get_recon(seed: int) -> ReconResult:
    return run_reconciliation(get_dataset(seed))


@st.cache_data(show_spinner="Computing drift across the book...")
def get_book_drift(seed: int) -> pd.DataFrame:
    from .metrics import book_drift
    return book_drift(get_dataset(seed))


def seed() -> int:
    return int(st.session_state.get("seed", DEFAULT_SEED))


def sidebar(ds: Dataset) -> None:
    with st.sidebar:
        st.caption(f"As of **{ds.as_of:%a %b %d, %Y}**")
        st.caption(f"Prices: {ds.price_source}")
        st.caption("Accounts, trades and custodian files are synthetic. No client data.")
        if st.button("New scenario", help="Regenerate accounts, trades and breaks with a new seed"):
            st.session_state["seed"] = seed() + 1
            st.session_state.pop("rec_status", None)
            st.rerun()


def pct(x: float, d: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x * 100:+.{d}f}%" if x < 0 else f"{x * 100:.{d}f}%"


def money(x: float, d: int = 0) -> str:
    return f"${x:,.{d}f}"


def filter_in(df: pd.DataFrame, **cols) -> pd.DataFrame:
    """Keep rows whose column value is in the selected list; an empty selection keeps everything."""
    keep = pd.Series(True, index=df.index)
    for col, values in cols.items():
        if values:
            keep &= df[col].isin(values)
    return df[keep]
