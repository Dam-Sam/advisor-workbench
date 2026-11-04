import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from workbench.data import BENCHMARK, DRIFT_BAND
from workbench.facts import LOOKBACKS
from workbench.metrics import (allocation, daily_returns, drift_table, holdings, portfolio_returns,
                               risk_contributions, summary_stats)
from workbench.ui import filter_in, get_book_drift, get_dataset, money, pct, seed, sidebar

NAVY, GREY, RED, GREEN = "#1f3a5f", "#9aa5b1", "#c0392b", "#2e7d32"

ds = get_dataset(seed())
sidebar(ds)
book = get_book_drift(seed())

st.title("Portfolio")
st.caption(f"Look-through allocation vs. model target. Drift band: ±{DRIFT_BAND * 100:.0f} percentage points.")

# ---------------------------------------------------------------- book view
st.subheader("Book overview")
f1, f2, f3 = st.columns([1, 1, 1])
adv = f1.multiselect("Advisor", sorted(book.advisor.unique()))
mdl = f2.multiselect("Model", sorted(book.model.unique()))
only = f3.toggle("Only accounts needing rebalance", value=True)
view = filter_in(book, advisor=adv, model=mdl)
if only:
    view = view[view.needs_rebalance]

k1, k2, k3 = st.columns(3)
k1.metric("Accounts shown", len(view))
k2.metric("Assets shown", money(view.market_value.sum()))
k3.metric("Needing rebalance (whole book)", f"{int(book.needs_rebalance.sum())} of {len(book)}")

shown = view.assign(market_value=view.market_value.round(0),
                    **{c: view[c] * 100 for c in ["equity", "fixed_income", "alternative", "worst_drift"]})
st.dataframe(
    shown, hide_index=True, width="stretch",
    column_config={
        "account_id": "Account", "household": "Household", "advisor": "Advisor", "model": "Model",
        "market_value": st.column_config.NumberColumn("Value ($)", format="localized"),
        "equity": st.column_config.NumberColumn("Equity", format="%.1f%%"),
        "fixed_income": st.column_config.NumberColumn("Fixed income", format="%.1f%%"),
        "alternative": st.column_config.NumberColumn("Alternative", format="%.1f%%"),
        "worst_sleeve": "Largest drift in",
        "worst_drift": st.column_config.NumberColumn("Drift (pts)", format="%+.1f"),
        "needs_rebalance": st.column_config.CheckboxColumn("Rebalance?"),
    },
)
st.download_button("Download rebalance list (CSV)", view.to_csv(index=False), "rebalance_list.csv", "text/csv")

st.divider()

# ---------------------------------------------------------------- account drill-down
st.subheader("Account drill-down")
c1, c2, c3 = st.columns([2, 1, 1])
default = view.account_id.iloc[0] if len(view) else book.account_id.iloc[0]
acct_id = c1.selectbox("Account", book.account_id, index=int(book.index[book.account_id == default][0]),
                       format_func=lambda a: f"{a} · {book.set_index('account_id').at[a, 'household']} · "
                                             f"{book.set_index('account_id').at[a, 'model']}")
period = c2.radio("Period", list(LOOKBACKS), index=1, horizontal=True)
rf = c3.number_input("Risk-free rate (assumption)", 0.0, 10.0, 2.5, 0.25, format="%.2f",
                     help="Used only for the Sharpe ratio. Replace with the current 3-month T-bill yield.") / 100

acct = ds.accounts.set_index("account_id").loc[acct_id]
h = holdings(ds, acct_id)
alloc = allocation(h, acct.cash)
drift = drift_table(alloc, acct.model)
days = min(LOOKBACKS[period], len(ds.prices) - 1)
r = portfolio_returns(ds, h, acct.cash, days)
bench = ds.prices[BENCHMARK].pct_change().iloc[-days:]
s = summary_stats(r, bench, rf)

m = st.columns(6)
m[0].metric("Market value", f"${(h.market_value.sum() + acct.cash) / 1e6:.2f}M")
m[1].metric(f"Return ({period})", pct(s["total_return"]),
            delta=f"{(s['total_return'] - s['benchmark_total_return']) * 100:+.1f} vs bench",
            help="Delta is the difference from the 60/40 benchmark, in percentage points")
m[2].metric("Volatility (ann.)", pct(s["annualized_vol"]))
m[3].metric("Max drawdown", pct(s["max_drawdown"]))
m[4].metric("Sharpe ratio", f"{s['sharpe']:.2f}")
m[5].metric("Tracking error", pct(s["tracking_error"]), help="Volatility of the return difference vs. benchmark")
st.caption(f"Risk view: today's holdings held at constant weights over the period. Benchmark: {BENCHMARK} "
           "(60/40 balanced ETF).")

left, right = st.columns([3, 2])
with left:
    growth = pd.DataFrame({"Portfolio": (1 + r).cumprod() * 100, "Benchmark (60/40)": (1 + bench).cumprod() * 100})
    fig = go.Figure()
    fig.add_scatter(x=growth.index, y=growth["Portfolio"], name="Portfolio", line=dict(color=NAVY, width=2.5))
    fig.add_scatter(x=growth.index, y=growth["Benchmark (60/40)"], name="Benchmark (60/40)",
                    line=dict(color=GREY, width=2, dash="dot"))
    fig.update_layout(title="Growth of $100", height=360, margin=dict(l=10, r=10, t=50, b=10),
                      template="plotly_white", legend=dict(orientation="h", y=1.12, x=0), yaxis_title="$")
    st.plotly_chart(fig, width="stretch")
with right:
    d = drift.drop(index="Cash") if drift.at["Cash", "target"] == 0 else drift
    colors = [RED if o else NAVY for o in d.out_of_band]
    fig = go.Figure()
    fig.add_bar(y=d.index, x=d.target * 100, orientation="h", name="Target", marker_color="#dfe4ea",
                error_x=dict(type="constant", value=DRIFT_BAND * 100, color=GREY, thickness=1))
    fig.add_bar(y=d.index, x=d.actual * 100, orientation="h", name="Actual", marker_color=colors, width=0.35)
    fig.update_layout(title=f"Allocation vs. {acct.model} model", barmode="overlay", height=360,
                      margin=dict(l=10, r=10, t=50, b=10), template="plotly_white", xaxis_title="% of portfolio",
                      legend=dict(orientation="h", y=1.12, x=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    flagged = list(drift.index[drift.out_of_band])
    if flagged:
        st.warning(f"Outside band: {', '.join(flagged)}. Grey whiskers show the ±{DRIFT_BAND * 100:.0f} pt band.")
    else:
        st.success("All sleeves within band.")

# risk contribution vs weight
rets = daily_returns(ds.prices).iloc[-days:]
w = h.set_index("ticker").market_value / (h.market_value.sum() + acct.cash)
rc = risk_contributions(w, rets).reindex(w.index)
rc_df = pd.DataFrame({"Weight": w * 100, "Share of risk": rc * 100}).sort_values("Share of risk")
fig = go.Figure()
fig.add_bar(y=rc_df.index, x=rc_df["Weight"], orientation="h", name="Weight", marker_color="#c9d3df")
fig.add_bar(y=rc_df.index, x=rc_df["Share of risk"], orientation="h", name="Share of risk", marker_color=NAVY)
fig.update_layout(title="Where the risk comes from: weight vs. share of portfolio variance", barmode="group",
                  height=60 + 38 * len(rc_df), margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                  xaxis_title="%", legend=dict(orientation="h", y=1.08, x=0))
st.plotly_chart(fig, width="stretch")
st.caption("A holding whose share of risk is far above its weight (often equities and single stocks) drives "
           "most of the portfolio's swings; bonds usually contribute less risk than their weight.")

st.markdown("**Holdings**")
st.dataframe(
    h[["ticker", "name", "vehicle", "asset_class", "quantity", "price", "market_value", "weight"]]
    .assign(market_value=h.market_value.round(0), weight=h.weight * 100),
    hide_index=True, width="stretch",
    column_config={
        "ticker": "Ticker", "name": "Name", "vehicle": "Vehicle", "asset_class": "Asset class",
        "quantity": st.column_config.NumberColumn("Units", format="localized"),
        "price": st.column_config.NumberColumn("Price", format="dollar"),
        "market_value": st.column_config.NumberColumn("Market value ($)", format="localized"),
        "weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
    },
)
st.caption(f"Cash: {money(acct.cash, 2)}. Multi-asset funds are looked through (60% equity / 40% fixed income) "
           "for the allocation chart.")
