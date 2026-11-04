import pandas as pd
import streamlit as st

from workbench import ai
from workbench.ui import get_book_drift, get_dataset, get_recon, money, seed, sidebar


ds = get_dataset(seed())
rec = get_recon(seed())
sidebar(ds)
drift = get_book_drift(seed())

st.title("Advisor Portfolio Workbench")
st.markdown(
    "A working prototype of the tools an investment analytics team builds for portfolio managers and "
    "advisors: **a book-wide portfolio view**, **an automated daily trade and position reconciliation**, "
    "and **AI drafting with guardrails**. Built by Sam Zhang."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets in the demo book", money(drift.market_value.sum()))
c2.metric("Accounts outside target band", f"{int(drift.needs_rebalance.sum())} of {len(drift)}")
c3.metric("Open reconciliation breaks", rec.summary["open_breaks"], help="Trade, position and cash breaks")
c4.metric("AI mode", "Live model" if ai.live_available() else "Templates",
          help="Templates are used when no API key is configured")

st.divider()
a, b, c = st.columns(3)
with a:
    st.subheader("1 · Portfolio")
    st.write("Every account's look-through allocation against its model, drift flags, then a drill-down: "
             "performance vs. a 60/40 benchmark, volatility, drawdown and each holding's share of risk.")
    st.page_link("views/portfolio.py", label="Open portfolio view", icon="📈")
with b:
    st.subheader("2 · Reconciliation")
    st.write("Matches our book against the custodian's trade, position and dividend files in SQL. "
             "Breaks land in one queue with impact, age and severity; position breaks are traced to causes.")
    st.page_link("views/reconciliation.py", label="Open reconciliation", icon="🔁")
with c:
    st.subheader("3 · AI assistant")
    st.write("Drafts break explanations and client update notes. Python computes every number; outputs are "
             "checked for ungrounded figures and advice language, approved by a human, and logged.")
    st.page_link("views/ai_assistant.py", label="Open AI assistant", icon="🤖")

st.divider()
st.subheader("How this maps to the role")
st.table(pd.DataFrame([
    ("Design and maintain dashboards and analytical tools", "Portfolio page: book drift view, account drill-down, risk metrics"),
    ("Re-engineer manual workflows; reduce operational risk; improve controls",
     "Reconciliation page: daily rec that replaces a spreadsheet match, with tolerances, an exception queue and an audit trail"),
    ("Trade lifecycle and reconciliation", "Trades move executed → allocated → confirmed → settled (T+1); in-flight trades are excluded from the match"),
    ("Investment products: equities, fixed income, alternatives, pooled funds",
     "Stocks, bond ETFs, a REIT ETF, index ETFs, and a balanced mutual fund with look-through allocation"),
    ("Integrate AI; understand capabilities, limitations and risks",
     "AI page: grounded drafting, number and advice checks, human approval, call cap, audit log, accuracy eval"),
    ("Python, SQL, web apps, SDLC", "Python + DuckDB SQL + Streamlit; requirements doc, pytest suite, GitHub Actions CI"),
], columns=["Posting asks for", "Where it shows up"]))

st.caption("Synthetic accounts and custodian files; not investment advice. Source on GitHub.")
