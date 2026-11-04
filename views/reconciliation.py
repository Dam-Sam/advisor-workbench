import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from workbench.data import LIFECYCLE
from workbench.recon import BREAK_LABELS, TOLERANCES
from workbench.ui import filter_in, get_dataset, get_recon, money, seed, sidebar

NAVY, GREY, RED, AMBER = "#1f3a5f", "#9aa5b1", "#c0392b", "#d68910"

ds = get_dataset(seed())
rec = get_recon(seed())
sidebar(ds)
s = rec.summary

st.title("Reconciliation")
st.caption(f"Book of record vs. custodian files for {ds.as_of:%A %B %d, %Y}. Matching runs in SQL (DuckDB); "
           "the queries live in sql/ in the repo.")

k = st.columns(5)
k[0].metric("Trade match rate", f"{s['match_rate'] * 100:.1f}%", help=f"{s['trades_matched']} of {s['trades_compared']} trades")
k[1].metric("Open breaks", s["open_breaks"])
k[2].metric("Amount at stake", money(s["impact_cad"]), help="Cash or exposure affected across open breaks")
k[3].metric("Oldest break (bus. days)", int(rec.queue.age_bdays.max()) if len(rec.queue) else 0)
k[4].metric("Position breaks explained", f"{s['position_breaks_explained']} of {s['position_breaks']}",
            help="Traced to a trade-level break on the same account and security")

def dates_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date
    return out


# ---------------------------------------------------------------- lifecycle
with st.expander("Trade lifecycle this week", expanded=False):
    counts = ds.book_trades.status.value_counts().reindex(LIFECYCLE).fillna(0).astype(int)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[GREY, GREY, NAVY, NAVY],
                           text=counts.values, textposition="outside"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                      yaxis_title="Trades")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Orders are executed, allocated to client accounts, confirmed, then settle T+1. "
               f"{s['in_flight_trades']} trades are still in flight (executed or allocated) and are excluded "
               "from today's match; they are not breaks yet.")

# ---------------------------------------------------------------- exception queue
st.subheader("Exception queue")
if "rec_status" not in st.session_state:
    st.session_state["rec_status"] = {}
saved = st.session_state["rec_status"]

q = rec.queue.copy()
q["status"] = q.break_id.map(lambda b: saved.get(b, {}).get("status", "Open"))
q["comment"] = q.break_id.map(lambda b: saved.get(b, {}).get("comment", ""))

f1, f2, f3 = st.columns(3)
cats = f1.multiselect("Category", ["Trade", "Position", "Cash"])
sev = f2.multiselect("Severity", ["High", "Medium", "Low"])
hide_resolved = f3.toggle("Hide resolved", value=False)
view = filter_in(q, category=cats, severity=sev)
if hide_resolved:
    view = view[view.status != "Resolved"]

st.caption("Set **Status** and add a **Comment** directly in the table.")
edited = st.data_editor(
    view[["break_id", "status", "severity", "description", "account_id", "ticker", "impact_cad", "age_bdays",
          "book_value", "custodian_value", "comment", "category", "ref"]],
    hide_index=True, width="stretch", key="queue_editor",
    disabled=["break_id", "severity", "category", "description", "account_id", "ticker", "ref", "impact_cad",
              "age_bdays", "book_value", "custodian_value"],
    column_config={
        "break_id": "Break", "severity": "Severity", "category": "Type", "description": "What differs",
        "account_id": "Account", "ticker": "Security", "ref": "Reference",
        "impact_cad": st.column_config.NumberColumn("At stake", format="dollar"),
        "age_bdays": st.column_config.NumberColumn("Age (bd)"),
        "book_value": "Our book", "custodian_value": "Custodian",
        "status": st.column_config.SelectboxColumn("Status", options=["Open", "Investigating", "Resolved"],
                                                   required=True),
        "comment": st.column_config.TextColumn("Comment", max_chars=200),
    },
)
for row in edited[["break_id", "status", "comment"]].itertuples(index=False):
    if row.status != "Open" or row.comment:
        saved[row.break_id] = {"status": row.status, "comment": row.comment}
    else:
        saved.pop(row.break_id, None)

resolved = sum(v["status"] == "Resolved" for v in saved.values())
st.progress(resolved / max(1, len(q)), text=f"{resolved} of {len(q)} breaks resolved")
c1, c2 = st.columns([1, 3])
export = q.assign(status=q.break_id.map(lambda b: saved.get(b, {}).get("status", "Open")),
                  comment=q.break_id.map(lambda b: saved.get(b, {}).get("comment", "")))
c1.download_button("Export exceptions (CSV)", export.to_csv(index=False), f"breaks_{ds.as_of:%Y%m%d}.csv", "text/csv")
c2.caption("Status and comments are kept for your session only in this demo. In production they would be "
           "written to a breaks table with user and timestamp, so the queue carries over to tomorrow's run.")

st.page_link("views/ai_assistant.py", label="Ask the AI assistant to explain a break", icon="🤖")

# ---------------------------------------------------------------- detail tabs
st.subheader("Detail")
t1, t2, t3, t4 = st.tabs(["Trades", "Positions", "Dividends", "How it works"])
with t1:
    show_all = st.toggle("Show matched trades too", value=False)
    tr = rec.trade_rec if show_all else rec.trade_rec[rec.trade_rec.rec_status != "MATCHED"]
    st.dataframe(dates_only(tr.drop(columns=["in_book", "at_custodian"])), hide_index=True, width="stretch")
with t2:
    st.dataframe(rec.position_rec, hide_index=True, width="stretch",
                 column_config={"explained_by": "Explained by", "explained": st.column_config.CheckboxColumn("Explained")})
    st.caption("A position break with a trade-level cause clears when that trade break is fixed. "
               "Only unexplained ones enter the queue, so the same problem is never worked twice.")
with t3:
    st.dataframe(dates_only(rec.cash_rec[rec.cash_rec.rec_status != "MATCHED"]), hide_index=True, width="stretch")
    st.caption(f"{s['cash_events_compared']} dividend payments compared.")
with t4:
    st.markdown(f"""
**Manual process this replaces:** export the book and the custodian files to Excel, VLOOKUP trade by trade,
eyeball quantities and prices, email the differences around, track them in a separate sheet.

**Automated run:**
1. Load today's book trades, custodian confirmations, custodian position file and income activity.
2. Exclude trades still in flight (executed or allocated, not yet confirmed).
3. Match trades on our trade ID (sent to the custodian as the client reference) with a full outer join, so
   trades missing on either side surface instead of disappearing.
4. Compare side, quantity, price and settlement date with tolerances: quantity exact, price within
   ${TOLERANCES['price_abs_tol']:.2f} or {TOLERANCES['price_rel_tol'] * 100:.1f}% (whichever is larger).
5. Rebuild book positions (opening + settled trades) and compare them with the custodian position file.
   Trace each position break to a trade break on the same account and security.
6. Match dividend payments on account, security, pay date and type.
7. Put everything unexplained in one queue, scored by amount at stake and age.

**Break types:** {", ".join(f"{k.replace('_', ' ').lower()} ({v.lower()})" for k, v in BREAK_LABELS.items())}.
""")
