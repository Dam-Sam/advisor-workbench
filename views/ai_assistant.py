from pathlib import Path

import pandas as pd
import streamlit as st

from workbench import ai
from workbench.facts import LOOKBACKS, client_facts
from workbench.recon import break_evidence
from workbench.ui import get_dataset, get_recon, seed, sidebar


ds = get_dataset(seed())
rec = get_recon(seed())
sidebar(ds)

st.title("AI assistant")
if ai.live_available():
    st.success(f"Live model: `{ai.model_name()}` · {ai.CAP.used()} of {ai.CAP.limit} calls used today", icon="✅")
else:
    st.info("Running in template mode (no API key configured). Every feature still works with deterministic "
            "templates; add ANTHROPIC_API_KEY to the app's secrets to use the live model.", icon="ℹ️")

with st.expander("How the guardrails work", expanded=False):
    st.markdown("""
- **Python computes every number.** The model gets a fixed sheet of facts and only explains or narrates them.
- **No free-text prompt box.** Inputs are records and fixed choices (tone, period), which limits prompt injection.
- **Every output is checked before display:** each number must match a number in the facts, advice language
  ("you should buy…", "guaranteed", "will outperform") is blocked, and length is capped. A failed check blocks the draft.
- **A human approves.** Nothing is sent or booked automatically; approvals and rejections are logged.
- **Cost and failure controls.** A daily call cap protects the public demo; if the model is unavailable,
  the page falls back to templates instead of failing.
""")


def show_checks(checks: ai.Checks) -> None:
    c = st.columns(3)
    c[0].markdown(("✅" if checks.numbers_grounded else "❌") + " **Numbers grounded**"
                  + ("" if checks.numbers_grounded else f"  \nNot in facts: {', '.join(checks.ungrounded)}"))
    c[1].markdown(("✅" if checks.no_advice else "❌") + " **No advice language**"
                  + ("" if checks.no_advice else f"  \nFound: {', '.join(checks.advice_hits)}"))
    c[2].markdown(("✅" if checks.within_length else "❌") + f" **Length** ({checks.word_count} words)")


tab1, tab2, tab3 = st.tabs(["Explain a break", "Draft a client update", "Audit log & eval"])

# ---------------------------------------------------------------- explain a break
with tab1:
    st.caption("For the operations analyst working the reconciliation queue.")
    q = rec.queue
    pick = st.selectbox("Break", q.break_id,
                        format_func=lambda b: (lambda r: f"{b} · {r.severity} · {r.description} · {r.account_id} "
                                                         f"{r.ticker}")(q.set_index("break_id").loc[b]))
    brk = q.set_index("break_id").loc[pick].to_dict() | {"break_id": pick}
    brk_clean = {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v) for k, v in brk.items()}
    evidence = break_evidence(ds, brk)
    with st.expander("What the model sees (evidence)"):
        st.json({"break": brk_clean, "evidence": evidence})

    if st.button("Explain this break", type="primary"):
        st.session_state["explain"] = (pick, ai.explain_break(brk_clean, evidence))
    if st.session_state.get("explain", (None,))[0] == pick:
        out = st.session_state["explain"][1]
        if out["fallback_reason"]:
            st.warning(f"Live model unavailable ({out['fallback_reason']}); showing the rule-based playbook.")
        st.markdown(f"**Likely cause** — {out['likely_cause']}")
        st.markdown(f"**Evidence** — {out['evidence']}")
        st.markdown(f"**Next step** — {out['next_step']}")
        st.caption(f"Owner: {out['owner']} · Confidence: {out['confidence']} · Source: "
                   f"{out['model'] or 'rule-based playbook'}")
        show_checks(out["checks"])
        a, b, _ = st.columns([1, 1, 4])
        if a.button("Approve", key="ex_ok", disabled=not out["checks"].passed):
            ai.audit({"feature": "explain_break", "action": "approved", "break_id": pick})
            st.toast("Approved and logged")
        if b.button("Reject", key="ex_no"):
            ai.audit({"feature": "explain_break", "action": "rejected", "break_id": pick})
            st.toast("Rejected and logged")

# ---------------------------------------------------------------- client update
with tab2:
    st.caption("For the advisor: a first draft of a client note, for review and editing before anything is sent.")
    c1, c2, c3 = st.columns(3)
    acct = c1.selectbox("Account", ds.accounts.account_id,
                        format_func=lambda a: f"{a} · {ds.accounts.set_index('account_id').at[a, 'household']}")
    period = c2.selectbox("Period", list(LOOKBACKS), index=1)
    tone = c3.selectbox("Tone", list(ai.TONES))
    facts = client_facts(ds, acct, period)
    with st.expander("Facts the model is allowed to use"):
        st.json(facts)

    if st.button("Draft update", type="primary"):
        st.session_state["draft"] = (acct, period, tone, ai.draft_client_summary(facts, tone))
    d = st.session_state.get("draft")
    if d and d[:3] == (acct, period, tone):
        out = d[3]
        if out["fallback_reason"]:
            st.warning(f"Live model unavailable ({out['fallback_reason']}); showing the template draft.")
        show_checks(out["checks"])
        if out["checks"].passed:
            text = st.text_area("Draft (edit before sending)", out["text"], height=200)
            st.caption(f"Source: {out['model'] or 'template'} · Draft for advisor review. Not investment advice.")
            a, b, _ = st.columns([1, 1, 4])
            if a.button("Approve draft", key="cs_ok"):
                edited = text.strip() != out["text"].strip()
                ai.audit({"feature": "client_summary", "action": "approved", "account_id": acct,
                          "edited_by_advisor": edited})
                st.toast("Approved and logged")
            if b.button("Reject draft", key="cs_no"):
                ai.audit({"feature": "client_summary", "action": "rejected", "account_id": acct})
                st.toast("Rejected and logged")
        else:
            st.error("Draft blocked by guardrails and not shown. The failed checks above are logged.")

# ---------------------------------------------------------------- audit + eval
with tab3:
    st.markdown("**Audit log** (this server, most recent first)")
    log = ai.audit_log()
    if log:
        st.dataframe(pd.DataFrame(log).astype(str), hide_index=True, width="stretch")
    else:
        st.caption("No AI activity yet in this session.")
    st.markdown("**Accuracy eval**")
    eval_md = Path(__file__).resolve().parent.parent / "docs" / "eval_results.md"
    if eval_md.exists():
        st.markdown(eval_md.read_text())
    else:
        st.caption("Run `python scripts/eval_ai.py` to measure classification accuracy against labelled breaks.")
