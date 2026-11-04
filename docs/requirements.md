# Requirements: Advisor Portfolio Workbench

**Owner:** Sam Zhang · **Status:** v1 prototype · **Data:** synthetic accounts and custodian files; public market prices

## Problem

Advisors and portfolio managers run books of client accounts against model portfolios. Three daily jobs are
still largely manual:

1. Finding which accounts have drifted from their model and need a rebalance.
2. Reconciling the firm's trades, positions and income against the custodian, usually in spreadsheets.
3. Writing explanations: of a break for the operations team, and of performance for clients.

Manual work here is slow and is itself an operational risk: a missed break can mean a failed settlement or a
wrong client statement.

## Users and stories

| User | Story | Acceptance criteria |
| --- | --- | --- |
| Advisor / PM | See every account outside its target band in one list | Look-through allocation (funds split into asset classes); band ±5 pts; filter by advisor and model; CSV export |
| Advisor / PM | Drill into one account's performance and risk | Return vs. 60/40 benchmark, volatility, max drawdown, Sharpe, tracking error, risk contribution by holding |
| Operations analyst | Reconcile today's trades, positions and dividends automatically | Full match on trade ID; tolerances configurable; in-flight trades excluded; zero false breaks on clean files |
| Operations analyst | Work one prioritized queue | Every break has type, account, security, amount at stake, age, severity; status and comment per break; CSV export |
| Operations analyst | Not chase the same problem twice | Position breaks caused by a trade break are linked to it and kept out of the queue |
| Operations analyst | Get a first explanation for a break | AI draft of likely cause, evidence and next step from the raw records; human approves; logged |
| Advisor | Get a first draft of a client update | AI draft from computed facts only; blocked if it invents a number or gives advice; editable; logged |
| Compliance / risk | Trust the AI outputs | No free-text prompts; number grounding and advice checks; daily call cap; audit log; measured accuracy |

## Process map

**Before (manual):**

```
Export book ─┐
             ├─► Excel VLOOKUP by trade ─► eyeball qty/price ─► email differences ─► track in another sheet
Export custodian files ─┘
```

**After (automated daily run):**

```
Book trades ──────────┐
Custodian confirms ───┼─► SQL trade match (full outer join, tolerances) ─┐
Custodian positions ──┼─► SQL position match ─► trace to trade breaks ────┼─► one exception queue ─► analyst review
Income activity ──────┘─► SQL dividend match ─────────────────────────────┘        │                (status, comment)
                                                                                     └─► AI explanation draft ─► approve / reject ─► audit log
```

## Scope

**In v1:** the three pages, synthetic data generator with injected breaks and ground truth, SQL rec engine,
AI drafting with guardrails, pytest suite, CI, deployment to Streamlit Community Cloud.

**Out of v1 (next steps):** persistent breaks table (status carries over day to day); real custodian file
formats (e.g., SWIFT MT535/MT536 statements); corporate actions; FX for USD holdings; user login and roles;
rebalance trade proposals.

## Success criteria

- Reconciliation finds every injected break with zero false positives, across many random scenarios (tested).
- Clean files produce zero breaks (tested).
- AI drafts that contain an ungrounded number or advice language are never shown (tested).
- App loads in under 10 seconds and every page renders without errors (tested in CI).

## Risks and controls

| Risk | Control |
| --- | --- |
| Model invents a figure | Numbers check against the fact sheet; draft blocked on failure |
| Model gives investment advice | Advice-phrase check; system prompt; human approval before anything is sent |
| Prompt injection | Structured inputs and fixed choices only; no free-text prompt box |
| Cost runaway on a public demo | Daily call cap; deterministic template fallback |
| Model outage | Automatic fallback to rule-based playbook and templates |
| Wrong tolerance hides a real break | Tolerances in one place, shown on the page, and unit-tested at the boundary |
