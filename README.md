# Advisor Portfolio Workbench

**Live demo:** _add your `https://<name>.streamlit.app` link here_ · **Walkthrough video:** _add link_

A working prototype of the tools an investment analytics team builds for portfolio managers and advisors:

| Page | What it does |
| --- | --- |
| **Portfolio** | Every account's look-through allocation vs. its model (Conservative / Balanced / Growth), drift flags at ±5 pts, and a drill-down: return vs. a 60/40 benchmark, volatility, max drawdown, Sharpe, tracking error, and each holding's share of portfolio risk. |
| **Reconciliation** | Automated daily match of our book against the custodian's trade, position and dividend files, written in SQL (DuckDB). One exception queue scored by amount at stake and age; position breaks traced to their trade-level cause; status and comments per break; CSV export. |
| **AI assistant** | Drafts break explanations for operations and client update notes for advisors. Python computes every number; outputs are checked for invented figures and advice language, approved by a human, capped per day, and logged. Falls back to templates with no API key. |

Accounts, trades and custodian files are **synthetic** and generated with known, injected breaks, so the
reconciliation can be tested against ground truth. Prices are real Yahoo Finance closes once
`data/prices.csv` exists, otherwise seeded synthetic prices.

## Why it's built this way

- **Product coverage.** Stocks (RY, TD, ENB, CNR, SHOP), index ETFs (VCN, XAW), bond ETFs (ZAG, XBB), an
  alternative (REIT ETF ZRE), and a fictional balanced mutual fund looked through 60/40 into its asset classes.
- **Trade lifecycle.** Trades move executed → allocated → confirmed → settled, T+1. Trades still in flight are
  excluded from the match, because they are not breaks yet.
- **Full outer joins.** A trade missing on either side must surface as a break, not silently drop out of an
  inner join.
- **Explained vs. unexplained.** A position break caused by a trade break clears when the trade is fixed, so
  only unexplained position breaks enter the queue. Nobody works the same problem twice.
- **AI where it helps, never as the source of numbers.** The model writes words; Python writes numbers. See
  `workbench/ai.py` for the guardrails and `tests/test_ai_guardrails.py` for proof they block bad output.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/fetch_prices.py                         # real prices -> data/prices.csv (optional)
pytest -q                                              # 45 tests
streamlit run app.py
```

For the live model, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key.

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a **public** GitHub repo (include `data/prices.csv`).
2. Go to share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py`.
   Under **Advanced settings**, choose Python 3.12 and paste your secrets (`ANTHROPIC_API_KEY`, `AI_DAILY_CAP`).
3. Set a custom subdomain, e.g. `sam-advisor-workbench.streamlit.app`.
4. In GitHub → **Settings → Secrets and variables → Actions → Variables**, add `APP_URL` with that link.
   The **Keep app awake** workflow then opens it every 6 hours so the hiring manager never lands on a sleep screen.
5. The **Refresh prices** workflow updates `data/prices.csv` each weekday evening; **CI** runs the tests on every push.

## Measure the AI

```bash
ANTHROPIC_API_KEY=... python scripts/eval_ai.py --seeds 5
```

The model sees only the raw book and custodian records for each break (no label, no SQL result) and must name
the break type. Results go to `docs/eval_results.md` and show on the AI page. Without a key it reports the
rule-based baseline, which matches the SQL by construction.

## Project layout

```
app.py                  navigation entry point
views/                  the four pages
workbench/data.py       securities, prices, seeded generator for accounts, trades, custodian files, breaks
workbench/recon.py      runs sql/*.sql on DuckDB, builds the exception queue and per-break evidence
workbench/metrics.py    returns, risk, drawdown, risk contribution, look-through allocation, drift
workbench/facts.py      the fixed fact sheet the AI may use
workbench/ai.py         model calls, guardrails, daily cap, audit log, template fallback
sql/                    trade, position and dividend reconciliation queries
tests/                  metrics, reconciliation vs. ground truth, guardrails, every page rendered headlessly
scripts/                price download, AI eval, keep-awake
docs/requirements.md    user stories, process map, scope, success criteria, risks and controls
```

## Next steps

Persist break status in Postgres so the queue carries over day to day; parse real custodian formats
(SWIFT MT535/MT536); corporate actions and FX; rebalance trade proposals; user roles.

_Built with GitHub Copilot and Claude as pair programmers. Not investment advice._
