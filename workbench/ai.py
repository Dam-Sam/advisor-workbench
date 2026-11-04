"""AI features with guardrails.

Design rules (the interview talking points):
1. Python computes every number. The model only explains or narrates numbers it is given.
2. Inputs are structured facts and fixed choices (no free-text prompt box), which limits prompt injection.
3. Every output is checked before display: numbers must trace to the input facts, advice language is
   blocked, and length is bounded. A failed check blocks the draft instead of showing it.
4. A human approves outputs; every call is logged (inputs, output, checks, reviewer action).
5. A daily call cap protects a public demo from runaway cost. With no API key the app falls back to
   deterministic templates, so it always works.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-5-5"
AUDIT_PATH = Path(os.environ.get("AUDIT_LOG_PATH", "/tmp/advisor_workbench_audit.jsonl"))
BREAK_TYPES = ["MISSING_AT_CUSTODIAN", "MISSING_IN_BOOK", "QTY_MISMATCH", "PRICE_MISMATCH",
               "SETTLE_DATE_MISMATCH", "UNEXPLAINED_POSITION", "UNBOOKED_DIVIDEND",
               "DIVIDEND_AMOUNT_MISMATCH", "DIVIDEND_NOT_AT_CUSTODIAN"]
TONES = {
    "Plain and brief": "plain, brief and warm; short sentences",
    "Detailed": "complete but readable; explain each figure in one clause",
    "Reassuring after a down market": "calm and factual; acknowledge the decline without minimising it",
}

ADVICE_PATTERNS = [
    r"\byou should (buy|sell|invest|move|switch|add|reduce)\b",
    r"\bwe (recommend|suggest|advise)\b", r"\bi (recommend|suggest|advise)\b",
    r"\brecommend(ed)? (buying|selling|that you)\b",
    r"\bguarantee[sd]?\b", r"\bwill (outperform|rise|recover|beat)\b", r"\b(strong )?buy (now|signal)\b",
    r"\brisk[- ]free\b(?! rate)",
]


# --------------------------------------------------------------------------- configuration
def _secret(name: str, default: str | None = None) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def model_name() -> str:
    return _secret("ANTHROPIC_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def live_available() -> bool:
    return bool(_secret("ANTHROPIC_API_KEY"))


# --------------------------------------------------------------------------- daily cap + audit log
class UsageCap:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = dt.date.today()
        self._count = 0

    @property
    def limit(self) -> int:
        return int(_secret("AI_DAILY_CAP", "40") or 40)

    def used(self) -> int:
        with self._lock:
            if dt.date.today() != self._day:
                self._day, self._count = dt.date.today(), 0
            return self._count

    def try_consume(self) -> bool:
        with self._lock:
            if dt.date.today() != self._day:
                self._day, self._count = dt.date.today(), 0
            if self._count >= self.limit:
                return False
            self._count += 1
            return True


CAP = UsageCap()
_AUDIT: list[dict] = []
_AUDIT_LOCK = threading.Lock()


def audit(event: dict) -> dict:
    event = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **event}
    with _AUDIT_LOCK:
        _AUDIT.append(event)
        del _AUDIT[:-200]
        try:
            with AUDIT_PATH.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError:
            pass
    return event


def audit_log() -> list[dict]:
    with _AUDIT_LOCK:
        return list(reversed(_AUDIT))


# --------------------------------------------------------------------------- guardrails
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> list[tuple[float, int]]:
    """Numbers in text as (absolute value, decimals shown). Signs are ignored: 'fell 4.2%' == '-4.2%'."""
    out = []
    for m in _NUM.finditer(text):
        s = m.group().replace(",", "")
        if s.endswith("."):
            s = s[:-1]
        dec = len(s.split(".")[1]) if "." in s else 0
        out.append((float(s), dec))
    return out


def ungrounded_numbers(output: str, facts: dict) -> list[str]:
    """Numbers in the output that do not match (after rounding) any number in the facts."""
    fact_text = json.dumps(facts, default=str)
    allowed = {v for v, _ in extract_numbers(fact_text)}
    bad = []
    for value, dec in extract_numbers(output):
        if value <= 10 and dec == 0:          # small counts ("three holdings", "2 sleeves")
            continue
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if not any(abs(round(a, dec) - value) <= tol for a in allowed):
            bad.append(f"{value:,.{dec}f}")
    return bad


def advice_phrases(text: str) -> list[str]:
    hits = []
    for p in ADVICE_PATTERNS:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            hits.append(m.group())
    return hits


@dataclass
class Checks:
    numbers_grounded: bool
    ungrounded: list[str]
    no_advice: bool
    advice_hits: list[str]
    within_length: bool
    word_count: int
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.passed = self.numbers_grounded and self.no_advice and self.within_length


def run_checks(text: str, facts: dict, max_words: int) -> Checks:
    bad = ungrounded_numbers(text, facts)
    adv = advice_phrases(text)
    wc = len(text.split())
    return Checks(not bad, bad, not adv, adv, wc <= max_words, wc)


# --------------------------------------------------------------------------- model call
class LiveCallError(RuntimeError):
    pass


def _call_model(system: str, user: str, max_tokens: int = 600) -> str:
    key = _secret("ANTHROPIC_API_KEY")
    if not key:
        raise LiveCallError("no API key configured")
    if not CAP.try_consume():
        raise LiveCallError(f"daily cap of {CAP.limit} calls reached")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=1)
        resp = client.messages.create(
            model=model_name(), max_tokens=max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    except Exception as e:  # network, auth, model name: degrade to templates, never crash the page
        raise LiveCallError(f"{type(e).__name__}: {str(e)[:160]}") from e


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group())


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- feature 1: explain a break
_BREAK_PLAYBOOK = {
    "MISSING_AT_CUSTODIAN": ("The custodian has no record of a confirmed trade we booked. Most often the "
                             "settlement instruction was not sent or was rejected.",
                             "Check the outbound instruction status with the custodian; resend if rejected.",
                             "Trade support"),
    "MISSING_IN_BOOK": ("The custodian booked a trade with no client reference that we do not have. It may "
                        "be a trade placed outside our order system or booked to the wrong account.",
                        "Ask the custodian for the trade ticket; confirm with the advisor before booking.",
                        "Trade support + advisor"),
    "QTY_MISMATCH": ("Both sides have the trade but the quantity or side differs, typically a keying error "
                     "or a partial fill booked as a full fill.",
                     "Compare against the broker execution report; amend whichever side is wrong.",
                     "Trade support"),
    "PRICE_MISMATCH": ("Both sides have the trade but the price differs beyond tolerance: often an average "
                       "price vs. a single fill, or a price before vs. after commission.",
                       "Check the broker confirm for the all-in price; correct the book or dispute.",
                       "Trade support"),
    "SETTLE_DATE_MISMATCH": ("Settlement dates differ. Canadian equities and ETFs settle T+1, so a T+2 date "
                             "suggests a stale settlement convention at one side.",
                             "Confirm the settlement date with the custodian before cash is short.",
                             "Settlements"),
    "UNEXPLAINED_POSITION": ("Holdings differ with no trade-level cause in this window. Common causes are a "
                             "transfer-in not yet booked, a corporate action, or a reinvested distribution.",
                             "Review transfers and corporate actions for this account and security.",
                             "Operations (positions)"),
    "UNBOOKED_DIVIDEND": ("The custodian received a dividend we have not booked, so client cash is understated.",
                          "Book the income from the custodian statement and confirm the rate per share.",
                          "Income processing"),
    "DIVIDEND_AMOUNT_MISMATCH": ("The dividend amount differs, often because one side used a different "
                                 "record-date quantity or rate.",
                                 "Recompute from record-date units and the announced rate; adjust the book.",
                                 "Income processing"),
    "DIVIDEND_NOT_AT_CUSTODIAN": ("We booked income the custodian has not paid. The payment may be delayed "
                                  "or our entry may be wrong.",
                                  "Confirm the pay date with the issuer's announcement and the custodian.",
                                  "Income processing"),
}

EXPLAIN_SYSTEM = """You are an operations analyst at a wealth manager reviewing reconciliation breaks.
Given one break and its evidence as JSON, explain the most likely root cause and the next step.
Rules:
- Use only numbers that appear in the evidence. Do not compute new figures.
- Be specific to the evidence; say what in the data points to the cause.
- If the evidence is consistent with more than one cause, say so and lower confidence.
- Reply with JSON only: {"likely_cause": str, "evidence": str, "next_step": str, "owner": str,
  "confidence": "high"|"medium"|"low"}. Each string under 60 words."""


def explain_break(brk: dict, evidence: dict, prefer_live: bool = True) -> dict:
    facts = {"break": brk, "evidence": evidence}
    mode, error, out = "template", None, None
    if prefer_live and live_available():
        try:
            out = _parse_json(_call_model(EXPLAIN_SYSTEM, json.dumps(facts, default=str), 500))
            out = {k: str(out.get(k, "")) for k in ("likely_cause", "evidence", "next_step", "owner", "confidence")}
            mode = "live"
        except (LiveCallError, ValueError, json.JSONDecodeError) as e:
            error = str(e)
    if out is None:
        cause, step, owner = _BREAK_PLAYBOOK.get(brk.get("break_type"), ("Unknown break type.", "Escalate.", "Ops"))
        out = {"likely_cause": cause,
               "evidence": f"Book: {brk.get('book_value')}. Custodian: {brk.get('custodian_value')}.",
               "next_step": step, "owner": owner, "confidence": "rule-based"}
    text = " ".join(out[k] for k in ("likely_cause", "evidence", "next_step"))
    checks = run_checks(text, facts, max_words=200)
    out.update(mode=mode, model=model_name() if mode == "live" else None, fallback_reason=error, checks=checks)
    audit({"feature": "explain_break", "mode": mode, "model": out["model"], "input_hash": _hash(facts),
           "break_id": brk.get("break_id"), "output": {k: out[k] for k in ("likely_cause", "next_step")},
           "checks_passed": checks.passed, "fallback_reason": error})
    return out


# --------------------------------------------------------------------------- feature 2: client summary draft
SUMMARY_SYSTEM = """You draft short portfolio update notes that a financial advisor will review, edit and
send to their client. You are given facts as JSON; every figure is already computed.
Rules:
- Use only the figures in the facts, written exactly as given. Never calculate, estimate or forecast.
- Do not give advice or recommend any action, product or trade. Do not predict markets.
- If an asset class is out of its target band, say that the advisor will review it with the client.
- Address the client as "you". Do not mention internal operations, reconciliations or the model name.
- Plain text, no headings or bullet points, at most {max_words} words. Tone: {tone}."""


def _template_summary(f: dict) -> str:
    drift = f.get("out_of_band_sleeves") or []
    drift_line = (f" Your {', '.join(drift)} allocation has moved outside its target range, and your advisor "
                  "will review it with you." if drift else " Your allocation remains within its target ranges.")
    return (f"Here is an update on your {f['model']} portfolio as of {f['as_of']}. Over the last "
            f"{f['period']}, it returned {f['portfolio_return']}, compared with {f['benchmark_return']} for its "
            f"balanced benchmark. Annualized volatility over the period was {f['annualized_volatility']}, and the "
            f"largest decline from a peak was {f['max_drawdown'].lstrip('-')}. The portfolio is currently valued at "
            f"{f['market_value']}, with {f['equity_actual']} in equities and {f['fixed_income_actual']} in fixed "
            f"income.{drift_line}")


def draft_client_summary(facts: dict, tone: str = "Plain and brief", max_words: int = 150,
                         prefer_live: bool = True) -> dict:
    mode, error, text = "template", None, None
    if prefer_live and live_available():
        try:
            system = SUMMARY_SYSTEM.format(max_words=max_words, tone=TONES.get(tone, TONES["Plain and brief"]))
            text = _call_model(system, json.dumps(facts, default=str), 400)
            mode = "live"
        except LiveCallError as e:
            error = str(e)
    if text is None:
        text = _template_summary(facts)
    checks = run_checks(text, facts, max_words=max_words)
    audit({"feature": "client_summary", "mode": mode, "model": model_name() if mode == "live" else None,
           "input_hash": _hash(facts), "account_id": facts.get("account_id"), "tone": tone,
           "output_preview": text[:160], "checks_passed": checks.passed,
           "ungrounded": checks.ungrounded, "advice_hits": checks.advice_hits, "fallback_reason": error})
    return {"text": text, "mode": mode, "model": model_name() if mode == "live" else None,
            "fallback_reason": error, "checks": checks}


# --------------------------------------------------------------------------- feature 3: classification (eval)
CLASSIFY_SYSTEM = """You classify reconciliation breaks between an internal book and a custodian.
You get the raw records from each side (null means the record is missing on that side).
Answer with exactly one label from: """ + ", ".join(BREAK_TYPES) + """.
Price tolerance is the larger of $0.01 or 0.5% of price. Reply with the label only."""


def classify_break_rule(evidence: dict) -> str:
    """Deterministic baseline that mirrors the SQL rules."""
    b, c, kind = evidence.get("book"), evidence.get("custodian"), evidence.get("record_kind")
    if kind == "position":
        return "UNEXPLAINED_POSITION"
    if kind == "cash":
        if b is None:
            return "UNBOOKED_DIVIDEND"
        if c is None:
            return "DIVIDEND_NOT_AT_CUSTODIAN"
        return "DIVIDEND_AMOUNT_MISMATCH"
    if c is None:
        return "MISSING_AT_CUSTODIAN"
    if b is None:
        return "MISSING_IN_BOOK"
    if b["side"] != c["side"] or abs(b["quantity"] - c["quantity"]) > 1e-6:
        return "QTY_MISMATCH"
    if abs(b["price"] - c["price"]) > max(0.01, b["price"] * 0.005):
        return "PRICE_MISMATCH"
    return "SETTLE_DATE_MISMATCH"


def classify_break_llm(evidence: dict) -> str:
    raw = _call_model(CLASSIFY_SYSTEM, json.dumps(evidence, default=str), 20).upper()
    for label in sorted(BREAK_TYPES, key=len, reverse=True):
        if label in raw:
            return label
    return "UNPARSEABLE"
