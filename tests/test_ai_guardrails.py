import pytest

from workbench import ai
from workbench.facts import client_facts

FACTS = {"portfolio_return": "6.4%", "benchmark_return": "5.9%", "market_value": "$1,234,000",
         "max_drawdown": "-8.1%", "as_of": "November 03, 2026"}


def test_grounded_numbers_pass():
    text = "Your portfolio returned 6.4% vs 5.9%, is worth $1,234,000, and fell at most 8.1% (Nov 03, 2026)."
    assert ai.ungrounded_numbers(text, FACTS) == []


def test_rounded_figures_still_count_as_grounded():
    assert ai.ungrounded_numbers("It returned about 6% this year.", FACTS) == []


def test_invented_number_is_flagged():
    assert ai.ungrounded_numbers("We expect 12.5% next year.", FACTS) == ["12.5"]


def test_computed_difference_is_flagged():
    # 6.4 - 5.9 = 0.5: the model is not allowed to do arithmetic
    assert "0.5" in ai.ungrounded_numbers("You beat the benchmark by 0.5%.", FACTS)


@pytest.mark.parametrize("text", [
    "You should buy more equities now.", "We recommend selling the bond fund.",
    "This portfolio will outperform next year.", "Returns are guaranteed.",
])
def test_advice_language_is_blocked(text):
    assert ai.advice_phrases(text)


def test_neutral_language_passes():
    assert ai.advice_phrases("Your advisor will review the allocation with you.") == []


def test_template_summary_passes_its_own_checks(ds):
    facts = client_facts(ds, ds.accounts.account_id.iloc[3])
    out = ai.draft_client_summary(facts)
    assert out["mode"] == "template"
    assert out["checks"].passed, out["checks"]


def test_daily_cap_blocks_after_limit(monkeypatch):
    cap = ai.UsageCap()
    monkeypatch.setenv("AI_DAILY_CAP", "2")
    assert cap.try_consume() and cap.try_consume()
    assert not cap.try_consume()


def test_every_call_is_audited(ds):
    before = len(ai.audit_log())
    ai.draft_client_summary(client_facts(ds, ds.accounts.account_id.iloc[0]))
    assert len(ai.audit_log()) == before + 1


def test_live_output_with_invented_number_and_advice_is_blocked(ds, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ai, "_call_model", lambda *a, **k: "You should buy more stocks; expect 14.2% next year.")
    out = ai.draft_client_summary(client_facts(ds, ds.accounts.account_id.iloc[0]))
    assert out["mode"] == "live"
    assert not out["checks"].passed
    assert out["checks"].advice_hits and "14.2" in out["checks"].ungrounded


def test_live_failure_falls_back_to_template(ds, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def boom(*a, **k):
        raise ai.LiveCallError("APIConnectionError: offline")
    monkeypatch.setattr(ai, "_call_model", boom)
    out = ai.draft_client_summary(client_facts(ds, ds.accounts.account_id.iloc[0]))
    assert out["mode"] == "template" and out["checks"].passed and "offline" in out["fallback_reason"]


def test_live_break_explanation_is_parsed(ds, monkeypatch):
    from workbench.recon import break_evidence, run_reconciliation
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ai, "_call_model", lambda *a, **k: 'Sure: {"likely_cause": "Instruction not sent", '
                        '"evidence": "No custodian record", "next_step": "Resend", "owner": "Trade support", '
                        '"confidence": "high"}')
    brk = run_reconciliation(ds).queue.iloc[0].to_dict()
    out = ai.explain_break({k: str(v) for k, v in brk.items()}, break_evidence(ds, brk))
    assert out["mode"] == "live" and out["likely_cause"] == "Instruction not sent"
