"""Measure how well the model classifies reconciliation breaks from raw records alone.

Each synthetic scenario has ground-truth labels for the breaks we injected, so accuracy is exact.
The model sees only the two raw records (book vs custodian), never the label or the SQL result.

    python scripts/eval_ai.py --seeds 3          # uses ANTHROPIC_API_KEY if set, else the rule baseline
Writes docs/eval_results.md, which the AI assistant page displays.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from workbench import ai  # noqa: E402
from workbench.data import generate_dataset  # noqa: E402
from workbench.recon import break_evidence, run_reconciliation  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    live = ai.live_available()
    classify = ai.classify_break_llm if live else ai.classify_break_rule

    rows = []
    for seed in range(100, 100 + args.seeds):
        ds = generate_dataset(seed=seed, as_of=pd.Timestamp("2026-11-03"))
        truth = dict(zip(ds.truth.ref, ds.truth.break_type))
        for brk in run_reconciliation(ds).queue.to_dict("records"):
            pred = classify(break_evidence(ds, brk))
            rows.append((seed, brk["ref"], truth[brk["ref"]], pred))
    df = pd.DataFrame(rows, columns=["seed", "ref", "label", "predicted"])
    acc = (df.label == df.predicted).mean()
    misses = df[df.label != df.predicted]
    per_type = df.assign(ok=df.label == df.predicted).groupby("label").ok.agg(["sum", "count"])

    src = f"live model `{ai.model_name()}`" if live else "rule-based baseline (no API key set)"
    lines = [
        f"Classifier: {src} · {len(df)} labelled breaks across {args.seeds} scenarios · "
        f"run {dt.date.today():%Y-%m-%d}",
        "",
        f"**Accuracy: {acc * 100:.1f}%**",
        "",
        "| Break type | Correct | Total |",
        "| --- | --- | --- |",
        *[f"| {t} | {int(r['sum'])} | {int(r['count'])} |" for t, r in per_type.iterrows()],
    ]
    if len(misses):
        common = Counter(zip(misses.label, misses.predicted)).most_common(3)
        lines += ["", "Most common confusions: " + "; ".join(f"{a} → {b} ({n})" for (a, b), n in common)]
    out = ROOT / "docs" / "eval_results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
