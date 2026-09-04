#!/usr/bin/env python3
"""Build one reference card per calculator for the open-book operator.

Why this exists
---------------
A 2026 audit of MedCalc-Bench showed that handing the model the calculator's
formula ("open-book") lifted one model from 52% to 81.5%, i.e. much of the
benchmark's apparent difficulty is formula *recall*, not clinical reasoning.
To study what difficulty remains after recall is handled, the open-book
operator needs a specification to retrieve - which is also what a clinician
actually has at the bedside (an MDCalc page), so this is a realistic operator
rather than a cheat.

Leakage control
---------------
Cards are generated from the **train** split only and are keyed by Calculator
ID, never by case. They describe the general formula or scoring rule and carry
no patient-specific values, so nothing about a test case reaches the card.

    OPENAI_API_KEY=... python3 scripts/build_cards.py --out data/calculator_cards.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from medflow.data import load_rows  # noqa: E402
from medflow.llm import LLMClient  # noqa: E402

NUM = re.compile(r"\d")

PROMPT = """You are writing a reference card for a clinical calculator, the kind a
clinician would look up before using it.

Calculator: {name}
Category: {category}
Expected output: {otype}

Below are {k} worked solutions for DIFFERENT patients. Use them only to recover the
general rule. Do NOT copy any patient's numbers.

{examples}

Write the reference card with exactly these sections:

FORMULA OR RULE:
  For an equation: the equation in symbols, with every coefficient.
  For a score: every criterion and the points it contributes, one per line.
INPUTS:
  Each value required, with its expected unit.
CONVENTIONS:
  Unit conversions, rounding, defaults for values not reported in a note,
  and any threshold or boundary rule that is easy to get wrong.

Rules: no patient-specific numbers anywhere. No worked example. Under 220 words."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "calculator_cards.json"))
    ap.add_argument("--model", default=os.environ.get("MEDFLOW_MODEL", "qwen3-30b-a3b-instruct-2507"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--k", type=int, default=3, help="worked solutions shown per calculator")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--cache", default=os.path.join(REPO, "runs", "cache.jsonl"))
    ap.add_argument("--mode", default="auto", choices=["auto", "live", "offline"])
    args = ap.parse_args()

    train = load_rows(split="train")
    by_calc = defaultdict(list)
    for r in train:
        by_calc[r["Calculator ID"]].append(r)

    client = LLMClient(model=args.model, temperature=0.0, max_tokens=args.max_tokens,
                       mode=args.mode, cache_path=args.cache, base_url=args.base_url)

    cards = {}
    ids = sorted(by_calc, key=lambda x: int(x))
    for i, cid in enumerate(ids, 1):
        rows = sorted(by_calc[cid], key=lambda r: int(r["Row Number"]))[: args.k]
        head = rows[0]
        examples = "\n\n".join(
            f"--- worked solution {j} ---\n{r['Ground Truth Explanation']}"
            for j, r in enumerate(rows, 1)
        )
        msg = PROMPT.format(name=head["Calculator Name"], category=head["Category"],
                            otype=head["Output Type"], k=len(rows), examples=examples)
        text = client.chat([{"role": "user", "content": msg}], tag=f"card:{cid}")
        cards[cid] = {
            "calculator_id": cid,
            "name": head["Calculator Name"],
            "category": head["Category"],
            "output_type": head["Output Type"],
            "card": text.strip(),
            "built_from": "train split",
            "n_examples": len(rows),
        }
        print(f"[{i:>2}/{len(ids)}] {head['Calculator Name'][:52]:<52} "
              f"{len(text.split()):>4} words", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(cards, fh, indent=2, ensure_ascii=False)
    print(f"\n{len(cards)} cards -> {os.path.relpath(args.out, REPO)}")
    print(f"llm: {client.usage.calls} live, {client.usage.cached} cached, "
          f"{client.usage.prompt_tokens + client.usage.completion_tokens} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
