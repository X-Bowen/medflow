#!/usr/bin/env python3
"""Compare finished runs and report the headroom a workflow search could claim.

    python scripts/compare.py runs/direct_*_n50.json runs/cot_*_n50.json runs/pot_*_n50.json

Prints an overall table, a per-category breakdown, and two ceilings:
  best-single    - the best of the given workflows, one workflow for everything
  per-category   - pick the best workflow per calculator category (what an
                   AFlow-style search over this operator set could plausibly reach)
  oracle         - correct if ANY workflow got it right (a loose upper bound)
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

def load(paths):
    runs = {}
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            d = json.load(open(p, encoding="utf-8"))
            runs[d["summary"]["workflow"]] = d
    return runs


def main() -> int:
    args = sys.argv[1:] or ["runs/direct_*_n50.json", "runs/cot_*_n50.json", "runs/pot_*_n50.json",
                            "runs/openbook_cot_*_n50.json", "runs/openbook_pot_*_n50.json"]
    runs = load(args)
    if not runs:
        print("no run files matched", file=sys.stderr)
        return 1
    CANON = ("direct", "cot", "pot", "openbook_cot", "openbook_pot")
    order = [w for w in CANON if w in runs] + [w for w in runs if w not in CANON]
    model = runs[order[0]]["summary"]["model"]
    n = runs[order[0]]["summary"]["n"]
    print(f"model={model}  n={n}\n")

    print(f"{'workflow':<14} {'acc':>7} {'parse-fail':>11} {'calls':>7} {'tokens':>9} {'llm_s':>8}")
    print("-" * 62)
    for w in order:
        s = runs[w]["summary"]
        u = s["usage"]
        tok = u["prompt_tokens"] + u["completion_tokens"]
        print(f"{w:<14} {s['accuracy']:>6.1%} {s['parse_failure_rate']:>11.1%} "
              f"{u['calls'] + u['cached']:>7} {tok:>9} {u.get('llm_s', 0):>8.1f}")

    cats = sorted({r["category"] for r in runs[order[0]]["results"]})
    print(f"\n{'category':<12} {'n':>3} " + " ".join(f"{w:>13}" for w in order) + f" {'best':>14}")
    print("-" * (17 + 14 * len(order) + 15))
    per_cat_correct = 0
    for c in cats:
        accs, ns = {}, 0
        for w in order:
            rs = [r for r in runs[w]["results"] if r["category"] == c]
            ns = len(rs)
            accs[w] = sum(r["correct"] for r in rs) / ns
        best = max(accs, key=lambda w: accs[w])
        per_cat_correct += accs[best] * ns
        print(f"{c:<12} {ns:>3} " + " ".join(f"{accs[w]:>12.0%}" for w in order) + f" {best:>14}")

    by_id = defaultdict(dict)
    for w in order:
        for r in runs[w]["results"]:
            by_id[r["id"]][w] = r["correct"]
    oracle = sum(any(v.values()) for v in by_id.values()) / len(by_id)
    best_single = max(runs[w]["summary"]["accuracy"] for w in order)

    print(f"\nbest single workflow          : {best_single:.1%}")
    print(f"best workflow per category    : {per_cat_correct / n:.1%}   "
          f"({per_cat_correct / n - best_single:+.1%} vs best single)")
    print(f"any-workflow-correct (oracle) : {oracle:.1%}   "
          f"({oracle - best_single:+.1%} vs best single)")
    print("\nThe gap between 'best single' and the two ceilings is the headroom an\n"
          "automated per-task workflow search is trying to capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
