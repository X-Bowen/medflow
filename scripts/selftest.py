#!/usr/bin/env python3
"""Offline self-test: no API key, no network beyond the one dataset download.

Checks the scorer's three answer formats and then runs all three workflows
end-to-end against the mock LLM, which is fed the ground truth. Every case must
score correct; anything less means the harness itself is broken.

    python scripts/selftest.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from medflow.data import load_rows, stratified_subset  # noqa: E402
from medflow.evaluate import score_one  # noqa: E402
from medflow.sandbox import run_python  # noqa: E402

failures = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")
    if not ok:
        failures.append(label)


def main() -> int:
    print("1. scorer")
    dec = {"id": "d", "Calculator Name": "CrCl", "Category": "lab test", "Output Type": "decimal",
           "Ground Truth Answer": "25.2381", "Lower Limit": "23.97619", "Upper Limit": "26.50001"}
    check("decimal inside band", score_one("ANSWER: 25.3 mL/min", dec)["correct"], True)
    check("decimal outside band", score_one("ANSWER: 40", dec)["correct"], False)
    check("unparseable is wrong", score_one("I cannot answer.", dec)["parsed"], False)

    inte = {"id": "i", "Calculator Name": "CHA2DS2", "Category": "risk", "Output Type": "integer",
            "Ground Truth Answer": "2", "Lower Limit": "2", "Upper Limit": "2"}
    check("integer exact", score_one("ANSWER: 2 points", inte)["correct"], True)

    dat = {"id": "t", "Calculator Name": "EDD", "Category": "date", "Output Type": "date",
           "Ground Truth Answer": "09/11/2014", "Lower Limit": "", "Upper Limit": ""}
    check("date normalised", score_one("ANSWER: 9/11/2014", dat)["correct"], True)

    ges = {"id": "g", "Calculator Name": "EGA", "Category": "date", "Output Type": "date",
           "Ground Truth Answer": "('29 weeks', '3 days')", "Lower Limit": "", "Upper Limit": ""}
    check("weeks/days tuple", score_one("ANSWER: (29 weeks, 3 days)", ges)["correct"], True)

    print("2. sandbox")
    check("arithmetic", run_python("answer = (140-87)*48/(1.4*72)")[0][:6], "25.238")
    check("import blocked", run_python("import os\nanswer=1")[1], "blocked construct: import")
    check("infinite loop killed", run_python("while True: pass", timeout=2)[1], "timeout after 2s")

    print("3. dataset")
    rows = load_rows()
    check("row count", len(rows), 1100)
    def digest(n, seed):
        ids = ",".join(r["id"] for r in stratified_subset(rows, n, seed))
        return hashlib.sha256(ids.encode()).hexdigest()[:12]
    check("subset (20, seed 0) reproducible", digest(20, 0), digest(20, 0))
    check("subset (50, seed 0) reproducible", digest(50, 0), digest(50, 0))
    check("a different seed gives a different subset", digest(20, 1) != digest(20, 0), True)

    print("4. workflows end-to-end (mock LLM)")
    for wf in ("direct", "cot", "pot"):
        out = subprocess.run(
            [sys.executable, "run_baseline.py", "--workflow", wf, "--n", "30",
             "--mode", "mock", "--quiet"],
            cwd=REPO, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if "accuracy" in l), "")
        check(f"{wf} scores 100% against ground-truth mock", "100.0%" in line, True)

    print("5. replay determinism (needs runs/cache.jsonl)")
    cache = os.path.join(REPO, "runs", "cache.jsonl")
    if not os.path.exists(cache):
        print("  skip  no cache present")
    else:
        accs = set()
        for workers in (1, 4, 8):
            out = subprocess.run(
                [sys.executable, "run_baseline.py", "--workflow", "pot", "--n", "50",
                 "--mode", "offline", "--model", "qwen3-30b-a3b-instruct-2507",
                 "--max-tokens", "1500", "--workers", str(workers), "--quiet"],
                cwd=REPO, capture_output=True, text=True,
                env={**os.environ, "OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""})
            line = next((l for l in out.stdout.splitlines() if l.startswith("accuracy")), "")
            accs.add(line.split(":")[1].strip() if ":" in line else out.stderr[-200:])
        check("pot replays identically at --workers 1/4/8", len(accs), 1)
        print(f"       -> {accs.pop() if len(accs) == 1 else accs}")

    print()
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
