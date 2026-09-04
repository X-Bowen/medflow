#!/usr/bin/env python3
"""Run a MedFlow baseline workflow over MedCalc-Bench and score it.

Examples
--------
  # single concrete test case, no API key needed (replays the shipped cache)
  python run_baseline.py --input examples/test1.json --mode offline

  # the reported 50-case baseline, replayed offline
  python run_baseline.py --n 50 --workflow cot --mode offline

  # live run against any OpenAI-compatible endpoint
  OPENAI_API_KEY=sk-... python run_baseline.py --n 50 --workflow cot --mode auto
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medflow.data import DEFAULT_CSV, load_rows, stratified_subset  # noqa: E402
from medflow.evaluate import aggregate, score_one  # noqa: E402
from medflow.llm import CacheMiss, LLMClient  # noqa: E402
from medflow.mock import MockClient  # noqa: E402
from medflow.workflows import WORKFLOWS  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(REPO, "runs", "cache.jsonl")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workflow", default="cot", choices=sorted(WORKFLOWS), help="workflow to run (default: cot = the baseline)")
    p.add_argument("--n", type=int, default=50, help="number of cases from the stratified subset (default: 50)")
    p.add_argument("--seed", type=int, default=0, help="subset seed (default: 0)")
    p.add_argument("--input", default=None, help="JSON file with explicit case ids, e.g. examples/test1.json")
    p.add_argument("--csv", default=DEFAULT_CSV, help="path to MedCalc-Bench test_data.csv")
    # The defaults are exactly the configuration runs/cache.jsonl was recorded
    # with, so the bare command in the README replays without extra flags.
    p.add_argument("--model", default=os.environ.get("MEDFLOW_MODEL", "qwen3-30b-a3b-instruct-2507"))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--disable-thinking", action="store_true",
                   help="send vLLM chat_template_kwargs.enable_thinking=false (for hybrid reasoning models)")
    p.add_argument("--mode", default="offline", choices=["offline", "auto", "live", "mock"],
                   help="offline=cache only (default, no key needed); auto=cache then API; live=always API; mock=fake LLM for plumbing tests")
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--out", default=None, help="where to write results JSON (default: runs/<workflow>_<model>_n<N>.json)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel LLM requests (default: 4). Results stay in case order; "
                        "set 1 for strictly sequential.")
    p.add_argument("--quiet", action="store_true")
    return p


def select_rows(args, rows):
    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            spec = json.load(fh)
        want = spec["case_ids"] if isinstance(spec, dict) else list(spec)
        index = {r["id"]: r for r in rows}
        missing = [c for c in want if c not in index]
        if missing:
            raise SystemExit(f"unknown case ids in {args.input}: {missing}")
        return [index[c] for c in want]
    return stratified_subset(rows, args.n, args.seed)


def main() -> int:
    args = build_parser().parse_args()
    rows = load_rows(args.csv)
    cases = select_rows(args, rows)

    if args.mode == "mock":
        client = MockClient(rows)
        model_name = "mock"
    else:
        client = LLMClient(
            model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
            mode=args.mode, cache_path=args.cache, base_url=args.base_url,
            disable_thinking=args.disable_thinking,
        )
        model_name = args.model

    run = WORKFLOWS[args.workflow]
    out_path = args.out or os.path.join(
        REPO, "runs", f"{args.workflow}_{model_name.replace('/', '-')}_n{len(cases)}.json"
    )

    print(f"MedFlow baseline | workflow={args.workflow} model={model_name} "
          f"mode={args.mode} cases={len(cases)}")
    if args.mode == "mock":
        print("  !! mock mode: the fake LLM is fed the ground truth. Accuracy here is "
              "a plumbing check, NOT a result.")
    print("-" * 78)

    t0 = time.time()
    miss: List[str] = []
    print_lock = threading.Lock()
    done = [0]

    def one(row):
        if miss:                      # a sibling already missed; stop doing work
            return None, None
        try:
            out = run(client, row)
            text = out["answer_text"]
        except CacheMiss as exc:
            miss.append(str(exc))
            return None, None
        except Exception as exc:  # noqa: BLE001
            out, text = {"trace": [], "error": str(exc)}, ""
        rec = score_one(text, row)
        rec["fallback"] = bool(out.get("fallback"))
        rec["error"] = out.get("error")
        trace = {"id": row["id"], "trace": out.get("trace", []), "answer_text": text}
        if not args.quiet:
            with print_lock:
                done[0] += 1
                mark = "PASS" if rec["correct"] else "FAIL"
                print(f"[{done[0]:>3}/{len(cases)}] {mark}  {row['id']:<8} "
                      f"pred={str(rec['prediction']):<12} gt={rec['ground_truth']:<12} "
                      f"{row['Calculator Name'][:38]}", flush=True)
        return rec, trace

    workers = max(1, args.workers if args.mode != "mock" else 1)
    if workers == 1:
        pairs = [one(r) for r in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(one, cases))   # map preserves input order
    if miss:
        print(f"\nCACHE MISS: {miss[0]}", file=sys.stderr)
        print("\nThe shipped cache was recorded with:\n"
              f"  --model qwen3-30b-a3b-instruct-2507 --temperature 0.0 --max-tokens 1500\n"
              f"  --workflow {{direct,cot,pot}} --n 50 --seed 0   (and examples/test1.json)\n"
              f"You asked for: --model {args.model} --temperature {args.temperature} "
              f"--max-tokens {args.max_tokens} --workflow {args.workflow} --n {args.n} "
              f"--seed {args.seed}", file=sys.stderr)
        return 2
    results = [p_[0] for p_ in pairs]
    traces = [p_[1] for p_ in pairs]

    summary = aggregate(results)
    summary.update({
        "workflow": args.workflow, "model": model_name, "mode": args.mode,
        "temperature": args.temperature, "seed": args.seed,
        "max_tokens": args.max_tokens, "disable_thinking": args.disable_thinking,
        "workers": workers,
        "wall_clock_s": round(time.time() - t0, 2),
        "usage": client.usage.as_dict(),
    })

    print("-" * 78)
    print(f"accuracy            : {summary['accuracy']:.1%}  "
          f"({sum(r['correct'] for r in results)}/{summary['n']})")
    print(f"parse failure rate  : {summary['parse_failure_rate']:.1%}")
    print("by category         : " + "  ".join(
        f"{k}={v['acc']:.0%}(n={v['n']})" for k, v in summary["by_category"].items()))
    print("by output type      : " + "  ".join(
        f"{k}={v['acc']:.0%}(n={v['n']})" for k, v in summary["by_output_type"].items()))
    u = summary["usage"]
    print(f"llm calls           : {u['calls']} live, {u['cached']} from cache")
    print(f"tokens              : {u['prompt_tokens']} prompt + {u['completion_tokens']} completion")
    print(f"llm time            : {u['llm_s']}s summed over calls (recorded, workers-independent)")
    print(f"wall clock          : {summary['wall_clock_s']}s")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2)
    trace_path = out_path.replace(".json", ".traces.json")
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(traces, fh, indent=2)
    print(f"\nresults -> {os.path.relpath(out_path, REPO)}")
    print(f"traces  -> {os.path.relpath(trace_path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
