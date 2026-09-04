#!/usr/bin/env python3
"""Drop later duplicate recordings of the same request from runs/cache.jsonl.

Live re-runs append a fresh response for a request that is already recorded,
because these endpoints are not bit-exact even at temperature 0. Replay keeps
the first recording, so the extra lines are dead weight; this rewrites the file
to only the lines replay actually uses.

    python scripts/dedupe_cache.py [--check]
"""
from __future__ import annotations

import json
import os
import sys

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "runs", "cache.jsonl")


def main() -> int:
    check = "--check" in sys.argv
    seen, keep, dropped = set(), [], 0
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            k = json.loads(line)["key"]
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
            keep.append(line)
    print(f"{len(keep)} unique requests, {dropped} later duplicates")
    if check:
        return 1 if dropped else 0
    if dropped:
        with open(CACHE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(keep) + "\n")
        print(f"rewrote {os.path.relpath(CACHE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
