"""Dataset loading for MedCalc-Bench (test split).

The CSV is fetched from the official NCBI GitHub mirror, which is ungated
(the HuggingFace copy requires manual approval). We pin a sha256 so that
everyone evaluates on byte-identical data.
"""
from __future__ import annotations

import csv
import hashlib
import os
import random
import sys
import urllib.request
from collections import defaultdict
from typing import Dict, List

DATA_URL = (
    "https://raw.githubusercontent.com/ncbi-nlp/MedCalc-Bench/"
    "main/datasets/test_data.csv"
)
DATA_SHA256 = "c9d219a30e43f50646b73fa5fe5fe86d1ef47e7d9b2c7403a0edb7f3e5290e3a"
EXPECTED_ROWS = 1100

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(REPO_ROOT, "data", "test_data.csv")

csv.field_size_limit(10 ** 7)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(path: str = DEFAULT_CSV, verify: bool = True) -> str:
    """Download the MedCalc-Bench test CSV if absent; verify its checksum."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[data] downloading MedCalc-Bench test split -> {path}", file=sys.stderr)
        urllib.request.urlretrieve(DATA_URL, path)
    if verify:
        got = _sha256(path)
        if got != DATA_SHA256:
            raise RuntimeError(
                f"checksum mismatch for {path}\n  expected {DATA_SHA256}\n  got      {got}\n"
                "Delete the file and re-run to re-download."
            )
    return path


def load_rows(path: str = DEFAULT_CSV) -> List[Dict[str, str]]:
    ensure_dataset(path)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    for r in rows:
        r["id"] = f"row{r['Row Number']}"
    return rows


def stratified_subset(rows: List[Dict[str, str]], n: int, seed: int = 0) -> List[Dict[str, str]]:
    """Deterministic subset that keeps the Category mix of the full test set.

    Sampling is seeded and sorted by Row Number, so the same (n, seed) always
    yields the same cases on any machine.
    """
    if n >= len(rows):
        return list(rows)
    by_cat: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_cat[r["Category"]].append(r)

    rng = random.Random(seed)
    picked: List[Dict[str, str]] = []
    cats = sorted(by_cat)
    # Proportional allocation, then round-robin for the remainder.
    quota = {c: int(round(n * len(by_cat[c]) / len(rows))) for c in cats}
    for c in cats:
        pool = sorted(by_cat[c], key=lambda r: int(r["Row Number"]))
        k = min(quota[c], len(pool))
        picked.extend(rng.sample(pool, k) if k else [])
    leftovers = [r for r in rows if r not in picked]
    leftovers.sort(key=lambda r: int(r["Row Number"]))
    while len(picked) < n and leftovers:
        picked.append(leftovers.pop(rng.randrange(len(leftovers))))
    picked = picked[:n]
    picked.sort(key=lambda r: int(r["Row Number"]))
    return picked
