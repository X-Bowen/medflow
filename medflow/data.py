"""Dataset loading for MedCalc-Bench Verified.

We use **MedCalc-Bench Verified**, the maintained successor to the original
NeurIPS 2024 release, not the original. A 2026 audit found formula, threshold
and implementation errors in the original that propagated into ground-truth
labels across whole calculator categories; searching workflows against wrong
labels would fit bugs. Diffing the two releases over the 1,100 shared test rows:
15 answers and their tolerance bands changed, concentrated in three rule-based
scores (Caprini 11, HEART 3, Child-Pugh 1), plus 80 questions and 24 patient
notes reworded.

Verified is ungated on HuggingFace, so nothing here needs an approval step. Both
splits are pinned by sha256 so everyone evaluates on byte-identical data.

The train split (10,538 rows, 49 MB) is a build-time dependency only - it is used
to generate the calculator reference cards for the open-book operator and is not
committed. The test split is committed.
"""
from __future__ import annotations

import csv
import hashlib
import os
import random
import sys
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional

_HF = "https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified/resolve/main"

SPLITS = {
    "test": {
        "url": f"{_HF}/test_data.csv",
        "sha256": "34579020b9c7127f7956f785f91c432f2d0d287e0fb973b27ecd787586dc8798",
        "rows": 1100,
        "file": "test_data.csv",
    },
    "train": {
        "url": f"{_HF}/train_data.csv",
        "sha256": "503db8197c55438640e66bb8a20a114ef2d5e8c6a1b12b79ccd1aa9c3e33b2c1",
        "rows": 10538,
        "file": "train_data.csv",
    },
}

# Kept for reference: the original, superseded release.
ORIGINAL_URL = (
    "https://raw.githubusercontent.com/ncbi-nlp/MedCalc-Bench/main/datasets/test_data.csv"
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_CSV = os.path.join(DATA_DIR, SPLITS["test"]["file"])
DATA_URL = SPLITS["test"]["url"]
DATA_SHA256 = SPLITS["test"]["sha256"]
EXPECTED_ROWS = SPLITS["test"]["rows"]

csv.field_size_limit(10 ** 7)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(path: str = DEFAULT_CSV, verify: bool = True, split: str = "test") -> str:
    """Download a MedCalc-Bench Verified split if absent; verify its checksum."""
    spec = SPLITS[split]
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"[data] downloading MedCalc-Bench Verified {split} split -> {path}",
              file=sys.stderr)
        urllib.request.urlretrieve(spec["url"], path)
    if verify:
        got = _sha256(path)
        if got != spec["sha256"]:
            raise RuntimeError(
                f"checksum mismatch for {path}\n  expected {spec['sha256']}\n  got      {got}\n"
                "Delete the file and re-run to re-download."
            )
    return path


def load_rows(path: Optional[str] = None, split: str = "test") -> List[Dict[str, str]]:
    """Load one split. `path` overrides the default location for that split."""
    spec = SPLITS[split]
    path = path or os.path.join(DATA_DIR, spec["file"])
    ensure_dataset(path, split=split)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != spec["rows"]:
        raise RuntimeError(f"expected {spec['rows']} rows in {split}, got {len(rows)}")
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
