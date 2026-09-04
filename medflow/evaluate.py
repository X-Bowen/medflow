"""Answer parsing and scoring for MedCalc-Bench.

Scoring follows the benchmark's own convention:
  decimal  -> correct iff Lower Limit <= pred <= Upper Limit (a 5% band)
  integer  -> exact match after rounding
  date     -> exact match on mm/dd/yyyy, EXCEPT the gestational-age calculator,
              whose ground truth is a ('N weeks', 'M days') tuple and is scored
              as an exact match on that (weeks, days) pair.
A response we cannot parse into a value counts as WRONG and is also tallied
separately as a parse failure, so format brittleness never hides as accuracy.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Sentinel written by llm.py when the endpoint returned no content at all.
# It contains digits, so it must never reach the number fallback.
NO_CONTENT_PREFIX = "[NO CONTENT:"

ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*(.+)", re.IGNORECASE)
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+")
DATE_RE = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})")
GEST_GT_RE = re.compile(r"\(\s*['\"]?\s*(\d+)\s*weeks?.*?(\d+)\s*days?", re.S | re.IGNORECASE)
GEST_PRED_RE = re.compile(r"(\d+)\s*weeks?\D{0,12}?(\d+)\s*days?", re.IGNORECASE)


def extract_answer_field(text: str) -> Optional[str]:
    """Pull the value after the last `ANSWER:` marker."""
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip().strip("*`").strip()


def _to_float(s: str) -> Optional[float]:
    m = NUM_RE.search(s.replace(",", "") if "," in s else s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def is_gestational(row: Dict[str, str]) -> bool:
    """The one `date` calculator whose answer is a (weeks, days) tuple."""
    return bool(GEST_GT_RE.search(row.get("Ground Truth Answer", "")))


def _to_weeks_days(s: str) -> Optional[Tuple[int, int]]:
    m = GEST_PRED_RE.search(s or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _to_date(s: str) -> Optional[str]:
    m = DATE_RE.search(s or "")
    if not m:
        return None
    mm, dd, yyyy = m.groups()
    return f"{int(mm):02d}/{int(dd):02d}/{yyyy}"


def parse_prediction(text: str, output_type: str, gestational: bool = False) -> Tuple[Optional[object], bool]:
    """Return (value, parsed_ok). Falls back to the last number in the text."""
    if (text or "").lstrip().startswith(NO_CONTENT_PREFIX):
        return None, False
    field = extract_answer_field(text)
    source = field if field is not None else (text or "")
    if gestational:
        val = _to_weeks_days(source) or _to_weeks_days(text or "")
        return val, val is not None
    if output_type == "date":
        val = _to_date(source)
        if val is None and field is not None:
            val = _to_date(text or "")
        return val, val is not None
    val = _to_float(source)
    if val is None and field is None:
        nums = NUM_RE.findall((text or "").replace(",", ""))
        if nums:
            try:
                val = float(nums[-1])
            except ValueError:
                val = None
    return val, val is not None


def score_one(text: str, row: Dict[str, str]) -> Dict:
    ot = row["Output Type"]
    gest = is_gestational(row)
    pred, ok = parse_prediction(text, ot, gestational=gest)
    correct = False
    if ok:
        if gest:
            m = GEST_GT_RE.search(row["Ground Truth Answer"])
            correct = pred == (int(m.group(1)), int(m.group(2)))
        elif ot == "date":
            correct = pred == _to_date(row["Ground Truth Answer"])
        elif ot == "integer":
            try:
                correct = round(float(pred)) == round(float(row["Ground Truth Answer"]))
            except (TypeError, ValueError):
                correct = False
        else:  # decimal
            try:
                lo, hi = float(row["Lower Limit"]), float(row["Upper Limit"])
                lo, hi = min(lo, hi), max(lo, hi)
                correct = lo <= float(pred) <= hi
            except (TypeError, ValueError):
                correct = False
    return {
        "id": row["id"],
        "calculator": row["Calculator Name"],
        "category": row["Category"],
        "output_type": "weeks_days" if gest else ot,
        "ground_truth": row["Ground Truth Answer"],
        "prediction": list(pred) if isinstance(pred, tuple) else pred,
        "parsed": ok,
        "correct": bool(correct),
    }


def aggregate(results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    by_cat = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][0] += int(r["correct"])
        by_cat[r["category"]][1] += 1
        by_type[r["output_type"]][0] += int(r["correct"])
        by_type[r["output_type"]][1] += 1
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "parse_failure_rate": sum(not r["parsed"] for r in results) / n,
        "by_category": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_cat.items())},
        "by_output_type": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_type.items())},
    }
