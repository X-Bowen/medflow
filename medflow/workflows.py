"""Workflows = compositions of operators over one MedCalc-Bench case.

Each workflow takes (client, row) and returns a dict with the final answer
text plus a step-by-step trace. Keeping them behind one signature is what
makes the next phase (automated workflow search) a drop-in: the searcher
emits a new `run(client, row)` body over the same operator vocabulary and the
same evaluator scores it.

Operator vocabulary in this baseline:
  op_answer_only   - one call, value only
  op_cot           - one call, reason then value            <-- BASELINE
  op_extract       - one call, pull the needed clinical values out of the note
  op_program       - one call, emit Python; executed in the sandbox
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .evaluate import is_gestational
from .sandbox import extract_code, run_python

SYSTEM = (
    "You are a careful clinical calculation assistant. You compute standard "
    "medical scores and formulas from a patient note. Be precise with units "
    "and do not round intermediate values."
)

_FORMAT_HEAD = (
    "End your reply with a single final line in exactly this form:\n"
    "ANSWER: <value>\n"
)


def format_rule(row: Dict[str, str]) -> str:
    """The answer-format instruction, matched to what this case expects."""
    if is_gestational(row):
        return _FORMAT_HEAD + "The value must be a tuple like (29 weeks, 3 days)."
    if row["Output Type"] == "date":
        return _FORMAT_HEAD + "The value must be a date as mm/dd/yyyy, nothing else."
    if row["Output Type"] == "integer":
        return _FORMAT_HEAD + "The value must be a bare integer (no units, no words)."
    return _FORMAT_HEAD + "The value must be a bare number (no units, no words)."


def _case(row: Dict[str, str]) -> str:
    return (
        f"Patient note:\n{row['Patient Note']}\n\n"
        f"Question: {row['Question']}"
    )


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
def op_answer_only(client, row: Dict[str, str]) -> Dict:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _case(row) + "\n\nGive only the final value. " + format_rule(row)},
    ]
    return {"name": "answer_only", "text": client.chat(msgs, tag=f"{row['id']}:answer_only")}


def op_cot(client, row: Dict[str, str]) -> Dict:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            _case(row)
            + "\n\nWork step by step: state the formula or rule, list the values you "
              "read from the note with their units, convert units as needed, then "
              "compute.\n\n" + format_rule(row)
        )},
    ]
    return {"name": "cot", "text": client.chat(msgs, tag=f"{row['id']}:cot")}


def op_extract(client, row: Dict[str, str]) -> Dict:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            _case(row)
            + "\n\nDo NOT compute anything yet. List only the clinical values this "
              "calculation needs, one per line, as `name: value unit`. If a value is "
              "absent from the note, write `name: NOT REPORTED`."
        )},
    ]
    return {"name": "extract", "text": client.chat(msgs, tag=f"{row['id']}:extract")}


def op_program(client, row: Dict[str, str], facts: str = "") -> Dict:
    ctx = f"\n\nValues already extracted from the note:\n{facts}" if facts else ""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            _case(row) + ctx
            + "\n\nWrite Python that computes the answer. Put every value you read "
              "from the note in as a literal. Assign the final result to a variable "
              "named `answer` (a float, an int, a 'mm/dd/yyyy' string, or a "
              "(weeks, days) tuple if the question asks for a gestational age).\n"
              "Only `math`, `datetime`, `date` and `timedelta` are available and "
              "`import` is not allowed. Reply with one ```python code block and "
              "nothing else."
        )},
    ]
    text = client.chat(msgs, tag=f"{row['id']}:program")
    code = extract_code(text)
    step = {"name": "program", "text": text, "code": code}
    if not code:
        step["error"] = "no code block in reply"
        return step
    value, err = run_python(code)
    step["exec_value"], step["error"] = value, err
    return step


# --------------------------------------------------------------------------
# Workflows
# --------------------------------------------------------------------------
def wf_direct(client, row: Dict[str, str]) -> Dict:
    s = op_answer_only(client, row)
    return {"answer_text": s["text"], "trace": [s]}


def wf_cot(client, row: Dict[str, str]) -> Dict:
    """THE BASELINE: a single chain-of-thought call."""
    s = op_cot(client, row)
    return {"answer_text": s["text"], "trace": [s]}


def wf_pot(client, row: Dict[str, str]) -> Dict:
    """extract -> program -> execute. Falls back to CoT if the code fails."""
    trace: List[Dict] = []
    e = op_extract(client, row)
    trace.append(e)
    p = op_program(client, row, facts=e["text"])
    trace.append(p)
    if p.get("exec_value") is not None and p.get("error") is None:
        return {"answer_text": f"ANSWER: {p['exec_value'].strip(chr(39))}", "trace": trace}
    f = op_cot(client, row)
    f["name"] = "cot_fallback"
    trace.append(f)
    return {"answer_text": f["text"], "trace": trace, "fallback": True}


WORKFLOWS: Dict[str, Callable] = {
    "direct": wf_direct,
    "cot": wf_cot,
    "pot": wf_pot,
}
