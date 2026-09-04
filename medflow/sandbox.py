"""A deliberately small sandbox for the model-generated Python in the `pot` workflow.

This is NOT a security boundary against an adversary; it is a guard against the
model importing the network, touching the filesystem, or looping forever. The
project only ever executes code produced by our own prompt against a public
benchmark, and the threat model is documented in the README.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import queue as _queue
import re
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

FORBIDDEN = re.compile(
    r"\b(import|open|exec|eval|compile|__import__|globals|locals|getattr|setattr|"
    r"delattr|input|breakpoint|exit|quit|help)\b"
)

SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int, "len": len,
    "list": list, "map": map, "max": max, "min": min, "pow": pow, "range": range,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "print": print, "ValueError": ValueError,
    "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError, "Exception": Exception,
}

SAFE_GLOBALS_TEMPLATE = {
    "math": math, "datetime": datetime, "date": date, "timedelta": timedelta,
}


def _worker(code: str, q) -> None:
    env = {"__builtins__": SAFE_BUILTINS, **SAFE_GLOBALS_TEMPLATE}
    try:
        exec(code, env)  # noqa: S102 - constrained env, see module docstring
        q.put(("ok", repr(env.get("answer"))))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_python(code: str, timeout: float = 5.0) -> Tuple[Optional[str], Optional[str]]:
    """Execute `code`; return (repr of its `answer` variable, error string)."""
    hit = FORBIDDEN.search(code)
    if hit:
        return None, f"blocked construct: {hit.group(0)}"
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, q))
    p.start()
    # Read the queue with a timeout rather than join()-then-empty(): a child can
    # exit before its write is drained, and empty() would then report "died" for
    # a run that actually succeeded. That race made replays non-deterministic.
    try:
        status, payload = q.get(timeout=timeout)
    except _queue.Empty:
        status, payload = None, None
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)
    if status is None:
        return None, f"timeout after {timeout}s"
    return (payload, None) if status == "ok" else (None, payload)


def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    if blocks:
        return blocks[-1].strip()
    return None
