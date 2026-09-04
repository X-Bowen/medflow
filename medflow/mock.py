"""A deterministic fake LLM used only to smoke-test the plumbing (`--mode mock`).

It never contacts a network and it is not a baseline: it echoes the ground
truth for CoT-style prompts so that a contributor can verify the harness,
scorer and I/O paths end-to-end offline. Reported accuracy under mock mode is
meaningless by construction and the runner labels it as such.
"""
from __future__ import annotations

from typing import Dict, List

from .llm import Usage


class MockClient:
    model = "mock"

    def __init__(self, rows: List[Dict[str, str]]):
        # Several cases share one patient note, so key on note+question.
        self.index = [((r["Patient Note"][:200], r["Question"][:120]), r) for r in rows]
        self.usage = Usage()

    def chat(self, messages: List[Dict[str, str]], tag: str = "") -> str:
        self.usage.calls += 1
        user = messages[-1]["content"]
        row = None
        for (note, question), r in self.index:
            if note and note in user and question and question in user:
                row = r
                break
        if row is None:
            return "ANSWER: 0"
        if "```python" in user or "Write Python" in user:
            gt = row["Ground Truth Answer"]
            lit = f"'{gt}'" if row["Output Type"] == "date" else gt
            return f"```python\nanswer = {lit}\n```"
        if "Do NOT compute" in user:
            return str(row["Relevant Entities"])
        return f"Mock reasoning for {row['Calculator Name']}.\nANSWER: {row['Ground Truth Answer']}"
