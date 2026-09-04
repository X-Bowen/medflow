"""OpenAI-compatible LLM client with an on-disk cache for exact replay.

Three modes:
  auto    - use a cached response when the exact request was seen before,
            otherwise call the API and append the response to the cache.
  offline - cache only. A cache miss is a hard error. This is how a grader
            reproduces every published number with no API key and no GPU.
  live    - always call the API (still writes the cache).

The cache key is a sha256 over the full request (model, temperature,
max_tokens, messages, extra_body), so any prompt change invalidates it rather
than silently returning a stale answer.

Two quirks of open-weight reasoning models served behind a vLLM gateway are
handled here, because both look like a broken endpoint if you do not name them:
  * a reasoning model that spends its whole budget thinking returns
    `content: None` with `finish_reason="length"`, everything in
    `reasoning_content`. We surface that as an explicit marker rather than an
    empty string that would silently score as a parse failure.
  * `disable_thinking=True` sends vLLM's
    `chat_template_kwargs.enable_thinking=false`, which is vendor-specific.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class CacheMiss(RuntimeError):
    pass


def _key(payload: Dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_s: float = 0.0        # server-side latency summed over calls, live or recorded
    errors: int = 0

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["llm_s"] = round(d["llm_s"], 2)
        return d


@dataclass
class LLMClient:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    mode: str = "auto"                     # auto | offline | live
    cache_path: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    disable_thinking: bool = False
    max_retries: int = 4
    usage: Usage = field(default_factory=Usage)
    _cache: Dict[str, Dict] = field(default_factory=dict, repr=False)
    _client: object = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.cache_path:
            self._load_cache()

    # ---------------- cache ----------------
    def _load_cache(self) -> None:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # First write wins. A live re-run appends a fresh response for
                # an already-recorded request (these endpoints are not bit-exact
                # even at temperature 0), and replay must keep reproducing the
                # run the README reports rather than drifting to the newest one.
                self._cache.setdefault(rec["key"], rec)

    def _append_cache(self, rec: Dict) -> None:
        """Append one record. Holds a lock so --workers > 1 cannot interleave lines."""
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with self._lock:
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------- api ----------------
    def _lazy_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The `openai` package is required for live calls. "
                    "Install it (pip install -r requirements.txt) or use --mode offline."
                ) from exc
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "No API key. Set OPENAI_API_KEY (and OPENAI_BASE_URL for a "
                    "non-OpenAI endpoint), or run with --mode offline to replay "
                    "the shipped cache."
                )
            self._client = OpenAI(
                api_key=key,
                base_url=self.base_url or os.environ.get("OPENAI_BASE_URL") or None,
            )
        return self._client

    def chat(self, messages: List[Dict[str, str]], tag: str = "") -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if self.disable_thinking else None
        )
        k = _key({**payload, "extra_body": extra_body})

        if self.mode in ("auto", "offline") and k in self._cache:
            rec = self._cache[k]
            with self._lock:
                # Charge the recorded cost, so token and latency accounting is
                # the same number whether a run was live or replayed.
                self.usage.cached += 1
                self.usage.prompt_tokens += rec.get("prompt_tokens", 0)
                self.usage.completion_tokens += rec.get("completion_tokens", 0)
                self.usage.llm_s += rec.get("latency_s", 0.0)
            return rec["text"]
        if self.mode == "offline":
            raise CacheMiss(
                f"cache miss for tag={tag!r} (key {k[:12]}). The shipped cache covers "
                "the exact configs listed in the README; change --model/--workflow/--n "
                "back, or run with --mode auto and an API key."
            )

        client = self._lazy_client()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                call = dict(payload)
                if extra_body:
                    call["extra_body"] = extra_body
                resp = client.chat.completions.create(**call)
                dt = time.time() - t0
                choice = resp.choices[0]
                text = choice.message.content
                finish = getattr(choice, "finish_reason", None)
                if not text:
                    # A reasoning model that burned the budget thinking returns
                    # content=None with everything in reasoning_content.
                    thinking = getattr(choice.message, "reasoning_content", None) or ""
                    text = (
                        f"[NO CONTENT: finish_reason={finish}; the model returned only "
                        f"reasoning_content ({len(thinking)} chars). Raise --max-tokens "
                        f"or pass --disable-thinking.]"
                    )
                u = getattr(resp, "usage", None)
                rec = {
                    "key": k,
                    "tag": tag,
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                    "text": text,
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "finish_reason": finish,
                    "latency_s": round(dt, 3),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._append_cache(rec)
                with self._lock:
                    self._cache[k] = rec
                    self.usage.calls += 1
                    self.usage.prompt_tokens += rec["prompt_tokens"]
                    self.usage.completion_tokens += rec["completion_tokens"]
                    self.usage.llm_s += dt
                return text
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last_err = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        with self._lock:
            self.usage.errors += 1
        raise RuntimeError(f"LLM call failed after {self.max_retries} tries: {last_err}")
