"""Client for locally hosted language models.

Supported servers:
  * Ollama            (POST {base}/api/chat, default http://localhost:11434)
  * OpenAI-compatible (POST {base}/v1/chat/completions): LM Studio, llama.cpp
    server, vLLM, LocalAI, text-generation-webui ...

Only the Python standard library is used. Requests bypass system proxies by
default because the model runs on this machine or LAN.

Robustness for small local models (settings in ``ai.scope``):
  * Structured outputs: a JSON schema is sent so the server can constrain
    decoding with a grammar (Ollama ``format: <schema>``, OpenAI-style
    ``response_format: json_schema``). Servers that reject it (HTTP 400/422/5xx)
    are retried with plain JSON mode and then without any format; the level
    that worked is remembered for later calls.
  * Generation options: context window (``num_ctx``), output cap
    (``num_predict`` / ``max_tokens``), temperature and a per-call seed so
    runs are reproducible on servers that honour seeds.
  * Repair loop: a reply that is not JSON or violates the schema gets a
    follow-up turn quoting the validation errors (up to ``retries`` extra
    calls), the usual "self-refine with validator feedback" pattern. If no
    attempt validates, the attempt with the fewest errors is returned and the
    caller re-validates each field.
  * Budget: ``max_llm_calls`` completed requests and ``max_llm_seconds`` of
    wall time per client. When the budget is spent ``chat_json`` returns None
    without contacting the server and ``budget_status()`` says why; callers
    then fall back to the deterministic heuristics. A cap of 0 disables the
    model.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..config import AIConfig
from .schema import validate as validate_schema

OLLAMA_DEFAULT = "http://localhost:11434"
OPENAI_DEFAULTS = ["http://localhost:1234", "http://localhost:8080", "http://localhost:8000"]


def extract_json(text: str) -> dict | None:
    """Return the first JSON object in ``text`` (handles code fences and chatter)."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        start = cand.find("{")
        while start >= 0:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cand[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            start = cand.find("{", start + 1)
    return None


@dataclass
class LLMStatus:
    provider: str
    base_url: str
    model: str
    available: bool
    detail: str = ""


REPAIR_PROMPT = ("Your previous reply could not be used:\n{errors}\n"
                 "Reply again with ONE corrected JSON object only, following the same format. No prose.")
FORMAT_LEVELS = ("schema", "json", "plain")
_FALLBACK_HTTP = (400, 415, 422, 500, 501)


class LocalLLM:
    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self.scope = cfg.scope
        self.model = os.environ.get("OFM_LLM_MODEL", cfg.model)
        self._status: LLMStatus | None = None
        handlers = [] if cfg.use_system_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)
        self.calls = 0                  # completed chat requests (count against max_llm_calls)
        self.seconds = 0.0              # wall time spent in chat requests
        self.repairs = 0                # follow-up turns sent to fix invalid replies
        self.failures = 0               # requests that failed (transport, HTTP, timeout)
        self.exhausted_reason = ""
        self.last_error = ""
        self.last_info: dict[str, Any] = {}
        self._format_level = 0 if self.scope.json_schema else 1

    # -- transport --------------------------------------------------------------
    def _request(self, url: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers={"Content-Type": "application/json"})
        with self._opener.open(req, timeout=timeout or self.cfg.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _probe(self, provider: str, base: str) -> bool:
        try:
            if provider == "ollama":
                tags = self._request(f"{base}/api/tags", timeout=2.0)
                names = [m.get("name", "") for m in tags.get("models", [])]
                if names and self.model not in names and not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
                    self._detail = f"model '{self.model}' not pulled (available: {', '.join(names[:6])})"
                    return False
                return True
            models = self._request(f"{base}/v1/models", timeout=2.0)
            ids = [m.get("id", "") for m in models.get("data", [])]
            if ids and self.model not in ids:
                # most single-model servers accept any name; prefer the loaded one
                self.model = ids[0]
            return True
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False

    def status(self) -> LLMStatus:
        if self._status is not None:
            return self._status
        self._detail = ""
        provider = self.cfg.provider
        base = os.environ.get("OFM_LLM_URL", self.cfg.base_url).rstrip("/")
        if provider == "none":
            self._status = LLMStatus("none", "", self.model, False, "disabled")
            return self._status
        tries: list[tuple[str, str]] = []
        if provider in ("auto", "ollama"):
            tries.append(("ollama", base or OLLAMA_DEFAULT))
        if provider in ("auto", "openai"):
            tries += [("openai", b) for b in ([base] if base else OPENAI_DEFAULTS)]
        for prov, url in tries:
            if self._probe(prov, url):
                self._status = LLMStatus(prov, url, self.model, True)
                return self._status
        self._status = LLMStatus(provider, base, self.model, False,
                                 self._detail or "no local model server reachable")
        return self._status

    @property
    def available(self) -> bool:
        return self.status().available

    # -- budget -----------------------------------------------------------------------
    def _check_budget(self) -> bool:
        sc = self.scope
        if self.calls >= sc.max_llm_calls:
            self.exhausted_reason = f"call budget spent ({self.calls}/{sc.max_llm_calls} calls)"
        elif self.seconds >= sc.max_llm_seconds:
            self.exhausted_reason = f"time budget spent ({self.seconds:.1f}/{sc.max_llm_seconds:g} s)"
        else:
            return True
        return False

    @property
    def budget_exhausted(self) -> bool:
        return not self._check_budget()

    def budget_status(self) -> dict:
        ok = self._check_budget()
        return {"calls": self.calls, "max_calls": self.scope.max_llm_calls,
                "seconds": round(self.seconds, 2), "max_seconds": self.scope.max_llm_seconds,
                "repairs": self.repairs, "failures": self.failures,
                "exhausted": not ok, "reason": "" if ok else self.exhausted_reason}

    def prompt_char_budget(self, system: str = "") -> int:
        """Largest user prompt (chars) that leaves room in ``num_ctx`` for the reply and one repair
        turn, at a conservative ~3 characters per token for JSON-heavy text."""
        sc = self.scope
        reserve = sc.max_output_tokens + (700 if sc.retries > 0 else 0)
        by_ctx = int((sc.num_ctx - reserve) * 3.0) - len(system)
        return min(int(sc.max_prompt_chars), max(800, by_ctx))

    # -- chat -----------------------------------------------------------------------
    def _post(self, st: LLMStatus, messages: list[dict], schema: dict | None, name: str,
              seed: int | None, temperature: float | None) -> str:
        """One chat request with format fallback. Returns the reply text (raises on transport errors)."""
        sc = self.scope
        temp = self.cfg.temperature if temperature is None else float(temperature)
        level = self._format_level if schema is not None else max(self._format_level, 1)
        while True:
            mode = FORMAT_LEVELS[level]
            remaining = sc.max_llm_seconds - self.seconds
            timeout = max(1.0, min(float(self.cfg.timeout), remaining))
            if st.provider == "ollama":
                options: dict[str, Any] = {"temperature": temp, "num_ctx": int(sc.num_ctx),
                                           "num_predict": int(sc.max_output_tokens)}
                if seed is not None:
                    options["seed"] = int(seed)
                payload: dict[str, Any] = {"model": self.model, "stream": False, "messages": messages,
                                           "options": options}
                if mode == "schema":
                    payload["format"] = schema
                elif mode == "json":
                    payload["format"] = "json"
                url = f"{st.base_url}/api/chat"
            else:
                payload = {"model": self.model, "temperature": temp, "messages": messages,
                           "max_tokens": int(sc.max_output_tokens)}
                if seed is not None:
                    payload["seed"] = int(seed)
                if mode == "schema":
                    payload["response_format"] = {"type": "json_schema",
                                                  "json_schema": {"name": name, "schema": schema}}
                elif mode == "json":
                    payload["response_format"] = {"type": "json_object"}
                url = f"{st.base_url}/v1/chat/completions"
            t0 = time.perf_counter()
            try:
                out = self._request(url, payload, timeout=timeout)
            except urllib.error.HTTPError as exc:
                self.seconds += time.perf_counter() - t0
                if exc.code in _FALLBACK_HTTP and level < len(FORMAT_LEVELS) - 1:
                    level += 1                       # server rejected the format: degrade and retry
                    self._format_level = max(self._format_level, level)
                    continue
                raise
            except BaseException:
                self.seconds += time.perf_counter() - t0
                raise
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            self.last_info["format"] = mode
            if st.provider == "ollama":
                return str(out.get("message", {}).get("content", ""))
            return str(out["choices"][0]["message"]["content"])

    def chat_json(self, system: str, user: str, images: list[bytes] | None = None, schema: dict | None = None,
                  seed: int | None = None, name: str = "ofm_reply", retries: int | None = None,
                  temperature: float | None = None) -> dict | None:
        """Ask for one JSON object. Validates against ``schema`` and runs the repair loop.

        Returns None when the model is unavailable, the budget is spent or no attempt parsed."""
        self.last_info = {"calls": 0, "repairs": 0, "errors": [], "valid": False, "format": None}
        st = self.status()
        if not st.available:
            self.last_error = st.detail or "model unavailable"
            return None
        images = images if (images and self.cfg.vision) else None
        user_msg: dict[str, Any] = {"role": "user", "content": user}
        if images:
            if st.provider == "ollama":
                user_msg["images"] = [base64.b64encode(b).decode("ascii") for b in images]
            else:
                user_msg["content"] = [{"type": "text", "text": user}] + [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode("ascii")}}
                    for b in images]
        messages: list[dict] = [{"role": "system", "content": system}, user_msg]
        retries = self.scope.retries if retries is None else retries
        best: dict | None = None
        best_errors: list[str] = []
        for attempt in range(1 + max(0, int(retries))):
            if not self._check_budget():
                self.last_error = self.exhausted_reason
                break
            try:
                text = self._post(st, messages, schema if self.scope.json_schema else None, name,
                                  None if seed is None else seed + attempt, temperature)
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TimeoutError) as exc:
                self.failures += 1
                self.last_error = f"request failed: {exc}"[:300]
                break
            self.last_info["calls"] += 1
            if attempt:
                self.repairs += 1
                self.last_info["repairs"] += 1
            obj = extract_json(text)
            if obj is None:
                errors = ["the reply is not a JSON object"]
            else:
                errors = validate_schema(obj, schema) if schema is not None else []
                if best is None or len(errors) < len(best_errors):
                    best, best_errors = obj, errors
            if obj is not None and not errors:
                self.last_info.update(valid=True, errors=[])
                self.last_error = ""
                return obj
            self.last_info["errors"] = errors
            messages = messages + [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content": REPAIR_PROMPT.format(errors="\n".join("- " + e for e in errors[:12]))},
            ]
        if best is not None:
            self.last_info["errors"] = best_errors
            self.last_error = "reply did not fully match the schema"
        return best
