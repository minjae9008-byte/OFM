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
from typing import Any, Callable

from ..config import AIConfig
from .schema import validate as validate_schema

OLLAMA_DEFAULT = "http://localhost:11434"
OPENAI_DEFAULTS = ["http://localhost:1234", "http://localhost:8080", "http://localhost:8000"]


def extract_json(text: str) -> dict | None:
    """Return the first JSON object in ``text`` (handles code fences and chatter)."""
    cands = json_candidates(text)
    return cands[0] if cands else None


def json_candidates(text: str, limit: int = 6) -> list[dict]:
    """JSON objects that ``text`` may mean, most likely first.

    A clean reply (or a fenced block) gives one object. Otherwise every balanced object is a
    candidate, scanning from the first ``{``; when an object never closes (the reply was cut
    off at the output token limit) its complete members are salvaged as a candidate too,
    before the objects nested inside it. ``chat_json`` keeps the candidate that validates
    best, so a truncated wrapper does not hide a complete payload inside it and vice versa."""
    if not text:
        return []
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    sources = [fence.group(1)] if fence else []
    sources.append(text)
    out: list[dict] = []
    for src in sources:
        try:
            obj = json.loads(src)
            if isinstance(obj, dict):
                return [obj]
        except (json.JSONDecodeError, ValueError):
            pass
        start = src.find("{")
        while start >= 0 and len(out) < limit:
            end = _balanced_end(src, start)
            if end is None:
                part = salvage_object(src, start)
                if part:
                    out.append(part)
                start = src.find("{", start + 1)        # look inside the unterminated object
                continue
            try:
                obj = json.loads(src[start:end + 1])
                if isinstance(obj, dict):
                    out.append(obj)
                    start = src.find("{", end + 1)      # skip its nested objects
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            start = src.find("{", start + 1)
        if out:
            break
    return out


def _balanced_end(text: str, start: int) -> int | None:
    """Index of the brace closing the object that opens at ``start`` (strings respected)."""
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
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
                return i
    return None


def salvage_object(text: str, start: int = 0) -> dict | None:
    """Complete ``"key": value`` members of a JSON object that is truncated after them.

    A member counts when a ``,`` or ``}`` follows its value, or when the text ends right
    after a complete string, object or array; a number cut at the end ("3" of "35", "1." of
    "1.5") or a missing comma ends the salvage."""
    dec = json.JSONDecoder()
    i = text.find("{", start)
    if i < 0:
        return None
    out: dict = {}
    i += 1
    n = len(text)

    def skip(k: int) -> int:
        while k < n and text[k] in " \t\r\n":
            k += 1
        return k

    while True:
        i = skip(i)
        if i >= n or text[i] != '"':
            break
        try:
            key, i = dec.raw_decode(text, i)
            i = skip(i)
            if i >= n or text[i] != ":":
                break
            value, i = dec.raw_decode(text, skip(i + 1))
        except (json.JSONDecodeError, ValueError):
            break
        i = skip(i)
        if i >= n:
            # cut right after the value: strings, objects and arrays are complete, a number
            # may not be ("3" of "35")
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
                out[key] = value
            break
        if text[i] not in ",}":
            break                   # "1." of "1.5", or a missing comma
        out[key] = value
        if text[i] == "}":
            break
        i += 1
    return out or None


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
_FORMAT_REJECTED = (400, 415, 422, 501)


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
                    if exc.code in _FORMAT_REJECTED:
                        # only a clear "format not supported" is remembered; a 5xx may be a
                        # transient failure (model reloading) and must not disable structured
                        # output for the rest of the run
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

    @staticmethod
    def _judge(obj: dict, schema: dict | None, check) -> tuple[list[str], tuple[int, int]]:
        """Errors of one parsed reply and its rank: schema errors first, then semantic ones,
        so a schema-valid reply with a malformed action beats a repair that lost a key."""
        errors = validate_schema(obj, schema) if schema is not None else []
        sem: list[str] = []
        if check is not None and not errors:
            try:
                sem = list(check(obj))[:12]
            except Exception as exc:      # a checker bug must not abort the request
                sem = [f"reply could not be checked ({type(exc).__name__})"]
        return (errors or sem), (len(errors), len(sem))

    def chat_json(self, system: str, user: str, images: list[bytes] | None = None, schema: dict | None = None,
                  seed: int | None = None, name: str = "ofm_reply", retries: int | None = None,
                  temperature: float | None = None,
                  check: Callable[[dict], list[str]] | None = None) -> dict | None:
        """Ask for one JSON object. Validates against ``schema`` and runs the repair loop.

        ``check`` adds semantic errors the schema cannot express (e.g. an ``add_poi`` without a
        region); they are quoted back in the repair turn like schema errors.

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
        base: list[dict] = [{"role": "system", "content": system}, user_msg]
        messages = base
        retries = self.scope.retries if retries is None else retries
        best: dict | None = None
        best_errors: list[str] = []
        best_rank: tuple[int, int] = (1 << 30, 1 << 30)
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
            obj, errors, rank = None, ["the reply is not a JSON object"], None
            for cand in json_candidates(text):
                c_errors, c_rank = self._judge(cand, schema, check)
                if rank is None or c_rank < rank:
                    obj, errors, rank = cand, c_errors, c_rank
            if obj is not None and (best is None or rank < best_rank):
                best, best_errors, best_rank = obj, errors, rank
            if obj is not None and not errors:
                self.last_info.update(valid=True, errors=[])
                self.last_error = ""
                return obj
            self.last_info["errors"] = errors
            # only the latest attempt is quoted back: the prompt budget reserves room for one
            # repair turn, and accumulated turns would overflow num_ctx on the second repair
            messages = base + [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content": REPAIR_PROMPT.format(errors="\n".join("- " + e for e in errors[:12]))},
            ]
        if best is not None:
            self.last_info["errors"] = best_errors
            self.last_error = "reply did not fully match the schema"
        return best
