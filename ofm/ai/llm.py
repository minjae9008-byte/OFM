"""Client for locally hosted language models.

Supported servers:
  * Ollama            (POST {base}/api/chat, default http://localhost:11434)
  * OpenAI-compatible (POST {base}/v1/chat/completions): LM Studio, llama.cpp
    server, vLLM, LocalAI, text-generation-webui ...

Only the Python standard library is used. Requests bypass system proxies by
default because the model runs on this machine or LAN. Responses are parsed
as JSON objects; anything that does not parse is treated as "no answer" and
the caller falls back to the deterministic heuristics.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..config import AIConfig

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


class LocalLLM:
    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self.model = os.environ.get("OFM_LLM_MODEL", cfg.model)
        self._status: LLMStatus | None = None
        handlers = [] if cfg.use_system_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)
        self.calls = 0

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

    # -- chat -----------------------------------------------------------------------
    def chat_json(self, system: str, user: str, images: list[bytes] | None = None) -> dict | None:
        st = self.status()
        if not st.available:
            return None
        images = images if (images and self.cfg.vision) else None
        try:
            self.calls += 1
            if st.provider == "ollama":
                msg = {"role": "user", "content": user}
                if images:
                    msg["images"] = [base64.b64encode(b).decode("ascii") for b in images]
                out = self._request(f"{st.base_url}/api/chat", {
                    "model": self.model, "stream": False, "format": "json",
                    "messages": [{"role": "system", "content": system}, msg],
                    "options": {"temperature": self.cfg.temperature},
                })
                text = out.get("message", {}).get("content", "")
            else:
                content: list | str = user
                if images:
                    content = [{"type": "text", "text": user}] + [
                        {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode("ascii")}}
                        for b in images]
                payload = {
                    "model": self.model, "temperature": self.cfg.temperature,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                    "response_format": {"type": "json_object"},
                }
                try:
                    out = self._request(f"{st.base_url}/v1/chat/completions", payload)
                except urllib.error.HTTPError:
                    payload.pop("response_format")       # some servers reject json mode
                    out = self._request(f"{st.base_url}/v1/chat/completions", payload)
                text = out["choices"][0]["message"]["content"]
            return extract_json(text)
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TimeoutError):
            return None
