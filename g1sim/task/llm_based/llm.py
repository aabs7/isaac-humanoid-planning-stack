"""Minimal Ollama chat client -- the only place the planner talks to an LLM.

Zero third-party deps (stdlib ``urllib``): a thin wrapper over Ollama's
``/api/chat`` endpoint with JSON-schema-constrained output. Kept deliberately small
and close to the HTTP contract so swapping to another OpenAI-compatible server later
is a small change, not a rewrite.

Prereqs (one-time, the user runs these):
    curl -fsSL https://ollama.com/install.sh | sh     # if not already installed
    ollama pull qwen2.5:7b-instruct                   # ~5 GB, fits the RTX 5090 easily
    # `ollama serve` runs as a daemon after install; verify: curl localhost:11434/api/tags
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_BASE_URL = "http://localhost:11434"


class LLMError(RuntimeError):
    """Raised when the LLM server is unreachable or returns an error. Carries a
    hint about the usual cause (server not running / model not pulled)."""


class OllamaChat:
    """A single-model chat handle. Call :meth:`chat` with a message list; if a
    ``schema`` is given the reply is constrained to it and returned as a parsed dict."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 temperature: float = 0.0, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: list, schema: Optional[dict] = None) -> str:
        """POST ``messages`` to /api/chat and return the assistant's raw content
        string. If ``schema`` (a JSON schema dict) is given, Ollama constrains the
        output to it, so the returned string is valid JSON for that schema."""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if schema is not None:
            body["format"] = schema
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError(
                f"cannot reach Ollama at {self.base_url} ({e}). Is `ollama serve` "
                f"running?  (verify: curl {self.base_url}/api/tags)") from e
        if "message" not in payload:
            raise LLMError(f"unexpected Ollama response: {payload}")
        return payload["message"]["content"]

    def chat_json(self, messages: list, schema: dict) -> dict:
        """Like :meth:`chat` but parse the schema-constrained reply into a dict."""
        raw = self.chat(messages, schema=schema)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned non-JSON despite schema: {raw!r}") from e

    def available(self) -> bool:
        """True if the server is reachable and the model is pulled. Cheap preflight
        so entry points can fail fast with a clear message."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        # Ollama tags carry a ``:latest`` / ``:tag`` suffix; match on the base too.
        return any(n == self.model or n.split(":")[0] == self.model.split(":")[0]
                   for n in names)
