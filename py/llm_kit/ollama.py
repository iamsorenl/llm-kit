"""Ollama /api/chat client (stdlib only)."""
from __future__ import annotations

import json
import os

from .errors import LlmError, Reply, post_json


def ollama_chat(
    messages: list[dict],
    model: str,
    *,
    url: str | None = None,
    format: str | dict | None = None,  # "json" or a JSON schema dict for structured output
    temperature: float = 0.0,
    num_ctx: int | None = None,
    think: bool | None = None,  # omitted unless set
    timeout: float = 90.0,  # a cold model load is ~25s
    options: dict | None = None,
) -> Reply:
    base = (url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    opts = {"temperature": temperature, **({"num_ctx": num_ctx} if num_ctx else {}), **(options or {})}
    body: dict = {"model": model, "stream": False, "messages": messages, "options": opts}
    if format is not None:
        body["format"] = format
    if think is not None:
        body["think"] = think

    status, text, ms = post_json(f"{base}/api/chat", body, timeout)
    if status != 200:
        raise LlmError("api", f"Ollama error ({status}): {text[:300]}", status)
    try:
        env = json.loads(text)
        content = env["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise LlmError("bad-response", "Ollama reply has no message.content") from exc
    if not isinstance(content, str):
        raise LlmError("bad-response", "Ollama reply has no message.content")
    return Reply(content, env.get("prompt_eval_count") or 0, env.get("eval_count") or 0, ms)
