from __future__ import annotations

from .errors import Reply


class FakeChat:
    """Offline stand-in for ollama_chat / compat_chat: returns scripted replies, records calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict], *args, **kwargs) -> Reply:
        self.calls.append(messages)
        if not self._replies:
            raise RuntimeError("FakeChat exhausted")
        return Reply(self._replies.pop(0))
