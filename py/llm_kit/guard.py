from __future__ import annotations

import re
from datetime import datetime, timezone

_PAID = ("gpt-4", "gpt-5", "claude")
# short reasoning-model names need boundaries so "o1" cannot match inside a free model id
_PAID_BOUNDED = re.compile(r"(?<![a-z0-9])(o1|o3|o4)(?![a-z0-9])")


def assert_free_model(model: str) -> None:
    """Raise ValueError if `model` looks like a paid model. "gpt-oss*" is free. A tripwire, not a price list."""
    m = model.lower().replace("gpt-oss", "")
    hit = next((p for p in _PAID if p in m), None) or (x.group(1) if (x := _PAID_BOUNDED.search(m)) else None)
    if hit:
        raise ValueError(f"model '{model}' looks like a paid model (matched '{hit}')")


class DailyCap:
    """In-memory call counter that resets at UTC midnight. Single process only."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._date = None
        self._count = 0

    def try_consume(self) -> bool:
        today = datetime.now(timezone.utc).date()
        if self._date != today:
            self._date, self._count = today, 0
        if self._count >= self.limit:
            return False
        self._count += 1
        return True
