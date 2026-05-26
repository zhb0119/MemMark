"""Shared env-resolution helpers for OpenAI-compatible LLM clients.

Both ``OpenAIChatClient`` and ``AsyncOpenAIChatClient`` accept api_key /
base_url / model via constructor or via a list of fallback environment
variables. The per-client preference order differs intentionally (sync
defaults to ``OPENAI_API_KEY`` first; async defaults to
``MEMMARK_API_KEY`` first), so this module dedupes the *mechanism* only —
each client still passes its own env-name list.
"""

from __future__ import annotations

import os
from typing import Optional


def first_env(*names: str) -> Optional[str]:
    """Return the first non-empty ``os.environ`` value among ``names``."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def resolve(explicit: Optional[str], *env_names: str) -> Optional[str]:
    """Prefer an explicit constructor arg, else first non-empty env var."""

    if explicit:
        return explicit
    return first_env(*env_names)
