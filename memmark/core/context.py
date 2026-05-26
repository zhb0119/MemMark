from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Iterable, Optional


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_context(
    *,
    agent_id: str,
    user_id: str = "",
    session_id: str,
    turn_id: int,
    tau: str,
    event_text: str,
    memory_snapshot: Any,
    timestamp: Optional[float] = None,
    recent_dialog_ids: Optional[Iterable[str]] = None,
    retrieved_memory_ids: Optional[Iterable[str]] = None,
    previous_commitment: str = "",
) -> str:
    """Build the per-decision context string ctx_t.

    Aligned with README §8 fields plus a `previous_commitment` chain that
    threads commitments without requiring a full Merkle proof at scoring
    time. The result is canonical JSON so the verifier can reproduce
    `ctx_t` byte-for-byte.
    """

    payload = {
        "agent_id": agent_id,
        "user_id": user_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "timestamp": timestamp if timestamp is not None else _now_seconds(),
        "tau": tau,
        "event_hash": sha256_text(event_text),
        "memory_hash": sha256_text(stable_json(memory_snapshot)),
        "recent_dialog_ids": list(recent_dialog_ids or []),
        "retrieved_memory_ids": list(retrieved_memory_ids or []),
        "previous_commitment": previous_commitment,
    }
    return stable_json(payload)


def derive_nonce(secret_key: str, ctx_string: str) -> str:
    """nonce_t = HMAC-SHA256(K, ctx_t).

    Per README §9.1: nonce_t = PRF(K, ctx_t). HMAC-SHA256 is a standard
    PRF instantiation. Without this step, the commitment is reproducible
    from public information alone and the watermark loses its keyed
    security property.
    """

    if not secret_key:
        raise ValueError(
            "secret_key must be set; pass MEMMARK_KEY env var or constructor arg"
        )
    return hmac.new(
        secret_key.encode("utf-8"),
        ctx_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def make_watermark_version(
    *,
    sdk_version: str = "memmark-mvp",
    model: str = "",
    api_version: str = "",
    weights_hash: str = "",
    t_score: float = 0.0,
    t_enum: float = 0.7,
    json_mode: bool = True,
    extra: str = "",
) -> str:
    """README §9.3 watermark_version string template.

    Encodes everything that could shift the LLM's distribution so that a
    cross-LLM replay (e.g., DeepSeek vs Qwen) fails commitment checks.
    """

    bits = [
        f"{sdk_version}",
        f"model={model or 'unknown'}",
    ]
    if api_version:
        bits.append(f"api={api_version}")
    if weights_hash:
        bits.append(f"weights={weights_hash}")
    bits.append(f"T_score={t_score}")
    bits.append(f"T_enum={t_enum}")
    bits.append(f"json_mode={'true' if json_mode else 'false'}")
    if extra:
        bits.append(extra)
    return "::".join(bits)


def resolve_secret_key(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.getenv("MEMMARK_KEY") or os.getenv("MEMMARK_SECRET")
    if env:
        return env
    raise RuntimeError(
        "MemMark requires a secret key. Set MEMMARK_KEY env var or pass secret_key=..."
    )


def _now_seconds() -> float:
    return float(int(time.time()))
