"""Multi-provider LLM router.

Spreads parallel requests across multiple OpenAI-compatible endpoints
(e.g., DeepSeek + DashScope-Qwen + OpenRouter) to dodge per-provider
rate limits and double the effective throughput. Watermark replay
still works because each call goes through the same `complete()`
interface and `watermark_version` records which model produced each
trace (so paper experiments must keep one provider per trace).

Two routing modes:

* `round_robin` — request i goes to provider i % N. Best when all
  providers serve the same model and are equally fast.
* `weighted_random` — pick by latency-inverse weights. Best when one
  provider is faster than others.

Use this only for the *speed* of the assess/score loop, not for the
RQ trace itself: a single audit trace must be tied to a single LLM
identity (otherwise commitment / version mismatch).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from memmark.llm.async_client import AsyncOpenAIChatClient
from memmark.llm.openai_client import OpenAIChatClient


@dataclass
class Provider:
    name: str
    client: Any                              # OpenAIChatClient or AsyncOpenAIChatClient
    weight: float = 1.0
    moving_avg_latency_s: float = 1.0
    request_count: int = 0


class MultiProviderClient:
    """Drop-in replacement for OpenAIChatClient that spreads load.

    Construct with a list of (base_url, api_key, model) triples; each
    becomes a sub-client and `complete()` round-robins across them.
    """

    def __init__(
        self,
        configs: Sequence[Dict[str, Any]],
        *,
        mode: str = "round_robin",
        async_mode: bool = False,
    ) -> None:
        if not configs:
            raise ValueError("MultiProviderClient needs ≥1 provider config")
        if mode not in ("round_robin", "weighted_random"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.async_mode = async_mode
        self.providers: List[Provider] = []
        for cfg in configs:
            kwargs = {
                "api_key": cfg.get("api_key"),
                "base_url": cfg.get("base_url"),
                "model": cfg.get("model"),
            }
            client = (
                AsyncOpenAIChatClient(**kwargs)
                if async_mode
                else OpenAIChatClient(**kwargs)
            )
            self.providers.append(
                Provider(
                    name=cfg.get("name", cfg.get("model", "provider")),
                    client=client,
                    weight=float(cfg.get("weight", 1.0)),
                )
            )
        self._next_idx = 0
        self._lock = asyncio.Lock() if async_mode else None

    @property
    def model(self) -> str:
        # Multi-model — return a synthetic identifier
        return "+".join(p.name for p in self.providers)

    # -- sync API ----------------------------------------------------- #
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        provider = self._pick()
        start = time.time()
        try:
            return provider.client.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        finally:
            self._record_latency(provider, time.time() - start)

    # -- async API ---------------------------------------------------- #
    async def complete_async(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        provider = self._pick()
        start = time.time()
        try:
            client = provider.client
            if hasattr(client, "complete_async"):
                return await client.complete_async(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
            # Fallback: run sync in a thread
            return await asyncio.to_thread(
                client.complete,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        finally:
            self._record_latency(provider, time.time() - start)

    async def complete_many(
        self,
        prompts: Sequence[List[Dict[str, str]]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        return await asyncio.gather(
            *[
                self.complete_async(
                    p, temperature=temperature, max_tokens=max_tokens
                )
                for p in prompts
            ]
        )

    # -- routing ------------------------------------------------------ #
    def _pick(self) -> Provider:
        if self.mode == "round_robin":
            p = self.providers[self._next_idx % len(self.providers)]
            self._next_idx += 1
            return p
        # weighted random by inverse latency
        weights = [p.weight / max(p.moving_avg_latency_s, 1e-3) for p in self.providers]
        total = sum(weights)
        if total <= 0:
            return self.providers[0]
        r = random.uniform(0, total)
        cum = 0.0
        for p, w in zip(self.providers, weights):
            cum += w
            if r <= cum:
                return p
        return self.providers[-1]

    def _record_latency(self, provider: Provider, latency: float) -> None:
        provider.request_count += 1
        provider.moving_avg_latency_s = (
            0.8 * provider.moving_avg_latency_s + 0.2 * latency
        )

    def stats(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": p.name,
                "request_count": p.request_count,
                "avg_latency_s": p.moving_avg_latency_s,
            }
            for p in self.providers
        ]
