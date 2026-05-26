"""Async OpenAI-compatible client for parallel LLM calls.

Used by the planner to fan out the 3 carrier-assessment + scoring
prompts in parallel instead of running them sequentially. With a 1 s
end-to-end latency per call this turns ~5 sequential calls/decision
into ~2 (assess fan-out is one batch, gen+score is one batch each).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from memmark.llm.config import resolve


class AsyncOpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client: Any = None,
        max_concurrency: int = 8,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "openai package is required for AsyncOpenAIChatClient"
                ) from exc
            resolved_api_key = resolve(
                api_key,
                "MEMMARK_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "DASHSCOPE_API_KEY",
            )
            if not resolved_api_key:
                raise RuntimeError(
                    "Set MEMMARK_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY"
                )
            resolved_base_url = resolve(base_url, "MEMMARK_BASE_URL", "TARGET_LLM_BASE")
            self.client = AsyncOpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
            )
        self.model = (
            resolve(model, "MEMMARK_MODEL", "TARGET_LLM_MODEL") or "deepseek-chat"
        )
        self._sem = asyncio.Semaphore(max_concurrency)

    async def complete_async(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs["extra_body"] = {
            "enable_thinking": False,
            "thinking": {"type": "disabled"},
            "reasoning": {"enabled": False},
        }
        async with self._sem:
            response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def complete_many(
        self,
        prompts: Sequence[List[Dict[str, str]]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Run N prompts concurrently, return aligned list of responses.

        Use this for fan-out: e.g., assessing 3 carrier types in
        parallel, or scoring 5 candidate sets in parallel.
        """

        return await asyncio.gather(
            *[
                self.complete_async(
                    p, temperature=temperature, max_tokens=max_tokens
                )
                for p in prompts
            ]
        )

    # Sync convenience wrapper for code that lives outside an event loop.
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return _run_async(
            self.complete_async(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        )


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)
