from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from memmark.llm.config import first_env, resolve


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client: Any = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError("openai package is required for LLMMemoryAgent") from exc
            resolved_api_key = resolve(
                api_key,
                "OPENAI_API_KEY",
                "MEMMARK_API_KEY",
                "TARGET_LLM_API_KEY",
                "DEEPSEEK_API_KEY",
            )
            if not resolved_api_key:
                raise RuntimeError("Set OPENAI_API_KEY, MEMMARK_API_KEY, TARGET_LLM_API_KEY, or DEEPSEEK_API_KEY")
            resolved_base_url = resolve(
                base_url, "OPENAI_BASE_URL", "MEMMARK_BASE_URL", "TARGET_LLM_BASE"
            )
            default_headers = self._default_headers()
            timeout = self._timeout()
            self.client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                default_headers=default_headers or None,
                timeout=timeout,
            )
        self.model = (
            resolve(model, "OPENAI_MODEL", "MEMMARK_MODEL", "TARGET_LLM_MODEL")
            or "deepseek-chat"
        )

    def complete(
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
        extra_body = self._resolved_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def disable_thinking_extra_body(self) -> Dict[str, Any]:
        """Return provider-specific flags that disable reasoning/thinking."""
        model_lower = self.model.lower()
        if any(name in model_lower for name in ["kimi", "glm", "qwen", "deepseek"]):
            return {"enable_thinking": False}
        return {"enable_thinking": False}

    def _resolved_extra_body(self) -> Optional[Dict[str, Any]]:
        env_value = first_env(
            "TARGET_LLM_EXTRA_BODY", "MEMMARK_EXTRA_BODY", "OPENAI_EXTRA_BODY"
        )
        if env_value:
            try:
                parsed = json.loads(env_value)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSON in TARGET_LLM_EXTRA_BODY/MEMMARK_EXTRA_BODY/OPENAI_EXTRA_BODY"
                ) from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(
                    "TARGET_LLM_EXTRA_BODY/MEMMARK_EXTRA_BODY/OPENAI_EXTRA_BODY must decode to a JSON object"
                )
            return parsed
        model_lower = self.model.lower()
        if any(name in model_lower for name in ["deepseek", "glm", "kimi", "qwen"]):
            return self.disable_thinking_extra_body()
        return None

    @staticmethod
    def _default_headers() -> Dict[str, str]:
        headers = {}
        site_url = OpenAIChatClient._ascii_header_value(os.getenv("OPENROUTER_SITE_URL"))
        app_name = OpenAIChatClient._ascii_header_value(os.getenv("OPENROUTER_APP_NAME"))
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        return headers

    @staticmethod
    def _ascii_header_value(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        encoded = value.encode("ascii", errors="ignore").decode("ascii").strip()
        return encoded or None

    @staticmethod
    def _timeout() -> Optional[float]:
        raw = first_env("MEMMARK_OPENAI_TIMEOUT", "OPENAI_TIMEOUT")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None
