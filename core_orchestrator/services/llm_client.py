"""
llm_client.py

Lightweight HTTP client for LLM inference with API-format adapters.
Supports llama.cpp native completion, OpenAI chat, and OpenAI responses.

This module does NOT own process lifecycle.  It only sends HTTP requests
to a running or external LLM server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from .settings_store import LLMServiceSettings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Normalized LLM inference result."""
    content: str
    model_id: Optional[str] = None
    usage: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


class LLMInferenceError(Exception):
    """Raised when LLM inference fails."""


class LLMInferenceClient:
    """
    Client for LLM inference with multiple API format adapters.

    The client is stateless beyond the ``requests.Session`` — it does not
    own or manage any process lifecycle.
    """

    def __init__(self) -> None:
        self._session = requests.Session()

    def evaluate(
        self,
        settings: LLMServiceSettings,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Send an inference request using the configured API format.

        The *settings* object determines the endpoint URL and request payload.
        """
        base_url = self._resolve_base_url(settings)
        timeout = settings.request_timeout_seconds

        if settings.api_format == "llama-completion":
            return self._eval_llama_completion(
                base_url, settings, prompt, context,
                max_tokens, temperature, top_p, timeout, **kwargs
            )
        elif settings.api_format == "openai-chat":
            return self._eval_openai_chat(
                base_url, settings, prompt, context,
                max_tokens, temperature, top_p, timeout, **kwargs
            )
        elif settings.api_format == "openai-responses":
            return self._eval_openai_responses(
                base_url, settings, prompt, context,
                max_tokens, temperature, top_p, timeout, **kwargs
            )
        else:
            raise LLMInferenceError(f"Unsupported API format: {settings.api_format}")

    def health_check(self, settings: LLMServiceSettings, timeout: int = 5) -> bool:
        """Check whether the LLM server responds to a health probe."""
        base_url = self._resolve_base_url(settings)
        health_url = self._resolve_health_url(base_url, settings.api_format)
        if not health_url:
            return False
        try:
            resp = self._session.get(health_url, timeout=timeout)
            return 200 <= resp.status_code < 300
        except requests.RequestException:
            return False

    # -- URL resolution -----------------------------------------------------

    @staticmethod
    def _resolve_base_url(settings: LLMServiceSettings) -> str:
        if settings.base_url:
            return settings.base_url.rstrip("/")
        if hasattr(settings, "host") and hasattr(settings, "port"):
            return f"http://{settings.host}:{settings.port}"
        raise LLMInferenceError("No base_url or host/port configured for LLM service")

    @staticmethod
    def _resolve_health_url(base_url: str, api_format: str) -> Optional[str]:
        if api_format == "llama-completion":
            return f"{base_url}/health"
        elif api_format in ("openai-chat", "openai-responses"):
            # Normalize: ensure /v1 is present
            if "/v1" not in base_url:
                return f"{base_url}/v1/models"
            return f"{base_url}/models"
        return base_url

    # -- Format adapters ----------------------------------------------------

    def _eval_llama_completion(
        self, base_url: str, settings: LLMServiceSettings,
        prompt: str, context: Optional[str],
        max_tokens: int, temperature: float, top_p: float,
        timeout: int, **kwargs: Any,
    ) -> LLMResult:
        full_prompt = self._compose_prompt(prompt, context)
        payload = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        url = f"{base_url}/completion"
        try:
            resp = self._session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise LLMInferenceError(f"LLM request to {url} failed: {exc}") from exc
        except (ValueError, KeyError) as exc:
            raise LLMInferenceError(f"Invalid response from {url}: {exc}") from exc

        content = data.get("content", data.get("generation", ""))
        return LLMResult(
            content=content,
            model_id=settings.model_id or None,
            usage=data.get("usage"),
            metadata={"api_format": "llama-completion"},
        )

    def _eval_openai_chat(
        self, base_url: str, settings: LLMServiceSettings,
        prompt: str, context: Optional[str],
        max_tokens: int, temperature: float, top_p: float,
        timeout: int, **kwargs: Any,
    ) -> LLMResult:
        messages = self._compose_messages(prompt, context)
        payload = {
            "model": settings.model_id or "",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Ensure /v1/chat/completions
        if "/v1" in base_url:
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"

        if not settings.model_id:
            raise LLMInferenceError("model_id is required for OpenAI chat format")

        try:
            resp = self._session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise LLMInferenceError(f"OpenAI chat request to {url} failed: {exc}") from exc
        except (ValueError, KeyError) as exc:
            raise LLMInferenceError(f"Invalid response from {url}: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInferenceError(f"Could not extract content from OpenAI response: {exc}") from exc

        return LLMResult(
            content=content,
            model_id=settings.model_id,
            usage=data.get("usage"),
            metadata={"api_format": "openai-chat"},
        )

    def _eval_openai_responses(
        self, base_url: str, settings: LLMServiceSettings,
        prompt: str, context: Optional[str],
        max_tokens: int, temperature: float, top_p: float,
        timeout: int, **kwargs: Any,
    ) -> LLMResult:
        full_input = self._compose_prompt(prompt, context)
        payload = {
            "model": settings.model_id or "",
            "input": full_input,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if "/v1" in base_url:
            url = f"{base_url}/responses"
        else:
            url = f"{base_url}/v1/responses"

        if not settings.model_id:
            raise LLMInferenceError("model_id is required for OpenAI responses format")

        try:
            resp = self._session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise LLMInferenceError(f"OpenAI responses request to {url} failed: {exc}") from exc
        except (ValueError, KeyError) as exc:
            raise LLMInferenceError(f"Invalid response from {url}: {exc}") from exc

        # Responses API returns content in different shapes
        content = ""
        if "output" in data and isinstance(data["output"], list):
            for item in data["output"]:
                if isinstance(item, dict) and item.get("type") == "message":
                    for part in item.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            content = part.get("text", "")
                            break
                    if content:
                        break
        elif "content" in data:
            content = str(data["content"])

        if not content:
            content = str(data.get("output_text", ""))

        return LLMResult(
            content=content,
            model_id=settings.model_id,
            usage=data.get("usage"),
            metadata={"api_format": "openai-responses"},
        )

    # -- Prompt composition -------------------------------------------------

    @staticmethod
    def _compose_prompt(prompt: str, context: Optional[str]) -> str:
        if context:
            return f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        return prompt

    @staticmethod
    def _compose_messages(prompt: str, context: Optional[str]) -> list[dict]:
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        return messages