from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Iterable

import requests

from core.errors import InferenceError
from core.inference.results import LLMResult
from core.services.specs import ServiceEndpoint


class LLMClient:
    """Call an already-running OpenAI-compatible LLM data-plane endpoint."""

    def __init__(self, endpoint: ServiceEndpoint, timeout: float = 120.0) -> None:
        if endpoint.service_type != "llm":
            raise ValueError("LLMClient requires an LLM endpoint.")
        self.endpoint = endpoint
        self.timeout = timeout

    def chat(
        self,
        prompt: str,
        image_paths: Iterable[str | Path] | None = None,
        *,
        images: Iterable[bytes | tuple[str, bytes]] | None = None,
        model: str = "local-model",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths or []:
            path = Path(image_path)
            content.append({"type": "image_url", "image_url": {"url": self._data_uri(path.read_bytes(), path)}})
        for image in images or []:
            if isinstance(image, tuple):
                mime, raw = image
            else:
                mime, raw = "image/jpeg", image
            content.append({"type": "image_url", "image_url": {"url": self._data_uri(raw, mime=mime)}})

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        try:
            response = requests.post(
                f"{self.endpoint.base_url}/v1/chat/completions",
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, OSError, ValueError) as exc:
            raise InferenceError(f"LLM request failed: {exc}", service_type="llm") from exc
        try:
            message = payload["choices"][0]["message"]["content"]
            if isinstance(message, list):
                text = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in message)
            else:
                text = str(message)
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                "LLM response did not contain choices[0].message.content.",
                service_type="llm",
            ) from exc
        return LLMResult(text=text, raw_response=payload if isinstance(payload, dict) else None)

    @staticmethod
    def _data_uri(raw: bytes, source: Path | str | None = None, *, mime: str | None = None) -> str:
        if mime is None:
            mime = mimetypes.guess_type(str(source))[0] if source is not None else None
        mime = mime or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
