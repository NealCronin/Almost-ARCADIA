from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterable

import requests

from core.errors import InferenceError
from core.inference.results import LLMResult
from core.services.specs import ServiceEndpoint

ImageInput = tuple[str, bytes]


class LLMClient:
    def __init__(
        self,
        endpoint: ServiceEndpoint,
        timeout: float = 180.0,
        *,
        role_defaults: dict[str, Any] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.role_defaults = dict(role_defaults or {})
        self._discovered_model: str | None = None

    def _model_id(self) -> str:
        """Read the model ID exposed by llama-server without changing its alias."""
        configured = self.role_defaults.get("model")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        if self._discovered_model is not None:
            return self._discovered_model

        try:
            response = requests.get(
                f"{self.endpoint.base_url}/v1/models",
                timeout=min(self.timeout, 30.0),
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or not data:
                raise ValueError("response did not contain a model list")
            first = data[0]
            model_id = first.get("id") if isinstance(first, dict) else None
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("first model did not contain a usable id")
        except (requests.RequestException, TypeError, ValueError) as exc:
            raise InferenceError(
                f"{self.endpoint.service_type} could not discover its model ID from "
                f"{self.endpoint.base_url}/v1/models: {exc}",
                service_type=self.endpoint.service_type,
            ) from exc

        self._discovered_model = model_id.strip()
        return self._discovered_model

    def chat(
        self,
        prompt: str,
        *,
        images: Iterable[ImageInput] | None = None,
        image_paths: Iterable[str | Path] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for mime_type, raw in images or []:
            encoded = base64.b64encode(raw).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})
        for raw_path in image_paths or []:
            path = Path(raw_path)
            suffix = path.suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(suffix, "image/jpeg")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})

        selected_model = model.strip() if isinstance(model, str) and model.strip() else self._model_id()
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": [{"role": "user", "content": content}],
        }
        effective_temperature = temperature if temperature is not None else self.role_defaults.get("temperature")
        effective_max_tokens = max_tokens if max_tokens is not None else self.role_defaults.get("max_tokens")
        if effective_temperature is not None:
            payload["temperature"] = float(effective_temperature)
        if effective_max_tokens is not None:
            payload["max_tokens"] = int(effective_max_tokens)

        try:
            response = requests.post(
                f"{self.endpoint.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_payload = response.json()
            text = raw_payload["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            raise InferenceError(
                f"{self.endpoint.service_type} inference failed at {self.endpoint.base_url}: {exc}",
                service_type=self.endpoint.service_type,
            ) from exc
        return LLMResult(str(text), raw_payload)
