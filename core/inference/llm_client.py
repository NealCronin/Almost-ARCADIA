from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

import requests

from core.services.specs import ServiceEndpoint


class LLMClient:
    def __init__(self, endpoint: ServiceEndpoint, timeout: float = 120.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def chat(
        self,
        prompt: str,
        image_paths: Iterable[str] | None = None,
        model: str = "local-model",
    ) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]

        for image_path in image_paths or []:
            path = Path(image_path)
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            suffix = path.suffix.lower().lstrip(".") or "jpeg"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{suffix};base64,{encoded}",
                    },
                }
            )

        response = requests.post(
            f"{self.endpoint.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]
