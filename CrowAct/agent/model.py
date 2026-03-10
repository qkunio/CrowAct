from dataclasses import dataclass
from typing import Any, Iterator

import requests

from .provider import LLMProvider


@dataclass
class AnthropicToolCallModel:
    provider: LLMProvider
    model: str
    max_tokens: int = 1024
    temperature: float = 0.2
    timeout: int = 60

    def _build_payload(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages,
            "tools": tools,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if stream:
            payload["stream"] = True
        return payload

    def generate_content_blocks(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = self._build_payload(
            system_prompt,
            messages,
            tools,
            stream=False,
        )
        response = requests.post(
            self.provider.build_endpoint(),
            headers=self.provider.build_headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("content") or []

    def iter_stream_chunks(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        content_blocks: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        from .sse import iter_sse_chunks

        payload = self._build_payload(
            system_prompt,
            messages,
            tools,
            stream=True,
        )
        with requests.post(
            self.provider.build_endpoint(),
            headers=self.provider.build_headers(),
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            response.encoding = "utf-8"
            for chunk in iter_sse_chunks(response, content_blocks):
                yield chunk
