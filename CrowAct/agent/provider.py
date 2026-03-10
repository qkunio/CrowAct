import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv


@dataclass
class LLMProvider:
    base_url: str
    api_key: str
    style: Literal["anthropic", "openai"]
    default_model: Optional[str] = None
    api_version: str = "2023-06-01"

    @classmethod
    def from_anthropic_env(cls, env_file: str = ".env") -> "LLMProvider":
        load_dotenv(env_file)

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not base_url:
            raise RuntimeError("Missing ANTHROPIC_BASE_URL")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")

        return cls(
            base_url=base_url,
            api_key=api_key,
            style="anthropic",
        )

    @classmethod
    def from_openai_env(cls, env_file: str = ".env") -> "LLMProvider":
        load_dotenv(env_file)

        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")

        if not base_url:
            raise RuntimeError("Missing OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        return cls(
            base_url=base_url,
            api_key=api_key,
            style="openai",
        )

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.style == "anthropic":
            headers["anthropic-version"] = self.api_version
        return headers

    def build_endpoint(self) -> str:
        if self.style == "anthropic":
            return self.base_url.rstrip("/") + "/v1/messages"
        return self.base_url.rstrip("/") + "/chat/completions"
