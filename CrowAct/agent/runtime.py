from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .model import AnthropicToolCallModel
from .provider import LLMProvider
from .tools import execute_tool_call, get_tools

MAX_TURNS = 20


def _collect_text(content_blocks: list[dict[str, Any]]) -> str:
    return "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    ).strip()


@dataclass
class Agent:
    provider: LLMProvider
    system_prompt: str
    model: str
    tools: Optional[list[dict[str, Any]]] = None
    history: Optional[list[dict[str, Any]]] = None
    max_turns: int = MAX_TURNS
    max_tokens: int = 1024
    temperature: float = 0.4
    timeout: int = 60
    last_answer: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = get_tools()
        self.client = AnthropicToolCallModel(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    def run(self, user_query: str, stream: bool = False) -> Iterator[dict[str, Any]]:
        messages: list[dict[str, Any]] = list(self.history or [])
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_query}],
            }
        )

        for _ in range(self.max_turns):
            content_blocks: list[dict[str, Any]] = []

            if stream:
                for chunk in self.client.iter_stream_chunks(
                    self.system_prompt,
                    messages,
                    tools=self.tools or [],
                    content_blocks=content_blocks,
                ):
                    yield chunk
            else:
                content_blocks = self.client.generate_content_blocks(
                    self.system_prompt,
                    messages,
                    tools=self.tools or [],
                )
                for block in content_blocks:
                    yield {
                        key: value
                        for key, value in block.items()
                        if not key.startswith("_")
                    }

            messages.append({"role": "assistant", "content": content_blocks})
            tool_calls = [
                block for block in content_blocks if block.get("type") == "tool_use"
            ]

            if not tool_calls:
                self.last_answer = _collect_text(content_blocks)
                return

            tool_results: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": execute_tool_call(tool_call),
                }
                tool_results.append(tool_result)
                yield tool_result

            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError("Exceeded the maximum number of turns without a final answer")


def run_agent(user_query: str, *, stream: bool = False) -> str:
    provider = LLMProvider.from_anthropic_env()
    agent = Agent(
        provider=provider,
        model="claude-sonnet-4-20250514",
        system_prompt=(
            "You are a concise ReAct-style assistant. "
            "Use tools first when needed. "
            "You may call tools in multiple steps. "
            "After receiving tool results, provide the final answer."
        ),
    )
    for _ in agent.run(user_query, stream=stream):
        pass
    return agent.last_answer
