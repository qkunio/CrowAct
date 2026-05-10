from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .model import AnthropicToolCallModel
from .provider import LLMProvider
from .tools import execute_tool_call, get_tools

MAX_TURNS = 20
DEFAULT_HISTORY_WINDOW = 20


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
    history_window: int = DEFAULT_HISTORY_WINDOW
    max_turns: int = MAX_TURNS
    max_tokens: int = 1024
    temperature: float = 0.4
    timeout: int = 60
    last_answer: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = get_tools()
        self.history = list(self.history or [])
        if self.history_window < 1:
            raise ValueError("history_window must be at least 1")
        self._trim_history()
        self.client = AnthropicToolCallModel(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    def _trim_history(self) -> None:
        self.history = self.history[-self.history_window :]

    def _append_history(self, *messages: dict[str, Any]) -> None:
        self.history.extend(messages)
        self._trim_history()

    def run(self, user_query: str, stream: bool = False) -> Iterator[dict[str, Any]]:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": user_query}],
        }
        self._append_history(user_message)
        messages: list[dict[str, Any]] = list(self.history)

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

            assistant_message = {"role": "assistant", "content": content_blocks}
            self._append_history(assistant_message)
            messages = list(self.history)
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

            tool_result_message = {"role": "user", "content": tool_results}
            self._append_history(tool_result_message)
            messages = list(self.history)

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
