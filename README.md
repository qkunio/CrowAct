# CrowAct

`crowact` is a lightweight ReAct agent framework with tool calling, streaming chunks, and pluggable providers.

Distribution name: `crowact`  
Python import name: `CrowAct`

## Requirements

- Python 3.10+
- `requests`
- `python-dotenv`

## Install

```bash
pip install crowact
```

For local development in this repository:

```bash
pip install -e .
```

## Quick Start

```python
from CrowAct import Agent, LLMProvider
from CrowAct.agent.tools import get_tools

TOOLS = get_tools("test_folder")


def main() -> None:
    provider = LLMProvider.from_anthropic_env("byte2.env")
    agent = Agent(
        provider=provider,
        model="deepseek-v3.2",
        system_prompt=(
            "You are a concise ReAct-style assistant. "
            "Use tools first when needed. "
            "You may call tools in multiple steps. "
            "After receiving tool results, provide the final answer."
        ),
        tools=TOOLS,
    )

    question = (
        "Please use tools to calculate this step by step: "
        "first compute 1+19, then take the square root of the result, "
        "and finally add 100."
    )

    for chunk in agent.run(question, stream=True):
        print(chunk)

    print(agent.last_answer)


if __name__ == "__main__":
    main()
```

## Tool Files

`get_tools(folder)` loads every `*.py` file in the target folder and registers functions marked with the `@tool(...)` decorator.

Example folder:

```text
test_folder/
  tool1.py
  tool2.py
```

Example tool:

```python
from CrowAct.agent.tools import tool


@tool(
    description="Compute the square root of a number.",
    param_descriptions={"number": "The number to take the square root of"},
)
def sqrt_tool(number: float) -> float:
    return number ** 0.5
```

## Agent Output

`Agent.run(...)` always returns an iterator of chunks.

Chunk types:

- `{"type": "text", "text": "..."}`
- `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}`
- `{"type": "tool_result", "tool_use_id": "...", "content": "..."}`

Behavior:

- `stream=True`: text is yielded incrementally as streaming chunks.
- `stream=False`: text is yielded as complete blocks, but the API remains iterator-based.

The final plain-text answer is stored in `agent.last_answer`.

## History

You can pass prior conversation context through `history`.

```python
agent = Agent(
    provider=provider,
    model="deepseek-v3.2",
    system_prompt="You are helpful.",
    history=[
        {"role": "user", "content": [{"type": "text", "text": "Previous question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Previous answer"}]},
    ],
)
```

`history` is prepended to the current run, but it is not mutated after execution.

## Provider Setup

### Anthropic-style

```python
provider = LLMProvider.from_anthropic_env(".env")
```

Expected environment variables:

```env
ANTHROPIC_BASE_URL=https://your-endpoint
ANTHROPIC_API_KEY=your-api-key
```

Endpoint pattern:

- `.../v1/messages`

### OpenAI-style

```python
provider = LLMProvider.from_openai_env(".env")
```

Expected environment variables:

```env
OPENAI_BASE_URL=https://your-endpoint
OPENAI_API_KEY=your-api-key
```

Endpoint pattern:

- `.../chat/completions`

## Current Limitation

`LLMProvider` supports both `anthropic` and `openai` endpoint styles, but the request body and tool-calling flow are currently implemented around the Anthropic-style message format. If you use an OpenAI-compatible endpoint, confirm that it accepts the same request shape before relying on it.

## Local Example

This repository includes:

- [main.py](/Users/qinkun/Documents/code/ClawClerk/main.py)
- [test_folder/tool1.py](/Users/qinkun/Documents/code/ClawClerk/test_folder/tool1.py)
- [test_folder/tool2.py](/Users/qinkun/Documents/code/ClawClerk/test_folder/tool2.py)

## License

Add a license before publishing to PyPI.
