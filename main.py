from CrowAct import Agent, LLMProvider, load_prompt_from
from CrowAct.agent.tools import get_tools

TOOLS = get_tools("your_tool_folder_name")


def main() -> None:
    bytedance = LLMProvider(
        base_url="your base url",
        api_key="your api key",
        style="anthropic",
    )
    # Or: bytedance = LLMProvider.from_anthropic_env("byte.env")
    # .env content:
    # ANTHROPIC_BASE_URL=...
    # ANTHROPIC_API_KEY=...

    agent = Agent(
        provider=bytedance,
        model="deepseek-v3.2",
        system_prompt=load_prompt_from(["xxx.md", "yyy.txt"]),
        tools=TOOLS,
    )
    question = (
        "Please use tools to calculate this step by step: "
        "first compute 1+19, then take the square root of the result, "
        "and finally add 100."
    )

    for chunk in agent.run(question, stream=True):
        print(chunk)

    # print("Final answer:", agent.last_answer)


if __name__ == "__main__":
    main()
