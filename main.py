from CrowAct import Agent, LLMProvider


def main() -> None:
    provider = LLMProvider.from_anthropic_env("deepseek.env")
    agent = Agent(
        provider=provider,
        model="deepseek-v4-flash",
        system_prompt="你是一个简洁、友好的中文助手。",
    )

    print("CrowAct CLI chat")
    print("输入 exit 或 quit 退出。")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        print("Assistant: ", end="", flush=True)
        for chunk in agent.run(question, stream=True):
            if chunk.get("type") == "text":
                print(chunk.get("text", ""), end="", flush=True)
        print()


if __name__ == "__main__":
    main()
