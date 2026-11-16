
from fastmcp import Client
from langchain_core.messages import HumanMessage, ToolMessage

from mcp_server import build_llm, load_mcp_tools, run_async

SERVER_URL = "http://127.0.0.1:8000/mcp"
MAX_TOOL_ROUNDS = 5


async def main() -> None:
    async with Client(SERVER_URL) as client:
        tools = await load_mcp_tools(client)
        by_name = {t.name: t for t in tools}
        print(f"Tools from {SERVER_URL}: {', '.join(by_name)}")

        llm_with_tools = build_llm().bind_tools(tools)
        messages = [HumanMessage(content=input("User: "))]

        for _ in range(MAX_TOOL_ROUNDS):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for call in response.tool_calls:
                result = await by_name[call["name"]].ainvoke(call["args"])
                print(f"[tool] {call['name']}({call['args']}) -> {result}")
                messages.append(
                    ToolMessage(
                        content=str(result),
                        name=call["name"],
                        tool_call_id=call["id"],
                    )
                )

        print(f"AI: {response.content}")


if __name__ == "__main__":
    run_async(main())
