import sys
from pathlib import Path

from fastmcp import Client
from langchain_core.messages import HumanMessage, ToolMessage

from mcp_server import build_llm, load_mcp_tools, run_async

STDIO_SERVER = Path(__file__).parent / "mcp_server_stdio.py"
HTTP_URL = "http://127.0.0.1:8000/mcp"
MAX_TOOL_ROUNDS = 10

# Tools under the stdio server's own agent. Binding these would mean our LLM
# calling a second LLM, so they are left out of the tool set.
EXCLUDED = {"stdio_chat"}

CONFIG = {
    "mcpServers": {
        "stdio": {"command": sys.executable, "args": [str(STDIO_SERVER)]},
        "http": {"url": HTTP_URL, "transport": "http"},
    }
}


async def main() -> None:
    async with Client(CONFIG) as client:
        all_tools = await load_mcp_tools(client)
        tools = [t for t in all_tools if t.name not in EXCLUDED]
        by_name = {t.name: t for t in tools}

        print(f"stdio server : {STDIO_SERVER.name}")
        print(f"http server  : {HTTP_URL}")
        print(f"tools        : {', '.join(by_name)}")
        if EXCLUDED & {t.name for t in all_tools}:
            print(f"not bound    : {', '.join(sorted(EXCLUDED))}")

        llm_with_tools = build_llm().bind_tools(tools)
        messages = [HumanMessage(content=input("\nUser: "))]

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
