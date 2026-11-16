from pathlib import Path

from fastmcp import Client

from mcp_server import run_async

SERVER_PATH = Path(__file__).parent / "mcp_server_stdio.py"


async def main() -> None:
    async with Client(SERVER_PATH) as client:
        tools = await client.list_tools()
        print("Tools exposed by the server:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        print("\nCalling the raw tools directly:")
        today = await client.call_tool("get_current_date", {})
        print(f"  get_current_date() -> {today.data}")
        product = await client.call_tool("multiply", {"num1": 6, "num2": 7})
        print(f"  multiply(6, 7)     -> {product.data}")

        print("\nLetting the server's agent orchestrate instead:")
        answer = await client.call_tool(
            "chat", {"question": "what is 6 times 7, and what is today?"}
        )
        print(f"  chat(...)          -> {answer.data}")


if __name__ == "__main__":
    run_async(main())
