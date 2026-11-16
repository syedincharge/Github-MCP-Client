from pathlib import Path

from fastmcp import Client

from mcp_server import run_async

SERVER_PATH = Path(__file__).parent / "mcp_server_stdio.py"


async def main() -> None:
    async with Client(SERVER_PATH) as client:
        question = input("User: ")
        answer = await client.call_tool("chat", {"question": question})
        print(f"AI: {answer.data}")


if __name__ == "__main__":
    run_async(main())
