from datetime import datetime

from fastmcp import FastMCP
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from mcp_server.llm import build_llm

mcp = FastMCP("My MCP Stdio Server")

MAX_TOOL_ROUNDS = 5


def get_current_date() -> str:
    """Returns the current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def multiply(num1: int, num2: int) -> int:
    """Multiplies two integers and return the result."""
    return num1 * num2


# Exposed to MCP clients as plain tools, and to the server's own agent below.
IMPLEMENTATIONS = [get_current_date, multiply]
for _fn in IMPLEMENTATIONS:
    mcp.tool(_fn)

_LC_TOOLS = [StructuredTool.from_function(fn) for fn in IMPLEMENTATIONS]
_LC_BY_NAME = {t.name: t for t in _LC_TOOLS}


@mcp.tool
async def chat(question: str) -> str:
    """Answer a question, choosing and running any needed tools server-side."""
    llm = build_llm().bind_tools(_LC_TOOLS)
    messages = [HumanMessage(content=question)]

    for _ in range(MAX_TOOL_ROUNDS):
        response = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for call in response.tool_calls:
            result = _LC_BY_NAME[call["name"]].invoke(call["args"])
            messages.append(
                ToolMessage(
                    content=str(result),
                    name=call["name"],
                    tool_call_id=call["id"],
                )
            )

    return "Stopped: the model kept requesting tools past the round limit."


if __name__ == "__main__":
    mcp.run(transport="stdio")
