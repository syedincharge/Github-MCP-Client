"""Expose the tools of a connected `fastmcp` MCP server as LangChain tools.

This covers the `fastmcp.Client` case, where the session is managed by the
caller. For GitHub's remote MCP server see `mcp_server.github`, which uses
`langchain-mcp-adapters` over streamable HTTP instead.
"""

from typing import Any

from fastmcp import Client
from langchain_core.tools import BaseTool, StructuredTool


def wrap_mcp_tool(client: Client, mcp_tool: Any) -> StructuredTool:
    """Wrap one MCP tool descriptor as an awaitable LangChain tool."""

    async def call(**kwargs: Any) -> Any:
        result = await client.call_tool(mcp_tool.name, kwargs)
        return result.data

    return StructuredTool.from_function(
        coroutine=call,
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=mcp_tool.input_schema,
    )


async def load_mcp_tools(client: Client) -> list[BaseTool]:
    """Return every tool on a connected MCP client as a LangChain tool.

    The tools call back through `client`, so they are only usable while its
    session is open. They are async-only: use `ainvoke`, not `invoke`.
    """
    return [wrap_mcp_tool(client, t) for t in await client.list_tools()]
