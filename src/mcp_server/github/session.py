"""Connection to GitHub's hosted MCP server."""

import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp"
TOKEN_ENV_VAR = "GITHUB_PERSONAL_ACCESS_TOKEN"


def build_github_client() -> MultiServerMCPClient:
    """Return a client for GitHub's remote MCP server.

    The client opens a fresh session per tool call, so it needs no explicit
    lifecycle management.
    """
    load_dotenv()

    token = os.getenv(TOKEN_ENV_VAR)
    if not token:
        raise RuntimeError(
            f"{TOKEN_ENV_VAR} is not set. Create a GitHub personal access token "
            "with 'repo' scope and add it to your .env file."
        )

    return MultiServerMCPClient(
        {
            "github": {
                "transport": "streamable_http",
                "url": GITHUB_MCP_URL,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )


async def load_raw_tools(client: MultiServerMCPClient) -> dict:
    """Return every GitHub MCP tool, keyed by name."""
    return {tool.name: tool for tool in await client.get_tools()}
