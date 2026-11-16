from mcp_server.bridge import load_mcp_tools, wrap_mcp_tool
from mcp_server.llm import ask, build_llm
from mcp_server.runtime import run as run_async

__all__ = [
    "ask",
    "build_llm",
    "load_mcp_tools",
    "run_async",
    "wrap_mcp_tool",
    "main",
]

# `mcp_server.github` is imported explicitly rather than re-exported here: it
# reaches the network and needs GITHUB_PERSONAL_ACCESS_TOKEN, so importing this
# package should not require either.


def main() -> None:
    print(ask("Hello, how are you?"))
