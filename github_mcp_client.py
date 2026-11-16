"""Prompt-driven GitHub workflow agent.

Usage:
    uv run github-agent "create a repo called demo and add a README"
    uv run github-agent --tools
    uv run github-agent                    # interactive

Equivalently, without the console script:
    uv run python -m mcp_server.github.github_mcp_client ...

Every GitHub operation is a separate tool call; the LLM decides which to use
and in what order from your instruction.
"""

import sys

from mcp_server import run_async
from mcp_server.github import build_github_client, load_raw_tools, stream_prompt
from mcp_server.github.tools import build_workflow_tools
from mcp_server.llm import build_llm


def _force_utf8_output() -> None:
    """Make stdout/stderr able to carry the model's text on Windows.

    The Windows console defaults to cp1252, which cannot encode characters the
    model uses freely -- a non-breaking hyphen in a reply is enough to end the
    run in UnicodeEncodeError after all the GitHub work has already succeeded.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

BANNER = """GitHub workflow agent. Describe what you want done, or 'quit' to exit.

Examples:
  status of MCP_Basic
  create a branch called feature/login in MCP_Basic off main
  add a hello.py printing hi to feature/login in MCP_Basic
  review src/app.py in MCP_Basic
  open a PR from feature/login into main in MCP_Basic
  check PR 1 in MCP_Basic, then merge it with squash
"""


async def show_tools() -> None:
    """Print the curated workflow tools and their arguments."""
    raw_tools = await load_raw_tools(build_github_client())
    tools = build_workflow_tools(raw_tools, build_llm())

    print(f"{len(tools)} workflow tools (over {len(raw_tools)} raw GitHub MCP tools):\n")
    for item in tools:
        summary = " ".join((item.description or "").split())
        print(f"  {item.name}")
        print(f"      {summary[:150]}")


async def one_shot(prompt: str) -> None:
    print(f"\n> {prompt}")
    reply = await stream_prompt(prompt)
    print(f"\n{reply}")


async def interactive() -> None:
    print(BANNER)
    while True:
        try:
            prompt = input("github> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            return

        try:
            reply = await stream_prompt(prompt)
            print(f"\n{reply}\n")
        except Exception as error:  # keep the session alive on a failed turn
            print(f"\n[error] {type(error).__name__}: {error}\n")


def main() -> None:
    _force_utf8_output()
    args = sys.argv[1:]

    if args and args[0] in {"--tools", "-t"}:
        run_async(show_tools())
    elif args:
        run_async(one_shot(" ".join(args)))
    else:
        run_async(interactive())


if __name__ == "__main__":
    main()
