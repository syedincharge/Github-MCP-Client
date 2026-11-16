"""Invoke raw GitHub MCP tools and normalise their replies.

A reply is a list of content blocks, not a single value. Most tools return one
JSON block, but some return several: `get_file_contents` answers with a status
line ("successfully downloaded text file (SHA: ...)") in block 0 and the file's
actual text in block 1. Reading only the first block silently loses the payload,
so everything here works from the whole block list.
"""

import json
import re
from typing import Any

# `get_file_contents` reports the blob sha in its status line, e.g.
# "successfully downloaded text file (SHA: 4ea1c75...)".
SHA_PATTERN = re.compile(r"SHA:\s*([0-9a-f]{40})")


def extract_blocks(response: Any) -> list[str]:
    """Return every text block of a tool reply, in order."""
    if isinstance(response, str):
        return [response]

    if isinstance(response, (list, tuple)):
        blocks = []
        for block in response:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str) and text:
                blocks.append(text)
        return blocks

    return [str(response)]


def extract_text(response: Any) -> str:
    """Return a tool reply's text blocks joined into one string."""
    return "\n".join(extract_blocks(response))


def parse(response: Any) -> Any:
    """Decode a tool reply as JSON, falling back to its joined text.

    A structured reply arrives as a single JSON block, so the first block that
    decodes is the payload. Replies that are prose or file content do not
    decode, and come back as text.
    """
    blocks = extract_blocks(response)
    for block in blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return "\n".join(blocks)


def _clean(args: dict) -> dict:
    """Drop None values so optional arguments keep their server-side defaults."""
    return {key: value for key, value in args.items() if value is not None}


def _require(raw_tools: dict, name: str):
    tool = raw_tools.get(name)
    if tool is None:
        raise KeyError(
            f"The GitHub MCP server does not expose a '{name}' tool. "
            f"Available: {', '.join(sorted(raw_tools))}"
        )
    return tool


async def invoke(raw_tools: dict, name: str, args: dict) -> Any:
    """Call a GitHub MCP tool by name and return its decoded reply."""
    tool = _require(raw_tools, name)
    return parse(await tool.ainvoke(_clean(args)))


async def invoke_blocks(raw_tools: dict, name: str, args: dict) -> list[str]:
    """Call a GitHub MCP tool and return its raw text blocks, undecoded.

    Use this when the payload is file content rather than structured data:
    decoding would mangle a file that happens to be valid JSON.
    """
    tool = _require(raw_tools, name)
    return extract_blocks(await tool.ainvoke(_clean(args)))


def payload_block(blocks: list[str]) -> str:
    """Return the content block of a reply that leads with a status line.

    `get_file_contents` returns [status, content]; a miss returns the error on
    its own. So the payload is the last block when there are several, and the
    only block otherwise.
    """
    if not blocks:
        return ""
    return blocks[-1]


def find_sha(blocks: list[str]) -> str | None:
    """Return the blob sha reported in a `get_file_contents` reply, if any.

    Returns None when the path does not exist, since the reply is then an error
    rather than a status line.
    """
    for block in blocks:
        match = SHA_PATTERN.search(block)
        if match:
            return match.group(1)
    return None


def summarise(value: Any, limit: int = 4000) -> str:
    """Render a tool reply as compact text an LLM can read."""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, default=str)

    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated, {len(text)} chars total]"
