"""A prompt-driven agent over the GitHub workflow tools."""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from mcp_server.github.session import build_github_client, load_raw_tools
from mcp_server.github.tools import build_workflow_tools
from mcp_server.llm import build_llm

# A full feature flow is status -> branch -> push -> review -> PR -> status ->
# merge, and each step is a model turn plus a tool turn, so the default limit
# of 25 is not enough headroom.
RECURSION_LIMIT = 80

# `create_agent` names its model node "model"; a ToolMessage arrives under
# "tools". The distinction matters when picking out the final answer.
MODEL_NODE = "model"
TOOLS_NODE = "tools"

SYSTEM_PROMPT = """You are a GitHub workflow assistant for a developer. You \
drive real repositories through GitHub's API, one tool call at a time.

How to work:
- Resolve the repository owner before anything else. Unless the user gave you \
an explicit owner, your FIRST call is gh_whoami; reuse the login it returns for \
every later call. Never guess an owner and never use the repository name as the \
owner.
- Issue ONE tool call at a time and wait for its result. These operations are \
order-dependent: creating a branch must finish before anything is pushed to it, \
and a push must finish before you open a pull request over it.
- Carry out the WHOLE request before you reply. Chain as many tool calls as it \
takes; do not stop after the first step and describe the rest as next steps. A \
typical feature flow is: gh_repo_status -> gh_create_branch -> gh_push_files \
-> gh_review_code -> gh_create_pull_request -> gh_pull_request_status -> \
gh_merge_pull_request.
- Put the work where the user asked for it. When they just want files added to \
a repository, commit to the default branch; that is what "add this file to my \
repo" means. Create a branch first only when they ask for a branch, or when \
they ask for a pull request or review, since a PR needs a branch to open from.
- To write files, prefer gh_push_files. It takes several files in one commit \
and needs no blob sha.
- When updating an existing file with gh_commit_file, read it first with \
gh_read_file to get its blob sha; without the sha the write is rejected.
- Report what each step actually returned, including URLs and PR numbers. If a \
tool fails, say so plainly and state what you tried -- never describe a step as \
done when its call errored.

Merging a pull request is hard to undo. Do it when the user's request covers \
it, and ask first when they have not actually asked for a merge.

Be concise. Prose about what you are about to do is not useful; the results \
are."""


async def build_github_agent(llm: BaseChatModel | None = None):
    """Return an agent wired to the GitHub workflow tools.

    Parallel tool calling is switched off deliberately. These operations are
    order-dependent: a model that emits `gh_create_branch` and `gh_push_files`
    in one turn races itself, because push also creates the branch when it is
    missing, and GitHub rejects the loser with 422 "Reference already exists" --
    leaving the branch made and the files unwritten.
    """
    model = llm or build_llm()
    client = build_github_client()
    raw_tools = await load_raw_tools(client)

    # The review tools call the model with no tools bound, where
    # `parallel_tool_calls` would be an invalid request parameter, so they get
    # the unbound model.
    tools = build_workflow_tools(raw_tools, model)

    return create_agent(
        model=model.bind(parallel_tool_calls=False),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


async def run_prompt(prompt: str, llm: BaseChatModel | None = None) -> str:
    """Run one natural-language instruction through the agent."""
    agent = await build_github_agent(llm)
    result = await agent.ainvoke(
        {"messages": [("user", prompt)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return result["messages"][-1].content


async def stream_prompt(prompt: str, llm: BaseChatModel | None = None) -> str:
    """Run one instruction, printing each tool call as it happens.

    Returns the agent's final reply.
    """
    agent = await build_github_agent(llm)
    final = ""

    async for step in agent.astream(
        {"messages": [("user", prompt)]},
        stream_mode="updates",
        config={"recursion_limit": RECURSION_LIMIT},
    ):
        for node, update in step.items():
            for message in _messages(update):
                _trace(node, message)
                # Only the model's own replies can be the final answer. A
                # ToolMessage also carries content and no tool_calls, so
                # testing for those alone would return raw tool JSON instead.
                if node == MODEL_NODE and getattr(message, "content", None):
                    final = message.content

    return final


def _messages(update) -> list:
    if isinstance(update, dict):
        return update.get("messages", []) or []
    return []


def _tool_calls(message) -> list:
    return getattr(message, "tool_calls", None) or []


def _trace(node: str, message) -> None:
    """Print a one-line trace of an agent step."""
    for call in _tool_calls(message):
        print(f"  -> {call['name']}({_short_args(call.get('args', {}))})")

    if node == TOOLS_NODE:
        name = getattr(message, "name", "tool")
        body = (getattr(message, "content", "") or "").strip()
        first = next((line for line in body.splitlines() if line.strip()), "(empty)")
        print(f"  <- {name}: {first[:160]}")


def _short_args(args: dict, limit: int = 120) -> str:
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return rendered if len(rendered) <= limit else f"{rendered[:limit]}..."
