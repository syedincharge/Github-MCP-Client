# mcp-server

An LLM-driven GitHub workflow built on GitHub's remote MCP server, plus a small
local MCP server and clients for experimenting with MCP over stdio and HTTP.

## The GitHub workflow agent

Every GitHub operation is exposed as its own tool. You describe the task in
plain language; the LLM picks the tools and the order.

```bash
uv run github-agent --tools             # list the workflow tools
uv run github-agent                     # interactive session
uv run github-agent "status of MCP_Basic"
```

`github-agent` is a console script, so it works from any directory. The
equivalent module form is
`uv run python -m mcp_server.github.github_mcp_client ...`.

```
github> create a branch feature/login in MCP_Basic, add a login.py to it,
         review it, then open a PR into main
```

Each tool call is traced as it happens (`-> tool(args)` / `<- result`), so you
can see what the model actually did rather than only its summary.

### The tools

| Tool | What it does |
| --- | --- |
| `gh_whoami` | Resolve the authenticated user, for the `owner` argument |
| `gh_create_repo` | Create a repository |
| `gh_create_branch` | Branch off the default branch, or another branch |
| `gh_list_branches` | List branches |
| `gh_repo_status` | Branches + recent commits + open PRs |
| `gh_read_file` | Read a file or list a directory |
| `gh_commit_file` | Commit one file to a branch |
| `gh_push_files` | Commit several files to a branch as one commit |
| `gh_review_code` | LLM review of one file, findings by severity |
| `gh_review_pull_request` | LLM review of a PR's diff |
| `gh_create_pull_request` | Open a PR |
| `gh_pull_request_status` | PR details, mergeability and CI check runs |
| `gh_merge_pull_request` | Merge a PR (`merge`, `squash` or `rebase`) |

A typical chain:

```
gh_repo_status -> gh_create_branch -> gh_push_files -> gh_review_code
  -> gh_create_pull_request -> gh_pull_request_status -> gh_merge_pull_request
```

### How this differs from local git

These tools drive the GitHub **API**, not a local checkout. Two consequences
worth knowing before you read the tool list as a git equivalent:

- **Commit and push are one operation.** There is no working tree and no
  staging area, so `gh_commit_file` and `gh_push_files` write a commit straight
  onto a remote branch. There is no separate push step to call, and nothing is
  staged locally first.
- **There is no `git status`.** `gh_repo_status` reports what the remote holds —
  branches, recent commits, open PRs — which is the useful equivalent for
  orienting yourself, but it is not a diff of uncommitted local work.

Updating an existing file with `gh_commit_file` needs that file's current blob
sha (read it with `gh_read_file` first); GitHub rejects the write without it.
`gh_push_files` has no such requirement, so prefer it for edits.

### Where files land

"Add this file to my repo" commits to the **default branch** — that is what it
means, and an earlier version of the system prompt got this wrong by always
cutting a feature branch first, so files appeared to vanish. A branch is only
created when you ask for one, or when you ask for a PR or a review (a PR needs
a branch to open from).

`gh_push_files` is the normal way to write files: any number of them, one
commit, no blob sha needed even when overwriting. `gh_commit_file` handles a
single file and looks the sha up itself when you do not pass one, so it no
longer fails on an existing path.

### Why parallel tool calls are disabled

`build_github_agent` binds the model with `parallel_tool_calls=False` on
purpose. These writes are order-dependent, and `gh_push_files` creates the
branch itself when it is missing. A model that emits `gh_create_branch` and
`gh_push_files` in one turn therefore races itself: GitHub answers the loser
with `422 Reference already exists`, and you end up with the branch created and
the files never written. Serialising the calls removes the race.

Two other sharp edges handled in `calls.py`: a reply is a *list* of content
blocks, and `get_file_contents` puts a status line in block 0 with the file's
text in block 1 — reading only block 0 silently reviews the status message
instead of the code. File and diff payloads therefore bypass JSON decoding, so
a file that happens to be valid JSON is not mangled.

### Layout

```
src/mcp_server/github/
  session.py   connection to https://api.githubcopilot.com/mcp
  calls.py     invoke a raw MCP tool, decode and truncate the reply
  tools.py     the 13 workflow tools
  review.py    the code-review prompt
  agent.py     LangGraph ReAct agent + system prompt
  github_mcp_client.py   CLI (one-shot, interactive, --tools)
```

## Setup

```bash
uv sync
cp .env.example .env     # then fill it in
```

`.env` needs:

- `GITHUB_PERSONAL_ACCESS_TOKEN` — a token with `repo` scope
- `OPENROUTER_API_KEY` or `OPENAI_API_KEY` — OpenRouter is preferred when both
  are set; override the model with `OPENROUTER_MODEL` / `OPENAI_MODEL`

## Dependency pins

`fastmcp` is held below 4.0 and `mcp` below 2.0 on purpose. `mcp` 2.x is a
breaking rewrite (`RequestContext` became `BaseContext`), `fastmcp` 4.x requires
it, and `langchain-mcp-adapters` still targets the 1.x API. Lifting either cap
alone breaks the imports; lift them together once the adapters ship mcp 2.x
support.

## The local MCP server

Separate from the GitHub agent — a small server for trying MCP transports
directly.

```bash
uv run mcp_server_stdio.py    # stdio transport
uv run mcp_server_http.py     # HTTP transport
uv run client_stdio.py        # stdio client
uv run client_http.py         # HTTP client
uv run client_tools.py        # tools exposed to an LLM
```
