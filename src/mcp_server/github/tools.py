"""The developer-facing GitHub workflow, one LangChain tool per operation.

Each tool is a thin, documented wrapper over GitHub's remote MCP server, so an
LLM agent can pick operations individually instead of being handed all 46 raw
MCP tools at once.

One thing to be clear about: this talks to the GitHub *API*, not to a local
git checkout. There is no working tree and no staging area, so "commit" and
"push" are a single remote operation -- `gh_commit_file` and `gh_push_files`
both write a commit straight onto a branch. Likewise there is no `git status`;
`gh_repo_status` reports the remote's branches, recent commits and open PRs.
"""

import json
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from mcp_server.github.calls import (
    find_sha,
    invoke,
    invoke_blocks,
    payload_block,
    summarise,
)
from mcp_server.github.review import review_text

MERGE_METHODS = frozenset({"merge", "squash", "rebase"})


def build_workflow_tools(raw_tools: dict, llm: BaseChatModel) -> list[BaseTool]:
    """Return the curated workflow tools, bound to a live MCP tool set."""

    # ---------------------------------------------------------------- identity

    @tool
    async def gh_whoami() -> str:
        """Get the authenticated GitHub user. Call this first when you need the
        repository owner and the user has not told you their username."""
        return summarise(await invoke(raw_tools, "get_me", {}))

    # ------------------------------------------------------------------- repos

    @tool
    async def gh_create_repo(
        name: Annotated[str, "Name of the repository to create"],
        description: Annotated[str, "Short repository description"] = "",
        private: Annotated[bool, "Keep the repository private"] = False,
        auto_init: Annotated[bool, "Seed it with an initial commit and README"] = True,
    ) -> str:
        """Create a new repository in the authenticated user's account.

        Leave auto_init true unless asked otherwise: a repository with no
        initial commit has no default branch, and branch and commit operations
        will fail against it.
        """
        return summarise(
            await invoke(
                raw_tools,
                "create_repository",
                {
                    "name": name,
                    "description": description or None,
                    "private": private,
                    "autoInit": auto_init,
                },
            )
        )

    # ---------------------------------------------------------------- branches

    @tool
    async def gh_create_branch(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        branch: Annotated[str, "Name for the new branch"],
        from_branch: Annotated[
            str | None, "Source branch; the default branch if omitted"
        ] = None,
    ) -> str:
        """Create a new branch. Use this before committing work, so changes do
        not land directly on the default branch."""
        return summarise(
            await invoke(
                raw_tools,
                "create_branch",
                {
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "from_branch": from_branch,
                },
            )
        )

    @tool
    async def gh_list_branches(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
    ) -> str:
        """List the branches of a repository."""
        return summarise(
            await invoke(raw_tools, "list_branches", {"owner": owner, "repo": repo})
        )

    # ------------------------------------------------------------------ status

    @tool
    async def gh_repo_status(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        branch: Annotated[str | None, "Branch to report commits for"] = None,
    ) -> str:
        """Check the state of a repository: its branches, its most recent
        commits and its open pull requests.

        This is the remote equivalent of orienting yourself with `git status` --
        there is no local working tree, so it reports what the remote holds.
        """
        branches = await invoke(
            raw_tools, "list_branches", {"owner": owner, "repo": repo, "perPage": 100}
        )
        commits = await invoke(
            raw_tools,
            "list_commits",
            {"owner": owner, "repo": repo, "sha": branch, "perPage": 10},
        )
        pulls = await invoke(
            raw_tools,
            "list_pull_requests",
            {"owner": owner, "repo": repo, "state": "open", "perPage": 20},
        )
        label = branch or "default branch"
        return "\n\n".join(
            [
                f"=== BRANCHES ===\n{summarise(branches, 1500)}",
                f"=== RECENT COMMITS ({label}) ===\n{summarise(commits, 2500)}",
                f"=== OPEN PULL REQUESTS ===\n{summarise(pulls, 2500)}",
            ]
        )

    # ------------------------------------------------------------------- files

    @tool
    async def gh_read_file(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        path: Annotated[str, "File or directory path in the repository"],
        ref: Annotated[
            str | None, "Branch, tag or commit SHA; the default branch if omitted"
        ] = None,
    ) -> str:
        """Read a file, or list a directory, from a repository. Use this to see
        existing content before changing it. For an existing file it also
        reports the blob sha that `gh_commit_file` needs."""
        blocks = await invoke_blocks(
            raw_tools,
            "get_file_contents",
            {"owner": owner, "repo": repo, "path": path, "ref": ref},
        )
        # Block 0 carries the status line, including the sha; keep both so the
        # model can read the content and the sha from one call.
        return summarise("\n".join(blocks), limit=8000)

    # --------------------------------------------------------- commit and push

    @tool
    async def gh_commit_file(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        branch: Annotated[str, "Branch to commit onto"],
        path: Annotated[str, "Path of the file to create or update"],
        content: Annotated[str, "Full file content, exactly as it should be written"],
        message: Annotated[str, "Commit message"],
        sha: Annotated[
            str | None,
            "Blob SHA of the file being replaced; looked up automatically when omitted",
        ] = None,
    ) -> str:
        """Commit a single file to a branch, creating it or replacing it.

        This commits and pushes in one remote operation -- there is no separate
        push step. Overwriting an existing file needs its current blob sha; if
        you do not pass one this looks it up first, so the call works whether
        or not the file is already there.
        """
        # GitHub rejects an overwrite that carries no sha, and the model cannot
        # be relied on to fetch one first, so resolve it here. A path that does
        # not exist yields no sha, which is exactly what a create needs.
        if sha is None:
            existing = await invoke_blocks(
                raw_tools,
                "get_file_contents",
                {"owner": owner, "repo": repo, "path": path, "ref": branch},
            )
            sha = find_sha(existing)

        return summarise(
            await invoke(
                raw_tools,
                "create_or_update_file",
                {
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "path": path,
                    "content": content,
                    "message": message,
                    "sha": sha,
                },
            )
        )

    @tool
    async def gh_push_files(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        branch: Annotated[str, "Branch to push to"],
        files: Annotated[
            list[dict[str, str]] | str,
            "Files to push, each an object with 'path' and 'content' keys",
        ],
        message: Annotated[str, "Commit message covering all the files"],
    ) -> str:
        """Push one or more files to a branch as a single commit.

        This is the normal way to write files. It takes any number of them,
        makes one commit, and needs no blob sha even when it overwrites.
        """
        try:
            parsed = _coerce_files(files)
        except ValueError as error:
            return str(error)

        return summarise(
            await invoke(
                raw_tools,
                "push_files",
                {
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "files": parsed,
                    "message": message,
                },
            )
        )

    # ------------------------------------------------------------------ review

    @tool
    async def gh_review_code(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        path: Annotated[str, "Path of the file to review"],
        ref: Annotated[
            str | None, "Branch, tag or commit SHA; the default branch if omitted"
        ] = None,
    ) -> str:
        """Review one file's code and report findings by severity. Use this when
        asked to review a file rather than a pull request."""
        blocks = await invoke_blocks(
            raw_tools,
            "get_file_contents",
            {"owner": owner, "repo": repo, "path": path, "ref": ref},
        )
        return await review_text(
            llm, f"{owner}/{repo}:{path}", payload_block(blocks)
        )

    @tool
    async def gh_review_pull_request(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        pull_number: Annotated[int, "Pull request number"],
    ) -> str:
        """Review the diff of a pull request and report findings by severity."""
        blocks = await invoke_blocks(
            raw_tools,
            "pull_request_read",
            {
                "method": "get_diff",
                "owner": owner,
                "repo": repo,
                "pullNumber": pull_number,
            },
        )
        return await review_text(
            llm,
            f"{owner}/{repo} PR #{pull_number}",
            summarise(payload_block(blocks), 20000),
        )

    # ----------------------------------------------------------- pull requests

    @tool
    async def gh_create_pull_request(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        title: Annotated[str, "Pull request title"],
        head: Annotated[str, "Branch containing the changes"],
        base: Annotated[str, "Branch to merge into"],
        body: Annotated[str, "Pull request description"] = "",
        draft: Annotated[bool, "Open it as a draft"] = False,
    ) -> str:
        """Open a pull request from one branch into another. Returns the PR
        number, which the status, review and merge tools need."""
        return summarise(
            await invoke(
                raw_tools,
                "create_pull_request",
                {
                    "owner": owner,
                    "repo": repo,
                    "title": title,
                    "head": head,
                    "base": base,
                    "body": body or None,
                    "draft": draft,
                },
            )
        )

    @tool
    async def gh_pull_request_status(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        pull_number: Annotated[int, "Pull request number"],
    ) -> str:
        """Check a pull request: its details, its mergeability and its CI check
        runs. Call this before merging."""
        details = await invoke(
            raw_tools,
            "pull_request_read",
            {"method": "get", "owner": owner, "repo": repo, "pullNumber": pull_number},
        )
        checks = await invoke(
            raw_tools,
            "pull_request_read",
            {
                "method": "get_check_runs",
                "owner": owner,
                "repo": repo,
                "pullNumber": pull_number,
            },
        )
        return "\n\n".join(
            [
                f"=== PULL REQUEST #{pull_number} ===\n{summarise(details, 3000)}",
                f"=== CHECK RUNS ===\n{summarise(checks, 2000)}",
            ]
        )

    @tool
    async def gh_merge_pull_request(
        owner: Annotated[str, "Repository owner"],
        repo: Annotated[str, "Repository name"],
        pull_number: Annotated[int, "Pull request number"],
        merge_method: Annotated[str, "One of 'merge', 'squash' or 'rebase'"] = "squash",
        commit_title: Annotated[str | None, "Title for the merge commit"] = None,
        commit_message: Annotated[str | None, "Extra detail for the merge commit"] = None,
    ) -> str:
        """Merge a pull request. This is not easily reversible, so check
        `gh_pull_request_status` first and only merge when the user has asked
        for it in this turn."""
        if merge_method not in MERGE_METHODS:
            return (
                f"Invalid merge_method {merge_method!r}: "
                "use 'merge', 'squash' or 'rebase'."
            )
        return summarise(
            await invoke(
                raw_tools,
                "merge_pull_request",
                {
                    "owner": owner,
                    "repo": repo,
                    "pullNumber": pull_number,
                    "merge_method": merge_method,
                    "commit_title": commit_title,
                    "commit_message": commit_message,
                },
            )
        )

    return [
        gh_whoami,
        gh_create_repo,
        gh_create_branch,
        gh_list_branches,
        gh_repo_status,
        gh_read_file,
        gh_commit_file,
        gh_push_files,
        gh_review_code,
        gh_review_pull_request,
        gh_create_pull_request,
        gh_pull_request_status,
        gh_merge_pull_request,
    ]


def _coerce_files(files: Any) -> list[dict[str, str]]:
    """Normalise a `files` argument into the list of objects push_files wants.

    Models routinely hand this over as a JSON *string* rather than a list. That
    is a serialisation slip, not a different intention, so decode it instead of
    failing the call and relying on the model to retry.
    """
    if isinstance(files, str):
        try:
            files = json.loads(files)
        except json.JSONDecodeError as error:
            raise ValueError(
                "The 'files' argument was a string that is not valid JSON "
                f"({error}). Pass a list of objects, each with 'path' and "
                "'content'."
            ) from error

    if isinstance(files, dict):
        files = [files]

    if not isinstance(files, list) or not files:
        raise ValueError(
            "The 'files' argument must be a non-empty list of objects, each "
            "with 'path' and 'content'."
        )

    for entry in files:
        if not isinstance(entry, dict) or "path" not in entry or "content" not in entry:
            raise ValueError(
                f"Each file needs 'path' and 'content' keys; got {entry!r}."
            )

    return files
