"""LLM-backed code review over content fetched from GitHub."""

from langchain_core.language_models import BaseChatModel

REVIEW_SYSTEM_PROMPT = """You are a careful senior code reviewer.

Review the supplied code or diff and report only findings that would change \
what a developer does. For each finding give:
  - severity: BLOCKER | MAJOR | MINOR | NIT
  - location: file and, where you can tell, the line or symbol
  - the defect, in one sentence
  - a concrete failure case (inputs or state -> wrong behaviour), or state \
plainly that it is a style or maintainability point rather than a bug

Prioritise correctness bugs, unhandled errors, security problems, resource \
leaks and race conditions over style. Do not invent issues to fill space: if \
the code looks sound, say so and name what you checked. Do not restate what \
the code does. End with a one-line verdict: APPROVE, APPROVE WITH COMMENTS, \
or REQUEST CHANGES."""


FAILURE_MARKERS = ("failed to ", "404 Not Found", "403 Forbidden", "401 Unauthorized")


async def review_text(llm: BaseChatModel, label: str, body: str) -> str:
    """Return a review of one blob of code or one diff.

    A fetch that failed is reported as a failure rather than sent to the model,
    which would otherwise dutifully "review" an API error message.
    """
    if not body.strip():
        return f"Nothing to review for {label}: the content came back empty."

    first = body.strip().splitlines()[0]
    if any(marker in first for marker in FAILURE_MARKERS):
        return f"Could not fetch {label} to review it. GitHub said: {first}"

    reply = await llm.ainvoke(
        [
            ("system", REVIEW_SYSTEM_PROMPT),
            ("human", f"Reviewing: {label}\n\n```\n{body}\n```"),
        ]
    )
    return reply.content
