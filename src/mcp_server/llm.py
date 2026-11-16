import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_llm() -> ChatOpenAI:
    """Return a chat model, preferring OpenRouter and falling back to OpenAI."""
    load_dotenv()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return ChatOpenAI(
            model=os.getenv(
                "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
            ),
            api_key=openrouter_key,
            base_url=OPENROUTER_BASE_URL,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=openai_key,
    )


def ask(prompt: str) -> str:
    """Send a single prompt to the configured model and return its text reply."""
    return build_llm().invoke(prompt).content
