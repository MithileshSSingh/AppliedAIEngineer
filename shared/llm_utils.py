"""LLM API wrappers for OpenAI and Anthropic."""

from shared.config import OPENAI_API_KEY, ANTHROPIC_API_KEY


def get_openai_client():
    """Get an OpenAI client (lazy import)."""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def get_anthropic_client():
    """Get an Anthropic client (lazy import)."""
    from anthropic import Anthropic
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def quick_chat(prompt: str, model: str = "gpt-4o-mini", system: str = "") -> str:
    """Send a quick chat message to OpenAI and return the response text."""
    client = get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
