"""Environment helpers. Secrets are never logged or hardcoded."""

from __future__ import annotations

import os

_PLACEHOLDER_KEYS = {
    "",
    "your_anthropic_api_key_here",
    "changeme",
    "replace_me",
    "<your_anthropic_api_key_here>",
}


def anthropic_api_key() -> str | None:
    """Return the Anthropic key from the environment, or None if unset/placeholder."""

    raw = os.environ.get("ANTHROPIC_API_KEY", "").strip().strip("\"'")
    if raw.lower() in _PLACEHOLDER_KEYS:
        return None
    if raw.startswith("<") and raw.endswith(">"):
        return None
    return raw or None
