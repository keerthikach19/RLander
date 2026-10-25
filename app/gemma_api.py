"""Minimal Gemma 4 API client used by the Mission Test Lab.

This module contains only the API plumbing. No training, reward weights,
or PPO refinement logic lives here; that workflow was removed.
"""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "gemma-4-26b-a4b-it"


def strict_json(text):
    """Parse JSON with duplicate-key and nonfinite-number rejection."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def gemma_request(prompt, model, api_key):
    """Make a single Gemma 4 generate-content request and return the raw payload."""
    if not re.fullmatch(r"gemma-4-[a-zA-Z0-9.-]+", model):
        raise ValueError("Select an explicit Gemma 4 model ID, not a Gemini model")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096,
                             "thinkingConfig": {"thinkingLevel": "minimal"}},
    }
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = strict_json(response.read(2_000_000).decode())
    except HTTPError as exc:
        raise RuntimeError(
            f"Gemma API returned HTTP {exc.code}; check model access, key and quota"
        ) from None
    except (URLError, TimeoutError):
        raise RuntimeError("Gemma API connection failed or timed out") from None
    return payload


def response_text(payload):
    """Extract the plain-text response from a Gemma generateContent payload."""
    candidates = payload.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Gemma response was blocked, truncated, or did not finish normally")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought", False))
    if not text.strip():
        raise ValueError("Gemma returned no proposal text")
    return text
