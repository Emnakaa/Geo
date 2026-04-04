"""Shared LLM call utility backed by the dynamic ModelRegistry.

Model selection is driven by real-time availability state — no hardcoded
fallback lists. When a model is rate-limited the registry is updated and
the next call automatically picks the next best available model.

Usage:
    from llm_utils import call_llm, call_llm_json

    text   = call_llm(prompt="...", system="...", preferred_model="llama-3.3-70b-versatile")
    result = call_llm_json(prompt="...", preferred_model="llama-3.3-70b-versatile")
"""

import re
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from groq import Groq, RateLimitError
from config import GROQ_API_KEY
from model_registry import registry


_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _parse_wait_seconds(error_message: str) -> float:
    """Extract retry-after from Groq error message."""
    m = re.search(
        r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?",
        error_message,
    )
    if not m:
        return 60.0
    h    = float(m.group(1) or 0)
    mins = float(m.group(2) or 0)
    s    = float(m.group(3) or 0)
    return h * 3600 + mins * 60 + s


def _is_daily_limit(error_message: str) -> bool:
    return "tokens per day" in error_message or "per day" in error_message


def call_llm(
    prompt: str,
    system: str = "",
    preferred_model: str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.2,
    tpm_max_wait: float = 90.0,
    max_attempts: int = 6,
) -> str:
    """Call the best available Groq model with automatic failover.

    The registry selects the model dynamically based on current availability.
    On rate-limit errors the registry is updated and the next call picks
    a different model — no hardcoded fallback chain.

    Args:
        prompt:               User message.
        system:               System message (optional).
        preferred_model:      Hint to the registry — used if available.
        max_completion_tokens: Token budget for the response.
        temperature:          Sampling temperature.
        tpm_max_wait:         Max seconds to wait on TPM before trying another model.
        max_attempts:         Total attempts across all models before giving up.

    Returns:
        Response text string.

    Raises:
        RuntimeError: If all models are exhausted.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None

    for attempt in range(max_attempts):
        model = registry.select(preferred=preferred_model)

        try:
            resp = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                print(f"[llm] {model}: empty response — blocking 60s and trying another model")
                registry.mark_tpm(model, 60)
                continue
            return content.strip()

        except RateLimitError as e:
            msg = str(e)
            last_error = e

            if _is_daily_limit(msg):
                registry.mark_tpd_exhausted(model)
                # Loop immediately — registry will pick a different model
                continue

            # TPM limit
            wait = _parse_wait_seconds(msg)
            if wait <= tpm_max_wait:
                registry.mark_tpm(model, wait)
                print(f"[llm] Waiting {wait:.0f}s for TPM reset on {model}...")
                time.sleep(wait + 1)
                # After wait, preferred model should be available again
            else:
                # Wait too long — block this model temporarily and try another
                registry.mark_tpm(model, wait)
                print(f"[llm] TPM wait {wait:.0f}s too long — trying another model")
                continue

        except Exception as e:
            last_error = e
            msg = str(e)
            print(f"[llm] {model}: unexpected error ({e}) — trying another model")
            if "400" in msg or "invalid_request_error" in msg:
                # Bad request — this model won't recover, block it for the session
                registry.mark_tpm(model, 3600)
            else:
                time.sleep(2 ** min(attempt, 4))
            continue

    raise RuntimeError(
        f"All models exhausted after {max_attempts} attempts. "
        f"Last error: {last_error}\n"
        f"Registry status: {registry.status()}"
    )


def call_llm_json(
    prompt: str,
    system: str = "",
    preferred_model: str | None = None,
    max_retries: int = 2,
    **kwargs,
) -> list | dict:
    """Like call_llm but parses and returns JSON. Retries on parse failure."""
    current_prompt = prompt

    for attempt in range(max_retries):
        raw = call_llm(
            prompt=current_prompt,
            system=system,
            preferred_model=preferred_model,
            **kwargs,
        )
        clean = (raw.strip()
                 .removeprefix("```json")
                 .removeprefix("```")
                 .removesuffix("```")
                 .strip())
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                current_prompt = (
                    f"Your previous response was not valid JSON.\n"
                    f"Error: {e}\nYour response was:\n{raw}\n\n"
                    f"Fix it and return ONLY valid JSON, nothing else."
                )
            else:
                print(f"[llm] JSON parse failed after {max_retries} attempts")
                return []

    return []
