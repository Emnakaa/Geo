"""Shared LLM call utility — multi-provider, fully autonomous model selection.

Providers: Groq (fast, free tier) + OpenRouter (broad model catalog).
The registry selects the best available model+provider for each call based on:
  - Current availability (not TPD-exhausted, not TPM-blocked)
  - Learned token capacity (models too small for the prompt are auto-skipped)
  - Capability tier (best quality model wins)

No hardcoded fallback chains. The system decides autonomously.

Usage:
    from llm_utils import call_llm, call_llm_json

    text   = call_llm(prompt="...", preferred_model="llama-3.3-70b-versatile")
    result = call_llm_json(prompt="...", preferred_model="llama-3.3-70b-versatile")
"""

import re
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from groq import Groq
from config import GROQ_API_KEYS, OPENROUTER_API_KEYS
from model_registry import registry

GROQ_API_KEY       = GROQ_API_KEYS[0]       if GROQ_API_KEYS       else ""
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else ""

try:
    from groq import RateLimitError as _GroqRLE
except ImportError:
    _GroqRLE = Exception

try:
    from openai import OpenAI as _OpenAI, RateLimitError as _OpenAIRLE
    _HAS_OPENAI = True
except ImportError:
    _OpenAI = None
    _OpenAIRLE = type("_never", (), {})
    _HAS_OPENAI = False

_RateLimitErrors = (_GroqRLE, _OpenAIRLE)

 
# Clients lazy singletons
 

_client_cache: dict = {}


def _client_for(provider: str, api_key: str = "") -> Groq:
    """Return a cached client for the given provider and key."""
    key = f"{provider}::{api_key}"
    if key in _client_cache:
        return _client_cache[key]
    if provider == "openrouter":
        if not _HAS_OPENAI:
            raise RuntimeError("openai package required for OpenRouter  pip install openai")
        client = _OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://github.com/geo-pipeline",
                "X-Title": "GEO Pipeline",
            },
        )
    else:
        client = Groq(api_key=api_key or GROQ_API_KEY)
    _client_cache[key] = client
    return client


 
# Helpers
 

def _parse_wait_seconds(msg: str) -> float:
    m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", msg)
    if not m:
        return 60.0
    return float(m.group(1) or 0) * 3600 + float(m.group(2) or 0) * 60 + float(m.group(3) or 0)


def _parse_413(msg: str):
    """Extract (limit, requested) token counts from 413 message."""
    m = re.search(r"Limit\s+(\d+),\s*Requested\s+(\d+)", msg, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _is_daily_limit(msg: str) -> bool:
    return "tokens per day" in msg or "per day" in msg or "daily" in msg.lower()


def _is_too_large(msg: str) -> bool:
    return "413" in msg or "request too large" in msg.lower() or "context_length_exceeded" in msg.lower()


def _is_insufficient_credits(msg: str) -> bool:
    return "402" in msg or "requires more credits" in msg.lower() or "can only afford" in msg.lower()


def _estimate_tokens(messages: list) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


 
# Core call
 

def call_llm(
    prompt: str,
    system: str = "",
    preferred_model: str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.2,
    tpm_max_wait: float = 90.0,
    max_attempts: int = 10,
) -> str:
    """Call the best available model with automatic multi-provider failover.

    Returns response text. Raises RuntimeError if all models are exhausted.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    estimated = _estimate_tokens(messages)
    last_error = None

    for attempt in range(max_attempts):
        model, provider, api_key = registry.select(
            preferred=preferred_model,
            estimated_tokens=estimated,
        )

        try:
            client = _client_for(provider, api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_completion_tokens,
            )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                print(f"[llm] {provider}/{model}: empty response — blocking 60s")
                registry.mark_tpm(model, provider, 60, api_key)
                continue
            return content.strip()

        except _RateLimitErrors as e:
            msg = str(e)
            last_error = e

            if _is_too_large(msg):
                parsed = _parse_413(msg)
                if parsed:
                    registry.mark_too_large(model, provider, *parsed)
                else:
                    registry.mark_tpm(model, provider, 3600, api_key)
                continue

            if _is_daily_limit(msg) or _is_insufficient_credits(msg):
                registry.mark_tpd_exhausted(model, provider, api_key)
                print(f"[llm] {provider}/{model}: credits/daily limit — skipping")
                continue

            wait = _parse_wait_seconds(msg)
            if wait <= tpm_max_wait:
                registry.mark_tpm(model, provider, wait, api_key)
                print(f"[llm] {provider}/{model}: TPM — waiting {wait:.0f}s...")
                time.sleep(wait + 1)
            else:
                registry.mark_tpm(model, provider, wait, api_key)
                print(f"[llm] {provider}/{model}: TPM wait {wait:.0f}s too long — skipping")
                continue

        except Exception as e:
            msg = str(e)
            last_error = e
            print(f"[llm] {provider}/{model}: error ({e})")

            if _is_too_large(msg):
                parsed = _parse_413(msg)
                if parsed:
                    registry.mark_too_large(model, provider, *parsed)
                else:
                    registry.mark_tpm(model, provider, 3600, api_key)
            elif _is_insufficient_credits(msg):
                print(f"[llm] {provider}/{model}: insufficient credits — skipping permanently")
                registry.mark_tpd_exhausted(model, provider, api_key)
            elif "400" in msg or "invalid_request_error" in msg:
                registry.mark_tpm(model, provider, 3600, api_key)
            else:
                time.sleep(2 ** min(attempt, 4))
            continue

    raise RuntimeError(
        f"All models exhausted after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


 
# JSON wrapper
 

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
