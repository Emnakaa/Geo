"""Supervisor Agent — LLM-based error recovery for Agent 2.

Instead of hardcoded if/elif chains, a small LLM reasons about each error
and returns a structured recovery decision. This is the agentic approach:
the system reflects on what went wrong and decides what to do next.

The supervisor runs OUTSIDE the failing agent call, which is the key insight:
infrastructure errors (400, 429, TPD) kill the agent before it can think,
so a separate supervisor is needed to reason about them.

Decision schema:
{
  "action": "retry" | "switch_model" | "wait_retry" | "skip",
  "mark_exhausted": true/false,   # mark current model as TPD-exhausted
  "wait_seconds": 0,              # seconds to wait before retrying
  "reason": "short explanation"   # supervisor's reasoning (logged)
}
"""

import json
import re
import time
from typing import Optional


_SUPERVISOR_PROMPT = """You are a pipeline supervisor. An LLM agent failed with an error.
Your job: reason about the error and return a recovery decision as JSON.

Available actions:
- "switch_model"  : this model/provider cannot handle this request — mark exhausted, use next model
- "wait_retry"    : temporary rate limit — wait N seconds then retry same model
- "retry"         : transient error — retry immediately with same model
- "skip"          : unrecoverable for this entity — give up, save empty row

Rules:
- 400 + "max_tokens" or "max_completion_tokens" or "maximum allowed" → switch_model (provider caps output)
- 413 + "tokens per minute" → switch_model (model TPM budget too small)
- 429 + "tokens per minute" or "TPM" → wait_retry (extract seconds from message)
- 429 + "tokens per day" or "daily" or "TPD" → switch_model (daily quota exhausted)
- 429 + "requests per minute" or "RPM" or "free-models-per-min" → wait_retry (60s default)
- 402 or "requires more credits" or "can only afford" → switch_model (billing limit)
- "no endpoints found" or "does not support tool" or "does not support function" → switch_model
- 500 or 502 or 503 → retry (transient server error)
- attempt >= 4 → skip (too many failures)

Error message:
{error_message}

Current model: {model_id} | Provider: {provider} | Attempt: {attempt}

Respond ONLY with valid JSON, no explanation outside the JSON:
{{"action": "...", "mark_exhausted": true/false, "wait_seconds": 0, "reason": "..."}}"""


class SupervisorAgent:
    """
    LLM-based supervisor that reasons about agent errors and decides recovery actions.
    Uses a fast/cheap model — it only needs to parse an error message and return JSON.
    """

    def __init__(self, registry):
        self._registry = registry
        self._llm = None
        self._failure_history: dict[str, list[str]] = {}  # entity -> list of errors

    def _get_llm(self):
        """Lazy-init a fast LLM for supervision decisions."""
        if self._llm is not None:
            return self._llm
        # Use the best available model — supervisor reasoning is lightweight
        try:
            model_id, provider, api_key = self._registry.select()
            self._llm = (model_id, provider, api_key)
        except RuntimeError:
            self._llm = None
        return self._llm

    def _llm_decide(self, error_message: str, model_id: str,
                    provider: str, attempt: int) -> dict:
        """Call the LLM to reason about the error and return a decision."""
        llm_info = self._get_llm()
        if llm_info is None:
            return {"action": "skip", "mark_exhausted": False,
                    "wait_seconds": 0, "reason": "no models available for supervision"}

        sup_model, sup_provider, sup_key = llm_info

        prompt = _SUPERVISOR_PROMPT.format(
            error_message=error_message[:600],
            model_id=model_id,
            provider=provider,
            attempt=attempt,
        )

        try:
            if sup_provider == "groq":
                from langchain_groq import ChatGroq
                llm = ChatGroq(api_key=sup_key, model=sup_model,
                               temperature=0.0, max_tokens=256)
            elif sup_provider == "openrouter":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=sup_key, model=sup_model,
                    temperature=0.0, max_tokens=256,
                    default_headers={"HTTP-Referer": "https://github.com/geo-pipeline",
                                     "X-Title": "GEO Pipeline"},
                )
            elif sup_provider == "mistral":
                from langchain_mistralai import ChatMistralAI
                llm = ChatMistralAI(api_key=sup_key, model=sup_model,
                                    temperature=0.0, max_tokens=256)
            else:
                raise ValueError(f"Unknown provider: {sup_provider}")

            response = llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                # Validate required fields
                if "action" not in decision:
                    raise ValueError("missing 'action' field")
                decision.setdefault("mark_exhausted", False)
                decision.setdefault("wait_seconds", 0)
                decision.setdefault("reason", "")
                return decision

        except Exception as e:
            print(f"[supervisor] LLM reasoning failed ({e}) — falling back to rule-based")

        # Fallback: rule-based decision if LLM fails
        return self._rule_based_decide(error_message, attempt)

    def _rule_based_decide(self, error_message: str, attempt: int) -> dict:
        """Deterministic fallback — mirrors original if/elif logic."""
        msg = error_message.lower()

        # RPM (requests per minute) — always wait+retry, never count against attempt limit
        if "429" in msg and any(kw in msg for kw in (
                "limit_rpm", "requests per minute", "free-models-per-min",
                "rate limit exceeded", "temporarily rate-limited", "retry shortly")):
            # Extract wait time if given, default 10s for RPM
            m = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", error_message)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 10
            return {"action": "wait_retry", "mark_exhausted": False,
                    "wait_seconds": wait, "reason": f"RPM rate limit — wait {wait:.0f}s"}

        if attempt >= 4:
            return {"action": "skip", "mark_exhausted": False,
                    "wait_seconds": 0, "reason": "max attempts reached"}

        # 400 max_tokens
        if "400" in msg and ("max_tokens" in msg or "maximum allowed" in msg
                              or "max_completion_tokens" in msg):
            return {"action": "switch_model", "mark_exhausted": True,
                    "wait_seconds": 0, "reason": "provider max_tokens cap"}

        # No tool support
        if ("no endpoints found" in msg or "does not support tool" in msg
                or "does not support function" in msg):
            return {"action": "switch_model", "mark_exhausted": True,
                    "wait_seconds": 0, "reason": "model does not support tool calling"}

        # 413 TPM
        if "413" in msg and "tokens per minute" in msg:
            return {"action": "switch_model", "mark_exhausted": True,
                    "wait_seconds": 0, "reason": "413 TPM budget too small"}

        # TPD / billing
        if ("tokens per day" in msg or "daily" in msg or "402" in msg
                or "requires more credits" in msg or "can only afford" in msg):
            return {"action": "switch_model", "mark_exhausted": True,
                    "wait_seconds": 0, "reason": "daily quota or billing exhausted"}

        # TPM rate limit — extract wait time
        if "429" in msg and ("tokens per minute" in msg or "tpm" in msg
                              or "requests per minute" in msg or "free-models-per-min" in msg):
            m = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", error_message)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 30
            return {"action": "wait_retry", "mark_exhausted": False,
                    "wait_seconds": wait, "reason": f"TPM rate limit — wait {wait:.0f}s"}

        # Transient server errors
        if any(code in msg for code in ("500", "502", "503")):
            return {"action": "retry", "mark_exhausted": False,
                    "wait_seconds": 2, "reason": "transient server error"}

        # Unknown — skip after too many attempts
        return {"action": "skip", "mark_exhausted": False,
                "wait_seconds": 0, "reason": f"unrecognised error at attempt {attempt}"}

    def decide(self, error_message: str, entity: str, model_id: str,
               provider: str, api_key: str, attempt: int) -> dict:
        """
        Main entry point. Returns a recovery decision dict:
          action        : "retry" | "switch_model" | "wait_retry" | "skip"
          mark_exhausted: whether to mark current model as TPD-exhausted
          wait_seconds  : seconds to wait before next attempt
          reason        : supervisor's explanation (for logging)
        """
        # Track failure history per entity
        history = self._failure_history.setdefault(entity, [])
        history.append(error_message[:120])

        # Hard cap — never attempt more than 4 times per entity
        if attempt >= 4:
            return {"action": "skip", "mark_exhausted": False,
                    "wait_seconds": 0, "reason": "max attempts (4) reached"}

        decision = self._llm_decide(error_message, model_id, provider, attempt)

        print(f"[supervisor] {entity} | attempt {attempt} | "
              f"action={decision['action']} | {decision['reason']}")

        return decision
