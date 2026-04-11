"""Dynamic multi-provider model registry for the GEO pipeline.

Providers supported: Groq, OpenRouter (added automatically if key present).

At startup:
  1. Discovers all available models from each provider's API.
  2. Assigns capability tiers for ranking.

At runtime:
  3. Tracks per-model state: TPM blocks, TPD exhaustion, learned token capacity.
  4. select() returns the best available model+provider for the current prompt size.
     — The system decides autonomously. No hardcoded model names or fallback chains.

When a 413 (request too large) occurs, the model's capacity is learned and stored.
Future calls with large prompts automatically skip models that are too small.
"""

import re
import time
import threading
import requests
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GROQ_API_KEYS, OPENROUTER_API_KEYS

# File to persist TPD-exhausted models across runs (reset daily)
_EXHAUSTED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "agent2_output", "exhausted_models.json")


def _load_exhausted() -> dict:
    """Load persisted TPD exhaustion state. Returns {slot: exhausted_at_timestamp}."""
    if not os.path.exists(_EXHAUSTED_FILE):
        return {}
    try:
        with open(_EXHAUSTED_FILE, "r") as f:
            data = json.load(f)
        # Discard entries older than 24 hours (quota resets daily)
        now = time.time()
        return {k: v for k, v in data.items() if now - v < 86400}
    except Exception:
        return {}


def _save_exhausted(slot: str):
    """Persist a newly exhausted model slot to disk."""
    os.makedirs(os.path.dirname(_EXHAUSTED_FILE), exist_ok=True)
    data = _load_exhausted()
    data[slot] = time.time()
    try:
        with open(_EXHAUSTED_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Capability tiers — same keywords apply across all providers
# ---------------------------------------------------------------------------

_TIER_KEYWORDS = [
    # Tier 0 — largest / best quality
    ["70b", "72b", "120b", "405b", "scout", "maverick",
     "qwen3-32b", "qwen-32b", "claude-3-5", "claude-3.5", "claude-3-opus",
     "gemini-pro", "gemini-1.5-pro", "llama-4", "deepseek-r1", "r1"],
    # Tier 1 — medium
    ["32b", "27b", "22b", "13b", "14b", "20b", "8b-instant",
     "claude-3-haiku", "gemini-flash", "gemini-1.5-flash",
     "mistral-small", "mistral-medium", "qwen-14b", "small"],
    # Tier 2 — fast / lightweight
    ["8b", "7b", "3b", "instant", "mini", "nano", "phi-3"],
]

# Provider priority for tie-breaking — prefer faster/cheaper providers first
_PROVIDER_PRIORITY = {"groq": 0, "mistral": 1, "openrouter": 2}


def _tier(model_id: str) -> int:
    mid = model_id.lower().replace(":free", "")  # strip OpenRouter :free suffix
    for i, keywords in enumerate(_TIER_KEYWORDS):
        if any(kw in mid for kw in keywords):
            return i
    return 99  # unknown — lowest preference


# ---------------------------------------------------------------------------
# Per-model state
# ---------------------------------------------------------------------------

class _ModelState:
    def __init__(self, model_id: str, provider: str, api_key: str,
                 key_index: int = 0, context_length: int = 0):
        self.id             = model_id
        self.provider       = provider          # "groq" | "openrouter"
        self.api_key        = api_key           # actual key to use for this slot
        self.key_index      = key_index         # which key in the pool (0-based)
        self.context_length = context_length    # from API, 0 = unknown
        self.tpd_exhausted  = False
        self.tpm_retry_at   = 0.0
        # Learned from 413 errors — the actual token-per-request cap.
        # None = unknown, int = learned cap in tokens.
        self.token_capacity: int | None = None

    @property
    def available(self) -> bool:
        if self.tpd_exhausted:
            return False
        if self.tpm_retry_at > time.time():
            return False
        return True

    def fits(self, estimated_tokens: int) -> bool:
        """Return False only if we *know* this model cannot handle the prompt."""
        if estimated_tokens == 0:
            return True
        # Use learned capacity if available, else fall back to context_length
        cap = self.token_capacity
        if cap is None and self.context_length > 0:
            cap = self.context_length
        if cap is None:
            return True  # unknown — try optimistically
        return estimated_tokens <= cap

    def mark_tpd_exhausted(self):
        self.tpd_exhausted = True
        print(f"[registry] {self.provider}[key{self.key_index}]/{self.id}: daily limit exhausted")

    def mark_tpm(self, retry_after_seconds: float):
        self.tpm_retry_at = time.time() + retry_after_seconds
        print(f"[registry] {self.provider}/{self.id}: TPM — retry in {retry_after_seconds:.0f}s")

    def learn_capacity(self, limit: int):
        if self.token_capacity is None or limit < self.token_capacity:
            self.token_capacity = limit
            print(f"[registry] {self.provider}/{self.id}: learned capacity = {limit} tokens")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Multi-provider model registry. Thread-safe."""

    _SKIP_PATTERNS = re.compile(
        r"whisper|tts|vision|guard|distil|embed|rerank|reward|tool|"
        r"image|audio|speech|video|ocr|translate",
        re.I,
    )

    def __init__(self):
        self._lock   = threading.Lock()
        self._states: dict[str, _ModelState] = {}  # key = "provider/model_id"
        self._loaded = False

    # ── Discovery ──────────────────────────────────────────────────────────────

    def _load(self):
        self._load_groq()
        self._load_openrouter()
        self._load_mistral()
        # Restore persisted TPD exhaustion state from previous runs
        exhausted = _load_exhausted()
        restored = 0
        for slot, ts in exhausted.items():
            if slot in self._states:
                self._states[slot].tpd_exhausted = True
                restored += 1
        if restored:
            print(f"[registry] Restored {restored} TPD-exhausted models from disk")
        total = len(self._states)
        by_provider = {}
        for s in self._states.values():
            by_provider.setdefault(s.provider, []).append(s.id)
        for prov, ids in by_provider.items():
            print(f"[registry] {prov}: {len(ids)} models available")
        print(f"[registry] Total pool: {total} models across {len(by_provider)} providers")
        self._loaded = True

    def _load_groq(self):
        if not GROQ_API_KEYS:
            print("[registry] Groq: no API keys — skipping")
            return
        _DEFAULT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                           "qwen/qwen3-32b", "llama3-8b-8192"]
        for ki, api_key in enumerate(GROQ_API_KEYS):
            try:
                resp = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                resp.raise_for_status()
                models = resp.json().get("data", [])
                added = 0
                for m in models:
                    mid = m.get("id", "")
                    if self._SKIP_PATTERNS.search(mid):
                        continue
                    slot = f"groq:{ki}/{mid}"
                    self._states[slot] = _ModelState(mid, "groq", api_key,
                                                     key_index=ki)
                    added += 1
                print(f"[registry] Groq key{ki}: discovered {added} models")
            except Exception as e:
                print(f"[registry] Groq key{ki} discovery failed ({e}) — using defaults")
                for mid in _DEFAULT_MODELS:
                    slot = f"groq:{ki}/{mid}"
                    self._states[slot] = _ModelState(mid, "groq", api_key,
                                                     key_index=ki)

    def _load_openrouter(self):
        if not OPENROUTER_API_KEYS:
            print("[registry] OpenRouter: no API keys — skipping")
            return
        for ki, api_key in enumerate(OPENROUTER_API_KEYS):
            try:
                resp = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                resp.raise_for_status()
                models = resp.json().get("data", [])
                added = 0
                for m in models:
                    mid = m.get("id", "")
                    ctx = int(m.get("context_length") or 0)
                    if self._SKIP_PATTERNS.search(mid):
                        continue
                    if ctx > 0 and ctx < 8192:
                        continue
                    slot = f"openrouter:{ki}/{mid}"
                    self._states[slot] = _ModelState(mid, "openrouter", api_key,
                                                     key_index=ki, context_length=ctx)
                    added += 1
                print(f"[registry] OpenRouter key{ki}: discovered {added} models")
            except Exception as e:
                print(f"[registry] OpenRouter key{ki} discovery failed ({e}) — skipping")

    def _load_mistral(self):
        from config import MISTRAL_API_KEYS
        if not MISTRAL_API_KEYS:
            print("[registry] Mistral: no API keys — skipping")
            return
        _MISTRAL_TOOL_MODELS = [
            "mistral-small-latest",
            "mistral-medium-latest",
            "open-mixtral-8x7b",
        ]
        for ki, api_key in enumerate(MISTRAL_API_KEYS):
            try:
                resp = requests.get(
                    "https://api.mistral.ai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                resp.raise_for_status()
                models = resp.json().get("data", [])
                added = 0
                for m in models:
                    mid = m.get("id", "")
                    if self._SKIP_PATTERNS.search(mid):
                        continue
                    if mid not in _MISTRAL_TOOL_MODELS:
                        continue  # only tool-capable models
                    slot = f"mistral:{ki}/{mid}"
                    self._states[slot] = _ModelState(mid, "mistral", api_key, key_index=ki)
                    added += 1
                print(f"[registry] Mistral key{ki}: {added} tool-capable models")
            except Exception as e:
                print(f"[registry] Mistral key{ki} discovery failed ({e}) — using defaults")
                for mid in _MISTRAL_TOOL_MODELS:
                    slot = f"mistral:{ki}/{mid}"
                    self._states[slot] = _ModelState(mid, "mistral", api_key, key_index=ki)

    def _ensure_loaded(self):
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def select(self, preferred: str | None = None,
               estimated_tokens: int = 0) -> tuple[str, str, str]:
        """Return (model_id, provider, api_key) for the best available model.

        Selection logic (fully autonomous):
        1. If `preferred` model is available on any key/provider and fits
           the prompt size — use it (Groq key0 preferred for speed/cost).
        2. Otherwise pick the best available slot by:
           a. Capability tier (lower = better)
           b. Provider priority (Groq before OpenRouter)
           c. Key index (lower = earlier account, usually more quota)
           d. Alphabetical for stability

        Raises RuntimeError if no models are available at all.
        """
        self._ensure_loaded()
        with self._lock:
            # Try preferred model across all keys, Groq first
            if preferred:
                for s in sorted(self._states.values(),
                                key=lambda x: (_PROVIDER_PRIORITY.get(x.provider, 99), x.key_index)):
                    if s.id == preferred and s.available and s.fits(estimated_tokens):
                        return s.id, s.provider, s.api_key

            # Filter all available slots that fit the prompt size
            candidates = [
                s for s in self._states.values()
                if s.available and s.fits(estimated_tokens)
            ]

            if not candidates:
                candidates = [s for s in self._states.values() if s.available]
                if candidates:
                    candidates.sort(key=lambda s: (
                        _tier(s.id),
                        _PROVIDER_PRIORITY.get(s.provider, 99),
                        s.key_index,
                        s.id,
                    ))
                    chosen = candidates[0]
                    print(f"[registry] No model fits {estimated_tokens} tokens — "
                          f"trying: {chosen.provider}[key{chosen.key_index}]/{chosen.id}")
                    return chosen.id, chosen.provider, chosen.api_key

                exhausted   = [f"{s.provider}[k{s.key_index}]/{s.id}"
                               for s in self._states.values() if s.tpd_exhausted]
                tpm_blocked = [f"{s.provider}[k{s.key_index}]/{s.id}"
                               for s in self._states.values()
                               if not s.tpd_exhausted and not s.available]
                raise RuntimeError(
                    f"No models available across all providers and keys.\n"
                    f"  Daily exhausted : {exhausted}\n"
                    f"  TPM blocked     : {tpm_blocked}"
                )

            candidates.sort(key=lambda s: (
                _tier(s.id),
                _PROVIDER_PRIORITY.get(s.provider, 99),
                s.key_index,
                s.id,
            ))
            chosen = candidates[0]
            print(f"[registry] Selected: {chosen.provider}[key{chosen.key_index}]/{chosen.id}")
            return chosen.id, chosen.provider, chosen.api_key

    def mark_tpd_exhausted(self, model_id: str, provider: str, api_key: str = ""):
        """Mark all slots for this model+provider+key as TPD exhausted and persist to disk."""
        self._ensure_loaded()
        with self._lock:
            for slot, s in self._states.items():
                if s.id == model_id and s.provider == provider:
                    if not api_key or s.api_key == api_key:
                        s.mark_tpd_exhausted()
                        _save_exhausted(slot)  # persist across runs

    def mark_tpm(self, model_id: str, provider: str, retry_after_seconds: float,
                 api_key: str = ""):
        self._ensure_loaded()
        with self._lock:
            for s in self._states.values():
                if s.id == model_id and s.provider == provider:
                    if not api_key or s.api_key == api_key:
                        s.mark_tpm(retry_after_seconds)

    def mark_too_large(self, model_id: str, provider: str, limit: int, requested: int):
        """413: record learned capacity so future large prompts skip this model."""
        self._ensure_loaded()
        with self._lock:
            for s in self._states.values():
                if s.id == model_id and s.provider == provider:
                    s.learn_capacity(limit)
        print(f"[registry] {provider}/{model_id}: too small for {requested} tokens "
              f"(cap={limit}) — auto-skipped for larger prompts")

    def status(self) -> dict:
        self._ensure_loaded()
        with self._lock:
            return {
                f"{s.provider}/{s.id}": {
                    "available":      s.available,
                    "tpd_exhausted":  s.tpd_exhausted,
                    "tpm_retry_in":   max(0.0, s.tpm_retry_at - time.time()),
                    "token_capacity": s.token_capacity,
                    "context_length": s.context_length,
                }
                for s in self._states.values()
            }


# Singleton — shared across all agents in the process.
registry = ModelRegistry()
