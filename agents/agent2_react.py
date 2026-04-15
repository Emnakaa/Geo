"""Agent 2 (ReAct/MCP) - Web feature extraction via MCP tool servers.

Replaces the hardcoded process_entity() loop with a LangGraph ReAct agent
that decides which tools to call per entity based on what it finds.

All entities are researched inside a single async context so MCP server
subprocesses are started once and reused — avoiding per-entity startup cost.
"""

import sys
import os
import json
import asyncio
import csv
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, GROQ_MODEL, OPENROUTER_API_KEY
from pipeline_state import PipelineState
from model_registry import registry as _registry
from agents.supervisor_agent import SupervisorAgent

_supervisor = SupervisorAgent(_registry)

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# MCP server definitions
# Use sys.executable so the correct Python interpreter is always found.
# ---------------------------------------------------------------------------

_SERVERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_servers")

MCP_SERVERS = {
    "search": {
        "command": sys.executable,
        "args": [os.path.join(_SERVERS_DIR, "search_server.py")],
        "transport": "stdio",
    },
    "scrape": {
        "command": sys.executable,
        "args": [os.path.join(_SERVERS_DIR, "scrape_server.py")],
        "transport": "stdio",
    },
    "wiki": {
        "command": sys.executable,
        "args": [os.path.join(_SERVERS_DIR, "wiki_server.py")],
        "transport": "stdio",
    },
    "enrichment": {
        "command": sys.executable,
        "args": [os.path.join(_SERVERS_DIR, "enrichment_server.py")],
        "transport": "stdio",
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a GEO research agent. Collect REAL web data for a TUNISIAN restaurant. Never invent values — use null if a tool returns nothing.

CRITICAL RULE — TUNISIA ONLY:
- You are researching restaurants located in TUNISIA (North Africa).
- Always include "Tunisia" or a Tunisian city (Tunis, Sfax, Sousse, Djerba, Hammamet, Nabeul, Bizerte, Kairouan) in every search query.
- If a result shows a country other than Tunisia (e.g. France, Morocco, Spain, Italy, Nicaragua) — DISCARD it and search again with more specific Tunisian location terms.
- If wikidata_lookup returns wd_country != Tunisia/Tunisie — set wd_country=null and wd_entity_type=null (wrong entity).
- If tripadvisor_search returns a URL for a non-Tunisian city — discard and try again with the Tunisian city name.

STEPS (in order):
1. google_maps_search(entity, location="Tunisia")
2. wikipedia_lookup(entity + " Tunisia")
3. tripadvisor_search(entity, location="Tunisia")
4. wikidata_lookup(entity + " Tunisia")
5. If website found in step 1: extract_website_socials(website_url)
6. If ig_handle found: scrape_instagram(handle)
7. If fb_handle found: scrape_facebook(handle)
8. If no social found: ddg_search("<entity> tunisia instagram OR facebook")

After all steps output ONLY this JSON:
{"canonical_entity":"<name>","is_restaurant":true/false,"gm_rating":null,"gm_review_count":null,"gm_address":null,"gm_phone":null,"gm_website":null,"gm_category":null,"ta_rating":null,"ta_review_count":null,"ta_url":null,"ta_ranking":null,"has_wikipedia":false,"wd_entity_type":null,"wd_country":null,"wd_official_website":null,"wd_founded":null,"ig_handle":null,"ig_followers":null,"ig_posts":null,"ig_engagement_rate":null,"ig_bio":null,"fb_handle":null,"fb_page_likes":null,"fb_post_engagement":null,"overall_confidence":0.0}
Replace nulls/false with real tool data. overall_confidence: 0.0=no data, 0.5=partial, 0.9+=rating+reviews+2 signals."""

# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def _compute_derived(row: dict) -> dict:
    try:
        from rapidfuzz import fuzz
        bio_match = fuzz.partial_ratio(
            (row.get("canonical_entity") or "").lower(),
            (row.get("ig_bio") or "").lower()
        ) > 60
    except Exception:
        bio_match = False

    hw  = bool(row.get("gm_website") or row.get("wd_official_website"))
    hp  = bool(row.get("gm_phone"))
    ha  = bool(row.get("gm_address"))
    hi  = bool(row.get("ig_handle"))
    hf  = bool(row.get("fb_handle"))
    ht  = bool(row.get("ta_url"))
    hwd = bool(row.get("wd_entity_type"))
    hwk = bool(row.get("has_wikipedia"))

    return {
        "has_website":           hw,
        "has_phone":             hp,
        "has_address":           ha,
        "has_instagram":         hi,
        "has_facebook":          hf,
        "has_tripadvisor":       ht,
        "has_wikidata":          hwd,
        "data_source_count":     sum([hw, hp, hi, hf, ht, hwk, hwd]),
        "review_total":          (row.get("gm_review_count") or 0) + (row.get("ta_review_count") or 0),
        "ig_bio_mentions_brand": bio_match,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

OUTPUT_DIR      = "agent2_output"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "web_features.csv")


def _load_done() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(CHECKPOINT_FILE)
        # Only count rows that were successfully enriched (confidence > 0, no error)
        if "overall_confidence" in df.columns:
            mask = df["overall_confidence"].fillna(0) > 0
            if "error" in df.columns:
                mask = mask & df["error"].isna()
            df = df[mask]
        return set(df["canonical_entity"].dropna().str.lower().tolist())
    except Exception:
        return set()


# Fixed column schema — every row written has exactly these columns
_CSV_COLUMNS = [
    "canonical_entity", "is_restaurant", "data_quality",
    # Google Maps
    "gm_rating", "gm_review_count", "gm_address", "gm_phone", "gm_website", "gm_category",
    # TripAdvisor
    "ta_rating", "ta_review_count", "ta_url", "ta_ranking",
    # Wikipedia
    "has_wikipedia",
    # Wikidata
    "wd_entity_type", "wd_country", "wd_official_website", "wd_founded",
    # Social
    "ig_handle", "ig_followers", "ig_posts", "ig_engagement_rate", "ig_bio",
    "fb_handle", "fb_page_likes", "fb_post_engagement",
    "overall_confidence", "scrape_timestamp",
    # derived
    "has_website", "has_phone", "has_address", "has_instagram",
    "has_facebook", "has_tripadvisor", "has_wikidata",
    "data_source_count", "review_total", "ig_bio_mentions_brand",
    # error info
    "parse_error", "error",
]

def _sanitise(value) -> str:
    """Convert any value to a CSV-safe string.

    Specifically:
    - None / NaN  → empty string
    - Booleans    → "True" / "False"  (not Python repr)
    - Strings     → strip embedded newlines, tabs, and carriage returns
                    (these break the pandas C CSV parser even inside quoted fields)
    - Numbers     → str as-is
    Truncates strings longer than 500 chars to avoid runaway values.
    """
    if value is None:
        return ""
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        # Replace all whitespace control chars except regular space
        cleaned = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # Collapse multiple spaces from the replacement
        import re as _re
        cleaned = _re.sub(r" {2,}", " ", cleaned).strip()
        return cleaned[:500]
    return str(value)


def _save_row(row: dict):
    """Write one research result row to the checkpoint CSV.

    Sanitises every value at write time so the output CSV is always
    parseable without post-hoc cleaning — no embedded newlines, no
    schema mismatches, no truncation surprises.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exists = os.path.exists(CHECKPOINT_FILE)
    # Normalise to fixed schema + sanitise every value
    normalised = {col: _sanitise(row.get(col, "")) for col in _CSV_COLUMNS}
    with open(CHECKPOINT_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, quoting=csv.QUOTE_ALL,
                           extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(normalised)


# ---------------------------------------------------------------------------
# Async research — all entities share one MCP client session
# ---------------------------------------------------------------------------

# Expected keys in a valid research result
_RESULT_KEYS = {
    "canonical_entity", "is_restaurant", "gm_rating", "gm_review_count",
    "gm_address", "gm_phone", "gm_website", "gm_category",
    "ta_rating", "ta_review_count", "ta_url",
    "has_wikipedia",
    "wd_entity_type", "wd_country",
    "ig_handle", "ig_followers", "ig_posts", "ig_engagement_rate", "ig_bio",
    "fb_handle", "fb_page_likes", "fb_post_engagement", "overall_confidence",
}

def _is_valid_result(data: dict, entity: str) -> bool:
    """Return True only if the parsed dict looks like a real research result."""
    if not isinstance(data, dict):
        return False
    # Must have at least half the expected keys
    overlap = len(_RESULT_KEYS & set(data.keys()))
    if overlap < len(_RESULT_KEYS) // 2:
        return False
    # canonical_entity must match what we asked for (not a tool name or garbage)
    ce = str(data.get("canonical_entity", "")).lower().strip()
    if not ce or ce in {"google_maps_search", "wikipedia_lookup", "tripadvisor_search",
                        "ddg_search", "scrape_instagram", "scrape_facebook",
                        "extract_website_socials"}:
        return False
    # overall_confidence must be a number 0-1
    try:
        conf = float(data.get("overall_confidence", -1))
        if not (0.0 <= conf <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _parse_row(text: str, entity: str) -> dict:
    """Extract and validate JSON from agent response.

    Returns a valid result dict, or an empty-data dict with parse_error set.
    Never returns hallucinated or corrupted data.
    """
    empty = {
        "canonical_entity":  entity,
        "overall_confidence": 0.0,
        "data_quality":       "parse_failed",
    }

    clean = text.strip()
    # Strip markdown fences
    if "```" in clean:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean)
        if m:
            clean = m.group(1).strip()

    # Try full parse
    try:
        data = json.loads(clean)
        if _is_valid_result(data, entity):
            data["data_quality"] = "ok"
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting first {...} block
    m = re.search(r"\{[\s\S]*?\}", clean)
    if m:
        try:
            data = json.loads(m.group())
            if _is_valid_result(data, entity):
                data["data_quality"] = "ok"
                return data
        except json.JSONDecodeError:
            pass

    empty["parse_error"] = clean[:300]
    return empty


def _make_llm(model_id: str, provider: str, api_key: str = ""):
    """Instantiate the right LLM client for a given model+provider+key."""
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or OPENROUTER_API_KEY,
            model=model_id,
            temperature=0.0,
            max_tokens=4096,   # cap output — Venice and other providers cap at 16384
            default_headers={
                "HTTP-Referer": "https://github.com/geo-pipeline",
                "X-Title": "GEO Pipeline",
            },
        )
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            api_key=api_key,
            model=model_id,
            temperature=0.0,
            max_tokens=4096,
        )
    return ChatGroq(api_key=api_key or GROQ_API_KEY, model=model_id, temperature=0.0,
                    max_tokens=4096)


# Error classification delegated to SupervisorAgent (agents/supervisor_agent.py).
# The supervisor uses an LLM to reason about errors rather than hardcoded patterns.


# Models known to support tool calling — in priority order.
# Groq first (fastest, highest quality), then OpenRouter as fallback.
# Registry skips TPD-exhausted slots automatically — no hardcoded provider skip.
_TOOL_CAPABLE = [
    "llama-3.3-70b-versatile",              # Groq  — priority 1
    "meta-llama/llama-3.3-70b-instruct",    # OpenRouter — priority 2
    "mistralai/mistral-small",              # OpenRouter
    "mistralai/mistral-medium",             # OpenRouter
    "anthropic/claude-3-haiku",             # OpenRouter
    "openai/gpt-4o-mini",                   # OpenRouter
    "google/gemini-flash-1.5",              # OpenRouter
    "qwen/qwen3-32b",                       # OpenRouter
]

def _batch_size(provider: str) -> int:
    # All providers run sequentially (batch=1) to avoid TPM/RPM rate limits.
    # Groq TPM=12k/min, parallel batches multiply token usage and cause 429s.
    return 1


async def _research_one(entity: str, tools: list,
                         model_id: str, provider: str, api_key: str,
                         idx: int, total: int) -> dict:
    """Research a single entity with its own agent instance. Handles TPD/TPM retry."""
    current_model    = model_id
    current_provider = provider
    current_api_key  = api_key
    llm   = _make_llm(current_model, current_provider, current_api_key)
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    print(f"[{idx}/{total}] START: {entity} [{current_provider}/{current_model}]")
    attempt = 0
    row = None
    while row is None and attempt < 6:  # extra headroom for wait_retry cycles
        attempt += 1
        try:
            response = await agent.ainvoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Research this restaurant entity:\n"
                        f"Entity: {entity}\n"
                        f"Domain: Tunisian restaurants\n"
                        f"Follow the research protocol and return the JSON."
                    ),
                }]
            })
            row = _parse_row(response["messages"][-1].content, entity)

        except Exception as e:
            err_msg = str(e)
            print(f"[{idx}/{total}] ERROR {entity} (attempt {attempt}): {e}")

            # ── Supervisor reasons about the error and decides the action ──────
            decision = _supervisor.decide(
                error_message=err_msg,
                entity=entity,
                model_id=current_model,
                provider=current_provider,
                api_key=current_api_key,
                attempt=attempt,
            )
            action = decision["action"]

            # Mark exhausted BEFORE switching so registry skips it immediately
            if decision.get("mark_exhausted"):
                _registry.mark_tpd_exhausted(current_model, current_provider, current_api_key)
                attempt -= 1  # exhaustion switch is free — don't burn an attempt

            if action == "switch_model":
                # Iterate _TOOL_CAPABLE; registry.select() naturally skips exhausted
                # slots. No provider is hardcoded — the system self-heals by exhausting
                # each model in turn until one works.
                switched = False
                for cap_model in _TOOL_CAPABLE:
                    try:
                        m_id, m_prov, m_key = _registry.select(preferred=cap_model)
                        if m_id not in _TOOL_CAPABLE:
                            continue  # registry gave a random fallback — skip
                        if m_id == current_model and m_prov == current_provider:
                            continue  # skip the model that just failed
                        current_model, current_provider, current_api_key = m_id, m_prov, m_key
                        llm   = _make_llm(current_model, current_provider, current_api_key)
                        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
                        print(f"[{idx}/{total}] Supervisor: switched to "
                              f"{current_provider}/{current_model}")
                        switched = True
                        break
                    except RuntimeError:
                        continue
                # if not switched: no model available right now — loop continues,
                # next attempt will retry; eventually hits attempt >= 6 and exits.

            elif action == "wait_retry":
                wait = float(decision.get("wait_seconds") or 10)
                _registry.mark_tpm(current_model, current_provider, wait, current_api_key)
                print(f"[{idx}/{total}] Supervisor: waiting {wait:.0f}s then retrying")
                await asyncio.sleep(wait + 1)
                attempt -= 1  # RPM wait is free — don't burn an attempt

            elif action == "retry":
                wait = float(decision.get("wait_seconds") or 0)
                if wait > 0:
                    await asyncio.sleep(wait)

            else:  # "skip"
                row = {"canonical_entity": entity, "error": err_msg[:300],
                       "overall_confidence": 0.0}

    if row is None:
        row = {"canonical_entity": entity, "error": "max attempts reached",
               "overall_confidence": 0.0}

    row["scrape_timestamp"] = datetime.now().isoformat()
    derived = _compute_derived(row)
    row.update(derived)
    _save_row(row)
    print(f"[{idx}/{total}] DONE: {entity} "
          f"confidence={row.get('overall_confidence','?')} "
          f"gm_rating={row.get('gm_rating','?')}")
    return row


async def _research_all_async(entities: list[str], initial_model: str,
                               initial_provider: str, initial_api_key: str = "") -> list[dict]:
    """Research all entities in parallel batches. Each entity has its own agent.
    MCP client is recreated per batch to avoid stdio connection timeouts on long runs.
    """
    total   = len(entities)
    results = []

    bsize = _batch_size(initial_provider)
    print(f"[Agent2] Provider={initial_provider} model={initial_model} batch_size={bsize}")

    # MCP client reconnects every N entities to avoid stdio connection timeout.
    # Reconnect interval: larger batches for fast providers, 4 for slow/free ones.
    reconnect_every = max(bsize, 4)

    batch_start = 0
    batch_num   = 0
    tools       = None

    while batch_start < total:
        batch = entities[batch_start: batch_start + bsize]
        batch_num += 1
        print(f"\n[Agent2] Batch {batch_num} — "
              f"entities {batch_start + 1}–{min(batch_start + bsize, total)}/{total}")

        # Reconnect MCP client every `reconnect_every` entities
        if tools is None or (batch_start % reconnect_every == 0 and batch_start > 0):
            mcp_client = MultiServerMCPClient(MCP_SERVERS)
            tools = await mcp_client.get_tools()
            if batch_num == 1:
                print(f"[Agent2] MCP tools: {[t.name for t in tools]}")

        tasks = [
            _research_one(
                entity   = entity,
                tools    = tools,
                model_id = initial_model,
                provider = initial_provider,
                api_key  = initial_api_key,
                idx      = batch_start + i + 1,
                total    = total,
            )
            for i, entity in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        connection_errors = []
        for i, res in enumerate(batch_results):
            if isinstance(res, Exception):
                ent_name = batch[i] if i < len(batch) else "unknown"
                err_str  = str(res)
                print(f"[Agent2] Unhandled exception for '{ent_name}': {err_str[:120]}")
                if "connection" in err_str.lower() or "connect" in err_str.lower():
                    connection_errors.append((i, ent_name))
                error_row = {"canonical_entity": ent_name, "error": err_str[:300],
                             "overall_confidence": 0.0}
                _save_row(error_row)
                results.append(error_row)
            else:
                results.append(res)

        # If connection errors occurred, force MCP reconnect before next batch
        if connection_errors:
            print(f"[Agent2] {len(connection_errors)} connection error(s) — "
                  f"forcing MCP reconnect before next batch")
            try:
                mcp_client = MultiServerMCPClient(MCP_SERVERS)
                tools = await mcp_client.get_tools()
            except Exception as e:
                print(f"[Agent2] MCP reconnect failed: {e}")
                tools = None

        batch_start += bsize

    return results


# ---------------------------------------------------------------------------
# Main orchestrator (sync entry point)
# ---------------------------------------------------------------------------

def run_agent2_react(entities: list[str], fresh_start: bool = False) -> list[dict]:
    """Research all entities using MCP tools. Returns list of feature dicts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if fresh_start and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    done  = _load_done()
    queue = [e for e in entities if e.lower() not in done]
    print(f"\n[Agent2-ReAct] {len(queue)} to research, {len(done)} already done")

    if not queue:
        # All done — reload from checkpoint
        import pandas as pd
        return pd.read_csv(CHECKPOINT_FILE).to_dict(orient="records")

    # Select initial model — try tool-capable models in priority order.
    # Select first available tool-capable model. Registry skips TPD-exhausted slots
    # automatically — no provider is hardcoded here. The supervisor handles switching.
    model_id, provider, api_key = None, None, None
    for cap_model in _TOOL_CAPABLE:
        try:
            m_id, m_prov, m_key = _registry.select(preferred=cap_model)
            if m_id not in _TOOL_CAPABLE:
                continue  # registry returned a random fallback — skip
            model_id, provider, api_key = m_id, m_prov, m_key
            break
        except RuntimeError:
            continue
    if model_id is None:
        raise RuntimeError("No tool-capable models available across all providers")
    print(f"[Agent2-ReAct] Starting with {provider}/{model_id}")

    # Single event loop for all entities — model switching happens inside
    new_rows = asyncio.run(_research_all_async(queue, model_id, provider, api_key))
    return new_rows


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def run_agent2_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs Agent 2 (ReAct/MCP) and merges results into state."""
    errors = list(state.get("errors", []))

    # Only research entities with at least 2 mentions — single-mention entities
    # are likely hallucinations or noise; filtering them saves ~70% of Agent 2 calls.
    entities = [
        row["canonical_entity"]
        for row in state.get("entity_features_global", [])
        if row.get("canonical_entity") and (row.get("mention_count", 0) >= 2
           or row.get("stability_score", 0) >= 0.05)
    ]

    if not entities:
        errors.append("agent2: no entities in state - skipping")
        return {**state, "errors": errors, "current_step": "agent2_skipped"}

    try:
        web_rows = run_agent2_react(entities, fresh_start=False)
    except Exception as exc:
        errors.append(f"agent2: {exc}")
        web_rows = []

    return {
        **state,
        "web_features": web_rows,
        "errors":       errors,
        "current_step": "agent2",
    }
