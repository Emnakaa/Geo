# -*- coding: utf-8 -*-
"""Agent 1 — Entity extraction, enrichment, cleaning, and GEO feature computation."""

import os
import json
import re
import uuid
import time
import requests
import pandas as pd
import csv
from groq import Groq
from tabulate import tabulate

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY
from model_registry import registry as _model_registry

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

RateLimitError = (_GroqRLE, _OpenAIRLE)

# Client cache — keyed by "provider::api_key" so each key gets its own instance.
_client_cache: dict = {}


def _get_client(provider: str = "groq", api_key: str = ""):
    """Return an API client for the given provider and key.

    Clients are cached per (provider, api_key) pair so we don't recreate
    them on every call, but different keys get separate client instances.
    """
    cache_key = f"{provider}::{api_key}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    if provider == "openrouter":
        if not _HAS_OPENAI:
            raise RuntimeError("pip install openai required for OpenRouter")
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

    _client_cache[cache_key] = client
    return client

from config import QUERY_MODELS as QUERY_MODELS

MODEL_EXTRACTOR       ="mistral-small-latest" #"llama-3.3-70b-versatile"
MODEL_EXTRACTOR2      ="llama-3.3-70b-versatile"
MODEL_EXTRACTOR3      ="llama-3.1-8b-instant"
MODEL_ANALYST         = "mistral-small-latest"
MODEL_ANALYST2        = "mistral-small-latest"
MODEL_FALLBACK_HEAVY  = "qwen/qwen3-32b"

from config import N_RUNS
OUTPUT_DIR      = "geo_output"
RAW_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "raw_responses.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per-run token accumulator — reset by the node function before each pipeline
# invocation so state from one run never bleeds into another.
TOKEN_USAGE: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def reset_token_usage() -> None:
    """Reset TOKEN_USAGE before a new node execution."""
    TOKEN_USAGE.update({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

def _track_tokens(usage: dict) -> None:
    for key in TOKEN_USAGE:
        TOKEN_USAGE[key] += usage.get(key, 0)

# model params
def _model_params(model: str, role: str = "analyst", provider: str = "groq") -> dict:
    if role == "query":
        base = {"temperature": 0.7, "max_tokens": 8192, "top_p": 0.9}
    elif role == "extractor":
        base = {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0}
    elif role == "analyst":
        base = {"temperature": 0.1, "max_tokens": 8192, "top_p": 0.95}
    else:
        base = {"temperature": 0.2, "max_tokens": 8192, "top_p": 0.9}

    # Groq-specific params — not supported on OpenRouter
    if provider == "groq":
        if "gpt-oss" in model.lower():
            base["reasoning_effort"] = "low"
        if "qwen3" in model.lower():
            base["reasoning_effort"] = "none"

    return base

def _parse_wait_time(err_msg: str) -> float:
    m = re.search(r"try again in (\d+)m([\d.]+)s", err_msg)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in ([\d.]+)s", err_msg)
    if m:
        return float(m.group(1))
    m = re.search(r"try again in (\d+) minute", err_msg)
    if m:
        return int(m.group(1)) * 60
    return 0.0

FATAL_ERRORS = [
    "model not found",
    "invalid api key",
    "authentication",
    "permission denied",
    "does not exist"
]

def _parse_413_agent1(err_msg: str):
    """Extract (limit, requested) from a 413 error. Returns (limit, requested) or None."""
    m = re.search(r"Limit\s+(\d+),\s*Requested\s+(\d+)", err_msg, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def query_llm(model: str | None, prompt: str, system: str = "",
              retries: int = 3, role: str = "analyst") -> dict:
    """
    Single LLM call with retry, rate limit handling, token tracking,
    and automatic model fallback via the dynamic ModelRegistry.

    413 (request too large) errors teach the registry the model's token
    capacity so future large prompts skip too-small models automatically.

    role: query | extractor | analyst
    """
    empty = {"raw_response": "", "completion_tokens": 0,
             "prompt_tokens": 0, "total_tokens": 0}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Estimated tokens = prompt chars/4 + max_completion_tokens for this role.
    # Including completion budget is critical: a 6000 TPM model cannot serve
    # a 56-token prompt if max_tokens=8192 (total = 8248 > 6000).
    _role_max = {"query": 8192, "extractor": 1024, "analyst": 8192}
    prompt_tokens_est = sum(len(m.get("content", "")) for m in messages) // 4
    estimated = prompt_tokens_est + _role_max.get(role, 4096)

    for attempt in range(retries * 4):  # enough room to cycle through models
        try:
            candidate, provider, api_key = _model_registry.select(
                preferred=model, estimated_tokens=estimated
            )
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return empty

        try:
            params   = _model_params(candidate, role=role, provider=provider)
            response = _get_client(provider, api_key).chat.completions.create(
                model    = candidate,
                messages = messages,
                stop     = None,
                **params,
            )
            result = {
                "raw_response":      response.choices[0].message.content.strip(),
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens":     response.usage.prompt_tokens,
                "total_tokens":      response.usage.total_tokens,
                "model_id":          candidate,
                "provider":          provider,
            }
            _track_tokens(result)
            if candidate != model or provider != "groq":
                print(f"[llm] Used: {provider}/{candidate}")
            return result

        except RateLimitError as e:
            err_msg = str(e)
            if "413" in err_msg or "request too large" in err_msg.lower():
                parsed = _parse_413_agent1(err_msg)
                if parsed:
                    _model_registry.mark_too_large(candidate, provider, *parsed)
                else:
                    _model_registry.mark_tpm(candidate, provider, 3600)
                continue
            if ("tokens per day" in err_msg or "per day" in err_msg
                    or "daily" in err_msg.lower()
                    or "402" in err_msg or "requires more credits" in err_msg.lower()
                    or "can only afford" in err_msg.lower()):
                _model_registry.mark_tpd_exhausted(candidate, provider, api_key)
                print(f"[llm] {provider}/{candidate}: credits/daily limit — skipping")
            else:
                wait = _parse_wait_time(err_msg)
                if wait > 0 and wait <= 90:
                    _model_registry.mark_tpm(candidate, provider, wait, api_key)
                    print(f"[RATE LIMIT] {provider}/{candidate}: waiting {wait:.0f}s...")
                    time.sleep(wait + 2)
                else:
                    _model_registry.mark_tpm(candidate, provider, max(wait, 60), api_key)
                    print(f"[RATE LIMIT] {provider}/{candidate}: wait {wait:.0f}s — skipping")

        except Exception as e:
            err_msg = str(e)
            print(f"[WARN] {provider}/{candidate}: {e}")
            if any(msg in err_msg.lower() for msg in FATAL_ERRORS):
                print(f"[FATAL] Non-recoverable error: {err_msg}")
                return empty
            if "413" in err_msg or "request too large" in err_msg.lower():
                parsed = _parse_413_agent1(err_msg)
                if parsed:
                    _model_registry.mark_too_large(candidate, provider, *parsed)
                else:
                    _model_registry.mark_tpm(candidate, provider, 3600, api_key)
                continue
            if ("402" in err_msg or "requires more credits" in err_msg.lower()
                    or "can only afford" in err_msg.lower()):
                print(f"[llm] {provider}/{candidate}: insufficient credits — skipping permanently")
                _model_registry.mark_tpd_exhausted(candidate, provider, api_key)
                continue
            time.sleep(2 ** min(attempt, 4))

    print(f"[ERROR] All models exhausted.")
    return empty


def agent1_load_prompts(prompt_set_path: str) -> list:

    if not os.path.exists(prompt_set_path):
        print(f"[ERROR] Prompt set not found: {prompt_set_path}")
        return []

    df = pd.read_csv(prompt_set_path, encoding='utf-8-sig')

    # verify required columns
    required_cols = ['prompt_id', 'intent_id', 'language', 'prompt_text']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in prompt set: {missing}")
        return []

    prompts = df.to_dict(orient='records')

    print(f"   Step 1 : Batch loader complete")
    print(f"   Prompts loaded : {len(prompts)}")
    print(f"   Intent types   : {df['intent_id'].unique().tolist()}")
    print(f"   Languages      : {df['language'].unique().tolist()}")

    return prompts

PROMPT_SET_PATH = "prompt_set_Tunisian_restaurants.csv"  # default path used by node


def agent1_query_prompts(
    prompts:      list,
    query_models: list | None = None,
    n_runs:       int  = N_RUNS,
    output_path:  str  = RAW_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Step 2: LLM querying — each prompt is sent to every model in query_models,
    repeated n_runs times per model. This cross-model, multi-run design is the
    core of GEO measurement: visibility is only meaningful when measured across
    diverse LLMs, not a single model.

    Resume support: skips (prompt_id, model_id, run_id) triples already saved.
    """
    if query_models is None:
        query_models = QUERY_MODELS

    # Column schema — must match the record dict keys below
    _RAW_COLS = ["response_id","prompt_id","model_slot","model_id","provider",
                 "run_id","response_text","completion_tokens","prompt_tokens",
                 "total_tokens","timestamp"]

    # resume support — key is (prompt_id, slot, run_id)
    done_keys = set()
    file_has_data = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    if file_has_data:
        try:
            df_existing = pd.read_csv(output_path, encoding='utf-8-sig')
            if 'prompt_id' in df_existing.columns:
                for _, row in df_existing.iterrows():
                    done_keys.add((row['prompt_id'], str(row.get('model_slot', row.get('model_id',''))), row['run_id']))
                print(f"Resuming - {len(done_keys)} records already saved")
            else:
                # File exists but has corrupted/missing header — start fresh
                print(f"[WARN] {output_path} has no valid header — rewriting")
                os.remove(output_path)
                file_has_data = False
        except Exception as e:
            print(f"[WARN] Could not read {output_path}: {e} — rewriting")
            os.remove(output_path)
            file_has_data = False

    # Write header row explicitly if file is new/empty
    if not file_has_data:
        pd.DataFrame(columns=_RAW_COLS).to_csv(
            output_path, mode='w', index=False,
            encoding='utf-8-sig', quoting=csv.QUOTE_ALL
        )

    total_prompts  = len(prompts)
    total_calls    = total_prompts * len(query_models) * n_runs
    done_calls     = len(done_keys)
    header_written = True  # always True now — header pre-written above

    slot_names = [s["slot"] if isinstance(s, dict) else s for s in query_models]
    print(f"\nStep 2: LLM querying (multi-model, multi-run)")
    print(f"   Prompts        : {total_prompts}")
    print(f"   Query slots    : {slot_names}")
    print(f"   Runs/slot      : {n_runs}")
    print(f"   Total calls    : {total_calls}")
    print(f"   Already done   : {done_calls}")
    print(f"   Remaining      : {total_calls - done_calls}\n")

    for i, prompt_dict in enumerate(prompts):
        prompt_id   = prompt_dict['prompt_id']
        prompt_text = prompt_dict['prompt_text']
        intent_id   = prompt_dict.get('intent_id', '')
        language    = prompt_dict.get('language', '')

        print(f"\n[{i+1}/{total_prompts}] {prompt_id} | {intent_id} | {language}")
        print(f"  Prompt: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")

        for slot_def in query_models:
            # Support both old string format and new slot dict format
            if isinstance(slot_def, dict):
                slot      = slot_def["slot"]
                groq_id   = slot_def.get("groq", "")
                or_id     = slot_def.get("openrouter", "")
                # Ordered candidate list: Groq first, OpenRouter as fallback
                candidates = [(groq_id, "groq"), (or_id, "openrouter")]
            else:
                slot      = slot_def
                candidates = [(slot_def, None)]   # old behaviour — registry decides

            for run_idx in range(1, n_runs + 1):
                run_id = f"run_{run_idx}"

                if (prompt_id, slot, run_id) in done_keys:
                    print(f"     [{slot}][{run_id}] already done — skipping")
                    continue

                result = None
                used_model = None
                used_provider = None

                for cand_model, cand_provider in candidates:
                    if not cand_model:
                        continue
                    r = query_llm(
                        model  = cand_model,
                        prompt = prompt_text,
                        role   = "query",
                    )
                    returned_model    = r.get("model_id", "")
                    returned_provider = r.get("provider", "")
                    # Accept only if the registry actually used the intended model
                    # (not a fallback to a different model family/slot)
                    if r.get('raw_response') and returned_model == cand_model:
                        result        = r
                        used_model    = returned_model or cand_model
                        used_provider = returned_provider or cand_provider or ""
                        break
                    if r.get('raw_response'):
                        # Got a response but from a wrong slot model — skip
                        print(f"     [{slot}] {cand_provider}/{cand_model} redirected to {returned_provider}/{returned_model} — trying next slot candidate")
                    else:
                        print(f"     [{slot}] {cand_provider}/{cand_model} failed — trying next")

                if not result or not result.get('raw_response'):
                    print(f"     [{slot}][{run_id}] all candidates failed — skipping")
                    continue

                response_id = f"{prompt_id}__{slot}__{run_id}"

                record = {
                    "response_id"      : response_id,
                    "prompt_id"        : prompt_id,
                    "model_slot"       : slot,
                    "model_id"         : used_model,
                    "provider"         : used_provider,
                    "run_id"           : run_id,
                    "response_text"    : result['raw_response'],
                    "completion_tokens": result['completion_tokens'],
                    "prompt_tokens"    : result['prompt_tokens'],
                    "total_tokens"     : result['total_tokens'],
                    "timestamp"        : time.time(),
                }
                pd.DataFrame([record]).to_csv(
                    output_path,
                    mode     = 'a',
                    header   = not header_written,
                    index    = False,
                    encoding = 'utf-8-sig',
                    quoting  = csv.QUOTE_ALL
                )
                header_written = True
                done_keys.add((prompt_id, slot, run_id))
                print(f"     [{slot} → {used_provider}/{used_model}][{run_id}] {result['total_tokens']} tokens")

    df_raw = pd.read_csv(output_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    print(f"\n{'='*55}")
    print(f"Step 2 complete")
    print(f"   Total records  : {len(df_raw)}")
    print(f"   Prompts covered: {df_raw['prompt_id'].nunique()}")
    print(f"   Slots used     : {df_raw['model_slot'].unique().tolist() if 'model_slot' in df_raw.columns else df_raw['model_id'].unique().tolist()}")
    print(f"   Total tokens   : {TOKEN_USAGE['total_tokens']}")
    print(f"   Saved to       : {output_path}")
    print(f"{'='*55}")

    return df_raw



def extract_entity_description(raw_text: str, entity: str) -> str:
    """
    Extract the text block dedicated to this entity from a full LLM response.
    Handles 4 formats: numbered list, bold header, markdown table, free-form prose.
    From notebook — kept as-is.
    """
    if not raw_text or not entity:
        return ""

    entity_lower      = entity.lower()
    lines             = raw_text.split("\n")
    description_lines = []
    inside_entity     = False
    consecutive_empty = 0

    # Format 3: markdown table
    for line in lines:
        if "|" in line and entity_lower in line.lower():
            cells = [c.strip() for c in line.split("|") if c.strip()]
            return " — ".join(cells)

    # Formats 1, 2, 4: list / bold header / prose
    for line in lines:
        if not inside_entity:
            if entity_lower in line.lower():
                inside_entity = True
                description_lines.append(line)
            continue

        stripped = line.strip()

        # stop: new numbered item → next entity
        if re.match(r"^\s*\d+\.\s+", line) and description_lines:
            break

        # stop: bold header that is NOT this entity → next entity
        if re.match(r"^\s*\*\*[^*]+\*\*", line) and description_lines:
            if entity_lower not in line.lower():
                break

        # stop: markdown table row for a different entity
        if "|" in line and entity_lower not in line.lower() and description_lines:
            break

        # stop: two consecutive blank lines
        if stripped == "":
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            description_lines.append(line)
            continue

        consecutive_empty = 0
        description_lines.append(line)

    return "\n".join(description_lines).strip()

MISTRAL_MODEL = "mistral-small-latest"

def query_mistral(prompt: str, system: str = "") -> dict:
    """
    Mistral API call — returns same dict format as query_llm
    so it's a drop-in replacement for extraction tasks.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model"      : MISTRAL_MODEL,
            "messages"   : messages,
            "temperature": 0.0,        # deterministic — critical for JSON output
            "max_tokens" : 1024,
        }
    )

    data = response.json()

    # handle API errors gracefully
    if "choices" not in data:
        print(f"[WARN] Mistral API error: {data}")
        return {"raw_response": "", "completion_tokens": 0,
                "prompt_tokens": 0, "total_tokens": 0}

    return {
        "raw_response"     : data["choices"][0]["message"]["content"].strip(),
        "completion_tokens": data["usage"]["completion_tokens"],
        "prompt_tokens"    : data["usage"]["prompt_tokens"],
        "total_tokens"     : data["usage"]["total_tokens"],
    }

ENTITIES_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "extracted_entities.csv")


BRAND_EXTRACTION_SYSTEM = """\
You are a restaurant name extractor working in a GEO analysis pipeline
that studies TUNISIAN restaurants — establishments serving Tunisian /
North-African cuisine, or restaurants located inside Tunisia.

Your ONLY task is to find every named restaurant, cafe, or food business
in the text. The text may be a numbered list, markdown table, bold headers,
or free-form prose - read ALL formats.

════════════════════════════════════════════════════
DOMAIN SCOPE — CRITICAL
════════════════════════════════════════════════════
Only extract establishments that are plausibly Tunisian:
  + Located in Tunisia (any city)
  + Serving Tunisian / North-African / Maghrebi cuisine
  + Tunisian-diaspora restaurants abroad known for Tunisian food

EXCLUDE any establishment that has NO Tunisian connection:
  - Any internationally known chain or restaurant with no Tunisian /
    North-African cuisine context in the surrounding text
  - If the surrounding text shows no Tunisian / Maghrebi food signal
    for that name → skip it

════════════════════════════════════════════════════
WHAT TO EXTRACT
════════════════════════════════════════════════════
Extract ONLY named establishments - places with a proper name:
  + Named restaurants     "Dar El Jeld", "Le Corsaire", "Chez Ahmed"
  + Named cafes           "Cafe des Nattes", "Le Petit Cafe"
  + Named sandwich shops  "Sandwich Chez Ali", "Le Snack du Port"
  + Named food businesses "Patisserie Mabrouk", "Boulangerie Paris"

════════════════════════════════════════════════════
WHAT TO EXCLUDE
════════════════════════════════════════════════════
Exclude items that are NOT a proper establishment name:
  - Standalone dish names with no business name
    ("couscous", "brik", "mlewi", "harissa", "shakshuka", "falafel")
  - Purely generic descriptions with no proper name
    ("a local restaurant", "the best cafe", "street food stalls")
  - City or region names  ("Tunis", "Sfax", "Djerba", "Tunisia")
  - People names          ("Chef Ahmed", "Mohamed")
  - Digital platforms, apps, or websites — anything that is a service for
    finding or booking restaurants, not a physical food establishment itself

NOTE: "sandwich" or "pizza" as PART of a business name is allowed.
  KEEP  -> "sandwich chez ahmed"
  KEEP  -> "pizza napoli"
  KEEP  -> "Mlewi Lazher"
  SKIP  -> "sandwich"
  SKIP  -> "mlewi"

════════════════════════════════════════════════════
NORMALIZATION - CRITICAL
════════════════════════════════════════════════════
Every name must be byte-for-byte identical across all runs.

  1. Lowercase only
       "Dar El Jeld"       -> "dar el jeld"
       "Le Corsaire"       -> "le corsaire"

  2. Keep geographic/cultural articles
       "Dar", "El", "Le", "La", "Les", "Al", "Chez" -> always keep

  3. Strip generic prefix "Restaurant" if a proper name follows
       "Restaurant Rami"      -> "rami"
       "Restaurant La Medina" -> "la medina"
       "Restaurant El Ksar"   -> "el ksar"
       BUT: if "restaurant" IS the name -> keep as-is
       "Le Restaurant"        -> "le restaurant"

  3b. Strip ONLY the leading word "Restaurant/Cafe/Snack"
      when immediately followed by a proper name or article
       "Restaurant du Port"   -> "du port"
       "Cafe Chez Ali"        -> "chez ali"
       "Snack Le Palmier"     -> "le palmier"

  4. Remove duplicates - return each name ONCE only

  5. Arabic names - keep as-is in Arabic script
       "دار الجلد"  -> "دار الجلد"
       "مطعم رامي"  -> "رامي"

  6. Mixed script - normalize Latin part only
       "Dar El Jeld دار الجلد" -> "dar el jeld"

════════════════════════════════════════════════════
OUTPUT FORMAT - STRICT
════════════════════════════════════════════════════
Return ONLY a flat JSON array of strings.
First character MUST be [   Last character MUST be ]
No objects. No nesting. No prose. No markdown. No explanation.
Return at most 20 restaurant names per response.

Correct:  ["dar el jeld", "le corsaire", "sandwich chez ahmed"]
Wrong:    [{"brand": "dar el jeld"}]
Wrong:    [["dar el jeld"]]

If NO named establishments found: []
Never return null, never return {}, never return an empty string.
The response must ALWAYS be a valid JSON array, even when empty.
"""


def build_brand_extraction_prompt(raw_text: str, language: str = "fr") -> str:
    return (
        f"Language of text: {language}\n"
        "Extract all restaurant names from the text below.\n"
        "Return ONLY a flat JSON array of lowercase strings.\n"
        "Example: [\"dar el jeld\", \"le corsaire\"]\n\n"
        f"---\n{raw_text}\n---"
    )


def clean_and_parse_json(raw: str, label: str = "") -> list:
    """
    Multi-layer JSON parser from notebook.
    Handles markdown fences, nested arrays, trailing commas,
    smart quotes, unescaped newlines, qwen3 thinking blocks.
    """
    if not raw or not raw.strip():
        print(f"[WARN] {label}: empty response.")
        return []

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _unwrap(parsed):
        if (isinstance(parsed, list)
                and len(parsed) == 1
                and isinstance(parsed[0], list)):
            return parsed[0]
        return parsed

    try:
        return _unwrap(json.loads(text))
    except json.JSONDecodeError:
        pass

    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return _unwrap(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass

    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    try:
        return _unwrap(json.loads(fixed))
    except json.JSONDecodeError:
        pass

    try:
        repaired = re.sub(r'(?<!\\)\n', r'\\n', text)
        return _unwrap(json.loads(repaired))
    except json.JSONDecodeError:
        pass

    print(f"[WARN] {label}: could not parse JSON.")
    print(f"[RAW] {raw[:400]}")
    return []


def agent1_extract_entities(
    df_raw:      pd.DataFrame,
    model:       str = MODEL_EXTRACTOR,
    output_path: str = ENTITIES_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Step 3: Entity extraction.
    For each raw response extract entity names using extractor LLM.

    Input  : raw_responses.csv (df_raw)
    Output : extracted_entities.csv
    Schema : entity_id · response_id · prompt_id · run_id · model_id ·
             intent_id · language · entity · brand_raw_text
    """

    # resume support
    _ENTITY_COLS = ["entity_id", "response_id", "entity", "brand_raw_text"]
    done_response_ids = set()
    file_has_data = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    if file_has_data:
        try:
            df_existing = pd.read_csv(output_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            if 'response_id' in df_existing.columns:
                done_response_ids = set(df_existing['response_id'].unique())
                print(f"Resuming - {len(done_response_ids)} responses already extracted")
            else:
                os.remove(output_path)
                file_has_data = False
        except Exception:
            os.remove(output_path)
            file_has_data = False

    if not file_has_data:
        pd.DataFrame(columns=_ENTITY_COLS).to_csv(
            output_path, mode='w', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL
        )

    total = len(df_raw)

    print(f"\nStep 3: Entity extraction")
    print(f"   Responses to process : {total}")
    print(f"   Already done         : {len(done_response_ids)}")
    print(f"   Remaining            : {total - len(done_response_ids)}")
    print(f"   Extractor model      : {model}\n")

    for i, row in df_raw.iterrows():
        response_id = row['response_id']

        if response_id in done_response_ids:
            print(f"  [{i+1}/{total}] {response_id} skipped")
            continue

        print(f"  [{i+1}/{total}] {response_id}")

        extraction_prompt = build_brand_extraction_prompt(
            raw_text=row['response_text'],
            language=row.get('language', 'fr')
        )

        result = query_mistral(
            prompt = extraction_prompt,
            system = BRAND_EXTRACTION_SYSTEM
        )

        raw_entities = clean_and_parse_json(
            result['raw_response'],
            label=f"Entity extraction [{response_id}]"
        )

        if not isinstance(raw_entities, list):
            raw_entities = []

        records = []
        for item in raw_entities:
            if isinstance(item, str):
                entity_name = item.lower().strip()
            elif isinstance(item, dict):
                entity_name = item.get("brand", "").lower().strip()
            else:
                continue

            if not entity_name:
                continue

            records.append({
                "entity_id"     : f"{response_id}__{entity_name.replace(' ', '_')}",
                "response_id"   : response_id,      # FK → raw_responses.csv
                "entity"        : entity_name,
                "brand_raw_text": extract_entity_description(
                                    row['response_text'], entity_name
                                  ),
            })

        print(f"           {len(records)} entities: {[r['entity'] for r in records]}")

        if records:
            pd.DataFrame(records).to_csv(
                output_path,
                mode='a',
                header=False,
                index=False,
                encoding='utf-8-sig',
                quoting=csv.QUOTE_ALL
            )

        done_response_ids.add(response_id)

    # final summary
    df_entities = pd.read_csv(
        output_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL
    )

    print(f"\n{'='*55}")
    print(f"Step 3 complete")
    print(f"   Total entity records : {len(df_entities)}")
    print(f"   Unique entities      : {df_entities['entity'].nunique()}")
    print(f"   Responses processed  : {df_entities['response_id'].nunique()}")
    print(f"   Saved to             : {output_path}")
    print(f"{'='*55}")

    return df_entities



ENRICHED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "enriched_entities.csv")

# SYSTEM PROMPT — ranking only

RANKING_EXTRACTION_SYSTEM = """\
You are a ranking extraction specialist for a GEO analysis pipeline.

Your ONLY task is to read a restaurant recommendation response and return
the ordered list of ALL restaurant entities mentioned, in the exact order
they first appear.

HOW TO RANK
- Numbered list response: use the item numbers directly
- Prose response: use the sentence order of first mention
- Position 1 = first mentioned entity

OUTPUT FORMAT — STRICT
Return ONLY a flat JSON array of strings in ranked order.
First character MUST be [   Last character MUST be ]

Example:
["dar el jeld", "le corsaire", "la kasbah", "dar zarrouk"]

If no entities found: []
"""

# HELPERS

def compute_description_length(entity: str, run_records: list) -> int:
    """
    Estimate average tokens dedicated to this entity across all runs.
    Uses char ratio × completion_tokens as proxy. No API call.
    """
    estimates = []

    for rec in run_records:
        raw_text = rec.get("response_text", "")
        completion_tokens = rec.get("completion_tokens", 0)

        description_block = rec.get("brand_raw_text") or extract_entity_description(raw_text, entity)
        description_block = description_block if isinstance(description_block, str) else ""

        resp_chars = len(raw_text)
        desc_chars = len(description_block)

        if resp_chars > 0 and completion_tokens > 0:
            char_ratio = desc_chars / resp_chars
            estimated_tokens = round(char_ratio * completion_tokens)
        else:
            estimated_tokens = 0

        estimates.append(estimated_tokens)

    return round(sum(estimates) / len(estimates)) if estimates else 0


def extract_rankings_for_response(response_id: str, response_text: str, entities: list, model: str) -> dict:
    """
    Extract ranking positions for ALL entities in one response in ONE LLM call.
    Returns: {entity_name: ranking_position}
    """
    prompt = (
        "Read this restaurant recommendation response and return "
        "ALL restaurant names in the exact order they first appear.\n\n"
        "Known entities to look for (but include any others too):\n"
        f"{json.dumps(entities, ensure_ascii=False)}\n\n"
        f"Response:\n---\n{response_text}\n---\n\n"
        "Return ONLY a flat JSON array of entity names in ranked order."
    )

    result = query_llm(
        model=model,
        prompt=prompt,
        system=RANKING_EXTRACTION_SYSTEM,
        role="extractor"
    )

    ranked_list = clean_and_parse_json(
        result["raw_response"],
        label=f"Ranking [{response_id}]"
    )

    if not isinstance(ranked_list, list):
        return {}

    return {
        entity.lower().strip(): idx + 1
        for idx, entity in enumerate(ranked_list)
        if isinstance(entity, str)
    }


# MAIN FUNCTION

def agent1_enrich_entities(
    df_entities: pd.DataFrame,
    df_raw: pd.DataFrame,
    model: str = MODEL_EXTRACTOR,
    output_path: str = ENRICHED_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Step 4 — Entity enrichment.

    Phase A : ranking extraction — 1 LLM call per response
    Phase B : description length — pure Python, no LLM

    Input  : extracted_entities.csv + raw_responses.csv
    Output : enriched_entities.csv
    Schema : response_id · prompt_id · entity · run_id ·
             ranking_position · description_length_tokens
    """

    # join entities with raw responses
    df_joined = df_entities.merge(
        df_raw[["response_id", "prompt_id", "response_text",
                "run_id", "model_id", "completion_tokens"]],
        on="response_id",
        how="left"
    )

    # resume support
    _ENRICH_COLS = ["response_id", "prompt_id", "entity", "run_id",
                    "ranking_position", "description_length_tokens"]
    done_response_ids = set()
    file_has_data = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    if file_has_data:
        try:
            df_existing = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
            if "response_id" in df_existing.columns:
                done_response_ids = set(df_existing["response_id"].unique())
                print(f"Resuming — {len(done_response_ids)} responses already done")
            else:
                os.remove(output_path)
                file_has_data = False
        except Exception:
            os.remove(output_path)
            file_has_data = False

    if not file_has_data:
        pd.DataFrame(columns=_ENRICH_COLS).to_csv(
            output_path, mode="w", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
        )

    # PHASE A — ranking extraction
    print("\nPhase A — Ranking extraction per response")

    ranking_map = {}
    responses = df_joined[["response_id", "response_text"]].drop_duplicates()
    total_resp = len(responses)

    for idx, (_, row) in enumerate(responses.iterrows()):
        response_id = row["response_id"]
        response_text = row["response_text"]

        entities_in_response = df_joined[
            df_joined["response_id"] == response_id
        ]["entity"].tolist()

        print(f"[{idx+1}/{total_resp}] {response_id} — {len(entities_in_response)} entities")

        ranking_map[response_id] = extract_rankings_for_response(
            response_id=response_id,
            response_text=response_text,
            entities=entities_in_response,
            model=model,
        )

        print(f" -> {ranking_map[response_id]}")

    # PHASE B — description length
    print("\nPhase B — Description length computation")

    groups = list(df_joined.groupby(["prompt_id", "entity"]))
    desc_map = {}

    for (prompt_id, entity), group in groups:
        run_records = group.to_dict(orient="records")
        desc_map[(prompt_id, entity)] = compute_description_length(
            entity=entity,
            run_records=run_records,
        )

    print(f"Computed {len(groups)} (prompt × entity) pairs")

    # BUILD FINAL RECORDS
    print("\nBuilding final records")

    records = []

    for _, row in df_joined.iterrows():
        response_id = row["response_id"]

        if response_id in done_response_ids:
            continue

        prompt_id = row["prompt_id"]
        entity = row["entity"]
        run_id = row["run_id"]

        records.append({
            "response_id": response_id,
            "prompt_id": prompt_id,
            "entity": entity,
            "run_id": run_id,
            "ranking_position": ranking_map.get(response_id, {}).get(entity),
            "description_length_tokens": desc_map.get((prompt_id, entity), 0),
        })

    if records:
        pd.DataFrame(records).to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL
        )

    # final summary
    df_enriched = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    print("\nStep 4 complete")
    print(f"Total records: {len(df_enriched)}")
    print(f"Unique entities: {df_enriched['entity'].nunique()}")
    print(f"Prompts covered: {df_enriched['prompt_id'].nunique()}")
    print(f"Saved to: {output_path}")

    return df_enriched


from rapidfuzz import fuzz, process

CLEAN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clean_entities.csv")
CLEAN_LOG_PATH    = os.path.join(OUTPUT_DIR, "cleaning_log.csv")

FUZZY_MERGE_THRESHOLD = 88
FUZZY_AMBIG_LOWER     = 75
FUZZY_AMBIG_UPPER     = 87
MIN_ENTITY_LENGTH     = 3
NULL_RANK_PENALTY     = 999
MIN_MENTION_COUNT     = 2          # drop singletons — noise, not signal


def normalize_name(e: str) -> str:
    """
    Canonical normalization for entity comparison:
    strip → lower → NFD decompose → strip combining marks (accents).
    'Café des Nattes' and 'cafe des nattes' both → 'cafe des nattes'.
    """
    e = str(e).strip().lower()
    e = unicodedata.normalize("NFD", e)
    e = "".join(c for c in e if unicodedata.category(c) != "Mn")
    return e


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A — Hard rules only
# ══════════════════════════════════════════════════════════════════════════════

def phase_a_prefilter(df: pd.DataFrame, entity_counts: dict) -> tuple:
    """
    Hard drop — empty, numeric, too short, non-Tunisian, generic names,
    and singletons (mention count < MIN_MENTION_COUNT).
    entity_counts keys must already be normalized via normalize_name().
    """
    log = []
    df  = df.copy()
    df['quality_flag'] = 'ok'

    def hard_invalid(entity: str) -> str:
        raw = str(entity).strip()
        if not raw:
            return "empty"
        e = normalize_name(raw)
        if len(e) < MIN_ENTITY_LENGTH:
            return "too_short"
        if e.isdigit():
            return "numeric"
        if entity_counts.get(e, 0) < MIN_MENTION_COUNT:
            return "singleton"
        return None

    drop_mask = pd.Series([False] * len(df), index=df.index)

    for idx, row in df.iterrows():
        reason = hard_invalid(str(row.get('entity', '')))
        if reason:
            drop_mask[idx] = True
            log.append({
                "entity": row['entity'],
                "action": "dropped",
                "reason": reason,
                "phase" : "A",
            })

    dropped = df[drop_mask]
    df      = df[~drop_mask].reset_index(drop=True)

    print(f"   Dropped (hard) : {len(dropped)}")
    print(f"   Kept           : {len(df)}")

    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B — Fuzzy clustering with frequency-based canonical
# ══════════════════════════════════════════════════════════════════════════════

def phase_b_fuzzy_cluster(
    entities:      list,
    entity_counts: dict,
) -> tuple:
    """
    Fuzzy deduplication using token_sort_ratio.
    Canonical = most frequent entity in cluster.
    """
    entities    = sorted(set(normalize_name(e) for e in entities if e))
    assigned    = {}
    clusters    = {}
    pair_scores = {}

    for entity in entities:
        if entity in assigned:
            continue

        candidates = process.extract(
            entity,
            [e for e in entities if e not in assigned],
            scorer       = fuzz.token_sort_ratio,
            limit        = None,
            score_cutoff = FUZZY_AMBIG_LOWER,
        )

        group = {entity}
        for match, score, _ in candidates:
            if match == entity:
                continue
            pair_scores[(entity, match)] = score
            if score >= FUZZY_MERGE_THRESHOLD:
                group.add(match)

        canonical = max(group, key=lambda e: entity_counts.get(e, 0))
        clusters[canonical] = list(group)
        for member in group:
            assigned[member] = canonical

    for entity in entities:
        if entity not in assigned:
            assigned[entity] = entity
            clusters[entity] = [entity]

    return clusters, pair_scores, assigned


# ══════════════════════════════════════════════════════════════════════════════
# PHASE C — LLM: pair resolution + entity validation using brand context
# ══════════════════════════════════════════════════════════════════════════════

ARBITRATION_SYSTEM = """\
You are an entity resolution and validation specialist for a restaurant
GEO analysis pipeline. The domain is TUNISIAN restaurants — physical
establishments serving Tunisian / North-African cuisine, OR restaurants
physically located in Tunisia.

You have TWO tasks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 1 — PAIR RESOLUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decide if each pair refers to the SAME physical place or DIFFERENT places.
Default: DIFFERENT. Flip to SAME only for pure surface variants.

SAME — only for:
  • Accent/spelling variant of identical name
      cafe des nattes / café des nattes         -> SAME
  • Article swap with no new word
      el kasbah / la kasbah / kasbah            -> SAME
      le corsaire / corsaire                    -> SAME
  • Generic prefix that adds nothing
      restaurant dar zarrouk / dar zarrouk      -> SAME

DIFFERENT — always when:
  • Any substantive word differs (even one letter in a proper noun)
      dar el jedid / dar el jeld                -> DIFFERENT (distinct places)
  • Different city/neighbourhood qualifier
  • Any genuine doubt                           -> DIFFERENT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 2 — ENTITY VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each entity decide: is it a real, SPECIFIC, identifiable Tunisian or
North-African restaurant / café / food establishment?

REJECT (valid = false) when ANY of these apply:

  R1. NOT A SPECIFIC PLACE — the name is a generic descriptor that could
      refer to dozens of different restaurants and cannot identify one
      physical establishment. Reject aggressively:
        "la medina", "le jardin", "la kasbah", "la terrasse", "le patio",
        "le palais", "la cour", "le bistro", "le grill", "le snack",
        "restaurant tunisien", "cuisine tunisienne", "le restaurant" → REJECT

  R2. WRONG GEOGRAPHY — the entity is a well-known restaurant from another
      country with no Tunisian / North-African connection in the context.
      Examples that MUST be rejected:
        "le grand vefour" (Paris), "tour d argent" (Paris),
        "noma" (Copenhagen), "el bulli" (Spain), "nobu" (global chain)
      Apply this rule broadly — if the name is internationally famous
      outside Tunisia and the context has no Tunisian signal → REJECT.

  R3. NOT A FOOD BUSINESS — digital platform, booking app, monument,
      museum, mosque, city name alone, mathematician, historical figure.

  R4. NO SIGNAL — context provides no Tunisian / Maghrebi / North-African
      food signal AND the name alone does not evoke Tunisian culture → REJECT.

ACCEPT (valid = true) when:
  A1. Context explicitly mentions Tunisian / North-African food culture:
      couscous, brik, harissa, lablabi, merguez, malouf, tajine, fricassé,
      Tunisian, Maghrebi, Carthage, medina of Tunis, Djerba, Sfax, Sousse …
  A2. Name clearly evokes Tunisian culture even with thin context:
      "dar zarrouk", "chez slah", "le sfax", "le djerba", "el ali",
      "dar el jeld", "dar ben gacem", "la mamma tunisienne" …
  A3. Name contains Tunisian proper nouns / Arabic restaurant naming
      patterns AND context does not contradict Tunisian origin.

BIAS: When uncertain, REJECT. A false negative (missed restaurant) is
      preferable to a false positive (non-Tunisian entity in the output).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — strict JSON, no prose, no markdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "pair_resolutions": [
    {
      "pair"      : ["name_a", "name_b"],
      "decision"  : "SAME" | "DIFFERENT",
      "canonical" : "chosen_name" | null,
      "confidence": "high" | "low"
    }
  ],
  "entity_validations": [
    {
      "entity": "entity_name",
      "valid" : true | false,
      "reason": "R1|R2|R3|R4|A1|A2|A3 — one sentence"
    }
  ]
}

If no pairs: "pair_resolutions": []
If no entities: "entity_validations": []
"""


def phase_c_llm_arbitration(
    ambiguous_pairs: list,
    all_entities:    list,
    entity_context:  dict,
    entity_counts:   dict,
    model:           str = MODEL_ANALYST,
    batch_size:      int = 20,
) -> tuple:
    """
    Phase C — LLM arbitration and entity validation.
    Sends brand_raw_text context so LLM judges semantic validity accurately.
    """
    merge_decisions = {}
    invalid_set     = set()
    flag_set        = set()
    log             = []

    entities_with_context = [
        {
            "entity" : e,
            "context": entity_context.get(e, "")[:400],
        }
        for e in all_entities
    ]

    pair_batches   = [ambiguous_pairs[i:i+batch_size]
                      for i in range(0, max(len(ambiguous_pairs), 1), batch_size)]
    entity_batches = [entities_with_context[i:i+batch_size]
                      for i in range(0, len(entities_with_context), batch_size)]

    while len(pair_batches)   < len(entity_batches):
        pair_batches.append([])
    while len(entity_batches) < len(pair_batches):
        entity_batches.append([])

    print(f"   Ambiguous pairs : {len(ambiguous_pairs)}")
    print(f"   Entities        : {len(all_entities)}")
    print(f"   Batches         : {len(pair_batches)}")

    for b_idx, (pair_batch, entity_batch) in enumerate(
        zip(pair_batches, entity_batches)
    ):
        print(f"   Batch {b_idx+1}/{len(pair_batches)} "
              f"— {len(pair_batch)} pairs, {len(entity_batch)} entities")

        prompt = (
            f"TASK 1 — Resolve {len(pair_batch)} restaurant name pairs.\n"
            f"TASK 2 — Validate {len(entity_batch)} entities using context.\n\n"
            + (
                "Pairs:\n"
                + json.dumps(
                    [{"name_a": a, "name_b": b, "fuzzy_score": round(s, 1)}
                     for a, b, s in pair_batch],
                    ensure_ascii=False, indent=2
                ) + "\n\n"
                if pair_batch else
                "Pairs: []\n\n"
            )
            + "Entities:\n"
            + json.dumps(entity_batch, ensure_ascii=False, indent=2)
            + "\n\nReturn ONLY the JSON object."
        )

        result = query_llm(
            model  = model,
            prompt = prompt,
            system = ARBITRATION_SYSTEM,
            role   = "analyst"
        )

        parsed = clean_and_parse_json(
            result['raw_response'],
            label=f"Phase C batch {b_idx+1}"
        )

        if not isinstance(parsed, dict):
            print(f"   [WARN] Batch {b_idx+1} parse failed — skipping")
            for a, b, _ in pair_batch:
                flag_set.add(a)
                flag_set.add(b)
            continue

        for item in parsed.get('pair_resolutions', []):
            pair       = item.get('pair', [])
            decision   = item.get('decision', 'DIFFERENT')
            chosen     = item.get('canonical')
            confidence = item.get('confidence', 'low')

            if len(pair) != 2:
                continue
            a, b = pair[0], pair[1]

            if decision == 'SAME' and chosen and confidence == 'high':
                other = b if chosen == a else a
                merge_decisions[other] = chosen
                log.append({
                    "entity": other,
                    "action": "merged_by_llm",
                    "reason": f"LLM -> {chosen} (conf=high)",
                    "phase" : "C",
                })
            elif confidence == 'low':
                flag_set.add(a)
                flag_set.add(b)
                log.append({
                    "entity": f"{a} | {b}",
                    "action": "flagged",
                    "reason": "LLM low confidence",
                    "phase" : "C",
                })

        for item in parsed.get('entity_validations', []):
            entity = item.get('entity', '').strip().lower()
            valid  = item.get('valid', True)
            reason = item.get('reason', '')

            if not valid and entity:
                invalid_set.add(entity)
                log.append({
                    "entity": entity,
                    "action": "invalidated",
                    "reason": reason,
                    "phase" : "C",
                })

        time.sleep(0.3)

    print(f"   LLM merged    : {len(merge_decisions)}")
    print(f"   Invalidated   : {len(invalid_set)}")
    print(f"   Flagged       : {len(flag_set)}")

    return merge_decisions, invalid_set, flag_set, log


# ══════════════════════════════════════════════════════════════════════════════
# PHASE D — Self-reflection on all cleaning decisions
# ══════════════════════════════════════════════════════════════════════════════

REFLECTION_SYSTEM = """\
You are a quality assurance specialist reviewing data cleaning decisions
for a Tunisian restaurant GEO analysis pipeline.

Review the cleaning log and current canonical mappings.
Identify any suspicious or incorrect decisions.

Be critical — a wrong merge is worse than a missed merge.
A wrongly invalidated restaurant loses real data.

IMPORTANT — restaurants named after cities like "le sfax", "le djerba",
"le bardo", "le marrakech" are valid Paris Tunisian restaurants.
Flag any invalidation of these as suspicious.

Return ONLY valid JSON:
{
  "quality_score": <0-10>,
  "proceed"      : <true | false>,
  "suspicious"   : [
    {
      "entity" : <string>,
      "issue"  : <what looks wrong>,
      "action" : "undo_merge" | "undo_flag" | "undo_invalidation" | "review"
    }
  ]
}
"""


def phase_d_self_reflect(
    cleaning_log:  list,
    canonical_map: dict,
    invalid_set:   set,
    model:         str = MODEL_ANALYST,
) -> dict:
    """
    Agent reflects on its own cleaning decisions.
    Returns correction instructions.
    """
    if not cleaning_log:
        print("   No decisions to reflect on")
        return {"quality_score": 10, "proceed": True, "suspicious": []}

    prompt = (
        f"Review these {len(cleaning_log)} data cleaning decisions "
        f"for Tunisian restaurant entities.\n\n"
        f"Cleaning log:\n"
        + json.dumps(cleaning_log, ensure_ascii=False, indent=2)
        + f"\n\nCurrent merges:\n"
        + json.dumps(
            {k: v for k, v in canonical_map.items() if k != v},
            ensure_ascii=False, indent=2
          )
        + f"\n\nCurrently invalidated:\n"
        + json.dumps(list(invalid_set), ensure_ascii=False, indent=2)
        + "\n\nIdentify suspicious or incorrect decisions."
    )

    result = query_llm(
        model  = model,
        prompt = prompt,
        system = REFLECTION_SYSTEM,
        role   = "analyst"
    )

    reflection = clean_and_parse_json(
        result['raw_response'],
        label="Phase D reflection"
    )

    if not isinstance(reflection, dict):
        print("   Parse failed — proceeding without correction")
        return {"quality_score": 5, "proceed": True, "suspicious": []}

    print(f"   Quality score : {reflection.get('quality_score', '?')}/10")
    print(f"   Proceed       : {reflection.get('proceed', True)}")

    suspicious = reflection.get('suspicious', [])
    if suspicious:
        print(f"   Suspicious    : {len(suspicious)}")
        for s in suspicious:
            print(f"   -> '{s.get('entity')}': {s.get('issue')}")

    return reflection


# ══════════════════════════════════════════════════════════════════════════════
# PHASE E — Apply canonical mapping + null rank penalty
# ══════════════════════════════════════════════════════════════════════════════

def phase_e_apply_mapping(
    df:            pd.DataFrame,
    canonical_map: dict,
    invalid_set:   set,
    flag_set:      set,
) -> pd.DataFrame:
    """
    Apply all decisions from Phases A-D.
    """
    df = df.copy()

    df['canonical_entity'] = df['entity'].map(
        lambda e: canonical_map.get(e, e)
    )

    df['ranking_position_filled'] = df['ranking_position'].fillna(
        NULL_RANK_PENALTY
    ).astype(int)

    df['quality_flag'] = df['entity'].apply(
        lambda e: 'invalid' if e in invalid_set else 'ok'
    )

    df['clean_flag'] = df['entity'].apply(
        lambda e: 'invalid' if e in invalid_set
             else 'flagged' if e in flag_set
             else 'clean'
    )

    df['merge_source'] = df.apply(
        lambda row: row['entity']
        if row['entity'] != row['canonical_entity'] else "",
        axis=1
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def agent1_clean_entities(
    df_enriched: pd.DataFrame,
    df_raw:      pd.DataFrame,
    df_entities: pd.DataFrame,
    model:       str = MODEL_ANALYST,
    output_path: str = CLEAN_OUTPUT_PATH,
    log_path:    str = CLEAN_LOG_PATH,
) -> pd.DataFrame:
    """
    Step 5 — Agentic data cleaning pipeline.

    Phase A : Hard rules only — empty, numeric, too short    (Python)
    Phase B : Fuzzy clustering, frequency-based canonical    (rapidfuzz)
    Phase C : LLM pair resolution + entity validation        (LLM)
    Phase D : Self-reflection + correction                   (LLM)
    Phase E : Apply all decisions                            (Python)

    Input  : enriched_entities.csv + raw_responses.csv +
             extracted_entities.csv (for brand context)
    Output : clean_entities.csv + cleaning_log.csv
    Schema : all enriched columns + canonical_entity +
             quality_flag + clean_flag +
             ranking_position_filled + merge_source
    """
    print(f"\n{'='*55}")
    print(f"Step 5 — Agentic data cleaning")
    print(f"   Input rows      : {len(df_enriched)}")
    print(f"   Unique entities : {df_enriched['entity'].nunique()}")
    print(f"   Model           : {model}")
    print(f"{'='*55}\n")

    cleaning_log  = []
    # Normalize keys so accent variants are counted together
    entity_counts: dict = {}
    for raw_entity in df_enriched['entity'].dropna():
        key = normalize_name(str(raw_entity))
        entity_counts[key] = entity_counts.get(key, 0) + 1

    # build entity context from brand_raw_text (extracted entities)
    # brand_raw_text is the specific description block for this entity
    # much more accurate than full response_text for validation
    print("Building entity context from brand descriptions...")
    entity_context = {}

    for _, row in df_enriched.iterrows():
        entity = row['entity']
        if entity not in entity_context:
            # try brand_raw_text from extracted entities first
            match = df_entities[
                (df_entities['response_id'] == row['response_id']) &
                (df_entities['entity'] == entity)
            ]
            if not match.empty:
                brand_text = match.iloc[0].get('brand_raw_text', '')
                if pd.notna(brand_text) and str(brand_text).strip():
                    entity_context[entity] = str(brand_text)[:400]
                    continue

            # fallback — use response_text skipping first 100 chars (intro)
            raw_match = df_raw[df_raw['response_id'] == row['response_id']]
            if not raw_match.empty:
                entity_context[entity] = str(
                    raw_match.iloc[0]['response_text']
                )[100:400]

    print(f"   Context built for {len(entity_context)} entities\n")

    # Phase A
    print("Phase A — Hard pre-filter")
    df_filtered, log_a = phase_a_prefilter(df_enriched, entity_counts)
    cleaning_log.extend(log_a)

    # Phase B
    print("\nPhase B — Fuzzy clustering")
    all_entities = list({
        normalize_name(e)
        for e in df_filtered['entity'].dropna()
        if normalize_name(str(e))
    })

    clusters, pair_scores, canonical_map = phase_b_fuzzy_cluster(
        entities      = all_entities,
        entity_counts = entity_counts,
    )

    auto_merged = 0
    for canonical, members in clusters.items():
        for m in members:
            if m != canonical:
                auto_merged += 1
                cleaning_log.append({
                    "entity": m,
                    "action": "merged",
                    "reason": f"fuzzy >= {FUZZY_MERGE_THRESHOLD} -> {canonical}",
                    "phase" : "B",
                })

    ambiguous_pairs = [
        (a, b, score)
        for (a, b), score in pair_scores.items()
        if FUZZY_AMBIG_LOWER <= score <= FUZZY_AMBIG_UPPER
    ]

    print(f"   Unique entities : {len(all_entities)}")
    print(f"   Auto-merged     : {auto_merged}")
    print(f"   Ambiguous pairs : {len(ambiguous_pairs)}")

    # Phase C
    print("\nPhase C — LLM arbitration + entity validation")
    merge_decisions, invalid_set, flag_set, log_c = phase_c_llm_arbitration(
        ambiguous_pairs = ambiguous_pairs,
        all_entities    = all_entities,
        entity_context  = entity_context,
        entity_counts   = entity_counts,
        model           = model,
    )
    cleaning_log.extend(log_c)
    canonical_map.update(merge_decisions)

    # Phase D
    print("\nPhase D — Self-reflection")
    reflection = phase_d_self_reflect(
        cleaning_log  = cleaning_log,
        canonical_map = canonical_map,
        invalid_set   = invalid_set,
        model         = model,
    )

    for correction in reflection.get('suspicious', []):
        entity = correction.get('entity', '').strip().lower()
        action = correction.get('action', '')

        if action == 'undo_merge' and entity in canonical_map:
            old_canonical = canonical_map[entity]
            canonical_map[entity] = entity
            cleaning_log.append({
                "entity": entity,
                "action": "merge_undone",
                "reason": correction.get('issue', ''),
                "phase" : "D",
            })
            print(f"   Merge undone : '{entity}' <- '{old_canonical}'")

        elif action == 'undo_flag' and entity in flag_set:
            flag_set.discard(entity)
            cleaning_log.append({
                "entity": entity,
                "action": "flag_removed",
                "reason": correction.get('issue', ''),
                "phase" : "D",
            })
            print(f"   Flag removed : '{entity}'")

        elif action == 'undo_invalidation' and entity in invalid_set:
            invalid_set.discard(entity)
            cleaning_log.append({
                "entity": entity,
                "action": "invalidation_undone",
                "reason": correction.get('issue', ''),
                "phase" : "D",
            })
            print(f"   Invalidation undone : '{entity}'")

    # Phase E
    print("\nPhase E — Apply canonical mapping")
    df_clean = phase_e_apply_mapping(
        df            = df_filtered,
        canonical_map = canonical_map,
        invalid_set   = invalid_set,
        flag_set      = flag_set,
    )

    # Save
    df_clean.to_csv(
        output_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL
    )
    pd.DataFrame(cleaning_log).to_csv(
        log_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL
    )

    # Summary
    merged_rows  = (df_clean['merge_source'] != "").sum()
    clean_rows   = df_clean[df_clean['clean_flag'] == 'clean']
    flagged_rows = df_clean[df_clean['clean_flag'] == 'flagged']
    invalid_rows = df_clean[df_clean['clean_flag'] == 'invalid']
    null_penalty = (df_clean['ranking_position_filled'] == NULL_RANK_PENALTY).sum()

    print(f"\n{'='*55}")
    print(f"Step 5 complete")
    print(f"   Input rows          : {len(df_enriched)}")
    print(f"   After Phase A       : {len(df_filtered)}")
    print(f"   Output rows         : {len(df_clean)}")
    print(f"   Canonical entities  : {df_clean['canonical_entity'].nunique()}")
    print(f"   Clean entities      : {clean_rows['canonical_entity'].nunique()}")
    print(f"   Flagged entities    : {flagged_rows['canonical_entity'].nunique()}")
    print(f"   Invalid entities    : {invalid_rows['canonical_entity'].nunique()}")
    print(f"   Rows with merge     : {merged_rows}")
    print(f"   Null rank->penalty  : {null_penalty}")
    print(f"   Saved to            : {output_path}")
    print(f"   Log saved to        : {log_path}")
    print(f"{'='*55}")

    return df_clean



import math

FEATURES_PATH        = os.path.join(OUTPUT_DIR, "entity_features.csv")
FEATURES_GLOBAL_PATH = os.path.join(OUTPUT_DIR, "entity_features_global.csv")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — co-mention rate
# ══════════════════════════════════════════════════════════════════════════════

def compute_co_mention_rate(df: pd.DataFrame) -> dict:
    """
    For each entity, compute how often it appears
    alongside each other entity in the same response.
    Returns: {entity: {co_entity: rate}}
    """
    co_mention_map = {}

    responses = df.groupby('response_id')['canonical_entity'].apply(list).to_dict()

    for response_id, entities in responses.items():
        entities = list(set(entities))
        for entity in entities:
            if entity not in co_mention_map:
                co_mention_map[entity] = {}
            for other in entities:
                if other == entity:
                    continue
                co_mention_map[entity][other] = \
                    co_mention_map[entity].get(other, 0) + 1

    # normalize by entity mention count
    entity_counts = df.groupby('canonical_entity').size().to_dict()
    co_rates = {}
    for entity, co_counts in co_mention_map.items():
        total = entity_counts.get(entity, 1)
        co_rates[entity] = {
            other: round(count / total, 4)
            for other, count in co_counts.items()
        }

    return co_rates


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — prompt type response (FIXED signature)
# ══════════════════════════════════════════════════════════════════════════════

def compute_prompt_type_response(
    df:      pd.DataFrame,
    df_raw:  pd.DataFrame,
    prompts: list,
) -> dict:
    """
    For each entity, compute mention_rate per intent_id.
    intent_id comes from the original prompt set, joined via prompt_id.
    """
    # build prompt_id → intent_id lookup from Step 1 output
    intent_lookup = {
        str(p['prompt_id']): str(p['intent_id'])
        for p in prompts
        if 'intent_id' in p and 'prompt_id' in p
    }

    df_with_intent = df.copy()
    df_with_intent['intent_id'] = (
        df_with_intent['prompt_id'].astype(str).map(intent_lookup)
    )

    result = {}
    intents = df_with_intent['intent_id'].dropna().unique().tolist()
    total_per_intent = (
        df_with_intent.groupby('intent_id')['response_id']
        .nunique().to_dict()
    )

    for entity, group in df_with_intent.groupby('canonical_entity'):
        intent_rates = {}
        for intent in intents:
            intent_group = group[group['intent_id'] == intent]
            total = total_per_intent.get(intent, 1)
            intent_rates[intent] = round(len(intent_group) / total, 4)
        result[entity] = intent_rates

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PER-PROMPT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_entity_features_per_prompt(
    df:           pd.DataFrame,
    total_models: int,
) -> pd.DataFrame:
    """
    Compute features per (prompt_id x canonical_entity).
    Denominators use actual valid response counts per prompt.
    """
    # responses per prompt from the FILTERED data (not df_raw)
    total_responses_per_prompt = (
        df.groupby('prompt_id')['response_id'].nunique().to_dict()
    )

    rows = []

    for (prompt_id, entity), group in df.groupby(
        ['prompt_id', 'canonical_entity']
    ):
        total_resp  = total_responses_per_prompt.get(prompt_id, 1)
        valid_ranks = group[
            group['ranking_position_filled'] != NULL_RANK_PENALTY
        ]['ranking_position_filled'].tolist()

        mention_count = len(group)
        mention_rate  = round(mention_count / total_resp, 4)
        top1_rate     = round(
            sum(1 for r in valid_ranks if r == 1) / mention_count, 4
        ) if mention_count > 0 else 0.0

        avg_rank = round(
            sum(valid_ranks) / len(valid_ranks), 4
        ) if valid_ranks else None

        rank_variance = round(
            math.sqrt(
                sum((r - (sum(valid_ranks) / len(valid_ranks))) ** 2
                    for r in valid_ranks) / len(valid_ranks)
            ), 4
        ) if len(valid_ranks) >= 2 else 0.0

        stability_score = round(
            mention_rate * (1 / (1 + rank_variance)), 4
        )

        if stability_score >= 0.75:
            consistency_label = "stable"
        elif stability_score >= 0.40:
            consistency_label = "variable"
        else:
            consistency_label = "unstable"

        distinct_models = group['response_id'].apply(
            lambda r: r.split('__')[1] if '__' in str(r) else ''
        ).nunique()
        model_presence_count = distinct_models
        cross_model_rate     = round(model_presence_count / total_models, 4)

        rows.append({
            "prompt_id":                prompt_id,
            "canonical_entity":         entity,
            "mention_count":            mention_count,
            "mention_rate":             mention_rate,
            "average_ranking_position": avg_rank,
            "rank_variance":            rank_variance,
            "top1_rate":                top1_rate,
            "frequency_across_runs":    group['run_id'].nunique(),
            "stability_score":          stability_score,
            "consistency_label":        consistency_label,
            "avg_description_length":   round(
                group['description_length_tokens'].mean(), 1
            ),
            "model_presence_count":     model_presence_count,
            "cross_model_rate":         cross_model_rate,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_entity_features_global(
    df:           pd.DataFrame,
    df_raw:       pd.DataFrame,
    total_models: int,
    prompts:      list,
) -> pd.DataFrame:
    """
    Compute global features per canonical_entity across all prompts and runs.
    This is the main input to the Phase 3 recommendation model.
    """
    # Use actual valid responses, not total raw (which includes truncated)
    total_responses = df['response_id'].nunique()
    co_mention_map  = compute_co_mention_rate(df)
    ptr_map         = compute_prompt_type_response(df, df_raw, prompts)

    rows = []

    for entity, group in df.groupby('canonical_entity'):
        valid_ranks = group[
            group['ranking_position_filled'] != NULL_RANK_PENALTY
        ]['ranking_position_filled'].tolist()

        mention_count = len(group)
        mention_rate  = round(mention_count / total_responses, 4)
        top1_rate     = round(
            sum(1 for r in valid_ranks if r == 1) / mention_count, 4
        ) if mention_count > 0 else 0.0

        avg_rank = round(
            sum(valid_ranks) / len(valid_ranks), 4
        ) if valid_ranks else None

        rank_variance = round(
            math.sqrt(
                sum((r - (sum(valid_ranks) / len(valid_ranks))) ** 2
                    for r in valid_ranks) / len(valid_ranks)
            ), 4
        ) if len(valid_ranks) >= 2 else 0.0

        stability_score = round(
            mention_rate * (1 / (1 + rank_variance)), 4
        )

        if stability_score >= 0.75:
            consistency_label = "stable"
        elif stability_score >= 0.40:
            consistency_label = "variable"
        else:
            consistency_label = "unstable"

        distinct_models = group['response_id'].apply(
            lambda r: r.split('__')[1] if '__' in str(r) else ''
        ).nunique()
        model_presence_count = distinct_models
        cross_model_rate     = round(model_presence_count / total_models, 4)

        # mention_prominence: % of mentions where entity is in top 3
        mention_prominence = round(
            sum(1 for r in valid_ranks if r <= 3) / mention_count, 4
        ) if mention_count > 0 else 0.0

        rows.append({
            "canonical_entity":         entity,
            "mention_count":            mention_count,
            "mention_rate":             mention_rate,
            "average_ranking_position": avg_rank,
            "rank_variance":            rank_variance,
            "top1_rate":                top1_rate,
            "mention_prominence":       mention_prominence,
            "frequency_across_runs":    group['run_id'].nunique(),
            "stability_score":          stability_score,
            "consistency_label":        consistency_label,
            "avg_description_length":   round(
                group['description_length_tokens'].mean(), 1
            ),
            "model_presence_count":     model_presence_count,
            "cross_model_rate":         cross_model_rate,
            "prompt_type_response":     json.dumps(
                ptr_map.get(entity, {}), ensure_ascii=False
            ),
            "co_mention_rate":          json.dumps(
                co_mention_map.get(entity, {}), ensure_ascii=False
            ),
        })

    df_global = pd.DataFrame(rows).sort_values(
        'stability_score', ascending=False
    ).reset_index(drop=True)

    return df_global


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def agent1_compute_metrics(
    df_clean:     pd.DataFrame,
    df_raw:       pd.DataFrame,
    prompts:      list,
    output_path:  str = FEATURES_PATH,
    global_path:  str = FEATURES_GLOBAL_PATH,
) -> tuple:
    """
    Step 6 — Compute all GEO features.
    Pure Python — no LLM calls.

    Input  : clean_entities.csv + raw_responses.csv
    Output : entity_features.csv (per prompt x entity)
             entity_features_global.csv (global — input to Phase 3)
    """
    print(f"\n{'='*55}")
    print(f"Step 6 — Compute metrics")
    print(f"   Input rows      : {len(df_clean)}")
    print(f"   Unique entities : {df_clean['canonical_entity'].nunique()}")
    print(f"{'='*55}\n")

    # ── Filter: keep clean + flagged, drop invalid only ──
    df = df_clean[df_clean['clean_flag'] != 'invalid'].copy()
    print(f"   After filtering invalid : {len(df)} rows "
          f"({df['canonical_entity'].nunique()} entities)")

    if len(df) == 0:
        print("   [ERROR] No valid entities after filtering — aborting")
        return pd.DataFrame(), pd.DataFrame()

    # ── Derive total_models from actual data ──
    total_models = df['response_id'].apply(
        lambda r: r.split('__')[1] if '__' in str(r) else ''
    ).nunique()

    total_responses = df['response_id'].nunique()

    print(f"   Total models (actual)   : {total_models}")
    print(f"   Total responses (valid) : {total_responses}\n")

    # ── Per-prompt features ──
    print("Computing per-prompt features...")
    df_features = compute_entity_features_per_prompt(
        df           = df,
        total_models = total_models,
    )
    df_features.to_csv(
        output_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL,
    )
    print(f"   Saved : {output_path}")

    # ── Global features ──
    print("\nComputing global features...")
    df_global = compute_entity_features_global(
        df           = df,
        df_raw       = df_raw,
        total_models = total_models,
        prompts      = prompts,
    )
    df_global.to_csv(
        global_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL,
    )
    print(f"   Saved : {global_path}")

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"Step 6 complete")
    print(f"   Per-prompt rows    : {len(df_features)}")
    print(f"   Global entities    : {len(df_global)}")
    print(f"   Stable entities    : "
          f"{len(df_global[df_global['consistency_label']=='stable'])}")
    print(f"   Variable entities  : "
          f"{len(df_global[df_global['consistency_label']=='variable'])}")
    print(f"   Unstable entities  : "
          f"{len(df_global[df_global['consistency_label']=='unstable'])}")
    print(f"   Top 5 by GEO Score :")
    cols = ['canonical_entity', 'geo_score', 'mention_rate',
            'mention_prominence', 'stability_score', 'cross_model_rate']
    cols = [c for c in cols if c in df_global.columns]
    print(df_global[cols].head(5).to_string(index=False))
    print(f"\n   ⚠ Running on partial data: {total_responses} valid responses")
    print(f"   Rates normalized to actual valid responses.")
    print(f"   Re-run after backfilling truncated responses.")
    print(f"{'='*55}")

    return df_features, df_global



# ── LangGraph node ────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline_state import PipelineState


def run_agent1_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs Agent 1 steps 1-6 and merges results into state."""
    errors = list(state.get("errors", []))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Reset token counter for this invocation so counts don't accumulate
    # across repeated graph executions.
    reset_token_usage()

    # Prefer prompt_set already in state; fall back to CSV on disk.
    prompt_records = state.get("prompt_set") or []
    if prompt_records:
        prompts = prompt_records
    else:
        prompts = agent1_load_prompts(PROMPT_SET_PATH)

    if not prompts:
        errors.append("agent1: no prompts available — skipping")
        return {**state, "errors": errors, "current_step": "agent1_aborted"}

    try:
        # Step 2
        df_raw = agent1_query_prompts(
            prompts=prompts,
            query_models=QUERY_MODELS,
            n_runs=N_RUNS, output_path=RAW_OUTPUT_PATH,
        )
        # Step 3
        df_entities = agent1_extract_entities(
            df_raw=df_raw, model=MODEL_EXTRACTOR,
            output_path=ENTITIES_OUTPUT_PATH,
        )
        # Step 4
        df_enriched = agent1_enrich_entities(
            df_entities=df_entities, df_raw=df_raw,
            model=MODEL_EXTRACTOR3, output_path=ENRICHED_OUTPUT_PATH,
        )
        # Step 5
        df_clean = agent1_clean_entities(
            df_enriched=df_enriched, df_raw=df_raw, df_entities=df_entities,
            model=MODEL_ANALYST, output_path=CLEAN_OUTPUT_PATH,
            log_path=CLEAN_LOG_PATH,
        )
        # Step 6
        df_features, df_global = agent1_compute_metrics(
            df_clean=df_clean, df_raw=df_raw, prompts=prompts,
            output_path=FEATURES_PATH, global_path=FEATURES_GLOBAL_PATH,
        )
    except Exception as exc:
        errors.append(f"agent1: {exc}")
        return {**state, "errors": errors, "current_step": "agent1_error"}

    # Merge per-run token counts into the shared state accumulator.
    prior = state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    merged_tokens = {k: prior.get(k, 0) + TOKEN_USAGE.get(k, 0) for k in prior}

    return {
        **state,
        "raw_responses":          df_raw.to_dict(orient="records"),
        "extracted_entities":     df_entities.to_dict(orient="records"),
        "clean_entities":         df_clean.to_dict(orient="records"),
        "entity_features":        df_features.to_dict(orient="records"),
        "entity_features_global": df_global.to_dict(orient="records"),
        "token_usage":            merged_tokens,
        "errors":                 errors,
        "current_step":           "agent1",
    }