"""Autonomous GEO Pipeline Orchestrator.

Replaces the hardcoded LangGraph graph with a ReAct agent that decides:
  - Which research steps to run
  - In what order
  - With what parameters
  - Whether to retry or adjust
  - When the goal is complete

The agents (intent discovery, entity extraction, web research) are tools.
The LLM reasons about the goal and current state at every step.
No fixed execution graph. No hardcoded agent sequence.
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_utils import call_llm_json
from pipeline_state import initial_state
from agents.agent0 import run_agent0_node
from agents.agent1 import run_agent1_node
from agents.agent2_react import run_agent2_node
from agents.agent3_merge import run_agent3_node, run_agent3_merge


 
# Shared pipeline state
 

_state: dict = {}


def _init(domain: str, languages: list, n_intents: int,
          n_variants: int, max_retries: int):
    global _state
    _state = initial_state(
        domain=domain,
        languages=languages,
        n_intents=n_intents,
        n_variants=n_variants,
        max_retries=max_retries,
    )


# Tools : each returns a plain-text summary for the orchestrator to reason on

def _tool_run_intent_discovery(
    n_intents: int = 4,
    n_variants: int = 3,
    languages: list | None = None,
) -> str:
    """Discover user intent types and generate realistic search prompts."""
    global _state
    if languages:
        _state["languages"] = languages
    _state["n_intents"] = n_intents
    _state["n_variants"] = n_variants

    result = run_agent0_node(_state)
    _state.update(result)

    n = len(_state.get("prompt_set", []))
    score = _state.get("agent0_quality_score")
    errors = _state.get("errors", [])
    intents = list({p.get("intent_id") for p in _state.get("prompt_set", [])})

    return (
        f"Intent discovery done. "
        f"Prompts: {n}, Intents: {intents}, Quality score: {score}/10. "
        f"Errors: {errors if errors else 'none'}."
    )


def _tool_run_entity_extraction() -> str:
    """Query LLMs with the generated prompts and extract restaurant/business entities."""
    global _state
    if not _state.get("prompt_set"):
        return "ERROR: No prompts in state. Run intent discovery first."

    result = run_agent1_node(_state)
    _state.update(result)

    n_entities = len(_state.get("entity_features_global", []))
    top5 = [e.get("canonical_entity") for e in _state.get("entity_features_global", [])[:5]]
    errors = [e for e in _state.get("errors", []) if "agent1" in e]

    return (
        f"Entity extraction done. "
        f"Entities found: {n_entities}. Top 5: {top5}. "
        f"Errors: {errors if errors else 'none'}."
    )


def _tool_run_web_research(fresh_start: bool = False) -> str:
    """Collect web presence data (Google Maps, TripAdvisor, Wikipedia, Foursquare,
    Wikidata, OpenStreetMap, social media) for extracted entities.

    Params:
        fresh_start (bool): if True, delete the existing Agent 2 checkpoint and
            re-research all entities from scratch. Use this when the checkpoint
            was built with an older schema (missing Wikidata / OSM columns).
    """
    global _state
    if not _state.get("entity_features_global"):
        return "ERROR: No entities in state. Run entity extraction first."

    if fresh_start:
        import os
        checkpoint = os.path.join("agent2_output", "web_features.csv")
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
            print("[orchestrator] fresh_start=True — deleted old Agent 2 checkpoint")

    result = run_agent2_node(_state)
    _state.update(result)

    n_web = len(_state.get("web_features", []))
    n_entities = len(_state.get("entity_features_global", []))
    coverage = round(n_web / n_entities, 2) if n_entities else 0
    errors = [e for e in _state.get("errors", []) if "agent2" in e]

    return (
        f"Web research done. "
        f"Coverage: {n_web}/{n_entities} entities ({coverage*100:.0f}%). "
        f"Errors: {errors if errors else 'none'}."
    )


def _tool_run_merge_and_clean(retry_failed: bool = False) -> str:
    """Clean and merge Agent1 (LLM visibility) and Agent2 (web presence) outputs
    into a single unified feature matrix ready for Phase 4 analysis.

    Handles:
    - CSV corruption recovery (unescaped commas in text fields)
    - Geographic entity filtering (city names that leaked through extraction)
    - Entity name normalisation and deduplication
    - Left-join on normalised entity name (Agent1 is source of truth)
    - Per-row data completeness scoring
    - Quality flags: complete / partial / agent2_failed

    Params:
        retry_failed (bool): if True, re-runs Agent2 only for entities
            that had confidence=0 in the previous run, then re-merges.

    Output: geo_output/unified_features.csv + geo_output/merge_report.json
    """
    global _state

    if retry_failed:
        # Identify failed entities from last merge report
        import json, os
        report_path = "geo_output/merge_report.json"
        if os.path.exists(report_path):
            with open(report_path) as f:
                prev = json.load(f)
            failed = prev.get("failed_entities", [])
            if failed:
                # Override entity list in state to only research failed ones
                all_entities = _state.get("entity_features_global", [])
                failed_set = set(e.lower().strip() for e in failed)
                _state["_retry_entities"] = [
                    e for e in all_entities
                    if e.get("canonical_entity", "").lower().strip() in failed_set
                ]
                result = run_agent2_node(_state)
                _state.update(result)
                _state.pop("_retry_entities", None)

    report = run_agent3_merge()

    if "error" in report:
        return f"ERROR: {report['error']}"

    _state["merge_report"] = report

    return (
        f"Merge complete. Unified matrix: {report['unified_rows']} entities. "
        f"Quality — complete: {report['quality_complete']}, "
        f"partial: {report['quality_partial']}, "
        f"agent2_failed: {report['quality_agent2_failed']}. "
        f"Coverage — Google Maps: {report['gm_coverage_pct']}%, "
        f"TripAdvisor: {report['ta_coverage_pct']}%, "
        f"Wikipedia: {report['wikipedia_coverage_pct']}%. "
        f"Failed entities needing retry: {report['failed_entities'] or 'none'}."
    )


def _tool_get_status() -> str:
    """Return a summary of the current pipeline state."""
    prompt_set = _state.get("prompt_set", [])
    entities   = _state.get("entity_features_global", [])
    web        = _state.get("web_features", [])

    entity_names = [e.get("canonical_entity") for e in entities[:8]]
    web_coverage = round(len(web) / len(entities), 2) if entities else 0

    status = {
        "domain":                _state.get("domain"),
        "languages":             _state.get("languages"),
        "prompts_generated":     len(prompt_set),
        "intent_types":          list({p.get("intent_id") for p in prompt_set}),
        "entities_found":        len(entities),
        "top_entities":          entity_names,
        "web_coverage":          f"{len(web)}/{len(entities)} ({web_coverage*100:.0f}%)",
        "errors":                _state.get("errors", []),
        "token_usage":           _state.get("token_usage", {}),
    }
    return json.dumps(status, indent=2, ensure_ascii=False)


def _tool_adjust_parameters(n_intents: int | None = None,
                             n_variants: int | None = None,
                             languages: list | None = None) -> str:
    """Adjust pipeline parameters before re-running a step."""
    global _state
    changed = []
    if n_intents is not None:
        _state["n_intents"] = n_intents
        changed.append(f"n_intents={n_intents}")
    if n_variants is not None:
        _state["n_variants"] = n_variants
        changed.append(f"n_variants={n_variants}")
    if languages is not None:
        _state["languages"] = languages
        changed.append(f"languages={languages}")
    return f"Parameters updated: {', '.join(changed) if changed else 'nothing changed'}."


def _tool_finish(reason: str) -> str:
    """Signal that the research goal is complete."""
    return f"PIPELINE COMPLETE: {reason}"


 
# Tool registry : the orchestrator chooses from this
 

TOOLS = {
    "run_intent_discovery": {
        "fn": _tool_run_intent_discovery,
        "description": (
            "Discover user intent types and generate realistic search prompts for the domain. "
            "Params: n_intents (int, default 4), n_variants (int, default 3), "
            "languages (list of codes, e.g. ['fr','ar'], default from initial config)."
        ),
    },
    "run_entity_extraction": {
        "fn": _tool_run_entity_extraction,
        "description": (
            "Query LLMs with generated prompts and extract restaurant/business entities "
            "mentioned in responses. Requires intent discovery to have run first. No params."
        ),
    },
    "run_web_research": {
        "fn": _tool_run_web_research,
        "description": (
            "Collect web presence data (Google Maps, TripAdvisor, Foursquare, Wikidata, "
            "OpenStreetMap, Wikipedia, social media) for extracted entities. "
            "Requires entity extraction to have run first. "
            "Pass fresh_start=True to clear the old checkpoint and re-research all "
            "entities from scratch (needed when the schema changed)."
        ),
    },
    "run_merge_and_clean": {
        "fn": _tool_run_merge_and_clean,
        "description": (
            "Clean and merge Agent1 + Agent2 outputs into a unified feature matrix. "
            "Recovers from CSV corruption, filters spurious entities, normalises names, "
            "left-joins on entity, scores data completeness per row. "
            "Run this after web research completes. "
            "Param: retry_failed (bool, default False) — set True to re-run Agent2 "
            "only for entities that previously had 0 confidence before merging."
        ),
    },
    "get_status": {
        "fn": _tool_get_status,
        "description": "Get a summary of what has been collected so far. No params.",
    },
    "adjust_parameters": {
        "fn": _tool_adjust_parameters,
        "description": (
            "Adjust pipeline parameters before re-running a step. "
            "Params: n_intents (int), n_variants (int), languages (list). All optional."
        ),
    },
    "finish": {
        "fn": _tool_finish,
        "description": "End the pipeline when the research goal is complete. Params: reason (str).",
    },
}

_TOOLS_SCHEMA = "\n".join(
    f"- {name}: {spec['description']}"
    for name, spec in TOOLS.items()
)

 
# Orchestrator system prompt
 

_SYSTEM = """You are an autonomous research orchestrator for a GEO (Generative Engine Optimization) pipeline.

GEO measures how often brands/restaurants appear in AI-generated responses and how strong their web presence is.

Your job: achieve the research goal by calling tools in whatever order makes sense.
You decide everything  what to run, when to retry, when to stop.

Available tools:
{tools}

At each step respond ONLY with valid JSON:
{{
  "thought": "your reasoning about the current state and what to do next",
  "action": "tool_name",
  "action_input": {{ ...params or empty dict }}
}}

Guidelines:
- Always call get_status first to see what has already been done.
- If intent discovery produces fewer than 4 intents or the prompts look weak, adjust parameters and retry.
- If entity extraction finds fewer than 3 entities, consider retrying with more prompts.
- Web research is optional if the goal is only about entity mentions — use your judgment.
- After web research completes, ALWAYS call run_merge_and_clean to produce the unified feature matrix.
- If run_merge_and_clean reports failed entities, decide whether to call it again with retry_failed=True
  (worth it if fewer than 5 entities failed; skip if most are failed — indicates a systemic issue).
- Do not retry indefinitely — if 2 retries fail, move on and finish with what you have.
- Call finish when: the unified feature matrix is ready, OR you have tried and the data is the best available.
- The standard pipeline order is: intent_discovery → entity_extraction → web_research → merge_and_clean → finish.
""".format(tools=_TOOLS_SCHEMA)


 
# ReAct loop
 

def run_orchestrator(
    domain: str,
    goal: str | None = None,
    languages: list | None = None,
    n_intents: int = 4,
    n_variants: int = 3,
    max_retries: int = 2,
    max_steps: int = 12,
) -> dict:
    """Run the autonomous orchestrator and return the final pipeline state.

    Args:
        domain:      Research domain (e.g. "Tunisian restaurants").
        goal:        Optional natural-language description of the research goal.
                     Defaults to a standard GEO research goal for the domain.
        languages:   List of language codes for prompts (default ["fr"]).
        n_intents:   Initial hint for number of intents (orchestrator may adjust).
        n_variants:  Initial hint for prompt variants per intent.
        max_retries: Max retries hint passed to agents.
        max_steps:   Hard cap on orchestrator reasoning steps.

    Returns:
        Final pipeline state dict.
    """
    _init(domain, languages or ["fr"], n_intents, n_variants, max_retries)

    if goal is None:
        goal = (
            f"Conduct a full GEO (Generative Engine Optimization) analysis for '{domain}'. "
            f"1) Generate diverse user prompts covering different intents. "
            f"2) Query LLMs to discover which brands/restaurants are mentioned. "
            f"3) Collect web presence data for the top entities. "
            f"Aim for at least 4 distinct intents, 5+ entities, and web data for 70%+ of entities."
        )

    print(f"\n{'='*60}")
    print(f"[Orchestrator] Goal: {goal}")
    print(f"{'='*60}\n")

    history = []

    for step in range(1, max_steps + 1):
        history_text = "\n".join(
            f"Step {i+1}: {h}" for i, h in enumerate(history)
        ) if history else "No steps taken yet."

        prompt = (
            f"Research goal: {goal}\n\n"
            f"Steps taken so far:\n{history_text}\n\n"
            f"This is step {step}/{max_steps}. What do you do next?"
        )

        decision = call_llm_json(prompt=prompt, system=_SYSTEM, max_completion_tokens=1024)

        if not isinstance(decision, dict):
            print(f"[Orchestrator] Step {step}: invalid LLM response, skipping")
            history.append(f"(invalid response — skipped)")
            continue

        thought      = decision.get("thought", "")
        action       = decision.get("action", "")
        action_input = decision.get("action_input", {})
        if not isinstance(action_input, dict):
            action_input = {}

        print(f"[Orchestrator] Step {step}/{max_steps}")
        print(f"  Thought : {thought}")
        print(f"  Action  : {action}({action_input})")

        if action not in TOOLS:
            msg = f"Unknown action '{action}'. Available: {list(TOOLS.keys())}"
            print(f"  ERROR   : {msg}")
            history.append(f"{action}(...) -> ERROR: {msg}")
            continue

        try:
            tool_result = TOOLS[action]["fn"](**action_input)
        except TypeError as e:
            tool_result = f"ERROR calling {action}: {e}"

        preview = tool_result[:300].replace("\n", " ")
        print(f"  Result  : {preview}")
        history.append(f"{action}({action_input}) -> {tool_result[:400]}")

        if action == "finish":
            print(f"\n[Orchestrator] Pipeline complete after {step} steps.")
            break

    else:
        print(f"\n[Orchestrator] Reached max steps ({max_steps}) — stopping.")

    return _state
