"""Shared LangGraph state definition for the GEO pipeline."""

from typing import Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    domain: str
    languages: List[str]
    n_intents: int
    n_variants: int
    max_reflection_loops: int

    # ── Agent 0 output ────────────────────────────────────────────────────────
    intents: List[Dict[str, Any]]           # intent dicts (not DataFrame)
    prompt_set: List[Dict[str, Any]]        # prompt_set CSV rows (not DataFrame)
    agent0_quality_score: Optional[float]

    # ── Agent 1 output ────────────────────────────────────────────────────────
    raw_responses: List[Dict[str, Any]]          # raw_responses.csv rows
    extracted_entities: List[Dict[str, Any]]     # extracted_entities.csv rows
    clean_entities: List[Dict[str, Any]]         # clean_entities.csv rows
    entity_features: List[Dict[str, Any]]        # entity_features.csv rows
    entity_features_global: List[Dict[str, Any]] # entity_features_global.csv rows

    # ── Agent 2 output ────────────────────────────────────────────────────────
    web_features: List[Dict[str, Any]]      # web_features.csv rows

    # ── Token accounting (replaces global TOKEN_USAGE dict) ──────────────────
    # Accumulated across all agents; each node merges its own counts in.
    token_usage: Dict[str, int]  # keys: prompt_tokens, completion_tokens, total_tokens

    # ── Supervisor control ────────────────────────────────────────────────────
    agent0_retries: int          # how many times agent0 has been retried by supervisor
    agent1_retries: int
    agent2_retries: int
    max_retries: int             # ceiling applied by every supervisor node
    supervisor_notes: List[str]  # supervisor decisions appended each round

    # ── Control flow ──────────────────────────────────────────────────────────
    errors: List[str]    # non-fatal errors appended by each node
    current_step: str    # last completed step label
