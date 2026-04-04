"""LangGraph pipeline wiring for the GEO agentic pipeline.

Architecture:
    agent0 -> supervisor0 -> agent1 -> supervisor1 -> agent2 (ReAct/MCP) -> supervisor2 -> END
                  ^-- retry --|              ^-- retry --|                        ^-- retry --|

- agent0/1 use LangGraph nodes with internal reflection loops.
- agent2 is a ReAct agent that calls MCP tool servers (search, scrape, wiki)
  and decides which tools to invoke per entity.
- Each supervisor evaluates output quality and can trigger retries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END

from pipeline_state import PipelineState
from agents.agent0 import run_agent0_node
from agents.agent1 import run_agent1_node
from agents.agent2_react import run_agent2_node
from supervisor import (
    supervisor0_node, route_after_supervisor0,
    supervisor1_node, route_after_supervisor1,
    supervisor2_node, route_after_supervisor2,
)


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("agent0",      run_agent0_node)
    graph.add_node("supervisor0", supervisor0_node)
    graph.add_node("agent1",      run_agent1_node)
    graph.add_node("supervisor1", supervisor1_node)
    graph.add_node("agent2",      run_agent2_node)
    graph.add_node("supervisor2", supervisor2_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("agent0")
    graph.add_edge("agent0", "supervisor0")

    # supervisor0 -> agent1 (continue) | agent0 (retry) | END (abort)
    graph.add_conditional_edges(
        "supervisor0",
        route_after_supervisor0,
        {"agent0": "agent0", "agent1": "agent1", END: END},
    )

    graph.add_edge("agent1", "supervisor1")

    # supervisor1 -> agent2 (continue) | agent1 (retry) | END (abort)
    graph.add_conditional_edges(
        "supervisor1",
        route_after_supervisor1,
        {"agent1": "agent1", "agent2": "agent2", END: END},
    )

    graph.add_edge("agent2", "supervisor2")

    # supervisor2 -> END (continue/abort) | agent2 (retry)
    graph.add_conditional_edges(
        "supervisor2",
        route_after_supervisor2,
        {"agent2": "agent2", END: END},
    )

    return graph.compile()


def initial_state(
    domain: str,
    languages: list | None = None,
    n_intents: int = 4,
    n_variants: int = 3,
    max_reflection_loops: int = 2,
    max_retries: int = 1,
) -> PipelineState:
    """Return a fully-initialised state dict ready to pass to graph.invoke()."""
    return PipelineState(
        # inputs
        domain               = domain,
        languages            = languages or ["fr"],
        n_intents            = n_intents,
        n_variants           = n_variants,
        max_reflection_loops = max_reflection_loops,
        # agent 0
        intents              = [],
        prompt_set           = [],
        agent0_quality_score = None,
        # agent 1
        raw_responses        = [],
        extracted_entities   = [],
        clean_entities       = [],
        entity_features      = [],
        entity_features_global = [],
        # agent 2
        web_features         = [],
        # supervisor
        agent0_retries       = 0,
        agent1_retries       = 0,
        agent2_retries       = 0,
        max_retries          = max_retries,
        supervisor_notes     = [],
        # control
        token_usage          = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        errors               = [],
        current_step         = "init",
    )
