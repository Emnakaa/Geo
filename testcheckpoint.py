"""Run the remaining pipeline: Agent 2 retry on triage-confirmed entities, then Agent 3.

Usage:
    python testcheckpoint.py
"""

import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FEATURES_GLOBAL_PATH
from agents.agent2_react import run_agent2_node
from agents.agent3_merge import run_agent3_merge
from pipeline_state import initial_state

REPORT_PATH = "geo_output/merge_report.json"


def load_retry_entities() -> list:
    """Read the LLM-triaged retry list from the last merge report."""
    if not os.path.exists(REPORT_PATH):
        print("[run] No merge report found — run Agent 3 first.")
        return []
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    retry = report.get("retry_entities", [])
    print(f"[run] Retry queue: {len(retry)} entities from merge report")
    return retry


def load_entity_features() -> list:
    """Load Agent 1 entity features as list of dicts."""
    if not os.path.exists(FEATURES_GLOBAL_PATH):
        print(f"[run] Entity features not found at {FEATURES_GLOBAL_PATH}")
        return []
    df = pd.read_csv(FEATURES_GLOBAL_PATH, encoding="utf-8-sig")
    return df.to_dict(orient="records")


def run_agent2_retry(retry_entities: list, all_entities: list) -> None:
    """Run Agent 2 only on the retry-list entities."""
    if not retry_entities:
        print("[run] Nothing to retry.")
        return

    retry_set = {e.lower().strip() for e in retry_entities}
    retry_rows = [
        row for row in all_entities
        if row.get("canonical_entity", "").lower().strip() in retry_set
    ]

    print(f"[run] Agent 2 retry: {len(retry_rows)} entities matched in features")

    state = initial_state(domain="Tunisian restaurants")
    state["entity_features_global"] = all_entities
    state["_retry_entities"] = retry_rows   # Agent 2 node picks this up

    result = run_agent2_node(state)

    errors = [e for e in result.get("errors", []) if e]
    if errors:
        print(f"[run] Agent 2 errors: {errors}")
    else:
        print("[run] Agent 2 retry complete — no errors")


def run_agent3() -> dict:
    """Re-run Agent 3 to merge the updated web features."""
    print("\n[run] Running Agent 3 merge...")
    report = run_agent3_merge()
    return report


def main():
    print("=" * 55)
    print("GEO Pipeline — Retry + Final Merge")
    print("=" * 55)

    retry_entities = load_retry_entities()
    all_entities   = load_entity_features()

    # Step 1: re-run Agent 2 on triage-confirmed entities
    run_agent2_retry(retry_entities, all_entities)

    # Step 2: re-run Agent 3 (dedup + triage + merge on updated web_features)
    report = run_agent3()

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"  Entities        : {report.get('unified_rows')}")
    print(f"  Complete        : {report.get('quality_complete')}")
    print(f"  Partial         : {report.get('quality_partial')}")
    print(f"  Still failed    : {report.get('quality_agent2_failed')}")
    print(f"  Retry queue     : {len(report.get('retry_entities', []))}")
    print(f"  Google Maps     : {report.get('gm_coverage_pct')}%")
    print(f"  TripAdvisor     : {report.get('ta_coverage_pct')}%")
    print(f"  Avg completeness: {report.get('avg_data_completeness')}")
    print("=" * 55)


if __name__ == "__main__":
    main()
