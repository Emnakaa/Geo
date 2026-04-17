# -*- coding: utf-8 -*-
"""Agent 3 — Clean, merge, and validate pipeline outputs.

Phase 3 of the GEO pipeline:
    Input  : geo_output/entity_features_global.csv  (Agent 1 — LLM visibility)
             agent2_output/web_features.csv          (Agent 2 — web presence)
    Output : geo_output/unified_features.csv         (joined, clean, ready for Phase 4 PCA)

Design principles:
    - No hardcoded logic about which entities are "valid" — the LLM-agent (orchestrator)
      decides what to do with the merge report summary.
    - Agent3 is deterministic: clean, normalise, join, score completeness.
    - It reports every data quality issue so the orchestrator can reason about
      whether to re-run Agent 2 for failed entities or proceed to Phase 4.
"""

import os
import csv
import json
import unicodedata
import re
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    FEATURES_GLOBAL_PATH,
    CHECKPOINT_FILE,
    OUTPUT_DIR,
)

UNIFIED_PATH = os.path.join(OUTPUT_DIR, "unified_features.csv")
MERGE_REPORT_PATH = os.path.join(OUTPUT_DIR, "merge_report.json")
REJECTED_PATH = os.path.join(OUTPUT_DIR, "rejected_entities.csv")

# ── Entity name normalisation ─────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Lowercase, strip whitespace, collapse spaces, strip accents."""
    if not isinstance(name, str):
        return ""
    # Unicode normalisation: NFD separates accents from base chars
    nfd = unicodedata.normalize("NFD", name)
    # Remove combining accent marks
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip().lower()


# ── Non-restaurant entity filter ─────────────────────────────────────────────
# Entities that are cities, countries, or generic terms — not businesses.
_GEO_TERMS = {
    "tunis", "tunisia", "sfax", "sousse", "djerba", "monastir", "hammamet",
    "nabeul", "gabes", "kairouan", "bizerte", "restaurant", "cafe", "brasserie",
}


def _is_geo_entity(name: str) -> bool:
    return _normalise(name) in _GEO_TERMS


# ── Agent 2 CSV cleaner ───────────────────────────────────────────────────────

def clean_agent2(path: str = CHECKPOINT_FILE) -> pd.DataFrame:
    """
    Load and prepare the agent2 web_features.csv for merging.

    Agent2 now sanitises values at write time (_sanitise in _save_row),
    so the CSV is always well-formed. This function handles only the
    join-preparation concerns:
    - Deduplicate on canonical_entity (keep highest overall_confidence)
    - Filter geographic/generic entities that leaked through extraction
    - Type coercion for numerics and booleans
    - Row-level quality flag: 'complete' / 'partial' / 'failed'

    The `on_bad_lines='skip'` is kept as a safety net for files written
    by older versions of the pipeline.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        on_bad_lines="skip",   # safety net for legacy files only
        low_memory=False,
    )

    if df.empty:
        return df

    # Normalise entity name for joining
    df["_entity_norm"] = df["canonical_entity"].apply(_normalise)

    # Remove geographic / generic entities that leaked through
    df = df[~df["_entity_norm"].apply(_is_geo_entity)].copy()

    # Deduplicate: keep the row with highest overall_confidence
    df["overall_confidence"] = pd.to_numeric(df["overall_confidence"], errors="coerce").fillna(0)
    df = (
        df.sort_values("overall_confidence", ascending=False)
          .drop_duplicates(subset="_entity_norm", keep="first")
          .reset_index(drop=True)
    )

    # Numeric coercions
    for col in ["gm_rating", "gm_review_count", "ta_rating", "ta_review_count",
                "ig_followers", "ig_posts", "fb_page_likes",
                "overall_confidence", "data_source_count", "review_total",
                "wd_founded"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean coercions
    for col in ["has_wikipedia", "has_instagram", "has_facebook",
                "has_tripadvisor", "has_website", "has_phone", "has_address",
                "has_wikidata"]:
        if col in df.columns:
            df[col] = df[col].map(
                {"True": True, "False": False, True: True, False: False}
            ).fillna(False)

    # Row-level quality flag
    def _quality(row) -> str:
        conf = row.get("overall_confidence", 0) or 0
        if conf == 0:
            return "failed"
        if conf >= 0.7 and (row.get("data_source_count") or 0) >= 3:
            return "complete"
        return "partial"

    df["data_quality"] = df.apply(_quality, axis=1)

    return df


# ── Agent 1 loader ────────────────────────────────────────────────────────────

def load_agent1(path: str = FEATURES_GLOBAL_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    df["_entity_norm"] = df["canonical_entity"].apply(_normalise)

    # Apply the same threshold as Agent 2 — only keep entities worth enriching.
    # Entities below this bar are never processed by Agent 2, so keeping them
    # would produce permanent agent2_failed rows in the unified output.
    before = len(df)
    if "mention_count" in df.columns and "stability_score" in df.columns:
        df = df[(df["mention_count"] >= 2) | (df["stability_score"] >= 0.05)].copy()
    elif "mention_count" in df.columns:
        df = df[df["mention_count"] >= 2].copy()
    print(f"  Agent1 filter   : {before} → {len(df)} entities (mention_count>=2 or stability_score>=0.05)")

    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def _fuzzy_match_a2(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Build a mapping from Agent1 _entity_norm → Agent2 row index using:
    1. Exact normalised name match
    2. Substring containment (a1_norm in a2_norm or a2_norm in a1_norm)
    3. Word-overlap ≥ 2 shared tokens
    Returns df2 with an added '_entity_norm_a1' column so the join key matches df1.
    """
    if df2.empty:
        return df2

    a2_norms = df2["_entity_norm"].tolist()
    a2_idx_map: dict[str, int] = {n: i for i, n in enumerate(a2_norms)}

    matched_pairs: list[tuple[str, int]] = []  # (a1_norm, a2_row_index)

    for a1_norm in df1["_entity_norm"]:
        # 1. Exact match
        if a1_norm in a2_idx_map:
            matched_pairs.append((a1_norm, a2_idx_map[a1_norm]))
            continue
        # 2. Substring containment
        best_idx = None
        best_len = 0
        for i, a2_norm in enumerate(a2_norms):
            if a1_norm in a2_norm or a2_norm in a1_norm:
                length = max(len(a1_norm), len(a2_norm))
                if length > best_len:
                    best_len = length
                    best_idx = i
        if best_idx is not None:
            matched_pairs.append((a1_norm, best_idx))
            continue
        # 3. Word overlap
        a1_words = set(a1_norm.split())
        best_idx = None
        best_overlap = 1  # require at least 2 shared words
        for i, a2_norm in enumerate(a2_norms):
            overlap = len(a1_words & set(a2_norm.split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_idx is not None:
            matched_pairs.append((a1_norm, best_idx))

    # Build a lookup: a1_norm → a2_row
    a1_to_a2: dict[str, int] = {a1: idx for a1, idx in matched_pairs}

    # Add a join key to df2 rows based on matched a1 norms
    df2 = df2.copy()
    df2["_entity_norm_a1"] = df2.index.map(
        {idx: a1 for a1, idx in a1_to_a2.items()}
    )
    return df2, a1_to_a2


def merge_features(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join Agent1 (LLM visibility) ← Agent2 (web presence) on normalised entity name.
    Uses fuzzy matching (exact → substring → word-overlap) to handle Agent2 returning
    enriched canonical names different from Agent1's original extraction.

    Agent 1 is the source of truth: every entity the LLMs mentioned is kept.
    Agent 2 data fills in web presence where available; NaN means no data collected.

    Returns a unified DataFrame sorted by stability_score descending.
    """
    if df1.empty:
        return pd.DataFrame()

    if df2.empty:
        merged = df1.copy()
    else:
        # Build rename map for agent2 to avoid column collisions on non-key fields
        a2_cols = [c for c in df2.columns if c not in {"canonical_entity", "_entity_norm"}]
        df2_slim = df2[["_entity_norm"] + a2_cols].copy()

        # Drop columns that already exist identically in agent1
        dup_cols = set(df1.columns) & set(df2_slim.columns) - {"_entity_norm"}
        df2_slim = df2_slim.drop(columns=list(dup_cols), errors="ignore")

        # First try exact merge
        exact_merged = df1.merge(df2_slim, on="_entity_norm", how="left")
        exact_matched = exact_merged["overall_confidence"].notna().sum() if "overall_confidence" in exact_merged.columns else 0

        # Apply fuzzy matching to improve coverage
        df2_fuzzy, a1_to_a2 = _fuzzy_match_a2(df1, df2)

        # Build a clean lookup: a1_norm -> a2 data row (as dict)
        a2_data_cols = [c for c in df2.columns
                        if c not in {"canonical_entity", "_entity_norm", "_entity_norm_a1"}
                        and c not in set(df1.columns)]
        rows_for_merge = []
        seen_a1 = set()
        for a1_norm, a2_idx in a1_to_a2.items():
            if a1_norm in seen_a1:
                continue
            seen_a1.add(a1_norm)
            row = {"_entity_norm": a1_norm}
            for col in a2_data_cols:
                if col in df2.columns:
                    row[col] = df2.iloc[a2_idx][col]
            rows_for_merge.append(row)

        df2_fuzzy_slim = pd.DataFrame(rows_for_merge).reset_index(drop=True)

        fuzzy_merged = df1.merge(df2_fuzzy_slim, on="_entity_norm", how="left")
        fuzzy_matched = fuzzy_merged["overall_confidence"].notna().sum() if "overall_confidence" in fuzzy_merged.columns else 0

        print(f"  Exact join matched  : {exact_matched}/{len(df1)} entities")
        print(f"  Fuzzy join matched  : {fuzzy_matched}/{len(df1)} entities")
        merged = fuzzy_merged

    # ── Data completeness score ────────────────────────────────────────────────
    # Fraction of key observable fields that are non-null.
    key_fields = [
        "gm_rating", "gm_review_count", "gm_address",
        "ta_rating", "ta_review_count",
        "has_wikipedia", "wd_entity_type",
        "ig_followers", "fb_page_likes",
        "overall_confidence",
    ]
    available = [f for f in key_fields if f in merged.columns]
    if available:
        merged["data_completeness"] = (
            merged[available].notna().sum(axis=1) / len(available)
        ).round(3)
    else:
        merged["data_completeness"] = 0.0

    # ── Overall data quality label ─────────────────────────────────────────────
    def _unified_quality(row) -> str:
        dq = row.get("data_quality", "")
        if dq == "failed" or pd.isna(dq):
            return "agent2_failed"
        dc = row.get("data_completeness", 0) or 0
        if dc >= 0.7:
            return "complete"
        if dc >= 0.4:
            return "partial"
        return "sparse"

    merged["unified_quality"] = merged.apply(_unified_quality, axis=1)

    # Sort by stability_score if available
    if "stability_score" in merged.columns:
        merged = merged.sort_values("stability_score", ascending=False).reset_index(drop=True)

    # Remove internal join key
    merged = merged.drop(columns=["_entity_norm"], errors="ignore")

    return merged


# ── Report ────────────────────────────────────────────────────────────────────

def _build_report(df1: pd.DataFrame, df2: pd.DataFrame, unified: pd.DataFrame) -> dict:
    a1_entities = set(df1["_entity_norm"].tolist()) if "_entity_norm" in df1.columns else set()
    a2_entities = set(df2["_entity_norm"].tolist()) if "_entity_norm" in df2.columns else set()

    failed = unified[unified["unified_quality"] == "agent2_failed"]["canonical_entity"].tolist()
    complete = unified[unified["unified_quality"] == "complete"]["canonical_entity"].tolist()
    partial  = unified[unified["unified_quality"] == "partial"]["canonical_entity"].tolist()

    return {
        "agent1_entities":       len(a1_entities),
        "agent2_entities":       len(a2_entities),
        "overlap":               len(a1_entities & a2_entities),
        "only_in_agent1":        list(a1_entities - a2_entities),
        "only_in_agent2":        list(a2_entities - a1_entities),
        "unified_rows":          len(unified),
        "quality_complete":      len(complete),
        "quality_partial":       len(partial),
        "quality_agent2_failed": len(failed),
        "failed_entities":       failed,
        "avg_data_completeness": round(unified["data_completeness"].mean(), 3)
                                  if "data_completeness" in unified.columns else None,
        "avg_overall_confidence": round(
            pd.to_numeric(unified.get("overall_confidence"), errors="coerce").mean(), 3
        ) if "overall_confidence" in unified.columns else None,
        "ta_coverage_pct":       round(
            unified["ta_url"].notna().sum() / len(unified) * 100, 1
        ) if "ta_url" in unified.columns else 0,
        "gm_coverage_pct":       round(
            unified["gm_rating"].notna().sum() / len(unified) * 100, 1
        ) if "gm_rating" in unified.columns else 0,
        "wikipedia_coverage_pct": round(
            unified["has_wikipedia"].sum() / len(unified) * 100, 1
        ) if "has_wikipedia" in unified.columns else 0,
    }


# ── Stop words for content-word matching ─────────────────────────────────────

_STOP_WORDS = frozenset({
    "le", "la", "les", "l", "de", "du", "des", "d", "el", "al", "et",
    "un", "une", "au", "aux", "en", "restaurant", "cafe", "dar", "maison",
    "chez", "brasserie", "bistrot", "the", "a", "an",
})


def _content_words(name: str) -> frozenset:
    """Extract meaningful (non-stop) words from a normalised entity name."""
    norm = _normalise(name)
    tokens = frozenset(re.split(r"[\s'\u2019\-]+", norm)) - _STOP_WORDS - {""}
    return tokens


# ── LLM deduplication ─────────────────────────────────────────────────────────

def llm_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and merge duplicate entities (same restaurant, different surface forms).

    Phase 1 — candidate detection : group entities by shared content words.
    Phase 2 — LLM verification    : ask LLM whether each candidate group is the same place.
    Phase 3 — merge                : sum mention counts, keep best web data, drop dup rows.
    """
    from llm_utils import call_llm

    names = df["canonical_entity"].tolist()
    content = {n: _content_words(n) for n in names}

    # Build candidate groups (content-word overlap >= 1)
    assigned: set = set()
    groups: list = []
    for i, a in enumerate(names):
        if a in assigned or not content[a]:
            continue
        group = [a]
        for b in names[i + 1:]:
            if b in assigned:
                continue
            if content[a] & content[b]:
                group.append(b)
        if len(group) > 1:
            assigned.update(group)
            groups.append(group)

    if not groups:
        print("  LLM dedup     : no candidate groups found")
        return df

    print(f"  LLM dedup     : {len(groups)} candidate groups → asking LLM")

    prompt = (
        "You are deduplicating a Tunisian restaurant dataset.\n\n"
        "For each group of names below, decide if ALL names refer to the SAME physical restaurant.\n\n"
        "Groups:\n"
        + json.dumps(groups, ensure_ascii=False, indent=2)
        + "\n\nRules:\n"
        "- Article/prefix differences only (\"le café des nattes\" vs \"café des nattes\") → same place\n"
        "- Different final word (\"dar el jeld\" vs \"dar el jaziri\") → different places\n"
        "- Short form vs full form (\"el mouradi\" vs \"restaurant el mouradi\") → same place\n"
        "- City suffix variant (\"el mouradi\" vs \"el mouradi sousse\") → same place\n\n"
        "Respond ONLY with a JSON array:\n"
        "[\n"
        "  {\"group\": [\"name1\", \"name2\"], \"same_place\": true, "
        "\"keep\": \"most complete name to keep as canonical\"}\n"
        "]"
    )

    try:
        response = call_llm(prompt=prompt, max_completion_tokens=2048)
        m = re.search(r'\[.*\]', response, re.DOTALL)
        if not m:
            print("  LLM dedup     : parse failed — skipping")
            return df
        decisions = json.loads(m.group())
    except Exception as e:
        print(f"  LLM dedup     : error ({e}) — skipping")
        return df

    # Build merge map: dup_name → keep_name
    merge_map: dict = {}
    for d in decisions:
        if not d.get("same_place"):
            continue
        keep = d.get("keep", "")
        group = d.get("group", [])
        if keep not in names:
            keep = group[0] if group else ""
        for name in group:
            if name != keep and name in names:
                merge_map[name] = keep

    if not merge_map:
        print("  LLM dedup     : no merges needed")
        return df

    # Apply merges: sum mention_count, fill nulls from duplicate row
    df = df.copy()
    for dup_name, keep_name in merge_map.items():
        dup_idxs  = df.index[df["canonical_entity"] == dup_name].tolist()
        keep_idxs = df.index[df["canonical_entity"] == keep_name].tolist()
        if not dup_idxs or not keep_idxs:
            continue
        di, ki = dup_idxs[0], keep_idxs[0]

        # Sum mention count
        df.at[ki, "mention_count"] = (
            (df.at[ki, "mention_count"] or 0) + (df.at[di, "mention_count"] or 0)
        )

        # Fill missing fields in keep row from duplicate row
        for col in df.columns:
            if col in ("canonical_entity", "mention_count"):
                continue
            kv, dv = df.at[ki, col], df.at[di, col]
            if (pd.isna(kv) or kv == "" or kv == 0) and not (pd.isna(dv) or dv == "" or dv == 0):
                df.at[ki, col] = dv

    df = df[~df["canonical_entity"].isin(merge_map)].reset_index(drop=True)
    print(f"  LLM dedup     : merged {len(merge_map)} duplicates → {len(df)} entities remain")
    return df


# ── Robust JSON object extractor ─────────────────────────────────────────────

def _parse_json_objects(text: str) -> list:
    """
    Extract as many valid JSON objects as possible from an LLM response.
    Handles truncated arrays and malformed entries by parsing object-by-object
    instead of the whole array at once — apostrophes in entity names won't
    break the entire batch.
    """
    # First try the whole array
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Fall back: extract individual {...} objects and parse each separately
    results = []
    for obj_match in re.finditer(r'\{[^{}]+\}', text, re.DOTALL):
        try:
            results.append(json.loads(obj_match.group()))
        except json.JSONDecodeError:
            # Try fixing common issues: unescaped apostrophes inside string values
            fixed = re.sub(r"(?<=\w)'(?=\w)", r"\\'", obj_match.group())
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    return results


# ── LLM triage of failed entities ─────────────────────────────────────────────

def llm_triage_failed(df: pd.DataFrame) -> tuple:
    """
    Classify agent2_failed entities and remove non-viable ones from the output.

    Classifications:
      retry         — likely a real Tunisian restaurant, Agent 2 missed it
      hallucination — LLM invented the name (e.g. a foreign or fictional restaurant)
      non_restaurant— real place but not a restaurant (hotel, city, souk, etc.)
      generic       — name too vague to ever identify uniquely

    Returns (cleaned_df, retry_list).
    Removed entities are saved to rejected_entities.csv for audit.
    """
    from llm_utils import call_llm
    from collections import Counter

    failed = df[df["unified_quality"] == "agent2_failed"].copy()
    if failed.empty:
        return df, []

    entities = (
        failed[["canonical_entity", "mention_count", "stability_score"]]
        .to_dict(orient="records")
    )

    prompt = (
        "You are auditing a Tunisian restaurant dataset.\n\n"
        "The following entities were extracted by LLMs but could NOT be verified online.\n"
        "Classify each as one of:\n"
        "- \"retry\"          : very likely a real Tunisian restaurant — Agent 2 missed it, worth retrying\n"
        "- \"hallucination\"  : LLM invented this name (e.g. foreign restaurant, fictional place)\n"
        "- \"non_restaurant\" : real place but not a restaurant (hotel chain, city landmark, souk, etc.)\n"
        "- \"generic\"        : name too vague to identify uniquely (\"le grand restaurant\", \"la villa\")\n\n"
        "Context: domain is Tunisian restaurants. Higher mention_count = more LLMs named it.\n\n"
        "Entities:\n"
        + json.dumps(entities, ensure_ascii=False, indent=2)
        + "\n\nRespond ONLY with a JSON array:\n"
        "[\n"
        "  {\"entity\": \"exact name\", \"decision\": \"retry|hallucination|non_restaurant|generic\","
        " \"reason\": \"one sentence\"}\n"
        "]"
    )

    try:
        response = call_llm(prompt=prompt, max_completion_tokens=3000)
        decisions = _parse_json_objects(response)
        if not decisions:
            print("  LLM triage    : parse failed — skipping")
            return df, []
    except Exception as e:
        print(f"  LLM triage    : error ({e}) — skipping")
        return df, []

    decision_map = {d["entity"]: d["decision"] for d in decisions}
    reason_map   = {d["entity"]: d.get("reason", "") for d in decisions}

    counts = Counter(decision_map.values())
    print(f"  LLM triage    : {dict(counts)}")

    retry_list = [name for name, dec in decision_map.items() if dec == "retry"]

    # Tag triaged entities
    df = df.copy()
    df["triage_decision"] = df["canonical_entity"].map(decision_map)
    df["triage_reason"]   = df["canonical_entity"].map(reason_map)

    # Save rejected entities to audit file, then remove from main output
    remove_set = {name for name, dec in decision_map.items()
                  if dec in ("hallucination", "non_restaurant", "generic")}
    rejected = df[df["canonical_entity"].isin(remove_set)].copy()
    if not rejected.empty:
        rejected.to_csv(REJECTED_PATH, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        print(f"  Rejected      : {len(rejected)} entities → {REJECTED_PATH}")

    df = df[~df["canonical_entity"].isin(remove_set)].reset_index(drop=True)
    print(f"  After triage  : {len(df)} entities remain  ({len(retry_list)} queued for retry)")

    return df, retry_list


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agent3_merge(
    agent1_path: str = FEATURES_GLOBAL_PATH,
    agent2_path: str = CHECKPOINT_FILE,
    output_path: str = UNIFIED_PATH,
    report_path: str = MERGE_REPORT_PATH,
) -> dict:
    """
    Clean, merge, and validate Agent1 + Agent2 outputs.

    Returns a report dict summarising data quality. The report is also
    saved as JSON so the orchestrator can reason about gaps (e.g. entities
    where Agent 2 failed and should be retried).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"Agent 3 — Merge & Clean")
    print(f"{'='*55}")

    df1 = load_agent1(agent1_path)
    if df1.empty:
        msg = f"Agent 1 output not found at {agent1_path}"
        print(f"  [ERROR] {msg}")
        return {"error": msg}

    df2 = clean_agent2(agent2_path)
    if df2.empty:
        print("  [WARN] Agent 2 output not found or empty — merging with agent1 only.")

    print(f"  Agent1 entities : {len(df1)}")
    print(f"  Agent2 entities : {len(df2)}")

    unified = merge_features(df1, df2)

    # ── LLM post-processing ───────────────────────────────────────────────────
    unified = llm_deduplicate(unified)
    unified, retry_list = llm_triage_failed(unified)

    # Save
    unified.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    print(f"  Saved unified   : {output_path}  ({len(unified)} rows)")

    report = _build_report(df1, df2, unified)
    report["retry_entities"] = retry_list
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved report    : {report_path}")

    print(f"\n  Quality breakdown:")
    print(f"    Complete   : {report['quality_complete']}")
    print(f"    Partial    : {report['quality_partial']}")
    print(f"    A2 failed  : {report['quality_agent2_failed']}")
    print(f"    Failed entities: {report['failed_entities']}")
    print(f"\n  Coverage:")
    print(f"    Google Maps   : {report['gm_coverage_pct']}%")
    print(f"    TripAdvisor   : {report['ta_coverage_pct']}%")
    print(f"    Wikipedia     : {report['wikipedia_coverage_pct']}%")
    print(f"    Avg completeness: {report['avg_data_completeness']}")
    if retry_list:
        print(f"\n  Retry queue ({len(retry_list)}): {retry_list[:5]}{'...' if len(retry_list) > 5 else ''}")
    print(f"{'='*55}")

    return report


# ── LangGraph node interface ──────────────────────────────────────────────────

from pipeline_state import PipelineState


def run_agent3_node(state: PipelineState) -> PipelineState:
    """LangGraph/orchestrator node: run Agent 3 merge and update state."""
    errors = list(state.get("errors", []))

    report = run_agent3_merge()

    if "error" in report:
        errors.append(f"agent3: {report['error']}")
        return {**state, "errors": errors, "current_step": "agent3_aborted"}

    return {
        **state,
        "merge_report":    report,
        "retry_entities":  report.get("retry_entities", []),
        "current_step":    "agent3_done",
        "errors":          errors,
    }
