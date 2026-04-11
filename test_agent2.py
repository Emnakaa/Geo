"""Run Agent 2 on all entities (mention_count >= 2 or stability_score >= 0.05)."""
import pandas as pd
from agents.agent2_react import run_agent2_react
from agents.agent3_merge import run_agent3_merge

# Load all entities that meet the quality threshold
df = pd.read_csv("geo_output/entity_features_global.csv")
col = "canonical_entity" if "canonical_entity" in df.columns else df.columns[0]
mask = pd.Series([True] * len(df))
if "mention_count" in df.columns and "stability_score" in df.columns:
    mask = (df["mention_count"] >= 2) | (df["stability_score"] >= 0.05)
elif "mention_count" in df.columns:
    mask = df["mention_count"] >= 2
entities = df[mask][col].dropna().tolist()

import os
# Re-run only entities that had errors (confidence=0 or error column set)
# Keep enriched rows (confidence > 0) intact
checkpoint = "agent2_output/web_features.csv"
if os.path.exists(checkpoint):
    cp = pd.read_csv(checkpoint)
    conf_col = "overall_confidence" if "overall_confidence" in cp.columns else None
    err_col  = "error" if "error" in cp.columns else None
    if conf_col:
        # Successfully enriched = confidence > 0 AND no error
        enriched_mask = cp[conf_col].fillna(0) > 0
        if err_col:
            enriched_mask = enriched_mask & cp[err_col].isna()
        done_with_data = set(cp[enriched_mask]["canonical_entity"].str.lower().tolist())
        entities = [e for e in entities if e.lower() not in done_with_data]
        print(f"  Skipping {len(done_with_data)} already enriched — {len(entities)} to retry")

print(f"\nRunning Agent 2 on {len(entities)} entities\n")

# fresh_start=False — keep existing enriched rows, only overwrite failed ones
rows = run_agent2_react(entities, fresh_start=False)

print("\n=== AGENT 2 RESULTS ===")
for r in rows:
    print(f"  {r.get('canonical_entity'):30s} "
          f"confidence={r.get('overall_confidence','?'):.2f}  "
          f"gm_rating={r.get('gm_rating','null')}  "
          f"ta_rating={r.get('ta_rating','null')}  "
          f"has_wikipedia={r.get('has_wikipedia','?')}")

# Run merge to see unified output
print("\n=== RUNNING MERGE ===")
report = run_agent3_merge()
print(f"  GM coverage : {report.get('gm_coverage_pct')}%")
print(f"  TA coverage : {report.get('ta_coverage_pct')}%")
print(f"  Avg confidence: {report.get('avg_overall_confidence')}")
