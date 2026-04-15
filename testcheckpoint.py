import sys, pandas as pd
sys.path.insert(0, '.')
df = pd.read_csv('geo_output/entity_features_global.csv')
entities = df[(df['mention_count']>=2)|(df['stability_score']>=0.05)]['canonical_entity'].tolist()
from agents.agent2_react import run_agent2_react
run_agent2_react(entities[:112])