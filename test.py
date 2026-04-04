from graph import build_graph, initial_state

graph = build_graph()
result = graph.invoke(initial_state(
    domain="Tunisian restaurants",
    languages=["fr"],
    n_intents=2,
    n_variants=2,
    max_retries=1,   # allow 1 supervisor-triggered retry per agent
))

print("\n=== PIPELINE RESULT ===")
print(f"Final step       : {result['current_step']}")
print(f"Errors           : {result['errors']}")
print(f"Prompts          : {len(result['prompt_set'])}")
print(f"Raw responses    : {len(result['raw_responses'])}")
print(f"Entities (clean) : {len(result['clean_entities'])}")
print(f"Global features  : {len(result['entity_features_global'])}")
print(f"Web features     : {len(result['web_features'])}")
print(f"Agent0 retries   : {result['agent0_retries']}")
print(f"Agent1 retries   : {result['agent1_retries']}")
print(f"Agent2 retries   : {result['agent2_retries']}")
print(f"Token usage      : {result['token_usage']}")
print(f"\nSupervisor notes :")
for note in result['supervisor_notes']:
    print(f"  {note}")
