from orchestrator import run_orchestrator

result = run_orchestrator(
    domain="Tunisian restaurants",
    languages=["fr"],
    n_intents=4,
    n_variants=2,
    max_retries=2,
    max_steps=12,
)

print("\n=== PIPELINE RESULT ===")
print(f"Domain           : {result['domain']}")
print(f"Prompts          : {len(result['prompt_set'])}")
print(f"Entities (global): {len(result['entity_features_global'])}")
print(f"Web features     : {len(result['web_features'])}")
print(f"Errors           : {result['errors']}")
print(f"Token usage      : {result['token_usage']}")
