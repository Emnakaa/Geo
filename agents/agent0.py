# -*- coding: utf-8 -*-
"""Agent 0 - Intent discovery and prompt generation."""

import os
import json
import pandas as pd

from config import MODEL_INTENT, MODEL_ANALYST2
from llm_utils import call_llm

MODEL_1 = MODEL_INTENT
MODEL_2 = MODEL_ANALYST2


def query_llm_structured(model: str, prompt: str, system: str = "") -> str:
    """Call LLM with automatic fallback on rate-limit errors."""
    return call_llm(prompt=prompt, system=system, preferred_model=model,
                    max_completion_tokens=4096, temperature=0.2)

import json
import pandas as pd
import time

def agent0_generate_intents(domain: str, n_intents: int, model: str) -> list:
    """
    Step 1: LLM discovers relevant intent types for the given domain.
    """
    system = """You are an expert in user behavior and search intent analysis, specialising in GEO (Generative Engine Optimization) research.
Your job is to identify the most realistic and diverse intent types that users express when querying an AI assistant about a specific domain.

STRICT RULES for GEO-valid intents:
1. Every intent must produce prompts where a user asks an AI ABOUT establishments — never TO an establishment.
   INVALID: "Quels sont les plats de votre restaurant ?" (addresses the AI as if it IS the restaurant)
   VALID:   "Quels sont les restaurants tunisiens les plus réputés à Tunis ?"
2. Every intent must be geographically anchored to the country/city in the domain — never drift to other cities or countries.
   INVALID for domain "Tunisian restaurants": prompts about Paris, Lyon, France
   VALID: prompts about Tunis, Sousse, Sfax, Djerba, Monastir, Hammamet
3. Every intent must naturally lead to naming SPECIFIC establishments — not generic cuisine descriptions.
   INVALID: "Comment prépare-t-on le couscous ?" (about food, not brands)
   VALID:   "Quel restaurant tunisien est réputé pour son couscous à Sousse ?"

Always respond in valid JSON only, no explanation, no markdown."""

    prompt = f"""Domain: {domain}

Generate exactly {n_intents} distinct intent categories that users would have
when asking an AI assistant about this domain.

For each intent provide:
- intent_id: short snake_case identifier
- intent_name: clear label
- description: one sentence explaining this intent type (must follow the 3 GEO rules above)

Respond ONLY with a JSON array like:
[
  {{
    "intent_id": "top_recommendation",
    "intent_name": "Top Restaurant Recommendation",
    "description": "User wants the AI to name the best-known or most reputed restaurants in a specific Tunisian city"
  }}
]"""

    response = query_llm_structured(model=model, prompt=prompt, system=system)

    # clean response in case LLM adds markdown
    intents = parse_json_response(response, model, prompt, system)

    print(f"Agent 0, Step 1: {len(intents)} intents discovered")
    return intents

def agent0_generate_prompts(domain: str, intents: list, languages: list,
                             n_variants: int, model: str) -> pd.DataFrame:
    """
    Step 2: For each intent, generate n_variants prompts per language.
    """
    system = """You are an expert in prompt engineering and multilingual NLP, specialising in GEO (Generative Engine Optimization) research.
Your job is to generate realistic, diverse user queries that will reveal which specific brands or establishments an AI assistant knows.

CRITICAL GEO RULE: At least 60% of prompts must be phrased to force the AI to recall and name SPECIFIC, WELL-KNOWN establishments.
- GOOD: "Quel est le restaurant tunisien le plus réputé à Tunis ?" → forces naming a known brand
- GOOD: "Cite-moi les 3 meilleures adresses pour manger tunisien à Tunis"  → forces a ranked list of real names
- BAD:  "Tu connais un bon resto tunisien ?" → too vague, invites invention
- BAD:  "Quels sont les plats de votre restaurant ?" → asks the AI as if it IS the restaurant

Prompts that ask for hours, menus of unnamed restaurants, or generic cuisine descriptions do NOT help GEO — avoid them.
Always respond in valid JSON only, no explanation, no markdown."""

    all_prompts = []

    for intent in intents:
        prompt = f"""Domain: {domain}
Intent: {intent['intent_name']} — {intent['description']}
Languages: {languages}
Number of variants per language: {n_variants}

Generate {n_variants} natural user queries for EACH of these languages: {languages}

Queries must:
- Sound like real user input, not formal questions
- Be phrased to elicit SPECIFIC establishment names (use superlatives, rankings, "cite", "liste", "le meilleur", "les plus connus")
- Be geographically precise (name a city or neighbourhood, not just "Tunisie" generically)
- Be diverse in phrasing and structure
- Stay within the domain and intent

Respond ONLY with a JSON array like:
[
  {{
    "intent_id": "{intent['intent_id']}",
    "language": "fr",
    "variant_id": 1,
    "prompt_text": "..."
  }}
]"""

        response = query_llm_structured(model=model, prompt=prompt, system=system)
        variants = parse_json_response(response, model, prompt, system)

        all_prompts.extend(variants)
        print(f"Intent '{intent['intent_id']}'  {len(variants)} prompts generated")
        time.sleep(0.5)  # avoid rate limiting

    # build DataFrame
    df = pd.DataFrame(all_prompts)

    # add prompt_id
    df = df.reset_index(drop=True)
    df['prompt_id'] = ['P' + str(i+1).zfill(3) for i in range(len(df))]

    # reorder columns
    df = df[['prompt_id', 'intent_id', 'language', 'variant_id', 'prompt_text']]

    return df

def parse_json_response(response: str, model: str,
                        original_prompt: str, system: str,
                        max_retries: int = 3) -> list:
    for attempt in range(max_retries):
        try:
            clean = (response.strip()
                     .removeprefix("```json")
                     .removeprefix("```")
                     .removesuffix("```")
                     .strip())
            return json.loads(clean) #json.loads() converts JSON string to Python object

        except json.JSONDecodeError as e:
            print(f"JSON parse failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                fix_prompt = f"""Your previous response was not valid JSON.
Error: {e}
Your response was:
{response}
Fix it and return ONLY valid JSON, nothing else."""
                response = query_llm_structured(
                    model=model,
                    prompt=fix_prompt,
                    system=system
                )
            else:
                print(f"[ERROR] Failed to parse JSON after {max_retries} attempts - returning empty list")
                return []

def agent0_reflect(domain: str, intents: list,
                   prompts_df: pd.DataFrame, model: str) -> dict:
    """
    Agent reflects on its own output quality before passing to Agent 1.
    Returns quality report + decision to proceed or regenerate.
    """
    system = """You are a critical evaluator of prompt sets for NLP research.
    Respond in valid JSON only."""

    prompt = f"""You generated this prompt set for domain: '{domain}'

Intents discovered: {[i['intent_id'] for i in intents]}
Total prompts generated: {len(prompts_df)}
Languages: {prompts_df['language'].unique().tolist()}
All prompts:
{prompts_df['prompt_text'].tolist()}

Evaluate the quality of this prompt set for GEO (Generative Engine Optimization) research:

1. Are the intents diverse enough for GEO analysis?
2. Are there important intents missing?
3. Are prompts natural and realistic?
4. Is language quality acceptable?
5. GEO CRITICAL — Do at least 60% of prompts force the AI to name SPECIFIC known establishments?
   Count prompts that use superlatives, rankings, "cite", "liste", "le meilleur", "les plus connus", or name a specific place.
   Prompts asking for hours, menus, or generic cuisine info do NOT count.
   If fewer than 60% force specific naming → set proceed: false and list them in issues.

Respond ONLY with:
{{
  "quality_score": 0-10,
  "proceed": true/false,
  "brand_eliciting_pct": <percentage of prompts that force specific brand naming>,
  "missing_intents": [],
  "issues": [],
  "recommendation": "..."
}}"""

    response = query_llm_structured(model=model, prompt=prompt, system=system)
    report = parse_json_response(response, model, prompt, system)

    print(f"\n Agent 0 Self-Reflection:")
    print(f"   Quality score : {report['quality_score']}/10")
    print(f"   Proceed       : {report['proceed']}")
    print(f"   Issues        : {report.get('issues', [])}")
    print(f"   Missing       : {report.get('missing_intents', [])}")

    return report

def agent0_run(domain: str, model: str, languages: list = ["fr", "ar"],
               n_intents: int = 4, n_variants: int = 3,
               max_reflection_loops: int = 2) -> pd.DataFrame:

    print(f"\n Agent 0 starting, Domain: '{domain}'")

    for loop in range(max_reflection_loops):
        print(f"\nLoop {loop + 1}/{max_reflection_loops}")

        # 1: discover intents
        intents = agent0_generate_intents(
            domain=domain, n_intents=n_intents, model=model
        )

        # 2: generate prompts
        prompt_df = agent0_generate_prompts(
            domain=domain, intents=intents,
            languages=languages, n_variants=n_variants, model=model
        )

        # 3:self reflection
        report = agent0_reflect(
            domain=domain, intents=intents,
            prompts_df=prompt_df, model=model
        )

        if report['proceed']:
            print(f"\n Agent 0 satisfied with output at loop {loop+1}")
            break
        else:
            print(f"\n Agent 0 not satisfied - regenerating...")
            # inject missing intents into next loop
            n_intents = n_intents + len(report.get('missing_intents', []))


    else:
        print(f"\n Agent 0 reached max loops - saving best available output")

    output_path = f"prompt_set_{domain.replace(' ', '_')}.csv"
    prompt_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n Agent 0 complete - {len(prompt_df)} prompts -> '{output_path}'")
    return prompt_df

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline_state import PipelineState


def run_agent0_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs Agent 0 and merges results into state."""
    errors = list(state.get("errors", []))
    try:
        prompt_df = agent0_run(
            domain               = state["domain"],
            model                = MODEL_1,
            languages            = state.get("languages", ["fr"]),
            n_intents            = state.get("n_intents", 4),
            n_variants           = state.get("n_variants", 3),
            max_reflection_loops = state.get("max_reflection_loops", 2),
        )
        prompt_set = prompt_df.to_dict(orient="records")
    except Exception as exc:
        errors.append(f"agent0: {exc}")
        prompt_set = []

    return {
        **state,
        "prompt_set":  prompt_set,
        "errors":      errors,
        "current_step": "agent0",
    }