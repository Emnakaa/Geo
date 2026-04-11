# Final Year Project Report — Academic Outline
## Generative Engine Optimization (GEO): An Agentic AI Pipeline for Brand Visibility Analysis in Large Language Models

**Field:** Computer Science / Artificial Intelligence  
**Level:** Final Year / Master's Project  
**Target Audience:** Academic supervisors, jury members, institutional evaluators

---

## TABLE OF CONTENTS

```
1. Introduction
   1.1 Context and Motivation
   1.2 Problem Statement
   1.3 Research Objectives
   1.4 Scope and Limitations
   1.5 Report Structure

2. Literature Review and State of the Art
   2.1 Search Engine Optimization (SEO): Background
   2.2 The Rise of Generative AI in Information Retrieval
   2.3 Generative Engine Optimization (GEO): Emerging Concept
   2.4 Large Language Models and Brand Representation
   2.5 Agentic AI Systems and Multi-Agent Architectures
   2.6 Related Work and Existing Approaches
   2.7 Identified Gaps and Research Opportunity

3. System Design and Architecture
   3.1 Research Approach and Design Philosophy
   3.2 Agentic vs. Pipeline Architecture: The Core Design Choice
   3.3 Data Collection Strategy
   3.4 Evaluation Framework
   3.5 Tools, Technologies, and APIs
   3.6 System Overview
   3.7 Agentic Orchestration Layer
   3.8 Dynamic Model Registry
   3.9 Agent 0: Intent Discovery and Prompt Generation
   3.10 Agent 1: LLM Querying and Entity Extraction
   3.11 Agent 2: Web Presence Research (ReAct/MCP)
   3.12 MCP Tool Servers
   3.13 Pipeline State and Data Flow
   3.14 System Architecture Diagram

4. Implementation and Evaluation
   4.1 Development Environment and Stack
   4.2 Orchestrator Implementation
   4.3 Model Registry and Autonomous Model Selection
   4.4 Agent 0 Implementation
   4.5 Agent 1 Implementation
   4.6 Agent 2 ReAct Loop and MCP Integration
   4.7 Rate Limit and Token Capacity Management
   4.8 Data Persistence and Checkpointing
   4.9 Challenges and Solutions
   4.10 Experimental Setup
   4.11 Dataset Summary
   4.12 Agent 0 Evaluation: Prompt Set Quality
   4.13 Agent 1 Evaluation: Entity Extraction Quality
   4.14 Agent 2 Evaluation: Web Feature Coverage
   4.15 Model Selection Behavior Analysis
   4.16 GEO Metrics: Brand Visibility Ranking
   4.17 Interpretation of Results
   4.18 Contribution to the GEO Field
   4.19 Agentic Architecture: Benefits and Trade-offs
   4.20 Limitations and Ethical Considerations

5. Conclusion and Future Work
   5.1 Summary of Contributions
   5.2 Answer to Research Objectives
   5.3 Future Work and Extensions

References
Appendices
   A. System Configuration and Setup
   B. Sample Pipeline Outputs
   C. System Prompt Texts
   D. Glossary
```

---

## SECTION-BY-SECTION GUIDE

---

### 1. INTRODUCTION

**Purpose:** Orient the reader, establish the research context, and define what the project aims to achieve.

**1.1 Context and Motivation**
- The shift from traditional search engines (Google, Bing) to AI-powered search (ChatGPT, Perplexity, Gemini)
- Businesses increasingly rely on being mentioned in LLM responses, not just search rankings
- The gap: no established methodology exists to measure or improve LLM brand visibility
- Why this matters for businesses in the Arabic/Francophone market (Tunisian context)

**1.2 Problem Statement**
Write as a clear, one-paragraph problem statement:
> "While SEO is well-understood for traditional search engines, the mechanisms by which Large Language Models select, mention, and rank businesses in generative responses remain poorly studied. This project addresses the lack of automated, scalable tools to measure brand visibility in LLM-generated content and to extract the web presence features that correlate with that visibility."

**1.3 Research Objectives**
List 3–5 numbered, measurable objectives:
1. Design an automated pipeline to measure how often named entities (restaurants) appear in LLM responses
2. Build an agentic system capable of autonomous research decisions without hardcoded execution graphs
3. Extract and analyze web presence features (Google Maps, TripAdvisor, social media, Wikipedia) for each entity
4. Evaluate the correlation between web presence strength and LLM mention frequency

**1.4 Scope and Limitations**
- Domain: local restaurant businesses in Tunisia
- Languages: French (primary), Arabic (extensible)
- LLM providers: Groq API, OpenRouter (free tier)
- Not in scope: real-time monitoring, paid advertising analysis, SEO recommendations engine

**1.5 Report Structure**
One paragraph summarizing each chapter in sequence.

> **Writing tip:** Keep the introduction to 4–6 pages. Avoid technical details — save them for later chapters.

---

### 2. LITERATURE REVIEW AND STATE OF THE ART

**Purpose:** Demonstrate academic awareness. Show that the project builds on existing knowledge and fills a real gap.

**2.1 Search Engine Optimization (SEO): Background**
- History and mechanisms of traditional SEO (Brin & Page, 1998; Moz, 2023)
- Key SEO signals: backlinks, page rank, structured data, on-page content
- Why SEO is insufficient for generative AI responses

**2.2 The Rise of Generative AI in Information Retrieval**
- Transformer architecture (Vaswani et al., 2017) — brief
- ChatGPT, GPT-4, LLaMA, Mistral — the modern LLM landscape
- AI Overviews (Google SGE), Perplexity, Bing Copilot — real-world deployment
- How users now receive synthesized answers instead of lists of links

**2.3 Generative Engine Optimization (GEO): Emerging Concept**
- GEO definition: optimizing content to appear in LLM-generated responses
- Key paper: "GEO: Generative Engine Optimization" (Aggarwal et al., 2023)
- Distinction from traditional SEO: no crawlable index, no keyword matching
- What signals influence LLM mentions: training data, web presence, structured data

**2.4 Large Language Models and Brand Representation**
- How LLMs encode entity knowledge during pre-training
- Frequency bias: well-known brands mentioned more often (Mallen et al., 2023)
- Hallucination risk in entity mentions
- Local vs. global brand awareness in LLMs

**2.5 Agentic AI Systems and Multi-Agent Architectures**
- Definition of AI agents (Russell & Norvig, 2020)
- ReAct framework (Yao et al., 2022): reason + act loops
- LangGraph: stateful multi-agent orchestration
- Model Context Protocol (MCP): standardized tool server communication
- Comparison: fixed pipelines vs. autonomous orchestrators

**2.6 Related Work and Existing Approaches**
- BrandGPT, ChatSEO tools (commercial, limited)
- Academic NLP pipelines for named entity recognition (NER)
- Web scraping and data enrichment pipelines
- Multi-provider LLM routing (OpenRouter, LiteLLM)

**2.7 Identified Gaps and Research Opportunity**

| Aspect | Existing Work | This Project |
|---|---|---|
| GEO measurement | Manual / limited | Automated, multi-prompt |
| Agent architecture | Fixed pipelines | Fully autonomous orchestrator |
| Model management | Hardcoded fallbacks | Dynamic registry, learned limits |
| Web feature extraction | Single source | Multi-source (Maps, TA, Wiki, Social) |

> **Writing tip:** Every claim needs a citation. Target 20–35 references. End with a synthesis paragraph that directly motivates your approach.

---

### 3. SYSTEM DESIGN AND ARCHITECTURE

**Purpose:** Explain the design decisions and present the full technical architecture. This chapter covers both the *why* (methodology) and the *what* (design).

**3.1 Research Approach and Design Philosophy**
- Empirical, design-science research methodology
- Iterative development (build → test → measure → improve)
- Why a pipeline approach was chosen, then why it was replaced with an agentic approach

**3.2 Agentic vs. Pipeline Architecture: The Core Design Choice**
- Initial design: hardcoded LangGraph DAG (agent0 → supervisor → agent1 → ...)
- Problem identified: not truly autonomous — LLM only rates quality, sequence is fixed
- Redesign: orchestrator LLM receives a goal, agents become tools, sequence emerges at runtime
- Justify this as the core contribution of the project

**3.3 Data Collection Strategy**
- Domain selection rationale: Tunisian restaurants (local, under-represented in LLMs)
- Language selection: French (primary market language)
- Intent taxonomy: how user intents were classified (recommendation, info, comparison, etc.)
- N-runs methodology: why each prompt is submitted multiple times (stability analysis)
- Web data sources: Google Maps, TripAdvisor, Wikipedia, Instagram, Facebook

**3.4 Evaluation Framework**
Define your metrics:
- **Mention rate**: proportion of LLM runs in which an entity is mentioned
- **Stability score**: consistency of mention rate across runs
- **Web coverage**: % of entities with at least one web data source
- **Data confidence**: agent-assigned 0–1 score per entity
- **Overall pipeline coverage**: entities reaching all 3 stages / total discovered

**3.5 Tools, Technologies, and APIs**

| Component | Technology | Purpose |
|---|---|---|
| Orchestration | LangGraph + custom orchestrator | Multi-agent coordination |
| LLM providers | Groq API + OpenRouter | Model inference |
| Model management | Custom ModelRegistry | Dynamic model selection |
| Tool protocol | FastMCP (stdio) | Tool server communication |
| Web search | SerpApi, DuckDuckGo | Google Maps, general search |
| Encyclopedia | Wikipedia API | Entity Wikipedia presence |
| Social media | Apify | Instagram/Facebook scraping |
| Data storage | CSV (pandas) | Progressive checkpointing |
| Language | Python 3.11 | Implementation |

**3.6 System Overview**
One-paragraph description of the system end-to-end. Include a high-level architecture diagram (Figure 1).

**3.7 Agentic Orchestration Layer**
```
Goal → Orchestrator LLM → selects tool → observes result → reasons → selects next tool → ...
```
- ReAct loop structure
- How the orchestrator decides when to retry, skip, or finish
- Tools available: run_intent_discovery, run_entity_extraction, run_web_research, get_status, finish

**3.8 Dynamic Model Registry**
- Discovery: queries provider APIs at startup (Groq, OpenRouter)
- Capability tier ranking (70b > 32b > 8b)
- Per-model state: TPD exhaustion, TPM blocking, learned token capacity
- `select(preferred, estimated_tokens)` — the autonomous selection algorithm
- State machine: available → TPM blocked → available / TPD exhausted

**3.9 Agent 0: Intent Discovery and Prompt Generation**
- Input: domain string, languages, n_intents, n_variants
- Step 1: LLM discovers intent taxonomy
- Step 2: LLM generates n_variants prompts per intent per language
- Step 3: Self-reflection loop — quality evaluation and regeneration
- Output: prompt_set (list of structured prompt dicts)

**3.10 Agent 1: LLM Querying and Entity Extraction**
- Query each prompt across n_runs using registry-selected models
- Extract named restaurant entities from responses (NER via LLM)
- Enrich entities with descriptive context
- Deduplicate via fuzzy matching + LLM arbitration (5-phase cleaning)
- Compute GEO metrics (mention_rate, stability_score, consistency_label)

**3.11 Agent 2: Web Presence Research (ReAct/MCP)**
- Input: list of canonical entities from Agent 1 (mention_count ≥ 2 or stability_score ≥ 0.05)
- ReAct loop: LLM decides which MCP tools to call and in what order
- Checkpoint system: resume from last saved entity
- Mid-loop model switching on TPD/TPM errors
- Output: web_features.csv with structured per-entity web data

**3.12 MCP Tool Servers**

| Server | Tool | Input | Output |
|---|---|---|---|
| search_server | google_maps_search | entity, location | rating, reviews, address, phone, website |
| search_server | tripadvisor_search | entity, location | ta_rating, ta_url, ta_ranking |
| search_server | ddg_search | query, max_results | list of {title, href, body} |
| scrape_server | scrape_instagram | handle | followers, posts, bio |
| scrape_server | scrape_facebook | handle | page_likes, engagement |
| scrape_server | extract_website_socials | url | ig_handle, fb_handle |
| wiki_server | wikipedia_lookup | entity | has_wikipedia, url, found_in |

**3.13 Pipeline State and Data Flow**
Show the `PipelineState` TypedDict fields and how each agent reads/writes them. Data flows: prompt_set → raw_responses → entity_features_global → web_features → unified_features.

**3.14 System Architecture Diagram**
Full diagram showing all components, connections, and data flows.

> **Writing tip:** Architecture diagrams are the most-read part of a technical report. One overview diagram + one detailed diagram per major component is the right level.

---

### 4. IMPLEMENTATION AND EVALUATION

**Purpose:** Show that you built it, prove that it works, and interpret what the results mean.

#### Implementation

**4.1 Development Environment and Stack**
- OS: Windows 10, Python 3.11
- Key libraries: LangGraph, LangChain-Groq, FastMCP, Groq SDK, Pandas, RapidFuzz
- Version control: Git

**4.2 Orchestrator Implementation**
- The `run_orchestrator()` function: goal → ReAct loop → state
- Tool registry pattern: `TOOLS` dict maps names to callables + descriptions
- History management: how the orchestrator maintains reasoning context across steps

**4.3 Model Registry and Autonomous Model Selection**
- `fits(estimated_tokens)` — how estimated prompt tokens pre-filter models before API calls
- Eliminates 413 retries after the first occurrence
- Key algorithmic decision: provider priority (Groq > OpenRouter) with per-key quota tracking

**4.4 Agent 0 Implementation**
- `agent0_generate_intents()` — structured LLM prompt with JSON output
- `agent0_generate_prompts()` — per-intent prompt generation loop
- `agent0_reflect()` — self-evaluation and regeneration trigger

**4.5 Agent 1 Implementation**
- `query_llm()` — provider-aware call with registry integration
- Entity cleaning: 5-phase pipeline (prefilter → fuzzy cluster → LLM arbitration → self-reflect → apply mapping)
- GEO metrics: `compute_entity_features_global()` — mention_rate, stability_score

**4.6 Agent 2 ReAct Loop and MCP Integration**
- `_research_all_async()` — async loop with mid-run model switching
- `_parse_row()` — validation logic (key overlap, entity name check, confidence range)
- `_save_row()` — fixed column schema normalisation

**4.7 Rate Limit and Token Capacity Management**
```
API Error
├── 413 (too large)   → learn capacity → skip for large prompts
├── 429 TPD (daily)   → mark exhausted → registry never picks again this session
├── 429 TPM (per-min) → mark blocked until retry_after → pick another model
└── 401 (invalid key) → remove from pool
```

**4.8 Data Persistence and Checkpointing**
- Progressive saving: one record per LLM call (Agent 1), one per entity (Agent 2)
- Resume logic: `done_keys` set loaded at startup, skip already-completed work
- UTF-8 BOM encoding for Excel compatibility

**4.9 Challenges and Solutions**

| Challenge | Solution |
|---|---|
| Groq TPD limits hit mid-pipeline | Dynamic registry with TPD state; OpenRouter fallback |
| 413 errors causing infinite retry loops | Token capacity learning; estimated_tokens pre-filter |
| TripAdvisor returning null data | DDG two-step: find URL → pass as startUrls to Apify actor |
| LLM hallucinating web data | Strict system prompt; `_is_valid_result()` validation |
| Inconsistent CSV columns across runs | Fixed `_CSV_COLUMNS` schema with normalisation |
| Agent 2 filter mismatch with merge | Applied same threshold (mention_count≥2 / stability≥0.05) to both sides |

#### Evaluation

**4.10 Experimental Setup**
- Domain: Tunisian restaurants
- Parameters: n_intents=4, n_variants=2, languages=["fr"], n_runs=2
- LLM providers: Groq (primary), OpenRouter (fallback)

**4.11 Dataset Summary**

| Stage | Metric | Value |
|---|---|---|
| Intent Discovery | Intents generated | 4 |
| Prompt Generation | Total prompts | ~12 |
| LLM Querying | Total responses collected | ~24 |
| Entity Extraction | Raw entities discovered | 308 |
| Entity Extraction | Filtered entities (quality threshold) | 99 |
| Web Research | Entities researched by Agent 2 | 99 |

**4.12 Agent 0 Evaluation: Prompt Set Quality**
- Self-reflection scores (avg quality score / 10)
- Intent diversity coverage
- Sample prompts table — show variety across intents

**4.13 Agent 1 Evaluation: Entity Extraction**

| Entity | Mention Rate | Stability Score | Consistency |
|---|---|---|---|
| Dar El Jeld | 0.50 | 0.50 | variable |
| Le Tunisien | 0.50 | 0.50 | variable |
| ... | ... | ... | ... |

Discuss: why are scores "unstable"? (expected with n_runs=2 — note as a limitation)

**4.14 Agent 2 Evaluation: Web Feature Coverage**

| Feature | Coverage |
|---|---|
| Google Maps rating | X/99 (X%) |
| TripAdvisor URL | X/99 (X%) |
| Instagram handle | X/99 (X%) |
| Wikipedia presence | X/99 (X%) |
| has_website | X/99 (X%) |
| Avg overall_confidence | 0.XX |

**4.15 Model Selection Behavior Analysis**
- Which models were selected per step and why
- How many TPD switches occurred during the run
- Registry state at pipeline end (models exhausted vs. available)

**4.16 GEO Metrics: Brand Visibility Ranking**
Present the final GEO ranking — a combined score for each entity:
```
GEO_Score = α × mention_rate + β × data_source_count + γ × overall_confidence
```
Visualize as a bar chart (Figure X).

#### Discussion

**4.17 Interpretation of Results**
- What do the mention rates tell us about LLM brand awareness of Tunisian restaurants?
- Why do certain entities (Dar El Jeld, etc.) appear more often?
- Is web presence correlated with LLM mention frequency? (preliminary evidence)

**4.18 Contribution to the GEO Field**
1. First automated GEO measurement pipeline for the North African restaurant domain
2. Agentic architecture where model selection and execution order are fully autonomous
3. Dynamic multi-provider model registry with learned token capacity
4. Validated MCP-based web presence extraction framework

**4.19 Agentic Architecture: Benefits and Trade-offs**

| Benefit | Trade-off |
|---|---|
| Adapts to unexpected states | Harder to debug and predict |
| No manual pipeline updates | LLM reasoning can be inconsistent |
| Self-corrects on model failures | Slightly higher latency per step |
| Extensible (new tools = new capabilities) | Requires strong system prompt engineering |

**4.20 Limitations and Ethical Considerations**
- Small dataset (n_runs=2 causes high instability scores)
- Groq free tier limits model diversity and scale
- Social media scraping requires paid Apify account
- LLM hallucination risk in web data (mitigated but not eliminated)
- Data privacy: only publicly available web data used
- Research bias: results reflect LLMs' training data biases about the region

> **Writing tip:** Never paste raw code into the report body. Use pseudocode, flowcharts, or short snippets (max 10 lines). Put full code in appendices.

---

### 5. CONCLUSION AND FUTURE WORK

**Purpose:** Close the report cleanly. Leave the reader with a clear sense of what was achieved.

**5.1 Summary of Contributions**
3–5 bullet points summarizing what was built and demonstrated.

**5.2 Answer to Research Objectives**
Return to each objective from Section 1.3 and explicitly state whether it was met:
> "Objective 1 — Design an automated pipeline to measure LLM brand visibility: **Met**. The system generates prompts, queries LLMs across multiple runs, extracts entities, and computes mention_rate and stability_score automatically."

**5.3 Future Work and Extensions**

| Extension | Description |
|---|---|
| Arabic language support | Add Arabic prompts and Arabic-tuned LLMs |
| Longitudinal tracking | Re-run pipeline monthly to track visibility trends |
| GEO optimization recommendations | Generate actionable advice for businesses |
| Fine-tuned entity extraction | Train a NER model on the collected responses |
| Real-time dashboard | Streamlit UI showing GEO scores live |
| Statistical validation | Increase n_runs to 10+ for reliable stability scores |

> **Writing tip:** Conclusion should be 2–3 pages maximum. No new information — restate, synthesize, and close.

---

## REFERENCES (IEEE Style)

```
[1]  S. Aggarwal et al., "GEO: Generative Engine Optimization," arXiv:2311.09735, 2023.
[2]  S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023.
[3]  A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
[4]  T. Brown et al., "Language Models are Few-Shot Learners," NeurIPS, 2020.
[5]  A. Mallen et al., "When Not to Trust Language Models," ACL, 2023.
[6]  S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2020.
[7]  LangChain, "LangGraph Documentation," 2024. [Online]. Available: https://langchain-ai.github.io/langgraph
[8]  Anthropic, "Model Context Protocol Specification," 2024.
[9]  Groq, "Groq API Documentation," 2024. [Online]. Available: https://console.groq.com/docs
[10] OpenRouter, "OpenRouter API Documentation," 2024. [Online]. Available: https://openrouter.ai/docs
```

---

## APPENDICES

**Appendix A — Setup and Configuration**
- `.env` file structure (with placeholders, not real keys)
- Installation commands
- Directory structure

**Appendix B — Sample Pipeline Outputs**
- Sample prompt_set CSV (5 rows)
- Sample entity_features_global CSV
- Sample web_features CSV (3 entities with real data)

**Appendix C — System Prompt Texts**
- Agent 0 intent discovery prompt
- Agent 2 research protocol prompt
- Orchestrator system prompt

**Appendix D — Glossary**

| Term | Definition |
|---|---|
| GEO | Generative Engine Optimization — optimizing content for visibility in LLM responses |
| LLM | Large Language Model — a deep learning model trained on text (e.g. GPT-4, LLaMA) |
| ReAct | Reason + Act — an agent architecture where the LLM alternates reasoning and tool use |
| MCP | Model Context Protocol — a standard for connecting LLMs to external tools via stdio |
| TPM | Tokens Per Minute — rate limit on token throughput per 60 seconds |
| TPD | Tokens Per Day — rate limit on total tokens per 24 hours |
| NER | Named Entity Recognition — the task of identifying named entities in text |
| Mention Rate | Fraction of LLM runs in which a given entity is mentioned |
| Stability Score | Consistency of mention rate across different prompts and runs |

---

## GENERAL WRITING TIPS

1. **Write for a jury that knows AI but not your specific project.** Never assume they read your code.
2. **Every figure and table needs a number, caption, and reference in the text.**
3. **Past tense for what you did** ("The system was designed..."), **present tense for results** ("Table 3 shows...").
4. **One idea per paragraph.** First sentence = topic sentence. Last sentence = transition.
5. **Avoid vague language:** "good results", "fast performance" — always quantify.
6. **Citation placement:** cite at the end of the sentence before the period: "...as shown in prior work [3]."
7. **Target length:** 50–70 pages excluding appendices. Quality over quantity.
8. **Defense preparation:** Introduction, Architecture diagram, Results table, and Conclusion are the 4 things examiners read first.
