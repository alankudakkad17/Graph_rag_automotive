"""
graph_rag_agent.py
───────────────────
Main LangGraph agent orchestrating the full Graph-RAG pipeline.

Graph nodes (agent steps):
  1. classify_query      → determine intent & complexity
  2. expand_query        → multi-query expansion for better recall
  3. select_pipeline     → choose VECTOR / GRAPH / HYBRID dynamically
  4. retrieve            → fetch from Neo4j + ChromaDB
  5. rerank              → cross-encoder reranking
  6. check_relevance     → relevance gate — refuse if docs not relevant
  7. optimize_prompt     → build context-aware system prompt
  8. generate            → Ollama LLM generates answer
  9. validate            → numeric grounding check
  10. refuse_answer      → clean refusal when relevance gate fails

State flows:
  START → classify → expand → select → retrieve → rerank
        → check_relevance ──► [relevant]   → optimize → generate → validate → END
                          └─► [irrelevant] → refuse → END
"""
from __future__ import annotations

import json
from typing import TypedDict, Annotated, Any
import operator

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

from backend.config import get_settings
from backend.retrieval.hybrid_retriever import (
    HybridRetriever, RetrievalStrategy, format_context
)
from backend.retrieval.reranker import RankedResult
from backend.utils import logger


# ── Agent State ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Input
    original_query: str
    chat_history: list[dict]

    # Query processing
    query_intent: str           # factual | diagnostic | exploratory | comparative
    query_complexity: str       # simple | moderate | complex
    expanded_queries: list[str]

    # Retrieval
    pipeline: str               # vector_only | graph_only | hybrid
    raw_results: list[dict]
    reranked_results: list[dict]

    # Relevance gate  ← NEW
    should_generate: bool       # True = proceed to generation, False = refuse
    refusal_reason: str         # no_results | low_score | llm_irrelevant

    # Generation
    system_prompt: str
    context: str
    answer: str
    sources: list[str]

    # Metadata
    confidence: float
    pipeline_reason: str
    error: str | None


# ── LLM singleton ──────────────────────────────────────────────────────────────
def _get_llm(temperature: float = 0.1) -> OllamaLLM:
    cfg = get_settings()
    return OllamaLLM(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=temperature,
        num_ctx=4096,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Node 1 — Query Classification
# ══════════════════════════════════════════════════════════════════════════════
def classify_query(state: AgentState) -> AgentState:
    """Classify query intent and complexity."""
    query = state["original_query"]
    logger.info(f"[Agent] classify_query: '{query[:60]}…'")

    llm = _get_llm(temperature=0.0)
    prompt = f"""Classify the following automotive query:

Query: "{query}"

Respond with ONLY a JSON object (no markdown) like:
{{
  "intent": "factual|diagnostic|exploratory|comparative",
  "complexity": "simple|moderate|complex",
  "domain": "engine|safety|infotainment|dimensions|features|recall|dtc|general"
}}"""
    try:
        raw = llm.invoke(prompt)
        # Extract JSON
        import re
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = {"intent": "factual", "complexity": "moderate", "domain": "general"}
    except Exception:
        parsed = {"intent": "factual", "complexity": "moderate", "domain": "general"}

    state["query_intent"] = parsed.get("intent", "factual")
    state["query_complexity"] = parsed.get("complexity", "moderate")
    logger.info(f"  Intent={state['query_intent']}, Complexity={state['query_complexity']}")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 2 — Query Expansion
# ══════════════════════════════════════════════════════════════════════════════
def expand_query(state: AgentState) -> AgentState:
    """
    Generate 2–3 alternative phrasings of the query to improve recall.
    Simple queries skip expansion.
    """
    query = state["original_query"]

    if state.get("query_complexity") == "simple":
        state["expanded_queries"] = [query]
        return state

    llm = _get_llm(temperature=0.3)
    prompt = f"""You are an automotive expert helping to retrieve information about BMW vehicles.

Original query: "{query}"

Generate 2 alternative phrasings that capture the same intent but use different keywords.
Focus on BMW 7 Series 2023 technical context.

Respond ONLY with a JSON array of strings:
["alternative 1", "alternative 2"]"""

    try:
        raw = llm.invoke(prompt)
        import re
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        alternatives = json.loads(match.group()) if match else []
    except Exception:
        alternatives = []

    state["expanded_queries"] = [query] + alternatives[:2]
    logger.info(f"  Expanded to {len(state['expanded_queries'])} queries")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 3 — Dynamic Pipeline Selection
# ══════════════════════════════════════════════════════════════════════════════
def select_pipeline(state: AgentState) -> AgentState:
    """
    Choose retrieval pipeline based on query characteristics.
    Rules:
      • diagnostic / DTC / recall → GRAPH_ONLY (structured, precise)
      • specifications / dimensions → HYBRID
      • exploratory / comparative → VECTOR_ONLY
      • default → HYBRID
    """
    intent = state.get("query_intent", "factual")
    query_l = state["original_query"].lower()

    if any(w in query_l for w in ["dtc", "diagnostic", "recall", "fault code",
                                   "nhtsa", "defect"]):
        pipeline = RetrievalStrategy.GRAPH_ONLY
        reason = "Diagnostic/recall query → graph traversal most precise"

    elif any(w in query_l for w in ["dimension", "weight", "wheelbase", "hp",
                                     "torque", "horsepower", "engine spec",
                                     "length", "width", "capacity"]):
        pipeline = RetrievalStrategy.HYBRID
        reason = "Specification query → hybrid for structured + semantic coverage"

    elif intent in ("exploratory", "comparative"):
        pipeline = RetrievalStrategy.VECTOR_ONLY
        reason = "Exploratory/comparative query → semantic search optimal"

    elif intent == "diagnostic":
        pipeline = RetrievalStrategy.GRAPH_ONLY
        reason = "Diagnostic intent → graph-native traversal"

    else:
        pipeline = RetrievalStrategy.HYBRID
        reason = "General query → hybrid pipeline"

    state["pipeline"] = pipeline.value
    state["pipeline_reason"] = reason
    logger.info(f"  Pipeline: {pipeline.value} — {reason}")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 4 — Retrieval
# ══════════════════════════════════════════════════════════════════════════════
def retrieve(state: AgentState) -> AgentState:
    """Execute retrieval using selected pipeline and expanded queries."""
    retriever = HybridRetriever()
    cfg = get_settings()
    strategy = RetrievalStrategy(state.get("pipeline", "hybrid"))
    queries = state.get("expanded_queries", [state["original_query"]])

    all_results: list[dict] = []
    seen_texts: set[str] = set()

    for q in queries:
        results = retriever.retrieve(q, strategy=strategy,
                                     top_k=cfg.retrieval_top_k)
        for r in results:
            key = r.text[:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_results.append({
                    "text": r.text,
                    "score": r.score,
                    "source": r.source,
                    "page": r.page,
                    "metadata": r.metadata,
                })

    state["raw_results"] = all_results
    logger.info(f"  Retrieved {len(all_results)} unique chunks")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 5 — Reranking
# ══════════════════════════════════════════════════════════════════════════════
def rerank_results(state: AgentState) -> AgentState:
    """Cross-encoder reranking of all retrieved chunks."""
    from backend.retrieval.reranker import Reranker
    cfg = get_settings()
    reranker = Reranker()
    ranked = reranker.rerank(
        state["original_query"],
        state.get("raw_results", []),
        top_k=cfg.rerank_top_k,
    )
    state["reranked_results"] = [
        {"text": r.text, "score": r.score, "source": r.source, "page": r.page}
        for r in ranked
    ]
    logger.info(f"  Top reranked score: {ranked[0].score:.4f}" if ranked else "  No results")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 6 — Relevance Gate  (pre-generation guard)
# ══════════════════════════════════════════════════════════════════════════════
def check_relevance(state: AgentState) -> AgentState:
    """
    Decides whether retrieved documents are relevant enough to answer
    the query before the LLM is invoked.

    Three score zones:
      0.0 – 0.3  → REFUSE immediately   (cross-encoder confident: irrelevant)
      0.3 – 0.6  → LLM decides          (borderline: need semantic judgement)
      0.6 – 1.0  → PROCEED              (cross-encoder confident: relevant)
    """
    import re
    results = state.get("reranked_results", [])
    query   = state["original_query"]

    # ── No results at all ────────────────────────────────────────────────────
    if not results:
        logger.info("  [RelevanceGate] No results → refusing")
        state["should_generate"] = False
        state["refusal_reason"]  = "no_results"
        return state

    top_score = results[0]["score"]
    logger.info(f"  [RelevanceGate] top_score={top_score:.4f}")

    # ── Score too low — refuse without LLM call ───────────────────────────
    if top_score < 0.3:
        logger.info("  [RelevanceGate] Score < 0.3 → refusing (low_score)")
        state["should_generate"] = False
        state["refusal_reason"]  = "low_score"
        return state

    # ── Score is high — proceed without LLM call ─────────────────────────
    if top_score >= 0.6:
        logger.info("  [RelevanceGate] Score ≥ 0.6 → proceeding")
        state["should_generate"] = True
        state["refusal_reason"]  = ""
        return state

    # ── Borderline (0.3 – 0.6) — ask LLM to judge ────────────────────────
    logger.info("  [RelevanceGate] Borderline score → asking LLM")
    relevant = _llm_relevance_check(query, results[:3])
    state["should_generate"] = relevant
    state["refusal_reason"]  = "" if relevant else "llm_irrelevant"
    logger.info(f"  [RelevanceGate] LLM decision → {'proceed' if relevant else 'refuse'}")
    return state


def _llm_relevance_check(query: str, results: list[dict]) -> bool:
    """
    Lightweight LLM call: can these documents answer this question?
    Uses temperature=0.0 for deterministic YES/NO output.
    Fails open (returns True) if LLM call errors — avoids silent data loss.
    """
    context_preview = "\n".join(
        f"Document {i+1}: {r['text'][:200]}"
        for i, r in enumerate(results)
    )
    llm = _get_llm(temperature=0.0)
    prompt = f"""You are evaluating whether retrieved documents can answer a user question about BMW vehicles.

Question: "{query}"

Retrieved documents:
{context_preview}

Can these documents provide a direct and accurate answer to the question?
Respond with ONLY one word: YES or NO"""

    try:
        response = llm.invoke(prompt).strip().upper()
        return response.startswith("YES")
    except Exception as exc:
        logger.warning(f"  [RelevanceGate] LLM check failed: {exc} — failing open")
        return True   # fail open: prefer generating over silent refusal


# ══════════════════════════════════════════════════════════════════════════════
#  Node 6b — Refuse Answer  (called when relevance gate fails)
# ══════════════════════════════════════════════════════════════════════════════
def refuse_answer(state: AgentState) -> AgentState:
    """
    Returns a clean, informative refusal message instead of a
    hallucinated or uncertain answer.
    Message is personalised to the specific reason for refusal.
    """
    reason = state.get("refusal_reason", "unknown")
    query  = state["original_query"]

    refusal_messages = {
        "no_results": (
            f"I could not find any information related to **\"{query}\"** "
            f"in the BMW 7 Series knowledge base.\n\n"
            f"**Suggestions:**\n"
            f"- Try rephrasing your question with different keywords\n"
            f"- Upload additional BMW documentation via the Upload tab\n"
            f"- Check that the ingestion pipeline completed successfully"
        ),
        "low_score": (
            f"The documents retrieved for **\"{query}\"** do not appear "
            f"relevant enough to provide a reliable answer.\n\n"
            f"The information may not be present in the current knowledge base. "
            f"I will not guess — please consult official BMW documentation "
            f"or upload a more specific document."
        ),
        "llm_irrelevant": (
            f"After reviewing the retrieved documents, they do not contain "
            f"sufficient information to accurately answer **\"{query}\"**.\n\n"
            f"I will not generate an answer from unrelated context as it would "
            f"risk providing incorrect information. Please try a more specific "
            f"question or upload relevant BMW documentation."
        ),
    }

    state["answer"]     = refusal_messages.get(
        reason,
        f"Unable to answer \"{query}\" — insufficient relevant information found."
    )
    state["confidence"] = 0.0
    state["sources"]    = []
    state["context"]    = ""
    logger.info(f"  [RefuseAnswer] Refused with reason: {reason}")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 7 — Prompt Optimization
# ══════════════════════════════════════════════════════════════════════════════
def optimize_prompt(state: AgentState) -> AgentState:
    """
    Dynamically construct a system prompt tailored to:
    • Query intent
    • Retrieved context quality
    • Conversation history
    """
    intent = state.get("query_intent", "factual")
    results = state.get("reranked_results", [])
    pipeline = state.get("pipeline", "hybrid")

    # Build context string
    context = format_context(
        [type("R", (), r)() for r in results],   # quick dataclass-like objects
        max_tokens=3000,
    ) if results else "No relevant information found in the knowledge base."

    # Adapt tone / instructions by intent
    tone_map = {
        "factual":     "Provide precise, data-driven answers with specific values.",
        "diagnostic":  "Analyze fault codes and provide systematic troubleshooting guidance.",
        "exploratory": "Give comprehensive explanations with relevant context.",
        "comparative": "Structure your response as a clear comparison with pros/cons.",
    }
    tone = tone_map.get(intent, "Provide clear, accurate automotive information.")

    # Confidence hint from top rerank score
    top_score = results[0]["score"] if results else 0.0
    confidence_hint = (
        "You have high-confidence relevant sources."
        if top_score > 0.7
        else "Sources have moderate relevance — note any uncertainty."
    )

    system_prompt = f"""You are an expert BMW automotive assistant with deep knowledge of the 2023 BMW 7 Series.
You have access to official BMW specification documents, technical manuals, and service data.

{tone}

{confidence_hint}

Guidelines:
- Ground your answer strictly in the provided context.
- Quote specific values (hp, torque, dimensions) when available.
- Cite the source number [Source N] when referencing specific data.
- If information is not in the context, say so clearly.
- Use technical but accessible language.
- Pipeline used: {pipeline}

Context from Knowledge Base:
{context}"""

    state["system_prompt"] = system_prompt
    state["context"] = context
    state["confidence"] = float(top_score)

    # Extract sources
    state["sources"] = list({r["source"] for r in results})

    logger.info(f"  Prompt optimized (intent={intent}, confidence={top_score:.3f})")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 8 — Answer Generation
# ══════════════════════════════════════════════════════════════════════════════
def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer using Ollama LLM."""
    llm = _get_llm(temperature=0.1)

    # Build conversation messages
    messages_text = ""
    for msg in state.get("chat_history", [])[-4:]:   # last 4 turns
        role = msg.get("role", "user")
        messages_text += f"\n{role.title()}: {msg.get('content', '')}"

    full_prompt = (
        f"{state['system_prompt']}\n\n"
        f"{'Conversation history:' + messages_text if messages_text else ''}\n\n"
        f"User Question: {state['original_query']}\n\n"
        f"Answer:"
    )

    try:
        answer = llm.invoke(full_prompt)
        state["answer"] = answer.strip()
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        state["answer"] = (
            "I encountered an error generating the response. "
            "Please check that Ollama is running with the correct model."
        )
        state["error"] = str(exc)

    logger.info(f"  Generated answer ({len(state['answer'])} chars)")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Node 9 — Validation (Numeric Grounding Check)
# ══════════════════════════════════════════════════════════════════════════════
def validate_answer(state: AgentState) -> AgentState:
    """
    Final numeric grounding check on the generated answer.
    The relevance gate (Node 6) already blocked irrelevant-context generation,
    so this focuses only on catching numbers the LLM may have fabricated
    despite having relevant context.
    """
    answer = state.get("answer", "")
    context = state.get("context", "")

    if not context or context == "No relevant information found in the knowledge base.":
        state["answer"] = (
            "⚠️ Limited information available in the knowledge base.\n\n" + answer
        )
        state["confidence"] = 0.1
        return state

    # Check for specific numeric claims not grounded in context
    import re
    numbers_in_answer = set(re.findall(r'\b\d{3,}\b', answer))
    numbers_in_context = set(re.findall(r'\b\d{3,}\b', context))
    ungrounded = numbers_in_answer - numbers_in_context

    if len(ungrounded) > 3:
        state["answer"] = (
            "📊 Note: Some specific values in this response may require verification "
            "against official BMW documentation.\n\n" + answer
        )

    return state


# ══════════════════════════════════════════════════════════════════════════════
#  Build LangGraph
# ══════════════════════════════════════════════════════════════════════════════
def _route_relevance(state: AgentState) -> str:
    """Conditional edge router after check_relevance node."""
    return "generate" if state.get("should_generate", False) else "refuse"


def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register all nodes ────────────────────────────────────────────────────
    graph.add_node("classify_query",  classify_query)
    graph.add_node("expand_query",    expand_query)
    graph.add_node("select_pipeline", select_pipeline)
    graph.add_node("retrieve",        retrieve)
    graph.add_node("rerank",          rerank_results)
    graph.add_node("check_relevance", check_relevance)   # ← NEW
    graph.add_node("refuse",          refuse_answer)     # ← NEW
    graph.add_node("optimize_prompt", optimize_prompt)
    graph.add_node("generate",        generate_answer)
    graph.add_node("validate",        validate_answer)

    # ── Linear edges ──────────────────────────────────────────────────────────
    graph.add_edge(START,              "classify_query")
    graph.add_edge("classify_query",   "expand_query")
    graph.add_edge("expand_query",     "select_pipeline")
    graph.add_edge("select_pipeline",  "retrieve")
    graph.add_edge("retrieve",         "rerank")
    graph.add_edge("rerank",           "check_relevance")  # ← NEW

    # ── Conditional edge — relevance gate branches here ───────────────────────
    #   relevant   → optimize_prompt → generate → validate → END
    #   irrelevant → refuse → END
    graph.add_conditional_edges(
        "check_relevance",
        _route_relevance,
        {
            "generate": "optimize_prompt",
            "refuse":   "refuse",
        },
    )

    # ── Generation path ───────────────────────────────────────────────────────
    graph.add_edge("optimize_prompt", "generate")
    graph.add_edge("generate",        "validate")
    graph.add_edge("validate",        END)

    # ── Refusal path ──────────────────────────────────────────────────────────
    graph.add_edge("refuse",          END)

    return graph.compile()


# ── Singleton compiled agent ──────────────────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
        logger.info("LangGraph agent compiled ✓")
    return _agent


# ── Main interface ─────────────────────────────────────────────────────────────
def run_agent(
    query: str,
    chat_history: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run the full Graph-RAG agent pipeline.

    Returns dict with: answer, sources, pipeline, confidence, pipeline_reason
    """
    agent = get_agent()
    initial_state: AgentState = {
        "original_query":  query,
        "chat_history":    chat_history or [],
        "query_intent":    "",
        "query_complexity":"",
        "expanded_queries":[],
        "pipeline":        "hybrid",
        "raw_results":     [],
        "reranked_results":[],
        "should_generate": True,    # ← NEW
        "refusal_reason":  "",      # ← NEW
        "system_prompt":   "",
        "context":         "",
        "answer":          "",
        "sources":         [],
        "confidence":      0.0,
        "pipeline_reason": "",
        "error":           None,
    }
    final_state = agent.invoke(initial_state)
    return {
        "answer":           final_state["answer"],
        "sources":          final_state["sources"],
        "pipeline":         final_state["pipeline"],
        "confidence":       final_state["confidence"],
        "pipeline_reason":  final_state["pipeline_reason"],
        "query_intent":     final_state["query_intent"],
        "num_results":      len(final_state["reranked_results"]),
        "was_refused":      not final_state.get("should_generate", True),  # ← NEW
        "refusal_reason":   final_state.get("refusal_reason", ""),         # ← NEW
        "error":            final_state.get("error"),
    }
