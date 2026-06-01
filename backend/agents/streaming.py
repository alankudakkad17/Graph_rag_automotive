"""
streaming.py
─────────────
Streaming version of the Graph-RAG pipeline.

Strategy:
  - Stages 1–6  (classify → check_relevance) run normally — no streaming needed
    because these are fast classification/retrieval steps
  - Stage 7  (generation) streams tokens via Ollama's native streaming API
  - Each token is yielded as a Server-Sent Event (SSE)

SSE event format:
  data: {"type": "stage",   "content": "Classifying query…"}
  data: {"type": "token",   "content": "The "}
  data: {"type": "token",   "content": "760i "}
  data: {"type": "metadata","content": {"pipeline": "hybrid", "confidence": 0.91}}
  data: {"type": "done",    "content": ""}
  data: {"type": "error",   "content": "error message"}
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from langchain_ollama import OllamaLLM
from backend.config import get_settings
from backend.retrieval.hybrid_retriever import HybridRetriever, RetrievalStrategy, format_context
from backend.retrieval.reranker import Reranker
from backend.agents.graph_rag_agent import (
    classify_query, expand_query, select_pipeline,
    check_relevance, AgentState
)
from backend.utils import logger


# ── SSE helper ────────────────────────────────────────────────────────────────
def _sse(event_type: str, content: str | dict) -> str:
    """Format a single SSE message."""
    payload = json.dumps({"type": event_type, "content": content})
    return f"data: {payload}\n\n"


# ── Main streaming generator ──────────────────────────────────────────────────
async def stream_agent(
    query: str,
    chat_history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """
    Async generator that yields SSE strings.
    Runs classification/retrieval synchronously, then streams generation tokens.
    """
    cfg = get_settings()
    chat_history = chat_history or []

    try:
        # ── Stage 1: Classify ────────────────────────────────────────────────
        yield _sse("stage", "🔍 Classifying query…")
        state: AgentState = {
            "original_query":   query,
            "chat_history":     chat_history,
            "query_intent":     "",
            "query_complexity": "",
            "expanded_queries": [],
            "pipeline":         "hybrid",
            "raw_results":      [],
            "reranked_results": [],
            "should_generate":  True,
            "refusal_reason":   "",
            "system_prompt":    "",
            "context":          "",
            "answer":           "",
            "sources":          [],
            "confidence":       0.0,
            "pipeline_reason":  "",
            "error":            None,
        }
        state = classify_query(state)
        yield _sse("stage", f"📋 Intent: {state['query_intent']} | Complexity: {state['query_complexity']}")

        # ── Stage 2: Expand ──────────────────────────────────────────────────
        yield _sse("stage", "🔄 Expanding query…")
        state = expand_query(state)
        yield _sse("stage", f"📝 {len(state['expanded_queries'])} query variants generated")

        # ── Stage 3: Select Pipeline ─────────────────────────────────────────
        yield _sse("stage", "⚙️ Selecting retrieval pipeline…")
        state = select_pipeline(state)
        yield _sse("stage", f"🔀 Pipeline: {state['pipeline']} — {state['pipeline_reason']}")

        # ── Stage 4: Retrieve ────────────────────────────────────────────────
        yield _sse("stage", "📚 Retrieving from knowledge base…")
        retriever = HybridRetriever()
        strategy  = RetrievalStrategy(state["pipeline"])
        all_results = []
        seen_texts:set[str] = set()

        for q in state["expanded_queries"]:
            results = retriever.retrieve(q, strategy=strategy, top_k=cfg.retrieval_top_k)
            for r in results:
                key = r.text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_results.append({
                        "text": r.text, "score": r.score,
                        "source": r.source, "page": r.page,
                        "metadata": getattr(r, "metadata", {}),
                    })

        state["raw_results"] = all_results
        yield _sse("stage", f"📄 Retrieved {len(all_results)} unique chunks")

        # ── Stage 5: Rerank ──────────────────────────────────────────────────
        yield _sse("stage", "🏆 Reranking with cross-encoder…")
        reranker = Reranker()
        ranked   = reranker.rerank(query, all_results, top_k=cfg.rerank_top_k)
        state["reranked_results"] = [
            {"text": r.text, "score": r.score, "source": r.source, "page": r.page}
            for r in ranked
        ]
        top_score = ranked[0].score if ranked else 0.0
        yield _sse("stage", f"✅ Top relevance score: {top_score:.3f}")

        # ── Stage 6: Relevance Gate ──────────────────────────────────────────
        yield _sse("stage", "🔎 Checking document relevance…")
        state = check_relevance(state)

        if not state["should_generate"]:
            reason = state["refusal_reason"]
            refusal_messages = {
                "no_results":    "❌ No relevant information found in the knowledge base.",
                "low_score":     "❌ Retrieved documents are not relevant enough to answer reliably.",
                "llm_irrelevant":"❌ Documents do not contain sufficient information for this query.",
            }
            yield _sse("refused", refusal_messages.get(reason, "❌ Unable to answer this query."))
            yield _sse("done", "")
            return

        yield _sse("stage", "✅ Documents are relevant — generating answer…")

        # ── Stage 7: Build prompt ────────────────────────────────────────────
        results  = state["reranked_results"]
        context  = format_context(
            [type("R", (), r)() for r in results],
            max_tokens=3000,
        )
        sources  = list({r["source"] for r in results})
        intent   = state.get("query_intent", "factual")

        tone_map = {
            "factual":     "Provide precise, data-driven answers with specific values.",
            "diagnostic":  "Analyze fault codes and provide systematic troubleshooting guidance.",
            "exploratory": "Give comprehensive explanations with relevant context.",
            "comparative": "Structure your response as a clear comparison with pros/cons.",
        }
        tone = tone_map.get(intent, "Provide clear, accurate automotive information.")
        confidence_hint = (
            "You have high-confidence relevant sources."
            if top_score > 0.7
            else "Sources have moderate relevance — note any uncertainty."
        )

        messages_text = ""
        for msg in chat_history[-4:]:
            role = msg.get("role", "user")
            messages_text += f"\n{role.title()}: {msg.get('content', '')}"

        system_prompt = f"""You are an expert BMW automotive assistant with deep knowledge of the 2023 BMW 7 Series.
{tone}
{confidence_hint}
Guidelines:
- Ground your answer strictly in the provided context.
- Quote specific values (hp, torque, dimensions) when available.
- Cite the source number [Source N] when referencing specific data.
- If information is not in the context, say so clearly.
Context from Knowledge Base:
{context}"""

        full_prompt = (
            f"{system_prompt}\n\n"
            f"{'Conversation history:' + messages_text if messages_text else ''}\n\n"
            f"User Question: {query}\n\nAnswer:"
        )

        # ── Stage 8: Stream tokens ───────────────────────────────────────────
        yield _sse("stage", "✍️ Generating answer…")

        llm = OllamaLLM(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0.1,
            num_ctx=4096,
        )

        # Stream tokens using LangChain's astream
        async for chunk in llm.astream(full_prompt):
            if chunk:
                yield _sse("token", chunk)

        # ── Done — send metadata ─────────────────────────────────────────────
        yield _sse("metadata", {
            "pipeline":        state["pipeline"],
            "pipeline_reason": state["pipeline_reason"],
            "confidence":      round(top_score, 3),
            "sources":         sources,
            "query_intent":    intent,
            "num_results":     len(results),
            "was_refused":     False,
        })
        yield _sse("done", "")

    except Exception as exc:
        logger.error(f"[Streaming] Error: {exc}", exc_info=True)
        yield _sse("error", str(exc))
        yield _sse("done", "")
