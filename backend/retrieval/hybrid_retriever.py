"""
hybrid_retriever.py
────────────────────
Orchestrates graph + vector retrieval with weighted score fusion.

Retrieval strategies:
  VECTOR_ONLY   → semantic similarity (fast, good for open questions)
  GRAPH_ONLY    → Cypher + graph traversal (precise, structured queries)
  HYBRID        → both + cross-encoder reranking (best quality)
  AUTO          → route automatically based on query type
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from backend.graph.graph_retriever import GraphRetriever, GraphResult
from backend.indexing.vector_indexer import embed_query, vector_search
from backend.retrieval.reranker import Reranker, RankedResult
from backend.config import get_settings
from backend.utils import logger


class RetrievalStrategy(str, Enum):
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY  = "graph_only"
    HYBRID      = "hybrid"
    AUTO        = "auto"


class HybridRetriever:
    """
    Combines ChromaDB vector search + Neo4j graph retrieval
    with cross-encoder reranking.
    """

    def __init__(self):
        self._graph = GraphRetriever()
        self._reranker = Reranker()

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.AUTO,
        top_k: int | None = None,
    ) -> list[RankedResult]:
        cfg = get_settings()
        top_k = top_k or cfg.retrieval_top_k

        # Auto-route based on query characteristics
        if strategy == RetrievalStrategy.AUTO:
            strategy = self._auto_route(query)
        logger.info(f"Retrieval strategy: {strategy.value}")

        vector_results: list[dict] = []
        graph_results: list[dict] = []

        # ── Vector retrieval ─────────────────────────────────────────────────
        if strategy in (RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.HYBRID):
            q_emb = embed_query(query)
            raw = vector_search(q_emb, top_k=top_k * 2)
            vector_results = raw

        # ── Graph retrieval ──────────────────────────────────────────────────
        if strategy in (RetrievalStrategy.GRAPH_ONLY, RetrievalStrategy.HYBRID):
            g_results: list[GraphResult] = self._graph.retrieve(query, top_k=top_k * 2)
            graph_results = [
                {
                    "text": r.text,
                    "score": r.score,
                    "source": r.source,
                    "page": r.page,
                    "metadata": r.metadata,
                }
                for r in g_results
            ]

        # ── Rerank combined results ───────────────────────────────────────────
        if not vector_results and not graph_results:
            logger.warning("No results from either retriever")
            return []

        ranked = self._reranker.rerank_mixed(
            query, vector_results, graph_results, top_k=top_k
        )
        logger.info(f"Retrieved {len(ranked)} results after reranking")
        return ranked

    # ── Auto routing logic ────────────────────────────────────────────────────
    @staticmethod
    def _auto_route(query: str) -> RetrievalStrategy:
        q = query.lower()

        # Structured / factual → graph is more precise
        structured_signals = [
            "what is", "how many", "list", "give me",
            "specification", "spec", "dimension", "weight",
            "horsepower", "engine", "fuel", "recall",
            "dtc", "diagnostic", "fault code", "when was",
        ]
        # Open-ended / conceptual → vector is better
        semantic_signals = [
            "explain", "describe", "how does", "why",
            "compare", "difference", "advantage", "disadvantage",
            "best", "recommend", "tell me about",
        ]

        struct_score = sum(1 for s in structured_signals if s in q)
        sem_score = sum(1 for s in semantic_signals if s in q)

        if struct_score > sem_score:
            return RetrievalStrategy.HYBRID   # graph + vector for structured
        elif sem_score > struct_score:
            return RetrievalStrategy.VECTOR_ONLY
        else:
            return RetrievalStrategy.HYBRID    # default: hybrid


def format_context(results: list[RankedResult], max_tokens: int = 3000) -> str:
    """Format retrieved results into a context string for the LLM."""
    parts = []
    total_chars = 0
    limit = max_tokens * 4   # rough char estimate

    for i, r in enumerate(results, 1):
        entry = (
            f"[Source {i} | {r.source} | Score: {r.score:.3f}]\n"
            f"{r.text}\n"
        )
        if total_chars + len(entry) > limit:
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n---\n".join(parts)
