"""
reranker.py
───────────
Cross-encoder reranker using sentence-transformers.
100% free, runs locally — no API key needed.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  •  Fast & accurate for passage relevance scoring
  •  Takes (query, passage) pairs → relevance score
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.utils import logger

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        cfg = get_settings()
        logger.info(f"Loading reranker: {cfg.reranker_model}")
        _model = CrossEncoder(cfg.reranker_model, max_length=512)
    return _model


@dataclass
class RankedResult:
    text: str
    score: float
    source: str = ""
    page: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Reranker:
    """
    Given a query and a list of candidate passages, returns them
    ranked by cross-encoder relevance score.
    """

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """
        candidates: list of dicts with at least {"text": str}
        Returns top_k RankedResults sorted by score descending.
        """
        if not candidates:
            return []

        cfg = get_settings()
        top_k = top_k or cfg.rerank_top_k
        model = _get_model()

        texts = [c.get("text", "") for c in candidates]
        pairs = [(query, t) for t in texts]

        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.error(f"Reranker prediction failed: {exc}")
            scores = [c.get("score", 0.5) for c in candidates]

        ranked = [
            RankedResult(
                text=cand.get("text", ""),
                score=float(score),
                source=cand.get("source", ""),
                page=cand.get("page", 0),
                metadata=cand.get("metadata", {}),
            )
            for cand, score in zip(candidates, scores)
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    def rerank_mixed(
        self,
        query: str,
        vector_results: list[dict],
        graph_results: list[dict],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Merge and rerank results from both vector and graph retrievers."""
        combined = vector_results + graph_results
        return self.rerank(query, combined, top_k)
