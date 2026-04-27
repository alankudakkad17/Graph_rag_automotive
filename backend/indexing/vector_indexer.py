"""
vector_indexer.py
──────────────────
Manages ChromaDB vector store and generates embeddings using
sentence-transformers (100% free, runs locally, no API key).
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from backend.config import get_settings
from backend.indexing.pdf_processor import TextChunk
from backend.utils import logger

_embed_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "automotive_chunks"


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        cfg = get_settings()
        logger.info(f"Loading embedding model: {cfg.hf_embed_model}")
        _embed_model = SentenceTransformer(cfg.hf_embed_model)
    return _embed_model


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        cfg = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=str(cfg.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready "
                    f"({_collection.count()} docs)")
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = _get_embed_model()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False,
                           normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_embed_model()
    vec = model.encode([text], normalize_embeddings=True)
    return vec[0].tolist()


def index_chunks(chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
    """Add chunks to ChromaDB with their pre-computed embeddings."""
    col = _get_collection()
    batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

    for chunk, emb in zip(chunks, embeddings):
        batch_ids.append(chunk.chunk_id)
        batch_docs.append(chunk.text)
        batch_embs.append(emb)
        batch_metas.append({
            "page": chunk.page,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "chunk_idx": chunk.chunk_idx,
        })

        if len(batch_ids) >= 100:
            col.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_metas,
            )
            batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

    if batch_ids:
        col.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas,
        )
    logger.info(f"ChromaDB: indexed {len(chunks)} chunks")


def vector_search(
    query_embedding: list[float],
    top_k: int = 10,
    doc_filter: str | None = None,
) -> list[dict]:
    """
    Semantic similarity search.
    Returns list of {text, score, page, doc_id, doc_title}.
    """
    col = _get_collection()
    where = {"doc_id": doc_filter} if doc_filter else None
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": doc,
            "score": float(1 - dist),   # cosine distance → similarity
            "page": meta.get("page", 0),
            "doc_id": meta.get("doc_id", ""),
            "doc_title": meta.get("doc_title", ""),
            "source": f"ChromaDB:p{meta.get('page', 0)}",
        })
    return output


def get_collection_stats() -> dict:
    col = _get_collection()
    return {"total_chunks": col.count(), "collection": COLLECTION_NAME}
