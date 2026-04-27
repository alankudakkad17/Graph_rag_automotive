"""
config.py — Centralised settings loaded from .env
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "automotive_rag"

    # ── Ollama LLM (free, local) ──────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    # ── HuggingFace local embeddings (fallback) ───────────────────────────────
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"

    # ── FastAPI ───────────────────────────────────────────────────────────────
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    # ── Gradio ────────────────────────────────────────────────────────────────
    gradio_host: str = "0.0.0.0"
    gradio_port: int = 7860
    fastapi_base_url: str = "http://localhost:8000"

    # ── Ingestion ─────────────────────────────────────────────────────────────
    data_dir: str = "./data/pdfs"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Pipeline thresholds ───────────────────────────────────────────────────
    graph_score_threshold: float = 0.6
    vector_score_threshold: float = 0.5
    rerank_top_k: int = 5
    retrieval_top_k: int = 10

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
