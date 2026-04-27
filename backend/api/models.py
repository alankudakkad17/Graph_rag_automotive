"""
models.py — Pydantic request/response schemas for the FastAPI API.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Query ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000,
                       description="Natural-language question about BMW vehicles")
    chat_history: list[dict] = Field(default_factory=list,
                                      description="Previous conversation turns")
    pipeline: str = Field(default="auto",
                          description="auto | vector_only | graph_only | hybrid")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    pipeline: str
    pipeline_reason: str
    confidence: float
    query_intent: str
    num_results: int
    error: Optional[str] = None


# ── Ingestion ─────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    status: str
    file: str
    chunks: int
    pages: int
    doc_id: str
    message: str = ""


# ── Graph ────────────────────────────────────────────────────────────────────
class GraphStatsResponse(BaseModel):
    nodes: dict[str, int]
    relationships: dict[str, int]
    total_chunks: int
    collection: str


class GraphDataResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    neo4j: bool
    chromadb: bool
    ollama: bool
    model: str
