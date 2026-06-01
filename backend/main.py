"""
main.py — FastAPI application entry point
─────────────────────────────────────────
Automotive Graph RAG System — BMW Edition
"""
from __future__ import annotations

import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import query, ingest, graph, stream
from backend.graph.neo4j_client import bootstrap_schema, close_driver
from backend.config import get_settings
from backend.utils import logger


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    logger.info("=== Automotive Graph RAG — Starting Up ===")

    # Bootstrap Neo4j schema
    try:
        bootstrap_schema()
        logger.info("✓ Neo4j schema ready")
    except Exception as exc:
        logger.warning(f"Neo4j not reachable at startup (will retry on first request): {exc}")

    # Pre-warm embedding model
    try:
        from backend.indexing.vector_indexer import embed_query
        embed_query("BMW 7 Series 2023 warm-up")
        logger.info("✓ Embedding model loaded")
    except Exception as exc:
        logger.warning(f"Embedding model warm-up failed: {exc}")

    logger.info(f"✓ FastAPI ready on http://{cfg.fastapi_host}:{cfg.fastapi_port}")
    yield

    # Shutdown
    close_driver()
    logger.info("=== Automotive Graph RAG — Shutdown ===")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="Automotive Graph RAG API",
        description=(
            "Graph-enhanced RAG system for BMW vehicles. "
            "Powered by Neo4j + ChromaDB + LangGraph + Ollama (free, local LLMs)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS (allow Gradio frontend)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(graph.router)
    app.include_router(stream.router)      # ← streaming SSE

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        cfg = get_settings()
        # Check Neo4j
        neo4j_ok = False
        try:
            from backend.graph.neo4j_client import run_query
            run_query("RETURN 1")
            neo4j_ok = True
        except Exception:
            pass

        # Check ChromaDB
        chroma_ok = False
        try:
            from backend.indexing.vector_indexer import get_collection_stats
            get_collection_stats()
            chroma_ok = True
        except Exception:
            pass

        # Check Ollama
        ollama_ok = False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{cfg.ollama_base_url}/api/tags")
                ollama_ok = resp.status_code == 200
        except Exception:
            pass

        return {
            "status": "healthy" if (neo4j_ok and chroma_ok) else "degraded",
            "neo4j": neo4j_ok,
            "chromadb": chroma_ok,
            "ollama": ollama_ok,
            "model": cfg.ollama_model,
        }

    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": "Automotive Graph RAG API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=cfg.fastapi_host,
        port=cfg.fastapi_port,
        reload=True,
        log_level="info",
    )
