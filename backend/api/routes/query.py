"""
routes/query.py — Query & chat endpoints.
"""
from fastapi import APIRouter, HTTPException
from backend.api.models import QueryRequest, QueryResponse
from backend.agents.graph_rag_agent import run_agent
from backend.utils import logger

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Main RAG query endpoint.
    Routes through the full LangGraph agent pipeline:
    classify → expand → select pipeline → retrieve → rerank → optimize → generate → validate
    """
    logger.info(f"Query received: '{req.query[:80]}'")
    try:
        result = run_agent(req.query, req.chat_history)
        return QueryResponse(**result)
    except Exception as exc:
        logger.error(f"Query error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/pipelines")
async def list_pipelines():
    """List available retrieval pipelines."""
    return {
        "pipelines": [
            {
                "name": "auto",
                "description": "AI automatically selects the best pipeline",
            },
            {
                "name": "hybrid",
                "description": "Neo4j graph + ChromaDB vector with cross-encoder reranking",
            },
            {
                "name": "graph_only",
                "description": "Neo4j Cypher + graph traversal only",
            },
            {
                "name": "vector_only",
                "description": "ChromaDB semantic search only",
            },
        ]
    }
