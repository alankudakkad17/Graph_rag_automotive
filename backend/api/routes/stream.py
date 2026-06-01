"""
routes/stream.py — Streaming query endpoint via Server-Sent Events (SSE).

SSE event types emitted:
  stage    → pipeline progress update ("Classifying query…")
  token    → one LLM output token
  metadata → final pipeline stats (confidence, sources, etc.)
  refused  → query was refused by the relevance gate
  error    → unexpected error occurred
  done     → stream is complete
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agents.streaming import stream_agent
from backend.utils import logger

router = APIRouter(prefix="/stream", tags=["Streaming"])


class StreamRequest(BaseModel):
    query:        str        = Field(..., min_length=3, max_length=1000)
    chat_history: list[dict] = Field(default_factory=list)


@router.post("/query")
async def stream_query(req: StreamRequest) -> StreamingResponse:
    """
    Stream the full RAG pipeline response via Server-Sent Events.

    Clients receive events in this order:
      1. Multiple `stage` events   — progress through classify/retrieve/rerank
      2. Multiple `token` events   — LLM answer tokens as they generate
      3. One `metadata` event      — pipeline stats after generation completes
      4. One `done` event          — signals stream end

    If documents are not relevant:
      1. Multiple `stage` events
      2. One `refused` event       — explains why answer was refused
      3. One `done` event

    Example client (JavaScript):
      const es = await fetch('/stream/query', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: 'What is the horsepower of the 760i?'})
      });
      const reader = es.body.getReader();
      // read chunks and parse SSE format
    """
    logger.info(f"[Stream] Query: '{req.query[:80]}'")

    async def event_generator():
        async for event in stream_agent(req.query, req.chat_history):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",     # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )
