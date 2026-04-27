"""
api_client.py — Synchronous HTTP client for the FastAPI backend.
"""
from __future__ import annotations

import os
import httpx

BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(120.0)   # allow time for LLM inference


def query(
    question: str,
    chat_history: list[dict] | None = None,
    pipeline: str = "auto",
) -> dict:
    payload = {
        "query": question,
        "chat_history": chat_history or [],
        "pipeline": pipeline,
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(f"{BASE_URL}/query/", json=payload)
        r.raise_for_status()
        return r.json()


def upload_pdf(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(
                f"{BASE_URL}/ingest/upload",
                files={"file": (os.path.basename(file_path), f, "application/pdf")},
            )
            r.raise_for_status()
            return r.json()


def get_graph_stats() -> dict:
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{BASE_URL}/graph/stats")
        r.raise_for_status()
        return r.json()


def get_graph_data(limit: int = 100) -> dict:
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{BASE_URL}/graph/visualize", params={"limit": limit})
        r.raise_for_status()
        return r.json()


def health_check() -> dict:
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{BASE_URL}/health")
            return r.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


def scan_data_dir() -> dict:
    with httpx.Client(timeout=300) as client:
        r = client.post(f"{BASE_URL}/ingest/scan-data-dir")
        r.raise_for_status()
        return r.json()
