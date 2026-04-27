"""
routes/ingest.py — Document ingestion endpoints.
"""
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.api.models import IngestResponse
from backend.pipeline.ingestion_pipeline import IngestionPipeline
from backend.config import get_settings
from backend.utils import logger

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> IngestResponse:
    """
    Upload a PDF document and trigger the full ingestion pipeline:
    PDF → chunks → embeddings → ChromaDB + Neo4j graph.
    """
    cfg = get_settings()
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    save_path = Path(cfg.data_dir) / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved upload: {save_path}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")

    # Run ingestion
    pipeline = IngestionPipeline()
    try:
        result = pipeline.run(save_path)
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message"))

    return IngestResponse(
        status=result["status"],
        file=result["file"],
        chunks=result["chunks"],
        pages=result["pages"],
        doc_id=result["doc_id"],
        message=f"Successfully indexed {result['chunks']} chunks from {result['pages']} pages",
    )


@router.post("/scan-data-dir")
async def scan_and_ingest_data_dir():
    """Ingest all PDFs found in the configured data directory."""
    cfg = get_settings()
    pipeline = IngestionPipeline()
    results = pipeline.run_directory(cfg.data_dir)
    return {"ingested": results, "total": len(results)}


@router.get("/status")
async def ingest_status():
    """List PDFs in the data directory."""
    cfg = get_settings()
    pdfs = list(Path(cfg.data_dir).glob("*.pdf"))
    return {
        "data_dir": str(cfg.data_dir),
        "pdfs": [p.name for p in pdfs],
        "count": len(pdfs),
    }
