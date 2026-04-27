"""
ingestion_pipeline.py
──────────────────────
End-to-end ingestion: PDF → chunks → embeddings → ChromaDB + Neo4j graph.
"""
from __future__ import annotations

from pathlib import Path

from backend.graph.neo4j_client import bootstrap_schema
from backend.indexing.pdf_processor import PDFProcessor
from backend.indexing.graph_builder import GraphBuilder
from backend.indexing.vector_indexer import embed_texts, index_chunks
from backend.utils import logger


class IngestionPipeline:
    def __init__(self):
        self._processor = PDFProcessor()
        self._builder = GraphBuilder()

    def run(self, pdf_path: str | Path) -> dict:
        """
        Full ingestion pipeline for a single PDF.
        Returns summary stats.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"=== Ingestion started: {pdf_path.name} ===")

        # Step 1: Bootstrap Neo4j schema
        bootstrap_schema()

        # Step 2: Parse PDF → chunks
        logger.info("Step 1/4 — Parsing PDF …")
        chunks = self._processor.process_file(pdf_path)
        if not chunks:
            return {"status": "error", "message": "No text extracted from PDF"}

        # Step 3: Generate embeddings
        logger.info(f"Step 2/4 — Generating embeddings for {len(chunks)} chunks …")
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        # Step 4: Index into ChromaDB
        logger.info("Step 3/4 — Indexing into ChromaDB …")
        index_chunks(chunks, embeddings)

        # Step 5: Build Neo4j graph
        logger.info("Step 4/4 — Building Neo4j knowledge graph …")
        self._builder.build_from_chunks(chunks, embeddings)

        summary = {
            "status": "success",
            "file": pdf_path.name,
            "chunks": len(chunks),
            "pages": max(c.page for c in chunks),
            "doc_id": chunks[0].doc_id,
        }
        logger.info(f"=== Ingestion complete: {summary} ===")
        return summary

    def run_directory(self, directory: str | Path) -> list[dict]:
        results = []
        for pdf in Path(directory).glob("**/*.pdf"):
            results.append(self.run(pdf))
        return results
