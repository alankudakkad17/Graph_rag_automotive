"""
pdf_processor.py
────────────────
Loads PDF files (BMW manuals, spec sheets, service docs) and splits them
into overlapping text chunks ready for embedding and graph ingestion.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.utils import logger


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    page: int
    chunk_idx: int
    doc_id: str
    doc_title: str
    doc_source: str
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    """
    Reads PDF → extracts clean text per page → splits into chunks.
    Handles multi-column layouts common in BMW specification sheets.
    """

    def __init__(self):
        cfg = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ── Public API ────────────────────────────────────────────────────────────
    def process_file(self, pdf_path: str | Path) -> list[TextChunk]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = self._make_id(str(path))
        doc_title = path.stem.replace("-", " ").replace("_", " ").title()
        logger.info(f"Processing PDF: {path.name}  (doc_id={doc_id})")

        pages_text = self._extract_pages(path)
        chunks = list(self._make_chunks(pages_text, doc_id, doc_title, str(path)))
        logger.info(f"  → {len(chunks)} chunks from {len(pages_text)} pages")
        return chunks

    def process_directory(self, directory: str | Path) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for pdf in Path(directory).glob("**/*.pdf"):
            try:
                chunks.extend(self.process_file(pdf))
            except Exception as exc:
                logger.error(f"Failed to process {pdf}: {exc}")
        return chunks

    # ── PDF text extraction ───────────────────────────────────────────────────
    def _extract_pages(self, path: Path) -> list[tuple[int, str]]:
        """Returns list of (page_number_1based, clean_text)."""
        pages: list[tuple[int, str]] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                cleaned = self._clean_text(raw)
                if len(cleaned.strip()) > 30:   # skip near-empty pages
                    pages.append((i, cleaned))
        return pages

    @staticmethod
    def _clean_text(text: str) -> str:
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        # Remove page headers/footers patterns common in BMW docs
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"BMW\s+Group\s+.*?\n", "", text, flags=re.IGNORECASE)
        return text.strip()

    # ── Chunking ─────────────────────────────────────────────────────────────
    def _make_chunks(
        self,
        pages: list[tuple[int, str]],
        doc_id: str,
        doc_title: str,
        doc_source: str,
    ) -> Iterator[TextChunk]:
        global_idx = 0
        for page_num, page_text in pages:
            # Prefix chunks with page context
            page_chunks = self._splitter.split_text(page_text)
            for local_idx, chunk_text in enumerate(page_chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{local_idx}"
                yield TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    page=page_num,
                    chunk_idx=global_idx,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_source=doc_source,
                    metadata={
                        "page": page_num,
                        "local_chunk_idx": local_idx,
                        "doc_title": doc_title,
                    },
                )
                global_idx += 1

    @staticmethod
    def _make_id(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
