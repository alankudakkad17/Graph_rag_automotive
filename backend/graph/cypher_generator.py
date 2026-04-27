"""
cypher_generator.py
────────────────────
Uses a local Ollama LLM to translate natural-language questions into
Cypher queries against the automotive graph schema.
"""
from __future__ import annotations

import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from backend.config import get_settings
from backend.utils import logger

# ─── Schema summary injected into the prompt ──────────────────────────────────
_SCHEMA_SUMMARY = """
Node labels & key properties:
  Vehicle      (id, make, model, variant, year, body_style, drivetrain, msrp)
  Engine       (id, type, displacement, cylinders, horsepower, torque, fuel_type, transmission)
  Feature      (id, name, category, description, standard)
  System       (id, name, type, description)
  Component    (id, name, part_number, category)
  Specification(id, attribute, value, unit, category)
  Recall       (id, title, description, date, remedy)
  DiagnosticCode(id, code, description, severity, system)
  Document     (id, title, source, type, pages)
  Chunk        (id, text, page, chunk_idx, doc_id)

Relationship types:
  (Vehicle)-[:HAS_ENGINE]->(Engine)
  (Vehicle)-[:HAS_FEATURE]->(Feature)
  (Vehicle)-[:HAS_SPECIFICATION]->(Specification)
  (Vehicle)-[:SUBJECT_TO_RECALL]->(Recall)
  (Feature)-[:BELONGS_TO_SYSTEM]->(System)
  (System)-[:HAS_COMPONENT]->(Component)
  (Component)-[:HAS_DTC]->(DiagnosticCode)
  (Chunk)-[:SOURCED_FROM]->(Document)
  (Chunk)-[:MENTIONS_VEHICLE]->(Vehicle)
  (Chunk)-[:MENTIONS_FEATURE]->(Feature)
  (Chunk)-[:NEXT_CHUNK]->(Chunk)
"""

_CYPHER_PROMPT = PromptTemplate.from_template(
    """You are a Neo4j Cypher expert working with an automotive knowledge graph about BMW vehicles.

Graph Schema:
{schema}

User question: {question}

Rules:
1. Return ONLY valid Cypher — no explanation, no markdown fences.
2. Always LIMIT results to 10 unless the question asks for all.
3. Use case-insensitive regex for string matching: =~ '(?i).*value.*'
4. If the question is about specifications, search both Specification nodes and Chunk text.
5. For unclear questions, return a broad MATCH that fetches relevant Chunk nodes.
6. Never use APOC unless necessary.

Cypher query:"""
)


class CypherGenerator:
    def __init__(self):
        cfg = get_settings()
        self._llm = OllamaLLM(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0.0,
        )
        self._chain = _CYPHER_PROMPT | self._llm

    def generate(self, question: str) -> str:
        """Return a Cypher string for the given natural-language question."""
        try:
            raw: str = self._chain.invoke(
                {"schema": _SCHEMA_SUMMARY, "question": question}
            )
            cypher = self._clean(raw)
            logger.debug(f"Generated Cypher:\n{cypher}")
            return cypher
        except Exception as exc:
            logger.error(f"CypherGenerator error: {exc}")
            # Fallback: broad chunk search
            return (
                "MATCH (c:Chunk) WHERE c.text =~ '(?i).*' "
                "RETURN c.text AS text, c.page AS page LIMIT 10"
            )

    @staticmethod
    def _clean(raw: str) -> str:
        """Strip markdown fences and extra whitespace."""
        raw = re.sub(r"```(?:cypher)?", "", raw, flags=re.IGNORECASE)
        raw = raw.replace("```", "").strip()
        return raw
