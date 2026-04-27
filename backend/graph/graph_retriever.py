"""
graph_retriever.py
──────────────────
Retrieves context from Neo4j using AI-generated Cypher queries plus
structured graph traversal patterns tailored to the automotive domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.graph.neo4j_client import run_query
from backend.graph.cypher_generator import CypherGenerator
from backend.utils import logger


@dataclass
class GraphResult:
    text: str
    source: str = ""
    page: int = 0
    score: float = 1.0
    metadata: dict = field(default_factory=dict)


class GraphRetriever:
    """
    Retrieves information from Neo4j using two strategies:
    1. AI-generated Cypher (flexible, handles complex natural-language)
    2. Template Cypher patterns (fast, reliable, domain-specific)
    """

    def __init__(self):
        self._cypher_gen = CypherGenerator()

    # ── Public interface ───────────────────────────────────────────────────────
    def retrieve(self, question: str, top_k: int = 10) -> list[GraphResult]:
        results: list[GraphResult] = []

        # Strategy 1: template-based (always reliable)
        results.extend(self._template_retrieve(question))

        # Strategy 2: LLM-generated Cypher
        results.extend(self._llm_cypher_retrieve(question))

        # Deduplicate by text
        seen: set[str] = set()
        unique: list[GraphResult] = []
        for r in results:
            key = r.text[:120]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:top_k]

    # ── LLM Cypher retrieval ───────────────────────────────────────────────────
    def _llm_cypher_retrieve(self, question: str) -> list[GraphResult]:
        try:
            cypher = self._cypher_gen.generate(question)
            rows = run_query(cypher)
            return [self._row_to_result(r) for r in rows]
        except Exception as exc:
            logger.warning(f"LLM Cypher retrieval failed: {exc}")
            return []

    # ── Template patterns ──────────────────────────────────────────────────────
    def _template_retrieve(self, question: str) -> list[GraphResult]:
        results: list[GraphResult] = []
        q_lower = question.lower()

        # Specification / dimension / performance queries
        if any(w in q_lower for w in ["dimension", "weight", "length", "width",
                                       "height", "wheelbase", "capacity",
                                       "horsepower", "hp", "torque", "range",
                                       "spec", "specification"]):
            results.extend(self._get_specifications(question))

        # Engine / powertrain queries
        if any(w in q_lower for w in ["engine", "motor", "power", "torque",
                                       "cylinder", "displacement", "turbo",
                                       "transmission", "hybrid", "electric"]):
            results.extend(self._get_engine_info())

        # Feature / technology queries
        if any(w in q_lower for w in ["feature", "technology", "assist",
                                       "safety", "adas", "lane", "camera",
                                       "sensor", "parking", "cruise",
                                       "infotainment", "idrive", "display"]):
            results.extend(self._get_features(question))

        # Recall queries
        if any(w in q_lower for w in ["recall", "safety issue", "defect", "nhtsa"]):
            results.extend(self._get_recalls())

        # Diagnostic code queries
        if any(w in q_lower for w in ["dtc", "diagnostic", "code", "fault",
                                       "error", "check engine", "p0"]):
            results.extend(self._get_dtc(question))

        # Generic chunk search
        results.extend(self._chunk_fulltext(question))

        return results

    # ── Specific query patterns ────────────────────────────────────────────────
    def _get_specifications(self, question: str) -> list[GraphResult]:
        rows = run_query(
            """
            MATCH (v:Vehicle)-[:HAS_SPECIFICATION]->(s:Specification)
            WITH v, s
            OPTIONAL MATCH (v)-[:HAS_ENGINE]->(e:Engine)
            RETURN v.variant AS variant,
                   s.attribute AS attribute, s.value AS value, s.unit AS unit,
                   s.category AS category
            ORDER BY s.category, s.attribute
            LIMIT 30
            """
        )
        texts = []
        for r in rows:
            t = (f"[Specification] {r.get('variant','')} | "
                 f"{r.get('attribute','')} = {r.get('value','')} {r.get('unit','')}")
            texts.append(GraphResult(text=t, source="Neo4j:Specification",
                                     metadata=dict(r)))
        return texts

    def _get_engine_info(self) -> list[GraphResult]:
        rows = run_query(
            """
            MATCH (v:Vehicle)-[:HAS_ENGINE]->(e:Engine)
            OPTIONAL MATCH (e)-[:ENGINE_SPEC]->(s:Specification)
            RETURN v.variant AS variant,
                   e.type AS engine_type, e.displacement AS disp,
                   e.horsepower AS hp, e.torque AS torque,
                   e.fuel_type AS fuel, e.transmission AS gearbox
            LIMIT 20
            """
        )
        results = []
        for r in rows:
            t = (f"[Engine] {r.get('variant','')} | "
                 f"{r.get('engine_type','')} {r.get('disp','')} | "
                 f"{r.get('hp','')} hp / {r.get('torque','')} lb-ft | "
                 f"{r.get('fuel','')} | {r.get('gearbox','')}")
            results.append(GraphResult(text=t, source="Neo4j:Engine", metadata=dict(r)))
        return results

    def _get_features(self, question: str) -> list[GraphResult]:
        rows = run_query(
            """
            MATCH (f:Feature)
            WHERE f.name =~ '(?i).*' OR f.description =~ '(?i).*'
            OPTIONAL MATCH (f)-[:BELONGS_TO_SYSTEM]->(sys:System)
            RETURN f.name AS name, f.category AS category,
                   f.description AS description, f.standard AS standard,
                   sys.name AS system
            LIMIT 20
            """
        )
        results = []
        for r in rows:
            std = "Standard" if r.get("standard") else "Optional"
            t = (f"[Feature] {r.get('name','')} ({r.get('category','')}) — "
                 f"{r.get('description','')} | {std} | System: {r.get('system','')}")
            results.append(GraphResult(text=t, source="Neo4j:Feature", metadata=dict(r)))
        return results

    def _get_recalls(self) -> list[GraphResult]:
        rows = run_query(
            "MATCH (v:Vehicle)-[:SUBJECT_TO_RECALL]->(r:Recall) "
            "RETURN v.variant AS variant, r.title AS title, "
            "r.description AS description, r.remedy AS remedy LIMIT 10"
        )
        results = []
        for r in rows:
            t = (f"[Recall] {r.get('variant','')} | {r.get('title','')} — "
                 f"{r.get('description','')} | Remedy: {r.get('remedy','')}")
            results.append(GraphResult(text=t, source="Neo4j:Recall", metadata=dict(r)))
        return results

    def _get_dtc(self, question: str) -> list[GraphResult]:
        rows = run_query(
            "MATCH (d:DiagnosticCode) "
            "RETURN d.code AS code, d.description AS description, "
            "d.severity AS severity, d.system AS system LIMIT 15"
        )
        results = []
        for r in rows:
            t = (f"[DTC] {r.get('code','')} | {r.get('description','')} | "
                 f"Severity: {r.get('severity','')} | System: {r.get('system','')}")
            results.append(GraphResult(text=t, source="Neo4j:DTC", metadata=dict(r)))
        return results

    def _chunk_fulltext(self, question: str) -> list[GraphResult]:
        try:
            # Use fulltext index if available
            rows = run_query(
                """
                CALL db.index.fulltext.queryNodes('chunk_text_idx', $q)
                YIELD node, score
                RETURN node.text AS text, node.page AS page,
                       node.doc_id AS doc_id, score
                LIMIT 10
                """,
                {"q": question},
            )
        except Exception:
            # Fallback: CONTAINS substring
            kw = question[:50]
            rows = run_query(
                "MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS toLower($kw) "
                "RETURN c.text AS text, c.page AS page, c.doc_id AS doc_id "
                "LIMIT 10",
                {"kw": kw},
            )
        results = []
        for r in rows:
            results.append(GraphResult(
                text=r.get("text", ""),
                source=f"Neo4j:Chunk:p{r.get('page', 0)}",
                page=r.get("page", 0),
                score=r.get("score", 0.8),
                metadata=dict(r),
            ))
        return results

    # ── Utility ───────────────────────────────────────────────────────────────
    @staticmethod
    def _row_to_result(row: dict[str, Any]) -> GraphResult:
        text = row.get("text", "") or str(row)
        return GraphResult(
            text=text,
            page=row.get("page", 0),
            source="Neo4j:LLMCypher",
            metadata=row,
        )
