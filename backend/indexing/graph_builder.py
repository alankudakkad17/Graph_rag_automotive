"""
graph_builder.py
────────────────
Populates Neo4j from TextChunks + extracted entities.
Handles:
  • Document & Chunk nodes (with NEXT_CHUNK edges)
  • Vehicle / Engine / Feature / Specification / Recall / DTC nodes
  • All relationship edges
  • Embedding storage on Chunk nodes
"""
from __future__ import annotations

from backend.graph.neo4j_client import run_write_batch, run_write_query
from backend.indexing.pdf_processor import TextChunk
from backend.indexing.entity_extractor import EntityExtractor, ExtractedEntities
from backend.utils import logger


class GraphBuilder:
    def __init__(self):
        self._extractor = EntityExtractor()

    # ── Main entry point ──────────────────────────────────────────────────────
    def build_from_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Ingest chunks + pre-computed embeddings into Neo4j.
        embeddings[i] corresponds to chunks[i].
        """
        if not chunks:
            logger.warning("No chunks to ingest")
            return

        # 1. Create Document node
        self._upsert_document(chunks[0])

        # 2. Create Chunk nodes with embeddings
        logger.info(f"Inserting {len(chunks)} chunk nodes …")
        self._upsert_chunks(chunks, embeddings)

        # 3. Create NEXT_CHUNK edges
        self._link_consecutive_chunks(chunks)

        # 4. Extract entities from each chunk and link to graph
        logger.info("Extracting entities and building graph …")
        self._extract_and_link(chunks)

        logger.info("Graph build complete ✓")

    # ── Document ──────────────────────────────────────────────────────────────
    def _upsert_document(self, chunk: TextChunk) -> None:
        run_write_query(
            """
            MERGE (d:Document {id: $id})
            SET d.title  = $title,
                d.source = $source,
                d.type   = 'Specification'
            """,
            {"id": chunk.doc_id, "title": chunk.doc_title, "source": chunk.doc_source},
        )

    # ── Chunks ────────────────────────────────────────────────────────────────
    def _upsert_chunks(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> None:
        batch: list[tuple[str, dict]] = []
        for chunk, emb in zip(chunks, embeddings):
            batch.append((
                """
                MERGE (c:Chunk {id: $id})
                SET c.text      = $text,
                    c.page      = $page,
                    c.chunk_idx = $idx,
                    c.doc_id    = $doc_id,
                    c.embedding = $embedding
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (c)-[:SOURCED_FROM]->(d)
                """,
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "page": chunk.page,
                    "idx": chunk.chunk_idx,
                    "doc_id": chunk.doc_id,
                    "embedding": emb,
                },
            ))
            if len(batch) >= 50:     # commit in batches of 50
                run_write_batch(batch)
                batch.clear()
        if batch:
            run_write_batch(batch)

    def _link_consecutive_chunks(self, chunks: list[TextChunk]) -> None:
        batch = []
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_idx)
        for a, b in zip(sorted_chunks, sorted_chunks[1:]):
            if a.doc_id == b.doc_id:
                batch.append((
                    "MATCH (a:Chunk {id:$a}),(b:Chunk {id:$b}) MERGE (a)-[:NEXT_CHUNK]->(b)",
                    {"a": a.chunk_id, "b": b.chunk_id},
                ))
        if batch:
            run_write_batch(batch)

    # ── Entity extraction & linking ───────────────────────────────────────────
    def _extract_and_link(self, chunks: list[TextChunk]) -> None:
        all_entities = ExtractedEntities()
        for chunk in chunks:
            ents = self._extractor.extract(chunk.text)
            # Merge into global entity lists
            all_entities.vehicles.extend(ents.vehicles)
            all_entities.engines.extend(ents.engines)
            all_entities.features.extend(ents.features)
            all_entities.specifications.extend(ents.specifications)
            all_entities.recalls.extend(ents.recalls)
            all_entities.dtc_codes.extend(ents.dtc_codes)

        # Deduplicate globally
        all_entities = self._deduplicate(all_entities)

        # Create nodes
        self._upsert_vehicles(all_entities)
        self._upsert_engines(all_entities)
        self._upsert_features(all_entities)
        self._upsert_specifications(all_entities)
        self._upsert_recalls(all_entities)
        self._upsert_dtc(all_entities)

        # Link chunks → entities
        self._link_chunks_to_entities(chunks, all_entities)

    # ── Upsert nodes ──────────────────────────────────────────────────────────
    @staticmethod
    def _upsert_vehicles(e: ExtractedEntities) -> None:
        batch = []
        for v in e.vehicles:
            batch.append((
                """
                MERGE (n:Vehicle {id: $id})
                SET n.make=$make, n.model=$model, n.variant=$variant,
                    n.year=$year, n.body_style=$body_style
                """,
                v,
            ))
        if batch:
            run_write_batch(batch)

    @staticmethod
    def _upsert_engines(e: ExtractedEntities) -> None:
        batch = []
        for eng in e.engines:
            batch.append((
                """
                MERGE (n:Engine {id: $id})
                SET n.type=$type, n.displacement=$displacement,
                    n.horsepower=$horsepower, n.torque=$torque,
                    n.fuel_type=$fuel_type
                """,
                {k: v for k, v in eng.items() if v is not None},
            ))
        if batch:
            run_write_batch(batch)

    @staticmethod
    def _upsert_features(e: ExtractedEntities) -> None:
        batch = []
        for f in e.features:
            batch.append((
                """
                MERGE (n:Feature {id: $id})
                SET n.name=$name, n.category=$category,
                    n.description=$description, n.standard=$standard
                """,
                f,
            ))
        if batch:
            run_write_batch(batch)

    @staticmethod
    def _upsert_specifications(e: ExtractedEntities) -> None:
        batch = []
        for s in e.specifications:
            batch.append((
                """
                MERGE (n:Specification {id: $id})
                SET n.attribute=$attribute, n.value=$value,
                    n.unit=$unit, n.category=$category
                """,
                s,
            ))
        if batch:
            run_write_batch(batch)

    @staticmethod
    def _upsert_recalls(e: ExtractedEntities) -> None:
        batch = []
        for r in e.recalls:
            batch.append((
                """
                MERGE (n:Recall {id: $id})
                SET n.title=$title, n.description=$description,
                    n.date=$date, n.remedy=$remedy
                """,
                r,
            ))
        if batch:
            run_write_batch(batch)

    @staticmethod
    def _upsert_dtc(e: ExtractedEntities) -> None:
        batch = []
        for d in e.dtc_codes:
            batch.append((
                """
                MERGE (n:DiagnosticCode {id: $id})
                SET n.code=$code, n.description=$description,
                    n.severity=$severity, n.system=$system
                """,
                d,
            ))
        if batch:
            run_write_batch(batch)

    # ── Chunk → entity edges ───────────────────────────────────────────────────
    def _link_chunks_to_entities(
        self, chunks: list[TextChunk], entities: ExtractedEntities
    ) -> None:
        batch: list[tuple[str, dict]] = []
        for chunk in chunks:
            text_lower = chunk.text.lower()

            for v in entities.vehicles:
                if v["variant"].lower() in text_lower:
                    batch.append((
                        "MATCH (c:Chunk {id:$cid}),(v:Vehicle {id:$vid}) "
                        "MERGE (c)-[:MENTIONS_VEHICLE]->(v)",
                        {"cid": chunk.chunk_id, "vid": v["id"]},
                    ))

            for f in entities.features:
                if f["name"].lower() in text_lower:
                    batch.append((
                        "MATCH (c:Chunk {id:$cid}),(f:Feature {id:$fid}) "
                        "MERGE (c)-[:MENTIONS_FEATURE]->(f)",
                        {"cid": chunk.chunk_id, "fid": f["id"]},
                    ))

            for s in entities.specifications:
                if s["attribute"].lower() in text_lower:
                    batch.append((
                        "MATCH (c:Chunk {id:$cid}),(s:Specification {id:$sid}) "
                        "MERGE (c)-[:MENTIONS_SPEC]->(s)",
                        {"cid": chunk.chunk_id, "sid": s["id"]},
                    ))

            # Vehicle → Engine link (if both exist)
            for v in entities.vehicles:
                for eng in entities.engines:
                    batch.append((
                        "MATCH (v:Vehicle {id:$vid}),(e:Engine {id:$eid}) "
                        "MERGE (v)-[:HAS_ENGINE]->(e)",
                        {"vid": v["id"], "eid": eng["id"]},
                    ))

            # Vehicle → Feature & Spec links
            for v in entities.vehicles:
                for f in entities.features:
                    batch.append((
                        "MATCH (v:Vehicle {id:$vid}),(f:Feature {id:$fid}) "
                        "MERGE (v)-[:HAS_FEATURE]->(f)",
                        {"vid": v["id"], "fid": f["id"]},
                    ))
                for s in entities.specifications:
                    batch.append((
                        "MATCH (v:Vehicle {id:$vid}),(s:Specification {id:$sid}) "
                        "MERGE (v)-[:HAS_SPECIFICATION]->(s)",
                        {"vid": v["id"], "sid": s["id"]},
                    ))

        # Commit in batches
        for i in range(0, len(batch), 100):
            run_write_batch(batch[i : i + 100])

    # ── Deduplication ─────────────────────────────────────────────────────────
    @staticmethod
    def _deduplicate(e: ExtractedEntities) -> ExtractedEntities:
        def dedup(lst: list[dict], key: str) -> list[dict]:
            seen: set = set()
            out: list[dict] = []
            for item in lst:
                k = item.get(key, "")
                if k and k not in seen:
                    seen.add(k)
                    out.append(item)
            return out

        e.vehicles = dedup(e.vehicles, "id")
        e.engines = dedup(e.engines, "id")
        e.features = dedup(e.features, "id")
        e.specifications = dedup(e.specifications, "id")
        e.recalls = dedup(e.recalls, "id")
        e.dtc_codes = dedup(e.dtc_codes, "id")
        return e
