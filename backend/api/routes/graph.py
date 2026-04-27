"""
routes/graph.py — Graph visualization & stats endpoints.
"""
from fastapi import APIRouter, HTTPException
from backend.api.models import GraphStatsResponse, GraphDataResponse
from backend.graph.neo4j_client import run_query
from backend.indexing.vector_indexer import get_collection_stats
from backend.utils import logger

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats() -> GraphStatsResponse:
    """Return node and relationship counts from Neo4j."""
    try:
        node_labels = ["Vehicle", "Engine", "Feature", "System", "Component",
                       "Specification", "Recall", "DiagnosticCode", "Document", "Chunk"]
        rel_types = ["HAS_ENGINE", "HAS_FEATURE", "HAS_SPECIFICATION",
                     "BELONGS_TO_SYSTEM", "SOURCED_FROM", "NEXT_CHUNK",
                     "MENTIONS_VEHICLE", "MENTIONS_FEATURE"]

        nodes: dict[str, int] = {}
        for label in node_labels:
            result = run_query(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            nodes[label] = result[0]["cnt"] if result else 0

        rels: dict[str, int] = {}
        for rel in rel_types:
            result = run_query(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt")
            rels[rel] = result[0]["cnt"] if result else 0

        chroma = get_collection_stats()
        return GraphStatsResponse(
            nodes=nodes,
            relationships=rels,
            total_chunks=chroma["total_chunks"],
            collection=chroma["collection"],
        )
    except Exception as exc:
        logger.error(f"Graph stats error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/visualize", response_model=GraphDataResponse)
async def get_graph_data(limit: int = 150) -> GraphDataResponse:
    """
    Return graph data for visualization (nodes + edges).
    Suitable for feeding into Plotly / PyVis / D3.js.
    """
    try:
        # Fetch non-chunk nodes
        node_rows = run_query(
            """
            MATCH (n)
            WHERE NOT n:Chunk
            RETURN id(n) AS id,
                   labels(n)[0] AS label,
                   COALESCE(n.name, n.variant, n.title, n.code, n.attribute, n.id) AS display,
                   labels(n)[0] AS type
            LIMIT $limit
            """,
            {"limit": limit},
        )

        # Fetch relationships between non-chunk nodes
        edge_rows = run_query(
            """
            MATCH (a)-[r]->(b)
            WHERE NOT a:Chunk AND NOT b:Chunk
            RETURN id(a) AS source, id(b) AS target, type(r) AS rel_type
            LIMIT $limit
            """,
            {"limit": limit * 2},
        )

        nodes = [
            {
                "id": str(r["id"]),
                "label": r.get("display", "unknown"),
                "type": r.get("type", "Node"),
            }
            for r in node_rows
        ]
        edges = [
            {
                "source": str(r["source"]),
                "target": str(r["target"]),
                "label": r.get("rel_type", ""),
            }
            for r in edge_rows
        ]
        return GraphDataResponse(nodes=nodes, edges=edges)

    except Exception as exc:
        logger.error(f"Graph visualize error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/vehicles")
async def get_vehicles():
    """Return all vehicle nodes with their features and engines."""
    try:
        rows = run_query(
            """
            MATCH (v:Vehicle)
            OPTIONAL MATCH (v)-[:HAS_ENGINE]->(e:Engine)
            OPTIONAL MATCH (v)-[:HAS_FEATURE]->(f:Feature)
            RETURN v.variant AS variant, v.year AS year, v.drivetrain AS drivetrain,
                   e.type AS engine, e.horsepower AS hp, e.torque AS torque,
                   collect(DISTINCT f.name)[..5] AS top_features
            LIMIT 20
            """
        )
        return {"vehicles": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/cypher")
async def run_custom_cypher(q: str):
    """Execute a read-only Cypher query (for debugging)."""
    if any(w in q.upper() for w in ["DELETE", "DETACH", "DROP", "CREATE", "MERGE", "SET"]):
        raise HTTPException(status_code=403, detail="Only read queries allowed")
    try:
        results = run_query(q)
        return {"results": results[:50]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
