"""
neo4j_client.py
───────────────
Thread-safe singleton Neo4j driver + helper utilities.
"""
from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase, Driver
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.utils import logger
from backend.graph.automotive_schema import SCHEMA_SETUP_QUERIES

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        cfg = get_settings()
        _driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            max_connection_pool_size=50,
        )
        _driver.verify_connectivity()
        logger.info(f"Neo4j connected → {cfg.neo4j_uri}")
    return _driver


def close_driver() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


# ─── Schema bootstrap ──────────────────────────────────────────────────────────
def bootstrap_schema() -> None:
    """Create constraints and indexes if they don't exist yet."""
    driver = get_driver()
    with driver.session() as session:
        for q in SCHEMA_SETUP_QUERIES:
            try:
                session.run(q)
            except Exception as exc:
                # Vector index may not be supported on older Neo4j – skip silently
                logger.warning(f"Schema query skipped: {exc!r}")
    logger.info("Neo4j schema bootstrapped")


# ─── Generic helpers ───────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def run_query(cypher: str, params: dict | None = None) -> list[dict[str, Any]]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


def run_write_query(cypher: str, params: dict | None = None) -> None:
    driver = get_driver()
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(cypher, params or {}))


def run_write_batch(queries: list[tuple[str, dict]]) -> None:
    """Execute multiple write queries in a single transaction."""
    driver = get_driver()
    with driver.session() as session:
        def _batch(tx):
            for cypher, params in queries:
                tx.run(cypher, params)
        session.execute_write(_batch)


def node_exists(label: str, id_value: str) -> bool:
    result = run_query(
        f"MATCH (n:{label} {{id: $id}}) RETURN count(n) AS cnt",
        {"id": id_value},
    )
    return result[0]["cnt"] > 0 if result else False
