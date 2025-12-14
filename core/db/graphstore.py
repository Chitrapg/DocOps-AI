# core/db/graphstore.py - Moved from app/graphstore.py
import os
import re
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase, exceptions


class Neo4jStore:
    """
    Thin wrapper around the Neo4j Python driver, inspired by the Neo4jDatabase
    in your reference code but adapted to your use-case:

    - Simple `query` method for debugging / future use
    - Helpers for creating Doc / Topic nodes and MENTIONS relationships
    - Robust `create_graph_from_documents` that:
        * MERGEs nodes by a stable `id` property
        * MERGEs relationships by matching on that `id`
        * Sanitizes labels & relationship types to valid Neo4j identifiers
    """

    def __init__(self) -> None:
        # Support multiple env var names for compatibility
        url = (
            os.getenv("NEO4J_URL")
            or os.getenv("NEO4J_URL")
            or os.getenv("NEO4J_BOLT")
            or "neo4j://localhost:7687"
        )
        user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
        pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS") or os.getenv("NEO4J_PASSWORD")

        if not pwd:
            raise RuntimeError(
                "NEO4J connection info missing. "
                "Set NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD (or NEO4J_PASS) in env."
            )

        self._database: Optional[str] = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB")
        self.driver = GraphDatabase.driver(url, auth=(user, pwd))

        # Verify connectivity once at startup (similar to Neo4jDatabase)
        try:
            self.driver.verify_connectivity()
        except exceptions.ServiceUnavailable as e:
            raise RuntimeError(
                f"Could not connect to Neo4j at {url}. Please ensure the URL is correct. Original error: {e}"
            ) from e
        except exceptions.AuthError as e:
            raise RuntimeError(
                "Could not authenticate to Neo4j. "
                "Please ensure NEO4J_USER and NEO4J_PASSWORD are correct. "
                f"Original error: {e}"
            ) from e

    # -------------------------
    # Basic helpers
    # -------------------------
    def _get_session(self):
        """
        Helper to open a session, with optional database parameter.
        """
        if self._database:
            return self.driver.session(database=self._database)
        return self.driver.session()

    def test_connection(self) -> bool:
        with self._get_session() as s:
            r = s.run("RETURN 1 AS ok")
            row = r.single()
            return bool(row and row.get("ok") == 1)

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generic query method (mostly for debugging or future features).
        Returns a list of dict rows (record.data()).
        """
        params = params or {}
        with self._get_session() as s:
            try:
                result = s.run(cypher, params)
                return [record.data() for record in result]
            except exceptions.CypherSyntaxError as e:
                # basic error structure, similar spirit to reference Neo4jDatabase
                return [
                    {
                        "code": "invalid_cypher",
                        "message": f"Invalid Cypher statement: {e}",
                    }
                ]
            except Exception as e:
                return [
                    {
                        "code": "error",
                        "message": str(e),
                    }
                ]

    def close(self) -> None:
        self.driver.close()

    # -------------------------
    # Sanitizers
    # -------------------------
    @staticmethod
    def _sanitize_label(raw: str) -> str:
        """
        Convert arbitrary string into a safe Neo4j label.
        - replace non-word characters with underscore
        - collapse repeated underscores
        - trim leading/trailing underscores
        - ensure it doesn't start with a digit (prefix with 'L' if so)
        - fallback to 'Entity' if empty
        """
        if raw is None:
            return "Entity"
        s = str(raw).strip()
        s = re.sub(r"\W+", "_", s, flags=re.UNICODE)
        s = re.sub(r"_+", "_", s)
        s = s.strip("_")
        if not s:
            return "Entity"
        if re.match(r"^\d", s):
            s = "L" + s
        return s

    @staticmethod
    def _sanitize_rel(raw: str) -> str:
        """
        Convert arbitrary string into a safe Neo4j relationship type.
        Same rules as label, but uppercase, fallback to RELATED_TO.
        """
        if raw is None:
            return "RELATED_TO"
        s = str(raw).strip()
        s = re.sub(r"\W+", "_", s, flags=re.UNICODE)
        s = re.sub(r"_+", "_", s)
        s = s.strip("_")
        if not s:
            return "RELATED_TO"
        if re.match(r"^\d", s):
            s = "R" + s
        return s.upper()

    # -------------------------
    # Coarse doc/topic helpers (used by ingest.py)
    # -------------------------
    def create_doc_node(self, doc_id: str, title: str, snippet: str) -> None:
        """
        Create or update a Doc node representing a PDF.
        """
        cypher = """
        MERGE (d:Doc {id: $doc_id})
        SET d.title = $title,
            d.snippet = $snippet,
            d.created_at = coalesce(d.created_at, datetime())
        """
        with self._get_session() as s:
            s.run(cypher, {"doc_id": doc_id, "title": title, "snippet": snippet or ""})

    def create_topic_node(self, topic: str) -> None:
        """
        Create or update a Topic node by name.
        """
        cypher = """
        MERGE (t:Topic {name: $name})
        """
        with self._get_session() as s:
            s.run(cypher, {"name": topic})

    def create_mention(self, doc_id: str, topic: str) -> None:
        """
        Connect a Doc to a Topic with a MENTIONS relationship.
        """
        cypher = """
        MATCH (d:Doc {id: $doc_id})
        MATCH (t:Topic {name: $topic})
        MERGE (d)-[:MENTIONS]->(t)
        """
        with self._get_session() as s:
            s.run(cypher, {"doc_id": doc_id, "topic": topic})

    def search_nodes(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for nodes containing the keyword in their properties.
        Returns list of matching nodes with their properties.
        """
        cypher = """
        MATCH (n)
        WHERE any(key in keys(n) WHERE toString(n[key]) CONTAINS $keyword)
        RETURN labels(n) as labels, properties(n) as props
        LIMIT $limit
        """
        try:
            with self._get_session() as s:
                result = s.run(cypher, {"keyword": keyword.lower(), "limit": limit})
                nodes = []
                for record in result:
                    labels = record.get("labels", [])
                    props = record.get("props", {})
                    nodes.append({
                        "labels": labels,
                        "properties": props,
                        "text": f"{'/'.join(labels)}: {props.get('id', props.get('name', props.get('title', str(props))))}"
                    })
                return nodes
        except Exception as e:
            print(f"Node search failed: {e}")
            return []

    def search_topics(self, keyword: str, limit: int = 10) -> List[str]:
        """
        Search for Topic nodes containing the keyword.
        """
        cypher = """
        MATCH (t:Topic)
        WHERE toLower(t.name) CONTAINS $keyword
        RETURN t.name as name
        LIMIT $limit
        """
        try:
            with self._get_session() as s:
                result = s.run(cypher, {"keyword": keyword.lower(), "limit": limit})
                return [record.get("name") for record in result if record.get("name")]
        except Exception:
            return []

    def get_related_nodes(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get nodes related to a specific node.
        """
        cypher = """
        MATCH (n {id: $node_id})-[r]-(m)
        RETURN type(r) as rel_type, labels(m) as labels, properties(m) as props
        LIMIT $limit
        """
        try:
            with self._get_session() as s:
                result = s.run(cypher, {"node_id": node_id, "limit": limit})
                related = []
                for record in result:
                    related.append({
                        "relationship": record.get("rel_type"),
                        "labels": record.get("labels", []),
                        "properties": record.get("props", {})
                    })
                return related
        except Exception:
            return []

    # -------------------------
    # MAIN: Insert graph from LLM graph_documents
    # -------------------------
    def create_graph_from_documents(self, graph_documents: List[Dict[str, Any]]) -> None:
        """
        Accepts a list of graph_documents (already normalized in ingest.py) in the shape:
            {
              "nodes": [
                {"id": "n1", "label": "Person", "properties": {...}},
                ...
              ],
              "edges": [
                {"source": "n1", "target": "n2", "label": "FOUNDED", "properties": {...}},
                ...
              ]
            }

        Strategy:
          - MERGE nodes by a stable `id` property (string)
          - Use that same `id` to MATCH endpoints when creating relationships
          - Sanitize labels and relationship types to valid Neo4j identifiers

        This fixes the issue where relationships were not being created reliably.
        """
        if not graph_documents:
            return

        with self._get_session() as session:
            # 1) Create / merge all nodes first
            for gd in graph_documents:
                nodes = gd.get("nodes") or []
                for n in nodes:
                    try:
                        node_id = n.get("id")
                        if not node_id:
                            # skip nodes without a stable id
                            continue

                        raw_label = n.get("label") or n.get("type") or "Entity"
                        label = self._sanitize_label(raw_label)
                        props = n.get("properties") or {}
                        # ensure id is stored on the node
                        props = dict(props)  # copy
                        props["id"] = str(node_id)

                        cypher_node = f"""
                        MERGE (x:{label} {{id: $id}})
                        SET x += $props
                        """
                        session.run(cypher_node, {"id": str(node_id), "props": props})
                    except Exception as e:
                        print(f"Warning: failed to insert node {n}: {e}")
                        continue

            # 2) Create / merge relationships
            for gd in graph_documents:
                edges = gd.get("edges") or gd.get("relationships") or []
                for e in edges:
                    try:
                        src = e.get("source") or e.get("source_node_id") or e.get("from")
                        tgt = e.get("target") or e.get("target_node_id") or e.get("to")
                        if not src or not tgt:
                            continue

                        raw_rel = (
                            e.get("label")
                            or e.get("relation")
                            or e.get("type")
                            or e.get("relationship")
                            or "RELATED_TO"
                        )
                        rel_type = self._sanitize_rel(raw_rel)
                        props = e.get("properties") or e.get("props") or {}
                        props = dict(props)

                        # relationships are created by matching endpoints on `id`
                        cypher_rel = (
                            f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                            f"MERGE (a)-[r:{rel_type}]->(b) "
                            f"SET r += $props"
                        )
                        session.run(
                            cypher_rel,
                            {"src": str(src), "tgt": str(tgt), "props": props},
                        )
                    except Exception as e_edge:
                        print(f"Warning: failed to insert edge {e}: {e_edge}")
                        continue
