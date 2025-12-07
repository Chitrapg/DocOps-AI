# app/vectorstore.py
import os
import urllib.parse
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine

# langchain-postgres (integration) provides PGVector
try:
    from langchain_postgres import PGVector
except Exception as e:
    raise ImportError(
        "langchain-postgres import failed. Install it with: pip install langchain-postgres"
    ) from e

# Document class used by langchain-postgres examples
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        # minimal shim
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

class LocalEmbeddingsAdapter:
    """
    Adapter to match the embedding interface expected by PGVector:
    - embed_documents(list[str]) -> list[list[float]]
    - embed_query(str) -> list[float]
    The embedder passed to PGVectorStore should provide embed_text(list[str]) and embed_query(str).
    """
    def __init__(self, embedder):
        self._embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed_text(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.embed_query(text)

def _build_connection_strings() -> Dict[str, str]:
    """
    Build and return:
      - 'pgvector_connection': uri for langchain-postgres (postgresql+psycopg://user:pass@host:port/db?params)
      - 'sqlalchemy_url': uri for SQLAlchemy (postgresql+psycopg://user:pass@host:port/db?params)
    Allow overriding by setting PG_CONNECTION_URI (used for both).
    Supports optional PGSSLMODE or PGSSLARGS to append ssl params.
    """
    # If user provides full connection URI, prefer it
    env_uri = os.getenv("PG_CONNECTION_URI")
    if env_uri:
        # Accept a full URI. We still return SQLALCHEMY & PGVECTOR variants (same)
        return {"pgvector_connection": env_uri, "sqlalchemy_url": env_uri}

    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DATABASE")

    if not all([user, password, db]):
        raise RuntimeError("PGUSER, PGPASSWORD and PGDATABASE must be set in environment (.env)")

    # URL-escape username/password
    user_enc = urllib.parse.quote_plus(user)
    pass_enc = urllib.parse.quote_plus(password)

    # Build optional query params (SSL etc.)
    query_params = {}
    sslmode = os.getenv("PGSSLMODE")
    if sslmode:
        query_params["sslmode"] = sslmode
    # Allow PGSSLARGS like "sslrootcert=/path/to/ca.pem&sslcert=..."
    sslargs = os.getenv("PGSSLARGS")
    if sslargs:
        # parse user-provided args and merge
        try:
            parsed = dict(urllib.parse.parse_qsl(sslargs))
            query_params.update(parsed)
        except Exception:
            # fallback to raw param append
            pass

    query_string = ""
    if query_params:
        query_string = "?" + urllib.parse.urlencode(query_params)

    # PGVector expects psycopg (psycopg3) style scheme in docs:
    pgvector_connection = f"postgresql+psycopg://{user_enc}:{pass_enc}@{host}:{port}/{db}{query_string}"
    sqlalchemy_url = pgvector_connection  # SQLAlchemy can use same scheme when psycopg3 is installed

    return {"pgvector_connection": pgvector_connection, "sqlalchemy_url": sqlalchemy_url}

class PGVectorStore:
    def __init__(self, embedder, table_name: str = "documents", use_jsonb: bool = True, create_engine_flag: bool = False):
        """
        embedder: an object with embed_text(list[str]) and embed_query(str).
        table_name: collection/table name to use in postgres (collection_name in PGVector).
        use_jsonb: whether to store metadata as jsonb (PGVector supports use_jsonb option).
        create_engine_flag: if True, builds a SQLAlchemy engine and attaches to self.engine (optional).
        """
        conns = _build_connection_strings()
        pgvector_conn = conns["pgvector_connection"]
        sqlalchemy_url = conns["sqlalchemy_url"]

        self.table = table_name
        self.embedder = LocalEmbeddingsAdapter(embedder)

        # Optionally create engine for direct SQL interactions (useful for migrations or manual inserts)
        self.engine = None
        if create_engine_flag:
            try:
                self.engine = create_engine(sqlalchemy_url, future=True)
            except Exception as e:
                # not fatal — library availability or driver issues may occur
                self.engine = None

        # Instantiate PGVector using the connection string
        try:
            # docs show signature: PGVector(embeddings=..., collection_name=..., connection=..., use_jsonb=...)
            self.vs = PGVector(
                embeddings=self.embedder,
                collection_name=self.table,
                connection=pgvector_conn,
                use_jsonb=use_jsonb,
            )
        except TypeError as e:
            # fallback attempt: some versions accept 'connection_string' or different param names
            tried = False
            try:
                self.vs = PGVector(
                    embeddings=self.embedder,
                    collection_name=self.table,
                    connection_string=pgvector_conn,  # older alias in some versions
                    use_jsonb=use_jsonb,
                )
                tried = True
            except Exception:
                pass
            if not tried:
                # Re-raise with helpful context
                raise TypeError(
                    "Failed to instantiate PGVector. The installed langchain-postgres expects a different constructor signature. "
                    "Original error: " + str(e)
                ) from e
        except Exception as e:
            # other runtime errors (db connection, missing driver, etc.)
            raise RuntimeError("Unable to instantiate PGVector: " + str(e)) from e

    def add_documents(self, docs: List[Dict[str, Any]], ids: Optional[List[Any]] = None):
        """
        docs: list of dicts with 'content' and optional 'metadata'
        ids: optional list of ids to store (langchain-postgres supports explicit ids)
        """
        lc_docs = []
        for d in docs:
            lc_docs.append(Document(page_content=d["content"], metadata=d.get("metadata", {})))

        # call likely method names; prefer add_documents
        if hasattr(self.vs, "add_documents"):
            try:
                if ids:
                    return self.vs.add_documents(lc_docs, ids=ids)
                return self.vs.add_documents(lc_docs)
            except TypeError:
                # fall through to other attempts
                pass

        if hasattr(self.vs, "add"):
            return self.vs.add(lc_docs)
        if hasattr(self.vs, "upsert"):
            return self.vs.upsert(lc_docs)

        raise RuntimeError("PGVector instance does not expose a known add method (add_documents/add/upsert).")

    def similarity_search(self, query: str, k: int = 5):
        """
        Returns list of Document-like objects from the vector store.
        """
        if hasattr(self.vs, "similarity_search"):
            return self.vs.similarity_search(query, k=k)
        if hasattr(self.vs, "search"):
            try:
                return self.vs.search(query, k=k)
            except TypeError:
                return self.vs.search(query, top_k=k)
        if hasattr(self.vs, "find"):
            return self.vs.find(query, k=k)
        raise RuntimeError("Underlying PGVector instance has no known search method (similarity_search/search/find).")
