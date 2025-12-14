# core/db/postgres.py - Moved from src/db.py
import psycopg2
import psycopg2.extras
from app.config import settings
import json

DSN = {
    'host': settings.PG_HOST,
    'port': settings.PG_PORT,
    'dbname': settings.PG_DB,
    'user': settings.PG_USER,
    'password': settings.PG_PASSWORD
}

def get_conn():
    return psycopg2.connect(**DSN)

def _to_vector_literal(vec):
    """
    Convert a Python list of floats to a pgvector literal string like:
      '[0.123,0.456,0.789]'
    This string can be cast to pg vector with '...::vector' in SQL.
    """
    # Ensure floats, compact representation (no trailing spaces)
    return '[' + ','.join((str(float(x)) for x in vec)) + ']'

def insert_document_chunk(filename: str, chunk_index: int, text: str, embedding: list, metadata: dict = None):
    """
    Stores a document chunk. Embedding is stored using a literal cast to vector.
    """
    # convert embedding to vector literal
    vec_lit = _to_vector_literal(embedding)
    sql = """
    INSERT INTO documents (filename, chunk_index, text, metadata, embedding)
    VALUES (%s, %s, %s, %s, %s::vector)
    RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                filename,
                chunk_index,
                text,
                json.dumps(metadata or {}),
                vec_lit
            ))
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None

def similarity_search(query_embedding: list, top_k: int = 5):
    """
    Performs similarity search using pgvector operator.
    Uses embedding <#> <vector> operator (distance) and orders ascending.
    We cast the RHS parameter to vector using ::vector and pass a vector literal string.
    """
    vec_lit = _to_vector_literal(query_embedding)
    sql = """
    SELECT id, filename, chunk_index, text, metadata, 1 - (embedding <#> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <#> %s::vector
    LIMIT %s
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (vec_lit, vec_lit, top_k))
            return cur.fetchall()

def store_generation(gen_id: str, prompt: str, grounding: list, output: str, metadata: dict = None):
    sql_create = """
    INSERT INTO generations (id, prompt, grounding, output, metadata)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET output = EXCLUDED.output
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_create, (gen_id, prompt, json.dumps(grounding), output, json.dumps(metadata or {})))
            conn.commit()
