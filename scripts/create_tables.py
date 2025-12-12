# create_tables.py  (put this in project root)
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root if present

PG_HOST = os.getenv('PG_HOST', 'localhost')
PG_PORT = int(os.getenv('PG_PORT', '5432'))
PG_DB = os.getenv('PG_DB', 'yourdb')
PG_USER = os.getenv('PG_USER', 'youruser')
PG_PASSWORD = os.getenv('PG_PASSWORD', 'yourpassword')

DSN = {
    'host': PG_HOST,
    'port': PG_PORT,
    'dbname': PG_DB,
    'user': PG_USER,
    'password': PG_PASSWORD
}

SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  filename text NOT NULL,
  chunk_index int NOT NULL,
  text text NOT NULL,
  metadata jsonb,
  embedding vector(384),
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
  ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS generations (
  id text PRIMARY KEY,
  prompt text,
  grounding jsonb,
  output text,
  metadata jsonb,
  created_at timestamptz DEFAULT now()
);
"""

def run():
    print(f"Connecting to {PG_HOST}:{PG_PORT}/{PG_DB} as {PG_USER}")
    conn = None
    try:
        conn = psycopg2.connect(**DSN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(SQL)
        print("Tables/extensions created successfully (or already exist).")
    except Exception as e:
        print("Error while creating tables/extensions:")
        print(e)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run()
