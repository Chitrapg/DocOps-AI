CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- Set embedding dimension to 384 (sentence-transformers/all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  filename text NOT NULL,
  chunk_index int NOT NULL,
  text text NOT NULL,
  metadata jsonb,
  embedding vector(384),
  created_at timestamptz DEFAULT now()
);

-- ivfflat index for similarity search (tune lists for your data)
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS generations (
  id text PRIMARY KEY,
  prompt text,
  grounding jsonb,
  output text,
  metadata jsonb,
  created_at timestamptz DEFAULT now()
);
