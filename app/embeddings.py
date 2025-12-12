# app/embeddings.py
import os
from typing import List
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

class Embeddings:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBED_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embs]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text([text])[0]
