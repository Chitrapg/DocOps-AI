#src/retriever.py
from src.embeddings import embed_text
from src.db import similarity_search

def retrieve_grounding_for_query(query: str, top_k: int = 6):
    q_emb = embed_text(query)
    rows = similarity_search(q_emb, top_k=top_k)
    grounding = []
    for r in rows:
        grounding.append({
            'id': r.get('id'),
            'text': r.get('text'),
            'metadata': r.get('metadata')
        })
    return grounding
