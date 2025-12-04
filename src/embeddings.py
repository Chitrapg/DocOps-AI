#src/embeddings.py

from sentence_transformers import SentenceTransformer
from app.config import settings

# local model: sentence-transformers/all-MiniLM-L6-v2 -> 384-dim
_MODEL = None

def load_local_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL

def embed_text(text: str) -> list:
    if settings.EMBEDDING_PROVIDER == 'openai' and settings.OPENAI_API_KEY:
        # If you switch to OpenAI later, implement call to get OpenAI embeddings here.
        raise NotImplementedError("OpenAI provider selected but not implemented in this file.")
    model = load_local_model()
    vec = model.encode(text, show_progress_bar=False)
    return vec.tolist()

def embed_texts(texts: list) -> list:
    model = load_local_model()
    vecs = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]
