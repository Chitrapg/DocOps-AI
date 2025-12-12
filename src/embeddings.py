#src/embeddings.py
import os
from app.config import settings

# Set offline mode to avoid HuggingFace network calls if already cached
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from sentence_transformers import SentenceTransformer

# local model: sentence-transformers/all-MiniLM-L6-v2 -> 384-dim
_MODEL = None
_MODEL_LOAD_ERROR = None

def load_local_model():
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        raise _MODEL_LOAD_ERROR
    try:
        # Try to load from cache first (offline mode)
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        return _MODEL
    except Exception as e1:
        # Try without local_files_only (will attempt network)
        try:
            _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            return _MODEL
        except Exception as e2:
            _MODEL_LOAD_ERROR = RuntimeError(
                f"Failed to load embedding model. Network error or model not cached. "
                f"Error: {e2}"
            )
            raise _MODEL_LOAD_ERROR

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
