# core/embeddings.py
"""
Unified embeddings module - singleton pattern for model loading.
Merged from app/embeddings.py and src/embeddings.py
"""
import os
from typing import List

# Set offline mode to avoid HuggingFace network calls if already cached
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from sentence_transformers import SentenceTransformer
from core.config import settings

# Singleton model cache
_MODEL_CACHE = {}
_MODEL_LOAD_ERROR = None


def _load_model(model_name: str) -> SentenceTransformer:
    """Load model with caching and offline mode support."""
    global _MODEL_LOAD_ERROR
    
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    
    if _MODEL_LOAD_ERROR is not None:
        raise _MODEL_LOAD_ERROR
    
    try:
        # Try to load from cache first (offline mode)
        model = SentenceTransformer(model_name, local_files_only=True)
        _MODEL_CACHE[model_name] = model
        return model
    except Exception:
        # Try without local_files_only (will attempt network)
        try:
            model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = model
            return model
        except Exception as e:
            _MODEL_LOAD_ERROR = RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                f"Network error or model not cached. Error: {e}"
            )
            raise _MODEL_LOAD_ERROR


class Embeddings:
    """
    Unified embeddings class with singleton model caching.
    Compatible with both app/ and src/ usage patterns.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBED_MODEL_NAME
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model on first access."""
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model
    
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (batch)."""
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embs]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_text([text])[0]


# Convenience functions for simple usage
def embed_text(text: str) -> List[float]:
    """Embed a single text string."""
    if settings.EMBEDDING_PROVIDER == 'openai' and settings.OPENAI_API_KEY:
        raise NotImplementedError("OpenAI embedding provider not implemented.")
    model = _load_model(settings.EMBED_MODEL_NAME)
    vec = model.encode(text, show_progress_bar=False)
    return vec.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts (batch)."""
    model = _load_model(settings.EMBED_MODEL_NAME)
    vecs = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]


# Singleton instance for shared use
_embeddings_instance = None


def get_embeddings() -> Embeddings:
    """Get singleton Embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = Embeddings()
    return _embeddings_instance
