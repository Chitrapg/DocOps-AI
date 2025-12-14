# core/llm/groq_langchain.py
"""
LangChain-based Groq client - used for RAG chains and graph queries.
Moved from app/groq_client.py
"""
import os
import time
from typing import Optional

try:
    from langchain_groq import ChatGroq
except Exception as e:
    raise ImportError(
        "langchain-groq import failed. Install with: pip install langchain-groq"
    ) from e

from core.config import settings

DEFAULT_GROQ_MODEL = settings.GROQ_MODEL


class GroqLangChainClient:
    """
    LangChain ChatGroq wrapper with retry logic.
    Used for RAG synthesis and graph query chains.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        **kwargs,
    ):
        self.model = model or DEFAULT_GROQ_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        try:
            self.llm = ChatGroq(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=self.max_retries,
                timeout=None,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create ChatGroq client. Ensure GROQ_API_KEY is set. Error: {e}"
            ) from e

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using LangChain ChatGroq."""
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in environment.")

        system_prompt = system_prompt or (
            "You are a helpful assistant. Use the provided CONTEXT to answer the user's question."
        )

        messages = [
            ("system", system_prompt),
            ("human", prompt),
        ]

        last_exc = None
        for attempt in range(1, self.max_retries + 2):
            try:
                ai_msg = self.llm.invoke(messages)
                content = getattr(ai_msg, "content", None)
                if content is None:
                    content = ai_msg.get("content") if isinstance(ai_msg, dict) else str(ai_msg)
                return content
            except Exception as e:
                last_exc = e
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
                raise RuntimeError(
                    f"Failed to get response from Groq after {attempt} attempts. Error: {e}"
                ) from e


# Singleton instance
_client_instance = None


def get_groq_client(**kwargs) -> GroqLangChainClient:
    """Get or create singleton GroqLangChainClient."""
    global _client_instance
    if _client_instance is None:
        _client_instance = GroqLangChainClient(**kwargs)
    return _client_instance


# Backward compatibility alias
GroqClient = GroqLangChainClient
