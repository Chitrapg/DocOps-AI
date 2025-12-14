# core/llm/groq_http.py
"""
Raw HTTP Groq client - used for direct API calls without LangChain.
Moved from src/groq_llm.py
"""
import time
import requests
from typing import Optional, Any, Dict
from core.config import settings

DEFAULT_TIMEOUT = 120
RETRIES = 3
BACKOFF = 1.5


class GroqHTTPClient:
    """
    Raw HTTP Groq client that supports OpenAI-style endpoints.
    Used for simple text generation without LangChain overhead.
    """

    def __init__(self):
        settings.require_groq()
        self.api_key = settings.GROQ_API_KEY
        self.url = settings.GROQ_API_URL.rstrip("/")
        self.model = settings.GROQ_MODEL

        # Auto-detect endpoint type
        lower = self.url.lower()
        if lower.endswith("/chat/completions"):
            self.mode = "chat"
        elif lower.endswith("/completions"):
            self.mode = "completions"
        else:
            self.mode = "chat"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _post(self, payload: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """POST with retries and backoff."""
        last_exc = None
        for attempt in range(1, RETRIES + 1):
            try:
                resp = requests.post(
                    self.url, 
                    headers=self._headers(), 
                    json=payload, 
                    timeout=timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                if attempt < RETRIES:
                    time.sleep(BACKOFF ** attempt)
                    continue
                raise
        raise last_exc

    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.0) -> str:
        """Generate text response from Groq API."""
        if self.mode == "chat":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

        try:
            data = self._post(payload)
        except requests.RequestException as e:
            raise ConnectionError(f"LLM call failed: {e}") from e

        # Parse response
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta")
                    if isinstance(msg, dict) and "content" in msg:
                        return msg.get("content", "")
                    if "text" in first:
                        return first.get("text", "")
                    if "content" in first:
                        return first.get("content", "")
        return str(data)


# Singleton instance
_http_client_instance = None


def get_groq_http_client() -> GroqHTTPClient:
    """Get singleton GroqHTTPClient."""
    global _http_client_instance
    if _http_client_instance is None:
        _http_client_instance = GroqHTTPClient()
    return _http_client_instance


# Backward compatibility alias
GroqClient = GroqHTTPClient
