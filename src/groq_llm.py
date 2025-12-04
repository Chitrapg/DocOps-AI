# src/groq_llm.py
import os
import time
import requests
from typing import Optional, Any, Dict
from app.config import settings

DEFAULT_TIMEOUT = 120
RETRIES = 3
BACKOFF = 1.5

class GroqClient:
    """
    Robust Groq HTTP client that supports:
      - OpenAI-style completions endpoint: /openai/v1/completions  (uses 'prompt')
      - OpenAI-style chat endpoint:        /openai/v1/chat/completions (uses 'messages')
    It auto-detects which payload to send based on GROQ_API_URL path.
    """

    def __init__(self):
        if not settings.GROQ_API_KEY or not settings.GROQ_API_URL:
            raise ValueError("GROQ_API_KEY and GROQ_API_URL must be set in env")
        self.api_key = settings.GROQ_API_KEY
        self.url = settings.GROQ_API_URL.rstrip("/")
        self.model = settings.GROQ_MODEL

        # Detect endpoint type by URL path
        lower = self.url.lower()
        if lower.endswith("/chat/completions"):
            self.mode = "chat"
        elif lower.endswith("/completions"):
            self.mode = "completions"
        else:
            # default: try chat style first
            self.mode = "chat"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _post(self, payload: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """
        Post with retries and backoff. Let requests use HTTP(S)_PROXY env vars if present.
        Raises requests.RequestException on failure.
        """
        last_exc = None
        for attempt in range(1, RETRIES + 1):
            try:
                resp = requests.post(self.url, headers=self._headers(), json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                # For DNS/name resolution specifically, requests raises a ConnectionError wrapping urllib3 NameResolutionError.
                # We retry a few times (useful for transient network).
                if attempt < RETRIES:
                    time.sleep(BACKOFF ** attempt)
                    continue
                # final raise
                raise

        # unreachable but for static typing:
        raise last_exc

    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.0) -> str:
        """
        Sends the prompt to the configured endpoint and returns the text content.
        For chat endpoints, we wrap prompt as a single user message.
        For completions endpoints, we use 'prompt'.
        Parsers handle common response shapes:
         - {'choices': [{'message': {'content': ...}}]}  (chat)
         - {'choices': [{'text': "..."}]}               (completion)
         - other shapes will be stringified for debugging
        """
        # Build payload depending on mode
        if self.mode == "chat":
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        else:
            # completions
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

        # Send request
        try:
            data = self._post(payload)
        except requests.RequestException as e:
            # Re-raise richer message for the caller to display
            raise ConnectionError(f"LLM call failed: {str(e)}") from e

        # Parse response
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices and isinstance(choices, list):
                first = choices[0]
                # Chat style: choice -> message -> content
                if isinstance(first, dict):
                    # OpenAI chat shape: {"message": {"role":"assistant","content":"..."}}
                    msg = first.get("message") or first.get("delta")
                    if isinstance(msg, dict) and ("content" in msg):
                        return msg.get("content", "")
                    # Another chat shape: {"message": {"content": "..."}}
                    if "text" in first:
                        return first.get("text", "")
                    # Some providers return {"message": {"content": "..."} } at top-level in different keys
                    # Try nested lookups safe-guarded:
                    if "content" in first:
                        return first.get("content", "")
                    # Sometimes first is a string-like
                    if isinstance(first, str):
                        return first
            # Fallback: some providers return 'result' or nested 'data'
            # try common keys
            if "data" in data and isinstance(data["data"], list) and data["data"] and isinstance(data["data"][0], dict):
                # example: {data:[{text:...}]}
                t0 = data["data"][0]
                for k in ("text", "content"):
                    if k in t0:
                        return t0[k]
            # If nothing matched, return full JSON as string for debugging
            return str(data)
        # non-dict -> return stringified
        return str(data)
