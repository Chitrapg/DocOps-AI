# app/groq_client.py
import os
import time
from typing import Optional

try:
    from langchain_groq import ChatGroq
except Exception as e:
    raise ImportError(
        "langchain-groq import failed. Install it with: pip install langchain-groq"
    ) from e

# Use env var to choose model; default to a sensible chat model name if you have one
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Don't raise here so app can start for dev; generate() will raise if used without key
    pass

class GroqClient:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        **kwargs,
    ):
        """
        Wrapper around langchain_groq.ChatGroq.

        model: Groq model name (see Groq docs). Default read from GROQ_MODEL env var or DEFAULT_GROQ_MODEL.
        temperature, max_tokens: passed to ChatGroq constructor.
        max_retries / retry_backoff: used for retrying transient failures.
        kwargs: forwarded to ChatGroq.
        """
        self.model = model or DEFAULT_GROQ_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Instantiate ChatGroq — the package reads GROQ_API_KEY from env by default.
        # You can add other ChatGroq params via kwargs if desired.
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
            # If init fails (missing keys / network), surface a helpful message
            raise RuntimeError(
                "Failed to create ChatGroq client. Ensure langchain-groq is installed and "
                "GROQ_API_KEY is set in environment. Original error: " + str(e)
            ) from e

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Synchronous call to Groq chat model.
        Combines an optional system_prompt with the user's prompt into the ChatGroq message format.

        Returns: str (the model's content). Raises on unrecoverable errors.
        """
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in environment. Set it before calling generate().")

        # default system prompt (keeps model behavior consistent with previous code)
        system_prompt = system_prompt or (
            "You are a helpful assistant. Use the provided CONTEXT to answer the user's question. "
            "Cite the source type in parentheses when you reference content."
        )

        # ChatGroq expects messages as list of tuples like ("system", "text"), ("human", "text")
        messages = [
            ("system", system_prompt),
            ("human", prompt),
        ]

        last_exc = None
        for attempt in range(1, self.max_retries + 2):  # `max_retries` retries after initial attempt
            try:
                # .invoke returns an AIMessage-like object in the docs; use .invoke(messages)
                ai_msg = self.llm.invoke(messages)
                # ai_msg likely has .content attribute (per docs)
                content = getattr(ai_msg, "content", None)
                if content is None:
                    # try other likely fields
                    content = ai_msg.get("content") if isinstance(ai_msg, dict) else str(ai_msg)
                return content
            except Exception as e:
                last_exc = e
                # transient network/timeout errors — retry
                if attempt <= self.max_retries:
                    wait = self.retry_backoff * attempt
                    time.sleep(wait)
                    continue
                # exhausted retries -> re-raise with helpful text
                raise RuntimeError(
                    f"Failed to get response from Groq after {attempt} attempts. Last error: {e}"
                ) from e

# convenience singleton
def get_groq_client(**kwargs) -> GroqClient:
    return GroqClient(**kwargs)
