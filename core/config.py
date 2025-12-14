# core/config.py
"""
Unified configuration loader using pydantic-settings.
Moved from app/config.py to core/ for centralized access.
"""
from functools import lru_cache
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Unified configuration loader using pydantic-settings (v2.12+ compatible).
    - Ignores extra/unknown env keys (prevents failures caused by stray keys in .env).
    - Makes some infra credentials optional at import-time.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General / app
    ENV: str = Field("production", env="ENV")
    DEBUG: bool = Field(False, env="DEBUG")
    UPLOAD_DIR: str = Field("./uploads", env="UPLOAD_DIR")
    RESULTS_DIR: str = Field("./results", env="RESULTS_DIR")

    # Postgres / pgvector
    PG_HOST: str = Field("localhost", env="PG_HOST")
    PG_PORT: int = Field(5432, env="PG_PORT")
    PG_DB: Optional[str] = Field(None, validation_alias="PG_DATABASE")  # Accept both PG_DB and PG_DATABASE
    PG_USER: Optional[str] = Field(None, env="PG_USER")
    PG_PASSWORD: Optional[str] = Field(None, env="PG_PASSWORD")
    PG_CONNECTION_URI: Optional[str] = Field(None, env="PG_CONNECTION_URI")
    EMBEDDING_DIM: int = Field(384, env="EMBEDDING_DIM")

    # Embedding model
    EMBEDDING_PROVIDER: str = Field("local", env="EMBEDDING_PROVIDER")
    EMBED_MODEL_NAME: str = Field("all-MiniLM-L6-v2", env="EMBED_MODEL_NAME")
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # Neo4j
    NEO4J_URL: Optional[str] = Field(None, env="NEO4J_URL")
    NEO4J_USER: Optional[str] = Field(None, env="NEO4J_USER")
    NEO4J_PASSWORD: Optional[str] = Field(None, env="NEO4J_PASSWORD")
    NEO4J_DATABASE: Optional[str] = Field(None, env="NEO4J_DATABASE")

    # Groq LLM
    GROQ_API_KEY: Optional[str] = Field(None, env="GROQ_API_KEY")
    GROQ_MODEL: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    GROQ_API_URL: str = Field("https://api.groq.com/openai/v1/chat/completions", env="GROQ_API_URL")
    GROQ_VISION_MODEL: str = Field("meta-llama/llama-4-scout-17b-16e-instruct", env="GROQ_VISION_MODEL")

    # Jira
    JIRA_SERVER: Optional[str] = Field(None, env="JIRA_SERVER")
    JIRA_EMAIL: Optional[str] = Field(None, env="JIRA_EMAIL")
    JIRA_API_TOKEN: Optional[str] = Field(None, env="JIRA_API_TOKEN")
    JIRA_PROJECT_KEY: str = Field("TFTEST", env="JIRA_PROJECT_KEY")

    # Confluence
    CONFLUENCE_BASE_URL: Optional[str] = Field(None, env="CONFLUENCE_BASE_URL")
    CONFLUENCE_USERNAME: Optional[str] = Field(None, env="CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN: Optional[str] = Field(None, env="CONFLUENCE_API_TOKEN")
    CONFLUENCE_SPACE_KEY: Optional[str] = Field(None, env="CONFLUENCE_SPACE_KEY")

    # Routing keywords
    TESTCASE_TRIGGER_KEYWORDS: str = Field(
        "generate test case,testcase,test cases,tc generation",
        env="TESTCASE_TRIGGER_KEYWORDS",
    )
    CONFLUENCE_TRIGGER_KEYWORDS: str = Field(
        "generate help text,help document,confluence,help page,user guide",
        env="CONFLUENCE_TRIGGER_KEYWORDS",
    )

    # Performance tuning
    SKIP_GRAPH_EXTRACTION: bool = Field(False, env="SKIP_GRAPH_EXTRACTION")
    GRAPH_BATCH_SIZE: int = Field(3, env="GRAPH_BATCH_SIZE")
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env="CHUNK_OVERLAP")

    @validator("PG_PORT", pre=True)
    def _port_to_int(cls, v):
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v

    def require_db(self) -> None:
        if self.PG_CONNECTION_URI:
            return
        if not all([self.PG_DB, self.PG_USER, self.PG_PASSWORD, self.PG_HOST]):
            raise RuntimeError(
                "Postgres configuration is missing. Set PG_DB, PG_USER, PG_PASSWORD (or PG_CONNECTION_URI) in the environment."
            )

    def require_groq(self) -> None:
        if not self.GROQ_API_KEY or not self.GROQ_API_URL:
            raise RuntimeError("GROQ_API_KEY and GROQ_API_URL must be set in environment.")

    def require_jira(self) -> None:
        if not (self.JIRA_SERVER and self.JIRA_EMAIL and self.JIRA_API_TOKEN):
            raise RuntimeError("JIRA_SERVER, JIRA_EMAIL, and JIRA_API_TOKEN are required to push to Jira.")

    def require_confluence(self) -> None:
        if not (self.CONFLUENCE_BASE_URL and self.CONFLUENCE_SPACE_KEY):
            raise RuntimeError("CONFLUENCE_BASE_URL and CONFLUENCE_SPACE_KEY are required.")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Cached settings instance
settings = get_settings()
