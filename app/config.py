# app/config.py
import os
from functools import lru_cache
from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings):
    """
    Unified configuration loader for the entire Multi-Agent System:
    - RAG Chatbot (Vector DB + Graph DB + Groq)
    - Test Case Generator (pgvector + Groq + Jira)
    - Confluence Generator (Groq Vision + Confluence REST)
    - Multi-Agent Orchestrator
    """

    # =========================
    # 🔹 Server / Application
    # =========================
    ENV: str = Field("production", env="ENV")
    DEBUG: bool = Field(False, env="DEBUG")
    UPLOAD_DIR: str = Field("./uploads", env="UPLOAD_DIR")
    RESULTS_DIR: str = Field("./results", env="RESULTS_DIR")

    # =========================
    # 🔹 Postgres / pgvector (RAG + Testcase Agent DB)
    # =========================
    PG_HOST: str = Field("localhost", env="PG_HOST")
    PG_PORT: int = Field(5432, env="PG_PORT")
    PG_DB: str = Field(..., env="PG_DB")
    PG_USER: str = Field(..., env="PG_USER")
    PG_PASSWORD: str = Field(..., env="PG_PASSWORD")
    PG_CONNECTION_URI: Optional[str] = Field(None, env="PG_CONNECTION_URI")
    EMBEDDING_DIM: int = Field(384, env="EMBEDDING_DIM")

    # =========================
    # 🔹 Embedding Model
    # =========================
    EMBEDDING_PROVIDER: str = Field("local", env="EMBEDDING_PROVIDER")
    EMBED_MODEL_NAME: str = Field("all-MiniLM-L6-v2", env="EMBED_MODEL_NAME")
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # =========================
    # 🔹 Neo4j Graph DB (RAG Knowledge Graph)
    # =========================
    NEO4J_URL: Optional[str] = Field(None, env="NEO4J_URL")
    NEO4J_USER: Optional[str] = Field(None, env="NEO4J_USER")
    NEO4J_PASSWORD: Optional[str] = Field(None, env="NEO4J_PASSWORD")
    NEO4J_DATABASE: Optional[str] = Field(None, env="NEO4J_DATABASE")

    # =========================
    # 🔹 Groq LLM (RAG + Testcase Agent + Confluence)
    # =========================
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    GROQ_MODEL: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    GROQ_API_URL: str = Field(
        "https://api.groq.com/openai/v1/chat/completions",
        env="GROQ_API_URL",
    )

    # Vision model (Confluence generator)
    GROQ_VISION_MODEL: str = Field(
        "meta-llama/llama-4-scout-17b-16e-instruct",
        env="GROQ_VISION_MODEL",
    )

    # =========================
    # 🔹 Jira Test Case Push
    # =========================
    JIRA_SERVER: Optional[str] = Field(None, env="JIRA_SERVER")
    JIRA_EMAIL: Optional[str] = Field(None, env="JIRA_EMAIL")
    JIRA_API_TOKEN: Optional[str] = Field(None, env="JIRA_API_TOKEN")
    JIRA_PROJECT_KEY: str = Field("TFTEST", env="JIRA_PROJECT_KEY")

    # =========================
    # 🔹 Confluence Help Doc Push
    # =========================
    CONFLUENCE_BASE_URL: Optional[str] = Field(None, env="CONFLUENCE_BASE_URL")
    CONFLUENCE_USERNAME: Optional[str] = Field(None, env="CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN: Optional[str] = Field(None, env="CONFLUENCE_API_TOKEN")
    CONFLUENCE_SPACE_KEY: Optional[str] = Field(None, env="CONFLUENCE_SPACE_KEY")

    # =========================
    # 🔹 Agent Routing (keywords)
    # =========================
    TESTCASE_TRIGGER_KEYWORDS: str = Field(
        "generate test case,testcase,test cases,tc generation",
        env="TESTCASE_TRIGGER_KEYWORDS",
    )
    CONFLUENCE_TRIGGER_KEYWORDS: str = Field(
        "generate help text,help document,confluence,help page,user guide",
        env="CONFLUENCE_TRIGGER_KEYWORDS",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Cached settings loader (only loads once)."""
    return Settings()


settings = get_settings()
