# TradeFin AI — RAG Chatbot + Multi-Agent Orchestrator

Overview:
- Upload PDFs / DOCX / screenshots via the Streamlit UI (app/streamlit_app.py).
- The orchestrator auto-routes user queries to:
  - RAG (vector + graph retrieval + Groq synthesis)
  - Testcase agent (generate testcases → optional push to Jira)
  - Confluence agent (generate help/confluence page → optional push)

Quickstart (local):
1. Copy `.env.example` -> `.env` and fill credentials (Postgres, Neo4j, Groq, Jira, Confluence).
2. Install requirements:
