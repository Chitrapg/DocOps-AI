# DocOps AI — RAG Chatbot + Multi-Agent Orchestrator

DocOps AI is a documentation assistant that combines **Hybrid RAG** (Vector + Graph) with a **Multi-Agent Orchestrator** to automate engineering workflows.

##  Features
- **Hybrid Retrieval**: Combines PostgreSQL (`pgvector`) and Neo4j (Graph) for deep context awareness.
- **Multi-Agent Routing**: Automatically routes queries to specialized agents:
  - **RAG Agent**: Technical Q&A using ingested documents.
  - **Testcase Agent**: Generates structured test cases and optionally pushes them to **Jira**.
  - **Confluence Agent**: Creates and publishes help documentation to **Confluence**.
- **Document Ingestion**: Supports PDF and DOCX processing with automated chunking and graph fragment extraction.
- **Streamlit UI**: A clean, interactive interface for chat and document management.

##  Tech Stack
- **LLM**: Groq (Llama 3.3 70B & Llama 4 Scout)
- **Orchestration**: LangChain
- **Databases**: PostgreSQL + `pgvector`, Neo4j
- **Frontend**: Streamlit
- **Integrations**: Jira API, Confluence API

##  Prerequisites
- Python 3.10+
- PostgreSQL with `pgvector` extension
- Neo4j Database
- API Keys for: Groq, Jira, and Confluence

##  Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd DocOps-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r .\requirements.txt
   ```

3. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   copy .env.example .env
   ```

4. **Initialize Database**:
   Run the script to create necessary PostgreSQL tables and extensions:
   ```bash
   python .\scripts\create_tables.py
   ```

##  Running the App

Start the Streamlit interface:
```bash
streamlit run .\ui\app.py
```

##  Project Structure
- `.\agents\`: Agent definitions and routing logic (`router.py`).
- `.\core\`: Core logic for embeddings, retrieval, and DB connections.
- `.\integrations\`: External service connectors (Jira/Confluence).
- `.\ui\`: Streamlit frontend implementation (`app.py`).
- `.\scripts\`: Database setup and utility scripts.


