# app/streamlit_app.py
import os
import sys
import tempfile
import importlib
import traceback
from dotenv import load_dotenv

load_dotenv()

# Ensure project root (parent folder of this file) is on sys.path so `import app.xxx` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure app package marker exists (helps some importers)
init_py = os.path.join(os.path.dirname(__file__), "__init__.py")
if not os.path.exists(init_py):
    open(init_py, "a").close()

import streamlit as st

# Local application components (these files are part of your repo)
# They may fail at import if environment is misconfigured; handle gracefully.
try:
    from app.ingest import ingest_pdf
except Exception as e:
    ingest_pdf = None
    ingest_import_err = e

try:
    from app.retriever import retriever
except Exception as e:
    retriever = None
    retriever_import_err = e

# Try orchestrator handler (optional)
try:
    from orchestrator.orchestrator import handle_user_input
except Exception:
    handle_user_input = None

# Try get_groq_client for status/info (optional)
try:
    from app.groq_client import get_groq_client
    groq_client = None
    try:
        groq_client = get_groq_client()
    except Exception:
        groq_client = None
except Exception:
    get_groq_client = None
    groq_client = None

st.set_page_config(page_title="RAG Chatbot & Agents", layout="wide")

st.title("DocOps AI — RAG Chatbot + Agents")

with st.sidebar:
    st.header("Admin — Ingest PDFs / Docs / Images")
    uploaded = st.file_uploader("Upload PDF/DOC/DOCX/Image(s)", type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], accept_multiple_files=True)
    ingest_button = st.button("Ingest uploaded files")

    if uploaded and ingest_button:
        progress_msgs = []
        for f in uploaded:
            # Save to temp and call ingest handler where supported
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1])
                tmp.write(f.getbuffer())
                tmp.flush()
                tmp.close()

                if ingest_pdf is None:
                    progress_msgs.append(f"INGEST SKIPPED for {f.name}: ingest handler not available (import error).")
                else:
                    try:
                        res = ingest_pdf(tmp.name, source_name=f.name)
                        if isinstance(res, dict):
                            progress_msgs.append(f"Ingested {f.name}: {res.get('added_chunks', 'N/A')} chunks, doc_id={res.get('doc_id')}")
                        else:
                            progress_msgs.append(f"Ingest returned for {f.name}: {res}")
                    except Exception as e:
                        progress_msgs.append(f"Error ingesting {f.name}: {e}")
            except Exception as e:
                progress_msgs.append(f"Failed to save uploaded file {f.name}: {e}")

        for m in progress_msgs:
            st.success(m)

st.markdown("---")
st.markdown(
    "Ask questions about your ingested documents. The system searches a vector DB (Postgres+pgvector) "
    "and a graph DB (Neo4j), synthesizes with the LLM, and can also run specialized agents (generate testcases -> Jira, "
    "generate Confluence help pages -> Confluence)."
)

if "history" not in st.session_state:
    st.session_state.history = []

if "approved" not in st.session_state:
    st.session_state.approved = False

if "approved_push" not in st.session_state:
    # When the user approves pushing to destinations (Jira/Confluence)
    st.session_state.approved_push = False

query = st.text_input("Your question or instruction (e.g. 'What is X?' or 'Generate test cases for feature Y')", "")

# Ask button handling
if st.button("Ask") and query.strip():
    with st.spinner("Retrieving answer..."):
        res = None
        # Prefer orchestrator if available
        try:
            if handle_user_input is not None:
                # pass dry_run false only if user has explicitly approved pushing
                dry_run_flag = not st.session_state.get("approved_push", False)
                res = handle_user_input(query, dry_run=dry_run_flag, chat_history=st.session_state.get("history", []))
            else:
                # fallback to RAG retriever
                if retriever is not None:
                    res = retriever.retrieve(query)
                else:
                    res = f"No retriever or orchestrator available. Import errors: ingest_err={locals().get('ingest_import_err', None)}, retriever_err={locals().get('retriever_import_err', None)}"
        except Exception as e:
            # Capture full traceback for debug in UI if desired
            tb = traceback.format_exc()
            res = f"Orchestrator/Tool invocation failed: {e}\n\nTraceback:\n{tb}"

    # Normalize result (dict preferred, but allow string)
    if isinstance(res, dict):
        route = res.get("route") or res.get("tool") or "rag"
        answer = (
            res.get("final_answer")
            or res.get("result")
            or res.get("output")
            or res.get("answer")
            or res.get("text")
            or res.get("graph_answer")
            or res.get("html")
            or str(res)
        )
    else:
        route = "rag"
        answer = str(res)

    st.session_state.history.append({"q": query, "route": route, "a": answer})

st.markdown("### Conversation")

for turn in reversed(st.session_state.history):
    q = turn.get("q", "")
    a = turn.get("a", "")
    route = turn.get("route", "rag")

    st.markdown(f"**Q:** {q}")
    st.markdown(f"**Handled by:** `{route}`")

    # Render HTML previews safely: show raw in expander and a short summary
    if isinstance(a, str) and a.strip().startswith("<") and ("<p" in a or "<div" in a or "<table" in a):
        st.markdown("**A (HTML preview):**")
        # show a short rendered snippet as plain text (do not use unsafe HTML rendering)
        preview_plain = a.replace("\n", " ")[:1000]
        st.markdown(preview_plain + ("..." if len(preview_plain) < len(a) else ""))
        with st.expander("Show raw HTML / full output"):
            st.code(a)
    else:
        # Markdown-escape answer for safe display
        st.markdown(f"**A:** {a}")

    st.markdown("---")

# small helpful debug panel (toggle)
if st.checkbox("Show debug info (connection / environment hints)"):
    try:
        import os, socket, json
        st.write("Environment variables (selected):")
        keys = [
            "PG_HOST", "PG_PORT", "PG_DB", "PG_USER", "GROQ_API_URL", "GROQ_API_KEY",
            "JIRA_SERVER", "JIRA_EMAIL", "JIRA_API_TOKEN", "CONFLUENCE_BASE_URL"
        ]
        env = {k: os.getenv(k) for k in keys}
        st.json(env)
        st.write("Local hostname / IP:")
        st.write(socket.gethostname())
        st.write("Parsed conversation turns:", len(st.session_state.history))
        st.write("Last approved_push:", st.session_state.get("approved_push", False))
        st.write("Last approved (UI):", st.session_state.get("approved", False))
        if groq_client:
            st.write("Groq client available:", True)
        else:
            st.write("Groq client available:", False)
    except Exception as e:
        st.write("Failed to gather debug info:", e)
