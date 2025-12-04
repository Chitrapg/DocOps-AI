# app/streamlit_app.py  (modified)
import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure app package marker exists
init_py = os.path.join(os.path.dirname(__file__), "__init__.py")
if not os.path.exists(init_py):
    open(init_py, "a").close()

from app.ingest import ingest_pdf
from orchestrator.orchestrator import handle_user_input
from app.retriever import retriever
from app.groq_client import get_groq_client

st.set_page_config(page_title="RAG Chatbot + Agents")
st.title("RAG Chatbot + AI Agents")

with st.sidebar:
    st.header("Admin — Ingest PDFs / Docs")
    uploaded = st.file_uploader("Upload PDF / DOC / DOCX / Image(s)", type=["pdf","doc","docx","png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded and st.button("Ingest uploaded files"):
        progress_msgs = []
        for f in uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1])
            tmp.write(f.getbuffer())
            tmp.flush()
            # For pdf/docx files call ingest_pdf; for images we store screenshot metadata
            if f.name.lower().endswith((".pdf", ".doc", ".docx")):
                res = ingest_pdf(tmp.name, source_name=f.name)
                progress_msgs.append(f"Ingested {f.name}: {res.get('added_chunks', 0)} chunks, doc_id={res.get('doc_id')}")
            else:
                # store image to uploads directory for Confluence generator to find
                uploads_dir = os.path.join(os.getcwd(), "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                dest = os.path.join(uploads_dir, f.name)
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                progress_msgs.append(f"Saved screenshot {f.name} to uploads folder.")
        for m in progress_msgs:
            st.success(m)

st.markdown("---")
st.markdown("Ask questions or request actions (generate test cases, create confluence page). The system will auto-route your query to the appropriate AI agent.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Your question / command", "")

# For destructive ops show an approval checkbox
if "approved_push" not in st.session_state:
    st.session_state.approved_push = False

push_confirm = st.checkbox("Allow the system to push created content (Jira / Confluence). Uncheck for dry-run.", value=False)
st.session_state.approved_push = push_confirm

if st.button("Send") and query.strip():
    with st.spinner("Orchestrating..."):
        res = handle_user_input(query, chat_history=st.session_state.history, dry_run=not st.session_state.approved_push)
    # Append into session history
    st.session_state.history.append({"q": query, "route": res.get("route"), "a": res.get("result")})
    st.success(f"Routed to: {res.get('route')}")

st.markdown("### Conversation")
for turn in reversed(st.session_state.history):
    st.markdown(f"**Q:** {turn['q']}")
    st.markdown(f"**Route:** {turn['route']}")
    st.markdown(f"**A:**\n\n```\n{turn['a']}\n```")
    st.markdown("---")
