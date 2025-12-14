# ui/app.py
"""
Streamlit UI for DocOps AI - Production-ready version.
Simplified from app/streamlit_app.py using new module structure.
"""
import os
import sys
import tempfile
import traceback

# Load .env file FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings

# Import agent router
try:
    from agents.router import route_query, handle_user_input
    router_available = True
except Exception as e:
    router_available = False
    router_error = str(e)

# Import ingest
try:
    from core.ingest.pdf import extract_pdf_text
    from core.embeddings import get_embeddings
    from core.db.vectorstore import PGVectorStore
    ingest_available = True
except Exception as e:
    ingest_available = False
    ingest_error = str(e)

# Page config
st.set_page_config(page_title="DocOps AI", layout="wide")
st.title("DocOps AI — RAG Chatbot + Agents")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Approve push toggle
    st.session_state.approved_push = st.checkbox(
        "✅ Approve push to external systems (Jira/Confluence)",
        value=st.session_state.get("approved_push", False),
        help="Enable to actually push to Jira/Confluence instead of dry-run"
    )
    
    st.divider()
    st.header("📄 Document Ingestion")
    
    if ingest_available:
        uploaded = st.file_uploader(
            "Upload PDF/DOCX to ingest",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        
        if st.button("Ingest Documents") and uploaded:
            with st.spinner("Ingesting..."):
                for f in uploaded:
                    try:
                        # Save to temp file
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1])
                        tmp.write(f.getbuffer())
                        tmp.close()
                        
                        # Extract text
                        text = extract_pdf_text(tmp.name)
                        st.success(f"✅ Ingested {f.name} ({len(text)} chars)")
                        
                        # Cleanup
                        os.unlink(tmp.name)
                    except Exception as e:
                        st.error(f"❌ Failed: {f.name} - {e}")
    else:
        st.warning(f"Ingest unavailable: {ingest_error}")

# Main area
st.markdown("""
**Ask questions or request actions:**
- Normal queries → RAG (vector + graph retrieval)
- "Generate test cases..." → Testcase agent
- "Generate confluence..." → Confluence agent
""")

query = st.text_input(
    "Your question or instruction:",
    placeholder="e.g., 'What is the user registration flow?' or 'Generate test cases for login'"
)

if st.button("🚀 Ask") and query.strip():
    if not router_available:
        st.error(f"Router not available: {router_error}")
    else:
        with st.spinner("Processing..."):
            try:
                dry_run = not st.session_state.get("approved_push", False)
                result = route_query(query, dry_run=dry_run)
                
                # Store in history
                st.session_state.history.append({
                    "q": query,
                    "result": result
                })
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

# Display history
st.markdown("### Conversation")
for turn in reversed(st.session_state.history[-10:]):
    st.markdown(f"**Q:** {turn['q']}")
    result = turn.get("result", {})
    
    if isinstance(result, dict):
        route = result.get("route", "unknown")
        success = result.get("success", False)
        
        st.caption(f"Handled by: {route}")
        
        if not success:
            st.error(result.get("error", "Unknown error"))
        else:
            # RAG response
            if "final_answer" in result:
                st.markdown(result["final_answer"])
            
            # Testcase response
            elif "md_table" in result:
                st.markdown(result["md_table"])
                if result.get("created"):
                    st.success(f"Created Jira issues: {', '.join(result['created'])}")
            
            # Confluence response
            elif "result" in result and isinstance(result["result"], dict):
                conf = result["result"]
                if conf.get("page_url"):
                    st.success(f"📄 Published: [{conf['page_url']}]({conf['page_url']})")
                if conf.get("html"):
                    with st.expander("View HTML Preview"):
                        st.markdown(conf["html"], unsafe_allow_html=True)
                if conf.get("error"):
                    st.warning(conf["error"])
            
            # Generic response
            else:
                st.json(result)
    else:
        st.write(str(result))
    
    st.divider()


def main():
    """Entry point for running as module."""
    pass  # Streamlit runs the script directly


if __name__ == "__main__":
    main()
