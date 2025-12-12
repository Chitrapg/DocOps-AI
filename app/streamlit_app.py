# app/streamlit_app.py
"""
Streamlit UI for DocOps AI (RAG Chatbot + Agents)

This file is an updated, self-contained Streamlit application that:
- Allows ingesting PDF/DOCX/images via the sidebar
- Exposes an "Approve push" checkbox that controls whether agents push to Jira/Confluence
- Invokes the orchestrator (if present) with dry_run flag (dry_run = not approved_push)
- Properly displays generated markdown tables (testcases) and allows pushing to Jira via a Push button
- Handles many shapes of orchestrator responses (dicts or plain strings) with good error messages
"""

import os
import sys
import tempfile
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

# Defensive imports of local modules
try:
    from app.ingest import ingest_pdf
    ingest_import_err = None
except Exception as e:
    ingest_pdf = None
    ingest_import_err = e

try:
    from app.retriever import retriever
    retriever_import_err = None
except Exception as e:
    retriever = None
    retriever_import_err = e

# Jira client for direct push (used when pushing stored test cases)
try:
    from src.jira_client import create_testcase_issue_from_payload
    jira_import_err = None
except Exception as e:
    create_testcase_issue_from_payload = None
    jira_import_err = e

# Orchestrator entrypoint (optional). Signature may vary; we call flexibly.
try:
    from orchestrator.orchestrator import handle_user_input
    orchestrator_import_err = None
except Exception as e:
    handle_user_input = None
    orchestrator_import_err = e

# Groq client availability check (optional)
try:
    from app.groq_client import get_groq_client
    try:
        groq_client = get_groq_client()
    except Exception:
        groq_client = None
except Exception:
    get_groq_client = None
    groq_client = None

st.set_page_config(page_title="DocOps AI — RAG Chatbot + Agents", layout="wide")
st.title("DocOps AI — RAG Chatbot + Agents")

# -----------------------
# Sidebar: ingest + approval control
# -----------------------
with st.sidebar:
    st.header("Admin — Ingest files")
    uploaded = st.file_uploader(
        "Upload PDF / DOCX / Image(s) to ingest (accepts multiple)",
        type=["pdf", "docx", "doc", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded:
        st.info(f"{len(uploaded)} file(s) selected")

    ingest_button = st.button("Ingest uploaded files")

    st.markdown("---")
    st.header("Approve external pushes")
    st.markdown(
        "When enabled, agent actions that push to external systems (Jira, Confluence) will be executed. "
        "If disabled, agents run in dry-run / preview mode."
    )
    if "approved_push" not in st.session_state:
        st.session_state.approved_push = False

    st.session_state.approved_push = st.checkbox(
        "Approve push to external systems (Jira / Confluence)",
        value=st.session_state.approved_push,
    )

    st.markdown("---")
    st.write("Status / Diagnostics")
    if ingest_import_err:
        st.error(f"Ingest module import error: {ingest_import_err}")
    if retriever_import_err:
        st.warning(f"Retriever import error: {retriever_import_err}")
    if orchestrator_import_err:
        st.warning(f"Orchestrator import error: {orchestrator_import_err}")
    if groq_client is None and get_groq_client is not None:
        st.info("Groq client importable but client init failed (check API key / network).")
    if get_groq_client is None:
        st.info("Groq client wrapper not available (app will use local fallback/mock if configured).")

# -----------------------
# Main area: query + conversation
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    "Ask questions about your ingested documents, or request actions:\n\n"
    "- Normal conversational queries => RAG (vector + graph) + LLM synthesis.\n"
    "- \"Generate test cases...\" => Testcase agent (dry-run unless Approve push is ON).\n"
    "- \"Generate confluence...\" => Confluence agent (dry-run unless Approve push is ON).\n"
)

query = st.text_input(
    "Your question or instruction (examples: 'What is the refund policy?', 'Generate test cases for the checkout module', 'Create confluence for feature X')",
    value="",
)

# Ingest handling
if ingest_button and uploaded:
    progress_msgs = []
    for f in uploaded:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1])
            tmp.write(f.getbuffer())
            tmp.flush()
            tmp.close()
            if ingest_pdf is None:
                progress_msgs.append(f"SKIPPED {f.name}: ingest handler not available.")
            else:
                try:
                    res = ingest_pdf(tmp.name, source_name=f.name)
                    if isinstance(res, dict):
                        progress_msgs.append(f"Ingested {f.name}: {res.get('added_chunks','N/A')} chunks, doc_id={res.get('doc_id')}")
                    else:
                        progress_msgs.append(f"Ingest returned for {f.name}: {res}")
                except Exception as e:
                    progress_msgs.append(f"Error ingesting {f.name}: {e}")
        except Exception as e:
            progress_msgs.append(f"Failed saving {f.name}: {e}")
    for m in progress_msgs:
        if "Error" in m or "Failed" in m or "SKIPPED" in m:
            st.error(m)
        else:
            st.success(m)

# Ask/Invoke orchestrator
if st.button("Ask") and query.strip():
    with st.spinner("Processing..."):
        try:
            approved_push = bool(st.session_state.get("approved_push", False))
            dry_run_flag = not approved_push
            print(f"[DEBUG] approved_push={approved_push}, dry_run_flag={dry_run_flag}")

            if handle_user_input is not None:
                # Try flexible calls to orchestrator to support varying signatures.
                try:
                    # Preferred modern signature: handle_user_input(query, dry_run=..., chat_history=...)
                    res = handle_user_input(query, dry_run=dry_run_flag, chat_history=st.session_state.get("history", []))
                except TypeError:
                    try:
                        # older signature: handle_user_input(query, chat_history, dry_run)
                        res = handle_user_input(query, st.session_state.get("history", []), dry_run_flag)
                    except TypeError:
                        try:
                            # simplest: handle_user_input(query)
                            res = handle_user_input(query)
                        except Exception as e:
                            tb = traceback.format_exc()
                            res = {"route": "orchestrator", "success": False, "error": f"Orchestrator call failed: {e}\n{tb}"}
                except Exception as e:
                    tb = traceback.format_exc()
                    res = {"route": "orchestrator", "success": False, "error": f"Orchestrator call failed: {e}\n{tb}"}
            else:
                # Fallback: call RAG retriever directly (if available)
                if retriever is not None:
                    try:
                        res = retriever.retrieve(query)
                        res = {"route": "rag", "success": True, "result": res}
                    except Exception as e:
                        tb = traceback.format_exc()
                        res = {"route": "rag", "success": False, "error": f"Retriever failed: {e}\n{tb}"}
                else:
                    res = {"route": "none", "success": False, "error": "No orchestrator or retriever available."}
        except Exception as e:
            tb = traceback.format_exc()
            res = {"route": "exception", "success": False, "error": f"Unexpected UI-level error: {e}\n{tb}"}

    # Normalize response into a conversation turn
    turn = {"q": query, "raw": res}
    route = "rag"
    if isinstance(res, dict):
        route = res.get("route") or res.get("tool") or route
        turn["route"] = route
        
        # Extract result dict if present (Confluence agent uses this pattern)
        result_val = res.get("result")
        
        # Prefer markdown table preview when available
        if res.get("md_table"):
            turn["md_table"] = res.get("md_table")
        elif res.get("html"):
            turn["html"] = res.get("html")
        elif isinstance(result_val, dict) and result_val.get("html"):
            # Handle nested result.html from Confluence agent
            turn["html"] = result_val.get("html")
            # Also extract page_url and error if present
            if result_val.get("page_url"):
                turn["page_url"] = result_val.get("page_url")
            if result_val.get("error"):
                turn["a"] = f"Error: {result_val.get('error')}"
        else:
            # try to find textual answer - handle nested RAG response
            t = None
            if isinstance(result_val, dict):
                # RAG agent returns {"result": {"final_answer": ..., "graph_answer": ..., "vector_contexts": ...}}
                t = result_val.get("final_answer") or result_val.get("answer") or result_val.get("output")
            if t is None:
                t = res.get("final_answer") or res.get("output") or res.get("answer") or res.get("text")
            if t is None and isinstance(result_val, str):
                t = result_val
            if t is None and res.get("success") is False and res.get("error"):
                t = res.get("error")
            turn["a"] = t
        # attach creation results/errors if present
        if "created" in res:
            turn["created"] = res.get("created")
        if "errors" in res:
            turn["errors"] = res.get("errors")
        if "gen_id" in res:
            turn["gen_id"] = res.get("gen_id")
        if "parsed_count" in res:
            turn["parsed_count"] = res.get("parsed_count")
        # Store parsed testcases for direct Jira push (avoid regenerating)
        if "parsed" in res:
            turn["parsed"] = res.get("parsed")
    else:
        # plain string response
        turn["route"] = route
        turn["a"] = str(res)

    st.session_state.history.append(turn)

# -----------------------
# Display conversation
# -----------------------
st.markdown("### Conversation")

# We'll present the most recent turns first
for idx, turn in enumerate(reversed(st.session_state.history)):
    q = turn.get("q", "")
    route = turn.get("route", "rag")
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**Handled by:** `{route}`")

    # If markdown table (testcases) present, render it nicely
    if "md_table" in turn and turn.get("md_table"):
        st.markdown("**Generated Testcases (Markdown Preview):**")
        # Render the markdown table and allow HTML because we use <br> inside steps for visual multiline
        st.markdown(turn.get("md_table"), unsafe_allow_html=True)
        if turn.get("parsed_count") is not None:
            st.caption(f"Parsed testcases: {turn.get('parsed_count')}")

        # If not pushed yet and approval is set, show a "Push to Jira" button
        already_created = bool(turn.get("created"))
        if not already_created and st.session_state.get("approved_push", False):
            # unique key for the push button ensures multiple turns don't conflict
            push_key = f"push_{len(st.session_state.history) - idx}_{hash(q) & 0xFFFF}"
            if st.button(f"Push generation to Jira", key=push_key):
                with st.spinner("Pushing to Jira..."):
                    created = []
                    errors = []
                    
                    # Use the stored parsed testcases directly instead of regenerating
                    parsed_testcases = turn.get("parsed", [])
                    
                    if not parsed_testcases:
                        errors.append("No parsed testcases available to push. Please regenerate.")
                    elif create_testcase_issue_from_payload is None:
                        errors.append(f"Jira client not available (import error: {jira_import_err})")
                    else:
                        # Push each parsed testcase directly to Jira
                        for tc in parsed_testcases:
                            try:
                                payload = {
                                    "testcase_id": tc.get("id"),
                                    "business_scenario": tc.get("title"),
                                    "preconditions": tc.get("preconditions"),
                                    "test_steps": tc.get("steps"),
                                    "expected_result": tc.get("expected_result"),
                                    "test_data_notes": tc.get("test_data"),
                                    "related_requirement_ids": tc.get("related_requirement_ids"),
                                    "tags": tc.get("tags", []),
                                }
                                key = create_testcase_issue_from_payload(payload)
                                created.append(key)
                            except Exception as e:
                                errors.append(f"Failed to create Jira issue for testcase {tc.get('id')}: {e}")

                    if created:
                        st.success(f"Created {len(created)} Jira issues: {created}")
                        # attach to turn (so after rerender we show created)
                        turn["created"] = created
                    if errors:
                        st.error(f"Errors when creating Jira issues: {errors}")
                        turn["errors"] = errors

    # If HTML preview available (Confluence), show a natural response and HTML preview
    elif "html" in turn and turn.get("html"):
        # Generate natural text response
        route_name = turn.get("route", "generate_confluence")
        html_content = turn.get("html", "")
        
        # Extract title from HTML if present
        import re
        title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html_content, re.IGNORECASE)
        doc_title = title_match.group(1) if title_match else "Confluence Help Document"
        
        # Show natural language response
        if turn.get("page_url"):
            st.markdown(f"**A:** ✅ I've generated and published the **{doc_title}** to Confluence successfully!")
            st.success(f"📄 [View on Confluence]({turn.get('page_url')})")
        elif turn.get("a") and "Error" in str(turn.get("a")):
            st.markdown(f"**A:** I've generated the **{doc_title}**, but there was an issue pushing to Confluence:")
            st.error(turn.get("a"))
        else:
            st.markdown(f"**A:** I've generated the **{doc_title}**. The HTML content is ready below. To push to Confluence, make sure your environment has `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN`, and `CONFLUENCE_SPACE_KEY` configured, then enable 'Approve push' and try again.")
        
        # Show HTML preview in expander
        with st.expander("📝 View Generated HTML Content", expanded=False):
            # Render HTML directly
            st.markdown(html_content, unsafe_allow_html=True)
        
        with st.expander("🔧 Show Raw HTML Code"):
            st.code(html_content, language="html")

    # If created keys are present, show success
    if "created" in turn and turn.get("created"):
        st.success(f"Created {len(turn.get('created'))} Jira issues: {turn.get('created')}")

    # If errors exist, show them
    if "errors" in turn and turn.get("errors"):
        st.error(f"Errors: {turn.get('errors')}")

    # Fallback textual answer
    if turn.get("a"):
        st.markdown(f"**A:** {turn.get('a')}")

    st.markdown("---")

# -----------------------
# Footer debug panel
# -----------------------
if st.checkbox("Show debug info (connection / environment hints)"):
    try:
        import socket, json
        st.write("Environment variables (selected):")
        keys = [
            "PG_HOST", "PG_PORT", "PG_DB", "PG_USER", "GROQ_API_URL", "GROQ_API_KEY",
            "JIRA_SERVER", "JIRA_EMAIL", "JIRA_API_TOKEN", "CONFLUENCE_BASE_URL", "USE_GROQ_MOCK"
        ]
        env = {k: os.getenv(k) for k in keys}
        st.json(env)
        st.write("Local hostname / IP:")
        st.write(socket.gethostname())
        st.write("Parsed conversation turns:", len(st.session_state.history))
        st.write("Approved push flag:", st.session_state.get("approved_push", False))
        if groq_client:
            st.write("Groq client available: True")
        else:
            st.write("Groq client available: False")
    except Exception as e:
        st.write("Failed to gather debug info:", e)
