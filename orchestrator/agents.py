# orchestrator/agents.py
import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Defensive imports for langchain tool decorator / create_agent (not strictly required)
try:
    from langchain.tools import tool
    from langchain.agents import create_agent
except Exception:
    # Provide a no-op fallback for decorator to avoid import-time hard errors
    def tool(*args, **kwargs):
        def _decorator(f):
            return f
        return _decorator

    def create_agent(*args, **kwargs):
        return None

# Local imports (may raise at import-time if modules missing)
try:
    from src.generator import generate_testcases  # returns (gen_id, md_table, parsed_testcases)
except Exception as e:
    logger.warning("Could not import generate_testcases: %s", e)
    generate_testcases = None

try:
    from src.jira_client import create_testcase_issue_from_payload
except Exception as e:
    logger.warning("Could not import Jira client: %s", e)
    create_testcase_issue_from_payload = None

try:
    from app.retriever import retriever as rag_retriever
except Exception as e:
    logger.warning("Could not import RAG retriever: %s", e)
    rag_retriever = None

# Helper: robust invoker for various tool shapes (functions, StructuredTool, etc.)
def _invoke_tool(tool_obj: Any, *args, **kwargs):
    """
    Try common ways to call a langchain tool object:
     - callable(tool_obj)(*args, **kwargs)
     - tool_obj.run(...)
     - tool_obj.invoke(...)
     - tool_obj.func(...)
     - nested .tool.func / .tool.run
    """
    # 1) direct call
    if callable(tool_obj):
        try:
            return tool_obj(*args, **kwargs)
        except TypeError:
            # maybe signature mismatch; fall through to other attempts
            pass

    for attr in ("run", "invoke", "func", "__call__"):
        fn = getattr(tool_obj, attr, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue

    # nested container patterns
    for path in (("tool", "func"), ("tool", "run"), ("fn", "func")):
        cur = tool_obj
        ok = True
        for p in path:
            cur = getattr(cur, p, None)
            if cur is None:
                ok = False
                break
        if ok and callable(cur):
            return cur(*args, **kwargs)

    raise RuntimeError("Unable to invoke tool object; unknown shape.")

# ========== Tools ==========

@tool("RAG", description="Answer conversational queries using the ingested document KB (vector + graph).")
def call_rag_tool(query: str, chat_history: Optional[list] = None) -> dict:
    logger.info("RAG tool invoked for query: %s", query)
    if rag_retriever is None:
        return {"route": "rag", "success": False, "error": "RAG retriever not available."}
    try:
        res = rag_retriever.retrieve(query)
        # normalize into dict response
        return {"route": "rag", "success": True, "result": res}
    except Exception as e:
        logger.exception("RAG tool failed")
        return {"route": "rag", "success": False, "error": str(e)}

@tool(
    "GenerateTestcases",
    description="Generate testcases from the ingested functional doc and optionally push to Jira."
)
def call_generate_testcases(query: str, dry_run: bool = True) -> dict:
    """
    Returns a dict:
      {
        "route": "generate_testcases",
        "success": True/False,
        "gen_id": "...",
        "md_table": "<markdown table string>",
        "parsed_count": N,
        "parsed": [...],                # list of normalized testcases (dicts)
        "created": ["JIRA-1","JIRA-2"], # only when dry_run==False
        "errors": ["..."]               # any errors during push
      }
    """
    logger.info("Testcase agent invoked: dry_run=%s, query=%s", dry_run, query)
    try:
        from src.retriever import retrieve_grounding_for_query
    except Exception:
        retrieve_grounding_for_query = None

    # 1) retrieve grounding
    grounding = []
    if retrieve_grounding_for_query:
        try:
            grounding = retrieve_grounding_for_query(query, top_k=8)
        except Exception as e:
            logger.warning("Grounding retrieval failed: %s", e)
            grounding = []

    # 2) call generator
    if generate_testcases is None:
        return {"route": "generate_testcases", "success": False, "error": "Generator not available."}

    try:
        gen_id, md_table, parsed = generate_testcases(grounding, query)
    except Exception as e:
        logger.exception("Generation error")
        return {"route": "generate_testcases", "success": False, "error": f"Generation failed: {e}"}

    parsed = parsed or []
    resp = {
        "route": "generate_testcases",
        "success": True,
        "gen_id": gen_id,
        "md_table": md_table,
        "parsed_count": len(parsed),
        "parsed": parsed,
        "created": [],
        "errors": []
    }

    # If dry_run, return preview only
    if dry_run:
        return resp

    # Otherwise attempt to push to Jira
    if create_testcase_issue_from_payload is None:
        resp["success"] = False
        resp["errors"].append("Jira client not available (missing imports or credentials).")
        return resp

    created = []
    errors = []

    for tc in parsed:
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
            logger.exception("Failed to create Jira issue for testcase %s", tc.get("id"))
            errors.append(str(e))

    resp["created"] = created
    resp["errors"] = errors
    resp["success"] = len(errors) == 0
    return resp


@tool("GenerateConfluence", description="Generate Confluence help document from FRD + screenshots and optionally push.")
def call_generate_confluence(query: str, dry_run: bool = True) -> dict:
    logger.info("Confluence agent invoked: dry_run=%s, query=%s", dry_run, query)
    try:
        # reuse the app-level wrapper
        from app.confluence_generator import create_confluence_page_from_grounding
    except Exception as e:
        logger.exception("Confluence helper not available: %s", e)
        return {"route": "generate_confluence", "success": False, "error": "Confluence helper not available."}

    try:
        # use retriever to build grounding
        from src.retriever import retrieve_grounding_for_query
        grounding = retrieve_grounding_for_query(query, top_k=8)
    except Exception:
        grounding = []

    title = f"Help - {query[:60]}"
    try:
        result = create_confluence_page_from_grounding(grounding, title, push=(not dry_run))
        # result may be dict with 'html', 'page_url', etc.
        return {"route": "generate_confluence", "success": True, "result": result}
    except Exception as e:
        logger.exception("Confluence generation failed")
        return {"route": "generate_confluence", "success": False, "error": str(e)}

# Expose mapping under multiple keys so routing works regardless of case
AGENT_TOOLS = {
    "rag": call_rag_tool,
    "RAG": call_rag_tool,
    "generate_testcases": call_generate_testcases,
    "GenerateTestcases": call_generate_testcases,
    "generate_confluence": call_generate_confluence,
    "GenerateConfluence": call_generate_confluence,
}
