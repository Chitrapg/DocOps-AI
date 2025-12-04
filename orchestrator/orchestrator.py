# orchestrator/orchestrator.py
import logging
from orchestrator.router import decide_route
from orchestrator.agents import AGENT_TOOLS

logger = logging.getLogger(__name__)

def handle_user_input(user_input: str, chat_history=None, dry_run=True) -> dict:
    """
    Orchestrator main entrypoint called by the Chat UI.
    Returns a dict {route: ..., result: "...", meta: {...}}
    dry_run: if True, do not perform destructive actions (e.g., push to Jira/Confluence).
    """
    route = decide_route(user_input)
    tool = AGENT_TOOLS.get(route)
    if tool is None:
        # fallback to RAG tool
        tool = AGENT_TOOLS["rag"]
    try:
        if route == "rag":
            out = tool(user_input, chat_history)
            return {"route": "rag", "result": out}
        elif route == "generate_testcases":
            # pass dry_run flag
            out = tool(user_input, dry_run=dry_run)
            return {"route": "generate_testcases", "result": out}
        elif route == "generate_confluence":
            out = tool(user_input, dry_run=dry_run)
            return {"route": "generate_confluence", "result": out}
        else:
            out = tool(user_input, dry_run=dry_run)
            return {"route": route, "result": out}
    except Exception as e:
        logger.exception("Orchestrator error")
        return {"route": route, "result": f"Orchestrator failed: {e}"}
