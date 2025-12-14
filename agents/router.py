# agents/router.py
"""
Agent Router - Routes user queries to the appropriate agent.
"""
from typing import Dict, Any, Optional
import logging

from core.config import settings

logger = logging.getLogger(__name__)


def route_query(
    query: str,
    dry_run: bool = True,
    chat_history: Optional[list] = None
) -> Dict[str, Any]:
    """
    Route user query to the appropriate agent.
    
    Uses keyword-based routing by default.
    Can be extended to use LLM-based routing.
    """
    text = (query or "").strip().lower()
    
    # Check for testcase keywords
    testcase_keywords = settings.TESTCASE_TRIGGER_KEYWORDS.lower().split(",")
    for kw in testcase_keywords:
        if kw.strip() in text:
            logger.info(f"Routing to testcase agent: {query[:50]}")
            from agents.testcase_agent import TestcaseAgent
            agent = TestcaseAgent(dry_run=dry_run)
            return agent.execute(query)
    
    # Check for confluence keywords
    confluence_keywords = settings.CONFLUENCE_TRIGGER_KEYWORDS.lower().split(",")
    for kw in confluence_keywords:
        if kw.strip() in text:
            logger.info(f"Routing to confluence agent: {query[:50]}")
            from agents.confluence_agent import ConfluenceAgent
            agent = ConfluenceAgent(dry_run=dry_run)
            return agent.execute(query)
    
    # Default to RAG
    logger.info(f"Routing to RAG agent: {query[:50]}")
    from agents.rag_agent import RAGAgent
    agent = RAGAgent(dry_run=dry_run)
    return agent.execute(query)


# Backward compatibility with orchestrator
def handle_user_input(
    user_input: str,
    agent_tools: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
    chat_history: Optional[list] = None
) -> Any:
    """
    Main entry point for handling user input.
    Backward compatible with old orchestrator signature.
    """
    return route_query(user_input, dry_run=dry_run, chat_history=chat_history)


# Export agent tools for backward compatibility
def get_agent_tools():
    """Get dictionary of available agent tools."""
    from agents.rag_agent import call_rag_tool
    from agents.testcase_agent import call_generate_testcases
    from agents.confluence_agent import call_generate_confluence
    
    return {
        "rag": call_rag_tool,
        "RAG": call_rag_tool,
        "generate_testcases": call_generate_testcases,
        "GenerateTestcases": call_generate_testcases,
        "generate_confluence": call_generate_confluence,
        "GenerateConfluence": call_generate_confluence,
    }


AGENT_TOOLS = None  # Lazy loaded

def _get_agent_tools():
    global AGENT_TOOLS
    if AGENT_TOOLS is None:
        AGENT_TOOLS = get_agent_tools()
    return AGENT_TOOLS
