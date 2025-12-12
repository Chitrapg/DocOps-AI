# orchestrator/router.py
import re
import logging
from app.groq_client import get_groq_client

logger = logging.getLogger(__name__)

# Keyword patterns (expand as needed)
TESTCASE_PATTERNS = [
    r"\bgenerate test(s|ing)?\b",
    r"\btest case(s)?\b",
    r"\bcreate test(s|cases)?\b",
    r"\bpush to jira\b",
    r"\bjira\b"
]

CONFLUENCE_PATTERNS = [
    r"\bconfluence\b",
    r"\bhelp text\b",
    r"\bgenerate help\b",
    r"\bcreate documentation\b",
    r"\bcreate doc(ument)?\b",
    r"\bcreate page\b",
    r"\bhelp page\b",
]

def _match_patterns(text: str, patterns):
    t = text.lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False

def decide_route(user_input: str) -> str:
    """
    Returns one of: 'rag', 'generate_testcases', 'generate_confluence'
    """
    # 1. deterministic keyword checks
    if _match_patterns(user_input, TESTCASE_PATTERNS):
        return "generate_testcases"
    if _match_patterns(user_input, CONFLUENCE_PATTERNS):
        return "generate_confluence"

    # 2. LLM fallback - ask groq to pick one
    try:
        groq = get_groq_client()
        prompt = (
            "You are a router that MUST choose exactly one of: rag, generate_testcases, generate_confluence.\n"
            "The user input is below. Decide the single correct destination and ONLY return that word.\n"
            f"User input: '''{user_input}'''\n"
            "Return the single token: rag OR generate_testcases OR generate_confluence."
        )
        out = groq.generate(prompt, temperature=0.0, max_tokens=16)
        out_lower = (out or "").strip().lower()
        if "generate_test" in out_lower or "testcase" in out_lower or "generate_testcases" in out_lower:
            return "generate_testcases"
        if "confluence" in out_lower or "help" in out_lower or "doc" in out_lower:
            return "generate_confluence"
        return "rag"
    except Exception as e:
        logger.warning("LLM router fallback failed: %s", e)
        return "rag"
