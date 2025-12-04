# orchestrator/agents.py
import logging
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI  # fallback if you'd like to use OpenAI; optional
from app.retriever import retriever as rag_retriever  # your retriever instance
from src.generator import generate_testcases  # returns (gen_id, md_table, parsed_testcases)
from src.jira_client import create_testcase_issue_from_payload
from confluence.confluence_generator import generate_confluence_from_frd  # wrapper we'll add
from app.groq_client import get_groq_client

logger = logging.getLogger(__name__)

# Instantiate a lightweight LLM for routing (cheap) and a stronger for generation if desired
# We rely on your Groq client; create_agent expects a model object - but create_agent can support a string
# Many LangChain adapters accept "model" string. Use default; if unavailable, fallback to "gpt-4o-mini" string.
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gpt-4o-mini")
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o")

# Build sub-agents as create_agent — keep them simple: for these subagents we'll not give extra tools (they use local functions)
# We will not instantiate heavy LLM wrappers here; Instead the tool functions call into your existing code.
# However we create a placeholder agent instance for signature compatibility when needed.

# Note: depending on your langchain SDK version, `create_agent` may accept string model identifiers.
# If it expects a model object, you can construct ChatOpenAI(...) or get_groq_client().llm if that matches interface.

try:
    # prefer Groq client if it exposes .llm
    groq_client = get_groq_client()
    router_agent_model = groq_client.llm  # may be a ChatGroq wrapper or similar
except Exception:
    router_agent_model = None

# If LangChain requires a model object and router_agent_model is None, use ChatOpenAI as fallback
if router_agent_model is None:
    try:
        router_agent_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception:
        router_agent_model = "gpt-4o-mini"  # fallback string if create_agent accepts string

# Create a top-level orchestrator agent (not strictly necessary, used if you want agent-ful reasoning)
orchestrator_agent = create_agent(model=router_agent_model, tools=[])


# ========== Tools: wrappers to call sub-systems ==========
@tool(
    name="RAG",
    description="Answer conversational queries using the ingested PDF knowledge base (vector + graph). Input: user question string."
)
def call_rag_tool(query: str, chat_history: list | None = None) -> str:
    logger.info("RAG tool invoked for query: %s", query)
    try:
        res = rag_retriever.retrieve(query)
        # return final synthesized answer
        return res.get("final_answer") or res.get("graph_answer") or "No answer found."
    except Exception as e:
        logger.exception("RAG tool error")
        return f"RAG tool failed: {e}"


@tool(
    name="GenerateTestcases",
    description="Generate testcases from the ingested functional doc and optionally push to Jira. Input: JSON string or plain query"
)
def call_generate_testcases(query: str, dry_run: bool = True) -> str:
    """
    query: user natural instruction like "Generate test cases for checkout module".
    dry_run: if True, do not push to Jira; just return generated testcases and parsed JSON.
    """
    logger.info("Testcase agent invoked: dry_run=%s, query=%s", dry_run, query)
    try:
        # 1) Retrieve grounding (we call the shared retriever to get context)
        from src.retriever import retrieve_grounding_for_query
        grounding = retrieve_grounding_for_query(query, top_k=8)

        gen_id, md_table, parsed = generate_testcases(grounding, query)

        # If dry_run, return summary + gen_id
        if dry_run:
            return f"GENERATION_ID: {gen_id}\n\nMARKDOWN_TABLE:\n{md_table}\n\nPARSED_COUNT: {len(parsed)}"

        # otherwise push to Jira
        created_keys = []
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
                    "tags": tc.get("tags", [])
                }
                key = create_testcase_issue_from_payload(payload)
                created_keys.append(key)
            except Exception as e:
                errors.append(str(e))
        return f"Created {len(created_keys)} Jira issues: {created_keys}. Errors: {errors}"
    except Exception as e:
        logger.exception("Testcase agent failed")
        return f"Testcase agent failed: {e}"


@tool(
    name="GenerateConfluence",
    description="Generate Confluence help document from FRD + screenshots. Input: expects query like 'create confluence for feature X' or an instruction plus attachments metadata."
)
def call_generate_confluence(query: str, dry_run: bool = True) -> str:
    """
    Here we call your confluence generator wrapper which expects an FRD path + screenshots folder or grounding text.
    We'll attempt to re-use the grounding + accept user instructions.
    """
    logger.info("Confluence agent invoked: query=%s dry_run=%s", query, dry_run)
    try:
        # fetch grounding (we use the same retriever)
        from src.retriever import retrieve_grounding_for_query
        grounding = retrieve_grounding_for_query(query, top_k=8)
        # Build FRD text by concatenating top chunks
        frd_text = "\n\n".join([g.get("text", "") for g in grounding])
        # Call generator wrapper: returns dict with 'title' and 'html' and 'page_url' if pushed
        from confluence.confluence_generator import create_confluence_page_from_text
        # create_confluence_page_from_text(frd_text, title, dry_run)
        title = f"Help - {query[:60]}"
        if dry_run:
            # return generated HTML without pushing
            html = create_confluence_page_from_text(frd_text, title, push=False)
            return f"CONFLUENCE_DRYRUN_TITLE: {title}\n\n{html[:4000]}..."
        else:
            result = create_confluence_page_from_text(frd_text, title, push=True)
            return f"Confluence page created: {result.get('url') or str(result)}"
    except Exception as e:
        logger.exception("Confluence agent failed")
        return f"Confluence agent failed: {e}"


# Expose a mapping of tool names to functions (for router)
AGENT_TOOLS = {
    "rag": call_rag_tool,
    "generate_testcases": call_generate_testcases,
    "generate_confluence": call_generate_confluence
}
