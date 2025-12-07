# orchestrator/orchestrator.py  (replace or add these functions)

import inspect
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# If you already import AGENT_TOOLS from orchestrator.agents, keep that import above.
# from orchestrator.agents import AGENT_TOOLS

def _invoke_tool_any(tool_obj: Any, *args, **kwargs) -> Any:
    """
    Robustly call a "tool" regardless of LangChain version.
    Tries, in order:
      - direct call tool_obj(*args, **kwargs)
      - tool_obj.run(*args, **kwargs)
      - tool_obj.func(*args, **kwargs)   (some wrappers expose .func)
      - tool_obj.invoke(*args, **kwargs)
      - tool_obj.__call__(*args, **kwargs) (fallback)
    Raises the original exception if nothing works.
    """
    # 1) Try direct call first (works if decorator returned a function)
    try:
        return tool_obj(*args, **kwargs)
    except TypeError as e_call:
        # keep the exception to potentially surface if nothing else works
        last_exc = e_call
    except Exception as e:
        # Some tools raise other exceptions internally; re-raise those (tool executed)
        raise

    # 2) Try common call methods
    for attr in ("run", "func", "invoke", "__call__"):
        fn = getattr(tool_obj, attr, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError as e_ty:
                # This method exists but expected different args; try next
                last_exc = e_ty
                continue
            except Exception:
                # tool ran but raised error -> bubble up
                raise

    # 3) For StructuredTool-like objects that wrap an inner function under .tool or .tool.func
    #    attempt common nested attribute patterns.
    nested_paths = [
        ("tool", "func"),
        ("tool", "run"),
        ("fn", "func"),
        ("func",),
    ]
    for path in nested_paths:
        cur = tool_obj
        ok = True
        for p in path:
            cur = getattr(cur, p, None)
            if cur is None:
                ok = False
                break
        if ok and callable(cur):
            try:
                return cur(*args, **kwargs)
            except TypeError as e_ty:
                last_exc = e_ty
                continue
            except Exception:
                raise

    # Nothing worked
    logger.debug("Invocation failure for tool object type: %s, attrs: %s", type(tool_obj), dir(tool_obj))
    raise last_exc


def _call_llm_generate_safe(llm_client: Any, prompt: str, **kwargs) -> Any:
    """
    Call an LLM client's `generate()` in a signature-safe way.

    Behavior:
      - Inspect the signature of llm_client.generate (or if llm_client itself is a callable,
        inspect that).
      - Construct a dict of only the supported keyword args from the provided kwargs.
      - Always pass prompt as first positional argument (safe across many wrappers).
    Supported optional kwargs: max_tokens, temperature, system_prompt, timeout
    """
    if llm_client is None:
        raise RuntimeError("LLM client is None")

    # Determine the callable to inspect: prefer .generate, else llm_client itself
    if hasattr(llm_client, "generate") and callable(getattr(llm_client, "generate")):
        fn = llm_client.generate
    elif callable(llm_client):
        fn = llm_client
    else:
        # try other possibilities: .invoke or .__call__
        if hasattr(llm_client, "invoke") and callable(llm_client.invoke):
            fn = llm_client.invoke
        elif hasattr(llm_client, "__call__") and callable(llm_client.__call__):
            fn = llm_client.__call__
        else:
            raise RuntimeError("No callable generate/invoke/__call__ found on LLM client")

    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())

    # map common names to preferred names the user passed
    candidate_kwargs = {}
    for k in ("max_tokens", "temperature", "system_prompt", "timeout"):
        if k in kwargs and (k in allowed or any(p.lower().startswith(k) for p in allowed)):
            # some wrappers use 'max_tokens' others 'max_tokens' etc — keep same name
            candidate_kwargs[k] = kwargs[k]

    # Some Groq wrappers expect 'system_prompt' as 'system_prompt' or expect 'prompt' only.
    # Call with prompt positional arg always.
    try:
        return fn(prompt, **candidate_kwargs)
    except TypeError as e:
        # Last resort: try calling with only prompt positional
        logger.debug("LLM generate signature mismatch, trying positional-only call: %s", e)
        try:
            return fn(prompt)
        except Exception as e2:
            logger.exception("LLM generate finally failed: %s", e2)
            raise


# import AGENT_TOOLS lazily to avoid circular import at module load time if needed
def _get_default_agent_tools() -> Dict[str, Any]:
    try:
        from orchestrator.agents import AGENT_TOOLS
    except Exception as e:
        logger.warning("Could not import AGENT_TOOLS from orchestrator.agents: %s", e)
        return {}
    return AGENT_TOOLS


def handle_user_input(user_input: str,
                      agent_tools: Optional[Dict[str, Any]] = None,
                      dry_run: bool = True,
                      chat_history: Optional[list] = None) -> Any:
    """
    Robust orchestrator entrypoint.

    - agent_tools: mapping name -> tool callable. If None, uses orchestrator.agents.AGENT_TOOLS.
    - dry_run: passed to tools where applicable.
    - chat_history: optional list of prior messages (not used by current simple router).
    """
    if agent_tools is None:
        agent_tools = _get_default_agent_tools()
        if not agent_tools:
            raise RuntimeError("No agent_tools available (AGENT_TOOLS import failed).")

    text = (user_input or "").strip().lower()

    # Simple keyword-based routing (replace with a router agent if desired)
    if any(k in text for k in ["test case", "testcase", "generate test", "generate testcases", "generate testcase"]):
        tool = agent_tools.get("generate_testcases") or agent_tools.get("GenerateTestcases")
    elif any(k in text for k in ["confluence", "help text", "help page", "generate help", "create confluence"]):
        tool = agent_tools.get("generate_confluence") or agent_tools.get("GenerateConfluence")
    else:
        tool = agent_tools.get("rag") or agent_tools.get("RAG")

    if tool is None:
        raise RuntimeError("No matching tool found for the input; available tools: " + ", ".join(agent_tools.keys()))

    # Try to call tool robustly. Tools often accept (query, dry_run) but not always.
    try:
        # first attempt: pass dry_run kwarg (most of our tools accept it)
        out = _invoke_tool_any(tool, user_input, dry_run=dry_run)
    except TypeError:
        # fallback: call with only the query
        try:
            out = _invoke_tool_any(tool, user_input)
        except Exception as e:
            logger.exception("Tool invocation failed (fallback without dry_run): %s", e)
            raise
    except Exception as e:
        logger.exception("Tool invocation failed: %s", e)
        raise

    # Return whatever the tool returned (string, dict, etc.)
    return out