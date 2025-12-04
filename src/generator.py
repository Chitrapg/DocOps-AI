# src/generator.py
import uuid
import json
import re
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

from src.groq_llm import GroqClient
from src.db import store_generation

PROMPT_TPL_PATH = Path(__file__).parent / "prompt_template.txt"

# Recovery settings
MAX_TOKENS_MAIN = 4000
MAX_TOKENS_COMPLETE = 1500
RETRY_SLEEP_SECONDS = 1.5

def build_prompt(grounding_chunks: list, user_request: str) -> str:
    """
    Loads the prompt template and injects grounding + user request.
    Additionally *prepends* a strict instruction block that forces compact JSON-only output.
    """
    base_tpl = PROMPT_TPL_PATH.read_text(encoding="utf-8")
    grounding_text = "\n\n".join([f"CHUNK {i+1}:\n{g['text']}" for i, g in enumerate(grounding_chunks)])

    strict_header = (
        "IMPORTANT INSTRUCTIONS (must follow exactly):\n"
        "1) Output ONLY a single valid JSON array (e.g. [ {...}, {...} ]) that strictly follows the TestCase schema.\n"
        "2) Do NOT output any explanatory text, headings, or backticks — JSON only.\n"
        "3) Keep JSON compact (no unnecessary whitespace) so it fits in one response.\n"
        "4) If you cannot generate full JSON due to token limits, return the JSON you can, still valid (complete array).\n\n"
    )

    tpl = strict_header + base_tpl
    tpl = tpl.replace("{grounding_chunks}", grounding_text)
    tpl = tpl.replace("{user_request}", user_request)
    return tpl

def _extract_json_from_text(text: str) -> Any:
    """
    Robust JSON extraction attempts:
    1) Try direct json.loads(text)
    2) Search for first '[' ... matching ']' and attempt to parse that slice
    3) Raise ValueError if no valid JSON found
    """
    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) look for first '[' and try to find a matching ending ']' (balanced)
    start = text.find('[')
    if start == -1:
        raise ValueError("No JSON array start '[' found in text.")

    depth = 0
    end_idx = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx != -1:
        candidate = text[start:end_idx + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # try to make a best-effort repair (remove trailing commas before closing)
            repaired = re.sub(r',\s*([\]\}])', r'\1', candidate)
            try:
                return json.loads(repaired)
            except Exception:
                raise ValueError("Found bracketed JSON but could not parse it after repair.")

    raise ValueError("No valid JSON array could be parsed from LLM output.")

def _ask_llm_to_complete_json(partial_json_text: str, groq: GroqClient) -> str:
    """
    Given a partial JSON string (truncated), call the LLM with a short instruction
    to complete/repair it and return the LLM raw text response.
    This prompt MUST instruct the model to return ONLY valid JSON (no commentary).
    """
    completion_prompt = (
        "You are given a PARTIAL JSON array that was truncated. "
        "Your job: return a single VALID JSON array (and nothing else) that completes/fixes the partial JSON. "
        "Do not include any text before or after the JSON. If you cannot reconstruct missing fields, "
        "fill them with reasonable defaults (e.g., empty arrays, empty strings). "
        "Partial JSON below:\n\n"
        + partial_json_text + "\n\n"
        "Return a valid JSON array only."
    )
    # request completion with smaller token budget
    return groq.generate(completion_prompt, max_tokens=MAX_TOKENS_COMPLETE, temperature=0.0)

def _validate_and_normalize(tc: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Ensure required fields exist and normalize types.
    """
    # Provide defaults for fields if missing
    return {
        "id": str(tc.get("id") or f"TC-{idx+1:03d}"),
        "title": str(tc.get("title") or ""),
        "category": str(tc.get("category") or ""),
        "priority": str(tc.get("priority") or "Medium"),
        "preconditions": tc.get("preconditions") or [],
        "steps": tc.get("steps") or [],
        "expected_result": str(tc.get("expected_result") or ""),
        "test_data": tc.get("test_data"),
        "related_requirement_ids": tc.get("related_requirement_ids") or [],
        "tags": tc.get("tags") or []
    }

def testcases_to_markdown_table(tcs: List[Dict[str, Any]]) -> str:
    """
    Render the parsed testcases into the markdown table (same logic as previous version).
    """
    if not tcs:
        header = "| ID | Title | Category | Priority | Preconditions | Steps | Expected Result | Test Data | Related Req IDs | Tags |\n"
        sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        return header + sep

    header = "| ID | Title | Category | Priority | Preconditions | Steps | Expected Result | Test Data | Related Req IDs | Tags |\n"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"

    def _row(tc):
        def safe_join_list(value, sep=", "):
            if value is None:
                return ""
            if isinstance(value, list):
                return sep.join(str(x) for x in value)
            return str(value)

        id_ = tc.get("id","")
        title = tc.get("title","")
        category = tc.get("category","")
        priority = tc.get("priority","")
        preconds = safe_join_list(tc.get("preconditions", []), sep="; ")
        steps_list = tc.get("steps", [])
        if isinstance(steps_list, list):
            steps_md = "<br>".join(f"{i+1}. {s}" for i,s in enumerate(steps_list))
        else:
            steps_md = str(steps_list)
        expected = tc.get("expected_result","")
        test_data = tc.get("test_data") or ""
        related = safe_join_list(tc.get("related_requirement_ids", []), sep=", ")
        tags = safe_join_list(tc.get("tags", []), sep=", ")
        cells = [id_, title, category, priority, preconds, steps_md, expected, str(test_data), related, tags]
        cells = [str(c).replace("|", "&#124;") for c in cells]
        return "| " + " | ".join(cells) + " |"

    rows = [_row(tc) for tc in tcs]
    return header + sep + "\n".join(rows)

def generate_testcases(grounding_chunks: list, user_request: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Main entrypoint:
      - Calls LLM with strict JSON instructions
      - Attempts to parse JSON; if truncated, tries recovery via extraction and completion
      - Returns (generation_id, markdown_table, parsed_testcases_list)
    """
    prompt = build_prompt(grounding_chunks, user_request)
    gen_id = str(uuid.uuid4())
    groq = GroqClient()

    # Primary call (ask for compact JSON-only output)
    try:
        raw = groq.generate(prompt, max_tokens=MAX_TOKENS_MAIN, temperature=0.0)
    except Exception as e:
        # store error for audit
        try:
            store_generation(gen_id, prompt, grounding_chunks, f"LLM-ERROR: {str(e)}", metadata={"user_request": user_request, "error": str(e)})
        except Exception:
            pass
        err_md = testcases_to_markdown_table([]) + f"\n\n**LLM call failed:** {str(e)}\n\nPlease check network / endpoint / API key."
        return gen_id, err_md, []

    # Try to parse JSON directly or with extraction
    parsed_list = None
    parse_error = None
    try:
        parsed = _extract_json_from_text(raw)
        # ensure it's a list of objects
        if isinstance(parsed, dict):
            parsed_list = [parsed]
        elif isinstance(parsed, list):
            parsed_list = parsed
        else:
            raise ValueError("Parsed JSON is not an object or array.")
    except Exception as e:
        parse_error = str(e)
        parsed_list = None

    # If parsing failed, try recovery: extract best bracketed substring and try to complete with LLM
    if parsed_list is None:
        # attempt to find the longest bracketed slice (first '[' -> last ']' as best-effort)
        start = raw.find('[')
        end = raw.rfind(']')
        partial = None
        if start != -1 and end != -1 and end > start:
            partial = raw[start:end+1]
        elif start != -1:
            partial = raw[start:]  # truncated end
        # If we have partial JSON, ask LLM to complete it
        if partial:
            # store partial attempt
            try:
                store_generation(gen_id, prompt, grounding_chunks, raw, metadata={"user_request": user_request, "parse_error": parse_error, "partial_attempt": partial})
            except Exception:
                pass

            # Small sleep to avoid immediate throttling
            time.sleep(RETRY_SLEEP_SECONDS)

            try:
                completion_raw = _ask_llm_to_complete_json(partial, groq)
            except Exception as e:
                # failed to complete
                try:
                    store_generation(gen_id, prompt, grounding_chunks, raw, metadata={"user_request": user_request, "parse_error": parse_error, "complete_error": str(e)})
                except Exception:
                    pass
                err_md = testcases_to_markdown_table([]) + "\n\n" + "LLM returned non-JSON and completion attempt failed.\n\n" + "Raw output:\n\n```\n" + raw + "\n```\n"
                return gen_id, err_md, []

            # Try parse the completion result
            try:
                parsed = _extract_json_from_text(completion_raw)
                if isinstance(parsed, dict):
                    parsed_list = [parsed]
                else:
                    parsed_list = parsed
            except Exception as e:
                # final failure: save and return helpful error
                try:
                    store_generation(gen_id, prompt, grounding_chunks, raw, metadata={"user_request": user_request, "parse_error": parse_error, "completion_raw": completion_raw, "completion_error": str(e)})
                except Exception:
                    pass
                err_md = testcases_to_markdown_table([]) + "\n\n" + "LLM returned non-JSON or unparsable output even after completion attempt. Raw output:\n\n```\n" + raw + "\n```\n\nCompletion attempt result:\n\n```\n" + completion_raw + "\n```\n"
                return gen_id, err_md, []

        else:
            # no partial JSON to try
            try:
                store_generation(gen_id, prompt, grounding_chunks, raw, metadata={"user_request": user_request, "parse_error": parse_error})
            except Exception:
                pass
            err_md = testcases_to_markdown_table([]) + "\n\n" + "LLM returned non-JSON and no partial JSON could be extracted. Raw output:\n\n```\n" + raw + "\n```\n"
            return gen_id, err_md, []

    # At this point parsed_list is available and should be a list of dicts
    validated = []
    for i, tc in enumerate(parsed_list):
        if not isinstance(tc, dict):
            continue
        validated.append(_validate_and_normalize(tc, i))

    md_table = testcases_to_markdown_table(validated)

    # store successful generation (store raw LLM output for audit)
    try:
        store_generation(gen_id, prompt, grounding_chunks, raw, metadata={"user_request": user_request})
    except Exception:
        pass

    return gen_id, md_table, validated
