# core/testcase_generator.py
"""
Test case generation from grounding documents.
Simplified from src/generator.py
"""
import uuid
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

from core.llm.groq_http import GroqHTTPClient


def generate_testcases(grounding: List[Dict[str, Any]], query: str) -> Tuple[str, str, List[Dict]]:
    """
    Generate test cases from grounding documents.
    
    Args:
        grounding: List of grounding chunks with 'text' key
        query: User query describing what to test
        
    Returns:
        Tuple of (generation_id, markdown_table, parsed_testcases)
    """
    gen_id = str(uuid.uuid4())[:8]
    
    if not grounding:
        return gen_id, "No grounding documents available.", []
    
    # Build context from grounding
    context = "\n\n".join([g.get("text", "") for g in grounding if g.get("text")])
    
    if not context.strip():
        return gen_id, "Grounding documents had no text content.", []
    
    # Generate test cases using LLM
    try:
        client = GroqHTTPClient()
    except Exception as e:
        return gen_id, f"LLM not available: {e}", []
    
    prompt = f"""Generate software test cases based on the following functional requirements.

FUNCTIONAL REQUIREMENTS:
{context[:6000]}

USER REQUEST: {query}

Generate test cases in this exact format for each test case:
| Test Case ID | Business Scenario | Preconditions | Test Steps | Expected Result | Test Data |

Create realistic, detailed test cases. Output ONLY a markdown table with the following columns:
- Test Case ID (e.g., TC001)
- Business Scenario (what is being tested)
- Preconditions (what must be true before test)
- Test Steps (numbered steps)
- Expected Result (what should happen)
- Test Data (sample data for the test)
"""

    try:
        response = client.generate(prompt, max_tokens=2000, temperature=0.2)
    except Exception as e:
        return gen_id, f"Generation failed: {e}", []
    
    # Parse the response
    md_table = response if isinstance(response, str) else str(response)
    parsed = _parse_testcase_table(md_table)
    
    return gen_id, md_table, parsed


def _parse_testcase_table(md_table: str) -> List[Dict[str, Any]]:
    """Parse markdown table into list of test case dicts."""
    testcases = []
    lines = md_table.strip().split("\n")
    
    # Find table rows (lines starting with |)
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        if "---" in line:  # Skip separator row
            continue
        if "Test Case ID" in line:  # Skip header
            continue
        
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]  # Remove empty parts
        
        if len(parts) >= 5:
            testcases.append({
                "testcase_id": parts[0] if len(parts) > 0 else f"TC{len(testcases)+1:03d}",
                "business_scenario": parts[1] if len(parts) > 1 else "",
                "preconditions": parts[2] if len(parts) > 2 else "",
                "test_steps": parts[3] if len(parts) > 3 else "",
                "expected_result": parts[4] if len(parts) > 4 else "",
                "test_data_notes": parts[5] if len(parts) > 5 else ""
            })
    
    return testcases
