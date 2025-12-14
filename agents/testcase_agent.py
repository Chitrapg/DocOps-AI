# agents/testcase_agent.py
"""
Testcase Agent - Generates test cases and optionally pushes to Jira.
"""
from typing import Dict, Any, List, Optional
from agents.base import BaseAgent


class TestcaseAgent(BaseAgent):
    """Agent for generating test cases from documents."""
    
    name = "testcase"
    description = "Generate test cases from FRD documents"
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Generate test cases from grounding documents."""
        self.log(f"Generating test cases for: {query[:50]}...")
        
        # Get grounding from retriever
        try:
            from core.retriever import retrieve_grounding_for_query
            grounding = retrieve_grounding_for_query(query, top_k=8)
            self.log(f"Retrieved {len(grounding)} grounding chunks")
        except Exception as e:
            self.log(f"Grounding retrieval failed: {e}", "warning")
            grounding = []
        
        # Generate test cases
        try:
            from src.generator import generate_testcases
            gen_id, md_table, parsed = generate_testcases(grounding, query)
        except Exception as e:
            self.log(f"Generation failed: {e}", "error")
            return {
                "route": "generate_testcases",
                "success": False,
                "error": f"Generation failed: {e}"
            }
        
        parsed = parsed or []
        result = {
            "route": "generate_testcases",
            "success": True,
            "gen_id": gen_id,
            "md_table": md_table,
            "parsed_count": len(parsed),
            "parsed": parsed,
            "created": [],
            "errors": []
        }
        
        # If dry run, return preview
        if self.dry_run:
            return result
        
        # Push to Jira
        self.log("Pushing to Jira...")
        try:
            from integrations.jira import push_testcases_to_jira
            created, errors = push_testcases_to_jira(parsed)
            result["created"] = created
            result["errors"] = errors
        except Exception as e:
            result["errors"].append(str(e))
            self.log(f"Jira push failed: {e}", "error")
        
        return result


# Convenience function for backward compatibility
def call_generate_testcases(query: str, dry_run: bool = True) -> Dict[str, Any]:
    """Call testcase agent as a tool."""
    agent = TestcaseAgent(dry_run=dry_run)
    return agent.execute(query)
