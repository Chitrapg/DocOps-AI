# agents/confluence_agent.py
"""
Confluence Agent - Generates help documents and optionally pushes to Confluence.
"""
from typing import Dict, Any
from agents.base import BaseAgent


class ConfluenceAgent(BaseAgent):
    """Agent for generating Confluence help pages."""
    
    name = "confluence"
    description = "Generate Confluence help documents from FRD"
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Generate Confluence page from grounding documents."""
        self.log(f"Generating Confluence page for: {query[:50]}...")
        
        # Get grounding from retriever
        try:
            from core.retriever import retrieve_grounding_for_query
            grounding = retrieve_grounding_for_query(query, top_k=8)
            self.log(f"Retrieved {len(grounding)} grounding chunks")
        except Exception as e:
            self.log(f"Grounding retrieval failed: {e}", "warning")
            grounding = []
        
        if not grounding:
            return {
                "route": "generate_confluence",
                "success": True,
                "result": {
                    "html": "<p>No documents found. Please ingest documents first.</p>",
                    "error": "No grounding documents available."
                }
            }
        
        # Generate HTML content
        try:
            from integrations.confluence import create_confluence_page_from_grounding
            title = f"Help - {query[:60]}"
            push = not self.dry_run
            result = create_confluence_page_from_grounding(grounding, title, push=push)
            
            self.log(f"Generation complete. Push={push}, has_url={'page_url' in result}")
            
            return {
                "route": "generate_confluence",
                "success": True,
                "result": result
            }
        except Exception as e:
            self.log(f"Generation failed: {e}", "error")
            return {
                "route": "generate_confluence",
                "success": False,
                "error": str(e)
            }


# Convenience function for backward compatibility
def call_generate_confluence(query: str, dry_run: bool = True) -> Dict[str, Any]:
    """Call confluence agent as a tool."""
    agent = ConfluenceAgent(dry_run=dry_run)
    return agent.execute(query)
