# agents/rag_agent.py
"""
RAG Agent - Handles conversational Q&A using hybrid retrieval.
"""
from typing import Dict, Any
from agents.base import BaseAgent


class RAGAgent(BaseAgent):
    """RAG agent for document Q&A using vector + graph retrieval."""
    
    name = "rag"
    description = "Answer questions using document retrieval and LLM synthesis"
    
    def __init__(self, retriever=None, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute RAG retrieval and synthesis."""
        self.log(f"Processing query: {query[:50]}...")
        
        if self.retriever is None:
            # Try to get default retriever
            try:
                from core.retriever import HybridRetriever
                from core.db.vectorstore import PGVectorStore
                from core.embeddings import get_embeddings
                from core.llm import get_groq_client
                
                embedder = get_embeddings()
                vectorstore = PGVectorStore(embedder)
                llm = get_groq_client()
                self.retriever = HybridRetriever(
                    vectorstore=vectorstore,
                    llm_client=llm
                )
            except Exception as e:
                self.log(f"Failed to initialize retriever: {e}", "error")
                return {
                    "route": "rag",
                    "success": False,
                    "error": f"Retriever not available: {e}"
                }
        
        try:
            result = self.retriever.retrieve(query)
            return result
        except Exception as e:
            self.log(f"Retrieval failed: {e}", "error")
            return {
                "route": "rag",
                "success": False,
                "error": str(e)
            }


# Convenience function for backward compatibility
def call_rag_tool(query: str, dry_run: bool = True) -> Dict[str, Any]:
    """Call RAG agent as a tool."""
    agent = RAGAgent(dry_run=dry_run)
    return agent.execute(query)
