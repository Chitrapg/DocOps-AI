# core/retriever.py
"""
Unified retriever module with hybrid vector + graph search.
Merged from app/retriever.py and src/retriever.py
"""
from typing import List, Dict, Any, Optional
import logging

from core.config import settings
from core.embeddings import embed_text, get_embeddings

logger = logging.getLogger(__name__)


def retrieve_grounding_for_query(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Simple vector-only retrieval for grounding.
    Used by testcase and confluence agents.
    
    Returns list of dicts with 'id', 'text', 'metadata' keys.
    """
    try:
        from core.db.postgres import similarity_search
        q_emb = embed_text(query)
        rows = similarity_search(q_emb, top_k=top_k)
        grounding = []
        for r in rows:
            grounding.append({
                'id': r.get('id'),
                'text': r.get('text'),
                'metadata': r.get('metadata')
            })
        return grounding
    except Exception as e:
        logger.warning(f"Grounding retrieval failed: {e}")
        return []


class HybridRetriever:
    """
    Hybrid retriever combining vector search and graph queries.
    Used by RAG agent for comprehensive Q&A.
    """
    
    def __init__(self, vectorstore=None, graphstore=None, llm_client=None):
        self.vectorstore = vectorstore
        self.graphstore = graphstore
        self.llm_client = llm_client
    
    def get_vector_contexts(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get contexts from vector store."""
        if self.vectorstore is None:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            contexts = []
            seen = set()
            
            for doc in results:
                content = getattr(doc, 'page_content', str(doc))
                if content in seen:
                    continue
                seen.add(content)
                
                meta = getattr(doc, 'metadata', {})
                contexts.append({
                    'text': content,
                    'meta': meta
                })
            return contexts
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
    
    def get_graph_answer(self, query: str) -> str:
        """Get answer from graph database using Cypher."""
        if self.graphstore is None:
            return ""
        
        try:
            # Use graph chain if available
            result = self.graphstore.query(query)
            if isinstance(result, dict):
                return result.get('result', str(result))
            return str(result) if result else ""
        except Exception as e:
            logger.warning(f"Graph query failed: {e}")
            return ""
    
    def synthesize(self, query: str, graph_answer: str, vector_contexts: List[Dict]) -> str:
        """Synthesize final answer from graph and vector results."""
        if self.llm_client is None:
            # Return concatenated contexts if no LLM
            parts = []
            if graph_answer:
                parts.append(f"Graph: {graph_answer}")
            for ctx in vector_contexts:
                parts.append(ctx.get('text', ''))
            return "\n\n".join(parts)
        
        # Build synthesis prompt
        context_text = "\n\n".join([ctx.get('text', '') for ctx in vector_contexts])
        
        prompt = f"""Answer the question based on the following context.

GRAPH KNOWLEDGE:
{graph_answer or '(No graph data available)'}

DOCUMENT CONTEXT:
{context_text or '(No document context available)'}

QUESTION: {query}

ANSWER:"""
        
        try:
            return self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Error generating answer: {e}"
    
    def retrieve(self, query: str, k_vector: int = 5) -> Dict[str, Any]:
        """
        Full hybrid retrieval: vector + graph + synthesis.
        
        Returns dict with:
            - 'final_answer': Synthesized answer
            - 'vector_contexts': Retrieved document chunks
            - 'graph_answer': Answer from graph
        """
        # Get vector contexts
        vector_contexts = self.get_vector_contexts(query, k=k_vector)
        
        # Get graph answer
        graph_answer = self.get_graph_answer(query)
        
        # Synthesize
        final_answer = self.synthesize(query, graph_answer, vector_contexts)
        
        return {
            'final_answer': final_answer,
            'vector_contexts': vector_contexts,
            'graph_answer': graph_answer,
            'route': 'rag',
            'success': True
        }
