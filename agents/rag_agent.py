# agents/rag_agent.py
"""
RAG Agent - Handles conversational Q&A using hybrid retrieval.
Uses GraphCypherQAChain for proper graph-based answers.
"""
from typing import Dict, Any, List, Optional
import logging
import os
from agents.base import BaseAgent

# Load env
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """RAG agent for document Q&A using vector + graph retrieval."""
    
    name = "rag"
    description = "Answer questions using document retrieval and LLM synthesis"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = None
        self._graph_chain = None
        self._llm = None
    
    def _init_graph_chain(self):
        """Initialize GraphCypherQAChain for proper graph queries."""
        if self._graph_chain is not None:
            return self._graph_chain
        
        try:
            from langchain_groq import ChatGroq
            from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
            
            # Create Neo4j graph connection
            graph = Neo4jGraph(
                url=os.getenv("NEO4J_URL"),
                username=os.getenv("NEO4J_USER"),
                password=os.getenv("NEO4J_PASSWORD"),
            )
            
            # Refresh schema so LLM knows the graph structure
            try:
                graph.refresh_schema()
                self.log(f"Neo4j Schema loaded: {graph.schema[:200]}...")
            except Exception as e:
                self.log(f"Schema refresh failed: {e}", "warning")
            
            # Create LLM for Cypher generation
            llm = ChatGroq(model_name="llama-3.3-70b-versatile")
            cypher_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
            
            # Create GraphCypherQAChain
            self._graph_chain = GraphCypherQAChain.from_llm(
                llm=llm,
                cypher_llm=cypher_llm,
                graph=graph,
                verbose=False,
                allow_dangerous_requests=True,
                validate_cypher=True,
            )
            
            self.log("GraphCypherQAChain initialized successfully")
            return self._graph_chain
            
        except Exception as e:
            self.log(f"GraphCypherQAChain init failed: {e}", "error")
            return None
    
    def _get_vector_contexts(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve contexts from vector database."""
        try:
            from core.embeddings import get_embeddings
            from core.db.vectorstore import PGVectorStore
            import hashlib
            
            if self._vectorstore is None:
                embedder = get_embeddings()
                self._vectorstore = PGVectorStore(embedder)
            
            results = self._vectorstore.similarity_search(query, k=k)
            contexts = []
            seen = set()
            
            for doc in results:
                content = getattr(doc, 'page_content', str(doc))
                # Dedupe by hash
                h = hashlib.sha256(content.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                
                meta = getattr(doc, 'metadata', {})
                contexts.append({
                    'text': content,
                    'source': meta.get('source', 'unknown'),
                    'type': 'vector'
                })
            
            self.log(f"Retrieved {len(contexts)} vector contexts")
            return contexts
        except Exception as e:
            self.log(f"Vector retrieval failed: {e}", "warning")
            return []
    
    def _get_graph_answer(self, query: str) -> str:
        """Get answer from graph using GraphCypherQAChain."""
        chain = self._init_graph_chain()
        if chain is None:
            return ""
        
        try:
            resp = chain.invoke({"query": query})
            
            # Normalize response to string
            if isinstance(resp, str):
                return resp
            if isinstance(resp, dict):
                for key in ("answer", "result", "output", "text"):
                    if key in resp and resp[key]:
                        return str(resp[key])
                return str(resp)
            return str(resp)
            
        except Exception as e:
            self.log(f"Graph QA failed: {e}", "warning")
            return ""
    
    def _synthesize_answer(self, query: str, graph_answer: str, vector_contexts: List[Dict]) -> str:
        """Synthesize final answer from graph and vector results."""
        try:
            from core.llm.groq_langchain import get_groq_client
            
            groq = get_groq_client()
            
            # Build synthesis prompt
            prompt_parts = [
                "You are a helpful assistant. Answer the user's question directly and concisely.",
                f"User question: {query}",
                "",
            ]
            
            # Add graph answer if available
            if graph_answer:
                prompt_parts.append("Knowledge Graph Answer:")
                prompt_parts.append(graph_answer)
                prompt_parts.append("")
            
            # Add vector contexts if available
            if vector_contexts:
                prompt_parts.append("Document Contexts:")
                for i, c in enumerate(vector_contexts[:3], 1):
                    snippet = c.get("text", "")[:500]
                    prompt_parts.append(f"[{i}] {snippet}")
                prompt_parts.append("")
            
            prompt_parts.append(
                "Task: Give a direct, concise answer. "
                "Do NOT explain which sources you used or didn't use. "
                "Do NOT say 'based on the graph' or 'from the contexts'. "
                "Just answer the question directly."
            )
            
            prompt = "\n".join(prompt_parts)
            answer = groq.generate(prompt)
            return answer.strip() if isinstance(answer, str) else str(answer)
            
        except Exception as e:
            self.log(f"Synthesis failed: {e}", "error")
            # Fallback - return graph answer directly if available
            return graph_answer or "Error generating answer."
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute RAG retrieval and synthesis."""
        self.log(f"Processing query: {query[:50]}...")
        
        # 1. Get graph answer (using GraphCypherQAChain)
        graph_answer = self._get_graph_answer(query)
        self.log(f"Graph answer: {graph_answer[:100] if graph_answer else 'None'}...")
        
        # 2. Get vector contexts
        vector_contexts = self._get_vector_contexts(query, k=5)
        
        # 3. Synthesize final answer
        final_answer = self._synthesize_answer(query, graph_answer, vector_contexts)
        
        return {
            "route": "rag",
            "success": True,
            "final_answer": final_answer,
            "graph_answer": graph_answer,
            "vector_count": len(vector_contexts),
            "graph_count": 1 if graph_answer else 0,
            "contexts": {
                "vector": vector_contexts[:3],
                "graph": [{"text": graph_answer, "source": "GraphCypherQA", "type": "graph"}] if graph_answer else []
            }
        }


# Convenience function for backward compatibility
def call_rag_tool(query: str, dry_run: bool = True) -> Dict[str, Any]:
    """Call RAG agent as a tool."""
    agent = RAGAgent(dry_run=dry_run)
    return agent.execute(query)
