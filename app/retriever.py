import hashlib
import os
from typing import List, Dict, Any

import streamlit as st
from langchain_groq import ChatGroq
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from app.vectorstore import PGVectorStore
from app.embeddings import Embeddings
from app.graphstore import Neo4jStore
from app.groq_client import get_groq_client  # optional safe client init

# -------------------
# Setup - reuse your existing vector store
# -------------------
embedder = Embeddings()
pg = PGVectorStore(embedder)

# Neo4j store (used by GraphCypherQAChain internally; we also keep Neo4jStore available)
neo = Neo4jStore()


# -------------------
# LLM and Graph chain (KG)
# -------------------
# Instantiate LLM used both by GraphCypherQAChain and synthesis step.
# If you prefer to centralize LLM init, change to use get_groq_client().llm where available.
try:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    cypher_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
except Exception:
    # Fallback: try to get client wrapper that exposes .llm
    try:
        client = get_groq_client()
        llm = client.llm
    except Exception:
        llm = None

# Create langchain Neo4j graph adapter (used by GraphCypherQAChain)
graph = Neo4jGraph(
    url=None if os.getenv("NEO4J_URL") is None else os.getenv("NEO4J_URL"),
    username=None if os.getenv("NEO4J_USER") is None else os.getenv("NEO4J_USER"),
    password=None if os.getenv("NEO4J_PASSWORD") is None else os.getenv("NEO4J_PASSWORD"),
)

# GraphCypherQAChain - prefer read-only answers from your KG
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    cypher_llm=cypher_llm,
    graph=graph,
    verbose=False,
    allow_dangerous_requests=True,
    validate_cypher=True,
)


# 🔍 Debug Neo4j Schema Visibility
try:
    graph.refresh_schema()
    print("\n=============[ Neo4j Schema Loaded ]==============")
    print(graph.schema)
    print("==================================================\n")

    # Optional: show in Streamlit debug UI
    try:
        import streamlit as st
        st.write("### Neo4j Schema (Debug)")
        st.code(graph.schema)
    except:
        pass
except Exception as e:
    print("[Neo4jGraph] Failed to refresh schema:", e)

# --- only the Retriever class and singleton need to be replaced/updated ---
from app.groq_client import get_groq_client  # ensure import at top of file

class Retriever:
    """
    Hybrid retriever with fixes:
      - normalize graph answer (dict -> string)
      - dedupe vector contexts
      - use get_groq_client().generate(...) for synthesis (stable API)
    """

    def __init__(self, pg: PGVectorStore, graph_chain: GraphCypherQAChain, llm):
        self.pg = pg
        self.graph_chain = graph_chain
        self.llm = llm

    def get_vector_contexts(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.pg.similarity_search(query, k=k)
            contexts = [
                {"source": "vector", "content": r.page_content, "meta": r.metadata}
                for r in results
            ]
            # dedupe by content hash while preserving order
            seen = set()
            unique = []
            for c in contexts:
                text = (c.get("content") or "").strip()
                if not text:
                    continue
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                unique.append(c)
            return unique
        except Exception as e:
            st.warning(f"Vector search failed: {e}")
            return []

    def graph_answer(self, query: str) -> str:
        try:
            resp = self.graph_chain.invoke({"query": query})
            # Normalize resp to a plain string for later prompt construction
            if isinstance(resp, str):
                return resp
            if isinstance(resp, dict):
                # common keys: 'answer', 'result', 'output'
                for k in ("answer", "result", "output", "text"):
                    if k in resp and resp[k]:
                        return str(resp[k])
                # some chains return {'query':..., 'result':...}
                if "result" in resp:
                    return str(resp["result"])
                # otherwise stringify the dict
                return str(resp)
            # fallback
            return str(resp)
        except Exception as e:
            print(f"[GraphCypherQA] Graph QA failed: {e}")
            st.warning(f"Graph QA failed: {e}")
            return ""

    def synthesize(self, query: str, graph_ans: str, vector_contexts: List[Dict[str, Any]]) -> str:
        """
        Use the configured groq client (get_groq_client().generate) for synthesis to avoid
        mismatched LLM wrapper shapes. Build a robust prompt and call the groq client.
        """

        # compact prompt
        prompt_parts = [
            "You are a helpful assistant that merges structured KG facts with textual evidence.",
            f"User question: {query}",
            "",
            "Graph-derived answer (facts from the knowledge graph):",
            graph_ans or "(no graph answer available)",
            "",
        ]

        if vector_contexts:
            prompt_parts.append("Top textual contexts (from document vector search):")
            for i, c in enumerate(vector_contexts[:5], start=1):
                snippet = (c.get("content") or "").strip()
                # truncate to reasonable length
                snippet = snippet[:800]
                prompt_parts.append(f"[{i}] {snippet}")
            prompt_parts.append("")

        prompt_parts.append(
            "Task: Provide a concise final answer to the user's question. "
            "Use the graph facts as authoritative for named entities/relations. "
            "If you use evidence from the contexts, indicate the context index in square brackets (e.g., [1]). "
            "If the information isn't present, say you don't have enough information."
        )

        prompt = "\n".join(prompt_parts)

        # Use the same groq client used elsewhere in your app for consistency
        try:
            groq_client = get_groq_client()
            # groq_client.generate returns string in your other code path
            out = groq_client.generate(prompt)
            if isinstance(out, str):
                return out.strip()
            # some clients may return objects
            if isinstance(out, dict):
                return out.get("text") or out.get("output") or str(out)
            return str(out)
        except Exception as e:
            st.warning(f"Synthesis generation failed: {e}. Returning graph answer + contexts.")
            # fallback: graph answer + top 3 contexts
            parts = [graph_ans or "(no graph answer)"]
            for i, c in enumerate(vector_contexts[:3], start=1):
                parts.append(f"[{i}] " + (c.get("content") or ""))
            return "\n\n".join(parts)

    def retrieve(self, query: str, k_vector: int = 5) -> Dict[str, Any]:
        # 1) KG answer
        graph_resp = self.graph_answer(query)
        print(graph_resp)

        # 2) Vector contexts (deduped inside get_vector_contexts)
        vector_contexts = self.get_vector_contexts(query, k=k_vector)
        print(vector_contexts)

        # 3) Synthesize final answer
        final_answer = self.synthesize(query, graph_resp, vector_contexts)

        # Streamlit debug display
        try:
            st.write("### Hybrid RAG — Results")

            # st.markdown("**Vector contexts (top):**")
            # for i, c in enumerate(vector_contexts, start=1):
            #     snippet = (c.get("content") or "")[:500]
            #     st.markdown(f"- [{i}] {snippet}")
        except Exception:
            pass

        return {
            "graph_answer": graph_resp,
            "vector_contexts": vector_contexts,
            "final_answer": final_answer
        }

# singleton (replace existing)
retriever = Retriever(pg=pg, graph_chain=graph_chain, llm=llm)
