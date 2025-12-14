
# core/ingest/pipeline.py
"""
Document ingestion pipeline - PDF/DOCX to vectors and graph.
Migrated from app/ingest.py
"""
import os
import uuid
import json
import re
import logging
from typing import List, Dict

# Load env
from dotenv import load_dotenv
load_dotenv()

from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use core modules instead of app modules
from core.embeddings import Embeddings
from core.db.vectorstore import PGVectorStore
from core.db.graphstore import Neo4jStore
from core.llm.groq_langchain import get_groq_client

# Optional: use LLMGraphTransformer to extract nodes/edges
try:
    from langchain_experimental.graph_transformers import LLMGraphTransformer
except Exception:
    LLMGraphTransformer = None

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
GRAPH_BATCH_SIZE = int(os.getenv("GRAPH_BATCH_SIZE", 6))

embedder = Embeddings()
pgstore = PGVectorStore(embedder)
neo4j = Neo4jStore()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# create a (singleton) groq client to avoid re-init many times
_groq_client = None

# silence noisy pdfminer font warnings (harmless for text extraction)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = get_groq_client()
    return _groq_client


def _get_llm_transformer():
    """
    Return an LLMGraphTransformer if available, constructed with the groq client's llm.
    """
    global _groq_client
    if LLMGraphTransformer is None:
        return None
    if _groq_client is None:
        _groq_client = get_groq_client()
    return LLMGraphTransformer(llm=_groq_client.llm)


def pdf_to_text(path: str) -> str:
    return extract_text(path)


# -------------------------
# Helpers: detect noisy chunks (TOC / page lists)
# -------------------------
def _looks_like_toc_or_list(text: str) -> bool:
    """
    Heuristic to detect TOC/table-of-contents or page-number lists that often confuse the LLM.
    Returns True if text likely is TOC-like.
    """
    if not text or len(text.strip()) < 60:
        # too short to be useful
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    # high proportion of lines that end with digits (page numbers)
    ends_with_digits = sum(1 for ln in lines if re.search(r"\b\d{1,4}$", ln))
    if ends_with_digits / max(1, len(lines)) > 0.4:
        return True
    # many short lines (< 8 words)
    short_lines = sum(1 for ln in lines if len(ln.split()) <= 6)
    if short_lines / max(1, len(lines)) > 0.6 and len(lines) > 5:
        return True
    # many dots like "Section ..... 123"
    dots = sum(1 for ln in lines if "..." in ln or "–" in ln or "—" in ln)
    if dots / max(1, len(lines)) > 0.2:
        return True
    return False


# -------------------------
# Robust JSON extraction from failed_generation / function-call wrappers
# -------------------------
def _find_balanced_braces(text: str, start_pos: int) -> str | None:
    """
    Given text and a start_pos pointing at the first '{', return the substring
    containing the balanced {...} block (handles nested braces). Returns None if not found.
    """
    if start_pos < 0 or start_pos >= len(text) or text[start_pos] != "{":
        return None
    depth = 0
    i = start_pos
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_pos:i + 1]
        i += 1
    return None  # unbalanced / not found


def _extract_json_from_failed_generation(raw_text: str) -> dict | None:
    """
    Robustly extract JSON object from model/tool wrapper or failed_generation text.

    Steps:
      - Look for '"arguments"' or '"failed_generation"' or 'arguments' literal
      - Find the first '{' after that position and extract balanced braces (handles nested)
      - Repair common issues: strip fences, remove trailing commas, convert single->double quotes if necessary
      - Try json.loads; if it fails attempt unicode-escape decoding then parse again
    """
    if not raw_text:
        return None

    text = str(raw_text).strip()

    candidate_positions = []

    # Look for "arguments" key (common wrapper)
    for m in re.finditer(r'["\']?arguments["\']?\s*:\s*', text, flags=re.I):
        after = m.end()
        brace_pos = text.find("{", after)
        if brace_pos != -1:
            candidate_positions.append(brace_pos)

    # Also try failed_generation key
    if not candidate_positions:
        for m in re.finditer(r'["\']?failed_generation["\']?\s*:\s*', text, flags=re.I):
            after = m.end()
            brace_pos = text.find("{", after)
            if brace_pos != -1:
                candidate_positions.append(brace_pos)

    # Fallback: first '{' in the text
    if not candidate_positions:
        brace_pos = text.find("{")
        if brace_pos != -1:
            candidate_positions.append(brace_pos)

    for pos in candidate_positions:
        block = _find_balanced_braces(text, pos)
        if not block:
            continue

        s = block.strip()

        # remove code fences if present
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
        # remove trailing commas
        s = re.sub(r",\s*([\]}])", r"\1", s)
        # naive single->double quote conversion if needed
        if ("'" in s) and ('"' not in s):
            s = s.replace("'", '"')

        # attempt parsing
        try:
            return json.loads(s)
        except Exception:
            # try unicode_escape decode
            try:
                s2 = s.encode("utf-8").decode("unicode_escape")
                s2 = re.sub(r'\\+"', '"', s2)
                s2 = re.sub(r"\\+'", '"', s2)
                s2 = re.sub(r",\s*([\]}])", r"\1", s2)
                return json.loads(s2)
            except Exception:
                # final fallback: find any { ... } inside s and parse
                m = re.search(r"\{[\s\S]*\}", s)
                if m:
                    try:
                        cand = m.group(0)
                        cand = re.sub(r",\s*([\]}])", r"\1", cand)
                        if ("'" in cand) and ('"' not in cand):
                            cand = cand.replace("'", '"')
                        return json.loads(cand)
                    except Exception:
                        continue
    return None


# -------------------------
# JSON repair & Groq fallback (strict prompt)
# -------------------------
def _repair_json_like(text: str) -> str:
    """
    Lightweight repair used by Groq fallback parsing.
    """
    if not text:
        return text

    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    if ("'" in s) and ('"' not in s):
        s = s.replace("'", '"')
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def _try_parse_json_from_text(text: str):
    """
    Try to parse JSON from text using repair heuristics.
    """
    try:
        return json.loads(text)
    except Exception:
        repaired = _repair_json_like(text)
        try:
            return json.loads(repaired)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", repaired)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None


def _extract_graph_via_groq(chunk_texts: List[str]) -> List[Dict]:
    """
    Fallback: ask Groq to emit a plain JSON `{"nodes":[...], "relationships":[...]}` for the given texts.
    Returns a list of graph_documents (same rough shape as transformer output) or an empty list on failure.
    """
    groq = _get_groq_client()
    joined = "\n\n".join([t[:3500] for t in chunk_texts])
    prompt = (
        "Extract a simple graph from the following text. "
        "Return ONLY a JSON object with exactly two keys: 'nodes' (array) and 'relationships' (array). "
        "Nodes must be objects with 'id' and 'label' and optional 'properties' dict. "
        "Relationships must be objects with 'source','target','label' and optional 'properties'. "
        "Do NOT include any explanation or text. Use only double-quoted JSON. "
        "If a page number appears, represent it as node id like 'Page 123' with label 'Page'.\n\n"
        f"TEXT:\n{joined}\n\nOUTPUT (JSON ONLY):"
    )

    try:
        gen = groq.generate(prompt)
    except Exception as e:
        print("Groq fallback generation failed:", e)
        return []

    out_text = gen if isinstance(gen, str) else str(gen)
    parsed = _try_parse_json_from_text(out_text)
    if not parsed:
        print("Groq fallback produced unparsable JSON.")
        return []

    nodes = parsed.get("nodes") or parsed.get("Nodes") or []
    rels = parsed.get("relationships") or parsed.get("Relationships") or parsed.get("edges") or []

    gd = {"nodes": [], "edges": []}
    for n in nodes:
        nid = n.get("id") or n.get("Id") or n.get("name") or n.get("title")
        label = n.get("label") or n.get("Label") or n.get("type") or "Entity"
        props = n.get("properties") or n.get("props") or {}
        if nid:
            gd["nodes"].append({"id": str(nid), "label": label, "properties": props})

    for r in rels:
        src = r.get("source") or r.get("source_node_id") or r.get("from")
        tgt = r.get("target") or r.get("target_node_id") or r.get("to")
        lab = r.get("label") or r.get("type") or r.get("relationship") or "RELATED_TO"
        props = r.get("properties") or r.get("props") or {}
        if src and tgt:
            gd["edges"].append({"source": str(src), "target": str(tgt), "label": lab, "properties": props})

    return [gd] if (gd["nodes"] or gd["edges"]) else []


# -------------------------
# Normalize GraphDocuments returned by transformers
# -------------------------
def _normalize_graph_documents(graph_docs_raw) -> List[Dict]:
    """
    Accepts:
      - list of dicts (already normalized)
      - list of GraphDocument objects with attributes .nodes/.edges or ._nodes/._edges
    Returns list of dicts in shape: {"nodes": [...], "edges": [...]}
    """
    normalized = []

    if not graph_docs_raw:
        return normalized

    for gd in graph_docs_raw:
        # direct dict-style
        if isinstance(gd, dict):
            nodes = gd.get("nodes") or gd.get("Nodes") or gd.get("node_list") or gd.get("elements") or []
            edges = gd.get("edges") or gd.get("Edges") or gd.get("relationships") or gd.get("Relations") or []
        else:
            # object-style GraphDocument: try common attributes
            nodes = getattr(gd, "nodes", None) or getattr(gd, "_nodes", None) or getattr(gd, "node_list", None) or []
            edges = getattr(gd, "edges", None) or getattr(gd, "_edges", None) or getattr(gd, "relationships", None) or []

            # Some GraphDocument implementations offer a to_dict method
            if (not nodes and not edges) and hasattr(gd, "to_dict"):
                try:
                    d = gd.to_dict()
                    nodes = d.get("nodes", []) or d.get("Nodes", [])
                    edges = d.get("edges", []) or d.get("relationships", []) or d.get("Edges", [])
                except Exception:
                    nodes = nodes or []
                    edges = edges or []

        # Normalize node entries
        norm_nodes = []
        try:
            for n in nodes or []:
                if n is None:
                    continue
                if isinstance(n, dict):
                    nid = n.get("id") or n.get("_id") or n.get("name") or n.get("title")
                    label = n.get("label") or n.get("type") or "Entity"
                    props = n.get("properties") or n.get("props") or {}
                    norm_nodes.append({"id": str(nid) if nid is not None else None, "label": label, "properties": props})
                else:
                    nid = getattr(n, "id", None) or getattr(n, "_id", None) or getattr(n, "name", None) or None
                    label = getattr(n, "label", None) or getattr(n, "type", None) or "Entity"
                    props = getattr(n, "properties", None) or getattr(n, "props", None) or {}
                    if hasattr(n, "get") and nid is None:
                        try:
                            nid = n.get("id") or n.get("_id") or n.get("name")
                        except Exception:
                            pass
                    norm_nodes.append({"id": str(nid) if nid is not None else None, "label": label, "properties": props})
        except Exception as e:
            print("Warning: failed to normalize some nodes:", e)

        # Normalize edges entries
        norm_edges = []
        try:
            for e in edges or []:
                if e is None:
                    continue
                if isinstance(e, dict):
                    src = e.get("source") or e.get("source_node_id") or e.get("from")
                    tgt = e.get("target") or e.get("target_node_id") or e.get("to")
                    lab = e.get("label") or e.get("type") or "RELATED_TO"
                    props = e.get("properties") or e.get("props") or {}
                    norm_edges.append({"source": str(src) if src is not None else None,
                                       "target": str(tgt) if tgt is not None else None,
                                       "label": lab,
                                       "properties": props})
                else:
                    src = getattr(e, "source", None) or getattr(e, "source_node_id", None) or getattr(e, "from", None)
                    tgt = getattr(e, "target", None) or getattr(e, "target_node_id", None) or getattr(e, "to", None)
                    lab = getattr(e, "label", None) or getattr(e, "type", None) or "RELATED_TO"
                    props = getattr(e, "properties", None) or getattr(e, "props", None) or {}
                    if hasattr(e, "get") and (src is None or tgt is None):
                        try:
                            src = src or e.get("source") or e.get("source_node_id") or e.get("from")
                            tgt = tgt or e.get("target") or e.get("target_node_id") or e.get("to")
                        except Exception:
                            pass
                    norm_edges.append({"source": str(src) if src is not None else None,
                                       "target": str(tgt) if tgt is not None else None,
                                       "label": lab,
                                       "properties": props})
        except Exception as e:
            print("Warning: failed to normalize some edges:", e)

        # Filter out nodes/edges with missing essential ids
        clean_nodes = [n for n in norm_nodes if n.get("id")]
        clean_edges = [r for r in norm_edges if r.get("source") and r.get("target")]

        if clean_nodes or clean_edges:
            normalized.append({"nodes": clean_nodes, "edges": clean_edges})

    return normalized


# -------------------------
# Main ingestion function (with robust fallback)
# -------------------------
def ingest_pdf(path: str, source_name: str = None):
    """
    Ingest PDF into vector DB and Neo4j.

    Steps:
      - extract text and split into chunks -> add to PG vectorstore
      - create a Doc node in Neo4j for the PDF (coarse)
      - naive topic extraction (existing behavior)
      - run LLMGraphTransformer on chunks in batches, with robust fallback to Groq JSON extraction
      - insert collected graph fragments into Neo4j (best-effort)
    """
    source_name = source_name or os.path.basename(path)
    try:
        full_text = pdf_to_text(path)
    except Exception as e:
        print("Error extracting text from PDF:", e)
        full_text = ""

    pages = full_text.split("\f") if full_text else []

    docs_to_add = []
    for i, page_text in enumerate(pages):
        if not page_text.strip():
            continue
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            docs_to_add.append({
                "content": chunk,
                "metadata": {"source": source_name, "page": i}
            })

    # add to PG vectorstore
    try:
        pgstore.add_documents(docs_to_add)
    except Exception as e:
        print("Failed to add documents to PG vectorstore:", e)

    # add coarse doc node to Neo4j (one per PDF)
    doc_id = str(uuid.uuid4())
    try:
        neo4j.create_doc_node(doc_id, source_name, (full_text[:300] if full_text else ""))
    except Exception as e:
        print("Failed creating doc node in Neo4j:", e)

    # naive topic extraction: unique long words (placeholder)
    try:
        words = set([w.lower().strip('.,()[]{}:;') for w in (full_text or "").split() if len(w) > 6])
        topics = list(words)[:30]
        for t in topics:
            try:
                neo4j.create_topic_node(t)
                neo4j.create_mention(doc_id, t)
            except Exception:
                pass
    except Exception:
        pass

    # -------------------
    # Graph extraction via LLM (batched) with robust fallback + TOC handling
    # -------------------
    all_graph_docs = []
    try:
        transformer = _get_llm_transformer()
        if transformer is not None:
            from langchain_core.documents import Document

            # build chunk-level Documents expected by transformer
            chunk_docs = [Document(page_content=d["content"]) for d in docs_to_add]

            for i in range(0, len(chunk_docs), GRAPH_BATCH_SIZE):
                batch = chunk_docs[i: i + GRAPH_BATCH_SIZE]

                # Pre-process: if most chunks look like TOC, prefer groq JSON fallback
                if all(_looks_like_toc_or_list(d.page_content) for d in batch):
                    print(f"Batch {i}-{i+GRAPH_BATCH_SIZE} looks TOC-like — using Groq JSON fallback.")
                    try:
                        texts = [d.page_content for d in batch]
                        gd_fallback = _extract_graph_via_groq(texts)
                        if gd_fallback:
                            all_graph_docs.extend(gd_fallback)
                            continue
                    except Exception as e_fb:
                        print("Groq fallback for TOC-like batch failed:", e_fb)
                        # fall through to attempt transformer per-chunk

                try:
                    gd_batch = transformer.convert_to_graph_documents(batch)
                    if gd_batch:
                        # Normalize transformer output into simple dicts
                        normalized_batch = _normalize_graph_documents(gd_batch)
                        all_graph_docs.extend(normalized_batch)
                except Exception as batch_err:
                    print(f"LLM graph extraction failed for batch {i}-{i+GRAPH_BATCH_SIZE}: {batch_err}")

                    # Try to extract JSON from failed_generation if present
                    try:
                        parsed_fg = _extract_json_from_failed_generation(str(batch_err))
                        if parsed_fg:
                            args = parsed_fg.get("arguments") or parsed_fg.get("args") or parsed_fg
                            if isinstance(args, dict):
                                nodes = args.get("nodes") or args.get("Nodes") or []
                                rels = args.get("relationships") or args.get("Relationships") or args.get("edges") or []
                                gd = {"nodes": [], "edges": []}
                                for n in nodes:
                                    nid = n.get("id") or n.get("name") or n.get("Id")
                                    label = n.get("type") or n.get("label") or "Entity"
                                    props = n.get("properties") or n.get("props") or {}
                                    if nid:
                                        gd["nodes"].append({"id": str(nid), "label": label, "properties": props})
                                for r in rels:
                                    src = r.get("source_node_id") or r.get("source") or r.get("from")
                                    tgt = r.get("target_node_id") or r.get("target") or r.get("to")
                                    lab = r.get("type") or r.get("label") or "RELATED_TO"
                                    props = r.get("properties") or r.get("props") or {}
                                    if src and tgt:
                                        gd["edges"].append({"source": str(src), "target": str(tgt), "label": lab, "properties": props})
                                if gd["nodes"] or gd["edges"]:
                                    all_graph_docs.append(gd)
                                    print(f"Recovered graph fragment from failed_generation for batch {i}.")
                                    continue
                    except Exception as e_fg:
                        print("Parsing failed_generation content failed:", e_fg)

                    # Next fallback: Groq JSON extraction for the entire batch
                    try:
                        texts = [d.page_content for d in batch]
                        gd_fallback = _extract_graph_via_groq(texts)
                        if gd_fallback:
                            all_graph_docs.extend(gd_fallback)
                            continue
                    except Exception as e_f:
                        print("Groq fallback extraction failed for batch:", e_f)

                    # Last resort: try each chunk individually via transformer, then groq fallback per chunk
                    for j, single_doc in enumerate(batch):
                        try:
                            single_gd = transformer.convert_to_graph_documents([single_doc])
                            if single_gd:
                                normalized_single = _normalize_graph_documents(single_gd)
                                all_graph_docs.extend(normalized_single)
                                continue
                        except Exception as single_err:
                            print(f"Single-chunk graph extraction failed (chunk idx {i+j}): {single_err}")
                            # Try parsing failed_generation inside single_err
                            try:
                                parsed2 = _extract_json_from_failed_generation(str(single_err))
                                if parsed2:
                                    args2 = parsed2.get("arguments") or parsed2
                                    nodes2 = args2.get("nodes") or args2.get("Nodes") or []
                                    rels2 = args2.get("relationships") or args2.get("Relationships") or args2.get("edges") or []
                                    gd2 = {"nodes": [], "edges": []}
                                    for n in nodes2:
                                        nid = n.get("id") or n.get("name")
                                        label = n.get("type") or n.get("label") or "Entity"
                                        props = n.get("properties") or n.get("props") or {}
                                        if nid:
                                            gd2["nodes"].append({"id": str(nid), "label": label, "properties": props})
                                    for r in rels2:
                                        src = r.get("source_node_id") or r.get("source") or r.get("from")
                                        tgt = r.get("target_node_id") or r.get("target") or r.get("to")
                                        lab = r.get("type") or r.get("label") or "RELATED_TO"
                                        props = r.get("properties") or r.get("props") or {}
                                        if src and tgt:
                                            gd2["edges"].append({"source": str(src), "target": str(tgt), "label": lab, "properties": props})
                                    if gd2["nodes"] or gd2["edges"]:
                                        all_graph_docs.append(gd2)
                                        print(f"Recovered graph fragment from failed_generation for chunk {i+j}.")
                                        continue
                            except Exception as e2:
                                print("Parsing failed_generation for single chunk failed:", e2)
                            # Groq fallback for single chunk
                            try:
                                gd_single_fb = _extract_graph_via_groq([single_doc.page_content])
                                if gd_single_fb:
                                    all_graph_docs.extend(gd_single_fb)
                                    continue
                            except Exception as fb_err:
                                print("Single-chunk groq fallback failed:", fb_err)
                                continue

            # Insert collected graph fragments into Neo4j (best-effort)
            if all_graph_docs:
                try:
                    neo4j.create_graph_from_documents(all_graph_docs)
                except Exception as e:
                    print("Failed inserting graph documents into Neo4j:", e)
    except Exception as e:
        # do not fail ingestion if transformer init or processing fails; log and continue
        print("LLM graph extraction failed (transformer init or processing):", e)

    return {
        "doc_id": doc_id,
        "added_chunks": len(docs_to_add),
        "graph_fragments": len(all_graph_docs)
    }
