# app/confluence_generator.py
import os
import logging
from typing import Optional, Dict, Any

# Prefer to reuse the standalone confluence script if present (it supports Groq vision flows)
try:
    # This is the script you pasted earlier, kept under confluence/
    from confluence.confluence_generator import (
        extract_pdf_text,
        encode_image_to_base64,
        generate_help_text_for_screen,
        create_confluence_page as _create_confluence_page_raw,
    )
    _HAS_STANDALONE = True
except Exception:
    _HAS_STANDALONE = False

# Fallback Groq HTTP client wrapper (text-only) if the vision path is not usable.
# This is the lightweight Groq client you added under src/groq_llm.py
try:
    from src.groq_llm import GroqClient
    _HAS_GROQ_HTTP = True
except Exception:
    GroqClient = None
    _HAS_GROQ_HTTP = False

# Config (Confluence envs - with fallback to Jira credentials)
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
# Allow fallback to JIRA_EMAIL if CONFLUENCE_USERNAME not set
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME") or os.getenv("JIRA_EMAIL")
# Allow fallback to JIRA_API_TOKEN if CONFLUENCE_API_TOKEN not set
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN") or os.getenv("JIRA_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

logger = logging.getLogger(__name__)


def _validate_confluence_config():
    missing = []
    if not CONFLUENCE_BASE_URL:
        missing.append("CONFLUENCE_BASE_URL")
    if not CONFLUENCE_USERNAME:
        missing.append("CONFLUENCE_USERNAME (or JIRA_EMAIL)")
    if not CONFLUENCE_API_TOKEN:
        missing.append("CONFLUENCE_API_TOKEN (or JIRA_API_TOKEN)")
    if not CONFLUENCE_SPACE_KEY:
        missing.append("CONFLUENCE_SPACE_KEY")
    
    if missing:
        raise RuntimeError(
            f"Confluence configuration incomplete. Missing: {', '.join(missing)}. "
            f"Please set these in your .env file."
        )


def _post_confluence_page(title: str, html_content: str) -> Dict[str, Any]:
    """
    Push page to Confluence via REST API and return parsed response.
    Raises requests.HTTPError on failure.
    """
    import requests

    _validate_confluence_config()
    url = f"{CONFLUENCE_BASE_URL.rstrip('/')}/rest/api/content"
    unique_title = f"{title} – {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    payload = {
        "type": "page",
        "title": unique_title,
        "space": {"key": CONFLUENCE_SPACE_KEY},
        "body": {
            "storage": {
                "value": html_content,
                "representation": "storage"
            }
        }
    }

    logger.debug("Creating Confluence page. Title: %s", unique_title)
    resp = requests.post(
        url,
        json=payload,
        auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def generate_help_html(frd_text: str, title: str, base64_image: Optional[str] = None) -> str:
    """
    Generate the Confluence-ready HTML help text for a screen/feature.

    Strategy:
      - If the standalone confluence generator (vision-capable) exists and a base64_image is provided,
        call its `generate_help_text_for_screen` to leverage vision + text synthesis.
      - Otherwise use the text-only Groq client (src.groq_llm.GroqClient) to synthesize an HTML page
        using a safe, deterministic prompt (temperature=0.0).

    Returns:
      - HTML string (storage-format HTML suitable for Confluence).
    """
    # Use the standalone vision flow if available and an image is provided
    if _HAS_STANDALONE and base64_image:
        try:
            # generate_help_text_for_screen expects (client, frd_text, base64_image, screen_name)
            # In the original script it used a Groq client instance; passing None may still work if that function
            # instantiates or uses the groq client internally. If it requires a client, we assume it creates one.
            html = generate_help_text_for_screen(client=None, frd_text=frd_text, base64_image=base64_image, screen_name=title)
            if html:
                return html
        except Exception as e:
            logger.exception("Vision-based confluence generation failed, falling back to text-only: %s", e)

    # Fallback: text-only generation using Groq HTTP client wrapper
    if not _HAS_GROQ_HTTP:
        raise RuntimeError("No available LLM client for Confluence generation (neither vision generator nor Groq HTTP client found).")

    try:
        groq = GroqClient()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GroqClient: {e}") from e

    # Build a deterministic prompt that asks for HTML output only
    prompt = (
        "You are an assistant that converts functional requirement text into a concise, "
        "developer-friendly Confluence help page in HTML (Confluence storage format is OK). "
        "Output ONLY the HTML content (no surrounding explanation). "
        "Include: Overview, Field Descriptions (as a small table), Buttons/Actions, Validation & Error Messages, Tips. "
        "Use the FRD text below as the sole source of truth.\n\n"
        "FRD:\n"
        "-----BEGIN FRD-----\n"
        f"{frd_text[:16000]}\n"  # safety: truncate to first N chars if huge
        "-----END FRD-----\n\n"
        "Return the HTML only."
    )

    try:
        html_out = groq.generate(prompt, max_tokens=2000, temperature=0.0)
        # The Groq wrapper returns a string; ensure it's string and strip it
        html_str = str(html_out).strip()
        # Basic sanity check: must contain at least one HTML tag
        if "<" not in html_str or ">" not in html_str:
            logger.warning("Groq returned non-HTML, wrapping in <div> for safety.")
            html_str = "<div>\n" + html_str + "\n</div>"
        return html_str
    except Exception as e:
        logger.exception("Text-only Confluence generation failed: %s", e)
        raise RuntimeError(f"Confluence generation failed: {e}") from e


def create_confluence_page_from_text(frd_text: str, title: str, push: bool = False) -> Dict[str, Any]:
    """
    Generate Confluence HTML and optionally push it to Confluence.

    Returns a dict:
      - if push is False: {"html": <html string>}
      - if push is True: {"html": <html string>, "api_response": <confluence response dict>, "page_url": <url>}
    """
    html = generate_help_html(frd_text, title)
    result: Dict[str, Any] = {"html": html}

    if not push:
        return result

    # Push to Confluence
    try:
        api_resp = _post_confluence_page(title, html)
        # derive a public URL if Confluence returns _links
        links = api_resp.get("_links", {})
        webui = links.get("webui")
        base = links.get("base", CONFLUENCE_BASE_URL)
        page_url = (base.rstrip("/") + webui) if webui else None

        result.update({"api_response": api_resp, "page_url": page_url})
        return result
    except Exception as e:
        logger.exception("Failed to create Confluence page: %s", e)
        # Surface a friendly error in the return value rather than raising, to let callers decide
        return {"html": html, "error": str(e)}


# Convenience helper used by orchestrator wrapper (keeps old name)
def create_confluence_page_from_grounding(grounding_chunks: list, title: str, push: bool = False) -> Dict[str, Any]:
    """
    Build FRD text from grounding chunks list (each chunk is dict with 'text') and call create_confluence_page_from_text.
    If grounding is empty, returns an error message instead of generating empty content.
    """
    if not grounding_chunks:
        return {
            "html": "<p>No documents found to generate help content from. Please ingest documents first using the sidebar, then try again.</p>",
            "error": "No grounding documents available. Please ingest some documents (PDF/DOCX) using the sidebar first."
        }
    
    frd_text = "\n\n".join([c.get("text", "") for c in grounding_chunks if c.get("text")])
    
    if not frd_text.strip():
        return {
            "html": "<p>The retrieved documents contained no text content. Please ensure your uploaded documents have readable text.</p>",
            "error": "Retrieved documents had no readable text content."
        }
    
    return create_confluence_page_from_text(frd_text=frd_text, title=title, push=push)

