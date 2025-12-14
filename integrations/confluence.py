# integrations/confluence.py
"""
Confluence integration - Generate and push help pages.
Merged from app/confluence_generator.py
"""
import os
import logging
import requests
from typing import Dict, Any, Optional, List

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from core.config import settings
from core.llm.groq_http import GroqHTTPClient

logger = logging.getLogger(__name__)


def _get_confluence_config() -> Dict[str, Optional[str]]:
    """Get Confluence config at runtime."""
    return {
        "base_url": os.getenv("CONFLUENCE_BASE_URL"),
        "username": os.getenv("CONFLUENCE_USERNAME") or os.getenv("JIRA_EMAIL"),
        "api_token": os.getenv("CONFLUENCE_API_TOKEN") or os.getenv("JIRA_API_TOKEN"),
        "space_key": os.getenv("CONFLUENCE_SPACE_KEY"),
    }


def _validate_confluence_config() -> Dict[str, str]:
    """Validate Confluence configuration."""
    config = _get_confluence_config()
    missing = []
    if not config["base_url"]:
        missing.append("CONFLUENCE_BASE_URL")
    if not config["username"]:
        missing.append("CONFLUENCE_USERNAME (or JIRA_EMAIL)")
    if not config["api_token"]:
        missing.append("CONFLUENCE_API_TOKEN (or JIRA_API_TOKEN)")
    if not config["space_key"]:
        missing.append("CONFLUENCE_SPACE_KEY")
    
    if missing:
        raise RuntimeError(f"Confluence config incomplete. Missing: {', '.join(missing)}")
    return config


def generate_help_html(frd_text: str, title: str) -> str:
    """Generate HTML help content from FRD text using LLM."""
    try:
        client = GroqHTTPClient()
    except Exception as e:
        logger.warning(f"LLM not available: {e}")
        return f"<h1>{title}</h1><p>LLM not available for content generation.</p>"
    
    if not frd_text or not frd_text.strip():
        return f"<h1>{title}</h1><p>No content provided for generation.</p>"
    
    prompt = f"""Generate a professional HTML help page from the following functional requirements.

REQUIREMENTS:
{frd_text[:6000]}

Create a help page with these sections (use HTML tags):
- <h1> Title
- <h2>Overview</h2> - Brief description
- <h2>Field Descriptions</h2> - Table of fields
- <h2>Actions</h2> - List of buttons/actions
- <h2>Tips</h2> - User tips

Output ONLY clean HTML, no markdown or code fences."""

    try:
        html = client.generate(prompt, max_tokens=2500, temperature=0.0)
        # Clean up any code fences
        if html.startswith("```"):
            html = html.split("```", 2)[1]
            if html.startswith("html"):
                html = html[4:]
        return html.strip()
    except Exception as e:
        logger.error(f"HTML generation failed: {e}")
        return f"<h1>{title}</h1><p>Error generating content: {e}</p>"


def _post_confluence_page(title: str, html_content: str) -> Dict[str, Any]:
    """Push page to Confluence via REST API."""
    import datetime
    
    config = _validate_confluence_config()
    url = f"{config['base_url'].rstrip('/')}/rest/api/content"
    unique_title = f"{title} – {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    payload = {
        "type": "page",
        "title": unique_title,
        "space": {"key": config['space_key']},
        "body": {
            "storage": {
                "value": html_content,
                "representation": "storage"
            }
        }
    }

    logger.info(f"Creating Confluence page: {unique_title}")
    resp = requests.post(
        url,
        json=payload,
        auth=(config['username'], config['api_token']),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def create_confluence_page_from_text(frd_text: str, title: str, push: bool = False) -> Dict[str, Any]:
    """Generate Confluence HTML and optionally push."""
    html = generate_help_html(frd_text, title)
    result = {"html": html}

    if not push:
        return result

    try:
        api_resp = _post_confluence_page(title, html)
        links = api_resp.get("_links", {})
        webui = links.get("webui")
        config = _get_confluence_config()
        base = links.get("base", config["base_url"])
        page_url = (base.rstrip("/") + webui) if webui else None

        result.update({"api_response": api_resp, "page_url": page_url})
        return result
    except Exception as e:
        logger.exception(f"Confluence push failed: {e}")
        return {"html": html, "error": str(e)}


def create_confluence_page_from_grounding(
    grounding_chunks: List[Dict[str, Any]], 
    title: str, 
    push: bool = False
) -> Dict[str, Any]:
    """Build FRD text from grounding chunks and create page."""
    if not grounding_chunks:
        return {
            "html": "<p>No documents found. Please ingest documents first.</p>",
            "error": "No grounding documents available."
        }
    
    frd_text = "\n\n".join([c.get("text", "") for c in grounding_chunks if c.get("text")])
    
    if not frd_text.strip():
        return {
            "html": "<p>Retrieved documents had no text content.</p>",
            "error": "No readable text in documents."
        }
    
    return create_confluence_page_from_text(frd_text, title, push=push)
