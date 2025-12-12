# confluence/confluence_generator.py
"""
Standalone Confluence page generator with proper implementations.
Provides functions for extracting text from PDFs, encoding images,
generating help text via LLM, and creating Confluence pages.
"""
import os
import base64
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Environment variables for Confluence
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    Returns the extracted text as a string.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed, trying pdfplumber")
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except ImportError:
            logger.error("No PDF library available (install PyMuPDF or pdfplumber)")
            return ""
    except Exception as e:
        logger.exception("Failed to extract PDF text: %s", e)
        return ""


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to a base64 string.
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.exception("Failed to encode image: %s", e)
        return ""


def generate_help_text_for_screen(
    client: Any,
    frd_text: str,
    base64_image: str = "",
    screen_name: str = "Screen"
) -> str:
    """
    Generate Confluence-ready HTML help text for a screen/feature.
    Uses the Groq LLM client to synthesize documentation from FRD text.
    
    Args:
        client: Optional LLM client (if None, will create GroqClient internally)
        frd_text: The functional requirement document text
        base64_image: Optional base64-encoded screenshot (not used in text-only mode)
        screen_name: Title/name of the screen for context
    
    Returns:
        HTML string suitable for Confluence storage format
    """
    # Import GroqClient lazily
    try:
        from src.groq_llm import GroqClient
    except ImportError:
        logger.error("GroqClient not available")
        return f"<div><h1>{screen_name}</h1><p>Error: LLM client not available.</p></div>"
    
    # Create client if not provided
    if client is None:
        try:
            client = GroqClient()
        except Exception as e:
            logger.exception("Failed to create GroqClient: %s", e)
            return f"<div><h1>{screen_name}</h1><p>Error: {e}</p></div>"
    
    # Build the prompt
    prompt = f"""You are a technical writer creating a Confluence help page for "{screen_name}".

Convert the following functional requirement document into a clear, user-friendly Confluence help page in HTML format.

Include these sections as appropriate:
1. **Overview** - Brief description of the feature/screen
2. **Field Descriptions** - Table with field names and descriptions
3. **Actions/Buttons** - What each button does
4. **Validation Rules** - Any validation or error messages users might see
5. **Tips & Best Practices** - Helpful hints for users

Output ONLY valid HTML that can be pasted into Confluence (storage format). Do not include markdown or explanations.

FRD Content:
---
{frd_text[:12000]}
---

Return the HTML only:"""

    try:
        # Try to call generate method
        if hasattr(client, 'generate'):
            html = client.generate(prompt, max_tokens=2000, temperature=0.0)
        else:
            html = client(prompt)
        
        html_str = str(html).strip()
        
        # Basic validation - ensure it looks like HTML
        if "<" not in html_str or ">" not in html_str:
            html_str = f"<div><h1>{screen_name}</h1>\n{html_str}\n</div>"
        
        return html_str
    except Exception as e:
        logger.exception("LLM generation failed: %s", e)
        return f"<div><h1>{screen_name}</h1><p>Error generating help text: {e}</p></div>"


def create_confluence_page(title: str, html_content: str) -> Dict[str, Any]:
    """
    Create a new page in Confluence with the given title and HTML content.
    
    Returns:
        Dict with 'id', 'title', 'url' on success, or 'error' on failure.
    """
    if not all([CONFLUENCE_BASE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, CONFLUENCE_SPACE_KEY]):
        return {
            "error": "Confluence configuration incomplete. Set CONFLUENCE_BASE_URL, "
                     "CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, CONFLUENCE_SPACE_KEY in environment."
        }
    
    # Add timestamp to title to avoid duplicates
    import datetime
    timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    unique_title = f"{title} – {timestamp}"
    
    url = f"{CONFLUENCE_BASE_URL.rstrip('/')}/rest/api/content"
    
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
    
    try:
        resp = requests.post(
            url,
            json=payload,
            auth=(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN),
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Build page URL
        links = data.get("_links", {})
        webui = links.get("webui", "")
        base = links.get("base", CONFLUENCE_BASE_URL)
        page_url = (base.rstrip("/") + webui) if webui else None
        
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "url": page_url,
            "api_response": data
        }
    except requests.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.text
        except:
            pass
        logger.exception("Confluence API error: %s", e)
        return {"error": f"Confluence API error: {e} - {error_detail}"}
    except Exception as e:
        logger.exception("Failed to create Confluence page: %s", e)
        return {"error": str(e)}
