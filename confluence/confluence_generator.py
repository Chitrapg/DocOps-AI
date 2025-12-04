# confluence/confluence_generator.py
import logging
from typing import Dict, Any, Optional
from confluence_generator import extract_pdf_text, encode_image_to_base64, generate_help_text_for_screen, create_confluence_page

logger = logging.getLogger(__name__)

def create_confluence_page_from_text(frd_text: str, title: str, push: bool = False) -> Dict[str, Any]:
    """
    If push==False: returns the generated HTML string for preview.
    If push==True: creates a Confluence page and returns API response dict.
    """
    # For now we do a single page per title. Generate HTML via Groq (vision not required if no images).
    # Use a minimal helper: call generate_help_text_for_screen with no image.
    html = generate_help_text_for_screen(client=None, frd_text=frd_text, base64_image="", screen_name=title)
    if not push:
        return {"html": html}
    # create page in Confluence
    res = create_confluence_page(title=title, html_content=html)
    return {"html": html, "api_response": res}
