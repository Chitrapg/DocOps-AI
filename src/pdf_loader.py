#src/pdf_loader.py
from pypdf import PdfReader

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts all text from a PDF file (Streamlit uploaded file -> bytes).
    Returns a clean text string.
    """
    reader = PdfReader(file_bytes)
    full_text = []
    
    for page in reader.pages:
        try:
            txt = page.extract_text()
            if txt:
                full_text.append(txt)
        except:
            continue

    return "\n".join(full_text)
