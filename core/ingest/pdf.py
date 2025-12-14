# core/ingest/pdf.py
"""
Fast PDF text extraction with multiple fallback methods.
Uses PyMuPDF (fitz) for speed, falls back to pdfminer or pypdf.
"""
from typing import Union
import io


def extract_pdf_text(source: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract text from PDF with multiple fallback methods.
    
    Args:
        source: File path (str), bytes, or BytesIO object
        
    Returns:
        Extracted text as string
    """
    # Try PyMuPDF first (fastest)
    try:
        return _extract_with_pymupdf(source)
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try pdfminer (most robust)
    try:
        return _extract_with_pdfminer(source)
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try pypdf (most compatible)
    try:
        return _extract_with_pypdf(source)
    except ImportError:
        pass
    except Exception:
        pass
    
    raise RuntimeError("No PDF extraction library available. Install pymupdf, pdfminer, or pypdf.")


def _extract_with_pymupdf(source: Union[str, bytes, io.BytesIO]) -> str:
    """Extract using PyMuPDF (fitz) - fastest method."""
    import fitz  # PyMuPDF
    
    if isinstance(source, str):
        doc = fitz.open(source)
    elif isinstance(source, bytes):
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        doc = fitz.open(stream=source.read(), filetype="pdf")
    
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def _extract_with_pdfminer(source: Union[str, bytes, io.BytesIO]) -> str:
    """Extract using pdfminer - most robust method."""
    from pdfminer.high_level import extract_text
    
    if isinstance(source, str):
        return extract_text(source)
    elif isinstance(source, bytes):
        return extract_text(io.BytesIO(source))
    else:
        return extract_text(source)


def _extract_with_pypdf(source: Union[str, bytes, io.BytesIO]) -> str:
    """Extract using pypdf - most compatible method."""
    from pypdf import PdfReader
    
    if isinstance(source, str):
        reader = PdfReader(source)
    elif isinstance(source, bytes):
        reader = PdfReader(io.BytesIO(source))
    else:
        reader = PdfReader(source)
    
    text_parts = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
        except Exception:
            continue
    return "\n".join(text_parts)


# Convenience alias for backward compatibility
pdf_to_text = extract_pdf_text
