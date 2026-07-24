"""
PDF Parser utility using pdfplumber to extract text and structured tables.
"""
import logging
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber library not found. PDF extraction will be disabled.")


def extract_text(file_path: str) -> str:
    """
    Extract raw text from a PDF file.
    """
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber is not installed.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
        
    text_content = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path}: {e}")
        raise


def extract_tables(file_path: str) -> List[List[List[Optional[str]]]]:
    """
    Extract tables from a PDF file. Returns a list of tables, where each table
    is a list of rows, and each row is a list of cell strings.
    """
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber is not installed.")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
        
    all_tables = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        all_tables.append(table)
        return all_tables
    except Exception as e:
        logger.error(f"Failed to extract tables from PDF {file_path}: {e}")
        raise


def extract_all(file_path: str) -> Dict[str, Any]:
    """
    Extract text, tables, and metadata from a PDF file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
        
    metadata = {
        "filename": os.path.basename(file_path),
        "file_size_bytes": os.path.getsize(file_path),
        "pages_count": 0
    }
    
    if not PDFPLUMBER_AVAILABLE:
        return {
            "text": "",
            "tables": [],
            "metadata": metadata,
            "error": "pdfplumber not available"
        }
        
    try:
        text_content = []
        all_tables = []
        all_links = []
        pages_detail = []
        
        with pdfplumber.open(file_path) as pdf:
            metadata["pages_count"] = len(pdf.pages)
            for idx, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                tables = page.extract_tables() or []
                
                # Extract embedded hyperlinks
                page_links = []
                if hasattr(page, 'hyperlinks') and page.hyperlinks:
                    for link in page.hyperlinks:
                        uri = link.get('uri')
                        if uri and uri not in page_links:
                            page_links.append(uri)
                            if uri not in all_links:
                                all_links.append(uri)

                table_md_list = []
                for table in tables:
                    if table and len(table) > 0:
                        all_tables.append(table)
                        # Convert table grid to Markdown table format
                        md_rows = []
                        for row in table:
                            cleaned_row = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row]
                            md_rows.append("| " + " | ".join(cleaned_row) + " |")
                        if len(md_rows) > 0:
                            table_md_list.append("\n".join(md_rows))
                
                table_str = "\n\n".join(table_md_list)
                links_str = "\n".join([f"Embedded Link: {l}" for l in page_links]) if page_links else ""
                combined_page = f"--- PAGE {idx+1} ---\n{page_text}\n\n{table_str}\n\n{links_str}".strip()
                text_content.append(combined_page)
                pages_detail.append({
                    "page_number": idx + 1,
                    "text": page_text,
                    "tables_markdown": table_str,
                    "links": page_links,
                    "combined": combined_page
                })
                        
        return {
            "text": "\n\n".join(text_content),
            "tables": all_tables,
            "links": all_links,
            "pages": pages_detail,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Failed to perform full PDF extraction for {file_path}: {e}")
        raise
