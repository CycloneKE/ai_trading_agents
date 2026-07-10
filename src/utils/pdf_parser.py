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
        with pdfplumber.open(file_path) as pdf:
            metadata["pages_count"] = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                    
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        all_tables.append(table)
                        
        return {
            "text": "\n".join(text_content),
            "tables": all_tables,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Failed to perform full PDF extraction for {file_path}: {e}")
        raise
