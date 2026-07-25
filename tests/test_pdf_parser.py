"""
Unit tests for pdf_parser.py.
"""
import pytest
import os
from src.utils import pdf_parser


def test_pdfplumber_available():
    """Verify pdfplumber availability flag is set correctly."""
    assert hasattr(pdf_parser, "PDFPLUMBER_AVAILABLE")


def test_extract_text_missing_file():
    """Verify FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        pdf_parser.extract_text("non_existent_file.pdf")


def test_extract_tables_missing_file():
    """Verify FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        pdf_parser.extract_tables("non_existent_file.pdf")


def test_extract_all_missing_file():
    """Verify FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        pdf_parser.extract_all("non_existent_file.pdf")
