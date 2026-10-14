"""
test_loader.py — Unit tests for src/rag/loader.py
===================================================
Run with: pytest tests/test_loader.py -v

We test the loader without any real PDFs by using Python's unittest.mock
to patch out PyMuPDF, so tests stay fast and don't need fixture files.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.loader import load_pdf


def test_load_pdf_raises_if_file_missing():
    """load_pdf should raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_pdf("/tmp/definitely_does_not_exist_policy.pdf")


def test_load_pdf_raises_if_no_text():
    """load_pdf should raise ValueError when PDF has no extractable text."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = "   "  # blank / whitespace only

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)

    with patch("src.rag.loader.fitz.open", return_value=mock_doc), \
         patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(ValueError, match="No extractable text"):
            load_pdf("data/policies/fake.pdf")


def test_load_pdf_returns_chunks():
    """load_pdf should return non-empty list of Document chunks for valid PDF.

    We force CHUNK_STRATEGY = "fixed_token" here so the test stays fast and
    offline — the semantic strategy makes live OpenAI embedding calls that
    are unsuitable for unit tests.
    """
    mock_page = MagicMock()
    mock_page.get_text.return_value = "This policy covers fire damage. " * 50

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)

    with patch("src.rag.loader.fitz.open", return_value=mock_doc), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("src.rag.loader.CHUNK_STRATEGY", "fixed_token"):
        chunks = load_pdf("data/policies/fake.pdf")

    assert len(chunks) >= 1
    assert all(hasattr(c, "page_content") for c in chunks)
    assert all(c.metadata["source"] == "fake.pdf" for c in chunks)
