
"""Unit tests for the LangChain PDF loader."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from mediquery.ingestion.loader import load_pdf


def test_load_pdf_returns_langchain_documents() -> None:
    """The source PDF should produce LangChain documents."""
    from mediquery.config import SOURCE_PDF

    documents = load_pdf(SOURCE_PDF)

    assert documents
    assert all(isinstance(document, Document) for document in documents)
    assert all(document.page_content.strip() for document in documents)
    assert all("source" in document.metadata for document in documents)
    assert all("page" in document.metadata for document in documents)
    assert min(document.metadata["page"] for document in documents) >= 1


def test_load_pdf_rejects_missing_file(tmp_path: Path) -> None:
    """A missing PDF should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="PDF introuvable"):
        load_pdf(tmp_path / "missing.pdf")
