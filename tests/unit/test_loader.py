
"""Unit tests for the LangChain PDF loader."""

from pathlib import Path

import pytest
from pypdf import PdfWriter
from langchain_core.documents import Document

from mediquery.ingestion.loader import load_pdf



def create_test_pdf(file_path: Path) -> None:
    """Create a PDF containing extractable text for testing."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(file_path), pagesize=letter)
    pdf.drawString(72, 720, "MediQuery test document.")
    pdf.drawString(72, 700, "This text must be extractable.")
    pdf.save()



def test_load_pdf_returns_langchain_documents(tmp_path: Path) -> None:
    """The loader should return LangChain documents with page metadata."""
    pdf_path = tmp_path / "test.pdf"
    create_test_pdf(pdf_path)

    documents = load_pdf(pdf_path)

    assert documents
    assert all(isinstance(document, Document) for document in documents)
    assert all("source" in document.metadata for document in documents)
    assert all("page" in document.metadata for document in documents)
    assert min(document.metadata["page"] for document in documents) >= 1


def test_load_pdf_rejects_missing_file(tmp_path: Path) -> None:
    """A missing PDF should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="PDF introuvable"):
        load_pdf(tmp_path / "missing.pdf")
