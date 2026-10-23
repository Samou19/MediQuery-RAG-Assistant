
"""Unit tests for the document chunker."""

import pytest
from langchain_core.documents import Document

from mediquery.ingestion.chunker import split_documents


def test_split_documents_preserves_metadata() -> None:
    """Chunks should preserve source metadata and have a start index."""
    document = Document(
        page_content=("Le diabète de type 2 nécessite un suivi régulier. " * 20),
        metadata={"source": "guide.pdf", "page": 3},
    )

    chunks = split_documents(
        [document],
        chunk_size=150,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "guide.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_split_documents_accepts_empty_input() -> None:
    """An empty document list should produce no chunks."""
    assert split_documents([]) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_split_documents_rejects_invalid_parameters(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Invalid chunking parameters should raise ValueError."""
    with pytest.raises(ValueError):
        split_documents(
            [Document(page_content="Texte de test")],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
