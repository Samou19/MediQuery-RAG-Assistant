
"""Unit tests for the ChromaDB indexer."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from mediquery.ingestion.indexer import get_chunk_id, index_documents


def test_get_chunk_id_is_deterministic() -> None:
    """Identical documents should produce identical IDs."""
    document = Document(
        page_content="Texte médical de test.",
        metadata={"source": "guide.pdf", "page": 1},
    )

    assert get_chunk_id(document) == get_chunk_id(document)


def test_get_chunk_id_changes_when_content_changes() -> None:
    """Different content should produce different IDs."""
    first = Document(
        page_content="Texte A",
        metadata={"source": "guide.pdf", "page": 1},
    )
    second = Document(
        page_content="Texte B",
        metadata={"source": "guide.pdf", "page": 1},
    )

    assert get_chunk_id(first) != get_chunk_id(second)


def test_index_documents_rejects_empty_list() -> None:
    """Empty input should be rejected."""
    with pytest.raises(ValueError, match="liste de documents"):
        index_documents([], embeddings=MagicMock())


@patch("mediquery.ingestion.indexer.Chroma")
def test_index_documents_uses_stable_ids(mock_chroma: MagicMock) -> None:
    """Documents should be indexed with deterministic IDs."""
    mock_store = MagicMock()
    mock_chroma.return_value = mock_store

    document = Document(
        page_content="Texte médical de test.",
        metadata={"source": "guide.pdf", "page": 1},
    )

    index_documents([document], embeddings=MagicMock())

    kwargs = mock_store.add_documents.call_args.kwargs
    assert kwargs["documents"] == [document]
    assert kwargs["ids"] == [get_chunk_id(document)]
