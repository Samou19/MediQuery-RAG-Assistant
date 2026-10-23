
"""Unit tests for the BM25 retriever."""

import pytest
from langchain_core.documents import Document

from mediquery.retrieval.bm25 import BM25Retriever, tokenize


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            page_content="Le dépistage du diabète repose sur la glycémie.",
            metadata={"page": 12},
        ),
        Document(
            page_content="Le traitement comprend une prise en charge adaptée.",
            metadata={"page": 24},
        ),
        Document(
            page_content="Le suivi recherche les complications du diabète.",
            metadata={"page": 42},
        ),
    ]


def test_tokenize_lowercases_text() -> None:
    assert tokenize("Diabète TYPE 2") == ["diabète", "type", "2"]


def test_retrieve_returns_relevant_document(documents: list[Document]) -> None:
    retriever = BM25Retriever(documents, k=2)

    results = retriever.retrieve("dépistage glycémie")

    assert results
    assert results[0][0].metadata["page"] == 12
    assert results[0][1] > 0


def test_retrieve_preserves_document_metadata(
    documents: list[Document],
) -> None:
    retriever = BM25Retriever(documents)

    results = retriever.retrieve("glycémie")

    assert results[0][0].metadata["page"] == 12


def test_empty_documents_are_rejected() -> None:
    with pytest.raises(ValueError, match="vide"):
        BM25Retriever([])


def test_empty_query_is_rejected(documents: list[Document]) -> None:
    retriever = BM25Retriever(documents)

    with pytest.raises(ValueError, match="requête"):
        retriever.retrieve("   ")
