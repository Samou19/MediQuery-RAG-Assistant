
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from mediquery.retrieval.reranker import (
    CrossEncoderReranker,
    select_diverse_pages,
)


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            page_content="Le suivi du diabète de type 2 comprend le contrôle de "
            "l'HbA1c et la recherche des complications.",
            metadata={"source": "guide.pdf", "page": 37},
        ),
        Document(
            page_content="La tour Eiffel se situe à Paris.",
            metadata={"source": "guide.pdf", "page": 90},
        ),
        Document(
            page_content="Le diagnostic du diabète repose notamment sur la mesure "
            "de la glycémie.",
            metadata={"source": "guide.pdf", "page": 12},
        ),
    ]


def test_rerank_orders_documents_by_score(documents: list[Document]) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.top_k = 2
    reranker.model = Mock()
    reranker.model.predict.return_value = [0.8, 0.1, 0.6]

    result = reranker.rerank(
        "Comment suivre un patient diabétique ?",
        documents,
    )

    assert len(result) == 2
    assert result[0] is documents[0]
    assert result[1] is documents[2]
    assert result[0].metadata["page"] == 37


def test_rerank_empty_documents() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.top_k = 5
    reranker.model = Mock()

    assert reranker.rerank("Question médicale", []) == []


def test_rerank_rejects_empty_query() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.top_k = 5
    reranker.model = Mock()

    with pytest.raises(ValueError, match="requête"):
        reranker.rerank("   ", [])


def test_select_diverse_pages_keeps_only_one_chunk_per_page() -> None:
    documents = [
        Document(
            page_content="Premier passage de la page 42",
            metadata={"source": "guide.pdf", "page": 42},
        ),
        Document(
            page_content="Deuxième passage de la page 42",
            metadata={"source": "guide.pdf", "page": 42},
        ),
        Document(
            page_content="Passage de la page 53",
            metadata={"source": "guide.pdf", "page": 53},
        ),
    ]

    result = select_diverse_pages(documents, top_k=2)

    assert len(result) == 2
    assert result[0] is documents[0]
    assert result[1] is documents[2]


def test_select_diverse_pages_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        select_diverse_pages([], top_k=0)
