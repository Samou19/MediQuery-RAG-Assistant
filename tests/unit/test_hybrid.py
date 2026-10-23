
"""Unit tests for reciprocal rank fusion."""

import pytest
from langchain_core.documents import Document

from mediquery.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_keeps_documents_present_in_both_lists() -> None:
    doc_a = Document(
        page_content="Diagnostic du diabète",
        metadata={"source": "guide.pdf", "page": 12, "start_index": 0},
    )
    doc_b = Document(
        page_content="Traitement du diabète",
        metadata={"source": "guide.pdf", "page": 24, "start_index": 0},
    )
    doc_c = Document(
        page_content="Suivi du patient",
        metadata={"source": "guide.pdf", "page": 42, "start_index": 0},
    )

    results = reciprocal_rank_fusion(
        [[doc_a, doc_b], [doc_b, doc_c]],
    )

    assert len(results) == 3
    assert results[0].page_content == "Traitement du diabète"


def test_rrf_deduplicates_same_chunk() -> None:
    doc_a = Document(
        page_content="Diagnostic du diabète",
        metadata={"source": "guide.pdf", "page": 12, "start_index": 0},
    )
    duplicate = Document(
        page_content="Diagnostic du diabète",
        metadata={"source": "guide.pdf", "page": 12, "start_index": 0},
    )

    results = reciprocal_rank_fusion([[doc_a], [duplicate]])

    assert len(results) == 1


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        reciprocal_rank_fusion([], k=0)
