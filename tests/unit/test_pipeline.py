
"""Unit tests for the MediQuery pipeline."""

from unittest.mock import Mock

import pytest

from mediquery.pipeline import MediQueryPipeline
from mediquery.schemas import MediQueryResponse, RetrievalResult


def test_pipeline_retrieves_context_and_generates_answer() -> None:
    """The pipeline should retrieve context and return the generated answer."""
    retriever = Mock()
    generator = Mock()

    retrieval_result = RetrievalResult(
        query="Qu'est-ce que le diabète de type 2 ?",
        chunks=(),
        strategy="hybrid",
    )
    expected_response = MediQueryResponse(
        question="Qu'est-ce que le diabète de type 2 ?",
        answer="Réponse de test.",
    )

    retriever.retrieve.return_value = retrieval_result
    generator.generate.return_value = expected_response

    pipeline = MediQueryPipeline(retriever, generator)
    response = pipeline.run("  Qu'est-ce que le diabète de type 2 ?  ")

    retriever.retrieve.assert_called_once_with(
        "Qu'est-ce que le diabète de type 2 ?"
    )
    generator.generate.assert_called_once_with(
        question="Qu'est-ce que le diabète de type 2 ?",
        context=retrieval_result,
    )
    assert response == expected_response


def test_pipeline_rejects_empty_question() -> None:
    """The pipeline should reject an empty question before retrieval."""
    retriever = Mock()
    generator = Mock()

    pipeline = MediQueryPipeline(retriever, generator)

    with pytest.raises(ValueError, match="question ne peut pas être vide"):
        pipeline.run("   ")

    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()
