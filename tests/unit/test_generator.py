
"""Unit tests for the MediQuery generator."""

from unittest.mock import Mock

from mediquery.generation.generator import MediQueryGenerator
from mediquery.schemas import RetrievalResult, RetrievedChunk


def test_generate_returns_answer_and_sources() -> None:
    """The generator should return an answer and its source references."""
    client = Mock()
    client.generate.return_value = "Le diabète de type 2 nécessite un suivi."

    generator = MediQueryGenerator(client=client)
    context = RetrievalResult(
        query="Comment suivre le diabète ?",
        chunks=(
            RetrievedChunk(
                chunk_id="guide::page=12::start=0",
                content="Le suivi repose sur des consultations régulières.",
                metadata={"source": "guide.pdf", "page": 12},
                source="guide.pdf",
            ),
            RetrievedChunk(
                chunk_id="guide::page=12::start=500",
                content="Le suivi est adapté au patient.",
                metadata={"source": "guide.pdf", "page": 12},
                source="guide.pdf",
            ),
        ),
        strategy="hybrid",
    )

    response = generator.generate("Comment suivre le diabète ?", context)

    assert response.answer == "Le diabète de type 2 nécessite un suivi."
    assert response.question == "Comment suivre le diabète ?"
    assert response.abstained is False
    assert response.metadata["strategy"] == "hybrid"
    assert len(response.sources) == 1
    assert response.sources[0].document_name == "guide.pdf"
    assert response.sources[0].page == 12

    client.generate.assert_called_once()
    prompt_context = client.generate.call_args.args[1]
    assert "Le suivi repose sur des consultations régulières." in prompt_context
    assert "Page : 12" in prompt_context


def test_generate_abstains_when_no_chunks_are_retrieved() -> None:
    """The generator should abstain when retrieval returns no chunks."""
    client = Mock()
    generator = MediQueryGenerator(client=client)
    context = RetrievalResult(
        query="Question sans résultat",
        chunks=(),
        strategy="hybrid",
    )

    response = generator.generate("Question sans résultat", context)

    assert response.abstained is True
    assert response.sources == ()
    assert "pas suffisamment d'informations" in response.answer
    client.generate.assert_not_called()


def test_generate_rejects_empty_question() -> None:
    """The generator should reject an empty question."""
    generator = MediQueryGenerator(client=Mock())
    context = RetrievalResult(
        query="Question valide",
        chunks=(),
        strategy="hybrid",
    )

    try:
        generator.generate("   ", context)
    except ValueError as exc:
        assert "question ne peut pas être vide" in str(exc)
    else:
        raise AssertionError("Une question vide aurait dû être rejetée.")
