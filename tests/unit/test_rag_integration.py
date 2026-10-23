
"""Integration tests for the MediQuery RAG pipeline."""

from unittest.mock import Mock

from mediquery.generation.generator import MediQueryGenerator
from mediquery.pipeline import MediQueryPipeline
from mediquery.retrieval.pipeline_retriever import HybridPipelineRetriever


def test_hybrid_retrieval_to_generated_answer() -> None:
    """The complete pipeline should pass retrieved context to the generator."""
    document = Mock()
    document.page_content = (
        "Le suivi du diabète de type 2 repose sur des consultations régulières."
    )
    document.metadata = {
        "source": "guide.pdf",
        "page": 12,
        "start_index": 0,
    }

    hybrid_retriever = Mock()
    hybrid_retriever.retrieve.return_value = [document]

    retriever = HybridPipelineRetriever(retriever=hybrid_retriever)

    llm_client = Mock()
    llm_client.generate.return_value = (
        "Le suivi repose sur des consultations régulières. (guide.pdf, p. 12)"
    )
    generator = MediQueryGenerator(client=llm_client)

    pipeline = MediQueryPipeline(retriever=retriever, generator=generator)
    response = pipeline.run("Comment suivre le diabète de type 2 ?")

    assert response.abstained is False
    assert "consultations régulières" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].document_name == "guide.pdf"
    assert response.sources[0].page == 12

    hybrid_retriever.retrieve.assert_called_once_with(
        "Comment suivre le diabète de type 2 ?"
    )
    llm_client.generate.assert_called_once()
