
"""Orchestration entry point for the MediQuery RAG pipeline."""

from typing import Protocol

from mediquery.schemas import MediQueryResponse, RetrievalResult


class Retriever(Protocol):
    """Interface expected from a retrieval component."""

    def retrieve(self, question: str) -> RetrievalResult:
        """Retrieve relevant document chunks for a question."""
        ...


class Generator(Protocol):
    """Interface expected from a generation component."""

    def generate(
        self,
        question: str,
        context: RetrievalResult,
    ) -> MediQueryResponse:
        """Generate an answer grounded in the retrieved context."""
        ...


class MediQueryPipeline:
    """Coordinate retrieval and answer generation."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def run(self, question: str) -> MediQueryResponse:
        """Run the retrieval and generation stages."""
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("La question ne peut pas être vide.")

        retrieval_result = self.retriever.retrieve(normalized_question)

        return self.generator.generate(
            question=normalized_question,
            context=retrieval_result,
        )
