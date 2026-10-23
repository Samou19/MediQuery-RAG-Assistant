
"""Lexical retrieval using BM25."""

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from mediquery.config import BM25_SEARCH_K


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase whitespace-separated terms."""
    return text.lower().split()


class BM25Retriever:
    """Retrieve documents using lexical similarity."""

    def __init__(
        self,
        documents: list[Document],
        k: int = BM25_SEARCH_K,
    ) -> None:
        if k <= 0:
            raise ValueError("k doit être strictement positif.")

        if not documents:
            raise ValueError("La liste de documents BM25 ne peut pas être vide.")

        self.documents = documents
        self.k = k
        self._tokenized_documents = [
            tokenize(document.page_content) for document in documents
        ]
        self._index = BM25Okapi(self._tokenized_documents)

    
    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Return the top-k documents ranked by BM25 score."""
        if not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # Return the top-k ranked documents, including zero or negative scores.
        # A score can be non-positive even for a relevant document.
        return [
            (self.documents[index], float(scores[index]))
            for index in ranked_indices[: self.k]
        ]
