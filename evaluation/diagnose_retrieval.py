
"""Inspect retrieval rankings for difficult evaluation questions."""

from evaluation.evaluate_retrieval import load_documents_from_chroma

from mediquery.config import BM25_SEARCH_K, VECTOR_SEARCH_K
from mediquery.retrieval.bm25 import BM25Retriever
from mediquery.retrieval.hybrid import HybridRetriever
from mediquery.retrieval.vector_store import get_vector_store

QUESTIONS = {
    "Q03 - Traitement": (
        "Quels sont les principes du traitement du diabète de type 2 ?"
    ),
    "Q04 - Suivi": (
        "Comment organiser le suivi du patient diabétique ?"
    ),
    "Q04bis - Suivi systématique": (
        "Quels éléments figurent dans la check-list du suivi systématique "
        "du diabète de type 2, et à quelle fréquence sont-ils contrôlés ?"
    )
}


def display_results(strategy, documents, limit=15):
    print(f"\n--- {strategy} ---")

    seen_pages = set()

    for rank, document in enumerate(documents, start=1):
        page = document.metadata.get("page", "?")
        marker = ""

        if page not in seen_pages:
            seen_pages.add(page)
            marker = " [nouvelle page]"

        print(
            f"Rang {rank:02d} | Page {page}{marker} | "
            f"{document.page_content[:180].replace(chr(10), ' ')}"
        )

        if rank >= limit:
            break


def main():
    documents = load_documents_from_chroma()
    vector_store = get_vector_store()

    bm25 = BM25Retriever(
        documents,
        k=max(BM25_SEARCH_K, 15),
    )

    hybrid = HybridRetriever(
        documents,
        bm25_k=max(BM25_SEARCH_K, 15),
        vector_k=max(VECTOR_SEARCH_K, 15),
        result_k=30,
    )

    for label, question in QUESTIONS.items():
        print(f"\n{'=' * 90}\n{label}\nQuestion : {question}")

        vector_results = vector_store.similarity_search(
            question, k=15
        )
        bm25_results = [
            document
            for document, _score in bm25.retrieve(question)
        ]
        hybrid_results = hybrid.retrieve(question)

        display_results("VECTOR", vector_results)
        display_results("BM25", bm25_results)
        display_results("HYBRID RRF", hybrid_results)


if __name__ == "__main__":
    main()
