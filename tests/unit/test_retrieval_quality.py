
"""Manual smoke test for semantic retrieval quality."""

from mediquery.config import VECTOR_SEARCH_K
from mediquery.ingestion.indexer import create_embeddings
from mediquery.retrieval.vector_store import get_vector_store

QUESTIONS = [
    "Quelles sont les étapes du parcours de soins du diabète de type 2 ?",
    "Comment dépister et diagnostiquer un diabète de type 2 ?",
    "Quels sont les principes du traitement du diabète de type 2 ?",
    "Comment organiser le suivi du patient diabétique ?",
    "Quelles complications faut-il rechercher chez un patient diabétique ?",
]


def main() -> None:
    vector_store = get_vector_store(embeddings=create_embeddings())

    for question in QUESTIONS:
        results = vector_store.similarity_search_with_score(
            question,
            k=VECTOR_SEARCH_K,
        )

        print(f"\n{'=' * 90}")
        print(f"QUESTION : {question}")

        for rank, (document, distance) in enumerate(results, start=1):
            page = document.metadata.get("page", "?")
            print(
                f"\n{rank}. Page {page} | Distance : {distance:.4f}\n"
                f"{document.page_content[:350].replace(chr(10), ' ')}"
            )


if __name__ == "__main__":
    main()
