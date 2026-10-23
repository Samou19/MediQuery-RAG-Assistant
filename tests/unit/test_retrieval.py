
"""Test semantic retrieval against the MediQuery ChromaDB index."""

from mediquery.config import VECTOR_SEARCH_K
from mediquery.ingestion.indexer import COLLECTION_NAME, create_embeddings
from mediquery.retrieval.vector_store import get_vector_store


def main() -> None:
    vector_store = get_vector_store(embeddings=create_embeddings())

    question = (
        "Quelles sont les étapes du parcours de soins "
        "d'un adulte vivant avec un diabète de type 2 ?"
    )

    results = vector_store.similarity_search_with_score(
        question,
        k=VECTOR_SEARCH_K,
    )

    print(f"\nQuestion : {question}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Nombre de résultats : {len(results)}")

    for rank, (document, score) in enumerate(results, start=1):
        print(f"\n--- Résultat {rank} | Distance : {score:.4f} ---")
        print(f"Source : {document.metadata.get('source', 'inconnue')}")
        print(f"Page : {document.metadata.get('page', 'inconnue')}")
        print(document.page_content[:800])


if __name__ == "__main__":
    main()
