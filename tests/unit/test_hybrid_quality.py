
"""Compare hybrid retrieval results on the indexed medical corpus."""

from langchain_core.documents import Document

from mediquery.config import BM25_SEARCH_K
from mediquery.retrieval.hybrid import HybridRetriever
from mediquery.retrieval.vector_store import get_vector_store

QUESTIONS = [
    "Quelles sont les étapes du parcours de soins du diabète de type 2 ?",
    "Comment dépister et diagnostiquer un diabète de type 2 ?",
    "Quels sont les principes du traitement du diabète de type 2 ?",
    "Comment organiser le suivi du patient diabétique ?",
    "Quelles complications faut-il rechercher chez un patient diabétique ?",
]


def main() -> None:
    vector_store = get_vector_store()
    records = vector_store._collection.get(
        include=["documents", "metadatas"],
    )

    documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(
            records["documents"],
            records["metadatas"],
            strict=True,
        )
        if content
    ]

    print(f"Chunks chargés : {len(documents)}")

    retriever = HybridRetriever(
        documents,
        bm25_k=BM25_SEARCH_K,
    )

    for question in QUESTIONS:
        print(f"\n{'=' * 90}")
        print(f"QUESTION : {question}")

        results = retriever.retrieve(question)

        for rank, document in enumerate(results, start=1):
            print(
                f"\n{rank}. Page {document.metadata.get('page', '?')} "
                f"| Chunk {document.metadata.get('chunk_index', '?')}"
            )
            print(document.page_content[:350].replace("\n", " "))


if __name__ == "__main__":
    main()
