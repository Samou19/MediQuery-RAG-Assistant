
"""List existing ChromaDB collections without modifying them."""

import chromadb

from mediquery.config import CHROMA_DIR


def main() -> None:
    """Display existing collections and their document counts."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print(f"Répertoire ChromaDB : {CHROMA_DIR}")

    collections = client.list_collections()

    if not collections:
        print("Aucune collection existante.")
        return

    for collection in collections:
        print(
            f"Collection : {collection.name} | "
            f"Documents : {collection.count()}"
        )


if __name__ == "__main__":
    main()
