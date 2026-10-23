
"""Run the MediQuery document ingestion pipeline."""

import logging

from mediquery.config import SOURCE_PDF
from mediquery.ingestion.chunker import split_documents
from mediquery.ingestion.indexer import index_documents
from mediquery.ingestion.loader import load_pdf

logger = logging.getLogger(__name__)


def run_ingestion() -> dict[str, int]:
    """Load, split and index the configured medical PDF."""
    logger.info("Démarrage de l'ingestion : %s", SOURCE_PDF.name)

    # 1. Load source PDF
    documents = load_pdf(SOURCE_PDF)
    logger.info("Pages chargées : %d", len(documents))

    # 2. Split documents into chunks
    chunks = split_documents(documents)
    logger.info("Chunks générés : %d", len(chunks))

    # 3. Index chunks in ChromaDB
    vector_store = index_documents(chunks)

    # 4. Report collection statistics
    collection_count = vector_store._collection.count()
    logger.info("Chunks présents dans la collection : %d", collection_count)

    return {
        "pages_loaded": len(documents),
        "chunks_generated": len(chunks),
        "collection_count": collection_count,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    statistics = run_ingestion()
    print("\nRésumé de l'ingestion")
    for name, value in statistics.items():
        print(f"{name}: {value}")
