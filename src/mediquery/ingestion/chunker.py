
"""Split loaded documents into chunks using LangChain."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mediquery.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks while preserving their metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap ne peut pas être négatif.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être inférieur à chunk_size.")

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks
