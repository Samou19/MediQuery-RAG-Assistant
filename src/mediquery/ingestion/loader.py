
"""Load PDF documents without legacy LangChain community loaders."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from mediquery.config import SOURCE_PDF


def load_pdf(file_path: Path = SOURCE_PDF) -> list[Document]:
    """Extract PDF pages as LangChain documents with page metadata."""
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {file_path}")

    try:
        reader = PdfReader(str(file_path))
        documents: list[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "page": page_number,
                    },
                )
            )

    except (OSError, ValueError) as exc:
        raise ValueError(f"Impossible de lire le PDF : {file_path}") from exc

    if not documents:
        raise ValueError(
            f"Aucun texte extractible trouvé dans le PDF : {file_path}"
        )

    return documents
