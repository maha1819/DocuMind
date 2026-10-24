
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from chunk_pdf import extract_pages, create_chunks


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddings(Embeddings):
    """Generate embeddings locally using Sentence Transformers."""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed_documents(self, texts):
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, text):
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return vector.tolist()


def build_vector_store(pdf_path):
    """
    Extract a PDF, split its text into chunks,
    and create a searchable FAISS vector store.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A FAISS vector store containing the PDF's chunks.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The uploaded file must be a PDF.")

    print(f"Processing: {pdf_path.name}")

    pages = extract_pages(str(pdf_path))
    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be empty or scanned."
        )

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": pdf_path.name,
                "page": chunk["page"],
            },
        )
        for chunk in chunks
    ]

    print(f"Pages with text: {len(pages)}")
    print(f"Chunks created: {len(documents)}")
    print("Loading embedding model...")

    embeddings = LocalEmbeddings()

    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    print("Document indexing complete!")

    return vector_store
