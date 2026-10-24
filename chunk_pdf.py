
import pymupdf


def extract_pages(pdf_path):
    """Extract text from a PDF while preserving page numbers."""
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if text:
                pages.append({
                    "text": text,
                    "page": page_number,
                })

    return pages


def create_chunks(pages, chunk_size=1000, overlap=200):
    """Split page text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                })

    return chunks


if __name__ == "__main__":
    pages = extract_pages("sample_paper.pdf")
    chunks = create_chunks(pages)

    print(f"Pages with text: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {index} | Page {chunk['page']} ---")
        print(chunk["text"])
