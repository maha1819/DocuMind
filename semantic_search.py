
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from chunk_pdf import extract_pages, create_chunks


# STEP 1: Extract text from the PDF and divide it into chunks
pdf_path = "sample_paper.pdf"

pages = extract_pages(pdf_path)
chunks = create_chunks(pages)

print(f"Pages with text: {len(pages)}")
print(f"Total chunks created: {len(chunks)}")


# STEP 2: Convert chunks into LangChain Documents.
# Metadata preserves the source filename and original PDF page number.
documents = [
    Document(
        page_content=chunk["text"],
        metadata={
            "source": pdf_path,
            "page": chunk["page"],
        },
    )
    for chunk in chunks
]

print(f"Documents prepared: {len(documents)}")


# STEP 3: Load a local embedding model.
# The model converts text into numerical vectors.
# The first run may download the model.
print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# STEP 4: Connect the local model to LangChain's Embeddings interface.
class LocalEmbeddings(Embeddings):

    def embed_documents(self, texts):
        """Convert multiple document chunks into vectors."""
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, text):
        """Convert a user's question into a vector."""
        vector = model.encode(
            text,
            normalize_embeddings=True,
        )
        return vector.tolist()


embeddings = LocalEmbeddings()


# STEP 5: Create a FAISS vector index from the document chunks.
# This index currently exists in memory while the script runs.
print("\nCreating FAISS vector index...")

vector_store = FAISS.from_documents(
    documents,
    embeddings,
)

print("FAISS index created successfully!")


# STEP 6: Search for the chunks most relevant to a question.
question = "What is the main contribution of this paper?"

print(f"\nQuestion: {question}")
print("\nSearching for relevant passages...")

results = vector_store.similarity_search(
    question,
    k=3,
)


# STEP 7: Display the top three results with their source pages.
print("\nMost relevant passages:")

for index, result in enumerate(results, start=1):
    print(f"\n{'=' * 60}")
    print(f"Result {index}")
    print(f"Source: {result.metadata['source']}")
    print(f"Page: {result.metadata['page']}")
    print("-" * 60)

    # Display at most 500 characters per result to keep output readable.
    print(result.page_content[:500])

print(f"\nSearch complete. Retrieved {len(results)} passages.")

