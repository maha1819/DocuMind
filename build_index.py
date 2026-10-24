
import argparse
from pathlib import Path

from document_index import build_vector_store


def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index for any PDF."
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to index.",
    )

    parser.add_argument(
        "--output",
        default="faiss_index",
        help="Folder where the FAISS index will be saved.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    vector_store = build_vector_store(args.pdf_path)

    output_path.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(output_path))

    print("\nIndex saved successfully!")
    print(f"Index folder: {output_path.resolve()}")


if __name__ == "__main__":
    main()

