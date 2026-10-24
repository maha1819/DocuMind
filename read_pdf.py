import pymupdf

pdf_path = "sample_paper.pdf"

with pymupdf.open(pdf_path) as document:
    print(f"Total pages: {len(document)}")

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        print(f"\n--- Page {page_number} ---")
        print(text[:1000])