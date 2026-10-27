import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVALUATION_DIR / "eval_dataset.json"
PDF_PATH = PROJECT_ROOT / "sample_paper.pdf"

sys.path.insert(0, str(PROJECT_ROOT))
from document_index import build_vector_store


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    cases = data if isinstance(data, list) else data.get("cases", data.get("evaluation_cases"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a non-empty JSON list or contain a 'cases' list.")

    seen_ids = set()
    for case in cases:
        required = {"id", "question", "answerable", "expected_pages"}
        missing = required - set(case)
        if missing:
            raise ValueError(f"Case is missing fields {sorted(missing)}: {case}")
        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        seen_ids.add(case["id"])
        if not isinstance(case["answerable"], bool):
            raise ValueError(f"{case['id']}: answerable must be true or false.")
        if not isinstance(case["expected_pages"], list):
            raise ValueError(f"{case['id']}: expected_pages must be a list.")
        if case["answerable"] and not case["expected_pages"]:
            raise ValueError(f"{case['id']}: answerable cases need expected_pages.")
        if not case["answerable"] and case["expected_pages"]:
            raise ValueError(f"{case['id']}: unanswerable cases should have no expected_pages.")
    return cases


def retrieve_documents(vector_store, question, k, method):
    if method == "similarity":
        return vector_store.similarity_search(question, k=k)
    if method == "mmr":
        index_size = vector_store.index.ntotal
        if index_size == 0:
            return []
        fetch_k = min(max(4 * k, 20), index_size)
        return vector_store.max_marginal_relevance_search(
            question, k=min(k, index_size), fetch_k=fetch_k
        )
    raise ValueError(f"Unsupported retrieval method: {method}")


def page_number(document):
    value = document.metadata.get("page")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def reciprocal_rank(expected_pages, retrieved_pages):
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            return 1.0 / rank
    return 0.0


def evaluate_cases(vector_store, cases, k, method, pdf_name):
    results = []
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    hits, recalls, reciprocal_ranks = [], [], []

    for case in cases:
        expected = set(int(p) for p in case["expected_pages"])
        docs = retrieve_documents(vector_store, case["question"], k, method)
        pages = [p for doc in docs if (p := page_number(doc)) is not None]
        matched = expected.intersection(pages)
        result = {
            "id": case["id"],
            "question": case["question"],
            "answerable": case["answerable"],
            "expected_pages": sorted(expected),
            "retrieved_pages": pages,
            "matched_expected_pages": sorted(matched),
            "retrieved_chunk_count": len(docs),
        }

        if case["answerable"]:
            hit = bool(matched)
            recall = len(matched) / len(expected) if expected else 0.0
            rr = reciprocal_rank(expected, pages)
            hits.append(float(hit))
            recalls.append(recall)
            reciprocal_ranks.append(rr)
            result.update({
                "retrieval_status": "HIT" if hit else "MISS",
                "first_expected_page_rank": next(
                    (rank for rank, page in enumerate(pages, 1) if page in expected),
                    None,
                ),
                "reciprocal_rank": rr,
                "expected_page_recall": recall,
            })
        else:
            result["retrieval_status"] = "UNANSWERABLE — retrieval diagnostic only"

        results.append(result)
        print(f"{case['id']}: {result['retrieval_status']}")
        print(f"  Expected pages: {sorted(expected)}")
        print(f"  Retrieved pages: {pages}")
        if case["answerable"]:
            print(f"  First expected-page rank: {result['first_expected_page_rank']}")
            print(f"  Reciprocal rank: {result['reciprocal_rank']:.3f}")

    summary = {
        "pdf": pdf_name,
        "method": method,
        "k": k,
        "total_cases": len(cases),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "page_hit_rate_at_k": round(sum(hits) / len(hits), 4) if hits else None,
        "mean_expected_page_recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else None,
        "metric_note": (
            "Retrieval metrics are calculated on answerable cases only. "
            "MRR ranks retrieved chunks; duplicate pages can occur because multiple chunks may come from one PDF page. "
            "Unanswerable-case retrieval rows are diagnostic and are not included in page-hit/recall/MRR."
        ),
    }
    return summary, results


def output_path_for(args, method):
    if args.output:
        path = Path(args.output)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if args.method == "both":
            return path.with_name(f"{path.stem}_{method}{path.suffix or '.json'}")
        return path
    return EVALUATION_DIR / f"retrieval_results_{method}_k{args.k}.json"


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare DocuMind retrieval quality.")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve (default: 5).")
    parser.add_argument("--method", choices=["similarity", "mmr", "both"], default="similarity")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                        help="Dataset JSON path. Defaults to evaluation/eval_dataset.json.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output JSON path. With --method both, method suffixes are added.")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1.")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Sample PDF not found: {PDF_PATH}")

    cases = load_dataset(dataset_path)
    methods = ["similarity", "mmr"] if args.method == "both" else [args.method]

    print(f"PDF: {PDF_PATH.name}")
    print(f"Dataset: {dataset_path}")
    print(f"Evaluation cases: {len(cases)}")
    print(f"Answerable: {sum(c['answerable'] for c in cases)}")
    print(f"Unanswerable: {sum(not c['answerable'] for c in cases)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Top-k: {args.k}")
    print("\nBuilding document index once for a fair comparison...")
    vector_store = build_vector_store(str(PDF_PATH))

    summaries = {}
    for method in methods:
        print(f"\n{'=' * 16} {method.upper()} {'=' * 16}")
        summary, results = evaluate_cases(vector_store, cases, args.k, method, PDF_PATH.name)
        output_path = output_path_for(args, method)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump({"summary": summary, "results": results}, file, indent=2, ensure_ascii=False)
        summaries[method] = summary
        print("\nSummary:")
        print(json.dumps(summary, indent=2))
        print(f"Saved: {output_path}")

    if len(methods) == 2:
        print("\n========== RETRIEVAL COMPARISON ==========")
        fields = [
            ("page_hit_rate_at_k", "Page hit rate"),
            ("mean_expected_page_recall_at_k", "Expected-page recall"),
            ("mean_reciprocal_rank", "Mean reciprocal rank"),
        ]
        print(f"{'Metric':34} {'Similarity':>12} {'MMR':>12}")
        for key, label in fields:
            print(f"{label:34} {summaries['similarity'][key]:>12.4f} {summaries['mmr'][key]:>12.4f}")
        print("\nInterpret differences rather than assuming MMR is always better.")


if __name__ == "__main__":
    main()
