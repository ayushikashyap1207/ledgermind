from __future__ import annotations

import argparse
import json

from app.rag.pipeline import ingest_documents
from eval.rag_eval import run_rag_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["docs/rag_corpus"])
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()

    if args.experiment:
        candidates = [(550, 80), (700, 120), (900, 160)]
        reports = []
        for chunk_size, chunk_overlap in candidates:
            report = run_rag_eval(
                corpus_paths=args.paths,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=3,
                reset=True,
            )
            reports.append(report)

        reports.sort(key=lambda r: (r["passed"], -r["chunk_count"]), reverse=True)
        print(json.dumps({"experiment": reports, "recommended": reports[0]}, indent=2))
        return

    report = ingest_documents(
        paths=args.paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )
    print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    main()
