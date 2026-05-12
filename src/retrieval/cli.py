from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DEVICE,
    DEFAULT_FINAL_K,
    DEFAULT_INITIAL_K,
    DEFAULT_PERSIST_DIR,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_POOL_K,
    DEFAULT_SEARCH_TYPE,
)
from .eval import evaluate_retrieval
from .indexing import build_vector_index
from .query import retrieve_contexts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _add_shared_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "gpu"], default=DEFAULT_DEVICE)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--chunk-type", choices=["text", "table"], default=None)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--disable-hybrid", action="store_true")
    parser.add_argument("--search-type", choices=["hybrid", "vector", "keyword"], default=DEFAULT_SEARCH_TYPE)
    parser.add_argument("--rerank-pool-k", type=int, default=DEFAULT_RERANK_POOL_K)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval pipeline: index build + query + evaluate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build-index")
    build_cmd.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    build_cmd.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    build_cmd.add_argument("--collection", default=DEFAULT_COLLECTION)
    build_cmd.add_argument("--batch-size", type=int, default=256)

    query_cmd = subparsers.add_parser("query")
    query_cmd.add_argument("--question", required=True)
    _add_shared_query_args(query_cmd)

    eval_cmd = subparsers.add_parser("evaluate")
    eval_cmd.add_argument("--eval-file", required=True)
    _add_shared_query_args(eval_cmd)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "build-index":
        print(json.dumps(build_vector_index(Path(args.artifacts_dir), Path(args.persist_dir), args.collection, args.batch_size), ensure_ascii=False, indent=2))
        return
    if args.command == "query":
        contexts = retrieve_contexts(
            question=args.question,
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            initial_k=args.initial_k,
            final_k=args.final_k,
            embed_model_name=args.model,
            device=args.device,
            doc_id=args.doc_id,
            chunk_type=args.chunk_type,
            reranker_model=args.reranker_model,
            disable_rerank=args.disable_rerank,
            enable_hybrid=not args.disable_hybrid,
            search_type=args.search_type,
            rerank_pool_k=args.rerank_pool_k,
        )
        print(json.dumps({"question": args.question, "result_count": len(contexts), "contexts": contexts}, ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        report = evaluate_retrieval(
            eval_path=Path(args.eval_file),
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            initial_k=args.initial_k,
            final_k=args.final_k,
            device=args.device,
            reranker_model=args.reranker_model,
            disable_rerank=args.disable_rerank,
            disable_hybrid=args.disable_hybrid,
            search_type=args.search_type,
            rerank_pool_k=args.rerank_pool_k,
            embed_model_name=args.model,
            doc_id=args.doc_id,
            chunk_type=args.chunk_type,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    raise ValueError(f"Bilinmeyen komut: {args.command}")


if __name__ == "__main__":
    main()

