from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .io import read_jsonl
from .query import retrieve_contexts


def _rank_hits(
    predicted: Sequence[Dict[str, Any]],
    relevant_chunks: set[str],
    relevant_docs: set[str],
    k: int,
) -> List[int]:
    hits: List[int] = []
    seen_relevant: set[str] = set()
    for idx, row in enumerate(predicted[:k], start=1):
        rel_key = str(row.get("chunk_id", "")) if relevant_chunks else str(row.get("doc_id", ""))
        is_hit = rel_key in relevant_chunks if relevant_chunks else rel_key in relevant_docs
        if is_hit and rel_key not in seen_relevant:
            hits.append(idx)
            seen_relevant.add(rel_key)
    return hits


def _compute_metrics(
    predictions: List[List[Dict[str, Any]]],
    labels: List[Dict[str, Any]],
    k: int,
) -> Dict[str, float]:
    if not predictions:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0}
    recall_values: List[float] = []
    rr_values: List[float] = []
    ndcg_values: List[float] = []
    for predicted, label in zip(predictions, labels):
        relevant_chunks = set(label.get("relevant_chunk_ids", []) or [])
        relevant_docs = set(label.get("relevant_doc_ids", []) or [])
        if not relevant_chunks and not relevant_docs:
            continue
        hits = _rank_hits(predicted, relevant_chunks, relevant_docs, k)
        denom = len(relevant_chunks) if relevant_chunks else len(relevant_docs)
        denom = max(denom, 1)
        recall_values.append(len(hits) / denom)
        rr_values.append(1.0 / hits[0] if hits else 0.0)
        dcg = sum(1.0 / math.log2(rank + 1.0) for rank in hits)
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, min(denom, k) + 1))
        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
    if not recall_values:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0}
    return {
        "recall@k": float(np.mean(recall_values)),
        "mrr": float(np.mean(rr_values)),
        "ndcg@k": float(np.mean(ndcg_values)),
    }


def evaluate_retrieval(
    eval_path: Path,
    persist_dir: Path,
    collection_name: str,
    initial_k: int,
    final_k: int,
    device: str,
    reranker_model: str,
    disable_rerank: bool,
    disable_hybrid: bool,
    search_type: str,
    rerank_pool_k: int,
    embed_model_name: Optional[str] = None,
    doc_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> Dict[str, Any]:
    rows = read_jsonl(eval_path)
    if not rows:
        raise ValueError(f"Eval dosyasi bos: {eval_path}")
    labels: List[Dict[str, Any]] = []
    predictions: List[List[Dict[str, Any]]] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        predictions.append(
            retrieve_contexts(
                question=question,
                persist_dir=persist_dir,
                collection_name=collection_name,
                initial_k=initial_k,
                final_k=final_k,
                embed_model_name=embed_model_name,
                device=device,
                doc_id=row.get("filter_doc_id") or doc_id,
                chunk_type=row.get("filter_chunk_type") or chunk_type,
                reranker_model=reranker_model,
                disable_rerank=disable_rerank,
                enable_hybrid=not (disable_hybrid or bool(row.get("disable_hybrid", False))),
                search_type=str(row.get("search_type", search_type)),
                rerank_pool_k=rerank_pool_k,
            )
        )
        labels.append(row)
    metrics = _compute_metrics(predictions=predictions, labels=labels, k=final_k)
    return {
        "query_count": len(predictions),
        "initial_k": initial_k,
        "final_k": final_k,
        "disable_rerank": disable_rerank,
        "disable_hybrid": disable_hybrid,
        "search_type": search_type,
        "rerank_pool_k": rerank_pool_k,
        "metrics": metrics,
    }

