from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import DEFAULT_EMBED_MODEL
from .runtime import (
    embed_question,
    get_chromadb,
    get_sparse_candidates,
    rerank_candidates,
    resolve_device,
    rrf_fuse,
)
from .types import RetrievalCandidate

LOGGER = logging.getLogger(__name__)


def _resolve_embed_model_for_query(collection: Any, explicit_model: Optional[str]) -> str:
    if explicit_model:
        return explicit_model
    metadata = getattr(collection, "metadata", None) or {}
    model_name = metadata.get("embedding_model")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return DEFAULT_EMBED_MODEL


def _build_where_filter(doc_id: Optional[str], chunk_type: Optional[str]) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    if doc_id:
        clauses.append({"doc_id": {"$eq": str(doc_id)}})
    if chunk_type:
        clauses.append({"chunk_type": {"$eq": str(chunk_type)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_contexts(
    question: str,
    persist_dir: Path,
    collection_name: str,
    initial_k: int,
    final_k: int,
    embed_model_name: Optional[str] = None,
    device: str = "auto",
    doc_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    reranker_model: str = "",
    disable_rerank: bool = False,
    enable_hybrid: bool = True,
    search_type: str = "hybrid",
    rerank_pool_k: int = 16,
) -> List[Dict[str, Any]]:
    if not question.strip():
        raise ValueError("Soru bos olamaz.")
    if initial_k < 1:
        raise ValueError("initial_k en az 1 olmali.")
    if final_k < 1:
        raise ValueError("final_k en az 1 olmali.")
    if final_k > initial_k:
        raise ValueError("final_k, initial_k'den buyuk olamaz.")

    chroma = get_chromadb()
    client = chroma.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)
    model_name = _resolve_embed_model_for_query(collection=collection, explicit_model=embed_model_name)
    resolved_device = resolve_device(device)
    query_vector = embed_question(question=question, model_name=model_name, device=resolved_device)
    where_filter = _build_where_filter(doc_id=doc_id, chunk_type=chunk_type)

    query_result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=initial_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    ids = query_result.get("ids", [[]])
    if not ids or not ids[0]:
        return []
    documents = query_result.get("documents", [[]])
    metadatas = query_result.get("metadatas", [[]])
    distances = query_result.get("distances", [[]])

    vector_candidates: List[RetrievalCandidate] = []
    for cid, doc_text, meta, dist in zip(
        ids[0],
        documents[0] if documents else ["" for _ in ids[0]],
        metadatas[0] if metadatas else [{} for _ in ids[0]],
        distances[0] if distances else [1.0 for _ in ids[0]],
    ):
        metadata = meta or {}
        vector_distance = float(dist) if dist is not None else 1.0
        vector_candidates.append(
            RetrievalCandidate(
                chunk_id=str(cid),
                doc_id=str(metadata.get("doc_id", "")),
                page=int(metadata.get("page", 0)),
                section=str(metadata.get("section", "ROOT")),
                chunk_type=str(metadata.get("chunk_type", "text")),
                text=str(doc_text or ""),
                char_len=int(metadata.get("char_len", len(str(doc_text or "")))),
                vector_distance=vector_distance,
                vector_score=1.0 - vector_distance,
            )
        )
    vector_candidates = sorted(vector_candidates, key=lambda x: x.vector_distance)

    sparse_candidates: List[RetrievalCandidate] = []
    if search_type in {"hybrid", "keyword"}:
        try:
            sparse_candidates = get_sparse_candidates(collection, question, initial_k, where_filter)
        except Exception as exc:
            LOGGER.warning("Sparse/BM25 asamasi hata verdi: %s", exc)

    hybrid_applied = False
    if search_type == "vector":
        candidates = vector_candidates
    elif search_type == "keyword":
        candidates = sparse_candidates
    elif search_type == "hybrid":
        if enable_hybrid and sparse_candidates:
            candidates = rrf_fuse(vector_candidates=vector_candidates, sparse_candidates=sparse_candidates)
            hybrid_applied = True
        else:
            candidates = vector_candidates
    else:
        raise ValueError("search_type yalnizca 'hybrid', 'vector' veya 'keyword' olabilir.")

    rerank_applied = False
    if not disable_rerank and len(candidates) > 1:
        rerank_pool = max(1, min(rerank_pool_k, len(candidates)))
        candidates, rerank_applied = rerank_candidates(
            question=question,
            candidates=candidates[:rerank_pool],
            reranker_model=reranker_model,
            device=resolved_device,
        )

    selected = candidates[:final_k]
    return [
        {
            "chunk_id": row.chunk_id,
            "doc_id": row.doc_id,
            "page": row.page,
            "section": row.section,
            "chunk_type": row.chunk_type,
            "text": row.text,
            "char_len": row.char_len,
            "vector_distance": row.vector_distance,
            "vector_score": row.vector_score,
            "sparse_score": row.sparse_score,
            "rrf_score": row.rrf_score,
            "hybrid_applied": hybrid_applied,
            "search_type": search_type,
            "rerank_score": row.rerank_score,
            "rerank_applied": rerank_applied,
            "model_name": model_name,
            "device": resolved_device,
        }
        for row in selected
    ]

