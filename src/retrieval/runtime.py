from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import EXPECTED_MURSIT_DIM
from .types import RetrievalCandidate

LOGGER = logging.getLogger(__name__)
_EMBEDDER_CACHE: Dict[Tuple[str, str], Any] = {}
_RERANKER_CACHE: Dict[Tuple[str, str], Any] = {}

try:
    import chromadb as _chromadb
except Exception:  # pragma: no cover
    _chromadb = None


def get_chromadb() -> Any:
    if _chromadb is None:
        raise RuntimeError("chromadb import edilemedi. Poetry bagimliliklarini kurun: `poetry install`.")
    return _chromadb


def get_sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers import edilemedi. Poetry bagimliliklarini kurun: `poetry install`."
        ) from exc


def get_cross_encoder_class() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder


def resolve_device(device_arg: str) -> str:
    if device_arg == "gpu":
        device_arg = "cuda"
    try:
        import torch
    except Exception:
        return "cpu"
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_question(question: str, model_name: str, device: str) -> np.ndarray:
    resolved_device = resolve_device(device)
    cache_key = (model_name, resolved_device)
    embedder = _EMBEDDER_CACHE.get(cache_key)
    if embedder is None:
        SentenceTransformer = get_sentence_transformer_class()
        embedder = SentenceTransformer(model_name, device=resolved_device)
        _EMBEDDER_CACHE[cache_key] = embedder
        LOGGER.info("Query embedding modeli cache'e alindi | model=%s | device=%s", model_name, resolved_device)
    tokenizer = getattr(embedder, "tokenizer", None)
    tok_name = tokenizer.__class__.__name__ if tokenizer is not None else "unknown"
    LOGGER.info("Query embedding modeli hazir | model=%s | device=%s | tokenizer=%s", model_name, resolved_device, tok_name)
    vector = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    vector = np.asarray(vector, dtype=np.float32)
    if model_name == "newmindai/Mursit-Large-TR-Retrieval" and vector.shape[1] != EXPECTED_MURSIT_DIM:
        raise RuntimeError(
            f"Mursit query embedding boyutu beklenenle uyusmuyor. Beklenen={EXPECTED_MURSIT_DIM}, gelen={vector.shape[1]}"
        )
    return vector[0]


def rerank_candidates(
    question: str,
    candidates: Sequence[RetrievalCandidate],
    reranker_model: str,
    device: str,
) -> Tuple[List[RetrievalCandidate], bool]:
    if len(candidates) <= 1:
        return list(candidates), False
    CrossEncoder = get_cross_encoder_class()
    resolved_device = resolve_device(device)
    cache_key = (reranker_model, resolved_device)
    reranker = _RERANKER_CACHE.get(cache_key)
    if reranker is None:
        reranker = CrossEncoder(reranker_model, device=resolved_device)
        _RERANKER_CACHE[cache_key] = reranker
        LOGGER.info("Cross-encoder cache'e alindi | model=%s | device=%s", reranker_model, resolved_device)
    pairs = [(question, row.text) for row in candidates]
    scores = reranker.predict(pairs)
    ordered = list(candidates)
    for row, score in zip(ordered, scores):
        row.rerank_score = float(score)
    ordered.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -math.inf, reverse=True)
    return ordered, True


def tokenize_tr(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]+", text.lower())
    return [tok for tok in tokens if len(tok) > 1]


def get_sparse_candidates(
    collection: Any,
    question: str,
    initial_k: int,
    where_filter: Optional[Dict[str, Any]],
) -> List[RetrievalCandidate]:
    q_tokens = tokenize_tr(question)
    if not q_tokens:
        return []
    # Chroma'da "ids" include listesine yazilmaz; ids her zaman doner.
    docs = collection.get(where=where_filter, include=["documents", "metadatas"])
    ids = docs.get("ids", []) or []
    documents = docs.get("documents", []) or []
    metadatas = docs.get("metadatas", []) or []
    if not ids:
        return []
    doc_tokens: List[List[str]] = [tokenize_tr(str(d or "")) for d in documents]
    N = len(doc_tokens)
    df: Counter[str] = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1
    q_tf = Counter(q_tokens)
    candidates: List[RetrievalCandidate] = []
    for cid, text, meta, toks in zip(ids, documents, metadatas, doc_tokens):
        d_tf = Counter(toks)
        d_len = max(len(toks), 1)
        score = 0.0
        for token, qf in q_tf.items():
            if token not in d_tf:
                continue
            tf = d_tf[token]
            idf = math.log((N - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
            score += (tf / d_len) * idf * (1.0 + 0.2 * qf)
        if score <= 0:
            continue
        metadata = meta or {}
        candidates.append(
            RetrievalCandidate(
                chunk_id=str(cid),
                doc_id=str(metadata.get("doc_id", "")),
                page=int(metadata.get("page", 0)),
                section=str(metadata.get("section", "ROOT")),
                chunk_type=str(metadata.get("chunk_type", "text")),
                text=str(text or ""),
                char_len=int(metadata.get("char_len", len(str(text or "")))),
                vector_distance=1.0,
                vector_score=0.0,
                sparse_score=float(score),
            )
        )
    candidates.sort(key=lambda x: x.sparse_score if x.sparse_score is not None else -math.inf, reverse=True)
    return candidates[:initial_k]


def rrf_fuse(
    vector_candidates: Sequence[RetrievalCandidate],
    sparse_candidates: Sequence[RetrievalCandidate],
    k: int = 60,
) -> List[RetrievalCandidate]:
    merged: Dict[str, RetrievalCandidate] = {}
    for rank, cand in enumerate(vector_candidates, start=1):
        score = 1.0 / (k + rank)
        existing = merged.get(cand.chunk_id)
        if existing is None:
            clone = RetrievalCandidate(**cand.__dict__)
            clone.rrf_score = score
            merged[cand.chunk_id] = clone
        else:
            existing.rrf_score = (existing.rrf_score or 0.0) + score
    for rank, cand in enumerate(sparse_candidates, start=1):
        score = 1.0 / (k + rank)
        existing = merged.get(cand.chunk_id)
        if existing is None:
            clone = RetrievalCandidate(**cand.__dict__)
            clone.rrf_score = score
            merged[cand.chunk_id] = clone
        else:
            existing.rrf_score = (existing.rrf_score or 0.0) + score
            if cand.sparse_score is not None:
                existing.sparse_score = cand.sparse_score
    fused = list(merged.values())
    fused.sort(key=lambda x: x.rrf_score if x.rrf_score is not None else -math.inf, reverse=True)
    return fused
