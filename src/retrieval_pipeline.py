"""Faz 3: Chroma + cosine retrieval + cross-encoder rerank pipeline.

Bu script, Faz 2 artefaktlarindan (chunks.jsonl + embeddings.npy + manifest.json)
vektor index kurar ve soru bazli retrieval yapar.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from config import (
    CHUNK_ARTIFACTS_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    VECTOR_PERSIST_DIR,
)

try:
    import chromadb as _chromadb
except Exception:  # pragma: no cover
    _chromadb = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


DEFAULT_ARTIFACTS_DIR = CHUNK_ARTIFACTS_DIR
DEFAULT_PERSIST_DIR = VECTOR_PERSIST_DIR
DEFAULT_COLLECTION = COLLECTION_NAME
DEFAULT_EMBED_MODEL = EMBED_MODEL_NAME
DEFAULT_RERANK_MODEL = RERANK_MODEL_NAME
DEFAULT_INITIAL_K = 24
DEFAULT_FINAL_K = 5
DEFAULT_DEVICE = "cuda"
EXPECTED_MURSIT_DIM = 1024
DEFAULT_SEARCH_TYPE = "hybrid"
DEFAULT_RERANK_POOL_K = 16
_EMBEDDER_CACHE: Dict[Tuple[str, str], Any] = {}
_RERANKER_CACHE: Dict[Tuple[str, str], Any] = {}


@dataclass
class RetrievalCandidate:
    chunk_id: str
    doc_id: str
    page: int
    section: str
    chunk_type: str
    text: str
    char_len: int
    vector_distance: float
    vector_score: float
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


def _get_chromadb() -> Any:
    if _chromadb is None:
        raise RuntimeError("chromadb import edilemedi. Poetry bagimliliklarini kurun: `poetry install`.")
    return _chromadb


def _get_sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers import edilemedi. Poetry bagimliliklarini kurun: `poetry install`."
        ) from exc


def _get_cross_encoder_class() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder


def _resolve_device(device_arg: str) -> str:
    if device_arg == "gpu":
        device_arg = "cuda"
    if device_arg != "auto":
        return device_arg
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                rows.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse hatasi ({path}:{line_no}): {exc}") from exc
    return rows


def _load_artifacts(artifacts_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json bulunamadi: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    chunks_name = artifacts.get("chunks_jsonl", "chunks.jsonl")
    embeddings_name = artifacts.get("embeddings_npy", "embeddings.npy")

    chunks_path = artifacts_dir / chunks_name
    embeddings_path = artifacts_dir / embeddings_name
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks dosyasi bulunamadi: {chunks_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"embeddings dosyasi bulunamadi: {embeddings_path}")

    chunks = _read_jsonl(chunks_path)
    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings 2 boyutlu olmali. Gelen sekil: {embeddings.shape}")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Satir sayisi uyusmuyor: chunks={len(chunks)} embeddings={embeddings.shape[0]}"
        )

    manifest_chunk_count = int(manifest.get("chunk_count", len(chunks)))
    if manifest_chunk_count != len(chunks):
        raise ValueError(
            f"Manifest chunk_count uyusmuyor: manifest={manifest_chunk_count}, gercek={len(chunks)}"
        )

    manifest_dim = int(manifest.get("embedding_dim", embeddings.shape[1]))
    if manifest_dim != embeddings.shape[1]:
        raise ValueError(
            f"Manifest embedding_dim uyusmuyor: manifest={manifest_dim}, gercek={embeddings.shape[1]}"
        )

    return chunks, embeddings, manifest


def _delete_collection_if_exists(client: Any, collection_name: str) -> bool:
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False


def _create_collection_cosine(client: Any, collection_name: str, metadata: Dict[str, Any]) -> Any:
    try:
        return client.create_collection(
            name=collection_name,
            metadata=metadata,
            configuration={"hnsw": {"space": "cosine"}},
        )
    except TypeError:
        # Chroma legacy API fallback
        legacy_metadata = dict(metadata)
        legacy_metadata["hnsw:space"] = "cosine"
        return client.create_collection(name=collection_name, metadata=legacy_metadata)


def _batched(seq: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, len(seq), batch_size):
        end = min(start + batch_size, len(seq))
        yield start, end


def build_vector_index(
    artifacts_dir: Path,
    persist_dir: Path,
    collection_name: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    chunks, embeddings, manifest = _load_artifacts(artifacts_dir=artifacts_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    requested_model = str(manifest.get("model_name", DEFAULT_EMBED_MODEL))
    requested_dim = int(manifest.get("embedding_dim", int(embeddings.shape[1])))

    if requested_model == "newmindai/Mursit-Large-TR-Retrieval" and requested_dim != EXPECTED_MURSIT_DIM:
        raise RuntimeError(
            f"Mursit modeli icin embedding boyutu uyumsuz. Beklenen={EXPECTED_MURSIT_DIM}, gelen={requested_dim}. "
            "Chunk/embedding artefaktlarini yeni model ile tekrar uretin."
        )

    chroma = _get_chromadb()
    client = chroma.PersistentClient(path=str(persist_dir))
    old_manifest = persist_dir / f"{collection_name}_index_manifest.json"
    if old_manifest.exists():
        try:
            old = json.loads(old_manifest.read_text(encoding="utf-8"))
            old_model = str(old.get("model_name", ""))
            old_dim = int(old.get("embedding_dim", -1))
            if old_model != requested_model or old_dim != requested_dim:
                LOGGER.warning(
                    "Index model/dim degisti. Re-index yapiliyor | old=%s/%s new=%s/%s",
                    old_model,
                    old_dim,
                    requested_model,
                    requested_dim,
                )
        except Exception:
            LOGGER.warning("Eski index manifest okunamadi; temiz re-index devam ediyor.")

    deleted = _delete_collection_if_exists(client=client, collection_name=collection_name)
    if deleted:
        LOGGER.info("Mevcut collection silindi: %s", collection_name)

    collection_meta: Dict[str, Any] = {
        "phase": "faz3",
        "embedding_model": str(manifest.get("model_name", DEFAULT_EMBED_MODEL)),
        "embedding_dim": int(manifest.get("embedding_dim", int(embeddings.shape[1]))),
        "source_chunk_count": int(len(chunks)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    collection = _create_collection_cosine(
        client=client,
        collection_name=collection_name,
        metadata=collection_meta,
    )

    for start, end in _batched(chunks, batch_size=batch_size):
        chunk_batch = chunks[start:end]
        vector_batch = embeddings[start:end]

        ids = [str(row["chunk_id"]) for row in chunk_batch]
        documents = [str(row.get("text", "")) for row in chunk_batch]
        metadatas: List[Dict[str, Any]] = []
        for row in chunk_batch:
            section_title = str(row.get("section_title", row.get("section", "ROOT")))
            page_no = int(row.get("page_no", row.get("page", 0)))
            is_table = bool(row.get("is_table", str(row.get("chunk_type", "text")) == "table"))
            metadatas.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "doc_id": str(row.get("doc_id", "")),
                    "page": page_no,
                    "page_no": page_no,
                    "section": section_title,
                    "section_title": section_title,
                    "is_table": is_table,
                    "chunk_type": str(row.get("chunk_type", "text")),
                    "char_len": int(row.get("char_len", len(str(row.get("text", ""))))),
                }
            )

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vector_batch.tolist(),
        )

    indexed_count = int(collection.count())
    if indexed_count != len(chunks):
        raise RuntimeError(
            f"Index kayit sayisi uyusmuyor: collection={indexed_count}, beklenen={len(chunks)}"
        )

    index_manifest = {
        "collection_name": collection_name,
        "persist_dir": str(persist_dir),
        "source_artifacts_dir": str(artifacts_dir),
        "indexed_chunk_count": indexed_count,
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": manifest.get("model_name", DEFAULT_EMBED_MODEL),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (persist_dir / f"{collection_name}_index_manifest.json").write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_manifest


def _resolve_embed_model_for_query(
    collection: Any,
    explicit_model: Optional[str],
) -> str:
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
        clauses.append({"doc_id": {"$eq": doc_id}})
    if chunk_type:
        clauses.append({"chunk_type": {"$eq": chunk_type}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _tokenize_tr(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+", text.lower())


def _get_sparse_candidates(
    collection: Any,
    question: str,
    initial_k: int,
    where_filter: Optional[Dict[str, Any]],
) -> List[RetrievalCandidate]:
    """Collection içindeki dokümanlar üzerinde BM25 benzeri sparse skorlama yapar."""
    data = collection.get(where=where_filter, include=["documents", "metadatas"])
    ids = data.get("ids", []) or []
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    if not ids:
        return []

    tokenized_docs: List[List[str]] = [_tokenize_tr(str(d or "")) for d in docs]
    query_tokens = _tokenize_tr(question)
    if not query_tokens:
        return []

    n_docs = len(tokenized_docs)
    doc_lens = [len(toks) for toks in tokenized_docs]
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0
    if avgdl <= 0:
        return []

    df: Counter[str] = Counter()
    for toks in tokenized_docs:
        for term in set(toks):
            df[term] += 1

    k1 = 1.5
    b = 0.75
    scored: List[Tuple[int, float]] = []
    for idx, toks in enumerate(tokenized_docs):
        tf = Counter(toks)
        score = 0.0
        dl = len(toks)
        norm = k1 * (1.0 - b + b * (dl / avgdl))
        for q in query_tokens:
            if q not in tf:
                continue
            term_df = df.get(q, 0)
            if term_df <= 0:
                continue
            idf = math.log(1.0 + ((n_docs - term_df + 0.5) / (term_df + 0.5)))
            term_tf = tf[q]
            score += idf * ((term_tf * (k1 + 1.0)) / (term_tf + norm))
        if score > 0:
            scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:initial_k]

    candidates: List[RetrievalCandidate] = []
    for idx, sparse_score in top:
        meta = metas[idx] or {}
        doc_text = str(docs[idx] or "")
        cid = str(ids[idx])
        candidates.append(
            RetrievalCandidate(
                chunk_id=cid,
                doc_id=str(meta.get("doc_id", "")),
                page=int(meta.get("page", 0)),
                section=str(meta.get("section", "ROOT")),
                chunk_type=str(meta.get("chunk_type", "text")),
                text=doc_text,
                char_len=int(meta.get("char_len", len(doc_text))),
                vector_distance=1.0,
                vector_score=0.0,
                sparse_score=float(sparse_score),
            )
        )
    return candidates


def _rrf_fuse(
    vector_candidates: List[RetrievalCandidate],
    sparse_candidates: List[RetrievalCandidate],
    rrf_k: int = 60,
) -> List[RetrievalCandidate]:
    by_id: Dict[str, RetrievalCandidate] = {}
    scores: Dict[str, float] = {}

    for rank, cand in enumerate(vector_candidates, start=1):
        cid = cand.chunk_id
        by_id[cid] = cand
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    for rank, cand in enumerate(sparse_candidates, start=1):
        cid = cand.chunk_id
        if cid not in by_id:
            by_id[cid] = cand
        else:
            by_id[cid].sparse_score = cand.sparse_score
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    merged = list(by_id.values())
    for row in merged:
        row.rrf_score = float(scores.get(row.chunk_id, 0.0))
    merged.sort(key=lambda x: x.rrf_score if x.rrf_score is not None else 0.0, reverse=True)
    return merged


def _embed_question(question: str, model_name: str, device: str) -> np.ndarray:
    if not question.strip():
        raise ValueError("Soru bos olamaz.")
    sentence_transformer_cls = _get_sentence_transformer_class()
    cache_key = (model_name, device)
    model = _EMBEDDER_CACHE.get(cache_key)
    if model is None:
        try:
            model = sentence_transformer_cls(model_name_or_path=model_name, device=device)
            _EMBEDDER_CACHE[cache_key] = model
        except Exception as exc:
            msg = str(exc)
            network_hints = ("Connection", "ReadTimeout", "NameResolutionError", "Temporary failure", "HTTPSConnectionPool")
            if any(h in msg for h in network_hints):
                raise RuntimeError(
                    "Embedding modeli indirilemedi/yuklenemedi. HuggingFace ag erisimi yok olabilir. "
                    "Air-gapped ortam icin modeli onceden local cache/mirror'a alin."
                ) from exc
            raise
        LOGGER.info("Query embedding modeli cache'e alindi | model=%s | device=%s", model_name, device)
    tokenizer = getattr(model, "tokenizer", None)
    tok_name = tokenizer.__class__.__name__ if tokenizer is not None else "unknown"
    LOGGER.info("Query embedding modeli hazir | model=%s | device=%s | tokenizer=%s", model_name, device, tok_name)
    vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vector[0], dtype=np.float32)


def _rerank_candidates(
    question: str,
    candidates: List[RetrievalCandidate],
    reranker_model: str,
    device: str,
) -> Tuple[List[RetrievalCandidate], bool]:
    if not candidates:
        return candidates, False
    try:
        cache_key = (reranker_model, device)
        reranker = _RERANKER_CACHE.get(cache_key)
        if reranker is None:
            cross_encoder_cls = _get_cross_encoder_class()
            reranker = cross_encoder_cls(reranker_model, device=device)
            _RERANKER_CACHE[cache_key] = reranker
            LOGGER.info("Cross-encoder cache'e alindi | model=%s | device=%s", reranker_model, device)
        pairs = [(question, row.text) for row in candidates]
        scores = reranker.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
    except Exception as exc:
        LOGGER.warning("Reranker yuklenemedi. Vector sirasi ile devam ediliyor. Sebep: %s", exc)
        return candidates, False

    for row, score in zip(candidates, scores):
        row.rerank_score = float(score)

    ranked = sorted(
        candidates,
        key=lambda x: x.rerank_score if x.rerank_score is not None else float("-inf"),
        reverse=True,
    )
    return ranked, True


def retrieve_contexts(
    question: str,
    persist_dir: Path,
    collection_name: str,
    initial_k: int = DEFAULT_INITIAL_K,
    final_k: int = DEFAULT_FINAL_K,
    embed_model_name: Optional[str] = None,
    device: str = DEFAULT_DEVICE,
    doc_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    reranker_model: str = DEFAULT_RERANK_MODEL,
    disable_rerank: bool = False,
    enable_hybrid: bool = True,
    search_type: str = DEFAULT_SEARCH_TYPE,
    rerank_pool_k: int = DEFAULT_RERANK_POOL_K,
) -> List[Dict[str, Any]]:
    if initial_k <= 0 or final_k <= 0:
        raise ValueError("initial_k ve final_k pozitif olmali.")
    if final_k > initial_k:
        LOGGER.warning("final_k > initial_k oldugu icin final_k, initial_k degerine cekildi.")
        final_k = initial_k

    resolved_device = _resolve_device(device)
    chroma = _get_chromadb()
    client = chroma.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)

    model_name = _resolve_embed_model_for_query(collection, explicit_model=embed_model_name)
    query_vector = _embed_question(question=question, model_name=model_name, device=resolved_device)
    where_filter = _build_where_filter(doc_id=doc_id, chunk_type=chunk_type)

    query_result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=initial_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    ids = query_result.get("ids", [[]])
    documents = query_result.get("documents", [[]])
    metadatas = query_result.get("metadatas", [[]])
    distances = query_result.get("distances", [[]])
    if not ids or not ids[0]:
        return []

    row_ids = ids[0]
    row_docs = documents[0] if documents else ["" for _ in row_ids]
    row_metas = metadatas[0] if metadatas else [{} for _ in row_ids]
    row_distances = distances[0] if distances else [1.0 for _ in row_ids]

    vector_candidates: List[RetrievalCandidate] = []
    for cid, doc_text, meta, dist in zip(row_ids, row_docs, row_metas, row_distances):
        metadata = meta or {}
        vector_distance = float(dist) if dist is not None else 1.0
        vector_score = 1.0 - vector_distance
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
                vector_score=vector_score,
            )
        )

    # Chroma cosine distance: dusuk mesafe daha iyi. Vector sirasi yuksek skora cevrildi.
    vector_candidates = sorted(vector_candidates, key=lambda x: x.vector_distance)
    candidates: List[RetrievalCandidate] = vector_candidates
    hybrid_applied = False
    sparse_candidates: List[RetrievalCandidate] = []
    if search_type in {"hybrid", "keyword"}:
        try:
            sparse_candidates = _get_sparse_candidates(
                collection=collection,
                question=question,
                initial_k=initial_k,
                where_filter=where_filter,
            )
        except Exception as exc:
            LOGGER.warning("Sparse/BM25 asamasi hata verdi: %s", exc)

    if search_type == "vector":
        candidates = vector_candidates
    elif search_type == "keyword":
        candidates = sparse_candidates
    elif search_type == "hybrid":
        if enable_hybrid and sparse_candidates:
            candidates = _rrf_fuse(
                vector_candidates=vector_candidates,
                sparse_candidates=sparse_candidates,
            )
            hybrid_applied = True
        else:
            candidates = vector_candidates
    else:
        raise ValueError("search_type yalnizca 'hybrid', 'vector' veya 'keyword' olabilir.")

    rerank_applied = False
    if not disable_rerank and len(candidates) > 1:
        rerank_pool = max(1, min(rerank_pool_k, len(candidates)))
        rerank_input = candidates[:rerank_pool]
        candidates, rerank_applied = _rerank_candidates(
            question=question,
            candidates=rerank_input,
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


def _rank_hits(
    predicted: Sequence[Dict[str, Any]],
    relevant_chunks: set[str],
    relevant_docs: set[str],
    k: int,
) -> List[int]:
    hits: List[int] = []
    seen_relevant: set[str] = set()
    top = predicted[:k]
    for idx, row in enumerate(top, start=1):
        if relevant_chunks:
            rel_key = str(row.get("chunk_id", ""))
            is_hit = rel_key in relevant_chunks
        else:
            rel_key = str(row.get("doc_id", ""))
            is_hit = rel_key in relevant_docs

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

        hits = _rank_hits(
            predicted=predicted,
            relevant_chunks=relevant_chunks,
            relevant_docs=relevant_docs,
            k=k,
        )
        if relevant_chunks:
            denom = len(relevant_chunks)
        else:
            denom = len(relevant_docs)
        denom = max(denom, 1)
        recall_values.append(len(hits) / denom)
        rr_values.append(1.0 / hits[0] if hits else 0.0)

        dcg = sum(1.0 / math.log2(rank + 1.0) for rank in hits)
        max_rel = min(denom, k)
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, max_rel + 1))
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
    """Eval dosya formati (JSONL):
    {"question":"...", "relevant_doc_ids":["..."]} veya
    {"question":"...", "relevant_chunk_ids":["..."]}.
    """

    rows = _read_jsonl(eval_path)
    if not rows:
        raise ValueError(f"Eval dosyasi bos: {eval_path}")

    labels: List[Dict[str, Any]] = []
    predictions: List[List[Dict[str, Any]]] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        result = retrieve_contexts(
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
        predictions.append(result)
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


def _add_shared_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Chroma persist klasoru",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection adi (varsayilan: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Query embedding model override. Bos birakilirsa collection metadata'dan okunur.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps", "gpu"],
        default=DEFAULT_DEVICE,
        help=f"Sorgu embedding cihazi (varsayilan: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--initial-k",
        type=int,
        default=DEFAULT_INITIAL_K,
        help=f"Ilk asama vektor retrieval aday sayisi (varsayilan: {DEFAULT_INITIAL_K})",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=DEFAULT_FINAL_K,
        help=f"Rerank sonrasi donulecek nihai baglam sayisi (varsayilan: {DEFAULT_FINAL_K})",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Opsiyonel metadata filtresi (doc_id)",
    )
    parser.add_argument(
        "--chunk-type",
        choices=["text", "table"],
        default=None,
        help="Opsiyonel metadata filtresi (chunk_type)",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANK_MODEL,
        help=f"Cross-encoder model (varsayilan: {DEFAULT_RERANK_MODEL})",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Rerank kapatilir, yalnizca vector skoru sirasi kullanilir.",
    )
    parser.add_argument(
        "--disable-hybrid",
        action="store_true",
        help="Dense+sparse hibrit retrieval'i kapatir (yalnizca dense).",
    )
    parser.add_argument(
        "--search-type",
        choices=["hybrid", "vector", "keyword"],
        default=DEFAULT_SEARCH_TYPE,
        help="Arama tipi: hybrid (RRF), vector veya keyword (BM25).",
    )
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        default=DEFAULT_RERANK_POOL_K,
        help=f"Re-ranker'a gidecek aday havuzu (varsayilan: {DEFAULT_RERANK_POOL_K})",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faz 3 retrieval pipeline: Chroma index build + query + evaluate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build-index", help="Faz 2 artefaktlarindan Chroma index kurar.")
    build_cmd.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Faz 2 artefakt klasoru",
    )
    build_cmd.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Chroma persist klasoru",
    )
    build_cmd.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection adi (varsayilan: {DEFAULT_COLLECTION})",
    )
    build_cmd.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Index yazma batch boyutu",
    )

    query_cmd = subparsers.add_parser("query", help="Soru sorarak retrieval sonucu dondurur.")
    query_cmd.add_argument("--question", required=True, help="Aranacak soru")
    _add_shared_query_args(query_cmd)

    eval_cmd = subparsers.add_parser("evaluate", help="JSONL soru seti ile Recall@k/MRR/nDCG raporu.")
    eval_cmd.add_argument(
        "--eval-file",
        required=True,
        help="JSONL eval dosyasi yolu",
    )
    _add_shared_query_args(eval_cmd)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "build-index":
        summary = build_vector_index(
            artifacts_dir=Path(args.artifacts_dir),
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
        print("Faz 3 index build tamamlandi.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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
        print(
            json.dumps(
                {
                    "question": args.question,
                    "result_count": len(contexts),
                    "contexts": contexts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
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
