"""Faz 3: Chroma + cosine retrieval + cross-encoder rerank pipeline.

Bu script, Faz 2 artefaktlarindan (chunks.jsonl + embeddings.npy + manifest.json)
vektor index kurar ve soru bazli retrieval yapar.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import chromadb as _chromadb
except Exception:  # pragma: no cover
    _chromadb = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


DEFAULT_ARTIFACTS_DIR = Path("src/results/chunkEmbeddings")
DEFAULT_PERSIST_DIR = Path("src/results/vectorStore/chroma")
DEFAULT_COLLECTION = "rag_chunks_v1"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


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
    rerank_score: Optional[float] = None


def _get_chromadb() -> Any:
    if _chromadb is None:
        raise RuntimeError("chromadb import edilemedi. requirements.txt bagimliliklarini kurun.")
    return _chromadb


def _get_sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers import edilemedi. requirements.txt bagimliliklarini kurun."
        ) from exc


def _get_cross_encoder_class() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder


def _resolve_device(device_arg: str) -> str:
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

    chroma = _get_chromadb()
    client = chroma.PersistentClient(path=str(persist_dir))
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
            metadatas.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "doc_id": str(row.get("doc_id", "")),
                    "page": int(row.get("page", 0)),
                    "section": str(row.get("section", "ROOT")),
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
    where: Dict[str, Any] = {}
    if doc_id:
        where["doc_id"] = {"$eq": doc_id}
    if chunk_type:
        where["chunk_type"] = {"$eq": chunk_type}
    return where or None


def _embed_question(question: str, model_name: str, device: str) -> np.ndarray:
    if not question.strip():
        raise ValueError("Soru bos olamaz.")
    sentence_transformer_cls = _get_sentence_transformer_class()
    model = sentence_transformer_cls(model_name_or_path=model_name, device=device)
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
        cross_encoder_cls = _get_cross_encoder_class()
        reranker = cross_encoder_cls(reranker_model, device=device)
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
    initial_k: int = 12,
    final_k: int = 4,
    embed_model_name: Optional[str] = None,
    device: str = "auto",
    doc_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    reranker_model: str = DEFAULT_RERANK_MODEL,
    disable_rerank: bool = False,
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

    candidates: List[RetrievalCandidate] = []
    for cid, doc_text, meta, dist in zip(row_ids, row_docs, row_metas, row_distances):
        metadata = meta or {}
        vector_distance = float(dist) if dist is not None else 1.0
        vector_score = 1.0 - vector_distance
        candidates.append(
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
    candidates = sorted(candidates, key=lambda x: x.vector_distance)

    rerank_applied = False
    if not disable_rerank and len(candidates) > 1:
        candidates, rerank_applied = _rerank_candidates(
            question=question,
            candidates=candidates,
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
        )
        predictions.append(result)
        labels.append(row)

    metrics = _compute_metrics(predictions=predictions, labels=labels, k=final_k)
    return {
        "query_count": len(predictions),
        "initial_k": initial_k,
        "final_k": final_k,
        "disable_rerank": disable_rerank,
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
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Sorgu embedding cihazi (varsayilan: auto)",
    )
    parser.add_argument(
        "--initial-k",
        type=int,
        default=12,
        help="Ilk asama vektor retrieval aday sayisi",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=4,
        help="Rerank sonrasi donulecek nihai baglam sayisi",
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
            embed_model_name=args.model,
            doc_id=args.doc_id,
            chunk_type=args.chunk_type,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Bilinmeyen komut: {args.command}")


if __name__ == "__main__":
    main()
