from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .constants import DEFAULT_EMBED_MODEL, EXPECTED_MURSIT_DIM
from .io import load_artifacts
from .runtime import get_chromadb

LOGGER = logging.getLogger(__name__)


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
        legacy_metadata = dict(metadata)
        legacy_metadata["hnsw:space"] = "cosine"
        return client.create_collection(name=collection_name, metadata=legacy_metadata)


def _batched(seq: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, len(seq), batch_size):
        yield start, min(start + batch_size, len(seq))


def build_vector_index(
    artifacts_dir: Path,
    persist_dir: Path,
    collection_name: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    chunks, embeddings, manifest = load_artifacts(artifacts_dir=artifacts_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    requested_model = str(manifest.get("model_name", DEFAULT_EMBED_MODEL))
    requested_dim = int(manifest.get("embedding_dim", int(embeddings.shape[1])))
    if requested_model == "newmindai/Mursit-Large-TR-Retrieval" and requested_dim != EXPECTED_MURSIT_DIM:
        raise RuntimeError(
            f"Mursit modeli icin embedding boyutu uyumsuz. Beklenen={EXPECTED_MURSIT_DIM}, gelen={requested_dim}."
        )

    chroma = get_chromadb()
    client = chroma.PersistentClient(path=str(persist_dir))
    old_manifest = persist_dir / f"{collection_name}_index_manifest.json"
    if old_manifest.exists():
        try:
            old = json.loads(old_manifest.read_text(encoding="utf-8"))
            old_model = str(old.get("model_name", ""))
            old_dim = int(old.get("embedding_dim", -1))
            if old_model != requested_model or old_dim != requested_dim:
                LOGGER.warning("Index model/dim degisti. Re-index | old=%s/%s new=%s/%s", old_model, old_dim, requested_model, requested_dim)
        except Exception:
            LOGGER.warning("Eski index manifest okunamadi; temiz re-index devam ediyor.")

    if _delete_collection_if_exists(client=client, collection_name=collection_name):
        LOGGER.info("Mevcut collection silindi: %s", collection_name)

    collection = _create_collection_cosine(
        client=client,
        collection_name=collection_name,
        metadata={
            "phase": "faz3",
            "embedding_model": requested_model,
            "embedding_dim": requested_dim,
            "source_chunk_count": int(len(chunks)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
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
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vector_batch.tolist())

    indexed_count = int(collection.count())
    if indexed_count != len(chunks):
        raise RuntimeError(f"Index kayit sayisi uyusmuyor: collection={indexed_count}, beklenen={len(chunks)}")

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

