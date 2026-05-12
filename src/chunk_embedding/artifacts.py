from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .types import ChunkRecord


def inject_chunk_context(chunk: ChunkRecord) -> str:
    return (
        f"[Dokuman: {chunk.doc_id} | Sayfa: {chunk.page_no} | "
        f"Bolum: {chunk.section_title} | Tablo: {str(chunk.is_table).lower()}]\n\n"
        f"{chunk.text}"
    )


def write_artifacts(
    output_dir: Path,
    chunks: Sequence[ChunkRecord],
    embeddings: np.ndarray,
    model_name: str,
    device: str,
    source_docs: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    embeddings_path = output_dir / "embeddings.npy"
    manifest_path = output_dir / "manifest.json"

    with chunks_path.open("w", encoding="utf-8") as f:
        for row in chunks:
            payload = {
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "page": row.page_no,
                "page_no": row.page_no,
                "section": row.section_title,
                "section_title": row.section_title,
                "is_table": row.is_table,
                "chunk_type": row.chunk_type,
                "text": inject_chunk_context(row),
                "char_len": row.char_len,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    np.save(embeddings_path, embeddings.astype(np.float32, copy=False))
    manifest: Dict[str, Any] = {
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "chunk_count": int(len(chunks)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_docs": list(sorted(set(source_docs))),
        "device": device,
        "dtype": "float32",
        "artifacts": {"chunks_jsonl": chunks_path.name, "embeddings_npy": embeddings_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

