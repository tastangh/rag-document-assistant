from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def load_artifacts(artifacts_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json bulunamadi: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    chunks_path = artifacts_dir / artifacts.get("chunks_jsonl", "chunks.jsonl")
    embeddings_path = artifacts_dir / artifacts.get("embeddings_npy", "embeddings.npy")
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks dosyasi bulunamadi: {chunks_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"embeddings dosyasi bulunamadi: {embeddings_path}")

    chunks = read_jsonl(chunks_path)
    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings 2 boyutlu olmali. Gelen sekil: {embeddings.shape}")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"Satir sayisi uyusmuyor: chunks={len(chunks)} embeddings={embeddings.shape[0]}")
    if int(manifest.get("chunk_count", len(chunks))) != len(chunks):
        raise ValueError("Manifest chunk_count uyusmuyor.")
    if int(manifest.get("embedding_dim", embeddings.shape[1])) != embeddings.shape[1]:
        raise ValueError("Manifest embedding_dim uyusmuyor.")
    return chunks, embeddings, manifest

