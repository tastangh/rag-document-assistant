from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
EXPECTED_MURSIT_DIM = 1024


def _get_sentence_transformer_class():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers import edilemedi. Bagimliliklari yeniden kur: "
            "pip install -r requirements.txt"
        ) from exc


def resolve_device(device_arg: str) -> str:
    if device_arg == "gpu":
        device_arg = "cuda"
    try:
        import torch
    except Exception:
        return "cpu"
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def _build_fallback_chain(model_name: str) -> List[str]:
    chain: List[str] = [model_name]

    # Local first: models/embeddings altindaki tum adaylari fallback zincirine ekle.
    local_roots = [Path("models/embeddings"), Path(__file__).resolve().parents[2] / "models" / "embeddings"]
    seen_roots = set()
    for root in local_roots:
        root_key = str(root.resolve()) if root.exists() else str(root)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        if not root.exists():
            continue
        for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            chain.append(str(d))

    offline_mode = str(os.getenv("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not offline_mode:
        chain.extend(
            [
                "newmindai/Mursit-Large-TR-Retrieval",
                "BAAI/bge-m3",
                "intfloat/multilingual-e5-base",
            ]
        )

    dedup: List[str] = []
    for model in chain:
        if model not in dedup:
            dedup.append(model)
    return dedup


def embed_texts(
    texts: Sequence[str],
    model_name: str,
    device_arg: str,
    batch_size: int,
) -> Tuple[np.ndarray, str, str]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), model_name, resolve_device(device_arg)

    device = resolve_device(device_arg)
    SentenceTransformer = _get_sentence_transformer_class()
    last_exc: Exception | None = None
    model = None
    selected_model = model_name

    for candidate in _build_fallback_chain(model_name):
        try:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                try:
                    model = SentenceTransformer(str(candidate_path), device=device, local_files_only=True)
                except TypeError:
                    model = SentenceTransformer(str(candidate_path), device=device)
            else:
                model = SentenceTransformer(candidate, device=device)
            selected_model = candidate
            tokenizer = getattr(model, "tokenizer", None)
            tok_name = tokenizer.__class__.__name__ if tokenizer is not None else "unknown"
            logger.info("Embedding modeli yuklendi | model=%s | device=%s | tokenizer=%s", candidate, device, tok_name)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("Embedding modeli yuklenemedi: %s | hata=%s", candidate, exc)

    if model is None:
        message = str(last_exc) if last_exc else ""
        if str(os.getenv("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                "Embedding modeli yuklenemedi: offline mod aktif ve lokal model bulunamadi/yuklenemedi. "
                f"Son hata: {message or 'bilinmiyor'}"
            ) from last_exc
        network_hints = ("Connection", "ReadTimeout", "NameResolutionError", "Temporary failure", "HTTPSConnectionPool")
        if any(hint in message for hint in network_hints):
            raise RuntimeError(
                "Embedding modeli indirilemedi. HuggingFace ag erisimi yok veya kesintili. "
                "Air-gapped ortam icin modeli onceden local cache/mirror'a alin."
            ) from last_exc
        raise RuntimeError(
            "Embedding modeli yuklenemedi (tum fallbackler denendi). "
            f"Son hata: {message or 'bilinmiyor'}"
        ) from last_exc

    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if selected_model == "newmindai/Mursit-Large-TR-Retrieval" and vectors.ndim == 2 and vectors.shape[1] != EXPECTED_MURSIT_DIM:
        raise RuntimeError(
            f"Mursit embedding boyutu beklenenle uyusmuyor. Beklenen={EXPECTED_MURSIT_DIM}, gelen={vectors.shape[1]}"
        )
    return vectors, selected_model, device
