from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _str_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


RESULTS_DIR = _path_env("RAG_RESULTS_DIR", "src/results")
RUNTIME_ROOT = _path_env("RAG_RUNTIME_ROOT", str(RESULTS_DIR / "ui_simple_runtime"))
SESSION_ROOT = _path_env("RAG_SESSION_ROOT", "data/sessions")
CHUNK_ARTIFACTS_DIR = _path_env("RAG_CHUNK_ARTIFACTS_DIR", str(RESULTS_DIR / "chunkEmbeddings"))
VECTOR_PERSIST_DIR = _path_env("RAG_VECTOR_PERSIST_DIR", str(RESULTS_DIR / "vectorStore/chroma"))
EVAL_DIR = _path_env("RAG_EVAL_DIR", str(RESULTS_DIR / "eval"))
OCR_CACHE_DIR = _path_env("RAG_OCR_CACHE_DIR", str(RESULTS_DIR / "cache/paddlex"))
COLLECTION_NAME = _str_env("RAG_COLLECTION_NAME", "rag_chunks_v1")
EMBED_MODEL_NAME = _str_env("RAG_EMBED_MODEL_NAME", "newmindai/Mursit-Large-TR-Retrieval")
RERANK_MODEL_NAME = _str_env("RAG_RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
OLLAMA_URL = _str_env("RAG_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = _str_env("RAG_OLLAMA_MODEL", "qwen3:8b")
GUARDRAIL_MODEL = _str_env("RAG_GUARDRAIL_MODEL", "newmindai/ettin-encoder-150M-TR-HD")
GUARDRAIL_THRESHOLD = _float_env("RAG_GUARDRAIL_THRESHOLD", 0.55)
OCR_USE_GPU = _bool_env("RAG_OCR_USE_GPU", True)
EMBED_DEVICE = _str_env("RAG_EMBED_DEVICE", "cuda")
RETRIEVAL_DEVICE = _str_env("RAG_RETRIEVAL_DEVICE", "cuda")


@dataclass(frozen=True)
class AppConfig:
    results_dir: Path = RESULTS_DIR
    runtime_root: Path = RUNTIME_ROOT
    session_root: Path = SESSION_ROOT
    chunk_artifacts_dir: Path = CHUNK_ARTIFACTS_DIR
    vector_persist_dir: Path = VECTOR_PERSIST_DIR
    eval_dir: Path = EVAL_DIR
    ocr_cache_dir: Path = OCR_CACHE_DIR
    collection_name: str = COLLECTION_NAME
    embed_model_name: str = EMBED_MODEL_NAME
    rerank_model_name: str = RERANK_MODEL_NAME
    ollama_url: str = OLLAMA_URL
    ollama_model: str = OLLAMA_MODEL
    guardrail_model: str = GUARDRAIL_MODEL
    guardrail_threshold: float = GUARDRAIL_THRESHOLD
    ocr_use_gpu: bool = OCR_USE_GPU
    embed_device: str = EMBED_DEVICE
    retrieval_device: str = RETRIEVAL_DEVICE
