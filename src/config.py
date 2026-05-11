from __future__ import annotations

import os
from pathlib import Path


def _path_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _str_env(name: str, default: str) -> str:
    return os.getenv(name, default)


RESULTS_DIR = _path_env("RAG_RESULTS_DIR", "src/results")
RUNTIME_ROOT = _path_env("RAG_RUNTIME_ROOT", str(RESULTS_DIR / "ui_simple_runtime"))
CHUNK_ARTIFACTS_DIR = _path_env("RAG_CHUNK_ARTIFACTS_DIR", str(RESULTS_DIR / "chunkEmbeddings"))
VECTOR_PERSIST_DIR = _path_env("RAG_VECTOR_PERSIST_DIR", str(RESULTS_DIR / "vectorStore/chroma"))
COLLECTION_NAME = _str_env("RAG_COLLECTION_NAME", "rag_chunks_v1")
EMBED_MODEL_NAME = _str_env("RAG_EMBED_MODEL_NAME", "newmindai/Mursit-Large-TR-Retrieval")
RERANK_MODEL_NAME = _str_env("RAG_RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
OLLAMA_URL = _str_env("RAG_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = _str_env("RAG_OLLAMA_MODEL", "qwen3:8b")
GUARDRAIL_MODEL = _str_env("RAG_GUARDRAIL_MODEL", "newmindai/ettin-encoder-150M-TR-HD")
GUARDRAIL_THRESHOLD = float(os.getenv("RAG_GUARDRAIL_THRESHOLD", "0.55"))
OCR_USE_GPU = os.getenv("RAG_OCR_USE_GPU", "1").strip() in {"1", "true", "True", "YES", "yes"}
EMBED_DEVICE = _str_env("RAG_EMBED_DEVICE", "cuda")
RETRIEVAL_DEVICE = _str_env("RAG_RETRIEVAL_DEVICE", "cuda")
