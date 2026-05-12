"""Retrieval package."""

from .cli import build_arg_parser, main
from .constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DEVICE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_FINAL_K,
    DEFAULT_INITIAL_K,
    DEFAULT_PERSIST_DIR,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_POOL_K,
    DEFAULT_SEARCH_TYPE,
)
from .eval import evaluate_retrieval
from .indexing import build_vector_index
from .query import retrieve_contexts

__all__ = [
    "build_vector_index",
    "retrieve_contexts",
    "evaluate_retrieval",
    "build_arg_parser",
    "main",
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_COLLECTION",
    "DEFAULT_DEVICE",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_FINAL_K",
    "DEFAULT_INITIAL_K",
    "DEFAULT_PERSIST_DIR",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RERANK_POOL_K",
    "DEFAULT_SEARCH_TYPE",
]
