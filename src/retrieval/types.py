from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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

