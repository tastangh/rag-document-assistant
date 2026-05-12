from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Block:
    doc_id: str
    page: int
    section: str
    block_type: str  # heading | text | table
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    page_no: int
    section_title: str
    is_table: bool
    chunk_type: str  # text | table
    text: str
    char_len: int

