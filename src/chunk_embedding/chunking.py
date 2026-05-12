from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .types import Block, ChunkRecord

PAGE_HEADER_RE = re.compile(r"^##\s+Sayfa\s+(\d+)\s*$", flags=re.IGNORECASE | re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_pages(markdown_text: str) -> List[Tuple[int, str]]:
    matches = list(PAGE_HEADER_RE.finditer(markdown_text))
    if not matches:
        text = markdown_text.strip()
        return [(1, text)] if text else []
    pages: List[Tuple[int, str]] = []
    for idx, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        if content:
            pages.append((page_num, content))
    return pages


def _current_section(heading_stack: Dict[int, str]) -> str:
    parts = [heading_stack[level] for level in (1, 2, 3) if heading_stack.get(level)]
    return " > ".join(parts).strip() if parts else "ROOT"


def parse_blocks(doc_id: str, page_num: int, page_text: str) -> List[Block]:
    lines = page_text.splitlines()
    blocks: List[Block] = []
    heading_stack: Dict[int, str] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue
        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            if level <= 3:
                heading_stack[level] = title
                for deeper in (lvl for lvl in list(heading_stack.keys()) if lvl > level and lvl <= 3):
                    heading_stack.pop(deeper, None)
            blocks.append(Block(doc_id, page_num, _current_section(heading_stack), "heading", title))
            idx += 1
            continue
        if stripped.startswith("|"):
            table_lines = [line]
            j = idx + 1
            while j < len(lines) and lines[j].rstrip().strip().startswith("|"):
                table_lines.append(lines[j].rstrip())
                j += 1
            blocks.append(Block(doc_id, page_num, _current_section(heading_stack), "table", "\n".join(table_lines).strip()))
            idx = j
            continue
        paragraph_lines = [stripped]
        j = idx + 1
        while j < len(lines):
            candidate = lines[j].rstrip()
            cstrip = candidate.strip()
            if not cstrip or cstrip.startswith("|") or HEADING_RE.match(cstrip):
                break
            paragraph_lines.append(cstrip)
            j += 1
        blocks.append(
            Block(
                doc_id,
                page_num,
                _current_section(heading_stack),
                "text",
                " ".join(paragraph_lines).strip(),
            )
        )
        idx = j
    return blocks


def normalize_chunk_text(text: str) -> str:
    cleaned = (text or "").replace("\uFFFD", " ")
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_noisy_text(text: str, min_chunk_size: int) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    alnum_count = sum(ch.isalnum() for ch in stripped)
    non_space_count = sum(not ch.isspace() for ch in stripped)
    if alnum_count == 0:
        return True
    if (alnum_count / max(non_space_count, 1)) < 0.22:
        return True
    symbol_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in stripped)
    if (symbol_count / max(non_space_count, 1)) > 0.45:
        return True
    return len(stripped) < min_chunk_size


def split_page_semantic(page_text: str) -> List[Tuple[str, str]]:
    lines = page_text.splitlines()
    parts: List[Tuple[str, str]] = []
    heading_stack: Dict[int, str] = {}
    current_lines: List[str] = []
    current_section = "ROOT"

    def flush_current() -> None:
        nonlocal current_lines
        content = normalize_chunk_text("\n".join(current_lines))
        if content:
            parts.append((current_section, content))
        current_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        m = HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            flush_current()
            if level <= 3:
                heading_stack[level] = title
                for deeper in (lvl for lvl in list(heading_stack.keys()) if lvl > level and lvl <= 3):
                    heading_stack.pop(deeper, None)
            section_parts = [heading_stack[lvl] for lvl in (1, 2, 3) if heading_stack.get(lvl)]
            current_section = " > ".join(section_parts).strip() if section_parts else "ROOT"
            current_lines.append(line)
            continue
        current_lines.append(line)

    flush_current()
    if parts:
        return parts
    fallback = normalize_chunk_text(page_text)
    return [("ROOT", fallback)] if fallback else []


def split_long_text(
    text: str,
    target_chunk_chars: int,
    overlap_chars: int,
    min_chunk_size: int,
) -> List[str]:
    normalized = normalize_chunk_text(text)
    if not normalized:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()] or [normalized]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= target_chunk_chars:
            current = candidate
            continue
        if current:
            cur_norm = normalize_chunk_text(current)
            if not is_noisy_text(cur_norm, min_chunk_size=min_chunk_size):
                chunks.append(cur_norm)
            tail = cur_norm[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{tail} {para}".strip() if tail else para
            continue
        step = max(target_chunk_chars - overlap_chars, max(min_chunk_size, 64))
        for i in range(0, len(para), step):
            piece = normalize_chunk_text(para[i : i + target_chunk_chars])
            if piece and not is_noisy_text(piece, min_chunk_size=min_chunk_size):
                chunks.append(piece)
        current = ""
    if current:
        cur_norm = normalize_chunk_text(current)
        if not is_noisy_text(cur_norm, min_chunk_size=min_chunk_size):
            chunks.append(cur_norm)
    return chunks


def hard_split_oversized_chunks(
    chunks: Sequence[ChunkRecord],
    max_text_len: int,
    overlap_chars: int,
    min_chunk_size: int,
) -> List[ChunkRecord]:
    counters: Dict[Tuple[str, int], int] = {}
    fixed: List[ChunkRecord] = []
    for row in chunks:
        key = (row.doc_id, row.page_no)
        counters.setdefault(key, 1)

        def _next_chunk_id() -> str:
            idx = counters[key]
            counters[key] += 1
            return f"{row.doc_id}::p{row.page_no}::c{idx}"

        if row.chunk_type != "text" or row.char_len <= max_text_len:
            fixed.append(
                ChunkRecord(
                    _next_chunk_id(),
                    row.doc_id,
                    row.page_no,
                    row.section_title,
                    row.is_table,
                    row.chunk_type,
                    row.text,
                    row.char_len,
                )
            )
            continue

        parts = split_long_text(
            text=row.text,
            target_chunk_chars=max_text_len,
            overlap_chars=min(overlap_chars, int(max_text_len * 0.25)),
            min_chunk_size=min_chunk_size,
        )
        if not parts:
            step = max(max_text_len - max(0, overlap_chars), max(min_chunk_size, 64))
            parts = [row.text[i : i + max_text_len].strip() for i in range(0, len(row.text), step) if row.text[i : i + max_text_len].strip()]
        for part in parts:
            fixed.append(
                ChunkRecord(
                    _next_chunk_id(),
                    row.doc_id,
                    row.page_no,
                    row.section_title,
                    False,
                    "text",
                    part,
                    len(part),
                )
            )
    return fixed


def build_chunks_for_document(
    doc_path: Path,
    target_chunk_chars: int,
    overlap_chars: int,
    min_chunk_size: int,
) -> List[ChunkRecord]:
    text = doc_path.read_text(encoding="utf-8")
    pages = split_pages(text)
    doc_id = doc_path.stem
    chunks: List[ChunkRecord] = []
    counter = 1
    for page_num, page_text in pages:
        semantic_sections = split_page_semantic(page_text)
        blocks = parse_blocks(doc_id=doc_id, page_num=page_num, page_text=page_text)
        section_texts: Dict[str, List[str]] = {}
        for section, section_text in semantic_sections:
            section_texts.setdefault(section, []).append(section_text)
        for block in blocks:
            if block.block_type == "heading":
                continue
            if block.block_type == "table":
                normalized = normalize_chunk_text(block.text)
                if not normalized or is_noisy_text(normalized, min_chunk_size=max(32, min_chunk_size // 2)):
                    continue
                chunks.append(
                    ChunkRecord(
                        f"{doc_id}::p{page_num}::c{counter}",
                        doc_id,
                        page_num,
                        block.section,
                        True,
                        "table",
                        normalized,
                        len(normalized),
                    )
                )
                counter += 1
        for section_title, section_parts in section_texts.items():
            section_text = "\n\n".join(section_parts).strip()
            for part in split_long_text(section_text, target_chunk_chars, overlap_chars, min_chunk_size):
                chunks.append(
                    ChunkRecord(
                        f"{doc_id}::p{page_num}::c{counter}",
                        doc_id,
                        page_num,
                        section_title,
                        False,
                        "text",
                        part,
                        len(part),
                    )
                )
                counter += 1
    return chunks
