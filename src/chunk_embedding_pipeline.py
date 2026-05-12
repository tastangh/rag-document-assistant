"""Faz 2: Yapısal/semantik chunking + embedding artefakt üretimi.

Çıktılar:
- chunks.jsonl
- embeddings.npy
- manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from config import CHUNK_ARTIFACTS_DIR, EMBED_MODEL_NAME, RESULTS_DIR
from langchain_text_splitters import MarkdownHeaderTextSplitter

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers import edilemedi. Poetry bagimliliklarini kurun: `poetry install`."
    ) from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
EXPECTED_MURSIT_DIM = 1024

PAGE_HEADER_RE = re.compile(r"^##\s+Sayfa\s+(\d+)\s*$", flags=re.IGNORECASE | re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
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
    chunk_type: str  # text | table (geri uyumluluk)
    text: str
    char_len: int


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
            blocks.append(
                Block(
                    doc_id=doc_id,
                    page=page_num,
                    section=_current_section(heading_stack),
                    block_type="heading",
                    text=title,
                )
            )
            idx += 1
            continue

        if stripped.startswith("|"):
            table_lines = [line]
            j = idx + 1
            while j < len(lines):
                candidate = lines[j].rstrip()
                if candidate.strip().startswith("|"):
                    table_lines.append(candidate)
                    j += 1
                    continue
                break
            blocks.append(
                Block(
                    doc_id=doc_id,
                    page=page_num,
                    section=_current_section(heading_stack),
                    block_type="table",
                    text="\n".join(table_lines).strip(),
                )
            )
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

        paragraph_text = " ".join(paragraph_lines).strip()
        blocks.append(
            Block(
                doc_id=doc_id,
                page=page_num,
                section=_current_section(heading_stack),
                block_type="text",
                text=paragraph_text,
            )
        )
        idx = j

    return blocks


def normalize_chunk_text(text: str) -> str:
    cleaned = (text or "")
    # OCR kaynakli kontrol/yerine gecen bozuk karakterleri temizle.
    cleaned = cleaned.replace("\uFFFD", " ")
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
    # Asiri sembol/karma karakter oranli OCR coplerini ele.
    symbol_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in stripped)
    if (symbol_count / max(non_space_count, 1)) > 0.45:
        return True
    if len(stripped) < min_chunk_size:
        return True
    return False


def semantic_split_text_block(text: str, min_chunk_size: int) -> List[str]:
    normalized = normalize_chunk_text(text)
    if not normalized:
        return []
    if is_noisy_text(normalized, min_chunk_size=min_chunk_size):
        return []
    return [normalized]


def _split_long_text_with_dynamic_overlap(
    text: str,
    target_chunk_chars: int,
    overlap_chars: int,
    min_chunk_size: int,
) -> List[str]:
    normalized = normalize_chunk_text(text)
    if not normalized:
        return []

    # Paragraf sinirlari varsa once onlari koru.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [normalized]

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
            # Dinamik overlap: onceki chunk'in son overlap_chars karakterini yeni chunk'a tasiyoruz.
            tail = cur_norm[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{tail} {para}".strip() if tail else para
        else:
            # Tek paragraf cok uzunsa sert bolme.
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


def split_page_semantic(page_text: str) -> List[Tuple[str, str]]:
    """Sayfa metnini Markdown baslik hiyerarsisine gore boler."""
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    docs = splitter.split_text(page_text)
    parts: List[Tuple[str, str]] = []
    for d in docs:
        content = normalize_chunk_text(d.page_content)
        if not content:
            continue
        md = d.metadata or {}
        section = str(md.get("h3") or md.get("h2") or md.get("h1") or "ROOT").strip() or "ROOT"
        parts.append((section, content))
    if not parts:
        fallback = normalize_chunk_text(page_text)
        return [("ROOT", fallback)] if fallback else []
    return parts


def _inject_chunk_context(chunk: ChunkRecord) -> str:
    return (
        f"[Dokuman: {chunk.doc_id} | Sayfa: {chunk.page_no} | "
        f"Bolum: {chunk.section_title} | Tablo: {str(chunk.is_table).lower()}]\n\n"
        f"{chunk.text}"
    )


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
        for s, t in semantic_sections:
            section_texts.setdefault(s, []).append(t)

        for block in blocks:
            if block.block_type == "heading":
                continue

            if block.block_type == "table":
                normalized = normalize_chunk_text(block.text)
                if not normalized or is_noisy_text(normalized, min_chunk_size=max(32, min_chunk_size // 2)):
                    continue
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}::p{page_num}::c{counter}",
                        doc_id=doc_id,
                        page_no=page_num,
                        section_title=block.section,
                        is_table=True,
                        chunk_type="table",
                        text=normalized,
                        char_len=len(normalized),
                    )
                )
                counter += 1
                continue

        # Text chunking: baslik/bolum bazli semantik birlestirme + dinamik overlap.
        for section_title, section_parts in section_texts.items():
            section_text = "\n\n".join(section_parts).strip()
            dynamic_parts = _split_long_text_with_dynamic_overlap(
                text=section_text,
                target_chunk_chars=target_chunk_chars,
                overlap_chars=overlap_chars,
                min_chunk_size=min_chunk_size,
            )
            for part in dynamic_parts:
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}::p{page_num}::c{counter}",
                        doc_id=doc_id,
                        page_no=page_num,
                        section_title=section_title,
                        is_table=False,
                        chunk_type="text",
                        text=part,
                        char_len=len(part),
                    )
                )
                counter += 1

    return chunks


def _resolve_device(device_arg: str) -> str:
    if device_arg == "gpu":
        device_arg = "cuda"
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embed_texts(
    texts: Sequence[str],
    model_name: str,
    device_arg: str,
    batch_size: int,
) -> Tuple[np.ndarray, str, str]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), model_name, _resolve_device(device_arg)

    device = _resolve_device(device_arg)
    fallback_chain = [
        model_name,
        "newmindai/Mursit-Large-TR-Retrieval",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-base",
    ]
    # tekrar edenleri temizle, sıra korunsun
    dedup_chain: List[str] = []
    for m in fallback_chain:
        if m not in dedup_chain:
            dedup_chain.append(m)

    last_exc: Exception | None = None
    model = None
    selected_model = dedup_chain[0]
    for candidate in dedup_chain:
        try:
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
        msg = str(last_exc) if last_exc else ""
        network_hints = ("Connection", "ReadTimeout", "NameResolutionError", "Temporary failure", "HTTPSConnectionPool")
        if any(h in msg for h in network_hints):
            raise RuntimeError(
                "Embedding modeli indirilemedi. HuggingFace ag erisimi yok veya kesintili. "
                "Air-gapped ortam icin modeli onceden local cache/mirror'a alin."
            ) from last_exc
        raise RuntimeError("Embedding modeli yuklenemedi (tum fallbackler denendi).") from last_exc

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
            contextual_text = _inject_chunk_context(row)
            payload = {
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "page": row.page_no,
                "page_no": row.page_no,
                "section": row.section_title,
                "section_title": row.section_title,
                "is_table": row.is_table,
                "chunk_type": row.chunk_type,
                "text": contextual_text,
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
        "artifacts": {
            "chunks_jsonl": chunks_path.name,
            "embeddings_npy": embeddings_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    md_files = sorted([p for p in input_dir.glob("*.md") if p.is_file()], key=lambda p: p.name.lower())
    if not md_files:
        raise FileNotFoundError(f"Girdi klasorunde .md dosyasi bulunamadi: {input_dir}")

    # Faz 2: boyut/overlap artik semantik bolutlemede dinamik olarak kullaniliyor.
    target_chunk_chars = max(chunk_size, min_chunk_size + 64)
    overlap_chars = min(max(chunk_overlap, 0), int(target_chunk_chars * 0.35))

    all_chunks: List[ChunkRecord] = []
    per_doc_counts: Dict[str, int] = {}

    for doc_path in md_files:
        doc_chunks = build_chunks_for_document(
            doc_path=doc_path,
            target_chunk_chars=target_chunk_chars,
            overlap_chars=overlap_chars,
            min_chunk_size=min_chunk_size,
        )
        if not doc_chunks:
            logger.warning("Belgeden gecerli chunk uretilemedi: %s", doc_path.name)
            continue

        all_chunks.extend(doc_chunks)
        per_doc_counts[doc_path.name] = len(doc_chunks)
        logger.info("Chunk uretildi: %s | %s chunk", doc_path.name, len(doc_chunks))

    if not all_chunks:
        raise RuntimeError("Hicbir dokumandan chunk uretilemedi.")

    max_text_len = int(chunk_size * 1.1)
    oversized = [c for c in all_chunks if c.chunk_type == "text" and c.char_len > max_text_len]
    if oversized:
        first = oversized[0]
        raise RuntimeError(
            f"Text chunk boyutu siniri asildi: {first.chunk_id} ({first.char_len} > {max_text_len})"
        )

    texts = [c.text for c in all_chunks]
    vectors, used_model, used_device = embed_texts(
        texts=texts,
        model_name=model_name,
        device_arg=device,
        batch_size=batch_size,
    )

    if vectors.shape[0] != len(all_chunks):
        raise RuntimeError(
            f"Embedding satir sayisi uyusmuyor: vectors={vectors.shape[0]} chunks={len(all_chunks)}"
        )

    write_artifacts(
        output_dir=output_dir,
        chunks=all_chunks,
        embeddings=vectors,
        model_name=used_model,
        device=used_device,
        source_docs=[p.name for p in md_files],
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "doc_count": len(md_files),
        "chunk_count": len(all_chunks),
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0,
        "model_name": used_model,
        "device": used_device,
        "per_doc_chunk_counts": per_doc_counts,
    }

    logger.info(
        "Faz 2 tamamlandi | doc=%s | chunk=%s | dim=%s | model=%s | device=%s",
        summary["doc_count"],
        summary["chunk_count"],
        summary["embedding_dim"],
        summary["model_name"],
        summary["device"],
    )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Faz 2 chunk + embedding pipeline")
    default_input_dir = str(RESULTS_DIR / "ocrMdResults")

    parser.add_argument(
        "--input-dir",
        default=default_input_dir,
        help=f"Girdi markdown klasoru (varsayilan: {default_input_dir})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CHUNK_ARTIFACTS_DIR),
        help=f"Cikti artefakt klasoru (varsayilan: {CHUNK_ARTIFACTS_DIR})",
    )
    parser.add_argument("--model", default=EMBED_MODEL_NAME, help="Embedding modeli")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Maksimum text chunk uzunlugu")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Geri uyumluluk parametresi (semantik chunking'de aktif kullanilmaz)",
    )
    parser.add_argument("--min-chunk-size", type=int, default=120, help="Minimum chunk uzunlugu")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "gpu"])
    parser.add_argument("--batch-size", type=int, default=32)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.chunk_size < 100:
        raise ValueError("chunk_size en az 100 olmali")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap, chunk_size'dan kucuk olmali")

    resolved_device = "cuda" if args.device == "gpu" else args.device
    summary = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        device=resolved_device,
        batch_size=args.batch_size,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
