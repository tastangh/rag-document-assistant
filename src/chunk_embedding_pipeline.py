"""Faz 2: Yapısal chunking + embedding artefakt üretimi.

Bu script, Faz 1 OCR markdown çıktılarından RAG'e hazır:
- chunks.jsonl
- embeddings.npy
- manifest.json
dosyalarını üretir.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "langchain-text-splitters import edilemedi. requirements.txt bağımlılıklarını kurun."
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers import edilemedi. requirements.txt bağımlılıklarını kurun."
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


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
    page: int
    section: str
    chunk_type: str  # text | table
    text: str
    char_len: int


def split_pages(markdown_text: str) -> List[Tuple[int, str]]:
    """`## Sayfa N` etiketlerine göre sayfaları ayırır."""
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
    """Sayfa metnini heading / paragraph / table bloklarına ayırır."""
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
            table_lines = [lines[idx].rstrip()]
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
            if not cstrip:
                break
            if cstrip.startswith("|"):
                break
            if HEADING_RE.match(cstrip):
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


def is_noisy_text(text: str, min_chunk_size: int) -> bool:
    """Aşırı kısa veya gürültülü chunk'ları eler."""
    stripped = text.strip()
    if not stripped:
        return True

    alnum_count = sum(ch.isalnum() for ch in stripped)
    non_space_count = sum(not ch.isspace() for ch in stripped)
    if alnum_count == 0:
        return True

    alnum_ratio = alnum_count / max(non_space_count, 1)
    if alnum_ratio < 0.22:
        return True

    # Çok kısa metinleri ele (başlık kırıntıları, OCR gürültüsü vb.)
    if len(stripped) < min_chunk_size:
        return True

    return False


def normalize_chunk_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean


def build_chunks_for_document(
    doc_path: Path,
    splitter: RecursiveCharacterTextSplitter,
    min_chunk_size: int,
) -> List[ChunkRecord]:
    text = doc_path.read_text(encoding="utf-8")
    pages = split_pages(text)
    doc_id = doc_path.stem

    chunks: List[ChunkRecord] = []
    counter = 1

    for page_num, page_text in pages:
        blocks = parse_blocks(doc_id=doc_id, page_num=page_num, page_text=page_text)

        for block in blocks:
            if block.block_type == "heading":
                continue

            if block.block_type == "table":
                table_text = block.text.strip()
                if is_noisy_text(table_text, min_chunk_size=max(32, min_chunk_size // 2)):
                    continue
                normalized = normalize_chunk_text(table_text)
                if not normalized:
                    continue
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}::p{page_num}::c{counter}",
                        doc_id=doc_id,
                        page=page_num,
                        section=block.section,
                        chunk_type="table",
                        text=normalized,
                        char_len=len(normalized),
                    )
                )
                counter += 1
                continue

            # block_type == "text"
            raw_parts = splitter.split_text(block.text)
            for part in raw_parts:
                normalized = normalize_chunk_text(part)
                if is_noisy_text(normalized, min_chunk_size=min_chunk_size):
                    continue

                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}::p{page_num}::c{counter}",
                        doc_id=doc_id,
                        page=page_num,
                        section=block.section,
                        chunk_type="text",
                        text=normalized,
                        char_len=len(normalized),
                    )
                )
                counter += 1

    return chunks


def _resolve_device(device_arg: str) -> str:
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
    """Metinleri embed eder. Dönüş: (embeddings, used_model_name, used_device)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), model_name, _resolve_device(device_arg)

    device = _resolve_device(device_arg)
    selected_model = model_name

    try:
        model = SentenceTransformer(selected_model, device=device)
    except Exception as exc:
        # Varsayılan modelde sorun varsa belirtilen fallback ile dene.
        if selected_model == "BAAI/bge-m3":
            fallback = "intfloat/multilingual-e5-base"
            LOGGER.warning(
                "Model yüklenemedi (%s). Fallback modele geçiliyor: %s",
                exc,
                fallback,
            )
            selected_model = fallback
            model = SentenceTransformer(selected_model, device=device)
        else:
            raise

    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)
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
            payload = {
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "page": row.page,
                "section": row.section,
                "chunk_type": row.chunk_type,
                "text": row.text,
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
        raise FileNotFoundError(f"Girdi klasöründe .md dosyası bulunamadı: {input_dir}")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: List[ChunkRecord] = []
    per_doc_counts: Dict[str, int] = {}

    for doc_path in md_files:
        doc_chunks = build_chunks_for_document(
            doc_path=doc_path,
            splitter=splitter,
            min_chunk_size=min_chunk_size,
        )
        if not doc_chunks:
            LOGGER.warning("Belgeden geçerli chunk üretilemedi: %s", doc_path.name)
            continue

        all_chunks.extend(doc_chunks)
        per_doc_counts[doc_path.name] = len(doc_chunks)
        LOGGER.info("Chunk üretildi: %s | %s chunk", doc_path.name, len(doc_chunks))

    if not all_chunks:
        raise RuntimeError("Hiçbir dokümandan chunk üretilemedi.")

    # Text chunk üst sınır doğrulaması (table chunk hariç).
    max_text_len = int(chunk_size * 1.1)
    oversized = [c for c in all_chunks if c.chunk_type == "text" and c.char_len > max_text_len]
    if oversized:
        first = oversized[0]
        raise RuntimeError(
            f"Text chunk boyutu sınırı aşıldı: {first.chunk_id} ({first.char_len} > {max_text_len})"
        )

    texts = [c.text for c in all_chunks]
    embeddings, used_model_name, used_device = embed_texts(
        texts=texts,
        model_name=model_name,
        device_arg=device,
        batch_size=batch_size,
    )

    if embeddings.shape[0] != len(all_chunks):
        raise RuntimeError(
            f"Embedding satır sayısı uyuşmuyor: {embeddings.shape[0]} != {len(all_chunks)}"
        )

    write_artifacts(
        output_dir=output_dir,
        chunks=all_chunks,
        embeddings=embeddings,
        model_name=used_model_name,
        device=used_device,
        source_docs=[p.name for p in md_files],
    )

    return {
        "doc_count": len(md_files),
        "chunk_count": len(all_chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "model_name": used_model_name,
        "device": used_device,
        "per_doc_counts": per_doc_counts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR markdown çıktılarından yapısal chunk + embedding artefaktları üretir."
    )
    parser.add_argument(
        "--input-dir",
        default="src/results/ocrMdResults",
        help="Girdi markdown klasörü (varsayılan: src/results/ocrMdResults)",
    )
    parser.add_argument(
        "--output-dir",
        default="src/results/chunkEmbeddings",
        help="Çıktı artefakt klasörü (varsayılan: src/results/chunkEmbeddings)",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="SentenceTransformer model adı (varsayılan: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Text chunk maksimum karakter uzunluğu",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Text chunk overlap uzunluğu",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=120,
        help="Text chunk minimum karakter uzunluğu",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Embedding cihazı (varsayılan: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch boyutu",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap, chunk_size'dan küçük olmalıdır.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Girdi klasörü bulunamadı: {input_dir}")

    summary = run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("Faz 2 tamamlandı.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
