from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import CHUNK_ARTIFACTS_DIR, EMBED_MODEL_NAME, RESULTS_DIR

from .artifacts import write_artifacts
from .chunking import build_chunks_for_document, hard_split_oversized_chunks
from .embedding import embed_texts
from .types import ChunkRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


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
        logger.warning(
            "Oversized text chunk tespit edildi (%s adet). Otomatik bolme uygulanacak. limit=%s",
            len(oversized),
            max_text_len,
        )
        all_chunks = hard_split_oversized_chunks(
            chunks=all_chunks,
            max_text_len=max_text_len,
            overlap_chars=overlap_chars,
            min_chunk_size=min_chunk_size,
        )
        oversized_after = [c for c in all_chunks if c.chunk_type == "text" and c.char_len > max_text_len]
        if oversized_after:
            first = oversized_after[0]
            raise RuntimeError(
                f"Text chunk boyutu siniri asildi (otomatik bolme sonrasi): {first.chunk_id} ({first.char_len} > {max_text_len})"
            )

    vectors, used_model, used_device = embed_texts(
        texts=[c.text for c in all_chunks],
        model_name=model_name,
        device_arg=device,
        batch_size=batch_size,
    )
    if vectors.shape[0] != len(all_chunks):
        raise RuntimeError(f"Embedding satir sayisi uyusmuyor: vectors={vectors.shape[0]} chunks={len(all_chunks)}")

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
    parser.add_argument("--input-dir", default=default_input_dir)
    parser.add_argument("--output-dir", default=str(CHUNK_ARTIFACTS_DIR))
    parser.add_argument("--model", default=EMBED_MODEL_NAME, help="Embedding modeli")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--min-chunk-size", type=int, default=120)
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

    summary = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        device="cuda" if args.device == "gpu" else args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

