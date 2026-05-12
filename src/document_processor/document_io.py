"""Document I/O and CLI helpers for document processing."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

import cv2
import fitz
import numpy as np

from config import OCR_CACHE_DIR

logger = logging.getLogger(__name__)


def extract_with_opendataloader_pdf(pdf_path: Path) -> str:
    try:
        import opendataloader_pdf  # type: ignore
    except Exception as exc:
        logger.warning("opendataloader_pdf import edilemedi, paddle fallback: %s", exc)
        return ""

    out_dir = OCR_CACHE_DIR / "opendataloader_tmp" / f"{pdf_path.stem}_{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        opendataloader_pdf.convert(input_path=[str(pdf_path)], output_dir=str(out_dir), format="markdown")
        md_files = sorted(out_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not md_files:
            return ""
        return md_files[0].read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("opendataloader_pdf convert basarisiz, paddle fallback: %s", exc)
        return ""


def extract_text_layer_from_pdf(pdf_path: Path) -> str:
    page_markdowns: List[str] = []
    useful_pages = 0
    min_chars_per_page = 80

    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            return ""
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = (page.get_text("text") or "").replace("\x00", "").strip()
            if len(text) >= min_chars_per_page:
                useful_pages += 1
            if text:
                page_markdowns.append(f"## Sayfa {page_idx + 1}\\n\\n{text}")
            else:
                page_markdowns.append(f"## Sayfa {page_idx + 1}\\n\\n[Bu sayfadan metin çıkarılamadı.]")

    coverage = useful_pages / max(len(page_markdowns), 1)
    if coverage >= 0.4:
        logger.info("PDF text-layer kullanıldı: %s (coverage=%.2f)", pdf_path.name, coverage)
        return "\\n\\n".join(page_markdowns).strip()
    return ""


def pdf_to_images(pdf_path: Path, pdf_zoom: float) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise ValueError("PDF sayfa içermiyor.")

        matrix = fitz.Matrix(pdf_zoom, pdf_zoom)
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            images.append(img_bgr)
    return images


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Resim okunamadı: {image_path}")
    return image


def split_double_page_image(image_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return [image_bgr]
    if (w / float(h)) < 1.35:
        return [image_bgr]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vertical_ink = binary_inv.sum(axis=0).astype(np.float64)
    if vertical_ink.size < 20:
        return [image_bgr]

    center = w // 2
    half_band = max(int(w * 0.15), 20)
    l_bound = max(center - half_band, 0)
    r_bound = min(center + half_band, w)
    center_band = vertical_ink[l_bound:r_bound]
    if center_band.size == 0:
        return [image_bgr]

    gutter_idx = l_bound + int(np.argmin(center_band))
    gutter_ink = float(vertical_ink[gutter_idx])
    mean_ink = float(np.mean(vertical_ink)) + 1e-6
    if gutter_ink > (0.45 * mean_ink):
        return [image_bgr]

    left, right = image_bgr[:, :gutter_idx], image_bgr[:, gutter_idx:]
    if left.size == 0 or right.size == 0:
        return [image_bgr]

    left_ink = float(binary_inv[:, :gutter_idx].sum())
    right_ink = float(binary_inv[:, gutter_idx:].sum())
    total_ink = left_ink + right_ink + 1e-6
    if (left_ink / total_ink) < 0.2 or (right_ink / total_ink) < 0.2:
        return [image_bgr]
    return [left, right]


def process_directory_to_markdown(
    input_dir: str | Path,
    output_dir: str | Path,
    use_gpu: bool = False,
    recursive: bool = True,
) -> List[Path]:
    from .core import DocumentProcessor

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Girdi klasörü bulunamadı: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    processor = DocumentProcessor(use_gpu=use_gpu)

    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in in_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in processor.SUPPORTED_EXTENSIONS
    ]
    files = sorted(files, key=lambda p: p.name.lower())

    if not files:
        logger.warning("İşlenecek destekli dosya bulunamadı: %s", in_dir)
        return []

    written: List[Path] = []
    for src in files:
        try:
            markdown = processor.process_document(src)
            out_path = out_dir / f"{src.stem}.md"
            out_path.write_text(markdown, encoding="utf-8")
            written.append(out_path)
            logger.info("Tamamlandı: %s -> %s", src.name, out_path.name)
        except Exception as exc:
            logger.exception("Dosya işlenemedi: %s | Hata: %s", src, exc)

    return written


def run_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF/JPG/PNG belgelerini OCR + layout analizi ile Markdown'a dönüştürür."
    )
    parser.add_argument("input_file", nargs="?", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", default="output.md")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    try:
        if args.input_dir:
            default_output_dir = Path(__file__).resolve().parent.parent / "results" / "ocrMdResults"
            output_dir = args.output_dir or str(default_output_dir)
            written = process_directory_to_markdown(
                input_dir=args.input_dir,
                output_dir=output_dir,
                use_gpu=args.gpu,
                recursive=not args.no_recursive,
            )
            print(f"Toplam çıktı: {len(written)} dosya")
            for path in written:
                print(path.resolve())
            return 0

        if not args.input_file:
            raise ValueError("Tek dosya modu için `input_file` veya klasör modu için `--input-dir` verin.")

        from .core import process_document_to_markdown

        markdown_output = process_document_to_markdown(args.input_file, use_gpu=args.gpu)
        output_path = Path(args.output)
        output_path.write_text(markdown_output, encoding="utf-8")

        print(f"Markdown başarıyla üretildi: {output_path.resolve()}")
        print("--- Önizleme (ilk 1200 karakter) ---")
        print(markdown_output[:1200])
        return 0
    except Exception as err:
        print(f"Hata: {err}")
        return 1
