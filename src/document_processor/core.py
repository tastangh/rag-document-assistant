"""Core document processing orchestration.

This module intentionally keeps only high-level workflow.
Heavy OCR/table/io utilities live in sibling modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .document_io import (
    extract_text_layer_from_pdf,
    extract_with_opendataloader_pdf,
    load_image,
    pdf_to_images,
    split_double_page_image,
)
from .markdown_tables import normalize_markdown_tables, preprocess_image_for_ocr
from .ocr_backends import (
    create_fallback_ocr_engine,
    extract_markdown_from_v3_result,
    extract_markdown_with_structure,
    extract_text_from_legacy_block,
    init_engines,
    legacy_result_to_markdown,
    ocr_to_text,
    process_single_page,
    run_fallback_ocr,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import logging as _py_logging

    _py_logging.getLogger("transformers").setLevel(_py_logging.ERROR)
except Exception:
    pass


class DocumentProcessor:
    """Convert PDF/JPG/PNG documents to Markdown."""

    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}

    def __init__(
        self,
        use_gpu: bool = False,
        pdf_zoom: float = 2.0,
        ocr_lang: str = "tr",
        ocr_profile: str = "default",
        ocr_backend: str = "paddle",
    ) -> None:
        # Paddle OCR is pinned to CPU for stability on Windows.
        self.use_gpu = False
        self.pdf_zoom = float(pdf_zoom)
        self.ocr_lang = (ocr_lang or "tr").strip().lower()
        self.ocr_profile = (ocr_profile or "default").strip().lower()
        self.ocr_backend = "paddle"

        self.structure_v3 = None
        self.legacy_structure = None
        self.ocr_tr = None
        self.ocr_en = None
        self._engines_ready = False

    def _init_engines(self) -> None:
        if self._engines_ready:
            return
        init_engines(self)
        self._engines_ready = True

    def _create_fallback_ocr_engine(self, lang: str) -> Any:
        return create_fallback_ocr_engine(self, lang)

    def _validate_input(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Desteklenmeyen dosya türü: {ext}. "
                f"Desteklenenler: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    def process_document(self, file_path: str | Path) -> str:
        path = Path(file_path)
        self._validate_input(path)

        try:
            if path.suffix.lower() == ".pdf" and self.ocr_backend in {"opendataloader_pdf", "auto"}:
                od_md = extract_with_opendataloader_pdf(path)
                if od_md.strip():
                    logger.info("OpenDataLoader PDF markdown extraction kullanildi: %s", path.name)
                    return od_md

            if path.suffix.lower() == ".pdf":
                direct_pdf_md = extract_text_layer_from_pdf(path)
                if direct_pdf_md.strip():
                    return direct_pdf_md
                self._init_engines()
                page_images = pdf_to_images(path, self.pdf_zoom)
            else:
                self._init_engines()
                page_images = [load_image(path)]
        except Exception as exc:
            raise RuntimeError(f"Belge yüklenemedi: {path}") from exc

        expanded_page_images: List[np.ndarray] = []
        for image in page_images:
            expanded_page_images.extend(split_double_page_image(image))
        page_images = expanded_page_images

        page_markdowns: List[str] = []
        for page_idx, page_image in enumerate(page_images, start=1):
            logger.info("Sayfa isleme basladi | page=%s", page_idx)
            try:
                page_md = process_single_page(self, page_image, preprocess_image_for_ocr)
                if not page_md.strip():
                    page_md = "[Bu sayfadan metin çıkarılamadı.]"
                    logger.warning("Sayfada anlamli OCR cikisi yok | page=%s", page_idx)
                else:
                    logger.info("Sayfa basariyla islendi | page=%s", page_idx)
                page_markdowns.append(f"## Sayfa {page_idx}\n\n{page_md.strip()}")
            except Exception as exc:
                logger.exception("Sayfa %s işlenirken hata: %s", page_idx, exc)
                page_markdowns.append(f"## Sayfa {page_idx}\n\n[Hata: Bu sayfa işlenemedi: {exc}]")

        return "\n\n".join(page_markdowns).strip()

    # Compatibility wrappers kept for callers/tests that may still use method forms.
    def _extract_with_opendataloader_pdf(self, pdf_path: Path) -> str:
        return extract_with_opendataloader_pdf(pdf_path)

    def _extract_text_layer_from_pdf(self, pdf_path: Path) -> str:
        return extract_text_layer_from_pdf(pdf_path)

    def _process_single_page(self, image_bgr: np.ndarray) -> str:
        return process_single_page(self, image_bgr, preprocess_image_for_ocr)

    def _extract_markdown_with_structure(self, image_bgr: np.ndarray) -> str:
        return extract_markdown_with_structure(self, image_bgr)

    @staticmethod
    def _extract_markdown_from_v3_result(result_obj: Any) -> str:
        return extract_markdown_from_v3_result(result_obj)

    @staticmethod
    def _legacy_result_to_markdown(result: List[Dict[str, Any]]) -> str:
        return legacy_result_to_markdown(result)

    @staticmethod
    def _extract_text_from_legacy_block(block_res: Any) -> str:
        return extract_text_from_legacy_block(block_res)

    def _run_fallback_ocr(self, image_bgr: np.ndarray) -> str:
        return run_fallback_ocr(self, image_bgr)

    @staticmethod
    def _ocr_to_text(ocr_engine: Any, image_bgr: np.ndarray) -> str:
        return ocr_to_text(ocr_engine, image_bgr)

    @staticmethod
    def _normalize_markdown_tables(markdown_text: str) -> str:
        return normalize_markdown_tables(markdown_text)


def process_document_to_markdown(
    file_path: str | Path,
    use_gpu: bool = False,
    ocr_lang: str = "tr",
    ocr_profile: str = "default",
    ocr_backend: str = "paddle",
) -> str:
    processor = DocumentProcessor(
        use_gpu=use_gpu,
        ocr_lang=ocr_lang,
        ocr_profile=ocr_profile,
        ocr_backend=ocr_backend,
    )
    return processor.process_document(file_path)
