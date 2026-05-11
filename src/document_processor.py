"""Faz 1 belge işleme: PDF/JPG/PNG -> yapılandırılmış Markdown.

Bu modül, PP-StructureV3 (ve geriye dönük PPStructure fallback) kullanarak
metin ve tablo içeriklerini çıkarır, tabloları Markdown grid formatında korur,
ve tüm sayfaları tek bir Markdown metninde birleştirir.
"""

from __future__ import annotations

import logging
import os
import re
import inspect
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from bs4 import BeautifulSoup

PaddleOCR = None
PPStructureV3 = None
PPStructure = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Reduce transformers warning spam triggered by PaddleX/PaddleOCR imports.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import logging as _py_logging

    _py_logging.getLogger("transformers").setLevel(_py_logging.ERROR)
except Exception:
    pass


class DocumentProcessor:
    """PDF ve resim belgelerini RAG için Markdown'a dönüştürür."""

    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}

    def __init__(self, use_gpu: bool = False, pdf_zoom: float = 2.0) -> None:
        """OCR ve layout pipeline'ını başlat.

        Args:
            use_gpu: GPU üzerinde çalıştırmak için True.
            pdf_zoom: PDF rasterization kalitesi (1.0-3.0 tipik aralık).
        """
        self.use_gpu = use_gpu
        self.pdf_zoom = pdf_zoom

        self.structure_v3 = None
        self.legacy_structure = None

        self.ocr_tr = None
        self.ocr_en = None

        self._init_engines()

    def _init_engines(self) -> None:
        """PP-StructureV3 ve fallback OCR motorlarını başlat."""
        device = "gpu" if self.use_gpu else "cpu"
        project_root = Path(__file__).resolve().parent
        local_paddlex_cache = project_root / ".paddlex_cache"
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(local_paddlex_cache))
        local_paddlex_cache.mkdir(parents=True, exist_ok=True)
        try:
            numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
        except Exception:
            numpy_major = 0

        if numpy_major >= 2:
            raise RuntimeError(
                "NumPy 2.x ile PaddleOCR/PaddleX ikili bağımlılıklarında uyumsuzluk var. "
                "Lütfen `numpy<2` kullanın ve bağımlılıkları yeniden kurun."
            )

        global PaddleOCR, PPStructureV3, PPStructure
        if PaddleOCR is None:
            try:
                from paddleocr import PaddleOCR as _PaddleOCR

                PaddleOCR = _PaddleOCR
            except Exception as exc:
                raise RuntimeError(
                    "paddleocr import edilemedi. Ortamı temiz bir venv içinde kurup tekrar deneyin."
                ) from exc

            try:
                from paddleocr import PPStructureV3 as _PPStructureV3

                PPStructureV3 = _PPStructureV3
            except ImportError:
                PPStructureV3 = None

            try:
                from paddleocr import PPStructure as _PPStructure

                PPStructure = _PPStructure
            except ImportError:
                PPStructure = None

        structure_errors: List[str] = []

        if PPStructureV3 is not None:
            try:
                # PP-StructureV3: layout + OCR + table parsing + reading order.
                # `lang='tr'` Türkçe modelini yükler, Latin karakter seti ile İngilizceyi de kapsar.
                self.structure_v3 = PPStructureV3(
                    lang="tr",
                    device=device,
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    enable_mkldnn=False,
                )
                LOGGER.info("PP-StructureV3 başarıyla başlatıldı (lang=tr, device=%s).", device)
            except Exception as exc:
                structure_errors.append(f"PPStructureV3 init hatası: {exc}")
                LOGGER.warning("PPStructureV3 başlatılamadı, PPStructure fallback denenecek.")

        if self.structure_v3 is None and PPStructure is not None:
            try:
                # Geriye uyumluluk: eski PPStructure API.
                self.legacy_structure = PPStructure(
                    show_log=False,
                    layout=True,
                    table=True,
                    ocr=True,
                    lang="tr",
                    return_ocr_result_in_table=True,
                )
                LOGGER.info("Legacy PPStructure başlatıldı (lang=tr).")
            except Exception as exc:
                structure_errors.append(f"PPStructure init hatası: {exc}")

        if self.structure_v3 is None and self.legacy_structure is None:
            detail = " | ".join(structure_errors) if structure_errors else "Bilinmeyen hata."
            raise RuntimeError(
                "PP-Structure başlatılamadı. `paddlex[ocr]` kurulu olmalı ve model indirme izni olmalı. "
                f"Ayrıntı: {detail}"
            )

        # İkincil OCR fallback: hem Türkçe hem İngilizce motorları başlat.
        fallback_errors: List[str] = []
        for lang in ("tr", "en"):
            try:
                engine = self._create_fallback_ocr_engine(lang)
                if lang == "tr":
                    self.ocr_tr = engine
                else:
                    self.ocr_en = engine
            except Exception as exc:
                fallback_errors.append(f"{lang}: {exc}")

        if self.ocr_tr is not None or self.ocr_en is not None:
            LOGGER.info("Fallback OCR motorları başlatıldı (tr/en, en az biri aktif).")
        else:
            # PP-Structure aktifse fallback olmadan da devam edebiliriz.
            LOGGER.warning(
                "Fallback OCR motorları başlatılamadı; yalnızca layout pipeline kullanılacak. Ayrıntı: %s",
                " | ".join(fallback_errors) if fallback_errors else "Bilinmeyen hata",
            )

    def _create_fallback_ocr_engine(self, lang: str) -> Any:
        """Yüklü PaddleOCR sürümüne uyumlu fallback OCR motoru oluşturur."""
        kwargs: Dict[str, Any] = {"lang": lang}

        try:
            sig = inspect.signature(PaddleOCR.__init__)
            params = sig.parameters
        except Exception:
            params = {}

        # PaddleOCR 3.x için önerilen parametre.
        if "use_textline_orientation" in params:
            kwargs["use_textline_orientation"] = True
        # Eski API için geri uyum.
        elif "use_angle_cls" in params:
            kwargs["use_angle_cls"] = True

        # Eski API.
        if "use_gpu" in params:
            kwargs["use_gpu"] = self.use_gpu
        # Yeni pipeline API genellikle `device` bekler.
        else:
            kwargs["device"] = "gpu:0" if self.use_gpu else "cpu"

        if "use_doc_orientation_classify" in params:
            kwargs["use_doc_orientation_classify"] = False
        if "use_doc_unwarping" in params:
            kwargs["use_doc_unwarping"] = False

        if "show_log" in params:
            kwargs["show_log"] = False

        # Bazı CPU ortamlarda Paddle static + oneDNN uyumsuzluğunu önlemek için.
        kwargs["enable_mkldnn"] = False

        return PaddleOCR(**kwargs)

    def process_document(self, file_path: str | Path) -> str:
        """Belgeyi işleyip tek bir Markdown metni döndür.

        Args:
            file_path: PDF/JPG/JPEG/PNG dosya yolu.

        Returns:
            Tüm sayfalar birleştirilmiş Markdown metni.
        """
        path = Path(file_path)
        self._validate_input(path)

        try:
            if path.suffix.lower() == ".pdf":
                # PDF zaten text-layer içeriyorsa önce doğrudan metin çıkar.
                # Bu yol, OCR/Paddle kaynaklı çalışma zamanı hatalarından etkilenmez.
                direct_pdf_md = self._extract_text_layer_from_pdf(path)
                if direct_pdf_md.strip():
                    return direct_pdf_md
                page_images = self._pdf_to_images(path)
            else:
                page_images = [self._load_image(path)]
        except Exception as exc:
            raise RuntimeError(f"Belge yüklenemedi: {path}") from exc

        # Yatayda çok geniş (muhtemel çift sayfa) taramaları otomatik böl.
        expanded_page_images: List[np.ndarray] = []
        for image in page_images:
            expanded_page_images.extend(self._split_double_page_image(image))
        page_images = expanded_page_images

        page_markdowns: List[str] = []

        for page_idx, page_image in enumerate(page_images, start=1):
            try:
                page_md = self._process_single_page(page_image)
                if not page_md.strip():
                    page_md = "[Bu sayfadan metin çıkarılamadı.]"

                page_markdowns.append(f"## Sayfa {page_idx}\n\n{page_md.strip()}")
            except Exception as exc:
                LOGGER.exception("Sayfa %s işlenirken hata: %s", page_idx, exc)
                page_markdowns.append(
                    f"## Sayfa {page_idx}\n\n[Hata: Bu sayfa işlenemedi: {exc}]"
                )

        return "\n\n".join(page_markdowns).strip()

    def _extract_text_layer_from_pdf(self, pdf_path: Path) -> str:
        """PDF text-layer varsa sayfa bazlı Markdown üretir, yoksa boş döner."""
        page_markdowns: List[str] = []
        useful_pages = 0
        min_chars_per_page = 80

        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                return ""

            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                text = page.get_text("text") or ""
                text = text.replace("\x00", "").strip()
                if len(text) >= min_chars_per_page:
                    useful_pages += 1
                if text:
                    page_markdowns.append(f"## Sayfa {page_idx + 1}\n\n{text}")
                else:
                    page_markdowns.append(f"## Sayfa {page_idx + 1}\n\n[Bu sayfadan metin çıkarılamadı.]")

        # En azından anlamlı bir kısmı text-layer ise bu yolu kullan.
        coverage = useful_pages / max(len(page_markdowns), 1)
        if coverage >= 0.4:
            LOGGER.info("PDF text-layer kullanıldı: %s (coverage=%.2f)", pdf_path.name, coverage)
            return "\n\n".join(page_markdowns).strip()
        return ""

    def _process_single_page(self, image_bgr: np.ndarray) -> str:
        """Tek sayfayı PP-Structure ile parse eder, gerekirse OCR fallback uygular."""
        candidates: List[str] = []

        # 1) Orijinal görüntü ile dene.
        structured_md = self._extract_markdown_with_structure(image_bgr)
        structured_md = self._normalize_markdown_tables(structured_md)
        if structured_md.strip():
            candidates.append(structured_md)

        # 2) Basit ön işlenmiş görüntü ile dene (zayıf taramalarda daha iyi sonuç verebilir).
        preprocessed = self._preprocess_image_for_ocr(image_bgr)
        structured_pre = self._extract_markdown_with_structure(preprocessed)
        structured_pre = self._normalize_markdown_tables(structured_pre)
        if structured_pre.strip():
            candidates.append(structured_pre)

        if candidates:
            return max(candidates, key=self._score_text_quality)

        # Layout pipeline boş döndüyse OCR fallback (orijinal + ön işlenmiş).
        fallback_orig = self._normalize_markdown_tables(self._run_fallback_ocr(image_bgr))
        fallback_pre = self._normalize_markdown_tables(self._run_fallback_ocr(preprocessed))

        for text in (fallback_orig, fallback_pre):
            if text.strip():
                candidates.append(text)

        if candidates:
            return max(candidates, key=self._score_text_quality)

        return ""

    def _extract_markdown_with_structure(self, image_bgr: np.ndarray) -> str:
        """PP-StructureV3/PPStructure sonuçlarından Markdown üretir."""
        if self.structure_v3 is not None:
            try:
                results = list(self.structure_v3.predict(input=image_bgr))
            except TypeError:
                # Bazı sürümlerde `predict(image_bgr)` imzası kullanılabiliyor.
                results = list(self.structure_v3.predict(image_bgr))

            if not results:
                return ""

            # Tek image girdisi için genellikle tek result döner; çoklu olasılığına karşı birleştiriyoruz.
            markdown_chunks = [self._extract_markdown_from_v3_result(res) for res in results]
            return "\n\n".join(chunk for chunk in markdown_chunks if chunk.strip()).strip()

        if self.legacy_structure is not None:
            result = self.legacy_structure(image_bgr)
            return self._legacy_result_to_markdown(result)

        return ""

    @staticmethod
    def _extract_markdown_from_v3_result(result_obj: Any) -> str:
        """PP-StructureV3 result objesinden Markdown metni alır."""
        md = getattr(result_obj, "markdown", None)
        if isinstance(md, dict):
            # Farklı sürümlerde farklı anahtar adları olabiliyor.
            for key in ("markdown_texts", "text", "markdown"):
                value = md.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        if isinstance(md, str) and md.strip():
            return md

        # `json` çıktısından ek fallback çıkarım.
        data = getattr(result_obj, "json", None)
        if isinstance(data, dict):
            page_texts: List[str] = []

            # parsing_res_list, PP-StructureV3'te okuma sırası korunmuş blok içeriklerini taşır.
            parsing_blocks = data.get("parsing_res_list") or []
            if isinstance(parsing_blocks, list):
                for block in parsing_blocks:
                    if not isinstance(block, dict):
                        continue
                    content = block.get("block_content")
                    if isinstance(content, str) and content.strip():
                        page_texts.append(content.strip())

            # Table HTML'lerini ayrıca alıp Markdown'a çevir (grid garantisi için).
            table_blocks = data.get("table_res_list") or []
            if isinstance(table_blocks, list):
                for tbl in table_blocks:
                    if not isinstance(tbl, dict):
                        continue
                    html = tbl.get("pred_html") or tbl.get("html")
                    if isinstance(html, str) and html.strip():
                        page_texts.append(DocumentProcessor._html_table_to_markdown(html))

            return "\n\n".join(page_texts).strip()

        return ""

    @staticmethod
    def _legacy_result_to_markdown(result: Sequence[Dict[str, Any]]) -> str:
        """Legacy PPStructure çıktısını Markdown'a dönüştürür (bbox sıralı)."""
        if not result:
            return ""

        def bbox_key(item: Dict[str, Any]) -> Tuple[float, float]:
            bbox = item.get("bbox") or [0, 0, 0, 0]
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                return float(bbox[1]), float(bbox[0])
            return (0.0, 0.0)

        blocks = sorted(result, key=bbox_key)
        parts: List[str] = []

        for block in blocks:
            block_type = str(block.get("type", "")).lower()
            block_res = block.get("res")

            if "table" in block_type and isinstance(block_res, dict):
                html = block_res.get("html")
                if isinstance(html, str) and html.strip():
                    parts.append(DocumentProcessor._html_table_to_markdown(html))
                continue

            text = DocumentProcessor._extract_text_from_legacy_block(block_res)
            if text:
                parts.append(text)

        return "\n\n".join(parts).strip()

    @staticmethod
    def _extract_text_from_legacy_block(block_res: Any) -> str:
        """Legacy OCR tuple/list yapılarından düz metin çıkarır."""
        # Örnek: (boxes, [(text, score), ...])
        if isinstance(block_res, tuple) and len(block_res) >= 2:
            rec_results = block_res[1]
            if isinstance(rec_results, list):
                texts = []
                for item in rec_results:
                    if isinstance(item, (list, tuple)) and item:
                        text = str(item[0]).strip()
                        if text:
                            texts.append(text)
                return "\n".join(texts).strip()

        if isinstance(block_res, list):
            texts = []
            for item in block_res:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("transcription")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                elif isinstance(item, (list, tuple)) and item:
                    text = str(item[0]).strip()
                    if text:
                        texts.append(text)
            return "\n".join(texts).strip()

        if isinstance(block_res, str):
            return block_res.strip()

        return ""

    def _run_fallback_ocr(self, image_bgr: np.ndarray) -> str:
        """Türkçe + İngilizce OCR sonuçlarını birleştirerek fallback metin üretir."""
        tr_text = self._ocr_to_text(self.ocr_tr, image_bgr)
        en_text = self._ocr_to_text(self.ocr_en, image_bgr)

        tr_text = tr_text.strip()
        en_text = en_text.strip()
        if not tr_text and not en_text:
            return ""

        candidates: List[str] = [text for text in (tr_text, en_text) if text]

        # Dillerin farklı güçlü taraflarını kaçırmamak için birleşik adayı da değerlendir.
        if tr_text and en_text and tr_text != en_text:
            combined = f"{tr_text}\n\n---\n\n{en_text}"
            candidates.append(combined)

        return max(candidates, key=self._score_text_quality)

    @staticmethod
    def _ocr_to_text(ocr_engine: Any, image_bgr: np.ndarray) -> str:
        """PaddleOCR düz OCR çıktısını satır satır metne dönüştürür."""
        if ocr_engine is None:
            return ""

        try:
            ocr_result = ocr_engine.ocr(image_bgr, cls=True)
        except Exception:
            return ""

        lines: List[str] = []

        # Beklenen yapı: [[ [box], (text, score) ], ... ]
        if isinstance(ocr_result, list):
            for page in ocr_result:
                if not isinstance(page, list):
                    continue
                for line in page:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue
                    rec = line[1]
                    if isinstance(rec, (list, tuple)) and rec:
                        text = str(rec[0]).strip()
                        if text:
                            lines.append(text)

        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_markdown_tables(markdown_text: str) -> str:
        """Markdown içindeki HTML tablolarını grid tabloya dönüştürür."""
        if not markdown_text:
            return ""

        table_pattern = re.compile(r"<table[\s\S]*?</table>", flags=re.IGNORECASE)

        def _replace(match: re.Match[str]) -> str:
            html_table = match.group(0)
            table_md = DocumentProcessor._html_table_to_markdown(html_table)
            if not table_md:
                return ""
            return DocumentProcessor._sanitize_markdown_table(table_md)

        normalized = table_pattern.sub(_replace, markdown_text)
        normalized = DocumentProcessor._strip_html_wrappers(normalized)
        normalized = DocumentProcessor._cleanup_markdown_whitespace(normalized)
        return normalized.strip()

    @staticmethod
    def _sanitize_markdown_table(table_md: str) -> str:
        """Bozuk/gürültülü tabloları düz metne indirger, iyi tabloları korur."""
        lines = [ln.rstrip() for ln in table_md.splitlines() if ln.strip()]
        if len(lines) < 2:
            return table_md.strip()

        data_rows = lines[2:] if len(lines) > 2 else []
        if not data_rows:
            return table_md.strip()

        cells: List[str] = []
        for row in data_rows:
            if not row.startswith("|"):
                continue
            parts = [cell.strip() for cell in row.strip().strip("|").split("|")]
            cells.extend(parts)

        if not cells:
            return table_md.strip()

        total_cells = len(cells)
        empty_cells = sum(1 for c in cells if not c)
        short_cells = sum(1 for c in cells if 0 < len(c) <= 1)
        alnum_chars = sum(sum(ch.isalnum() for ch in c) for c in cells)
        total_chars = sum(len(c) for c in cells)

        empty_ratio = empty_cells / max(total_cells, 1)
        short_ratio = short_cells / max(total_cells, 1)
        alnum_ratio = alnum_chars / max(total_chars, 1)

        # Heuristik: aşırı boş, aşırı tek-karakterli veya çok düşük alfanümerik yoğunluk.
        noisy = (
            (empty_ratio > 0.65 and len(data_rows) >= 3)
            or (short_ratio > 0.55 and len(data_rows) >= 4)
            or (alnum_ratio < 0.22 and len(data_rows) >= 4)
        )
        if not noisy:
            return table_md.strip()

        flattened_cells = [c for c in cells if c and len(c) > 1]
        if not flattened_cells:
            return ""

        compact_text = " ".join(flattened_cells)
        compact_text = re.sub(r"\s{2,}", " ", compact_text).strip()
        return compact_text

    @staticmethod
    def _strip_html_wrappers(text: str) -> str:
        """OCR çıktısındaki gereksiz HTML sarmalayıcılarını temizler."""
        if not text:
            return ""

        cleaned = re.sub(r"</?(html|body)\b[^>]*>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?div\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
        return unescape(cleaned)

    @staticmethod
    def _cleanup_markdown_whitespace(text: str) -> str:
        """Satır/satır sonu boşluklarını normalize eder."""
        if not text:
            return ""

        lines = [line.rstrip() for line in text.splitlines()]
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _score_text_quality(text: str) -> float:
        """Heuristik kalite skoru: daha okunabilir adayı seçmek için kullanılır."""
        if not text:
            return float("-inf")

        lowered = text.lower()
        letters = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        spaces = sum(ch.isspace() for ch in text)
        bad_chars = text.count("�")
        pipes = text.count("|")

        score = (
            letters
            + (0.5 * digits)
            + (0.2 * spaces)
            - (20.0 * bad_chars)
            - (0.1 * pipes)
        )

        # Temizlikten kaçan gereksiz HTML katmanlarını cezalandır.
        if "<html" in lowered or "<body" in lowered:
            score -= 150.0
        if "<div" in lowered:
            score -= 50.0

        return score

    @staticmethod
    def _deskew_grayscale(binary_image: np.ndarray) -> np.ndarray:
        """Tahmini küçük eğiklikleri düzeltir."""
        coords = np.column_stack(np.where(binary_image < 250))
        if coords.size == 0 or coords.shape[0] < 800:
            return binary_image

        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        # Aşırı veya anlamsız açılarla döndürme yapmayalım.
        if abs(angle) < 0.15 or abs(angle) > 20:
            return binary_image

        h, w = binary_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary_image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    @staticmethod
    def _preprocess_image_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
        """Zor taramalar için temel OCR ön işleme."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        max_side = max(h, w)
        if 0 < max_side < 1800:
            scale = 1800.0 / float(max_side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Kontrast iyileştirme + hafif gürültü azaltma.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # İkilileştirme OCR doğruluğunu özellikle eski belgelerde artırır.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        binary = cv2.medianBlur(binary, 3)
        binary = DocumentProcessor._deskew_grayscale(binary)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _html_table_to_markdown(html_table: str) -> str:
        """HTML tabloyu Markdown grid formatına çevirir.

        Not: Markdown grid tabloları rowspan/colspan'ı doğal olarak desteklemez.
        Bu nedenle hücre yapısını olabildiğince koruyacak şekilde boşluk doldurma
        stratejisi uygulanır.
        """
        soup = BeautifulSoup(html_table, "html.parser")
        table = soup.find("table")
        if table is None:
            return ""

        rows = table.find_all("tr")
        if not rows:
            return ""

        parsed_rows: List[List[str]] = []
        pending_rowspan: Dict[int, Tuple[int, str]] = {}

        for tr in rows:
            row_cells: List[str] = []
            col_idx = 0

            # Bu satıra devreden rowspan hücrelerini yerleştir.
            while col_idx in pending_rowspan:
                remain, value = pending_rowspan[col_idx]
                row_cells.append(value)
                if remain > 1:
                    pending_rowspan[col_idx] = (remain - 1, value)
                else:
                    pending_rowspan.pop(col_idx, None)
                col_idx += 1

            for cell in tr.find_all(["th", "td"]):
                while col_idx in pending_rowspan:
                    remain, value = pending_rowspan[col_idx]
                    row_cells.append(value)
                    if remain > 1:
                        pending_rowspan[col_idx] = (remain - 1, value)
                    else:
                        pending_rowspan.pop(col_idx, None)
                    col_idx += 1

                text = " ".join(cell.get_text(" ", strip=True).split())
                colspan = max(int(cell.get("colspan", 1) or 1), 1)
                rowspan = max(int(cell.get("rowspan", 1) or 1), 1)

                row_cells.append(text)

                # colspan kadar sütun doldur (ilk hücre metin, kalanı boş placeholder).
                for _ in range(colspan - 1):
                    row_cells.append("")

                # rowspan devri: mevcut ve colspan ile açılan tüm sütunlara yay.
                if rowspan > 1:
                    base_col = col_idx
                    for offset in range(colspan):
                        pending_rowspan[base_col + offset] = (rowspan - 1, "")

                col_idx += colspan

            # Satır bitiminde devreden span varsa satıra ekleyelim.
            while col_idx in pending_rowspan:
                remain, value = pending_rowspan[col_idx]
                row_cells.append(value)
                if remain > 1:
                    pending_rowspan[col_idx] = (remain - 1, value)
                else:
                    pending_rowspan.pop(col_idx, None)
                col_idx += 1

            parsed_rows.append(row_cells)

        # Satırları eşit sütun sayısına normalize et.
        max_cols = max(len(r) for r in parsed_rows)
        normalized_rows = [r + [""] * (max_cols - len(r)) for r in parsed_rows]

        if max_cols == 0:
            return ""

        header = normalized_rows[0]
        separator = ["---"] * max_cols
        body = normalized_rows[1:] if len(normalized_rows) > 1 else []

        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        md_lines.extend("| " + " | ".join(row) + " |" for row in body)

        return "\n".join(md_lines).strip()

    def _pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """PDF'i sayfa sayfa OpenCV BGR görüntüsüne çevirir."""
        images: List[np.ndarray] = []

        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                raise ValueError("PDF sayfa içermiyor.")

            matrix = fitz.Matrix(self.pdf_zoom, self.pdf_zoom)
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=matrix, alpha=False)

                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, pix.n)

                if pix.n == 4:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                images.append(img_bgr)

        return images

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        """Resim dosyasını OpenCV BGR formatında yükler."""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Resim okunamadı: {image_path}")
        return image

    @staticmethod
    def _split_double_page_image(image_bgr: np.ndarray) -> List[np.ndarray]:
        """Yan yana iki sayfa olabilecek taramaları otomatik left/right böler."""
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return [image_bgr]

        aspect_ratio = w / float(h)
        # Muhafazakar eşik: belirgin geniş görüntülerde split dene.
        if aspect_ratio < 1.35:
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

        # Ortadaki boşluk yeterince belirgin değilse split yapma.
        if gutter_ink > (0.45 * mean_ink):
            return [image_bgr]

        left = image_bgr[:, :gutter_idx]
        right = image_bgr[:, gutter_idx:]
        if left.size == 0 or right.size == 0:
            return [image_bgr]

        # İki parçanın da içerik taşıdığını doğrula.
        left_ink = float(binary_inv[:, :gutter_idx].sum())
        right_ink = float(binary_inv[:, gutter_idx:].sum())
        total_ink = left_ink + right_ink + 1e-6
        if (left_ink / total_ink) < 0.2 or (right_ink / total_ink) < 0.2:
            return [image_bgr]

        return [left, right]

    def _validate_input(self, path: Path) -> None:
        """Dosya türü ve erişilebilirlik kontrollerini yapar."""
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Desteklenmeyen dosya türü: {ext}. "
                f"Desteklenenler: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )


def process_document_to_markdown(file_path: str | Path, use_gpu: bool = False) -> str:
    """Kullanımı kolay fonksiyonel arayüz."""
    processor = DocumentProcessor(use_gpu=use_gpu)
    return processor.process_document(file_path)


def process_directory_to_markdown(
    input_dir: str | Path,
    output_dir: str | Path,
    use_gpu: bool = False,
    recursive: bool = True,
) -> List[Path]:
    """Bir klasördeki destekli dosyaları tek tek işleyip .md çıktıları üretir."""
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
        LOGGER.warning("İşlenecek destekli dosya bulunamadı: %s", in_dir)
        return []

    written: List[Path] = []
    for src in files:
        try:
            markdown = processor.process_document(src)
            out_path = out_dir / f"{src.stem}.md"
            out_path.write_text(markdown, encoding="utf-8")
            written.append(out_path)
            LOGGER.info("Tamamlandı: %s -> %s", src.name, out_path.name)
        except Exception as exc:
            LOGGER.exception("Dosya işlenemedi: %s | Hata: %s", src, exc)

    return written


if __name__ == "__main__":
    # Örnek kullanım:
    #   python document_processor.py test.pdf
    #   python document_processor.py test.jpg
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF/JPG/PNG belgelerini OCR + layout analizi ile Markdown'a dönüştürür."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Tek dosya modu: test.pdf veya test.jpg gibi giriş dosyası",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Klasör modu: destekli dosyaların okunacağı klasör",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="PaddleOCR inference için GPU kullan",
    )
    parser.add_argument(
        "--output",
        default="output.md",
        help="Tek dosya modu: Markdown çıktısının kaydedileceği dosya (varsayılan: output.md)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Klasör modu: .md çıktılarının kaydedileceği klasör",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Klasör modunda alt klasörleri tarama",
    )

    args = parser.parse_args()

    try:
        if args.input_dir:
            default_output_dir = Path(__file__).resolve().parent / "results" / "ocrMdResults"
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
        else:
            if not args.input_file:
                raise ValueError("Tek dosya modu için `input_file` veya klasör modu için `--input-dir` verin.")

            markdown_output = process_document_to_markdown(args.input_file, use_gpu=args.gpu)

            output_path = Path(args.output)
            output_path.write_text(markdown_output, encoding="utf-8")

            print(f"Markdown başarıyla üretildi: {output_path.resolve()}")
            print("--- Önizleme (ilk 1200 karakter) ---")
            print(markdown_output[:1200])
    except Exception as err:
        print(f"Hata: {err}")
        raise SystemExit(1)
