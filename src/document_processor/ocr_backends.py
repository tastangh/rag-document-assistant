from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from config import OCR_CACHE_DIR
from .markdown_tables import html_table_to_markdown, normalize_markdown_tables, score_text_quality

PaddleOCR = None
PPStructureV3 = None
PPStructure = None
logger = logging.getLogger(__name__)


def init_engines(processor: Any) -> None:
    device = "gpu" if processor.use_gpu else "cpu"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(OCR_CACHE_DIR))
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
    except Exception:
        numpy_major = 0
    if numpy_major >= 2:
        raise RuntimeError("NumPy 2.x ile PaddleOCR/PaddleX uyumsuz. `numpy<2` kullanin.")

    global PaddleOCR, PPStructureV3, PPStructure
    if PaddleOCR is None:
        from paddleocr import PaddleOCR as _PaddleOCR
        PaddleOCR = _PaddleOCR
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

    errors: List[str] = []
    if PPStructureV3 is not None:
        try:
            kwargs: Dict[str, Any] = {
                "lang": processor.ocr_lang if processor.ocr_lang in {"tr", "en"} else "tr",
                "device": device,
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "enable_mkldnn": False,
            }
            if processor.ocr_profile == "lightweight":
                kwargs["use_formula_recognition"] = False
                kwargs["use_textline_orientation"] = False
            processor.structure_v3 = PPStructureV3(**kwargs)
        except Exception as exc:
            errors.append(f"PPStructureV3: {exc}")

    if processor.structure_v3 is None and PPStructure is not None:
        try:
            processor.legacy_structure = PPStructure(
                show_log=False,
                layout=True,
                table=True,
                ocr=True,
                lang=processor.ocr_lang if processor.ocr_lang in {"tr", "en"} else "tr",
                return_ocr_result_in_table=True,
            )
        except Exception as exc:
            errors.append(f"PPStructure: {exc}")

    if processor.structure_v3 is None and processor.legacy_structure is None:
        detail = " | ".join(errors) if errors else "unknown"
        raise RuntimeError(f"PP-Structure baslatilamadi: {detail}")

    for lang in ("tr", "en"):
        try:
            engine = create_fallback_ocr_engine(processor, lang)
            if lang == "tr":
                processor.ocr_tr = engine
            else:
                processor.ocr_en = engine
        except Exception:
            pass


def create_fallback_ocr_engine(processor: Any, lang: str) -> Any:
    kwargs: Dict[str, Any] = {"lang": lang}
    try:
        params = inspect.signature(PaddleOCR.__init__).parameters
    except Exception:
        params = {}
    if "use_textline_orientation" in params:
        kwargs["use_textline_orientation"] = True
    elif "use_angle_cls" in params:
        kwargs["use_angle_cls"] = True
    if "use_gpu" in params:
        kwargs["use_gpu"] = processor.use_gpu
    else:
        kwargs["device"] = "gpu:0" if processor.use_gpu else "cpu"
    if "show_log" in params:
        kwargs["show_log"] = False
    kwargs["enable_mkldnn"] = False
    return PaddleOCR(**kwargs)


def extract_markdown_with_structure(processor: Any, image_bgr: np.ndarray) -> str:
    if processor.structure_v3 is not None:
        try:
            results = list(processor.structure_v3.predict(input=image_bgr))
        except TypeError:
            results = list(processor.structure_v3.predict(image_bgr))
        if not results:
            return ""
        chunks = [extract_markdown_from_v3_result(res) for res in results]
        return "\n\n".join(c for c in chunks if c.strip()).strip()
    if processor.legacy_structure is not None:
        return legacy_result_to_markdown(processor.legacy_structure(image_bgr))
    return ""


def extract_markdown_from_v3_result(result_obj: Any) -> str:
    md = getattr(result_obj, "markdown", None)
    if isinstance(md, dict):
        for key in ("markdown_texts", "text", "markdown"):
            value = md.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if isinstance(md, str) and md.strip():
        return md
    data = getattr(result_obj, "json", None)
    if isinstance(data, dict):
        page_texts: List[str] = []
        for block in data.get("parsing_res_list") or []:
            if isinstance(block, dict):
                content = block.get("block_content")
                if isinstance(content, str) and content.strip():
                    page_texts.append(content.strip())
        for tbl in data.get("table_res_list") or []:
            if isinstance(tbl, dict):
                html = tbl.get("pred_html") or tbl.get("html")
                if isinstance(html, str) and html.strip():
                    page_texts.append(html_table_to_markdown(html))
        return "\n\n".join(page_texts).strip()
    return ""


def legacy_result_to_markdown(result: Sequence[Dict[str, Any]]) -> str:
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
                parts.append(html_table_to_markdown(html))
            continue
        text = extract_text_from_legacy_block(block_res)
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts).strip()


def extract_text_from_legacy_block(block_res: Any) -> str:
    if isinstance(block_res, str):
        return block_res.strip()
    if isinstance(block_res, dict):
        for key in ("text", "content", "markdown"):
            value = block_res.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(block_res, list):
        lines: List[str] = []
        for item in block_res:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rec = item[1]
                if isinstance(rec, (list, tuple)) and rec:
                    t = str(rec[0]).strip()
                    if t:
                        lines.append(t)
        return "\n".join(lines).strip()
    return ""


def run_fallback_ocr(processor: Any, image_bgr: np.ndarray) -> str:
    tr_text = ocr_to_text(processor.ocr_tr, image_bgr)
    en_text = ocr_to_text(processor.ocr_en, image_bgr)
    candidates = [x for x in (tr_text, en_text) if x.strip()]
    if tr_text.strip() and en_text.strip() and tr_text != en_text:
        candidates.append(f"{tr_text}\n\n---\n\n{en_text}")
    if not candidates:
        return ""
    return max(candidates, key=score_text_quality)


def ocr_to_text(ocr_engine: Any, image_bgr: np.ndarray) -> str:
    if ocr_engine is None:
        return ""
    try:
        ocr_result = ocr_engine.ocr(image_bgr, cls=True)
    except Exception:
        return ""
    lines: List[str] = []
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


def process_single_page(processor: Any, image_bgr: np.ndarray, preprocess_fn: Any) -> str:
    candidates: List[str] = []
    structured_md = normalize_markdown_tables(extract_markdown_with_structure(processor, image_bgr))
    if structured_md.strip():
        candidates.append(structured_md)
    preprocessed = preprocess_fn(image_bgr)
    structured_pre = normalize_markdown_tables(extract_markdown_with_structure(processor, preprocessed))
    if structured_pre.strip():
        candidates.append(structured_pre)
    if candidates:
        return max(candidates, key=score_text_quality)
    fallback_orig = normalize_markdown_tables(run_fallback_ocr(processor, image_bgr))
    fallback_pre = normalize_markdown_tables(run_fallback_ocr(processor, preprocessed))
    for text in (fallback_orig, fallback_pre):
        if text.strip():
            candidates.append(text)
    if candidates:
        return max(candidates, key=score_text_quality)
    return ""

