from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from chunk_embedding import run_pipeline
from config import EMBED_DEVICE
from document_processor import process_document_to_markdown
from retrieval import DEFAULT_COLLECTION, build_vector_index

INLINE_CITATION_RE = re.compile(r"\[(?:doc_id:)?[^:\]]+:p\d+:[^\]]+\]")


def prepare_documents(
    file_paths: List[Path],
    ocr_md_dir: Path,
    chunk_dir: Path,
    vector_dir: Path,
    embed_model_name: str,
    ocr_profile: str,
) -> Dict[str, Any]:
    from .state import clear_dir

    clear_dir(ocr_md_dir)
    clear_dir(chunk_dir)
    failed: List[Dict[str, str]] = []
    written_md: List[str] = []

    for p in file_paths:
        try:
            markdown = process_document_to_markdown(
                p,
                use_gpu=False,
                ocr_lang="tr",
                ocr_profile=ocr_profile,
                ocr_backend="paddle",
            )
            md_path = ocr_md_dir / f"{p.stem}.md"
            md_path.write_text(markdown, encoding="utf-8")
            written_md.append(md_path.name)
        except Exception as exc:
            failed.append({"file": p.name, "error": str(exc)})
    if not written_md:
        return {"ok": False, "message": "Hicbir dosya OCR edilemedi.", "failed": failed}

    run_pipeline(
        input_dir=ocr_md_dir,
        output_dir=chunk_dir,
        model_name=embed_model_name,
        chunk_size=1200,
        chunk_overlap=200,
        min_chunk_size=120,
        device=EMBED_DEVICE,
        batch_size=32,
    )
    idx = build_vector_index(artifacts_dir=chunk_dir, persist_dir=vector_dir, collection_name=DEFAULT_COLLECTION, batch_size=256)
    return {
        "ok": True,
        "written_md": written_md,
        "failed": failed,
        "indexed_chunk_count": idx.get("indexed_chunk_count", 0),
        "ocr_gpu_enabled": False,
        "ocr_gpu_available": False,
        "embed_model_name": embed_model_name,
        "ocr_lang": "tr+en(auto)",
        "ocr_profile": ocr_profile,
        "ocr_backend": "paddle",
        "ocr_device_mode": "cpu",
    }


def strip_inline_citations(text: str) -> str:
    cleaned = INLINE_CITATION_RE.sub("", text or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def doc_id_from_doc_entry(doc: Dict[str, Any]) -> Optional[str]:
    path = str(doc.get("path", "")).strip()
    if not path:
        return None
    try:
        return Path(path).stem
    except Exception:
        return None


def doc_options() -> List[tuple[str, str]]:
    ready_docs = [d for d in st.session_state.get("docs", []) if d.get("status") == "ready"]
    options: List[tuple[str, str]] = []
    for d in ready_docs:
        did = doc_id_from_doc_entry(d)
        if did:
            options.append((str(d.get("name", did)), did))
    return options


def resolve_selected_doc_id(mode: str, manual_doc_id: str) -> Optional[str]:
    return manual_doc_id.strip() or None if mode == "manual" else None


def ollama_health(ollama_url: str = "http://localhost:11434/api/generate") -> tuple[bool, str]:
    tags_url = ollama_url.rstrip("/").replace("/api/generate", "") + "/api/tags"
    try:
        with urlopen(tags_url, timeout=2) as resp:
            status = int(getattr(resp, "status", 200))
            if status >= 400:
                return False, f"HTTP {status}"
        return True, "ok"
    except URLError as exc:
        return False, str(getattr(exc, "reason", exc))
    except Exception as exc:
        return False, str(exc)


def is_small_talk(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    small_talk_exact = {
        "selam",
        "merhaba",
        "hey",
        "hi",
        "hello",
        "nasilsin",
        "naber",
        "iyi misin",
        "tesekkurler",
        "teşekkürler",
        "sa",
        "slm",
    }
    if q in small_talk_exact:
        return True
    return any(q.startswith(prefix) for prefix in ("selam", "merhaba", "nasılsın", "nasilsin", "hello", "hi ", "hey "))


def classify_query_mode(text: str, rag_ready: bool) -> str:
    q = (text or "").strip().lower()
    if not q or is_small_talk(q):
        return "chat"
    if not rag_ready:
        return "chat"
    interpret_markers = ["yorumla", "değerlendir", "degerlendir", "özetle", "ozetle", "hangi konferans", "sence", "çıkarım", "cikarim", "öner", "oner", "uygun mu", "ne anlatıyor", "ne anlatiyor"]
    fact_markers = ["kaç", "kac", "nedir", "sayi", "sayı", "sayilar", "sayılar", "numara", "hangi dergi", "hangi sayfa", "tarih", "deadline", "teslim", "kim", "nerede", "madde", "başlık", "baslik"]
    if any(m in q for m in interpret_markers):
        return "rag_interpret"
    if any(m in q for m in fact_markers):
        return "rag_fact"
    return "rag_interpret"


def auto_rag_params(mode: str) -> Dict[str, Any]:
    if mode == "rag_fact":
        return {"strict_guardrail": True, "guardrail_threshold": 0.55, "citation_min_coverage": 0.90, "temperature_boost": 0.0, "top_p_floor": 0.85, "temperature_cap": 0.45, "allow_extractive_on_guardrail_fail": False}
    if mode == "rag_interpret":
        return {"strict_guardrail": True, "guardrail_threshold": 0.45, "citation_min_coverage": 0.70, "temperature_boost": 0.10, "top_p_floor": 0.90, "temperature_cap": 0.70, "allow_extractive_on_guardrail_fail": True}
    return {"strict_guardrail": False, "guardrail_threshold": 0.40, "citation_min_coverage": 0.0, "temperature_boost": 0.15, "top_p_floor": 0.90, "temperature_cap": 1.10, "allow_extractive_on_guardrail_fail": False}

