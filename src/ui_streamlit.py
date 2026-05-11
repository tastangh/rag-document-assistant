"""Sade chat UI: belge yukle -> hazirla butonu -> soru sor."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fitz
import streamlit as st

from chunk_embedding_pipeline import run_pipeline
from document_processor import process_document_to_markdown
from generation_pipeline import ask_question
from retrieval_pipeline import DEFAULT_COLLECTION, build_vector_index


UI_RUNTIME_ROOT = Path("src/results/ui_runtime")
GREETING_WORDS = {"merhaba", "selam", "selamlar", "hello", "hi", "hey"}
FALLBACK_ANSWER = "Baglamda yeterli bilgi yok."
ERROR_MARKERS = [
    re.compile(r"\bhata\b", re.IGNORECASE),
    re.compile(r"islenemedi", re.IGNORECASE),
    re.compile(r"unimplemented", re.IGNORECASE),
    re.compile(r"onednn[_\s-]?instruction", re.IGNORECASE),
    re.compile(r"convertpirattribute2runtimeattribute", re.IGNORECASE),
]


def _ensure_dirs() -> None:
    UI_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)


def _session_paths() -> Dict[str, Path]:
    sid = str(st.session_state.get("session_id", "chat_session"))
    root = UI_RUNTIME_ROOT / sid
    paths = {
        "root": root,
        "upload_dir": root / "uploads",
        "ocr_md_dir": root / "ocr_md",
        "chunk_dir": root / "chunks",
        "vector_dir": root / "vector",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _state_path(paths: Dict[str, Path]) -> Path:
    return paths["root"] / "state.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(list(value))
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _persist_state(paths: Dict[str, Path]) -> None:
    state = {
        "docs": st.session_state.get("docs", {}),
        "messages": st.session_state.get("messages", []),
        "index_signature": st.session_state.get("index_signature", ""),
        "ready": bool(st.session_state.get("ready", False)),
        "model_name": st.session_state.get("model_name", "qwen3:8b"),
        "strict_guardrail": bool(st.session_state.get("strict_guardrail", False)),
        "processed_upload_keys": sorted(list(st.session_state.get("processed_upload_keys", set()))),
        "last_error": st.session_state.get("last_error", ""),
    }
    target = _state_path(paths)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_safe(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def _load_state(paths: Dict[str, Path]) -> None:
    p = _state_path(paths)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return

    docs = data.get("docs", {}) or {}
    cleaned_docs: Dict[str, Dict[str, Any]] = {}
    for name, d in docs.items():
        doc_path = Path(str(d.get("path", "")))
        md_path = Path(str(d.get("md_path", "")))
        if not doc_path.exists():
            continue
        dd = dict(d)
        dd["path"] = str(doc_path)
        dd["md_path"] = str(md_path)
        cleaned_docs[name] = dd

    st.session_state.docs = cleaned_docs
    st.session_state.messages = data.get("messages", []) or []
    st.session_state.index_signature = str(data.get("index_signature", ""))
    st.session_state.ready = bool(data.get("ready", False))
    st.session_state.model_name = str(data.get("model_name", "qwen3:8b"))
    st.session_state.strict_guardrail = bool(data.get("strict_guardrail", False))
    st.session_state.processed_upload_keys = set(data.get("processed_upload_keys", []) or [])
    st.session_state.last_error = str(data.get("last_error", ""))


def _clear_dir(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            _clear_dir(item)
            item.rmdir()


def _save_upload(uploaded_file, upload_dir: Path) -> Path:
    out = upload_dir / uploaded_file.name
    out.write_bytes(uploaded_file.getbuffer())
    return out


def _extract_pdf_text_direct(file_path: Path) -> str:
    pages: List[str] = []
    with fitz.open(file_path) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            if not text:
                # Bazı PDF'lerde "text" boş gelebiliyor; bloklardan toparlamayı dene.
                blocks = page.get_text("blocks") or []
                block_texts = []
                for b in blocks:
                    if len(b) >= 5 and isinstance(b[4], str):
                        t = b[4].strip()
                        if t:
                            block_texts.append(t)
                text = "\n".join(block_texts).strip()
            if text:
                pages.append(f"## Sayfa {i + 1}\n\n{text}")
    return "\n\n".join(pages).strip()


def _document_to_markdown(file_path: Path, use_gpu: bool = True) -> str:
    if file_path.suffix.lower() == ".pdf":
        direct = _extract_pdf_text_direct(file_path)
        # Kısa da olsa anlamlı text-layer varsa OCR'a düşmeden bunu kullan.
        if len(direct) > 20:
            return direct
    return process_document_to_markdown(file_path, use_gpu=use_gpu)


def _looks_like_error_markdown(md: str) -> bool:
    normalized = unicodedata.normalize("NFKD", md)
    normalized = normalized.encode("ascii", "ignore").decode("ascii", errors="ignore").lower()
    compact = re.sub(r"\s+", " ", normalized).strip()
    for marker in ERROR_MARKERS:
        if marker.search(compact):
            return True
    return False


def _rebuild_index(active_docs: Dict[str, Dict[str, Any]], paths: Dict[str, Path], use_gpu: bool = True) -> Dict[str, Any]:
    ocr_md_dir = paths["ocr_md_dir"]
    chunk_dir = paths["chunk_dir"]
    vector_dir = paths["vector_dir"]
    _clear_dir(ocr_md_dir)
    _clear_dir(chunk_dir)
    _clear_dir(vector_dir)

    selected = [d for d in active_docs.values() if d.get("active") and d.get("status") == "ready"]
    if not selected:
        return {"indexed_chunk_count": 0, "processed_docs": [], "failed_count": 0, "first_error": ""}

    written: List[str] = []
    first_error = ""
    failed_count = 0
    for d in selected:
        src = Path(str(d["path"]))
        md = Path(str(d["md_path"]))
        if not md.exists():
            text = _document_to_markdown(src, use_gpu=use_gpu)
            if not text.strip() or _looks_like_error_markdown(text):
                d["status"] = "failed"
                d["error"] = "Metin cikartilamadi (OCR/PDF parse)."
                failed_count += 1
                if not first_error:
                    first_error = d["error"]
                continue
            md.write_text(text, encoding="utf-8")
            d["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if md.exists():
            written.append(md.name)

    if not written:
        return {
            "indexed_chunk_count": 0,
            "processed_docs": [],
            "failed_count": failed_count,
            "first_error": first_error,
        }

    run_pipeline(
        input_dir=ocr_md_dir,
        output_dir=chunk_dir,
        model_name="BAAI/bge-m3",
        chunk_size=1200,
        chunk_overlap=200,
        min_chunk_size=120,
        device="cuda" if use_gpu else "cpu",
        batch_size=32,
    )
    idx = build_vector_index(
        artifacts_dir=chunk_dir,
        persist_dir=vector_dir,
        collection_name=DEFAULT_COLLECTION,
        batch_size=256,
    )
    return {
        "indexed_chunk_count": idx.get("indexed_chunk_count", 0),
        "processed_docs": written,
        "failed_count": failed_count,
        "first_error": first_error,
    }


def _status_badge(status: str) -> str:
    if status == "ready":
        return "🟢 ready"
    if status == "failed":
        return "🔴 failed"
    return "🟡 analyzing"


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    with st.expander("Kaynaklar"):
        for src in sources:
            st.markdown(
                f"- **{src.get('doc_id','')}** | s{src.get('page', 0)} | `{src.get('chunk_id','')}`\n"
                f"  - {src.get('text_preview','')}"
            )


def _is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETING_WORDS


def _derive_term_definition(question: str, sources: List[Dict[str, Any]]) -> str | None:
    m = re.match(r"^\s*([A-Za-z0-9_\-]+)\s+nedir\??\s*$", question.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    term = m.group(1)
    term_l = term.lower()
    for src in sources:
        preview = str(src.get("text_preview", "")).strip()
        if not preview:
            continue
        compact = " ".join(preview.replace("_", " ").split())
        if term_l not in compact.lower():
            continue
        idx = compact.lower().find(term_l)
        tail = compact[idx + len(term) :].lstrip(" :;,-_")
        if tail:
            expansion = " ".join(tail.split()[:14]).strip(" .,:;")
            if expansion:
                return f"Bu belgede **{term}**, `{term} {expansion}` ifadesiyle geciyor."
        return f"Bu belgede **{term}** ifadesi geciyor; kaynakta tanim parcali gorunuyor."
    return None


def main() -> None:
    st.set_page_config(page_title="Belge Asistani", page_icon=":speech_balloon:", layout="centered")
    _ensure_dirs()
    paths = _session_paths()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs" not in st.session_state:
        st.session_state.docs = {}
    if "processed_upload_keys" not in st.session_state:
        st.session_state.processed_upload_keys = set()
    if "index_signature" not in st.session_state:
        st.session_state.index_signature = ""
    if "ready" not in st.session_state:
        st.session_state.ready = False
    if "model_name" not in st.session_state:
        st.session_state.model_name = "qwen3:8b"
    if "strict_guardrail" not in st.session_state:
        st.session_state.strict_guardrail = False
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    _load_state(paths)

    st.title("Belge Asistani")
    st.caption("PDF/resim yukle, sonra Belgeleri Hazirla butonuna bas ve soru sor.")

    with st.sidebar:
        st.subheader("Aktif Dokumanlar")
        if st.button("Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = []
            _persist_state(paths)
        if st.session_state.docs:
            names = sorted(st.session_state.docs.keys())
            for name in names:
                d = st.session_state.docs[name]
                c1, c2 = st.columns([5, 1])
                with c1:
                    new_active = st.checkbox(
                        f"{name} ({_status_badge(str(d.get('status', 'analyzing')))})",
                        value=bool(d.get("active", True)),
                        key=f"active_{name}",
                        help=f"Durum: {d.get('status', 'unknown')}",
                    )
                    old_active = bool(d.get("active", True))
                    d["active"] = new_active
                    if old_active != new_active:
                        _persist_state(paths)
                with c2:
                    if st.button("X", key=f"remove_{name}", help="Dokumani kaldir"):
                        try:
                            Path(str(d.get("path", ""))).unlink(missing_ok=True)
                            Path(str(d.get("md_path", ""))).unlink(missing_ok=True)
                        except Exception:
                            pass
                        del st.session_state.docs[name]
                        _persist_state(paths)
                        st.rerun()
                if d.get("status") == "failed":
                    st.caption(f"{name}: hata - {d.get('error', 'metin cikartilamadi')}")
        else:
            st.caption("Henuz dokuman yok.")

    uploaded_files = st.file_uploader(
        "Belgeleri yukle",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    prepare_clicked = st.button("Belgeleri Hazirla", type="primary", use_container_width=True)

    new_uploads = []
    if uploaded_files:
        for f in uploaded_files:
            up_key = f"{f.name}:{f.size}"
            if up_key not in st.session_state.processed_upload_keys:
                new_uploads.append((up_key, f))

    if prepare_clicked:
        if not uploaded_files:
            st.warning("Once en az bir belge sec.")
        elif not new_uploads:
            st.info("Secili belgeler zaten hazirlanmis. Yeni dosya secersen tekrar islenir.")
        else:
            with st.spinner("Yukleniyor..."):
                for up_key, f in new_uploads:
                    saved = _save_upload(f, upload_dir=paths["upload_dir"])
                    md_path = paths["ocr_md_dir"] / f"{saved.stem}.md"
                    status = "analyzing"
                    err = ""
                    created_at = datetime.now().isoformat(timespec="seconds")
                    st.session_state.docs[saved.name] = {
                        "path": str(saved),
                        "md_path": str(md_path),
                        "active": False,
                        "status": status,
                        "error": err,
                        "created_at": created_at,
                        "updated_at": created_at,
                    }
                    _persist_state(paths)

                    status = "ready"
                    try:
                        md = _document_to_markdown(saved, use_gpu=True)
                        if not md.strip() or _looks_like_error_markdown(md):
                            status = "failed"
                            first_line = (md.strip().splitlines()[0] if md.strip() else "")
                            err = f"Metin cikartma basarisiz. {first_line}".strip()
                        else:
                            md_path.write_text(md, encoding="utf-8")
                    except Exception as exc:
                        status = "failed"
                        err = str(exc)

                    st.session_state.docs[saved.name] = {
                        "path": str(saved),
                        "md_path": str(md_path),
                        "active": status == "ready",
                        "status": status,
                        "error": err,
                        "created_at": created_at,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    st.session_state.processed_upload_keys.add(up_key)
                    st.session_state.last_error = err if status == "failed" else st.session_state.last_error
                    _persist_state(paths)

    active_signature = "|".join(
        sorted(
            name
            for name, d in st.session_state.docs.items()
            if d.get("active") and d.get("status") == "ready"
        )
    )
    if active_signature != st.session_state.index_signature:
        with st.spinner("Analiz ve index guncelleniyor..."):
            summary = _rebuild_index(st.session_state.docs, paths, use_gpu=True)
            st.session_state.index_signature = active_signature
            st.session_state.ready = summary.get("indexed_chunk_count", 0) > 0
            if st.session_state.ready:
                st.caption(
                    f"Aktif: {len(summary.get('processed_docs', []))} dokuman | "
                    f"Index chunk: {summary.get('indexed_chunk_count', 0)}"
                )
            else:
                failed_count = int(summary.get("failed_count", 0))
                first_error = str(summary.get("first_error", "")) or str(st.session_state.get("last_error", ""))
                diagnostic = f"Aktif ready dokuman yok. failed dokumanlar: {failed_count}."
                if first_error:
                    diagnostic += f" en son hata: {first_error}"
                st.warning(diagnostic)
                st.session_state.last_error = first_error
            _persist_state(paths)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    prompt = st.chat_input("Belgeyle ilgili sorunu yaz...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        _persist_state(paths)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if _is_greeting(prompt):
                ans = "Merhaba. Belgeleri yukleyip Belgeleri Hazirla butonuna bastiktan sonra sorunu sorabilirsin."
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                _persist_state(paths)
                return

            if not st.session_state.ready:
                failed_count = sum(1 for d in st.session_state.docs.values() if d.get("status") == "failed")
                last_error = str(st.session_state.get("last_error", ""))
                ans = f"Aktif ready dokuman yok. failed dokumanlar: {failed_count}."
                if last_error:
                    ans += f" en son hata: {last_error}"
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                _persist_state(paths)
                return

            with st.spinner("Dusunuyorum..."):
                try:
                    result = ask_question(
                        question=prompt,
                        persist_dir=paths["vector_dir"],
                        collection_name=DEFAULT_COLLECTION,
                        initial_k=32,
                        final_k=8,
                        device="cuda",
                        disable_rerank=False,
                        model_name=st.session_state.model_name,
                        strict_guardrail=st.session_state.strict_guardrail,
                    )
                    answer = str(result.get("answer", ""))
                    sources = list(result.get("sources", []) or [])
                    if answer.strip() == FALLBACK_ANSWER and sources:
                        derived = _derive_term_definition(prompt, sources)
                        if derived:
                            answer = derived
                    st.markdown(answer)
                    if sources:
                        _render_sources(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                    _persist_state(paths)
                except Exception as exc:
                    err = f"Bir hata oldu: {exc}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    _persist_state(paths)


if __name__ == "__main__":
    main()
