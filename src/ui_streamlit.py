"""Basit calisan UI: Yukle -> OCR -> Index -> Soru sor.

Faz 7: session izolasyonu + state kaliciligi.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from chunk_embedding_pipeline import run_pipeline
from config import (
    EMBED_DEVICE,
    EMBED_MODEL_NAME,
    OCR_USE_GPU,
    OLLAMA_MODEL,
    RETRIEVAL_DEVICE,
    RUNTIME_ROOT,
)
from document_processor import process_document_to_markdown
from generation_pipeline import ask_question
from retrieval_pipeline import DEFAULT_COLLECTION, build_vector_index


def _effective_ocr_gpu_enabled() -> bool:
    if not OCR_USE_GPU:
        return False
    try:
        import paddle

        return bool(paddle.is_compiled_with_cuda())
    except Exception:
        return False


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _resolve_session_id() -> str:
    sid = str(st.query_params.get("sid", "")).strip()
    if not sid:
        sid = str(uuid.uuid4())
        st.query_params["sid"] = sid
    return sid


def _ensure_dirs(session_id: str) -> Dict[str, Path]:
    runtime = RUNTIME_ROOT / "sessions" / session_id
    upload_dir = runtime / "uploads"
    ocr_md_dir = runtime / "ocr_md"
    chunk_dir = runtime / "chunks"
    vector_dir = runtime / "vector"
    state_file = runtime / "state.json"
    for p in (runtime, upload_dir, ocr_md_dir, chunk_dir, vector_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "runtime": runtime,
        "upload_dir": upload_dir,
        "ocr_md_dir": ocr_md_dir,
        "chunk_dir": chunk_dir,
        "vector_dir": vector_dir,
        "state_file": state_file,
    }


def _clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for item in list(path.iterdir()):
        if item.is_file():
            try:
                item.unlink(missing_ok=True)
            except PermissionError:
                time.sleep(0.2)
                try:
                    item.unlink(missing_ok=True)
                except PermissionError:
                    # Windows dosya kilidi devam ediyorsa sonraki rebuild'de tekrar denenecek.
                    continue
        elif item.is_dir():
            _clear_dir(item)
            try:
                item.rmdir()
            except OSError:
                continue


def _save_uploaded_files(uploaded_files: List[Any], upload_dir: Path) -> List[Path]:
    saved: List[Path] = []
    for f in uploaded_files:
        out = upload_dir / f.name
        out.write_bytes(f.getbuffer())
        saved.append(out)
    return saved


def _default_state() -> Dict[str, Any]:
    return {
        "messages": [],
        "ready": False,
        "docs": [],
        "last_error": "",
        "model_name": OLLAMA_MODEL,
    }


def _load_state(state_file: Path) -> Dict[str, Any]:
    if not state_file.exists():
        return _default_state()
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    base = _default_state()
    base.update(payload if isinstance(payload, dict) else {})
    return base


def _save_state(paths: Dict[str, Path]) -> None:
    payload = {
        "messages": st.session_state.get("messages", []),
        "ready": bool(st.session_state.get("ready", False)),
        "docs": st.session_state.get("docs", []),
        "last_error": st.session_state.get("last_error", ""),
        "model_name": OLLAMA_MODEL,
    }
    _atomic_write_json(paths["state_file"], payload)


def _prepare_documents(
    file_paths: List[Path],
    ocr_md_dir: Path,
    chunk_dir: Path,
    vector_dir: Path,
) -> Dict[str, Any]:
    _clear_dir(ocr_md_dir)
    _clear_dir(chunk_dir)
    # Windows'ta Chroma dosyalari process tarafindan kilitli kalabildigi icin
    # vector klasorunu fiziksel olarak silmiyoruz.
    # Yeniden kurulum collection seviyesinde build_vector_index icinde yapiliyor.

    failed: List[Dict[str, str]] = []
    written_md: List[str] = []
    ocr_gpu_enabled = _effective_ocr_gpu_enabled()
    for p in file_paths:
        try:
            markdown = process_document_to_markdown(p, use_gpu=ocr_gpu_enabled)
            md_path = ocr_md_dir / f"{p.stem}.md"
            md_path.write_text(markdown, encoding="utf-8")
            written_md.append(md_path.name)
        except Exception as exc:
            failed.append({"file": p.name, "error": str(exc)})

    if not written_md:
        return {
            "ok": False,
            "message": "Hicbir dosya OCR edilemedi.",
            "failed": failed,
        }

    run_pipeline(
        input_dir=ocr_md_dir,
        output_dir=chunk_dir,
        model_name=EMBED_MODEL_NAME,
        chunk_size=1200,
        chunk_overlap=200,
        min_chunk_size=120,
        device=EMBED_DEVICE,
        batch_size=32,
    )
    idx = build_vector_index(
        artifacts_dir=chunk_dir,
        persist_dir=vector_dir,
        collection_name=DEFAULT_COLLECTION,
        batch_size=256,
    )
    return {
        "ok": True,
        "written_md": written_md,
        "failed": failed,
        "indexed_chunk_count": idx.get("indexed_chunk_count", 0),
        "ocr_gpu_enabled": ocr_gpu_enabled,
    }


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("Kaynaklar"):
        for s in sources:
            st.markdown(
                f"- **{s.get('doc_id', '')}** | s{s.get('page', 0)} | `{s.get('chunk_id', '')}`\n"
                f"  - {s.get('text_preview', '')}"
            )


def main() -> None:
    st.set_page_config(page_title="Belge Asistani", page_icon=":page_facing_up:", layout="centered")

    session_id = _resolve_session_id()
    paths = _ensure_dirs(session_id)

    if "bootstrapped" not in st.session_state:
        state = _load_state(paths["state_file"])
        st.session_state.messages = list(state.get("messages", []))
        st.session_state.ready = bool(state.get("ready", False))
        st.session_state.docs = list(state.get("docs", []))
        st.session_state.last_error = str(state.get("last_error", ""))
        st.session_state.bootstrapped = True
    else:
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("ready", False)
        st.session_state.setdefault("docs", [])
        st.session_state.setdefault("last_error", "")

    st.title("Belge Asistani")
    st.caption("1) PDF/resim yukle  2) Belgeleri Hazirla  3) Soru sor")
    st.caption(f"Oturum: `{session_id}`")

    uploaded_files = st.file_uploader(
        "Belgeleri yukle",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if st.button("Belgeleri Hazirla", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Once en az bir dosya sec.")
        else:
            with st.spinner("OCR ve index hazirlaniyor..."):
                saved = _save_uploaded_files(uploaded_files, paths["upload_dir"])
                result = _prepare_documents(
                    file_paths=saved,
                    ocr_md_dir=paths["ocr_md_dir"],
                    chunk_dir=paths["chunk_dir"],
                    vector_dir=paths["vector_dir"],
                )
            st.session_state.docs = [{"name": p.name, "path": str(p), "status": "uploaded"} for p in saved]
            if not result["ok"]:
                st.session_state.ready = False
                st.session_state.last_error = result["message"]
                st.error(result["message"])
                for f in result.get("failed", []):
                    st.caption(f"{f['file']}: {f['error']}")
            else:
                st.session_state.ready = True
                st.session_state.last_error = ""
                failed_names = {f["file"] for f in result.get("failed", [])}
                st.session_state.docs = [
                    {
                        "name": p.name,
                        "path": str(p),
                        "status": "failed" if p.name in failed_names else "ready",
                    }
                    for p in saved
                ]
                st.success(
                    f"Hazir. Index chunk sayisi: {result.get('indexed_chunk_count', 0)} | "
                    f"OCR basarili dosya: {len(result.get('written_md', []))}"
                )
                st.caption(f"OCR cihazi: {'gpu' if result.get('ocr_gpu_enabled') else 'cpu'}")
                for f in result.get("failed", []):
                    st.caption(f"{f['file']}: {f['error']}")
            _save_state(paths)

    if st.session_state.docs:
        with st.expander("Aktif dokumanlar", expanded=False):
            for d in st.session_state.docs:
                st.markdown(f"- `{d.get('name','')}` ({d.get('status','unknown')})")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_sources(msg.get("sources", []))

    prompt = st.chat_input("Belgeyle ilgili sorunu yaz...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        _save_state(paths)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.ready:
                ans = "Once belgeleri hazirlaman gerekiyor."
                if st.session_state.last_error:
                    ans += f"\n\nSon hata: {st.session_state.last_error}"
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                _save_state(paths)
            else:
                with st.spinner("Cevap uretiliyor..."):
                    try:
                        result = ask_question(
                            question=prompt,
                            persist_dir=paths["vector_dir"],
                            collection_name=DEFAULT_COLLECTION,
                            initial_k=16,
                            final_k=5,
                            device=RETRIEVAL_DEVICE,
                            disable_rerank=False,
                            model_name=OLLAMA_MODEL,
                            strict_guardrail=True,
                        )
                        answer = str(result.get("answer", ""))
                        sources = list(result.get("sources", []) or [])
                        st.markdown(answer)
                        _render_sources(sources)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                        _save_state(paths)
                    except Exception as exc:
                        err = f"Bir hata oldu: {exc}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                        _save_state(paths)


if __name__ == "__main__":
    main()
