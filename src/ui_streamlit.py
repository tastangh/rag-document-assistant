"""Basit calisan UI: Yukle -> OCR -> Index -> Soru sor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from chunk_embedding_pipeline import run_pipeline
from document_processor import process_document_to_markdown
from generation_pipeline import ask_question
from retrieval_pipeline import DEFAULT_COLLECTION, build_vector_index


RUNTIME_ROOT = Path("src/results/ui_simple_runtime")


def _ensure_dirs() -> Dict[str, Path]:
    runtime = RUNTIME_ROOT
    upload_dir = runtime / "uploads"
    ocr_md_dir = runtime / "ocr_md"
    chunk_dir = runtime / "chunks"
    vector_dir = runtime / "vector"
    for p in (runtime, upload_dir, ocr_md_dir, chunk_dir, vector_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "runtime": runtime,
        "upload_dir": upload_dir,
        "ocr_md_dir": ocr_md_dir,
        "chunk_dir": chunk_dir,
        "vector_dir": vector_dir,
    }


def _clear_dir(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            _clear_dir(item)
            item.rmdir()


def _save_uploaded_files(uploaded_files: List[Any], upload_dir: Path) -> List[Path]:
    saved: List[Path] = []
    for f in uploaded_files:
        out = upload_dir / f.name
        out.write_bytes(f.getbuffer())
        saved.append(out)
    return saved


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
    for p in file_paths:
        try:
            markdown = process_document_to_markdown(p, use_gpu=False)
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
        model_name="BAAI/bge-m3",
        chunk_size=1200,
        chunk_overlap=200,
        min_chunk_size=120,
        device="auto",
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

    paths = _ensure_dirs()
    if "ready" not in st.session_state:
        st.session_state.ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Belge Asistani")
    st.caption("1) PDF/resim yukle  2) Belgeleri Hazirla  3) Soru sor")

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
            if not result["ok"]:
                st.session_state.ready = False
                st.error(result["message"])
                for f in result.get("failed", []):
                    st.caption(f"{f['file']}: {f['error']}")
            else:
                st.session_state.ready = True
                st.success(
                    f"Hazir. Index chunk sayisi: {result.get('indexed_chunk_count', 0)} | "
                    f"OCR basarili dosya: {len(result.get('written_md', []))}"
                )
                for f in result.get("failed", []):
                    st.caption(f"{f['file']}: {f['error']}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_sources(msg.get("sources", []))

    prompt = st.chat_input("Belgeyle ilgili sorunu yaz...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.ready:
                ans = "Once belgeleri hazirlaman gerekiyor."
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
            else:
                with st.spinner("Cevap uretiliyor..."):
                    try:
                        result = ask_question(
                            question=prompt,
                            persist_dir=paths["vector_dir"],
                            collection_name=DEFAULT_COLLECTION,
                            initial_k=16,
                            final_k=5,
                            device="auto",
                            disable_rerank=False,
                            model_name="qwen3:8b",
                            strict_guardrail=True,
                        )
                        answer = str(result.get("answer", ""))
                        sources = list(result.get("sources", []) or [])
                        st.markdown(answer)
                        _render_sources(sources)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as exc:
                        err = f"Bir hata oldu: {exc}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
