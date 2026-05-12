from __future__ import annotations

import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from chunk_embedding_pipeline import run_pipeline
from config import (
    EMBED_DEVICE,
    EMBED_MODEL_NAME,
    OCR_USE_GPU,
    OLLAMA_MODEL,
    RETRIEVAL_DEVICE,
    SESSION_ROOT,
)
from document_processor import process_document_to_markdown
from generation_pipeline import ask_question
from model_catalog import (
    EMBEDDING_MODELS,
    LLM_MODELS,
    MODEL_PRESETS,
    OCR_LANG_OPTIONS,
    OCR_PROFILE_OPTIONS,
    RERANK_MODELS,
    resolve_local_hf_model,
)
from retrieval_pipeline import DEFAULT_COLLECTION, build_vector_index


DEFAULT_MODELS = list(dict.fromkeys([OLLAMA_MODEL] + LLM_MODELS))


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
    uuid_v4_re = re.compile(
        r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$",
        flags=re.IGNORECASE,
    )
    if not sid or not uuid_v4_re.match(sid):
        sid = str(uuid.uuid4())
        st.query_params["sid"] = sid
    return sid


def _resolve_access_key() -> str:
    key = str(st.query_params.get("sk", "")).strip()
    if not key:
        key = secrets.token_urlsafe(24)
        st.query_params["sk"] = key
    return key


def _ensure_dirs(session_id: str) -> Dict[str, Path]:
    runtime = SESSION_ROOT / session_id
    upload_dir = runtime / "uploads"
    ocr_md_dir = runtime / "ocr_md"
    chunk_dir = runtime / "chunks"
    vector_dir = runtime / "vector"
    state_file = runtime / f"state_{session_id}.json"
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
        "system_instructions": "",
        "temperature": 0.0,
        "thinking_level": "medium",
        "strict_guardrail": True,
        "fast_mode": True,
        "initial_k": 12,
        "final_k": 4,
        "retrieval_min_overlap": 0.08,
        "doc_filter_mode": "auto",
        "manual_doc_id": "",
        "preset_id": "tr_balanced",
        "embedding_model": EMBED_MODEL_NAME,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "ocr_lang": "tr",
        "ocr_profile": "default",
        "session_id": "",
        "access_key": "",
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


def _save_state(paths: Dict[str, Path], session_id: str, access_key: str) -> None:
    payload = {
        "messages": st.session_state.get("messages", []),
        "ready": bool(st.session_state.get("ready", False)),
        "docs": st.session_state.get("docs", []),
        "last_error": st.session_state.get("last_error", ""),
        "model_name": st.session_state.get("model_name", OLLAMA_MODEL),
        "system_instructions": st.session_state.get("system_instructions", ""),
        "temperature": float(st.session_state.get("temperature", 0.0)),
        "thinking_level": st.session_state.get("thinking_level", "medium"),
        "strict_guardrail": bool(st.session_state.get("strict_guardrail", True)),
        "fast_mode": bool(st.session_state.get("fast_mode", True)),
        "initial_k": int(st.session_state.get("initial_k", 12)),
        "final_k": int(st.session_state.get("final_k", 4)),
        "retrieval_min_overlap": float(st.session_state.get("retrieval_min_overlap", 0.08)),
        "doc_filter_mode": st.session_state.get("doc_filter_mode", "auto"),
        "manual_doc_id": st.session_state.get("manual_doc_id", ""),
        "preset_id": st.session_state.get("preset_id", "tr_balanced"),
        "embedding_model": st.session_state.get("embedding_model", EMBED_MODEL_NAME),
        "reranker_model": st.session_state.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        "ocr_lang": st.session_state.get("ocr_lang", "tr"),
        "ocr_profile": st.session_state.get("ocr_profile", "default"),
        "session_id": session_id,
        "access_key": access_key,
    }
    _atomic_write_json(paths["state_file"], payload)


def _cleanup_session(runtime_root: Path) -> None:
    resolved_root = SESSION_ROOT.resolve()
    resolved_target = runtime_root.resolve()
    if resolved_root not in resolved_target.parents and resolved_target != resolved_root:
        raise RuntimeError("Guvenlik hatasi: Session dizini beklenen kok altinda degil.")
    _clear_dir(resolved_target)
    try:
        resolved_target.rmdir()
    except OSError:
        pass


def _validate_or_recover_session(session_id: str, access_key: str, paths: Dict[str, Path]) -> tuple[str, str, Dict[str, Any]]:
    loaded = _load_state(paths["state_file"])
    stored_sid = str(loaded.get("session_id", "")).strip()
    stored_key = str(loaded.get("access_key", "")).strip()

    if not stored_sid and not stored_key:
        loaded["session_id"] = session_id
        loaded["access_key"] = access_key
        return session_id, access_key, loaded

    if stored_sid == session_id and stored_key and stored_key == access_key:
        return session_id, access_key, loaded

    new_sid = str(uuid.uuid4())
    new_key = secrets.token_urlsafe(24)
    st.query_params["sid"] = new_sid
    st.query_params["sk"] = new_key
    new_paths = _ensure_dirs(new_sid)
    new_state = _default_state()
    new_state["session_id"] = new_sid
    new_state["access_key"] = new_key
    _atomic_write_json(new_paths["state_file"], new_state)
    st.warning("Oturum kimligi dogrulanamadi. Guvenlik nedeniyle yeni izole oturum olusturuldu.")
    return new_sid, new_key, new_state


def _prepare_documents(
    file_paths: List[Path],
    ocr_md_dir: Path,
    chunk_dir: Path,
    vector_dir: Path,
    embed_model_name: str,
    ocr_lang: str,
    ocr_profile: str,
) -> Dict[str, Any]:
    _clear_dir(ocr_md_dir)
    _clear_dir(chunk_dir)

    failed: List[Dict[str, str]] = []
    written_md: List[str] = []
    ocr_gpu_enabled = _effective_ocr_gpu_enabled()
    for p in file_paths:
        try:
            markdown = process_document_to_markdown(
                p,
                use_gpu=ocr_gpu_enabled,
                ocr_lang=ocr_lang,
                ocr_profile=ocr_profile,
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
        "embed_model_name": embed_model_name,
        "ocr_lang": ocr_lang,
        "ocr_profile": ocr_profile,
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


def _render_debug(details: Dict[str, Any]) -> None:
    if not details:
        return
    with st.expander("Debug / Details", expanded=False):
        st.json(details)


def _doc_id_from_doc_entry(doc: Dict[str, Any]) -> Optional[str]:
    path = str(doc.get("path", "")).strip()
    if not path:
        return None
    try:
        return Path(path).stem
    except Exception:
        return None


def _doc_options() -> List[tuple[str, str]]:
    ready_docs = [d for d in st.session_state.get("docs", []) if d.get("status") == "ready"]
    options: List[tuple[str, str]] = []
    for d in ready_docs:
        did = _doc_id_from_doc_entry(d)
        if did:
            options.append((str(d.get("name", did)), did))
    return options


def _resolve_selected_doc_id(mode: str, manual_doc_id: str) -> Optional[str]:
    if mode == "manual":
        return manual_doc_id.strip() or None
    return None


def _ollama_health(ollama_url: str = "http://localhost:11434/api/generate") -> tuple[bool, str]:
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


def main() -> None:
    st.set_page_config(page_title="Belge Asistani", page_icon=":page_facing_up:", layout="centered")

    session_id = _resolve_session_id()
    access_key = _resolve_access_key()
    paths = _ensure_dirs(session_id)
    session_id, access_key, recovered_state = _validate_or_recover_session(
        session_id=session_id,
        access_key=access_key,
        paths=paths,
    )
    paths = _ensure_dirs(session_id)

    if "bootstrapped" not in st.session_state:
        state = recovered_state if recovered_state else _load_state(paths["state_file"])
        st.session_state.messages = list(state.get("messages", []))
        st.session_state.ready = bool(state.get("ready", False))
        st.session_state.docs = list(state.get("docs", []))
        st.session_state.last_error = str(state.get("last_error", ""))
        st.session_state.model_name = str(state.get("model_name", OLLAMA_MODEL))
        st.session_state.system_instructions = str(state.get("system_instructions", ""))
        st.session_state.temperature = float(state.get("temperature", 0.0))
        st.session_state.thinking_level = str(state.get("thinking_level", "medium"))
        st.session_state.strict_guardrail = bool(state.get("strict_guardrail", True))
        st.session_state.fast_mode = bool(state.get("fast_mode", True))
        st.session_state.initial_k = int(state.get("initial_k", 12))
        st.session_state.final_k = int(state.get("final_k", 4))
        st.session_state.retrieval_min_overlap = float(state.get("retrieval_min_overlap", 0.08))
        st.session_state.doc_filter_mode = str(state.get("doc_filter_mode", "auto"))
        st.session_state.manual_doc_id = str(state.get("manual_doc_id", ""))
        st.session_state.preset_id = str(state.get("preset_id", "tr_balanced"))
        st.session_state.embedding_model = str(state.get("embedding_model", EMBED_MODEL_NAME))
        st.session_state.reranker_model = str(state.get("reranker_model", "BAAI/bge-reranker-v2-m3"))
        st.session_state.ocr_lang = str(state.get("ocr_lang", "tr"))
        st.session_state.ocr_profile = str(state.get("ocr_profile", "default"))
        st.session_state.session_id = session_id
        st.session_state.access_key = access_key
        st.session_state.bootstrapped = True
        st.session_state.warmup_done = False
    else:
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("ready", False)
        st.session_state.setdefault("docs", [])
        st.session_state.setdefault("last_error", "")
        st.session_state.setdefault("model_name", OLLAMA_MODEL)
        st.session_state.setdefault("system_instructions", "")
        st.session_state.setdefault("temperature", 0.0)
        st.session_state.setdefault("thinking_level", "medium")
        st.session_state.setdefault("strict_guardrail", True)
        st.session_state.setdefault("fast_mode", True)
        st.session_state.setdefault("initial_k", 12)
        st.session_state.setdefault("final_k", 4)
        st.session_state.setdefault("retrieval_min_overlap", 0.08)
        st.session_state.setdefault("doc_filter_mode", "auto")
        st.session_state.setdefault("manual_doc_id", "")
        st.session_state.setdefault("preset_id", "tr_balanced")
        st.session_state.setdefault("embedding_model", EMBED_MODEL_NAME)
        st.session_state.setdefault("reranker_model", "BAAI/bge-reranker-v2-m3")
        st.session_state.setdefault("ocr_lang", "tr")
        st.session_state.setdefault("ocr_profile", "default")
        st.session_state.setdefault("session_id", session_id)
        st.session_state.setdefault("access_key", access_key)
        st.session_state.setdefault("warmup_done", False)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    with st.sidebar:
        st.subheader("Run settings")
        preset_labels = [p["label"] for p in MODEL_PRESETS]
        preset_ids = [p["id"] for p in MODEL_PRESETS]
        preset_idx = preset_ids.index(st.session_state.preset_id) if st.session_state.preset_id in preset_ids else 0
        picked_preset_label = st.selectbox("Preset", options=preset_labels, index=preset_idx)
        picked_preset = MODEL_PRESETS[preset_labels.index(picked_preset_label)]
        if picked_preset["id"] != st.session_state.preset_id:
            st.session_state.preset_id = picked_preset["id"]
            st.session_state.model_name = picked_preset["llm_model"]
            st.session_state.embedding_model = picked_preset["embedding_model"]
            st.session_state.reranker_model = picked_preset["reranker_model"]
            st.session_state.ocr_lang = picked_preset["ocr_lang"]
            st.session_state.ocr_profile = picked_preset["ocr_profile"]

        model_options = list(dict.fromkeys(DEFAULT_MODELS + [str(st.session_state.model_name)]))
        current_model = st.selectbox(
            "Model",
            options=model_options,
            index=max(0, model_options.index(st.session_state.model_name)) if st.session_state.model_name in model_options else 0,
        )
        custom_model = st.text_input("Custom model (optional)", value="")
        st.session_state.model_name = custom_model.strip() or current_model
        st.session_state.embedding_model = st.selectbox(
            "Embedding model",
            options=list(dict.fromkeys(EMBEDDING_MODELS + [str(st.session_state.embedding_model)])),
            index=(
                list(dict.fromkeys(EMBEDDING_MODELS + [str(st.session_state.embedding_model)])).index(
                    str(st.session_state.embedding_model)
                )
            ),
        )
        st.session_state.reranker_model = st.selectbox(
            "Reranker model",
            options=list(dict.fromkeys(RERANK_MODELS + [str(st.session_state.reranker_model)])),
            index=(
                list(dict.fromkeys(RERANK_MODELS + [str(st.session_state.reranker_model)])).index(
                    str(st.session_state.reranker_model)
                )
            ),
        )
        st.session_state.ocr_lang = st.selectbox(
            "OCR language",
            options=OCR_LANG_OPTIONS,
            index=OCR_LANG_OPTIONS.index(st.session_state.ocr_lang) if st.session_state.ocr_lang in OCR_LANG_OPTIONS else 0,
        )
        st.session_state.ocr_profile = st.selectbox(
            "OCR profile",
            options=OCR_PROFILE_OPTIONS,
            index=(
                OCR_PROFILE_OPTIONS.index(st.session_state.ocr_profile)
                if st.session_state.ocr_profile in OCR_PROFILE_OPTIONS
                else 0
            ),
        )

        st.session_state.system_instructions = st.text_area(
            "System instructions",
            value=st.session_state.system_instructions,
            height=110,
            placeholder="Model davranisi icin ek talimat yazabilirsin...",
        )
        st.session_state.temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.5, step=0.1, value=float(st.session_state.temperature)
        )
        st.session_state.thinking_level = st.selectbox(
            "Thinking level",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(str(st.session_state.thinking_level)) if str(st.session_state.thinking_level) in {"low", "medium", "high"} else 1,
        )

        st.markdown("---")
        st.subheader("RAG controls")
        st.session_state.strict_guardrail = st.toggle("Strict guardrail", value=bool(st.session_state.strict_guardrail))
        st.session_state.fast_mode = st.toggle("Fast mode", value=bool(st.session_state.fast_mode))
        st.session_state.initial_k = int(st.number_input("Initial k", min_value=1, max_value=64, value=int(st.session_state.initial_k), step=1))
        st.session_state.final_k = int(st.number_input("Final k", min_value=1, max_value=16, value=int(st.session_state.final_k), step=1))
        st.session_state.retrieval_min_overlap = float(
            st.slider("Retrieval min overlap", min_value=0.0, max_value=0.3, value=float(st.session_state.retrieval_min_overlap), step=0.01)
        )

        mode_label = st.selectbox(
            "Doc filter mode",
            options=["auto", "manual"],
            index=0 if st.session_state.doc_filter_mode == "auto" else 1,
        )
        st.session_state.doc_filter_mode = mode_label

        docs = _doc_options()
        if mode_label == "manual" and docs:
            labels = [name for name, _ in docs]
            ids = [did for _, did in docs]
            default_idx = ids.index(st.session_state.manual_doc_id) if st.session_state.manual_doc_id in ids else 0
            picked = st.selectbox("Manual doc", options=labels, index=default_idx)
            st.session_state.manual_doc_id = ids[labels.index(picked)]
        elif mode_label == "manual":
            st.caption("Manual doc secimi icin once belgeleri hazirla.")
            st.session_state.manual_doc_id = ""
        else:
            st.session_state.manual_doc_id = ""

        ok, msg = _ollama_health()
        st.caption(f"Model hazir mi?: {'Evet' if ok else 'Hayir'} ({msg})")

    st.title("Belge Asistani")
    st.caption("1) PDF/resim yukle  2) Belgeleri Hazirla  3) Soru sor")
    st.caption(f"Oturum: `{session_id}`")
    st.caption(
        "Run fingerprint: "
        f"llm={st.session_state.model_name} | emb={st.session_state.embedding_model} | "
        f"rerank={st.session_state.reranker_model} | temp={st.session_state.temperature:.1f} | "
        f"strict={st.session_state.strict_guardrail} | fast={st.session_state.fast_mode}"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset chat only", use_container_width=True):
            st.session_state.messages = []
            _save_state(paths, session_id=session_id, access_key=access_key)
            st.rerun()
    with c2:
        if st.button("Reset index + chat", use_container_width=True):
            _cleanup_session(paths["runtime"])
            st.session_state.clear()
            fresh_sid = str(uuid.uuid4())
            fresh_key = secrets.token_urlsafe(24)
            st.query_params["sid"] = fresh_sid
            st.query_params["sk"] = fresh_key
            st.rerun()

    uploaded_files = st.file_uploader("Belgeleri yukle", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

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
                    embed_model_name=resolve_local_hf_model(str(st.session_state.embedding_model), "embedding"),
                    ocr_lang=str(st.session_state.ocr_lang),
                    ocr_profile=str(st.session_state.ocr_profile),
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
                    {"name": p.name, "path": str(p), "status": "failed" if p.name in failed_names else "ready"}
                    for p in saved
                ]
                st.success(
                    f"Hazir. Index chunk sayisi: {result.get('indexed_chunk_count', 0)} | "
                    f"OCR basarili dosya: {len(result.get('written_md', []))}"
                )
                st.caption(
                    f"OCR cihazi: {'gpu' if result.get('ocr_gpu_enabled') else 'cpu'} | "
                    f"lang={result.get('ocr_lang')} | profile={result.get('ocr_profile')} | "
                    f"embedding={result.get('embed_model_name')}"
                )
                for f in result.get("failed", []):
                    st.caption(f"{f['file']}: {f['error']}")
                st.session_state.warmup_done = False
            _save_state(paths, session_id=session_id, access_key=access_key)

    if st.session_state.docs:
        with st.expander("Aktif dokumanlar", expanded=False):
            for d in st.session_state.docs:
                st.markdown(f"- `{d.get('name','')}` ({d.get('status','unknown')})")

    selected_doc_id = _resolve_selected_doc_id(
        mode=str(st.session_state.doc_filter_mode),
        manual_doc_id=str(st.session_state.manual_doc_id),
    )
    if selected_doc_id:
        st.caption(f"Aktif doc_id filtresi: `{selected_doc_id}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_sources(msg.get("sources", []))
            _render_debug(msg.get("details", {}))

    prompt = st.chat_input("Belgeyle ilgili sorunu yaz...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        _save_state(paths, session_id=session_id, access_key=access_key)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.ready:
                ans = "Once belgeleri hazirlaman gerekiyor."
                if st.session_state.last_error:
                    ans += f"\n\nSon hata: {st.session_state.last_error}"
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                _save_state(paths, session_id=session_id, access_key=access_key)
            else:
                with st.spinner("Cevap uretiliyor..."):
                    try:
                        if not st.session_state.get("warmup_done", False):
                            _ = ask_question(
                                question="dokuman ozeti",
                                persist_dir=paths["vector_dir"],
                                collection_name=DEFAULT_COLLECTION,
                                initial_k=8,
                                final_k=3,
                                device=RETRIEVAL_DEVICE,
                                disable_rerank=True,
                                model_name=st.session_state.model_name,
                                reranker_model=resolve_local_hf_model(str(st.session_state.reranker_model), "reranker"),
                                strict_guardrail=bool(st.session_state.strict_guardrail),
                                fast_mode=bool(st.session_state.fast_mode),
                                doc_id=selected_doc_id,
                                system_instructions=st.session_state.system_instructions,
                                temperature=float(st.session_state.temperature),
                                thinking_level=st.session_state.thinking_level,
                            )
                            st.session_state.warmup_done = True

                        t0 = time.perf_counter()
                        result = ask_question(
                            question=prompt,
                            persist_dir=paths["vector_dir"],
                            collection_name=DEFAULT_COLLECTION,
                            initial_k=int(st.session_state.initial_k),
                            final_k=int(st.session_state.final_k),
                            device=RETRIEVAL_DEVICE,
                            disable_rerank=False,
                            model_name=st.session_state.model_name,
                            reranker_model=resolve_local_hf_model(str(st.session_state.reranker_model), "reranker"),
                            strict_guardrail=bool(st.session_state.strict_guardrail),
                            fast_mode=bool(st.session_state.fast_mode),
                            context_limit=4,
                            doc_id=selected_doc_id,
                            retrieval_min_overlap=float(st.session_state.retrieval_min_overlap),
                            system_instructions=st.session_state.system_instructions,
                            temperature=float(st.session_state.temperature),
                            thinking_level=st.session_state.thinking_level,
                        )
                        latency = time.perf_counter() - t0

                        answer = str(result.get("answer", ""))
                        sources = list(result.get("sources", []) or [])
                        verification = result.get("verification", {}) or {}
                        details = {
                            "latency_sec": round(latency, 3),
                            "sources_count": len(sources),
                            "fallback_used": bool(verification.get("fallback_used", False)),
                            "supported_ratio": verification.get("supported_ratio"),
                            "confidence": verification.get("confidence_level", verification.get("confidence")),
                        }

                        st.markdown(answer)
                        _render_sources(sources)
                        _render_debug(details)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "details": details,
                            }
                        )
                        _save_state(paths, session_id=session_id, access_key=access_key)
                    except Exception as exc:
                        err = f"Bir hata oldu: {exc}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                        _save_state(paths, session_id=session_id, access_key=access_key)


if __name__ == "__main__":
    main()
