from __future__ import annotations

import os
import secrets
import time
import uuid

import streamlit as st

from config import EMBED_MODEL_NAME, OLLAMA_MODEL, RETRIEVAL_DEVICE
from generation import ask_question, chat_without_rag
from model_catalog import (
    EMBEDDING_MODELS,
    LLM_MODELS,
    MODEL_PRESETS,
    OCR_BACKEND_OPTIONS,
    OCR_DEVICE_MODE_OPTIONS,
    OCR_PROFILE_OPTIONS,
    RERANK_MODELS,
    resolve_local_hf_model,
)
from retrieval import DEFAULT_COLLECTION

from .rag import (
    auto_rag_params,
    classify_query_mode,
    doc_options,
    is_small_talk,
    ollama_health,
    prepare_documents,
    resolve_selected_doc_id,
    strip_inline_citations,
)
from .render import render_debug, render_sources
from .state import (
    cleanup_session,
    ensure_dirs,
    load_state,
    resolve_access_key,
    resolve_session_id,
    save_state,
    save_uploaded_files,
    validate_or_recover_session,
)

DEFAULT_MODELS = list(dict.fromkeys([OLLAMA_MODEL] + LLM_MODELS))


def _bootstrap_state(session_id: str, access_key: str, paths, recovered_state) -> None:
    if "bootstrapped" not in st.session_state:
        state = recovered_state if recovered_state else load_state(paths["state_file"])
        for k, v in state.items():
            st.session_state[k] = v
        st.session_state.session_id = session_id
        st.session_state.access_key = access_key
        st.session_state.bootstrapped = True
        st.session_state.warmup_done = False
        return
    defaults = {
        "messages": [],
        "ready": False,
        "docs": [],
        "last_error": "",
        "model_name": OLLAMA_MODEL,
        "system_instructions": "",
        "temperature": 0.35,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "thinking_level": "medium",
        "strict_guardrail": True,
        "fast_mode": True,
        "initial_k": 16,
        "final_k": 5,
        "retrieval_min_overlap": 0.05,
        "guardrail_threshold": 0.50,
        "citation_min_coverage": 0.8,
        "creativity": 35,
        "log_level": 1,
        "doc_filter_mode": "auto",
        "manual_doc_id": "",
        "preset_id": "tr_balanced",
        "embedding_model": EMBED_MODEL_NAME,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "ocr_profile": "default",
        "ocr_backend": "paddle",
        "ocr_device_mode": "cpu",
        "session_id": session_id,
        "access_key": access_key,
        "warmup_done": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _render_sidebar() -> None:
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
            st.session_state.ocr_profile = picked_preset["ocr_profile"]

        model_options = list(dict.fromkeys(DEFAULT_MODELS + [str(st.session_state.model_name)]))
        current_model = st.selectbox("Model", options=model_options, index=max(0, model_options.index(st.session_state.model_name)) if st.session_state.model_name in model_options else 0)
        custom_model = st.text_input("Custom model (optional)", value="")
        st.session_state.model_name = custom_model.strip() or current_model
        st.session_state.embedding_model = st.selectbox("Embedding model", options=list(dict.fromkeys(EMBEDDING_MODELS + [str(st.session_state.embedding_model)])), index=list(dict.fromkeys(EMBEDDING_MODELS + [str(st.session_state.embedding_model)])).index(str(st.session_state.embedding_model)))
        st.session_state.reranker_model = st.selectbox("Reranker model", options=list(dict.fromkeys(RERANK_MODELS + [str(st.session_state.reranker_model)])), index=list(dict.fromkeys(RERANK_MODELS + [str(st.session_state.reranker_model)])).index(str(st.session_state.reranker_model)))
        st.session_state.ocr_profile = st.selectbox("OCR profile", options=OCR_PROFILE_OPTIONS, index=OCR_PROFILE_OPTIONS.index(st.session_state.ocr_profile) if st.session_state.ocr_profile in OCR_PROFILE_OPTIONS else 0)
        st.session_state.ocr_backend = st.selectbox("OCR backend", options=OCR_BACKEND_OPTIONS, index=OCR_BACKEND_OPTIONS.index(st.session_state.ocr_backend) if str(st.session_state.ocr_backend) in OCR_BACKEND_OPTIONS else 0)
        st.session_state.ocr_device_mode = st.selectbox("OCR device mode", options=OCR_DEVICE_MODE_OPTIONS, index=OCR_DEVICE_MODE_OPTIONS.index(st.session_state.ocr_device_mode) if str(st.session_state.ocr_device_mode) in OCR_DEVICE_MODE_OPTIONS else 0)

        st.session_state.system_instructions = st.text_area("System instructions", value=st.session_state.system_instructions, height=110, placeholder="Model davranisi icin ek talimat yazabilirsin...")
        st.session_state.temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, step=0.1, value=float(st.session_state.temperature))
        st.session_state.top_k = int(st.number_input("Top-k (generation)", min_value=1, max_value=200, value=int(st.session_state.top_k), step=1))
        st.session_state.top_p = float(st.slider("Top-p (generation)", min_value=0.0, max_value=1.0, step=0.01, value=float(st.session_state.top_p)))
        st.session_state.repeat_penalty = float(st.slider("Repeat penalty", min_value=1.0, max_value=2.0, step=0.05, value=float(st.session_state.repeat_penalty)))
        st.session_state.creativity = int(st.slider("Yaraticilik", min_value=0, max_value=100, step=1, value=int(st.session_state.creativity)))
        st.session_state.thinking_level = st.selectbox("Thinking level", options=["low", "medium", "high"], index=["low", "medium", "high"].index(str(st.session_state.thinking_level)) if str(st.session_state.thinking_level) in {"low", "medium", "high"} else 1)
        st.markdown("---")
        st.subheader("RAG controls")
        st.session_state.strict_guardrail = st.toggle("Strict guardrail", value=bool(st.session_state.strict_guardrail))
        st.session_state.fast_mode = st.toggle("Fast mode", value=bool(st.session_state.fast_mode))
        st.session_state.initial_k = int(st.number_input("Initial k", min_value=1, max_value=64, value=int(st.session_state.initial_k), step=1))
        st.session_state.final_k = int(st.number_input("Final k", min_value=1, max_value=16, value=int(st.session_state.final_k), step=1))
        st.session_state.retrieval_min_overlap = float(st.slider("Retrieval min overlap", min_value=0.0, max_value=0.3, value=float(st.session_state.retrieval_min_overlap), step=0.01))
        st.session_state.guardrail_threshold = float(st.slider("Halusinasyon koruma hassasiyeti", min_value=0.0, max_value=1.0, value=float(st.session_state.guardrail_threshold), step=0.01))
        st.session_state.citation_min_coverage = float(st.slider("Citation coverage min", min_value=0.0, max_value=1.0, value=float(st.session_state.citation_min_coverage), step=0.05))
        st.session_state.log_level = int(st.number_input("C/Debug log seviyesi (0-3)", min_value=0, max_value=3, value=int(st.session_state.log_level), step=1))
        st.session_state.doc_filter_mode = st.selectbox("Doc filter mode", options=["auto", "manual"], index=0 if st.session_state.doc_filter_mode == "auto" else 1)
        docs = doc_options()
        if st.session_state.doc_filter_mode == "manual" and docs:
            labels = [name for name, _ in docs]
            ids = [did for _, did in docs]
            default_idx = ids.index(st.session_state.manual_doc_id) if st.session_state.manual_doc_id in ids else 0
            picked = st.selectbox("Manual doc", options=labels, index=default_idx)
            st.session_state.manual_doc_id = ids[labels.index(picked)]
        elif st.session_state.doc_filter_mode == "manual":
            st.caption("Manual doc secimi icin once belgeleri hazirla.")
            st.session_state.manual_doc_id = ""
        else:
            st.session_state.manual_doc_id = ""
        ok, msg = ollama_health()
        st.caption(f"Model hazir mi?: {'Evet' if ok else 'Hayir'} ({msg})")


def main() -> None:
    st.set_page_config(page_title="Belge Asistani", page_icon=":page_facing_up:", layout="centered")
    session_id = resolve_session_id()
    access_key = resolve_access_key()
    paths = ensure_dirs(session_id)
    session_id, access_key, recovered_state = validate_or_recover_session(session_id, access_key, paths)
    paths = ensure_dirs(session_id)
    _bootstrap_state(session_id, access_key, paths, recovered_state)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("GLOG_minloglevel", str(int(st.session_state.get("log_level", 1))))

    _render_sidebar()
    st.title("Belge Asistani")
    st.caption("1) PDF/resim yukle  2) Belgeleri Hazirla  3) Soru sor")
    st.caption(f"Oturum: `{session_id}`")
    st.caption(f"Run fingerprint: llm={st.session_state.model_name} | emb={st.session_state.embedding_model} | rerank={st.session_state.reranker_model} | temp={st.session_state.temperature:.1f} | strict={st.session_state.strict_guardrail} | fast={st.session_state.fast_mode}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset chat only", use_container_width=True):
            st.session_state.messages = []
            save_state(paths, session_id, access_key)
            st.rerun()
    with c2:
        if st.button("Reset index + chat", use_container_width=True):
            cleanup_session(paths["runtime"])
            st.session_state.clear()
            st.query_params["sid"] = str(uuid.uuid4())
            st.query_params["sk"] = secrets.token_urlsafe(24)
            st.rerun()

    uploaded_files = st.file_uploader("Belgeleri yukle", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Belgeleri Hazirla", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Once en az bir dosya sec.")
        else:
            with st.spinner("OCR ve index hazirlaniyor..."):
                saved = save_uploaded_files(uploaded_files, paths["upload_dir"])
                result = prepare_documents(saved, paths["ocr_md_dir"], paths["chunk_dir"], paths["vector_dir"], resolve_local_hf_model(str(st.session_state.embedding_model), "embedding"), str(st.session_state.ocr_profile))
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
                st.session_state.docs = [{"name": p.name, "path": str(p), "status": "failed" if p.name in failed_names else "ready"} for p in saved]
                st.success(f"Hazir. Index chunk sayisi: {result.get('indexed_chunk_count', 0)} | OCR basarili dosya: {len(result.get('written_md', []))}")
                st.session_state.warmup_done = False
            save_state(paths, session_id, access_key)

    if st.session_state.docs:
        with st.expander("Aktif dokumanlar", expanded=False):
            for d in st.session_state.docs:
                st.markdown(f"- `{d.get('name','')}` ({d.get('status','unknown')})")

    selected_doc_id = resolve_selected_doc_id(str(st.session_state.doc_filter_mode), str(st.session_state.manual_doc_id))
    if selected_doc_id:
        st.caption(f"Aktif doc_id filtresi: `{selected_doc_id}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))
            render_debug(msg.get("details", {}))

    prompt = st.chat_input("Belgeyle ilgili sorunu yaz...")
    if not prompt:
        return
    query_mode = classify_query_mode(prompt, rag_ready=bool(st.session_state.ready))
    auto_params = auto_rag_params(query_mode)
    creativity_ratio = max(0.0, min(1.0, float(st.session_state.creativity) / 100.0))
    base_temperature = max(float(st.session_state.temperature), round(0.2 + 1.0 * creativity_ratio, 2))
    effective_temperature = min(float(auto_params.get("temperature_cap", 1.5)), base_temperature + float(auto_params.get("temperature_boost", 0.0)))
    effective_top_p = max(float(st.session_state.top_p), round(0.5 + 0.5 * creativity_ratio, 2), float(auto_params.get("top_p_floor", 0.85)))
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_state(paths, session_id, access_key)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if query_mode == "chat" or is_small_talk(prompt):
                t0 = time.perf_counter()
                result = chat_without_rag(
                    question=prompt,
                    model_name=st.session_state.model_name,
                    system_instructions=st.session_state.system_instructions,
                    thinking_level=st.session_state.thinking_level,
                    temperature=float(effective_temperature),
                    top_k=int(st.session_state.top_k),
                    top_p=float(effective_top_p),
                    repeat_penalty=float(st.session_state.repeat_penalty),
                )
                latency = time.perf_counter() - t0
                ans = str(result.get("answer", "")).strip() or "Merhaba, buradayim."
                details = {"latency_sec": round(latency, 3), "sources_count": 0, "mode": "chat_without_rag", "query_mode": query_mode}
                st.markdown(ans)
                render_debug(details)
                st.session_state.messages.append({"role": "assistant", "content": ans, "details": details})
                save_state(paths, session_id, access_key)
                return

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
                    strict_guardrail=bool(auto_params.get("strict_guardrail", st.session_state.strict_guardrail)),
                    fast_mode=bool(st.session_state.fast_mode),
                    doc_id=selected_doc_id,
                    system_instructions=st.session_state.system_instructions,
                    temperature=float(effective_temperature),
                    thinking_level=st.session_state.thinking_level,
                    top_k=int(st.session_state.top_k),
                    top_p=float(effective_top_p),
                    repeat_penalty=float(st.session_state.repeat_penalty),
                    guardrail_threshold=float(auto_params.get("guardrail_threshold", st.session_state.guardrail_threshold)),
                    citation_min_coverage=float(auto_params.get("citation_min_coverage", st.session_state.citation_min_coverage)),
                    allow_extractive_on_guardrail_fail=bool(auto_params.get("allow_extractive_on_guardrail_fail", False)),
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
                strict_guardrail=bool(auto_params.get("strict_guardrail", st.session_state.strict_guardrail)),
                fast_mode=bool(st.session_state.fast_mode),
                context_limit=4,
                doc_id=selected_doc_id,
                retrieval_min_overlap=float(st.session_state.retrieval_min_overlap),
                system_instructions=st.session_state.system_instructions,
                temperature=float(effective_temperature),
                thinking_level=st.session_state.thinking_level,
                top_k=int(st.session_state.top_k),
                top_p=float(effective_top_p),
                repeat_penalty=float(st.session_state.repeat_penalty),
                guardrail_threshold=float(auto_params.get("guardrail_threshold", st.session_state.guardrail_threshold)),
                citation_min_coverage=float(auto_params.get("citation_min_coverage", st.session_state.citation_min_coverage)),
                allow_extractive_on_guardrail_fail=bool(auto_params.get("allow_extractive_on_guardrail_fail", False)),
            )
            latency = time.perf_counter() - t0
            answer = str(result.get("answer", ""))
            answer_for_ui = strip_inline_citations(answer)
            sources = list(result.get("sources", []) or [])
            verification = result.get("verification", {}) or {}
            details = {
                "latency_sec": round(latency, 3),
                "sources_count": len(sources),
                "fallback_used": bool(verification.get("fallback_used", False)),
                "supported_ratio": verification.get("supported_ratio"),
                "confidence": verification.get("confidence_level", verification.get("confidence")),
                "query_mode": query_mode,
            }
            st.markdown(answer_for_ui or answer)
            render_sources(sources)
            render_debug(details)
            st.session_state.messages.append({"role": "assistant", "content": (answer_for_ui or answer), "sources": sources, "details": details})
            save_state(paths, session_id, access_key)
        except Exception as exc:
            err = f"Bir hata oldu: {exc}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            save_state(paths, session_id, access_key)

