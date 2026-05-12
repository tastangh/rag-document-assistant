from __future__ import annotations

import json
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from config import EMBED_MODEL_NAME, OLLAMA_MODEL, SESSION_ROOT


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_session_id() -> str:
    sid = str(st.query_params.get("sid", "")).strip()
    uuid_v4_re = re.compile(
        r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$",
        flags=re.IGNORECASE,
    )
    if not sid or not uuid_v4_re.match(sid):
        sid = str(uuid.uuid4())
        st.query_params["sid"] = sid
    return sid


def resolve_access_key() -> str:
    key = str(st.query_params.get("sk", "")).strip()
    if not key:
        key = secrets.token_urlsafe(24)
        st.query_params["sk"] = key
    return key


def ensure_dirs(session_id: str) -> Dict[str, Path]:
    runtime = SESSION_ROOT / session_id
    paths = {
        "runtime": runtime,
        "upload_dir": runtime / "uploads",
        "ocr_md_dir": runtime / "ocr_md",
        "chunk_dir": runtime / "chunks",
        "vector_dir": runtime / "vector",
        "state_file": runtime / f"state_{session_id}.json",
    }
    for p in paths.values():
        if isinstance(p, Path) and p.suffix == "":
            p.mkdir(parents=True, exist_ok=True)
    return paths


def clear_dir(path: Path) -> None:
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
            clear_dir(item)
            try:
                item.rmdir()
            except OSError:
                continue


def default_state() -> Dict[str, Any]:
    return {
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
        "session_id": "",
        "access_key": "",
    }


def load_state(state_file: Path) -> Dict[str, Any]:
    if not state_file.exists():
        return default_state()
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return default_state()
    base = default_state()
    base.update(payload if isinstance(payload, dict) else {})
    return base


def save_state(paths: Dict[str, Path], session_id: str, access_key: str) -> None:
    payload = default_state()
    payload.update(
        {
            "messages": st.session_state.get("messages", []),
            "ready": bool(st.session_state.get("ready", False)),
            "docs": st.session_state.get("docs", []),
            "last_error": st.session_state.get("last_error", ""),
            "model_name": st.session_state.get("model_name", OLLAMA_MODEL),
            "system_instructions": st.session_state.get("system_instructions", ""),
            "temperature": float(st.session_state.get("temperature", 0.35)),
            "top_k": int(st.session_state.get("top_k", 40)),
            "top_p": float(st.session_state.get("top_p", 0.9)),
            "repeat_penalty": float(st.session_state.get("repeat_penalty", 1.1)),
            "thinking_level": st.session_state.get("thinking_level", "medium"),
            "strict_guardrail": bool(st.session_state.get("strict_guardrail", True)),
            "fast_mode": bool(st.session_state.get("fast_mode", True)),
            "initial_k": int(st.session_state.get("initial_k", 16)),
            "final_k": int(st.session_state.get("final_k", 5)),
            "retrieval_min_overlap": float(st.session_state.get("retrieval_min_overlap", 0.05)),
            "guardrail_threshold": float(st.session_state.get("guardrail_threshold", 0.5)),
            "citation_min_coverage": float(st.session_state.get("citation_min_coverage", 0.8)),
            "creativity": int(st.session_state.get("creativity", 35)),
            "log_level": int(st.session_state.get("log_level", 1)),
            "doc_filter_mode": st.session_state.get("doc_filter_mode", "auto"),
            "manual_doc_id": st.session_state.get("manual_doc_id", ""),
            "preset_id": st.session_state.get("preset_id", "tr_balanced"),
            "embedding_model": st.session_state.get("embedding_model", EMBED_MODEL_NAME),
            "reranker_model": st.session_state.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
            "ocr_profile": st.session_state.get("ocr_profile", "default"),
            "ocr_backend": st.session_state.get("ocr_backend", "paddle"),
            "ocr_device_mode": st.session_state.get("ocr_device_mode", "cpu"),
            "session_id": session_id,
            "access_key": access_key,
        }
    )
    atomic_write_json(paths["state_file"], payload)


def cleanup_session(runtime_root: Path) -> None:
    resolved_root = SESSION_ROOT.resolve()
    resolved_target = runtime_root.resolve()
    if resolved_root not in resolved_target.parents and resolved_target != resolved_root:
        raise RuntimeError("Guvenlik hatasi: Session dizini beklenen kok altinda degil.")
    clear_dir(resolved_target)
    try:
        resolved_target.rmdir()
    except OSError:
        pass


def validate_or_recover_session(session_id: str, access_key: str, paths: Dict[str, Path]) -> tuple[str, str, Dict[str, Any]]:
    loaded = load_state(paths["state_file"])
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
    new_paths = ensure_dirs(new_sid)
    new_state = default_state()
    new_state["session_id"] = new_sid
    new_state["access_key"] = new_key
    atomic_write_json(new_paths["state_file"], new_state)
    st.warning("Oturum kimligi dogrulanamadi. Guvenlik nedeniyle yeni izole oturum olusturuldu.")
    return new_sid, new_key, new_state


def save_uploaded_files(uploaded_files: List[Any], upload_dir: Path) -> List[Path]:
    saved: List[Path] = []
    for f in uploaded_files:
        out = upload_dir / f.name
        out.write_bytes(f.getbuffer())
        saved.append(out)
    return saved

